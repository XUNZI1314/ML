"""Parse optional per-pose sidecar outputs into numeric pose features.

The main pipeline treats these files as optional. Missing or malformed sidecar
files should never make a pose fail; they simply contribute fewer features.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np


FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][-+]?\d+)?")


def _to_float(value: Any) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return float("nan")


def _finite_or_nan(values: Iterable[float], fn: str) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    if fn == "sum":
        return float(np.sum(arr))
    if fn == "mean":
        return float(np.mean(arr))
    if fn == "min":
        return float(np.min(arr))
    if fn == "max":
        return float(np.max(arr))
    return float("nan")


def read_single_float(path: str | Path) -> float:
    """Read the last numeric value from a small text sidecar file."""
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        return float("nan")
    numbers = FLOAT_RE.findall(file_path.read_text(encoding="utf-8", errors="ignore"))
    return _to_float(numbers[-1]) if numbers else float("nan")


def parse_mmpbsa_normalized(path: str | Path) -> dict[str, float]:
    """Parse MMPBSA_normalized.txt when present."""
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        return {}
    lines = [line.strip() for line in file_path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    for line in reversed(lines):
        if line.startswith("-") or line.lower().startswith("vhh"):
            parts = [part.strip() for part in line.split("|")]
            if len(parts) >= 5:
                value = _to_float(parts[-1])
                if math.isfinite(value):
                    return {"mmpbsa_normalized": value}
    value = read_single_float(file_path)
    return {"mmpbsa_normalized": value} if math.isfinite(value) else {}


def parse_interface_sc(path: str | Path) -> dict[str, float]:
    """Parse Rosetta-style *_interface.sc score files."""
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        return {}
    header: list[str] | None = None
    values: list[str] | None = None
    for raw in file_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line.startswith("SCORE:"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        payload = parts[1:]
        if payload[0] == "total_score":
            header = payload
        elif header is not None:
            values = payload
            break
    if not header or not values:
        return {}

    parsed: dict[str, float] = {}
    for key, value in zip(header, values):
        if key == "description":
            continue
        number = _to_float(value)
        if math.isfinite(number):
            parsed[f"interface_{key}"] = number
    return parsed


def residue_numbers_from_keys(
    residue_keys: Iterable[str] | None,
    *,
    chain_id: str | None = None,
) -> set[int]:
    """Extract residue numbers from canonical keys such as B:82 or B:82:A."""
    if not residue_keys:
        return set()
    wanted_chain = str(chain_id).strip() if chain_id is not None else ""
    numbers: set[int] = set()
    for key in residue_keys:
        parts = [part.strip() for part in str(key).split(":") if part.strip()]
        if len(parts) < 2:
            continue
        key_chain, residue_text = parts[0], parts[1]
        if wanted_chain and key_chain != wanted_chain:
            continue
        try:
            numbers.add(int(residue_text))
        except ValueError:
            continue
    return numbers


def parse_decomp_mmpbsa(
    path: str | Path,
    *,
    pocket_residue_numbers: Iterable[int] | None = None,
) -> dict[str, float]:
    """Parse FINAL_DECOMP_MMPBSA.dat into generic antigen/nanobody features.

    The parser assumes receptor rows marked as ``R`` correspond to the antigen
    and ligand rows marked as ``L`` correspond to the nanobody, which matches the
    current VHH-CD38 result layout.
    """
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        return {}

    pocket_numbers = {int(item) for item in (pocket_residue_numbers or [])}
    antigen_totals: list[float] = []
    pocket_totals: list[float] = []
    nanobody_totals: list[float] = []

    for raw in file_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "|" not in raw or "+/-" not in raw:
            continue
        parts = raw.split("|")
        if len(parts) < 3:
            continue
        left = parts[0].strip().split()
        loc = parts[1].strip().split()
        total_text = parts[-1]
        numbers = [_to_float(item) for item in FLOAT_RE.findall(total_text)]
        numbers = [item for item in numbers if math.isfinite(item)]
        if len(left) < 2 or len(loc) < 3 or not numbers:
            continue
        try:
            residue_index = int(left[-1])
        except ValueError:
            continue

        chain_marker = loc[0]
        total = numbers[0]
        if chain_marker == "R":
            antigen_totals.append(total)
            if residue_index in pocket_numbers:
                pocket_totals.append(total)
        elif chain_marker == "L":
            nanobody_totals.append(total)

    return {
        "decomp_antigen_total_sum": _finite_or_nan(antigen_totals, "sum"),
        "decomp_antigen_total_mean": _finite_or_nan(antigen_totals, "mean"),
        "decomp_antigen_total_min": _finite_or_nan(antigen_totals, "min"),
        "decomp_antigen_total_max": _finite_or_nan(antigen_totals, "max"),
        "decomp_nanobody_total_sum": _finite_or_nan(nanobody_totals, "sum"),
        "decomp_nanobody_total_mean": _finite_or_nan(nanobody_totals, "mean"),
        "decomp_nanobody_total_min": _finite_or_nan(nanobody_totals, "min"),
        "decomp_nanobody_total_max": _finite_or_nan(nanobody_totals, "max"),
        "decomp_pocket_total_sum": _finite_or_nan(pocket_totals, "sum"),
        "decomp_pocket_total_mean": _finite_or_nan(pocket_totals, "mean"),
        "decomp_pocket_total_min": _finite_or_nan(pocket_totals, "min"),
        "decomp_pocket_total_max": _finite_or_nan(pocket_totals, "max"),
        "decomp_pocket_residue_count": float(len(pocket_totals)),
    }


def _find_first_existing(candidates: Iterable[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def parse_pose_sidecar_features(
    pose_dir: str | Path | None,
    *,
    pose_id: str | None = None,
    pocket_residue_keys: Iterable[str] | None = None,
    antigen_chain: str | None = None,
) -> dict[str, Any]:
    """Parse known optional sidecar files for one pose directory."""
    if pose_dir is None:
        return {
            "sidecar_parse_status": "missing_pose_dir",
            "sidecar_metric_count": 0.0,
        }
    directory = Path(pose_dir).expanduser()
    if not directory.is_dir():
        return {
            "sidecar_parse_status": "missing_pose_dir",
            "sidecar_metric_count": 0.0,
        }

    pose = str(pose_id or directory.name).strip()
    features: dict[str, Any] = {
        "sidecar_parse_status": "ok",
        "sidecar_metric_count": 0.0,
    }

    normalized = parse_mmpbsa_normalized(directory / "MMPBSA_normalized.txt")
    features.update(normalized)

    score_txt = read_single_float(directory / "score.txt")
    if math.isfinite(score_txt):
        features["score_txt"] = score_txt

    accuracy_candidates = []
    if pose:
        accuracy_candidates.append(directory / f"{pose}_accuracy.txt")
    accuracy_candidates.extend(sorted(directory.glob("*_accuracy.txt"), key=lambda p: p.name.lower()))
    accuracy_path = _find_first_existing(accuracy_candidates)
    if accuracy_path is not None:
        accuracy = read_single_float(accuracy_path)
        if math.isfinite(accuracy):
            features["rsite_accuracy"] = accuracy

    interface_candidates = []
    if pose:
        interface_candidates.append(directory / f"{pose}_interface.sc")
    interface_candidates.extend(sorted(directory.glob("*_interface.sc"), key=lambda p: p.name.lower()))
    interface_path = _find_first_existing(interface_candidates)
    if interface_path is not None:
        features.update(parse_interface_sc(interface_path))

    pocket_numbers = residue_numbers_from_keys(pocket_residue_keys, chain_id=antigen_chain)
    features.update(
        parse_decomp_mmpbsa(
            directory / "FINAL_DECOMP_MMPBSA.dat",
            pocket_residue_numbers=pocket_numbers,
        )
    )

    metric_count = sum(
        1
        for key, value in features.items()
        if key not in {"sidecar_parse_status", "sidecar_metric_count"}
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )
    features["sidecar_metric_count"] = float(metric_count)
    return features


__all__ = [
    "parse_pose_sidecar_features",
    "parse_interface_sc",
    "parse_decomp_mmpbsa",
    "parse_mmpbsa_normalized",
    "read_single_float",
    "residue_numbers_from_keys",
]
