"""Lightweight CD38 pocket-definition parameter sensitivity analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure

from compare_pocket_method_consensus import build_pocket_method_consensus
from extract_fpocket_pocket_residues import extract_fpocket_residue_keys
from pdb_parser import load_complex_pdb
from pocket_io import load_residue_set, normalize_residue_key


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze CD38 pocket parameter sensitivity")
    parser.add_argument("--manifest_csv", default="benchmarks/cd38/cd38_benchmark_manifest.csv")
    parser.add_argument("--results_root", default="benchmarks/cd38/results")
    parser.add_argument("--truth_file", default="benchmarks/cd38/cd38_active_site_truth.txt")
    parser.add_argument("--out_dir", default="benchmarks/cd38/parameter_sensitivity")
    parser.add_argument("--contact_cutoffs", default="3.5,4.0,4.5,5.0,5.5")
    parser.add_argument("--consensus_min_method_counts", default="1,2")
    parser.add_argument("--overwide_penalty_weights", default="0.0,0.15,0.30")
    return parser


def _clean(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    return text or None


def _float_grid(text: str, name: str) -> list[float]:
    vals: list[float] = []
    for raw in str(text or "").split(","):
        if raw.strip():
            try:
                vals.append(float(raw.strip()))
            except ValueError as exc:
                raise ValueError(f"Invalid {name}: {raw!r}") from exc
    if not vals:
        raise ValueError(f"{name} cannot be empty.")
    return sorted(set(vals))


def _int_grid(text: str, name: str) -> list[int]:
    vals: list[int] = []
    for raw in str(text or "").split(","):
        if raw.strip():
            vals.append(max(1, int(raw.strip())))
    if not vals:
        raise ValueError(f"{name} cannot be empty.")
    return sorted(set(vals))


def _sort_key(key: str) -> tuple[str, int, str]:
    parts = str(key).split(":")
    chain = parts[0] if parts else ""
    try:
        resseq = int(parts[1]) if len(parts) > 1 else 0
    except ValueError:
        resseq = 0
    icode = parts[2] if len(parts) > 2 else ""
    return chain, resseq, icode


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _enabled(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"manifest_csv not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    if "enabled" in df.columns:
        df = df[df["enabled"].map(_enabled)].reset_index(drop=True)
    if "result_name" not in df.columns or "method" not in df.columns:
        raise ValueError("Manifest must contain result_name and method columns.")
    return df


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_sanitize(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if np.isfinite(out) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _iter_residues(entity: Any) -> list[Residue]:
    if isinstance(entity, Residue):
        return [entity]
    if isinstance(entity, Chain):
        return list(entity.get_residues())
    if isinstance(entity, Model):
        return list(entity.get_residues())
    if isinstance(entity, Structure):
        models = list(entity.get_models())
        return list(models[0].get_residues()) if models else []
    if hasattr(entity, "get_residues"):
        return list(entity.get_residues())
    raise TypeError("Unsupported structure entity.")


def _coords(residue: Residue) -> np.ndarray:
    items: list[np.ndarray] = []
    for atom in residue.get_atoms():
        try:
            coord = np.asarray(atom.get_coord(), dtype=np.float64).reshape(3)
        except Exception:
            continue
        if np.isfinite(coord).all():
            items.append(coord)
    return np.vstack(items).astype(np.float64, copy=False) if items else np.empty((0, 3), dtype=np.float64)


def _min_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape[0] == 0 or b.shape[0] == 0:
        return float("nan")
    diff = a[:, None, :] - b[None, :, :]
    return float(np.sqrt(np.min(np.einsum("ijk,ijk->ij", diff, diff, optimize=True))))


def _chain_id(residue: Residue) -> str:
    return str(getattr(residue.get_parent(), "id", "")).strip() or "_"


def _canonical_key(residue: Residue) -> str:
    _, resseq, icode = residue.id
    return normalize_residue_key(_chain_id(residue), int(resseq), str(icode).strip())


def _protein_like(residue: Residue) -> bool:
    return str(residue.id[0]).strip() == ""


def _result_dir(results_root: Path, row: pd.Series) -> Path:
    result_name = _clean(row.get("result_name"))
    if not result_name:
        raise ValueError("Manifest row is missing result_name.")
    return results_root / result_name


def _resolve_pdb(results_root: Path, row: pd.Series) -> Path:
    explicit = _clean(row.get("pdb_path"))
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if path.exists():
            return path
    result_dir = _result_dir(results_root, row)
    for name in ["ligand_contact_selection_summary.json", "run_manifest.json"]:
        candidate = _clean(_read_json(result_dir / name).get("pdb_path"))
        if candidate:
            path = Path(candidate).expanduser().resolve()
            if path.exists():
                return path
    pdb_id = _clean(row.get("rcsb_pdb_id"))
    if pdb_id:
        path = result_dir / "inputs" / f"{pdb_id.upper()}.pdb"
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(f"Could not resolve local PDB for {row.get('result_name')!r}.")


def _name_set(value: Any) -> set[str]:
    return {x.strip().upper() for x in str(value or "").split(",") if x.strip()}


def _int_set(value: Any) -> set[int]:
    vals: set[int] = set()
    for item in str(value or "").split(","):
        if item.strip():
            vals.add(int(item.strip()))
    return vals


def _derive_contact_keys(
    pdb_path: Path,
    *,
    protein_chain: str | None,
    ligand_chain: str | None,
    ligand_resnames: set[str],
    ligand_resseqs: set[int],
    cutoff: float,
) -> set[str]:
    residues = _iter_residues(load_complex_pdb(str(pdb_path)))
    protein: list[Residue] = []
    ligands: list[Residue] = []
    for residue in residues:
        chain = _chain_id(residue)
        _, resseq, _ = residue.id
        resname = str(residue.get_resname()).strip().upper()
        if _protein_like(residue):
            if not protein_chain or chain == protein_chain:
                protein.append(residue)
            continue
        if ligand_chain and chain != ligand_chain:
            continue
        if ligand_resnames and resname not in ligand_resnames:
            continue
        if ligand_resseqs and int(resseq) not in ligand_resseqs:
            continue
        ligands.append(residue)
    if not ligands:
        raise ValueError(f"No ligand residues matched filters in {pdb_path}.")
    ligand_blocks = [block for block in (_coords(res) for res in ligands) if block.shape[0] > 0]
    if not ligand_blocks:
        raise ValueError(f"No ligand coordinates in {pdb_path}.")
    ligand_coords = np.vstack(ligand_blocks).astype(np.float64, copy=False)
    keys: set[str] = set()
    for residue in protein:
        distance = _min_distance(_coords(residue), ligand_coords)
        if np.isfinite(distance) and distance <= float(cutoff):
            keys.add(_canonical_key(residue))
    return keys


def _p2rank_key(token: str) -> str:
    parts = [part for part in str(token).strip().split("_") if part]
    if len(parts) < 2:
        raise ValueError(f"Unsupported P2Rank residue token: {token!r}")
    icode = parts[2] if len(parts) >= 3 and len(parts[2]) == 1 and not parts[2].isdigit() else ""
    return normalize_residue_key(parts[0], int(parts[1]), icode)


def _p2rank_residues(row: pd.Series, chain_filter: str | None) -> set[str]:
    out: set[str] = set()
    for token in str(row.get("residue_ids") or "").split():
        key = _p2rank_key(token)
        if not chain_filter or key.split(":", 1)[0] == chain_filter:
            out.add(key)
    return out


def _evaluate(predicted: set[str], truth: set[str]) -> dict[str, Any]:
    pred = set(predicted)
    truth_set = set(truth)
    overlap = pred & truth_set
    union = pred | truth_set
    coverage = float(len(overlap) / len(truth_set)) if truth_set else float("nan")
    precision = float(len(overlap) / len(pred)) if pred else float("nan")
    jaccard = float(len(overlap) / len(union)) if union else float("nan")
    f1 = (
        float(2 * coverage * precision / (coverage + precision))
        if np.isfinite(coverage) and np.isfinite(precision) and (coverage + precision) > 0
        else float("nan")
    )
    missing = sorted(truth_set - pred, key=_sort_key)
    extra = sorted(pred - truth_set, key=_sort_key)
    utility = float(np.clip(0.60 * coverage + 0.40 * precision, 0.0, 1.0)) if np.isfinite(coverage) and np.isfinite(precision) else 0.0
    return {
        "predicted_residue_count": int(len(pred)),
        "truth_count": int(len(truth_set)),
        "exact_overlap_count": int(len(overlap)),
        "exact_truth_coverage": coverage,
        "exact_predicted_precision": precision,
        "exact_jaccard": jaccard,
        "exact_f1": f1,
        "missing_truth_count": int(len(missing)),
        "extra_predicted_count": int(len(extra)),
        "missing_truth_risk": float(1 - coverage) if np.isfinite(coverage) else float("nan"),
        "overwide_risk": float(1 - precision) if np.isfinite(precision) else float("nan"),
        "utility_score": utility,
        "missing_truth_residues": ";".join(missing),
        "extra_predicted_residues": ";".join(extra),
        "predicted_residues": ";".join(sorted(pred, key=_sort_key)),
    }


def _contact_rows(df: pd.DataFrame, results_root: Path, truth: set[str], cutoffs: list[float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in df[df["method"].astype(str).str.strip().eq("ligand_contact")].iterrows():
        result_dir = _result_dir(results_root, row)
        payload = _read_json(result_dir / "ligand_contact_selection_summary.json")
        pdb_path = _resolve_pdb(results_root, row)
        protein_chain = _clean(row.get("protein_chain")) or _clean(payload.get("protein_chain"))
        ligand_chain = _clean(row.get("ligand_chain")) or _clean(payload.get("ligand_chain"))
        ligand_resnames = _name_set(_clean(row.get("ligand_resnames")) or ",".join(payload.get("ligand_resnames") or []))
        ligand_resseqs = _int_set(_clean(row.get("ligand_resseqs")) or ",".join(str(x) for x in payload.get("ligand_resseqs") or []))
        baseline = float(payload.get("distance_threshold") or row.get("distance_threshold") or 4.5)
        for cutoff in cutoffs:
            pred = _derive_contact_keys(
                pdb_path,
                protein_chain=protein_chain,
                ligand_chain=ligand_chain,
                ligand_resnames=ligand_resnames,
                ligand_resseqs=ligand_resseqs,
                cutoff=cutoff,
            )
            rows.append(
                {
                    "scenario_type": "ligand_contact_cutoff",
                    "result_name": _clean(row.get("result_name")),
                    "rcsb_pdb_id": _clean(row.get("rcsb_pdb_id")),
                    "parameter_name": "distance_threshold",
                    "parameter_value": float(cutoff),
                    "is_baseline_parameter": bool(abs(float(cutoff) - baseline) < 1e-9),
                    "protein_chain": protein_chain,
                    "ligand_chain": ligand_chain,
                    "ligand_resnames": ";".join(sorted(ligand_resnames)),
                    "source_path": str(pdb_path),
                    **_evaluate(pred, truth),
                }
            )
    return rows


def _p2rank_rows(df: pd.DataFrame, results_root: Path, truth: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in df[df["method"].astype(str).str.strip().eq("p2rank")].iterrows():
        result_dir = _result_dir(results_root, row)
        csv_text = _clean(row.get("predictions_csv"))
        source_csv = Path(csv_text).expanduser().resolve() if csv_text else result_dir / "source_predictions.csv"
        if not source_csv.exists():
            continue
        src = pd.read_csv(source_csv, skipinitialspace=True)
        src.columns = [str(col).strip() for col in src.columns]
        if "rank" not in src.columns or "residue_ids" not in src.columns:
            continue
        chain_filter = _clean(row.get("chain_filter")) or _clean(_read_json(result_dir / "run_manifest.json").get("chain_filter"))
        baseline_rank = int(float(row.get("rank"))) if _clean(row.get("rank")) else None
        for _, pocket in src.iterrows():
            rank = int(pd.to_numeric(pd.Series([pocket.get("rank")]), errors="coerce").iloc[0])
            pred = _p2rank_residues(pocket, chain_filter)
            rows.append(
                {
                    "scenario_type": "p2rank_rank_choice",
                    "result_name": _clean(row.get("result_name")),
                    "rcsb_pdb_id": _clean(row.get("rcsb_pdb_id")),
                    "parameter_name": "rank",
                    "parameter_value": rank,
                    "is_baseline_parameter": bool(baseline_rank is not None and rank == baseline_rank),
                    "pocket_name": str(pocket.get("name")).strip(),
                    "p2rank_score": float(pd.to_numeric(pd.Series([pocket.get("score")]), errors="coerce").iloc[0]),
                    "p2rank_probability": float(pd.to_numeric(pd.Series([pocket.get("probability")]), errors="coerce").iloc[0]) if "probability" in src.columns else float("nan"),
                    "chain_filter": chain_filter,
                    "source_path": str(source_csv),
                    **_evaluate(pred, truth),
                }
            )
    return rows


def _fpocket_rows(df: pd.DataFrame, results_root: Path, truth: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in df[df["method"].astype(str).str.strip().eq("fpocket")].iterrows():
        result_dir = _result_dir(results_root, row)
        result_name = _clean(row.get("result_name"))
        manifest_payload = _read_json(result_dir / "run_manifest.json")
        chain_filter = _clean(row.get("chain_filter")) or _clean(manifest_payload.get("chain_filter"))
        include_hetatm = _enabled(row.get("include_hetatm"))
        source_hint = (
            _clean(row.get("fpocket_pocket_pdb"))
            or _clean(manifest_payload.get("original_fpocket_pocket_pdb"))
            or _clean(manifest_payload.get("fpocket_pocket_pdb"))
        )
        source_path: Path | None = Path(source_hint).expanduser().resolve() if source_hint else None
        predicted = result_dir / "predicted_pocket.txt"
        if predicted.exists():
            pred = set(load_residue_set(str(predicted)))
        else:
            if source_path is None:
                continue
            if not source_path.exists():
                continue
            pred = set(
                extract_fpocket_residue_keys(
                    source_path,
                    chain_filter=chain_filter,
                    include_hetatm=include_hetatm,
                )[0]
            )
        pocket_number = None
        if source_path is not None and source_path.name:
            match = pd.Series([source_path.stem]).astype(str).str.extract(r"pocket(\d+)", expand=False).iloc[0]
            if pd.notna(match):
                pocket_number = int(match)
        rows.append(
            {
                "scenario_type": "fpocket_pocket_choice",
                "result_name": result_name,
                "rcsb_pdb_id": _clean(row.get("rcsb_pdb_id")),
                "parameter_name": "pocket_file",
                "parameter_value": pocket_number if pocket_number is not None else str(source_path.name if source_path else result_name),
                "is_baseline_parameter": True,
                "chain_filter": chain_filter,
                "include_hetatm": bool(include_hetatm),
                "source_path": str(source_path) if source_path is not None else str(predicted),
                **_evaluate(pred, truth),
            }
        )
    return rows


FPOCKET_SENSITIVITY_COLUMNS = [
    "scenario_type",
    "result_name",
    "rcsb_pdb_id",
    "parameter_name",
    "parameter_value",
    "is_baseline_parameter",
    "chain_filter",
    "include_hetatm",
    "source_path",
    "predicted_residue_count",
    "truth_count",
    "exact_overlap_count",
    "exact_truth_coverage",
    "exact_predicted_precision",
    "exact_jaccard",
    "exact_f1",
    "missing_truth_count",
    "extra_predicted_count",
    "missing_truth_risk",
    "overwide_risk",
    "utility_score",
    "missing_truth_residues",
    "extra_predicted_residues",
    "predicted_residues",
]


def _consensus_rows(df: pd.DataFrame, results_root: Path, truth: set[str], min_counts: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = df.groupby(df["rcsb_pdb_id"].astype(str).str.strip().str.upper(), dropna=True)
    for pdb_id, group in groups:
        method_paths: dict[str, Path] = {}
        for _, row in group.iterrows():
            method = str(row.get("method")).strip()
            if method not in {"ligand_contact", "p2rank", "fpocket", "manual"}:
                continue
            path = _result_dir(results_root, row) / "predicted_pocket.txt"
            if path.exists():
                method_paths[method] = path.resolve()
        if len(method_paths) < 2:
            continue
        methods = {
            name: {"source_path": str(path), "residues": set(load_residue_set(str(path)))}
            for name, path in sorted(method_paths.items())
        }
        for min_count in min_counts:
            if min_count > len(methods):
                continue
            result = build_pocket_method_consensus(
                methods,
                truth_residues=set(truth),
                min_method_count=int(min_count),
            )
            summary = result["summary"]
            truth_summary = summary.get("truth", {})
            rows.append(
                {
                    "scenario_type": "method_consensus_threshold",
                    "result_name": f"{pdb_id}_method_consensus_min{min_count}",
                    "rcsb_pdb_id": str(pdb_id),
                    "parameter_name": "min_method_count",
                    "parameter_value": int(min_count),
                    "is_baseline_parameter": bool(int(min_count) == len(methods)),
                    "method_count": int(summary.get("method_count", len(methods))),
                    "methods": ";".join(summary.get("methods", sorted(methods))),
                    "method_specific_residue_count": int(summary.get("method_specific_residue_count", 0)),
                    "consensus_tool_coverage": truth_summary.get("consensus_truth_coverage"),
                    "consensus_tool_precision": truth_summary.get("consensus_truth_precision"),
                    "source_path": ";".join(str(path) for path in method_paths.values()),
                    **_evaluate(set(result["consensus_residues"]), truth),
                }
            )
    return rows


def _penalty_rows(all_df: pd.DataFrame, weights: list[float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if all_df.empty:
        return pd.DataFrame()
    for _, row in all_df.iterrows():
        for weight in weights:
            utility = float(row.get("utility_score") or 0.0)
            overwide = float(row.get("overwide_risk") or 0.0)
            rows.append(
                {
                    "scenario_type": row.get("scenario_type"),
                    "result_name": row.get("result_name"),
                    "rcsb_pdb_id": row.get("rcsb_pdb_id"),
                    "parameter_name": row.get("parameter_name"),
                    "parameter_value": row.get("parameter_value"),
                    "overwide_penalty_weight": float(weight),
                    "utility_score": utility,
                    "overwide_risk": overwide,
                    "adjusted_utility_score": float(np.clip(utility - float(weight) * overwide, 0.0, 1.0)),
                    "exact_truth_coverage": row.get("exact_truth_coverage"),
                    "exact_predicted_precision": row.get("exact_predicted_precision"),
                    "predicted_residue_count": row.get("predicted_residue_count"),
                }
            )
    return pd.DataFrame(rows)


def _best_rows(df: pd.DataFrame, key: str) -> list[dict[str, Any]]:
    if df.empty:
        return []
    out: list[dict[str, Any]] = []
    for _, group in df.groupby(key, dropna=False):
        top = group.sort_values(
            by=["utility_score", "exact_truth_coverage", "exact_predicted_precision"],
            ascending=[False, False, False],
            na_position="last",
        ).head(1)
        out.extend(top.replace({np.nan: None}).to_dict(orient="records"))
    return out


def _table(df: pd.DataFrame, columns: list[str], max_rows: int = 12) -> list[str]:
    if df.empty:
        return ["_No rows._"]
    cols = [col for col in columns if col in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.head(max_rows).iterrows():
        values: list[str] = []
        for col in cols:
            value = row.get(col)
            if isinstance(value, float):
                values.append(f"{value:.4f}" if np.isfinite(value) else "")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _report(
    summary: dict[str, Any],
    contact: pd.DataFrame,
    p2rank: pd.DataFrame,
    fpocket: pd.DataFrame,
    consensus: pd.DataFrame,
    penalty: pd.DataFrame,
) -> str:
    lines = [
        "# CD38 Pocket Parameter Sensitivity",
        "",
        f"- Ligand-contact scenarios: `{summary['contact_scenario_count']}`",
        f"- P2Rank rank scenarios: `{summary['p2rank_rank_scenario_count']}`",
        f"- fpocket pocket scenarios: `{summary['fpocket_pocket_scenario_count']}`",
        f"- Method-consensus scenarios: `{summary['consensus_scenario_count']}`",
        f"- Overwide-penalty scenarios: `{summary['penalty_scenario_count']}`",
        "",
        "## Interpretation",
        "",
        "- `exact_truth_coverage` measures recovery of known CD38 key residues.",
        "- `exact_predicted_precision` measures how narrow the predicted pocket is against the conservative truth set.",
        "- `utility_score = 0.60 * coverage + 0.40 * precision`; adjusted utility subtracts an overwide penalty.",
        "- This is a robustness check using existing local benchmark files; it does not run external pocket finders.",
        "",
        "## Ligand-Contact Cutoff Scenarios",
        "",
    ]
    lines.extend(_table(contact.sort_values(["result_name", "parameter_value"]) if not contact.empty else contact, ["result_name", "parameter_value", "predicted_residue_count", "exact_truth_coverage", "exact_predicted_precision", "exact_f1", "utility_score"]))
    lines.extend(["", "## P2Rank Rank Choice Scenarios", ""])
    lines.extend(
        _table(
            p2rank.sort_values(["result_name", "parameter_value"]) if not p2rank.empty else p2rank,
            [
                "result_name",
                "parameter_value",
                "pocket_name",
                "predicted_residue_count",
                "exact_truth_coverage",
                "exact_predicted_precision",
                "utility_score",
            ],
            max_rows=30,
        )
    )
    lines.extend(["", "## fpocket Pocket Choice Scenarios", ""])
    lines.extend(
        _table(
            fpocket.sort_values(["rcsb_pdb_id", "result_name"]) if not fpocket.empty else fpocket,
            [
                "result_name",
                "parameter_value",
                "predicted_residue_count",
                "exact_truth_coverage",
                "exact_predicted_precision",
                "utility_score",
            ],
            max_rows=30,
        )
    )
    lines.extend(["", "## Method Consensus Threshold Scenarios", ""])
    lines.extend(_table(consensus.sort_values(["rcsb_pdb_id", "parameter_value"]) if not consensus.empty else consensus, ["rcsb_pdb_id", "parameter_value", "predicted_residue_count", "exact_truth_coverage", "exact_predicted_precision", "utility_score"]))
    lines.extend(["", "## Best Adjusted Utility Examples", ""])
    lines.extend(_table(penalty.sort_values("adjusted_utility_score", ascending=False) if not penalty.empty else penalty, ["scenario_type", "result_name", "parameter_value", "overwide_penalty_weight", "exact_truth_coverage", "exact_predicted_precision", "adjusted_utility_score"]))
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `contact_cutoff_sensitivity.csv`",
            "- `p2rank_rank_sensitivity.csv`",
            "- `fpocket_pocket_sensitivity.csv`",
            "- `method_consensus_threshold_sensitivity.csv`",
            "- `overwide_penalty_sensitivity.csv`",
            "- `cd38_pocket_parameter_sensitivity_summary.json`",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = _build_parser().parse_args()
    manifest_csv = Path(args.manifest_csv).expanduser().resolve()
    results_root = Path(args.results_root).expanduser().resolve()
    truth_file = Path(args.truth_file).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(manifest_csv)
    truth = set(load_residue_set(str(truth_file)))
    contact_cutoffs = _float_grid(args.contact_cutoffs, "contact_cutoffs")
    min_counts = _int_grid(args.consensus_min_method_counts, "consensus_min_method_counts")
    penalty_weights = _float_grid(args.overwide_penalty_weights, "overwide_penalty_weights")

    contact = pd.DataFrame(_contact_rows(manifest, results_root, truth, contact_cutoffs))
    p2rank = pd.DataFrame(_p2rank_rows(manifest, results_root, truth))
    fpocket = pd.DataFrame(_fpocket_rows(manifest, results_root, truth))
    if fpocket.empty:
        fpocket = pd.DataFrame(columns=FPOCKET_SENSITIVITY_COLUMNS)
    consensus = pd.DataFrame(_consensus_rows(manifest, results_root, truth, min_counts))
    parts = [df for df in [contact, p2rank, fpocket, consensus] if not df.empty]
    all_scenarios = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    penalty = _penalty_rows(all_scenarios, penalty_weights)

    summary = {
        "manifest_csv": str(manifest_csv),
        "results_root": str(results_root),
        "truth_file": str(truth_file),
        "contact_cutoffs": contact_cutoffs,
        "consensus_min_method_counts": min_counts,
        "overwide_penalty_weights": penalty_weights,
        "contact_scenario_count": int(contact.shape[0]),
        "p2rank_rank_scenario_count": int(p2rank.shape[0]),
        "fpocket_pocket_scenario_count": int(fpocket.shape[0]),
        "consensus_scenario_count": int(consensus.shape[0]),
        "penalty_scenario_count": int(penalty.shape[0]),
        "best_contact_by_result": _best_rows(contact, "result_name"),
        "best_p2rank_rank_by_result": _best_rows(p2rank, "result_name"),
        "best_fpocket_pocket_by_pdb": _best_rows(fpocket, "rcsb_pdb_id"),
        "best_consensus_by_pdb": _best_rows(consensus, "rcsb_pdb_id"),
        "minimum_contact_coverage": float(contact["exact_truth_coverage"].min()) if not contact.empty else None,
        "minimum_p2rank_rank_coverage": float(p2rank["exact_truth_coverage"].min()) if not p2rank.empty else None,
        "minimum_fpocket_pocket_coverage": float(fpocket["exact_truth_coverage"].min()) if not fpocket.empty else None,
        "minimum_consensus_coverage": float(consensus["exact_truth_coverage"].min()) if not consensus.empty else None,
    }

    outputs = {
        "contact": out_dir / "contact_cutoff_sensitivity.csv",
        "p2rank": out_dir / "p2rank_rank_sensitivity.csv",
        "fpocket": out_dir / "fpocket_pocket_sensitivity.csv",
        "consensus": out_dir / "method_consensus_threshold_sensitivity.csv",
        "penalty": out_dir / "overwide_penalty_sensitivity.csv",
        "summary": out_dir / "cd38_pocket_parameter_sensitivity_summary.json",
        "report": out_dir / "cd38_pocket_parameter_sensitivity_report.md",
    }
    contact.to_csv(outputs["contact"], index=False)
    p2rank.to_csv(outputs["p2rank"], index=False)
    fpocket.to_csv(outputs["fpocket"], index=False)
    consensus.to_csv(outputs["consensus"], index=False)
    penalty.to_csv(outputs["penalty"], index=False)
    outputs["summary"].write_text(json.dumps(_json_sanitize(summary), ensure_ascii=True, indent=2), encoding="utf-8")
    outputs["report"].write_text(_report(summary, contact, p2rank, fpocket, consensus, penalty), encoding="utf-8")

    for path in outputs.values():
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
