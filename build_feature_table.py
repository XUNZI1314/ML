"""Build per-pose feature table for nanobody pocket-blocking screening.

This module reads an input CSV where each row represents one pose/complex,
extracts geometry features with existing parser modules, merges optional numeric
features, and writes a consolidated pose_features table.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from geometry_features import compute_all_geometry_features
from pdb_parser import (
    extract_residues_from_entity,
    extract_atoms_from_entity,
    load_complex_pdb,
    split_antigen_nanobody,
)
from pocket_io import load_ligand_template_pdb, load_residue_set, match_residues_in_structure


REQUIRED_INPUT_COLUMNS = [
    "nanobody_id",
    "conformer_id",
    "pose_id",
    "pdb_path",
]

OPTIONAL_HINT_NUMERIC_COLUMNS = [
    "hdock_score",
    "MMPBSA_energy",
    "mmpbsa_energy",
    "MMGBSA_energy",
    "mmgbsa_energy",
    "mmgbsa",
    "interface_dg",
    "buried_sasa",
    "num_hbonds",
    "shape_complementarity",
    "iptm",
    "pae",
    "plddt",
    "rsite_accuracy",
    "label",
]


def _is_missing(value: Any) -> bool:
    """Return True for None/NaN/blank-like values."""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _clean_str(value: Any) -> str | None:
    """Normalize a value to trimmed string or None when missing."""
    if _is_missing(value):
        return None
    text = str(value).strip()
    return text if text else None


def _resolve_path(path_like: str | None, base_dir: Path) -> str | None:
    """Resolve file path with CSV directory as fallback base."""
    if path_like is None:
        return None
    p = Path(path_like).expanduser()
    if p.is_absolute():
        return str(p)
    return str((base_dir / p).resolve())


def _safe_float_or_nan(value: Any) -> float:
    """Convert value to float; invalid values become NaN."""
    if _is_missing(value):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _short_error_message(exc: Exception, max_len: int = 240) -> str:
    """Render short, readable error text for status columns."""
    msg = str(exc).strip() or exc.__class__.__name__
    if len(msg) <= max_len:
        return msg
    return msg[: max_len - 3].rstrip() + "..."


def detect_optional_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Auto-detect optional numeric columns while skipping obvious text/id fields."""
    excluded = {
        "nanobody_id",
        "conformer_id",
        "pose_id",
        "pdb_path",
        "antigen_chain",
        "nanobody_chain",
        "pocket_file",
        "catalytic_file",
        "ligand_file",
        "status",
        "error_message",
        "warning_message",
        "split_mode",
        "geometry_debug_summary",
        "_row_index",
        "target_variant_index",
        "pose_index",
        "sidecar_file_count",
        "mmpbsa_parse_status",
        "mmpbsa_parse_warning",
    }

    numeric_cols: list[str] = []
    for col in df.columns:
        if col in excluded:
            continue

        text_col = str(col).strip().lower()
        if text_col.endswith("_id") and col not in OPTIONAL_HINT_NUMERIC_COLUMNS:
            continue
        if text_col.endswith("_path"):
            continue

        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(col)
            continue

        converted = pd.to_numeric(series, errors="coerce")
        finite_count = int(np.isfinite(converted.to_numpy(dtype=np.float64)).sum())
        if finite_count > 0 or col in OPTIONAL_HINT_NUMERIC_COLUMNS:
            numeric_cols.append(col)

    return numeric_cols


def read_input_table(input_csv: str, required_columns: list[str] | None = None) -> pd.DataFrame:
    """Read and validate input pose table.

    Args:
        input_csv: Path to input CSV.
        required_columns: Required column names. Defaults to core pose columns.

    Returns:
        Input DataFrame.

    Raises:
        FileNotFoundError: If input CSV does not exist.
        ValueError: If required columns are missing.
    """
    required = required_columns or list(REQUIRED_INPUT_COLUMNS)

    path = Path(input_csv).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    if not path.is_file():
        raise ValueError(f"Input CSV path is not a file: {path}")

    try:
        df = pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required input columns: {missing}")

    if df.empty:
        return df.copy()

    # Keep row order stable and provide explicit row id for easier debugging.
    out = df.copy()
    out["_row_index"] = np.arange(len(out), dtype=int)
    return out


def merge_optional_numeric_features(
    base_result: dict[str, Any],
    row: pd.Series,
    numeric_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Merge optional numeric columns from input row into output feature row.

    Args:
        base_result: Existing output dictionary.
        row: Input row.
        numeric_columns: Optional explicit numeric columns to merge.

    Returns:
        Updated output dictionary with numeric features appended.
    """
    result = dict(base_result)

    if numeric_columns is None:
        candidate_cols = detect_optional_numeric_columns(pd.DataFrame([row]))
    else:
        candidate_cols = [c for c in numeric_columns if c in row.index]

    for col in candidate_cols:
        value = row[col]
        number = _safe_float_or_nan(value)

        # Auto-detection rule:
        # - always include known hint columns
        # - include extra columns when value can be converted to numeric
        if (col in OPTIONAL_HINT_NUMERIC_COLUMNS) or np.isfinite(number) or _is_missing(value):
            result[col] = number

    return result


def process_one_pose(
    row: pd.Series,
    base_dir: str | Path,
    optional_numeric_cols: list[str] | None = None,
    atom_contact_threshold: float = 4.5,
    catalytic_contact_threshold: float = 4.5,
    substrate_clash_threshold: float = 2.8,
    mouth_residue_fraction: float = 0.30,
    default_pocket_file: str | None = None,
    default_catalytic_file: str | None = None,
    default_ligand_file: str | None = None,
    default_antigen_chain: str | None = None,
    default_nanobody_chain: str | None = None,
    pdb_structure_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Process one pose row and return output record with status/error fields.

    Args:
        row: One input DataFrame row.
        base_dir: Base directory for resolving relative file paths.
        optional_numeric_cols: Numeric columns to merge.
        atom_contact_threshold: Atom contact threshold (Angstrom).
        catalytic_contact_threshold: Catalytic contact threshold (Angstrom).
        substrate_clash_threshold: Ligand clash threshold (Angstrom).
        mouth_residue_fraction: Fraction for inferred mouth residues.

    Returns:
        Output row dictionary containing basic ids, geometry features, and
        status/error_message.
    """
    base = Path(base_dir)

    output: dict[str, Any] = {
        "nanobody_id": _clean_str(row.get("nanobody_id")),
        "conformer_id": _clean_str(row.get("conformer_id")),
        "pose_id": _clean_str(row.get("pose_id")),
        "pdb_path": _clean_str(row.get("pdb_path")),
        "antigen_chain": _clean_str(row.get("antigen_chain")) or _clean_str(default_antigen_chain),
        "nanobody_chain": _clean_str(row.get("nanobody_chain")) or _clean_str(default_nanobody_chain),
        "pocket_file": _clean_str(row.get("pocket_file")) or _clean_str(default_pocket_file),
        "catalytic_file": _clean_str(row.get("catalytic_file")) or _clean_str(default_catalytic_file),
        "ligand_file": _clean_str(row.get("ligand_file")) or _clean_str(default_ligand_file),
        "status": "failed",
        "error_message": "",
        "warning_message": "",
        "split_mode": "",
        "pdb_cache_state": "",
        "num_antigen_residues": 0.0,
        "num_nanobody_residues": 0.0,
        "num_pocket_residues_defined": 0.0,
        "num_pocket_residues_matched": 0.0,
        "num_catalytic_residues_defined": 0.0,
        "num_catalytic_residues_matched": 0.0,
        "num_ligand_atoms": 0.0,
    }

    try:
        pdb_path = _resolve_path(_clean_str(row.get("pdb_path")), base)
        if pdb_path is None:
            raise ValueError("pdb_path is missing.")

        if pdb_structure_cache is not None and pdb_path in pdb_structure_cache:
            structure = pdb_structure_cache[pdb_path]
            output["pdb_cache_state"] = "hit"
        else:
            structure = load_complex_pdb(pdb_path)
            output["pdb_cache_state"] = "miss"
            if pdb_structure_cache is not None:
                pdb_structure_cache[pdb_path] = structure

        antigen_chain = output["antigen_chain"]
        nanobody_chain = output["nanobody_chain"]
        split = split_antigen_nanobody(
            structure,
            antigen_chain=antigen_chain,
            nanobody_chain=nanobody_chain,
            fallback_by_ter=True,
        )
        output["split_mode"] = str(getattr(split, "split_mode", "") or getattr(split, "method", ""))

        antigen_atoms = extract_atoms_from_entity(split.antigen, heavy_only=True)
        nanobody_atoms = extract_atoms_from_entity(split.nanobody, heavy_only=True)
        antigen_residues = [
            residue
            for chain_id in split.antigen.chain_ids
            for residue in split.antigen.model[chain_id].get_residues()
        ]
        nanobody_residues = [
            residue
            for chain_id in split.nanobody.chain_ids
            for residue in split.nanobody.model[chain_id].get_residues()
        ]
        nanobody_residue_infos = extract_residues_from_entity(split.nanobody)
        antigen_residue_infos = extract_residues_from_entity(split.antigen)

        output["num_antigen_residues"] = float(len(antigen_residue_infos))
        output["num_nanobody_residues"] = float(len(nanobody_residue_infos))

        pocket_path = _resolve_path(output["pocket_file"], base)
        catalytic_path = _resolve_path(output["catalytic_file"], base)
        ligand_path = _resolve_path(output["ligand_file"], base)

        pocket_keys = load_residue_set(pocket_path) if pocket_path else None
        catalytic_keys = load_residue_set(catalytic_path) if catalytic_path else None
        ligand_template = load_ligand_template_pdb(ligand_path) if ligand_path else None

        warnings: list[str] = []
        output["num_pocket_residues_defined"] = float(len(pocket_keys) if pocket_keys is not None else 0)
        output["num_catalytic_residues_defined"] = float(len(catalytic_keys) if catalytic_keys is not None else 0)

        if pocket_keys:
            pocket_match = match_residues_in_structure(antigen_residues, pocket_keys)
            output["num_pocket_residues_matched"] = float(len(pocket_match.matched_keys))
            if pocket_match.warnings:
                warnings.extend([f"pocket: {w}" for w in pocket_match.warnings[:3]])

        if catalytic_keys:
            catalytic_match = match_residues_in_structure(antigen_residues, catalytic_keys)
            output["num_catalytic_residues_matched"] = float(len(catalytic_match.matched_keys))
            if catalytic_match.warnings:
                warnings.extend([f"catalytic: {w}" for w in catalytic_match.warnings[:3]])

        if ligand_template is not None and hasattr(ligand_template, "coordinates"):
            try:
                ligand_coords = np.asarray(getattr(ligand_template, "coordinates"), dtype=np.float64)
                if ligand_coords.ndim == 2 and ligand_coords.shape[1] == 3:
                    output["num_ligand_atoms"] = float(ligand_coords.shape[0])
            except Exception:
                output["num_ligand_atoms"] = 0.0

        geom_features = compute_all_geometry_features(
            antigen_atoms=antigen_atoms,
            antigen_residues=antigen_residues,
            nanobody_atoms=nanobody_atoms,
            nanobody_residues=nanobody_residues,
            pocket_residues=pocket_keys,
            catalytic_residues=catalytic_keys,
            ligand_template=ligand_template,
            atom_contact_threshold=atom_contact_threshold,
            catalytic_contact_threshold=catalytic_contact_threshold,
            substrate_clash_threshold=substrate_clash_threshold,
            mouth_residue_fraction=mouth_residue_fraction,
        )

        output.update(geom_features)
        output["status"] = "ok"
        output["error_message"] = ""
        output["warning_message"] = " | ".join(warnings[:5])
    except Exception as exc:
        output["status"] = "failed"
        output["error_message"] = _short_error_message(exc)

    output = merge_optional_numeric_features(
        base_result=output,
        row=row,
        numeric_columns=optional_numeric_cols,
    )
    return output


def _detect_optional_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Backward-compatible alias for optional numeric detection."""
    return detect_optional_numeric_columns(df)


def safe_process_one_pose(
    row: pd.Series,
    base_dir: str | Path,
    optional_numeric_cols: list[str] | None = None,
    atom_contact_threshold: float = 4.5,
    catalytic_contact_threshold: float = 4.5,
    substrate_clash_threshold: float = 2.8,
    mouth_residue_fraction: float = 0.30,
    default_pocket_file: str | None = None,
    default_catalytic_file: str | None = None,
    default_ligand_file: str | None = None,
    default_antigen_chain: str | None = None,
    default_nanobody_chain: str | None = None,
    pdb_structure_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Safe wrapper around process_one_pose that guarantees status fields."""
    try:
        return process_one_pose(
            row=row,
            base_dir=base_dir,
            optional_numeric_cols=optional_numeric_cols,
            atom_contact_threshold=atom_contact_threshold,
            catalytic_contact_threshold=catalytic_contact_threshold,
            substrate_clash_threshold=substrate_clash_threshold,
            mouth_residue_fraction=mouth_residue_fraction,
            default_pocket_file=default_pocket_file,
            default_catalytic_file=default_catalytic_file,
            default_ligand_file=default_ligand_file,
            default_antigen_chain=default_antigen_chain,
            default_nanobody_chain=default_nanobody_chain,
            pdb_structure_cache=pdb_structure_cache,
        )
    except Exception as exc:
        fallback = {
            "nanobody_id": _clean_str(row.get("nanobody_id")),
            "conformer_id": _clean_str(row.get("conformer_id")),
            "pose_id": _clean_str(row.get("pose_id")),
            "pdb_path": _clean_str(row.get("pdb_path")),
            "status": "failed",
            "error_message": _short_error_message(exc),
            "warning_message": "safe wrapper fallback",
            "split_mode": "",
        }
        return merge_optional_numeric_features(
            base_result=fallback,
            row=row,
            numeric_columns=optional_numeric_cols,
        )


def collect_feature_qc_summary(
    feature_df: pd.DataFrame,
    near_constant_threshold: float = 0.99,
    pocket_overwide_threshold: float = 0.55,
) -> dict[str, Any]:
    """Collect per-column QC summary for generated feature tables."""
    if feature_df.empty:
        return {
            "row_count": 0,
            "column_count": int(feature_df.shape[1]),
            "all_empty_columns": [],
            "near_constant_columns": [],
            "missing_ratio": {},
            "numeric_stats": {},
            "status_counts": {},
            "pocket_shape_qc": {},
        }

    row_count = int(feature_df.shape[0])
    missing_ratio = {
        col: float(feature_df[col].isna().mean())
        for col in feature_df.columns
    }

    all_empty_columns = [col for col, r in missing_ratio.items() if r >= 1.0]

    numeric_df = feature_df.select_dtypes(include=[np.number])
    numeric_stats: dict[str, dict[str, float]] = {}
    near_constant_columns: list[str] = []
    for col in numeric_df.columns:
        series = pd.to_numeric(numeric_df[col], errors="coerce")
        finite = series[np.isfinite(series.to_numpy(dtype=np.float64))]
        if finite.empty:
            continue

        value_counts = finite.value_counts(normalize=True, dropna=True)
        top_freq = float(value_counts.iloc[0]) if not value_counts.empty else 0.0
        if top_freq >= float(near_constant_threshold):
            near_constant_columns.append(col)

        numeric_stats[col] = {
            "count": float(finite.shape[0]),
            "mean": float(np.nanmean(finite.to_numpy(dtype=np.float64))),
            "std": float(np.nanstd(finite.to_numpy(dtype=np.float64))),
            "min": float(np.nanmin(finite.to_numpy(dtype=np.float64))),
            "p05": float(np.nanpercentile(finite.to_numpy(dtype=np.float64), 5)),
            "p50": float(np.nanpercentile(finite.to_numpy(dtype=np.float64), 50)),
            "p95": float(np.nanpercentile(finite.to_numpy(dtype=np.float64), 95)),
            "max": float(np.nanmax(finite.to_numpy(dtype=np.float64))),
            "top_value_ratio": top_freq,
        }

    status_counts = {}
    if "status" in feature_df.columns:
        vc = feature_df["status"].astype(str).value_counts(dropna=False)
        status_counts = {str(k): int(v) for k, v in vc.items()}

    pocket_shape_qc: dict[str, Any] = {}
    if "pocket_shape_overwide_proxy" in feature_df.columns:
        overwide = pd.to_numeric(feature_df["pocket_shape_overwide_proxy"], errors="coerce")
        finite = overwide[np.isfinite(overwide.to_numpy(dtype=np.float64))]
        threshold = float(np.clip(pocket_overwide_threshold, 0.0, 1.0))
        if not finite.empty:
            high_mask = overwide >= threshold
            high_rows = feature_df.loc[high_mask.fillna(False)].copy()
            id_cols = [
                col
                for col in [
                    "nanobody_id",
                    "conformer_id",
                    "pose_id",
                    "pdb_path",
                    "pocket_shape_residue_count",
                    "pocket_shape_overwide_proxy",
                    "pocket_shape_tightness_proxy",
                ]
                if col in high_rows.columns
            ]
            top_rows = []
            if id_cols:
                high_rows = high_rows.sort_values(by="pocket_shape_overwide_proxy", ascending=False)
                for _, high_row in high_rows.loc[:, id_cols].head(10).iterrows():
                    record: dict[str, Any] = {}
                    for col in id_cols:
                        value = high_row.get(col)
                        if isinstance(value, (np.integer, np.floating)):
                            value = float(value)
                        record[col] = value
                    top_rows.append(record)

            pocket_shape_qc = {
                "overwide_threshold": threshold,
                "finite_row_count": int(finite.shape[0]),
                "high_overwide_row_count": int(high_mask.fillna(False).sum()),
                "high_overwide_row_fraction": float(high_mask.fillna(False).sum() / max(int(finite.shape[0]), 1)),
                "overwide_proxy_mean": float(np.nanmean(finite.to_numpy(dtype=np.float64))),
                "overwide_proxy_p50": float(np.nanpercentile(finite.to_numpy(dtype=np.float64), 50)),
                "overwide_proxy_p95": float(np.nanpercentile(finite.to_numpy(dtype=np.float64), 95)),
                "overwide_proxy_max": float(np.nanmax(finite.to_numpy(dtype=np.float64))),
                "top_overwide_rows": top_rows,
            }

    qc = {
        "row_count": row_count,
        "column_count": int(feature_df.shape[1]),
        "all_empty_columns": sorted(all_empty_columns),
        "near_constant_columns": sorted(set(near_constant_columns)),
        "missing_ratio": missing_ratio,
        "numeric_stats": numeric_stats,
        "status_counts": status_counts,
        "pocket_shape_qc": pocket_shape_qc,
    }
    return qc


def summarize_processing_results(feature_df: pd.DataFrame) -> dict[str, Any]:
    """Build compact processing summary for batch run diagnostics."""
    total = int(len(feature_df))
    ok = int((feature_df.get("status", pd.Series([], dtype=object)) == "ok").sum()) if total > 0 else 0
    failed = int((feature_df.get("status", pd.Series([], dtype=object)) == "failed").sum()) if total > 0 else 0
    warning_non_empty = 0
    if "warning_message" in feature_df.columns and total > 0:
        warning_non_empty = int(
            feature_df["warning_message"].fillna("").astype(str).str.strip().ne("").sum()
        )

    split_mode_counts: dict[str, int] = {}
    if "split_mode" in feature_df.columns and total > 0:
        vc = feature_df["split_mode"].fillna("").astype(str).value_counts(dropna=False)
        split_mode_counts = {str(k): int(v) for k, v in vc.items() if str(k).strip()}

    return {
        "total_rows": total,
        "ok_rows": ok,
        "failed_rows": failed,
        "rows_with_warning_message": warning_non_empty,
        "split_mode_counts": split_mode_counts,
    }


def build_feature_table(
    df: pd.DataFrame,
    base_dir: str | Path,
    atom_contact_threshold: float = 4.5,
    catalytic_contact_threshold: float = 4.5,
    substrate_clash_threshold: float = 2.8,
    mouth_residue_fraction: float = 0.30,
    default_pocket_file: str | None = None,
    default_catalytic_file: str | None = None,
    default_ligand_file: str | None = None,
    default_antigen_chain: str | None = None,
    default_nanobody_chain: str | None = None,
    skip_failed_rows: bool = False,
    pdb_structure_cache: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build output feature table row-by-row in serial mode.

    This function is failure-tolerant: one row failure does not stop the whole
    table construction.
    """
    if df.empty:
        return pd.DataFrame(
            columns=list(REQUIRED_INPUT_COLUMNS)
            + [
                "status",
                "error_message",
                "warning_message",
                "split_mode",
                "num_antigen_residues",
                "num_nanobody_residues",
                "num_pocket_residues_defined",
                "num_pocket_residues_matched",
                "num_catalytic_residues_defined",
                "num_catalytic_residues_matched",
                "num_ligand_atoms",
            ]
        )

    numeric_cols = detect_optional_numeric_columns(df)
    rows: list[dict[str, Any]] = []
    local_cache = pdb_structure_cache if pdb_structure_cache is not None else {}

    for _, row in df.iterrows():
        out = safe_process_one_pose(
            row=row,
            base_dir=base_dir,
            optional_numeric_cols=numeric_cols,
            atom_contact_threshold=atom_contact_threshold,
            catalytic_contact_threshold=catalytic_contact_threshold,
            substrate_clash_threshold=substrate_clash_threshold,
            mouth_residue_fraction=mouth_residue_fraction,
            default_pocket_file=default_pocket_file,
            default_catalytic_file=default_catalytic_file,
            default_ligand_file=default_ligand_file,
            default_antigen_chain=default_antigen_chain,
            default_nanobody_chain=default_nanobody_chain,
            pdb_structure_cache=local_cache,
        )
        if skip_failed_rows and str(out.get("status", "")).lower() == "failed":
            continue
        rows.append(out)

    out_df = pd.DataFrame(rows)

    # Keep basic identifiers and status columns at the front.
    front_cols = [
        "nanobody_id",
        "conformer_id",
        "pose_id",
        "pdb_path",
        "status",
        "error_message",
        "warning_message",
        "split_mode",
        "num_antigen_residues",
        "num_nanobody_residues",
        "num_pocket_residues_defined",
        "num_pocket_residues_matched",
        "num_catalytic_residues_defined",
        "num_catalytic_residues_matched",
        "num_ligand_atoms",
    ]
    ordered_cols = [c for c in front_cols if c in out_df.columns] + [
        c for c in out_df.columns if c not in front_cols
    ]
    return out_df.loc[:, ordered_cols]


def save_feature_table(
    feature_df: pd.DataFrame,
    out_csv: str,
    qc_json_path: str | None = None,
) -> None:
    """Save feature table to CSV.

    Args:
        feature_df: Output feature DataFrame.
        out_csv: Destination CSV path.
        qc_json_path: Optional feature QC JSON output path.
    """
    out_path = Path(out_csv).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(out_path, index=False)

    qc_path = Path(qc_json_path).expanduser() if qc_json_path else out_path.with_name("feature_qc.json")
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    qc_payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "output_csv": str(out_path),
        "processing_summary": summarize_processing_results(feature_df),
        "feature_qc": collect_feature_qc_summary(feature_df),
    }
    qc_path.write_text(
        json.dumps(qc_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _build_cli_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Build pose feature table from input pose CSV."
    )
    parser.add_argument("--input_csv", required=True, help="Input pose CSV path.")
    parser.add_argument(
        "--out_csv",
        default="pose_features.csv",
        help="Output feature CSV path (default: pose_features.csv).",
    )
    parser.add_argument(
        "--atom_contact_threshold",
        type=float,
        default=4.5,
        help="Atom contact threshold in Angstrom (default: 4.5).",
    )
    parser.add_argument(
        "--catalytic_contact_threshold",
        type=float,
        default=4.5,
        help="Catalytic contact threshold in Angstrom (default: 4.5).",
    )
    parser.add_argument(
        "--substrate_clash_threshold",
        type=float,
        default=2.8,
        help="Substrate clash threshold in Angstrom (default: 2.8).",
    )
    parser.add_argument(
        "--mouth_residue_fraction",
        type=float,
        default=0.30,
        help="Fraction of outer pocket residues used as mouth residues (default: 0.30).",
    )
    parser.add_argument(
        "--default_pocket_file",
        default=None,
        help="Default pocket residue file used when a row has empty pocket_file.",
    )
    parser.add_argument(
        "--default_catalytic_file",
        default=None,
        help="Default catalytic residue file used when a row has empty catalytic_file.",
    )
    parser.add_argument(
        "--default_ligand_file",
        default=None,
        help="Default ligand template file used when a row has empty ligand_file.",
    )
    parser.add_argument(
        "--default_antigen_chain",
        default=None,
        help="Default antigen chain ID used when a row has empty antigen_chain.",
    )
    parser.add_argument(
        "--default_nanobody_chain",
        default=None,
        help="Default nanobody chain ID used when a row has empty nanobody_chain.",
    )
    parser.add_argument(
        "--skip_failed_rows",
        action="store_true",
        help="If set, failed rows are excluded from the output CSV.",
    )
    parser.add_argument(
        "--qc_json",
        default=None,
        help="Optional output path for feature QC JSON (default: feature_qc.json next to out_csv).",
    )
    return parser


def main() -> None:
    """CLI entry point."""
    parser = _build_cli_parser()
    args = parser.parse_args()

    input_path = Path(args.input_csv).expanduser()
    base_dir = input_path.parent.resolve()

    df = read_input_table(str(input_path))
    feature_df = build_feature_table(
        df=df,
        base_dir=base_dir,
        atom_contact_threshold=args.atom_contact_threshold,
        catalytic_contact_threshold=args.catalytic_contact_threshold,
        substrate_clash_threshold=args.substrate_clash_threshold,
        mouth_residue_fraction=args.mouth_residue_fraction,
        default_pocket_file=args.default_pocket_file,
        default_catalytic_file=args.default_catalytic_file,
        default_ligand_file=args.default_ligand_file,
        default_antigen_chain=args.default_antigen_chain,
        default_nanobody_chain=args.default_nanobody_chain,
        skip_failed_rows=bool(args.skip_failed_rows),
    )
    save_feature_table(feature_df, args.out_csv, qc_json_path=args.qc_json)

    ok_count = int((feature_df["status"] == "ok").sum()) if "status" in feature_df.columns else 0
    fail_count = int((feature_df["status"] == "failed").sum()) if "status" in feature_df.columns else 0
    summary = summarize_processing_results(feature_df)
    print(
        f"Feature table saved to {args.out_csv}. "
        f"Total={len(feature_df)}, ok={ok_count}, failed={fail_count}, "
        f"warnings={summary.get('rows_with_warning_message', 0)}"
    )


__all__ = [
    "read_input_table",
    "process_one_pose",
    "safe_process_one_pose",
    "build_feature_table",
    "merge_optional_numeric_features",
    "detect_optional_numeric_columns",
    "collect_feature_qc_summary",
    "summarize_processing_results",
    "save_feature_table",
]


if __name__ == "__main__":
    main()
