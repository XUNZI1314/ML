"""Calibrate CD38 pocket-shape proxy decisions from local benchmark artifacts.

This script does not run external pocket finders. It reads existing CD38
benchmark outputs, recomputes the runtime pocket_shape_overwide_proxy for each
predicted pocket, compares it with truth-based benchmark metrics, and writes a
conservative recommendation for threshold/penalty settings.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from geometry_features import compute_pocket_shape_features
from pdb_parser import load_complex_pdb
from pocket_io import load_residue_set


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build CD38 proxy calibration report")
    parser.add_argument("--panel_csv", default="benchmarks/cd38/results/cd38_benchmark_panel.csv")
    parser.add_argument("--sensitivity_dir", default="benchmarks/cd38/parameter_sensitivity")
    parser.add_argument("--out_dir", default="benchmarks/cd38/proxy_calibration")
    parser.add_argument("--runtime_proxy_threshold", type=float, default=0.55)
    parser.add_argument("--truth_overwide_threshold", type=float, default=0.55)
    parser.add_argument("--coverage_threshold", type=float, default=0.80)
    parser.add_argument("--low_precision_threshold", type=float, default=0.35)
    parser.add_argument("--min_structures_for_default_penalty", type=int, default=5)
    parser.add_argument("--threshold_grid", default="0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70")
    return parser


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null", "na", "n/a"} else text


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _read_csv(path: str | Path | None) -> pd.DataFrame:
    if path is None or not str(path).strip():
        return pd.DataFrame()
    target = Path(path).expanduser()
    if not target.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(target, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(target, encoding="utf-8-sig", low_memory=False)


def _float_grid(text: str) -> list[float]:
    values: list[float] = []
    for raw in str(text or "").split(","):
        item = raw.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise ValueError("threshold_grid cannot be empty.")
    return sorted(set(values))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
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


def _format_float(value: Any, digits: int = 4) -> str:
    number = _safe_float(value)
    if not np.isfinite(number):
        return ""
    return f"{number:.{digits}f}"


def _resolve_path(value: Any, base_dir: Path | None = None) -> Path | None:
    text = _clean_text(value)
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path.resolve()


def _predicted_pocket_file(row: pd.Series) -> Path | None:
    result_dir = _resolve_path(row.get("result_dir"))
    if result_dir is not None:
        candidate = result_dir / "predicted_pocket.txt"
        if candidate.exists():
            return candidate.resolve()
    return None


def _compute_runtime_proxy(row: pd.Series) -> dict[str, Any]:
    pdb_path = _resolve_path(row.get("pdb_path"))
    pocket_file = _predicted_pocket_file(row)
    if pdb_path is None or not pdb_path.exists():
        return {"proxy_status": "missing_pdb", "pocket_shape_overwide_proxy": np.nan}
    if pocket_file is None or not pocket_file.exists():
        return {"proxy_status": "missing_predicted_pocket", "pocket_shape_overwide_proxy": np.nan}
    try:
        residue_keys = load_residue_set(pocket_file)
        structure = load_complex_pdb(str(pdb_path))
        features = compute_pocket_shape_features(residue_keys, residue_reference=structure)
    except Exception as exc:
        return {
            "proxy_status": "failed",
            "proxy_error": str(exc)[:240],
            "pocket_shape_overwide_proxy": np.nan,
        }
    return {
        "proxy_status": "ok",
        "predicted_pocket_file": str(pocket_file),
        "pocket_shape_overwide_proxy": _safe_float(features.get("pocket_shape_overwide_proxy")),
        "pocket_shape_residue_count": _safe_float(features.get("pocket_shape_residue_count")),
        "pocket_shape_centroid_radius_p90": _safe_float(features.get("pocket_shape_centroid_radius_p90")),
        "pocket_shape_atom_bbox_volume": _safe_float(features.get("pocket_shape_atom_bbox_volume")),
    }


def _build_calibration_rows(
    panel_df: pd.DataFrame,
    *,
    runtime_proxy_threshold: float,
    truth_overwide_threshold: float,
    coverage_threshold: float,
    low_precision_threshold: float,
) -> pd.DataFrame:
    if panel_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for _, row in panel_df.iterrows():
        proxy = _compute_runtime_proxy(row)
        coverage = _safe_float(row.get("exact_truth_coverage"))
        precision = _safe_float(row.get("exact_predicted_precision"))
        truth_overwide = _safe_float(row.get("overwide_pocket_score"))
        high_coverage = bool(np.isfinite(coverage) and coverage >= float(coverage_threshold))
        low_precision = bool(np.isfinite(precision) and precision <= float(low_precision_threshold))
        truth_overwide_flag = bool(np.isfinite(truth_overwide) and truth_overwide >= float(truth_overwide_threshold))
        high_coverage_low_precision = bool(high_coverage and low_precision)
        runtime_proxy = _safe_float(proxy.get("pocket_shape_overwide_proxy"))
        runtime_proxy_flag = bool(np.isfinite(runtime_proxy) and runtime_proxy >= float(runtime_proxy_threshold))
        rows.append(
            {
                "result_name": _clean_text(row.get("result_name")),
                "method": _clean_text(row.get("method")),
                "pdb_id": _clean_text(row.get("pdb_id") or row.get("rcsb_pdb_id")),
                "predicted_residue_count": _safe_int(row.get("predicted_residue_count")),
                "truth_residue_count": _safe_int(row.get("truth_residue_count")),
                "exact_truth_coverage": coverage,
                "exact_predicted_precision": precision,
                "exact_f1": _safe_float(row.get("exact_f1")),
                "overwide_pocket_score": truth_overwide,
                "truth_overwide_flag": truth_overwide_flag,
                "high_coverage_low_precision_flag": high_coverage_low_precision,
                "runtime_proxy_threshold": float(runtime_proxy_threshold),
                "runtime_proxy_flag": runtime_proxy_flag,
                **proxy,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["truth_risk_label"] = out["truth_overwide_flag"] | out["high_coverage_low_precision_flag"]
        out["proxy_truth_gap"] = pd.to_numeric(out["pocket_shape_overwide_proxy"], errors="coerce") - pd.to_numeric(
            out["overwide_pocket_score"], errors="coerce"
        )
    return out


def _threshold_candidates(rows: pd.DataFrame, grid: list[float]) -> pd.DataFrame:
    if rows.empty or "truth_risk_label" not in rows.columns:
        return pd.DataFrame()
    valid = rows[
        pd.to_numeric(rows["pocket_shape_overwide_proxy"], errors="coerce").notna()
        & rows["truth_risk_label"].notna()
    ].copy()
    if valid.empty:
        return pd.DataFrame()

    y_true = valid["truth_risk_label"].astype(bool).to_numpy()
    proxy = pd.to_numeric(valid["pocket_shape_overwide_proxy"], errors="coerce").to_numpy(dtype=float)
    out_rows: list[dict[str, Any]] = []
    for threshold in grid:
        y_pred = proxy >= float(threshold)
        tp = int(np.sum(y_true & y_pred))
        tn = int(np.sum(~y_true & ~y_pred))
        fp = int(np.sum(~y_true & y_pred))
        fn = int(np.sum(y_true & ~y_pred))
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        precision = tp / max(tp + fp, 1)
        accuracy = (tp + tn) / max(len(y_true), 1)
        f1 = 2 * precision * sensitivity / max(precision + sensitivity, 1e-12)
        out_rows.append(
            {
                "threshold": float(threshold),
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "precision": float(precision),
                "accuracy": float(accuracy),
                "balanced_accuracy": float(0.5 * (sensitivity + specificity)),
                "f1": float(f1),
            }
        )
    return pd.DataFrame(out_rows).sort_values(
        ["balanced_accuracy", "f1", "specificity", "threshold"],
        ascending=[False, False, False, True],
    )


def _load_sensitivity(sensitivity_dir: Path) -> dict[str, pd.DataFrame]:
    names = {
        "contact_cutoff": "contact_cutoff_sensitivity.csv",
        "p2rank_rank": "p2rank_rank_sensitivity.csv",
        "fpocket_pocket": "fpocket_pocket_sensitivity.csv",
        "method_consensus": "method_consensus_threshold_sensitivity.csv",
        "overwide_penalty": "overwide_penalty_sensitivity.csv",
    }
    return {key: _read_csv(sensitivity_dir / filename) for key, filename in names.items()}


def _method_summary(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    work = rows.copy()
    for column in [
        "exact_truth_coverage",
        "exact_predicted_precision",
        "overwide_pocket_score",
        "pocket_shape_overwide_proxy",
    ]:
        work[column] = pd.to_numeric(work.get(column), errors="coerce")
    grouped = work.groupby("method", dropna=False).agg(
        row_count=("result_name", "count"),
        pdb_count=("pdb_id", "nunique"),
        mean_coverage=("exact_truth_coverage", "mean"),
        mean_precision=("exact_predicted_precision", "mean"),
        mean_truth_overwide=("overwide_pocket_score", "mean"),
        mean_runtime_proxy=("pocket_shape_overwide_proxy", "mean"),
        truth_risk_count=("truth_risk_label", "sum"),
        runtime_proxy_flag_count=("runtime_proxy_flag", "sum"),
    )
    return grouped.reset_index()


def _penalty_summary(overwide_penalty_df: pd.DataFrame) -> pd.DataFrame:
    if overwide_penalty_df.empty or "overwide_penalty_weight" not in overwide_penalty_df.columns:
        return pd.DataFrame()
    work = overwide_penalty_df.copy()
    for column in ["overwide_penalty_weight", "utility_score", "overwide_risk", "adjusted_utility_score"]:
        work[column] = pd.to_numeric(work.get(column), errors="coerce")
    grouped = work.groupby("overwide_penalty_weight", dropna=False).agg(
        scenario_count=("scenario_type", "count"),
        mean_overwide_risk=("overwide_risk", "mean"),
    )
    finite = work[work["utility_score"].notna() & work["adjusted_utility_score"].notna()].copy()
    paired = finite.groupby("overwide_penalty_weight", dropna=False).agg(
        paired_scenario_count=("scenario_type", "count"),
        mean_utility_score=("utility_score", "mean"),
        mean_adjusted_utility_score=("adjusted_utility_score", "mean"),
        min_adjusted_utility_score=("adjusted_utility_score", "min"),
    )
    return grouped.join(paired, how="left").reset_index().sort_values("overwide_penalty_weight")


def _recommendation(
    rows: pd.DataFrame,
    threshold_df: pd.DataFrame,
    *,
    runtime_proxy_threshold: float,
    min_structures_for_default_penalty: int,
) -> dict[str, Any]:
    pdb_count = int(rows["pdb_id"].nunique()) if not rows.empty and "pdb_id" in rows.columns else 0
    row_count = int(len(rows))
    method_count = int(rows["method"].nunique()) if not rows.empty and "method" in rows.columns else 0
    fpocket_count = int(rows["method"].astype(str).str.lower().eq("fpocket").sum()) if "method" in rows.columns else 0
    valid_proxy_count = int(pd.to_numeric(rows.get("pocket_shape_overwide_proxy"), errors="coerce").notna().sum()) if not rows.empty else 0
    truth_risk_count = int(rows["truth_risk_label"].astype(bool).sum()) if "truth_risk_label" in rows.columns else 0
    non_risk_count = row_count - truth_risk_count
    enough_structures = pdb_count >= int(min_structures_for_default_penalty)
    enough_classes = truth_risk_count >= 2 and non_risk_count >= 2
    enough_methods = method_count >= 3 and fpocket_count > 0

    best_threshold = float(runtime_proxy_threshold)
    best_threshold_score = None
    if not threshold_df.empty:
        best = threshold_df.iloc[0]
        best_threshold = float(best.get("threshold"))
        best_threshold_score = {
            "balanced_accuracy": _safe_float(best.get("balanced_accuracy")),
            "f1": _safe_float(best.get("f1")),
            "sensitivity": _safe_float(best.get("sensitivity")),
            "specificity": _safe_float(best.get("specificity")),
        }

    blockers: list[str] = []
    if not enough_structures:
        blockers.append(f"benchmark structures are too few ({pdb_count} < {int(min_structures_for_default_penalty)})")
    if not enough_classes:
        blockers.append(f"truth risk/non-risk split is too small ({truth_risk_count}/{non_risk_count})")
    if not enough_methods:
        blockers.append("no real fpocket rows and fewer than three method families are present")
    if valid_proxy_count < row_count:
        blockers.append(f"runtime proxy could not be computed for all rows ({valid_proxy_count}/{row_count})")

    if blockers:
        return {
            "evidence_level": "low",
            "recommended_runtime_proxy_threshold": float(runtime_proxy_threshold),
            "recommended_default_penalty_weight": 0.0,
            "optional_experimental_penalty_weight": 0.15,
            "recommended_policy": "keep_default_penalty_off",
            "best_empirical_threshold": best_threshold,
            "best_empirical_threshold_score": best_threshold_score,
            "reason": "Current benchmark is useful for direction, but not large/diverse enough to change defaults.",
            "blockers": blockers,
            "next_actions": [
                "Add real fpocket outputs for 3ROP/4OGW/3F6Y and rerun finalize_cd38_external_benchmark.py --run_discovered --run_sensitivity.",
                "Add at least one non-CD38 benchmark protein before changing global ranking defaults.",
                "Keep --pocket_overwide_penalty_weight at 0.0 by default; use 0.15 only for sensitivity review runs.",
            ],
        }

    return {
        "evidence_level": "moderate",
        "recommended_runtime_proxy_threshold": best_threshold,
        "recommended_default_penalty_weight": 0.10,
        "optional_experimental_penalty_weight": 0.15,
        "recommended_policy": "enable_small_penalty_after_review",
        "best_empirical_threshold": best_threshold,
        "best_empirical_threshold_score": best_threshold_score,
        "reason": "Benchmark coverage is broad enough to consider a small default overwide penalty after manual review.",
        "blockers": [],
        "next_actions": [
            "Review threshold candidates before enabling the penalty in default pipeline templates.",
            "Continue tracking false positives and false negatives in cd38_proxy_calibration_rows.csv.",
        ],
    }


def _records(df: pd.DataFrame, max_rows: int = 10) -> list[dict[str, Any]]:
    if df.empty:
        return []
    return _json_safe(df.head(max_rows).replace({np.nan: None}).to_dict(orient="records"))


def _markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    view = df.loc[:, [col for col in columns if col in df.columns]].head(max_rows).copy()
    if view.empty:
        return "_No rows._"
    for col in view.columns:
        if pd.api.types.is_numeric_dtype(view[col]):
            view[col] = view[col].map(lambda value: _format_float(value) if pd.notna(value) else "")
    header = "| " + " | ".join(view.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = ["| " + " | ".join(str(row.get(col, "")).replace("|", "\\|") for col in view.columns) + " |" for _, row in view.iterrows()]
    return "\n".join([header, sep] + rows)


def _build_report(
    summary: dict[str, Any],
    rows: pd.DataFrame,
    thresholds: pd.DataFrame,
    methods: pd.DataFrame,
    penalty: pd.DataFrame,
) -> str:
    rec = summary.get("recommendation", {})
    lines = [
        "# CD38 Proxy Calibration Report",
        "",
        "This report compares the runtime `pocket_shape_overwide_proxy` against CD38 truth-based benchmark metrics.",
        "It is a calibration aid only; it does not change ranking defaults.",
        "",
        "## Recommendation",
        "",
        f"- Evidence level: `{rec.get('evidence_level', '')}`",
        f"- Policy: `{rec.get('recommended_policy', '')}`",
        f"- Recommended runtime proxy threshold: `{_format_float(rec.get('recommended_runtime_proxy_threshold'))}`",
        f"- Recommended default penalty weight: `{_format_float(rec.get('recommended_default_penalty_weight'))}`",
        f"- Optional experimental penalty weight: `{_format_float(rec.get('optional_experimental_penalty_weight'))}`",
        f"- Reason: {rec.get('reason', '')}",
        "",
        "## Current Evidence Coverage",
        "",
        f"- Benchmark rows: `{summary.get('benchmark_row_count', 0)}`",
        f"- PDB structures: `{summary.get('pdb_count', 0)}`",
        f"- Methods: `{summary.get('method_count', 0)}`",
        f"- fpocket rows: `{summary.get('fpocket_row_count', 0)}`",
        f"- Truth-risk rows: `{summary.get('truth_risk_count', 0)}`",
        f"- Runtime proxy computed rows: `{summary.get('valid_proxy_count', 0)}`",
        "",
        "## Blockers Before Changing Defaults",
        "",
    ]
    blockers = rec.get("blockers") if isinstance(rec.get("blockers"), list) else []
    if blockers:
        lines.extend([f"- {item}" for item in blockers])
    else:
        lines.append("- No blocker from the current evidence gate.")
    lines.extend(
        [
            "",
            "## Next Actions",
            "",
        ]
    )
    for action in rec.get("next_actions", []) or []:
        lines.append(f"- {action}")
    lines.extend(
        [
            "",
            "## Calibration Rows",
            "",
            _markdown_table(
                rows,
                [
                    "result_name",
                    "method",
                    "pdb_id",
                    "exact_truth_coverage",
                    "exact_predicted_precision",
                    "overwide_pocket_score",
                    "pocket_shape_overwide_proxy",
                    "truth_risk_label",
                    "runtime_proxy_flag",
                    "proxy_status",
                ],
            ),
            "",
            "## Method Summary",
            "",
            _markdown_table(
                methods,
                [
                    "method",
                    "row_count",
                    "pdb_count",
                    "mean_coverage",
                    "mean_precision",
                    "mean_truth_overwide",
                    "mean_runtime_proxy",
                    "truth_risk_count",
                    "runtime_proxy_flag_count",
                ],
            ),
            "",
            "## Threshold Candidates",
            "",
            _markdown_table(
                thresholds,
                [
                    "threshold",
                    "tp",
                    "tn",
                    "fp",
                    "fn",
                    "sensitivity",
                    "specificity",
                    "precision",
                    "balanced_accuracy",
                    "f1",
                ],
                max_rows=20,
            ),
            "",
            "## Penalty Simulation Summary",
            "",
            _markdown_table(
                penalty,
                [
                    "overwide_penalty_weight",
                    "scenario_count",
                    "paired_scenario_count",
                    "mean_utility_score",
                    "mean_overwide_risk",
                    "mean_adjusted_utility_score",
                    "min_adjusted_utility_score",
                ],
            ),
            "",
            "## Output Files",
            "",
            f"- Summary JSON: `{summary['outputs']['summary_json']}`",
            f"- Report Markdown: `{summary['outputs']['report_md']}`",
            f"- Calibration rows CSV: `{summary['outputs']['calibration_rows_csv']}`",
            f"- Threshold candidates CSV: `{summary['outputs']['threshold_candidates_csv']}`",
            f"- Method summary CSV: `{summary['outputs']['method_summary_csv']}`",
            f"- Penalty summary CSV: `{summary['outputs']['penalty_summary_csv']}`",
            "",
        ]
    )
    return "\n".join(lines)


def build_cd38_proxy_calibration_outputs(
    *,
    panel_csv: str | Path = "benchmarks/cd38/results/cd38_benchmark_panel.csv",
    sensitivity_dir: str | Path = "benchmarks/cd38/parameter_sensitivity",
    out_dir: str | Path = "benchmarks/cd38/proxy_calibration",
    runtime_proxy_threshold: float = 0.55,
    truth_overwide_threshold: float = 0.55,
    coverage_threshold: float = 0.80,
    low_precision_threshold: float = 0.35,
    min_structures_for_default_penalty: int = 5,
    threshold_grid: str = "0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70",
) -> dict[str, Any]:
    panel_path = Path(panel_csv).expanduser().resolve()
    sensitivity_path = Path(sensitivity_dir).expanduser().resolve()
    output_dir = Path(out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    panel_df = _read_csv(panel_path)
    rows = _build_calibration_rows(
        panel_df,
        runtime_proxy_threshold=float(runtime_proxy_threshold),
        truth_overwide_threshold=float(truth_overwide_threshold),
        coverage_threshold=float(coverage_threshold),
        low_precision_threshold=float(low_precision_threshold),
    )
    thresholds = _threshold_candidates(rows, _float_grid(threshold_grid))
    methods = _method_summary(rows)
    sensitivity = _load_sensitivity(sensitivity_path)
    penalty = _penalty_summary(sensitivity.get("overwide_penalty", pd.DataFrame()))
    recommendation = _recommendation(
        rows,
        thresholds,
        runtime_proxy_threshold=float(runtime_proxy_threshold),
        min_structures_for_default_penalty=int(min_structures_for_default_penalty),
    )

    summary_json = output_dir / "cd38_proxy_calibration_summary.json"
    report_md = output_dir / "cd38_proxy_calibration_report.md"
    rows_csv = output_dir / "cd38_proxy_calibration_rows.csv"
    thresholds_csv = output_dir / "cd38_proxy_threshold_candidates.csv"
    methods_csv = output_dir / "cd38_proxy_method_summary.csv"
    penalty_csv = output_dir / "cd38_proxy_penalty_summary.csv"

    rows.to_csv(rows_csv, index=False, encoding="utf-8")
    thresholds.to_csv(thresholds_csv, index=False, encoding="utf-8")
    methods.to_csv(methods_csv, index=False, encoding="utf-8")
    penalty.to_csv(penalty_csv, index=False, encoding="utf-8")

    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "panel_csv": str(panel_path),
        "sensitivity_dir": str(sensitivity_path),
        "benchmark_row_count": int(len(rows)),
        "pdb_count": int(rows["pdb_id"].nunique()) if not rows.empty and "pdb_id" in rows.columns else 0,
        "method_count": int(rows["method"].nunique()) if not rows.empty and "method" in rows.columns else 0,
        "fpocket_row_count": int(rows["method"].astype(str).str.lower().eq("fpocket").sum()) if "method" in rows.columns else 0,
        "truth_risk_count": int(rows["truth_risk_label"].astype(bool).sum()) if "truth_risk_label" in rows.columns else 0,
        "valid_proxy_count": int(pd.to_numeric(rows.get("pocket_shape_overwide_proxy"), errors="coerce").notna().sum()) if not rows.empty else 0,
        "thresholds": {
            "runtime_proxy_threshold": float(runtime_proxy_threshold),
            "truth_overwide_threshold": float(truth_overwide_threshold),
            "coverage_threshold": float(coverage_threshold),
            "low_precision_threshold": float(low_precision_threshold),
            "min_structures_for_default_penalty": int(min_structures_for_default_penalty),
        },
        "recommendation": recommendation,
        "top_threshold_candidates": _records(thresholds, max_rows=5),
        "method_summary": _records(methods, max_rows=20),
        "penalty_summary": _records(penalty, max_rows=20),
        "outputs": {
            "summary_json": str(summary_json),
            "report_md": str(report_md),
            "calibration_rows_csv": str(rows_csv),
            "threshold_candidates_csv": str(thresholds_csv),
            "method_summary_csv": str(methods_csv),
            "penalty_summary_csv": str(penalty_csv),
        },
    }
    summary = _json_safe(summary)
    summary_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    report_md.write_text(_build_report(summary, rows, thresholds, methods, penalty), encoding="utf-8")
    return summary


def main() -> None:
    args = _build_parser().parse_args()
    summary = build_cd38_proxy_calibration_outputs(
        panel_csv=args.panel_csv,
        sensitivity_dir=args.sensitivity_dir,
        out_dir=args.out_dir,
        runtime_proxy_threshold=float(args.runtime_proxy_threshold),
        truth_overwide_threshold=float(args.truth_overwide_threshold),
        coverage_threshold=float(args.coverage_threshold),
        low_precision_threshold=float(args.low_precision_threshold),
        min_structures_for_default_penalty=int(args.min_structures_for_default_penalty),
        threshold_grid=str(args.threshold_grid),
    )
    outputs = summary.get("outputs", {}) if isinstance(summary.get("outputs"), dict) else {}
    for key, value in outputs.items():
        print(f"Saved {key}: {value}")
    rec = summary.get("recommendation", {}) if isinstance(summary.get("recommendation"), dict) else {}
    print(f"Recommendation: {rec.get('recommended_policy', 'unknown')} (evidence={rec.get('evidence_level', 'unknown')})")


if __name__ == "__main__":
    main()
