from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


SKIP_LOCAL_APP_DIR_NAMES = {
    "_bundle_imports",
    "_generated_inputs",
    "compare_exports",
    "experiment_validation_feedback",
    "parameter_templates",
    "result_archive",
    "validation_retrain_comparisons",
}

KEY_ARTIFACTS = {
    "recommended_pipeline_summary_json": ("recommended_pipeline_summary.json",),
    "execution_report_md": ("recommended_pipeline_report.md",),
    "quality_gate_summary_json": ("quality_gate", "quality_gate_summary.json"),
    "quality_gate_checks_csv": ("quality_gate", "quality_gate_checks.csv"),
    "quality_gate_report_md": ("quality_gate", "quality_gate_report.md"),
    "batch_decision_summary_json": ("batch_decision_summary", "batch_decision_summary.json"),
    "batch_decision_summary_md": ("batch_decision_summary", "batch_decision_summary.md"),
    "batch_decision_summary_cards_csv": ("batch_decision_summary", "batch_decision_summary_cards.csv"),
    "consensus_ranking_csv": ("consensus_outputs", "consensus_ranking.csv"),
    "consensus_report_md": ("consensus_outputs", "consensus_report.md"),
    "score_explanation_cards_csv": ("score_explanation_cards", "score_explanation_cards.csv"),
    "score_explanation_cards_html": ("score_explanation_cards", "score_explanation_cards.html"),
    "score_explanation_cards_md": ("score_explanation_cards", "score_explanation_cards.md"),
    "candidate_report_index_html": ("candidate_report_cards", "index.html"),
    "candidate_report_zip": ("candidate_report_cards.zip",),
    "candidate_comparison_report_md": ("candidate_comparisons", "candidate_comparison_report.md"),
    "candidate_group_comparison_summary_csv": ("candidate_comparisons", "candidate_group_comparison_summary.csv"),
    "experiment_suggestions_csv": ("experiment_suggestions", "next_experiment_suggestions.csv"),
    "experiment_plan_md": ("experiment_suggestions", "experiment_plan.md"),
    "experiment_plan_state_ledger_csv": ("experiment_suggestions", "experiment_plan_state_ledger.csv"),
    "training_summary_json": ("model_outputs", "training_summary.json"),
    "pose_predictions_csv": ("model_outputs", "pose_predictions.csv"),
    "feature_qc_json": ("feature_qc.json",),
    "baseline_comparison_report_md": ("comparison_rule_vs_ml", "ranking_comparison_report.md"),
    "calibrated_comparison_report_md": ("comparison_calibrated_rule_vs_ml", "ranking_comparison_report.md"),
    "improvement_report_md": ("improvement_summary", "calibration_improvement_report.md"),
    "strategy_report_md": ("strategy_optimization", "recommended_strategy_report.md"),
    "run_provenance_card_md": ("provenance", "run_provenance_card.md"),
    "run_provenance_card_json": ("provenance", "run_provenance_card.json"),
    "run_artifact_manifest_csv": ("provenance", "run_artifact_manifest.csv"),
    "run_input_file_manifest_csv": ("provenance", "run_input_file_manifest.csv"),
    "run_provenance_integrity_json": ("provenance", "run_provenance_integrity.json"),
    "ai_run_summary_md": ("ai_outputs", "ai_run_summary.md"),
}

RUN_INDEX_COLUMNS = [
    "run_name",
    "status",
    "started_at",
    "start_mode",
    "execution_mode",
    "label_col",
    "label_valid_count",
    "label_class_count",
    "calibration_possible",
    "baseline_rank_spearman",
    "baseline_rule_auc",
    "calibrated_rank_spearman",
    "calibrated_rule_auc",
    "training_mode",
    "train_rows",
    "val_rows",
    "feature_count",
    "best_epoch",
    "best_val_loss",
    "failed_rows",
    "warning_rows",
    "run_root",
    "out_dir",
    "summary_json",
]

ARTIFACT_COLUMNS = [
    "run_name",
    "artifact_key",
    "path",
    "exists",
    "size_bytes",
    "modified_at",
]

VALIDATION_TREND_COLUMNS = [
    "comparison_name",
    "before_run",
    "after_run",
    "top_k",
    "top_k_overlap_count",
    "top_k_overlap_fraction",
    "top_k_overlap_jaccard",
    "validated_top_k_count",
    "label_valid_delta",
    "baseline_rank_spearman_delta",
    "baseline_rule_auc_delta",
    "calibrated_rank_spearman_delta",
    "calibrated_rule_auc_delta",
    "best_val_loss_delta",
    "before_label_col",
    "after_label_col",
    "summary_json",
    "report_md",
]

LINEAGE_COLUMNS = [
    "run_name",
    "started_at",
    "start_mode",
    "input_csv",
    "feature_csv",
    "input_csv_sha256",
    "feature_csv_sha256",
    "parameter_hash",
    "file_manifest_hash",
    "input_file_manifest_hash",
    "input_reference_count",
    "input_missing_count",
    "git_commit",
    "git_dirty",
    "same_input_manifest_group_size",
    "same_feature_hash_group_size",
    "same_parameter_hash_group_size",
    "previous_run_with_same_input_manifest",
    "previous_run_with_same_feature_hash",
    "previous_run_with_same_parameter_hash",
    "summary_json",
    "provenance_json",
]

LINEAGE_GRAPH_SPECS = [
    {
        "edge_type": "input_manifest",
        "hash_column": "input_file_manifest_hash",
        "size_column": "same_input_manifest_group_size",
        "previous_column": "previous_run_with_same_input_manifest",
        "label": "Input file manifest",
        "color": "#2563eb",
    },
    {
        "edge_type": "feature_csv",
        "hash_column": "feature_csv_sha256",
        "size_column": "same_feature_hash_group_size",
        "previous_column": "previous_run_with_same_feature_hash",
        "label": "Feature CSV",
        "color": "#059669",
    },
    {
        "edge_type": "parameters",
        "hash_column": "parameter_hash",
        "size_column": "same_parameter_hash_group_size",
        "previous_column": "previous_run_with_same_parameter_hash",
        "label": "Parameters",
        "color": "#d97706",
    },
]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def _delta(after_value: Any, before_value: Any) -> float | None:
    after_num = _safe_float(after_value)
    before_num = _safe_float(before_value)
    if after_num is None or before_num is None:
        return None
    return float(after_num - before_num)


def _path_mtime_text(path: Path) -> str:
    if not path.exists():
        return ""
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


def _resolve_artifact_path(summary: dict[str, Any], out_dir: Path, artifact_key: str) -> Path:
    artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
    artifact_text = str(artifacts.get(artifact_key) or "").strip()
    if artifact_text:
        artifact_path = Path(artifact_text).expanduser()
        if not artifact_path.is_absolute():
            artifact_path = out_dir / artifact_path
        return artifact_path.resolve()
    fallback = KEY_ARTIFACTS.get(artifact_key, ())
    return out_dir.joinpath(*fallback).resolve() if fallback else out_dir.resolve()


def _extract_manifest_hash(card: dict[str, Any], label: str) -> str:
    records = card.get("file_manifest") if isinstance(card.get("file_manifest"), list) else []
    for record in records:
        if not isinstance(record, dict):
            continue
        if record.get("group") == "input" and record.get("label") == label:
            return str(record.get("sha256") or "")
    return ""


def _load_provenance_card(summary: dict[str, Any], out_dir: Path) -> tuple[dict[str, Any], Path]:
    artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
    path_text = str(artifacts.get("run_provenance_card_json") or "").strip()
    path = Path(path_text).expanduser() if path_text else out_dir / "provenance" / "run_provenance_card.json"
    if not path.is_absolute():
        path = out_dir / path
    path = path.resolve()
    return _read_json(path), path


def _extract_lineage_row(run_root: Path, run_row: dict[str, Any] | None) -> dict[str, Any] | None:
    if run_row is None:
        return None
    summary_path_text = str(run_row.get("summary_json") or "")
    if not summary_path_text:
        return None
    summary_path = Path(summary_path_text).expanduser().resolve()
    summary = _read_json(summary_path)
    if not summary:
        return None
    out_dir = Path(str(summary.get("out_dir") or run_row.get("out_dir") or run_root / "outputs")).expanduser().resolve()
    provenance_card, provenance_path = _load_provenance_card(summary, out_dir)
    summary_provenance = summary.get("provenance") if isinstance(summary.get("provenance"), dict) else {}
    card_input_manifest = (
        provenance_card.get("input_file_manifest")
        if isinstance(provenance_card.get("input_file_manifest"), dict)
        else {}
    )
    summary_input_manifest = (
        summary_provenance.get("input_file_manifest")
        if isinstance(summary_provenance.get("input_file_manifest"), dict)
        else {}
    )
    git_payload = provenance_card.get("git") if isinstance(provenance_card.get("git"), dict) else {}
    return {
        "run_name": run_root.name,
        "started_at": run_row.get("started_at"),
        "start_mode": summary.get("start_mode") or run_row.get("start_mode"),
        "input_csv": summary.get("input_csv"),
        "feature_csv": summary.get("feature_csv"),
        "input_csv_sha256": _extract_manifest_hash(provenance_card, "input_csv"),
        "feature_csv_sha256": _extract_manifest_hash(provenance_card, "feature_csv"),
        "parameter_hash": summary_provenance.get("parameter_hash") or provenance_card.get("parameter_hash"),
        "file_manifest_hash": summary_provenance.get("file_manifest_hash") or provenance_card.get("file_manifest_hash"),
        "input_file_manifest_hash": summary_input_manifest.get("manifest_hash") or card_input_manifest.get("manifest_hash"),
        "input_reference_count": summary_input_manifest.get("reference_count") or card_input_manifest.get("reference_count"),
        "input_missing_count": summary_input_manifest.get("missing_count") or card_input_manifest.get("missing_count"),
        "git_commit": git_payload.get("commit"),
        "git_dirty": git_payload.get("dirty"),
        "same_input_manifest_group_size": 0,
        "same_feature_hash_group_size": 0,
        "same_parameter_hash_group_size": 0,
        "previous_run_with_same_input_manifest": "",
        "previous_run_with_same_feature_hash": "",
        "previous_run_with_same_parameter_hash": "",
        "summary_json": str(summary_path),
        "provenance_json": str(provenance_path) if provenance_path.exists() else "",
    }


def _load_training_payload(out_dir: Path) -> dict[str, Any]:
    return _read_json(out_dir / "model_outputs" / "training_summary.json")


def _load_feature_qc_payload(summary: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
    qc_text = str(artifacts.get("feature_qc_json") or "").strip()
    if qc_text:
        qc_path = Path(qc_text).expanduser()
        if not qc_path.is_absolute():
            qc_path = out_dir / qc_path
        return _read_json(qc_path.resolve())
    return _read_json(out_dir / "feature_qc.json")


def _extract_run_metrics(run_root: Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    metadata = _read_json(run_root / "app_run_metadata.json")
    summary_path = run_root / "outputs" / "recommended_pipeline_summary.json"
    summary = _read_json(summary_path)
    if not metadata and not summary:
        return None, []

    out_dir = Path(str(summary.get("out_dir") or metadata.get("out_dir") or run_root / "outputs")).expanduser().resolve()
    comparison = summary.get("comparison") if isinstance(summary.get("comparison"), dict) else {}
    baseline = comparison.get("baseline_rule_vs_ml") if isinstance(comparison.get("baseline_rule_vs_ml"), dict) else {}
    calibrated = comparison.get("calibrated_rule_vs_ml") if isinstance(comparison.get("calibrated_rule_vs_ml"), dict) else {}
    training_payload = _load_training_payload(out_dir)
    training_summary = (
        training_payload.get("summary")
        if isinstance(training_payload.get("summary"), dict)
        else {}
    )
    feature_qc_payload = _load_feature_qc_payload(summary, out_dir)
    processing_summary = (
        feature_qc_payload.get("processing_summary")
        if isinstance(feature_qc_payload.get("processing_summary"), dict)
        else {}
    )

    row = {
        "run_name": run_root.name,
        "status": str(metadata.get("status") or ("success" if summary else "unknown")),
        "started_at": str(
            metadata.get("started_at")
            or metadata.get("created_at")
            or _path_mtime_text(run_root)
        ),
        "start_mode": str(metadata.get("start_mode") or summary.get("start_mode") or ""),
        "execution_mode": str(metadata.get("execution_mode") or ""),
        "label_col": summary.get("label_col"),
        "label_valid_count": summary.get("label_valid_count"),
        "label_class_count": summary.get("label_class_count"),
        "calibration_possible": summary.get("calibration_possible"),
        "baseline_rank_spearman": baseline.get("rank_spearman"),
        "baseline_rule_auc": baseline.get("rule_auc"),
        "calibrated_rank_spearman": calibrated.get("rank_spearman"),
        "calibrated_rule_auc": calibrated.get("rule_auc"),
        "training_mode": training_payload.get("mode"),
        "train_rows": training_payload.get("n_rows_train"),
        "val_rows": training_payload.get("n_rows_val"),
        "feature_count": training_payload.get("n_features"),
        "best_epoch": training_summary.get("best_epoch"),
        "best_val_loss": training_summary.get("best_val_loss"),
        "failed_rows": processing_summary.get("failed_rows"),
        "warning_rows": processing_summary.get("rows_with_warning_message"),
        "run_root": str(run_root),
        "out_dir": str(out_dir),
        "summary_json": str(summary_path) if summary_path.exists() else "",
    }

    artifact_rows: list[dict[str, Any]] = []
    for artifact_key in KEY_ARTIFACTS:
        artifact_path = _resolve_artifact_path(summary, out_dir, artifact_key)
        exists = artifact_path.exists()
        artifact_rows.append(
            {
                "run_name": run_root.name,
                "artifact_key": artifact_key,
                "path": str(artifact_path),
                "exists": bool(exists),
                "size_bytes": int(artifact_path.stat().st_size) if exists and artifact_path.is_file() else None,
                "modified_at": _path_mtime_text(artifact_path) if exists else "",
            }
        )

    return row, artifact_rows


def _iter_run_roots(local_app_runs: Path) -> list[Path]:
    if not local_app_runs.exists():
        return []
    roots: list[Path] = []
    for child in local_app_runs.iterdir():
        if not child.is_dir():
            continue
        if child.name in SKIP_LOCAL_APP_DIR_NAMES or child.name.startswith("_"):
            continue
        if (child / "app_run_metadata.json").exists() or (child / "outputs" / "recommended_pipeline_summary.json").exists():
            roots.append(child.resolve())
    return sorted(roots, key=lambda path: path.stat().st_mtime, reverse=True)


def _run_name_from_summary_path(path_text: str) -> str:
    if not path_text:
        return ""
    path = Path(path_text)
    parts = list(path.parts)
    if "local_app_runs" in parts:
        idx = parts.index("local_app_runs")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    if len(path.parents) >= 2:
        return path.parents[1].name
    return path.stem


def _extract_validation_trend_row(summary_path: Path) -> dict[str, Any]:
    payload = _read_json(summary_path)
    top_k_overlap = payload.get("top_k_overlap") if isinstance(payload.get("top_k_overlap"), dict) else {}
    before_metrics = payload.get("before_metrics") if isinstance(payload.get("before_metrics"), dict) else {}
    after_metrics = payload.get("after_metrics") if isinstance(payload.get("after_metrics"), dict) else {}
    outputs = payload.get("outputs") if isinstance(payload.get("outputs"), dict) else {}
    before_summary = str(payload.get("before_summary") or "")
    after_summary = str(payload.get("after_summary") or "")
    return {
        "comparison_name": summary_path.parent.name,
        "before_run": _run_name_from_summary_path(before_summary),
        "after_run": _run_name_from_summary_path(after_summary),
        "top_k": payload.get("top_k"),
        "top_k_overlap_count": top_k_overlap.get("overlap_count"),
        "top_k_overlap_fraction": top_k_overlap.get("fraction"),
        "top_k_overlap_jaccard": top_k_overlap.get("jaccard"),
        "validated_top_k_count": payload.get("validated_top_k_count"),
        "label_valid_delta": _delta(after_metrics.get("label_valid_count"), before_metrics.get("label_valid_count")),
        "baseline_rank_spearman_delta": _delta(
            after_metrics.get("baseline_rank_spearman"),
            before_metrics.get("baseline_rank_spearman"),
        ),
        "baseline_rule_auc_delta": _delta(after_metrics.get("baseline_rule_auc"), before_metrics.get("baseline_rule_auc")),
        "calibrated_rank_spearman_delta": _delta(
            after_metrics.get("calibrated_rank_spearman"),
            before_metrics.get("calibrated_rank_spearman"),
        ),
        "calibrated_rule_auc_delta": _delta(
            after_metrics.get("calibrated_rule_auc"),
            before_metrics.get("calibrated_rule_auc"),
        ),
        "best_val_loss_delta": _delta(after_metrics.get("best_val_loss"), before_metrics.get("best_val_loss")),
        "before_label_col": before_metrics.get("label_col"),
        "after_label_col": after_metrics.get("label_col"),
        "summary_json": str(summary_path),
        "report_md": str(outputs.get("report_md") or summary_path.parent / "validation_retrain_comparison_report.md"),
    }


def _build_validation_trend_table(local_app_runs: Path) -> pd.DataFrame:
    root = local_app_runs / "validation_retrain_comparisons"
    if not root.exists():
        return pd.DataFrame(columns=VALIDATION_TREND_COLUMNS)
    rows = [
        _extract_validation_trend_row(path)
        for path in sorted(root.glob("*/validation_retrain_comparison_summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    ]
    if not rows:
        return pd.DataFrame(columns=VALIDATION_TREND_COLUMNS)
    return pd.DataFrame(rows).reindex(columns=VALIDATION_TREND_COLUMNS)


def _safe_sort_text(value: Any) -> str:
    return str(value or "")


def _add_lineage_group_columns(lineage_df: pd.DataFrame) -> pd.DataFrame:
    if lineage_df.empty:
        return pd.DataFrame(columns=LINEAGE_COLUMNS)
    df = lineage_df.copy()
    df = df.sort_values(["started_at", "run_name"], key=lambda series: series.map(_safe_sort_text)).reset_index(drop=True)
    specs = [
        ("input_file_manifest_hash", "same_input_manifest_group_size", "previous_run_with_same_input_manifest"),
        ("feature_csv_sha256", "same_feature_hash_group_size", "previous_run_with_same_feature_hash"),
        ("parameter_hash", "same_parameter_hash_group_size", "previous_run_with_same_parameter_hash"),
    ]
    for hash_col, size_col, previous_col in specs:
        if hash_col not in df.columns:
            df[size_col] = 0
            df[previous_col] = ""
            continue
        non_empty = df[hash_col].fillna("").astype(str).str.len() > 0
        counts = df.loc[non_empty].groupby(hash_col)["run_name"].transform("count")
        df[size_col] = 0
        df.loc[non_empty, size_col] = counts.astype(int)
        previous_by_hash: dict[str, str] = {}
        previous_values: list[str] = []
        for _, row in df.iterrows():
            key = str(row.get(hash_col) or "")
            previous_values.append(previous_by_hash.get(key, "") if key else "")
            if key:
                previous_by_hash[key] = str(row.get("run_name") or "")
        df[previous_col] = previous_values
    return df.reindex(columns=LINEAGE_COLUMNS)


def _build_lineage_table(lineage_rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not lineage_rows:
        return pd.DataFrame(columns=LINEAGE_COLUMNS)
    return _add_lineage_group_columns(pd.DataFrame(lineage_rows))


def _shared_group_count(lineage_df: pd.DataFrame, column: str) -> int:
    if lineage_df.empty or column not in lineage_df.columns:
        return 0
    values = lineage_df[column].fillna("").astype(str)
    non_empty = values[values.str.len() > 0]
    if non_empty.empty:
        return 0
    return int(sum(1 for count in non_empty.value_counts().to_dict().values() if int(count) > 1))


def _short_hash(value: Any, *, length: int = 12) -> str:
    text = str(value or "").strip()
    return text[:length] if text else ""


def _jsonable_value(value: Any) -> Any:
    if pd.isna(value):
        return ""
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _build_lineage_graph_payload(lineage_df: pd.DataFrame, *, generated_at: str) -> dict[str, Any]:
    if lineage_df.empty:
        return {
            "schema_version": 1,
            "generated_at": generated_at,
            "node_count": 0,
            "edge_count": 0,
            "shared_group_count": 0,
            "nodes": [],
            "edges": [],
            "groups": [],
        }

    ordered = lineage_df.sort_values(["started_at", "run_name"], key=lambda series: series.map(_safe_sort_text))
    nodes: list[dict[str, Any]] = []
    for _, row in ordered.iterrows():
        run_name = str(row.get("run_name") or "")
        nodes.append(
            {
                "id": run_name,
                "label": run_name,
                "started_at": _jsonable_value(row.get("started_at")),
                "start_mode": _jsonable_value(row.get("start_mode")),
                "input_reference_count": _jsonable_value(row.get("input_reference_count")),
                "input_missing_count": _jsonable_value(row.get("input_missing_count")),
                "input_file_manifest_hash": str(row.get("input_file_manifest_hash") or ""),
                "feature_csv_sha256": str(row.get("feature_csv_sha256") or ""),
                "parameter_hash": str(row.get("parameter_hash") or ""),
                "summary_json": str(row.get("summary_json") or ""),
                "provenance_json": str(row.get("provenance_json") or ""),
            }
        )

    edges: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    for spec in LINEAGE_GRAPH_SPECS:
        hash_column = spec["hash_column"]
        if hash_column not in ordered.columns:
            continue
        valid = ordered[ordered[hash_column].fillna("").astype(str).str.len() > 0]
        for group_hash, group_df in valid.groupby(hash_column, sort=False):
            runs = [str(value) for value in group_df["run_name"].fillna("").tolist() if str(value)]
            if not runs:
                continue
            groups.append(
                {
                    "group_type": spec["edge_type"],
                    "label": spec["label"],
                    "hash": str(group_hash),
                    "short_hash": _short_hash(group_hash),
                    "run_count": int(len(runs)),
                    "runs": runs,
                    "first_run": runs[0],
                    "last_run": runs[-1],
                    "is_shared": bool(len(runs) > 1),
                }
            )
        previous_column = spec["previous_column"]
        if previous_column not in ordered.columns:
            continue
        for _, row in ordered.iterrows():
            source = str(row.get(previous_column) or "")
            target = str(row.get("run_name") or "")
            group_hash = str(row.get(hash_column) or "")
            if not source or not target or not group_hash:
                continue
            edges.append(
                {
                    "source": source,
                    "target": target,
                    "type": spec["edge_type"],
                    "label": spec["label"],
                    "hash": group_hash,
                    "short_hash": _short_hash(group_hash),
                }
            )

    shared_groups = [group for group in groups if group["is_shared"]]
    return {
        "schema_version": 1,
        "generated_at": generated_at,
        "node_count": int(len(nodes)),
        "edge_count": int(len(edges)),
        "group_count": int(len(groups)),
        "shared_group_count": int(len(shared_groups)),
        "nodes": nodes,
        "edges": edges,
        "groups": groups,
    }


def _build_lineage_graph_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Result Archive Lineage Graph",
        "",
        f"- Generated at: `{payload.get('generated_at', '')}`",
        f"- Runs: `{payload.get('node_count', 0)}`",
        f"- Lineage edges: `{payload.get('edge_count', 0)}`",
        f"- Shared groups: `{payload.get('shared_group_count', 0)}`",
        "",
        "## Shared Lineage Groups",
        "",
    ]
    shared_groups = [group for group in payload.get("groups", []) if isinstance(group, dict) and group.get("is_shared")]
    if not shared_groups:
        lines.append("_No shared lineage groups were detected._")
    else:
        for group in shared_groups:
            runs = [str(run) for run in group.get("runs", [])]
            lines.extend(
                [
                    f"### {group.get('label', '')} `{group.get('short_hash', '')}`",
                    "",
                    f"- Run count: `{group.get('run_count', 0)}`",
                    f"- Runs: `{' -> '.join(runs)}`",
                    "",
                ]
            )
    lines.extend(["", "## Edges", ""])
    edges = [edge for edge in payload.get("edges", []) if isinstance(edge, dict)]
    if not edges:
        lines.append("_No lineage edges were detected._")
    else:
        lines.append("| Type | Source | Target | Hash |")
        lines.append("|---|---|---|---|")
        for edge in edges:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(edge.get("label") or edge.get("type") or ""),
                        str(edge.get("source") or ""),
                        str(edge.get("target") or ""),
                        str(edge.get("short_hash") or ""),
                    ]
                )
                + " |"
            )
    lines.append("")
    return "\n".join(lines)


def _build_lineage_graph_html(payload: dict[str, Any]) -> str:
    shared_groups = [group for group in payload.get("groups", []) if isinstance(group, dict) and group.get("is_shared")]
    nodes = [node for node in payload.get("nodes", []) if isinstance(node, dict)]
    node_by_id = {str(node.get("id") or ""): node for node in nodes}

    group_cards: list[str] = []
    for group in shared_groups:
        edge_type = str(group.get("group_type") or "")
        color = next((str(spec["color"]) for spec in LINEAGE_GRAPH_SPECS if spec["edge_type"] == edge_type), "#64748b")
        run_chips: list[str] = []
        for idx, run_name in enumerate(group.get("runs", [])):
            run_text = str(run_name)
            node = node_by_id.get(run_text, {})
            chip = (
                '<span class="run-chip">'
                f'<strong>{html.escape(run_text)}</strong>'
                f'<small>{html.escape(str(node.get("started_at") or ""))}</small>'
                "</span>"
            )
            run_chips.append(chip)
            if idx < len(group.get("runs", [])) - 1:
                run_chips.append(f'<span class="arrow" style="color:{color}">-></span>')
        group_cards.append(
            "\n".join(
                [
                    f'<section class="lineage-group" style="border-left-color:{color}">',
                    f'<h2>{html.escape(str(group.get("label") or ""))}</h2>',
                    (
                        '<p class="hash">'
                        f'{html.escape(str(group.get("short_hash") or ""))} | '
                        f'{html.escape(str(group.get("run_count") or 0))} runs'
                        "</p>"
                    ),
                    '<div class="run-path">',
                    "".join(run_chips),
                    "</div>",
                    "</section>",
                ]
            )
        )
    if not group_cards:
        group_cards.append('<p class="empty">No shared lineage groups were detected.</p>')

    rows: list[str] = []
    for node in nodes[:200]:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(node.get('label') or ''))}</td>"
            f"<td>{html.escape(str(node.get('started_at') or ''))}</td>"
            f"<td>{html.escape(str(node.get('start_mode') or ''))}</td>"
            f"<td>{html.escape(_short_hash(node.get('input_file_manifest_hash')))}</td>"
            f"<td>{html.escape(_short_hash(node.get('feature_csv_sha256')))}</td>"
            f"<td>{html.escape(_short_hash(node.get('parameter_hash')))}</td>"
            "</tr>"
        )
    table_body = "\n".join(rows) if rows else '<tr><td colspan="6">No runs.</td></tr>'

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>ML Result Archive Lineage Graph</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f8f4ec;
      --ink: #172033;
      --muted: #697386;
      --card: #ffffff;
      --line: #e5decf;
    }}
    body {{
      margin: 0;
      padding: 32px;
      background: radial-gradient(circle at top left, #fff7d6, transparent 34%), var(--bg);
      color: var(--ink);
      font-family: "Segoe UI", "Noto Sans", sans-serif;
    }}
    header {{
      max-width: 1120px;
      margin: 0 auto 24px auto;
    }}
    h1 {{
      margin: 0 0 8px 0;
      font-size: 30px;
      letter-spacing: -0.04em;
    }}
    .summary {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 16px;
    }}
    .metric {{
      min-width: 140px;
      padding: 12px 14px;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.78);
      box-shadow: 0 12px 30px rgba(32, 25, 16, 0.06);
    }}
    .metric strong {{
      display: block;
      font-size: 24px;
    }}
    .metric span {{
      color: var(--muted);
      font-size: 12px;
    }}
    main {{
      max-width: 1120px;
      margin: 0 auto;
    }}
    .lineage-group {{
      margin: 16px 0;
      padding: 18px 20px;
      border: 1px solid var(--line);
      border-left: 8px solid #64748b;
      border-radius: 20px;
      background: var(--card);
      box-shadow: 0 18px 50px rgba(32, 25, 16, 0.08);
    }}
    .lineage-group h2 {{
      margin: 0;
      font-size: 18px;
    }}
    .hash {{
      margin: 5px 0 16px 0;
      color: var(--muted);
      font-family: Consolas, monospace;
      font-size: 12px;
    }}
    .run-path {{
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .run-chip {{
      display: inline-flex;
      flex-direction: column;
      gap: 3px;
      min-width: 150px;
      padding: 10px 12px;
      border: 1px solid #d8d1c5;
      border-radius: 14px;
      background: #fffaf1;
    }}
    .run-chip small {{
      color: var(--muted);
      font-size: 11px;
    }}
    .arrow {{
      font-weight: 800;
      font-size: 18px;
    }}
    table {{
      width: 100%;
      margin-top: 24px;
      border-collapse: collapse;
      background: var(--card);
      border-radius: 16px;
      overflow: hidden;
      box-shadow: 0 18px 50px rgba(32, 25, 16, 0.08);
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      font-size: 13px;
    }}
    th {{
      background: #ede4d3;
    }}
    .empty {{
      padding: 18px;
      border-radius: 16px;
      background: #fff;
      border: 1px solid var(--line);
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <header>
    <h1>ML Result Archive Lineage Graph</h1>
    <p>Visualizes repeated runs that share the same input manifest, feature CSV hash, or parameter hash.</p>
    <div class="summary">
      <div class="metric"><strong>{int(payload.get("node_count") or 0)}</strong><span>Runs</span></div>
      <div class="metric"><strong>{int(payload.get("edge_count") or 0)}</strong><span>Lineage edges</span></div>
      <div class="metric"><strong>{int(payload.get("shared_group_count") or 0)}</strong><span>Shared groups</span></div>
    </div>
  </header>
  <main>
    {''.join(group_cards)}
    <table>
      <thead>
        <tr>
          <th>Run</th>
          <th>Started</th>
          <th>Mode</th>
          <th>Input manifest</th>
          <th>Feature CSV</th>
          <th>Parameters</th>
        </tr>
      </thead>
      <tbody>{table_body}</tbody>
    </table>
  </main>
</body>
</html>
"""


def _markdown_table(df: pd.DataFrame, columns: list[str], *, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    work = df.loc[:, [column for column in columns if column in df.columns]].head(max_rows).copy()
    if work.empty:
        return "_No rows._"
    header = "| " + " | ".join(work.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(work.columns)) + " |"
    body = ["| " + " | ".join("" if pd.isna(row.get(col)) else str(row.get(col)) for col in work.columns) + " |" for _, row in work.iterrows()]
    return "\n".join([header, sep] + body)


def _build_archive_report(
    summary: dict[str, Any],
    run_df: pd.DataFrame,
    validation_trend_df: pd.DataFrame,
    lineage_df: pd.DataFrame,
) -> str:
    lines = [
        "# ML Result Archive Report",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Local app runs: `{summary['local_app_runs']}`",
        f"- Archived runs: `{summary['run_count']}`",
        f"- Artifact rows: `{summary['artifact_row_count']}`",
        f"- Validation retrain comparisons: `{summary['validation_retrain_comparison_count']}`",
        f"- Lineage rows: `{summary['lineage_row_count']}`",
        f"- Lineage graph edges: `{summary.get('lineage_graph_edge_count', 0)}`",
        f"- Lineage graph shared groups: `{summary.get('lineage_graph_shared_group_count', 0)}`",
        f"- Shared input manifest groups: `{summary['shared_input_manifest_group_count']}`",
        "",
        "## Recent Runs",
        "",
        _markdown_table(
            run_df,
            [
                "run_name",
                "status",
                "started_at",
                "label_col",
                "label_valid_count",
                "baseline_rank_spearman",
                "calibrated_rank_spearman",
                "best_val_loss",
            ],
            max_rows=20,
        ),
        "",
        "## Validation Retrain Long-Term Trend",
        "",
        _markdown_table(
            validation_trend_df,
            [
                "comparison_name",
                "before_run",
                "after_run",
                "top_k_overlap_fraction",
                "validated_top_k_count",
                "label_valid_delta",
                "calibrated_rank_spearman_delta",
                "best_val_loss_delta",
            ],
            max_rows=20,
        ),
        "",
        "## Run Lineage",
        "",
        _markdown_table(
            lineage_df,
            [
                "run_name",
                "start_mode",
                "same_input_manifest_group_size",
                "same_feature_hash_group_size",
                "same_parameter_hash_group_size",
                "previous_run_with_same_input_manifest",
                "previous_run_with_same_feature_hash",
            ],
            max_rows=20,
        ),
        "",
        "## Outputs",
        "",
    ]
    for key, value in summary.get("outputs", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)


def build_result_archive(
    *,
    local_app_runs: str | Path = "local_app_runs",
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    run_root = Path(local_app_runs).expanduser().resolve()
    archive_dir = Path(out_dir).expanduser().resolve() if out_dir is not None else run_root / "result_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    lineage_rows: list[dict[str, Any]] = []
    for run_dir in _iter_run_roots(run_root):
        run_row, run_artifacts = _extract_run_metrics(run_dir)
        if run_row is None:
            continue
        run_rows.append(run_row)
        artifact_rows.extend(run_artifacts)
        lineage_row = _extract_lineage_row(run_dir, run_row)
        if lineage_row is not None:
            lineage_rows.append(lineage_row)

    run_df = pd.DataFrame(run_rows).reindex(columns=RUN_INDEX_COLUMNS)
    artifact_df = pd.DataFrame(artifact_rows).reindex(columns=ARTIFACT_COLUMNS)
    validation_trend_df = _build_validation_trend_table(run_root)
    lineage_df = _build_lineage_table(lineage_rows)

    runs_csv = archive_dir / "result_archive_runs.csv"
    artifacts_csv = archive_dir / "result_archive_artifact_manifest.csv"
    validation_trend_csv = archive_dir / "result_archive_validation_retrain_trends.csv"
    lineage_csv = archive_dir / "result_archive_lineage.csv"
    lineage_graph_json = archive_dir / "result_archive_lineage_graph.json"
    lineage_graph_html = archive_dir / "result_archive_lineage_graph.html"
    lineage_graph_md = archive_dir / "result_archive_lineage_graph.md"
    summary_json = archive_dir / "result_archive_summary.json"
    report_md = archive_dir / "result_archive_report.md"

    run_df.to_csv(runs_csv, index=False)
    artifact_df.to_csv(artifacts_csv, index=False)
    validation_trend_df.to_csv(validation_trend_csv, index=False)
    lineage_df.to_csv(lineage_csv, index=False)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lineage_graph_payload = _build_lineage_graph_payload(lineage_df, generated_at=generated_at)
    _write_json(lineage_graph_json, lineage_graph_payload)
    lineage_graph_html.write_text(_build_lineage_graph_html(lineage_graph_payload), encoding="utf-8")
    lineage_graph_md.write_text(_build_lineage_graph_markdown(lineage_graph_payload), encoding="utf-8")

    status_counts = (
        {str(k): int(v) for k, v in run_df["status"].value_counts(dropna=False).to_dict().items()}
        if not run_df.empty and "status" in run_df.columns
        else {}
    )
    summary = {
        "generated_at": generated_at,
        "local_app_runs": str(run_root),
        "out_dir": str(archive_dir),
        "run_count": int(len(run_df)),
        "artifact_row_count": int(len(artifact_df)),
        "existing_artifact_count": int(artifact_df["exists"].fillna(False).sum()) if not artifact_df.empty else 0,
        "validation_retrain_comparison_count": int(len(validation_trend_df)),
        "lineage_row_count": int(len(lineage_df)),
        "lineage_graph_edge_count": int(lineage_graph_payload.get("edge_count") or 0),
        "lineage_graph_shared_group_count": int(lineage_graph_payload.get("shared_group_count") or 0),
        "shared_input_manifest_group_count": _shared_group_count(lineage_df, "input_file_manifest_hash"),
        "shared_feature_hash_group_count": _shared_group_count(lineage_df, "feature_csv_sha256"),
        "shared_parameter_hash_group_count": _shared_group_count(lineage_df, "parameter_hash"),
        "status_counts": status_counts,
        "outputs": {
            "runs_csv": str(runs_csv),
            "artifact_manifest_csv": str(artifacts_csv),
            "validation_retrain_trends_csv": str(validation_trend_csv),
            "lineage_csv": str(lineage_csv),
            "lineage_graph_json": str(lineage_graph_json),
            "lineage_graph_html": str(lineage_graph_html),
            "lineage_graph_md": str(lineage_graph_md),
            "summary_json": str(summary_json),
            "report_md": str(report_md),
        },
    }
    _write_json(summary_json, summary)
    report_md.write_text(_build_archive_report(summary, run_df, validation_trend_df, lineage_df), encoding="utf-8")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build local app result archive indexes and long-term trend tables")
    parser.add_argument("--local_app_runs", default="local_app_runs", help="Path to local_app_runs")
    parser.add_argument("--out_dir", default=None, help="Output archive directory; default is local_app_runs/result_archive")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    summary = build_result_archive(local_app_runs=args.local_app_runs, out_dir=args.out_dir)
    outputs = summary.get("outputs") if isinstance(summary.get("outputs"), dict) else {}
    for key in [
        "runs_csv",
        "artifact_manifest_csv",
        "validation_retrain_trends_csv",
        "lineage_csv",
        "lineage_graph_json",
        "lineage_graph_html",
        "lineage_graph_md",
        "summary_json",
        "report_md",
    ]:
        if outputs.get(key):
            print(f"Saved: {outputs[key]}")


if __name__ == "__main__":
    main()
