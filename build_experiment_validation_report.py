"""Convert experiment ledger results into validation labels and audit reports."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


POSITIVE_VALUES = {
    "1",
    "true",
    "yes",
    "positive",
    "hit",
    "active",
    "effective",
    "blocker",
    "blocking",
    "validated_positive",
    "confirmed_positive",
    "confirmed_blocker",
}
NEGATIVE_VALUES = {
    "0",
    "false",
    "no",
    "negative",
    "no_hit",
    "inactive",
    "ineffective",
    "non_blocker",
    "not_blocking",
    "validated_negative",
    "confirmed_negative",
    "fail_to_block",
}
INCONCLUSIVE_VALUES = {"inconclusive", "ambiguous", "unclear", "partial", "unknown", "na", "n/a"}
BLOCKED_STATUS_VALUES = {"blocked", "cancelled", "canceled", "material_unavailable", "not_available"}
PENDING_STATUS_VALUES = {"planned", "pending", "queued", "standby", "defer", "deferred"}
IN_PROGRESS_STATUS_VALUES = {"running", "in_progress", "started", "ongoing", "current"}
COMPLETED_STATUS_VALUES = {"done", "complete", "completed", "finished", "tested"}


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def _norm(value: Any) -> str:
    return _clean_text(value).lower().replace("-", "_").replace(" ", "_")


def _parse_label(value: Any) -> tuple[float | None, str]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = None
    if numeric is not None:
        if numeric >= 0.5:
            return 1.0, "explicit_positive"
        if numeric == 0.0:
            return 0.0, "explicit_negative"

    text = _norm(value)
    if not text:
        return None, ""
    if text in POSITIVE_VALUES:
        return 1.0, "explicit_positive"
    if text in NEGATIVE_VALUES:
        return 0.0, "explicit_negative"
    if text in INCONCLUSIVE_VALUES:
        return None, "explicit_inconclusive"
    return None, "unrecognized_explicit_label"


def _infer_validation(row: pd.Series) -> dict[str, Any]:
    explicit_label, explicit_reason = _parse_label(row.get("validation_label") or row.get("input_validation_label"))
    if explicit_label is not None:
        return {
            "validation_label": explicit_label,
            "validation_label_ready": True,
            "validation_category": "label_positive" if explicit_label >= 0.5 else "label_negative",
            "recommended_training_use": "train_label",
            "validation_label_reason": explicit_reason,
        }

    result_label, result_reason = _parse_label(row.get("experiment_result"))
    if result_label is not None:
        return {
            "validation_label": result_label,
            "validation_label_ready": True,
            "validation_category": "label_positive" if result_label >= 0.5 else "label_negative",
            "recommended_training_use": "train_label",
            "validation_label_reason": result_reason.replace("explicit", "result"),
        }

    if result_reason == "explicit_inconclusive":
        return {
            "validation_label": None,
            "validation_label_ready": False,
            "validation_category": "inconclusive",
            "recommended_training_use": "exclude",
            "validation_label_reason": "experiment_result is inconclusive",
        }

    status = _norm(row.get("experiment_status"))
    if status in COMPLETED_STATUS_VALUES:
        return {
            "validation_label": None,
            "validation_label_ready": False,
            "validation_category": "completed_needs_result",
            "recommended_training_use": "needs_result",
            "validation_label_reason": "completed status requires experiment_result or validation_label",
        }
    if status in BLOCKED_STATUS_VALUES:
        return {
            "validation_label": None,
            "validation_label_ready": False,
            "validation_category": "blocked_or_cancelled",
            "recommended_training_use": "exclude",
            "validation_label_reason": "blocked/cancelled is not a biological negative label",
        }
    if status in IN_PROGRESS_STATUS_VALUES:
        return {
            "validation_label": None,
            "validation_label_ready": False,
            "validation_category": "in_progress",
            "recommended_training_use": "wait",
            "validation_label_reason": "experiment is still in progress",
        }
    if status in PENDING_STATUS_VALUES or not status:
        return {
            "validation_label": None,
            "validation_label_ready": False,
            "validation_category": "pending",
            "recommended_training_use": "wait",
            "validation_label_reason": "no validation result yet",
        }
    return {
        "validation_label": None,
        "validation_label_ready": False,
        "validation_category": "unknown_status",
        "recommended_training_use": "review",
        "validation_label_reason": f"unrecognized status/result: {status or 'empty'}",
    }


def _load_csv(path: str | Path | None) -> pd.DataFrame | None:
    if path is None or not str(path).strip():
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"CSV not found: {resolved}")
    return pd.read_csv(resolved, low_memory=False)


def build_validation_status_table(ledger_df: pd.DataFrame) -> pd.DataFrame:
    if ledger_df.empty:
        return pd.DataFrame()
    if "nanobody_id" not in ledger_df.columns:
        raise ValueError("ledger_csv must contain nanobody_id")

    work = ledger_df.copy()
    for col in [
        "experiment_status",
        "experiment_result",
        "validation_label",
        "experiment_owner",
        "experiment_cost",
        "experiment_note",
        "latest_source_run_name",
        "latest_source_kind",
    ]:
        if col not in work.columns:
            work[col] = ""

    work["input_validation_label"] = work["validation_label"].map(_clean_text)
    work = work.drop(columns=["validation_label"])
    inferred_rows = [_infer_validation(row) for _, row in work.iterrows()]
    inferred_df = pd.DataFrame(inferred_rows)
    out = pd.concat([work.reset_index(drop=True), inferred_df], axis=1)
    first_cols = [
        "nanobody_id",
        "validation_label",
        "validation_label_ready",
        "validation_category",
        "recommended_training_use",
        "validation_label_reason",
        "input_validation_label",
        "experiment_status",
        "experiment_result",
        "experiment_owner",
        "experiment_cost",
        "experiment_note",
    ]
    ordered_cols = [col for col in first_cols if col in out.columns] + [col for col in out.columns if col not in first_cols]
    return out[ordered_cols]


def _merge_consensus(status_df: pd.DataFrame, consensus_df: pd.DataFrame | None) -> pd.DataFrame:
    if consensus_df is None or consensus_df.empty or "nanobody_id" not in consensus_df.columns:
        return status_df.copy()
    keep_cols = [
        col
        for col in [
            "nanobody_id",
            "consensus_rank",
            "consensus_score",
            "confidence_score",
            "confidence_level",
            "decision_tier",
            "qc_risk_score",
            "review_reason_flags",
            "low_confidence_reasons",
        ]
        if col in consensus_df.columns
    ]
    return status_df.merge(consensus_df[keep_cols], on="nanobody_id", how="left")


def _annotate_features(feature_df: pd.DataFrame, labels_df: pd.DataFrame, *, label_col: str) -> pd.DataFrame:
    if "nanobody_id" not in feature_df.columns:
        raise ValueError("feature_csv must contain nanobody_id")
    label_map = labels_df[["nanobody_id", "validation_label"]].copy()
    label_map = label_map.dropna(subset=["validation_label"])
    label_map = label_map.drop_duplicates(subset=["nanobody_id"], keep="last")
    label_map = label_map.rename(columns={"validation_label": label_col})
    return feature_df.merge(label_map, on="nanobody_id", how="left")


def _summary_payload(status_df: pd.DataFrame, labels_df: pd.DataFrame, *, ledger_csv: str | Path) -> dict[str, Any]:
    category_counts = (
        {str(k): int(v) for k, v in status_df["validation_category"].value_counts(dropna=False).to_dict().items()}
        if "validation_category" in status_df.columns
        else {}
    )
    label_counts = (
        {str(k): int(v) for k, v in labels_df["validation_label"].value_counts(dropna=False).to_dict().items()}
        if not labels_df.empty
        else {}
    )
    return {
        "ledger_csv": str(Path(ledger_csv).expanduser().resolve()),
        "built_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "candidate_count": int(len(status_df)),
        "label_ready_count": int(len(labels_df)),
        "positive_label_count": int((labels_df["validation_label"].astype(float) >= 0.5).sum()) if not labels_df.empty else 0,
        "negative_label_count": int((labels_df["validation_label"].astype(float) < 0.5).sum()) if not labels_df.empty else 0,
        "category_counts": category_counts,
        "label_counts": label_counts,
        "formula": {
            "label_source_priority": "validation_label column first, then experiment_result; experiment_status alone never creates a biological label",
            "positive_values": sorted(POSITIVE_VALUES),
            "negative_values": sorted(NEGATIVE_VALUES),
        },
    }


def _report_text(summary: dict[str, Any], status_df: pd.DataFrame, labels_df: pd.DataFrame) -> str:
    lines = [
        "# Experiment Validation Feedback Report",
        "",
        f"- Candidates in ledger: `{summary['candidate_count']}`",
        f"- Label-ready candidates: `{summary['label_ready_count']}`",
        f"- Positive labels: `{summary['positive_label_count']}`",
        f"- Negative labels: `{summary['negative_label_count']}`",
        f"- Category counts: `{json.dumps(summary.get('category_counts', {}), ensure_ascii=False)}`",
        "",
        "## Label Safety Rule",
        "",
        "- `completed` alone is not converted into a positive or negative label.",
        "- Only explicit `validation_label` or interpretable `experiment_result` values become training labels.",
        "- `blocked` / `cancelled` are treated as non-biological exclusions, not negative labels.",
        "",
        "## Label-Ready Candidates",
        "",
    ]
    if labels_df.empty:
        lines.append("_No label-ready candidates yet._")
    else:
        cols = [
            "nanobody_id",
            "validation_label",
            "experiment_result",
            "experiment_status",
            "consensus_rank",
            "consensus_score",
            "validation_label_reason",
        ]
        cols = [col for col in cols if col in labels_df.columns]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, row in labels_df.head(50).iterrows():
            values = [str(row.get(col, "")).replace("\n", " ").replace("|", "/") for col in cols]
            lines.append("| " + " | ".join(values) + " |")

    needs_result = status_df[status_df["recommended_training_use"].astype(str).eq("needs_result")]
    lines.extend(["", "## Needs Result", ""])
    if needs_result.empty:
        lines.append("_No completed-without-result candidates._")
    else:
        lines.append(", ".join(needs_result["nanobody_id"].astype(str).head(50).tolist()))
    return "\n".join(lines) + "\n"


def build_experiment_validation_feedback(
    *,
    ledger_csv: str | Path,
    out_dir: str | Path,
    consensus_csv: str | Path | None = None,
    feature_csv: str | Path | None = None,
    label_col: str = "experiment_label",
) -> dict[str, Any]:
    ledger_path = Path(ledger_csv).expanduser().resolve()
    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    ledger_df = _load_csv(ledger_path)
    if ledger_df is None:
        raise FileNotFoundError(f"ledger_csv not found: {ledger_path}")
    consensus_df = _load_csv(consensus_csv)
    feature_df = _load_csv(feature_csv)

    status_df = build_validation_status_table(ledger_df)
    status_with_consensus = _merge_consensus(status_df, consensus_df)
    labels_df = status_with_consensus[status_with_consensus["validation_label_ready"].astype(bool)].copy()

    status_csv = out_path / "experiment_validation_status_report.csv"
    labels_csv = out_path / "experiment_validation_labels.csv"
    summary_json = out_path / "experiment_validation_summary.json"
    report_md = out_path / "experiment_validation_report.md"
    status_with_consensus.to_csv(status_csv, index=False)
    labels_df.to_csv(labels_csv, index=False)

    feature_out = None
    if feature_df is not None:
        annotated = _annotate_features(feature_df, labels_df, label_col=str(label_col))
        feature_out = out_path / "pose_features_with_experiment_labels.csv"
        annotated.to_csv(feature_out, index=False)

    summary = _summary_payload(status_with_consensus, labels_df, ledger_csv=ledger_path)
    summary["outputs"] = {
        "experiment_validation_status_report_csv": str(status_csv),
        "experiment_validation_labels_csv": str(labels_csv),
        "experiment_validation_summary_json": str(summary_json),
        "experiment_validation_report_md": str(report_md),
        "pose_features_with_experiment_labels_csv": str(feature_out) if feature_out is not None else None,
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    report_md.write_text(_report_text(summary, status_with_consensus, labels_df), encoding="utf-8")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build validation labels and reports from experiment ledger")
    parser.add_argument("--ledger_csv", required=True, help="experiment_state_ledger_global.csv or experiment_plan_state_ledger.csv")
    parser.add_argument("--out_dir", default="experiment_validation_feedback", help="Output directory")
    parser.add_argument("--consensus_csv", default=None, help="Optional consensus_ranking.csv for prediction-vs-validation context")
    parser.add_argument("--feature_csv", default=None, help="Optional pose_features.csv to annotate with experiment labels")
    parser.add_argument("--label_col", default="experiment_label", help="Label column name in annotated feature CSV")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    summary = build_experiment_validation_feedback(
        ledger_csv=args.ledger_csv,
        out_dir=args.out_dir,
        consensus_csv=args.consensus_csv,
        feature_csv=args.feature_csv,
        label_col=str(args.label_col),
    )
    for path in summary.get("outputs", {}).values():
        if path:
            print(f"Saved: {path}")


if __name__ == "__main__":
    main()
