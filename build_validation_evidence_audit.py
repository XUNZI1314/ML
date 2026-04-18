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
PENDING_STATUS_VALUES = {"planned", "pending", "queued", "standby", "defer", "deferred", "include_now"}
IN_PROGRESS_STATUS_VALUES = {"running", "in_progress", "started", "ongoing", "current"}
COMPLETED_STATUS_VALUES = {"done", "complete", "completed", "finished", "tested"}


SOURCE_PRIORITIES = {
    "validation_labels_csv": 0,
    "experiment_plan_state_ledger_csv": 1,
    "experiment_plan_csv": 2,
}

SOURCE_LABEL_COLUMNS = [
    "validation_label",
    "input_validation_label",
    "experiment_label",
    "assay_label",
    "label",
]
SOURCE_RESULT_COLUMNS = [
    "experiment_result",
    "result",
    "validation_result",
    "assay_result",
]
SOURCE_STATUS_COLUMNS = [
    "experiment_status",
    "status",
    "sample_status",
    "plan_decision",
    "last_plan_decision",
]

CONSENSUS_CONTEXT_COLUMNS = [
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

EVIDENCE_COLUMNS = [
    "nanobody_id",
    "consensus_rank",
    "consensus_score",
    "confidence_level",
    "decision_tier",
    "validation_label",
    "validation_label_ready",
    "validation_label_source",
    "validation_category",
    "experiment_status",
    "experiment_result",
    "validation_gap",
    "recommended_action",
    "in_top_k",
]

ACTION_COLUMNS = [
    "priority",
    "action_type",
    "nanobody_id",
    "consensus_rank",
    "reason",
    "recommended_action",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether current consensus candidates have enough real validation evidence. "
            "This step does not change model scores or rankings."
        )
    )
    parser.add_argument("--summary_json", default=None, help="Optional recommended_pipeline_summary.json.")
    parser.add_argument("--consensus_csv", default=None, help="Optional consensus_ranking.csv override.")
    parser.add_argument("--experiment_plan_csv", default=None, help="Optional experiment_plan.csv override.")
    parser.add_argument(
        "--experiment_plan_state_ledger_csv",
        default=None,
        help="Optional experiment_plan_state_ledger.csv override.",
    )
    parser.add_argument(
        "--validation_labels_csv",
        default=None,
        help="Optional experiment_validation_labels.csv or manual validation labels CSV.",
    )
    parser.add_argument("--out_dir", default="validation_evidence_audit", help="Output directory.")
    parser.add_argument("--top_k", type=int, default=10, help="Top-k candidates to audit most strictly.")
    parser.add_argument(
        "--min_compare_labels",
        type=int,
        default=4,
        help="Minimum label-ready candidates for lightweight validation-vs-ranking comparison.",
    )
    parser.add_argument(
        "--min_calibration_labels",
        type=int,
        default=8,
        help="Minimum label-ready candidates for calibration/retraining readiness.",
    )
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
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def _norm(value: Any) -> str:
    return _clean_text(value).lower().replace("-", "_").replace(" ", "_")


def _first_nonempty(row: pd.Series, columns: list[str]) -> str:
    for col in columns:
        if col not in row.index:
            continue
        text = _clean_text(row.get(col))
        if text:
            return text
    return ""


def _parse_label(value: Any) -> tuple[float | None, str]:
    text = _clean_text(value)
    if not text:
        return None, ""
    try:
        numeric = float(text)
    except (TypeError, ValueError):
        numeric = None
    if numeric is not None:
        if numeric >= 0.5:
            return 1.0, "explicit_positive"
        if numeric == 0.0:
            return 0.0, "explicit_negative"

    normalized = _norm(text)
    if normalized in POSITIVE_VALUES:
        return 1.0, "explicit_positive"
    if normalized in NEGATIVE_VALUES:
        return 0.0, "explicit_negative"
    if normalized in INCONCLUSIVE_VALUES:
        return None, "explicit_inconclusive"
    return None, "unrecognized_explicit_label"


def _infer_evidence(*, label_value: Any, result_value: Any, status_value: Any) -> dict[str, Any]:
    explicit_label, explicit_reason = _parse_label(label_value)
    if explicit_label is not None:
        return {
            "validation_label": explicit_label,
            "validation_label_ready": True,
            "validation_category": "label_positive" if explicit_label >= 0.5 else "label_negative",
            "validation_reason": explicit_reason,
            "status_priority": 100,
        }

    result_label, result_reason = _parse_label(result_value)
    if result_label is not None:
        return {
            "validation_label": result_label,
            "validation_label_ready": True,
            "validation_category": "result_positive" if result_label >= 0.5 else "result_negative",
            "validation_reason": result_reason.replace("explicit", "result"),
            "status_priority": 95,
        }

    if explicit_reason == "explicit_inconclusive" or result_reason == "explicit_inconclusive":
        return {
            "validation_label": "",
            "validation_label_ready": False,
            "validation_category": "inconclusive",
            "validation_reason": "explicit inconclusive result is not a trainable label",
            "status_priority": 55,
        }

    status = _norm(status_value)
    if status in COMPLETED_STATUS_VALUES:
        return {
            "validation_label": "",
            "validation_label_ready": False,
            "validation_category": "completed_needs_result",
            "validation_reason": "completed status requires experiment_result or validation_label",
            "status_priority": 80,
        }
    if status in BLOCKED_STATUS_VALUES:
        return {
            "validation_label": "",
            "validation_label_ready": False,
            "validation_category": "blocked_or_cancelled",
            "validation_reason": "blocked/cancelled is not a biological negative label",
            "status_priority": 65,
        }
    if status in IN_PROGRESS_STATUS_VALUES:
        return {
            "validation_label": "",
            "validation_label_ready": False,
            "validation_category": "in_progress",
            "validation_reason": "experiment is still in progress",
            "status_priority": 50,
        }
    if status in PENDING_STATUS_VALUES or not status:
        return {
            "validation_label": "",
            "validation_label_ready": False,
            "validation_category": "pending",
            "validation_reason": "no validation result yet",
            "status_priority": 10,
        }
    return {
        "validation_label": "",
        "validation_label_ready": False,
        "validation_category": "unknown_status",
        "validation_reason": f"unrecognized status/result: {status or 'empty'}",
        "status_priority": 30,
    }


def _load_csv(path: str | Path | None, *, required: bool = False) -> pd.DataFrame | None:
    if path is None or not str(path).strip():
        if required:
            raise FileNotFoundError("Required CSV path is missing.")
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        if required:
            raise FileNotFoundError(f"CSV not found: {resolved}")
        return None
    try:
        return pd.read_csv(resolved, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(resolved, encoding="utf-8-sig", low_memory=False)


def _load_summary(path: str | Path | None) -> dict[str, Any]:
    if path is None or not str(path).strip():
        return {}
    summary_path = Path(path).expanduser().resolve()
    if not summary_path.exists():
        raise FileNotFoundError(f"summary_json not found: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _artifact_path(summary: dict[str, Any], key: str) -> str | None:
    artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
    value = artifacts.get(key)
    if value:
        return str(value)
    return None


def _resolve_inputs(args: argparse.Namespace) -> dict[str, str | None]:
    summary = _load_summary(args.summary_json)
    return {
        "summary_json": str(Path(args.summary_json).expanduser().resolve()) if args.summary_json else None,
        "consensus_csv": args.consensus_csv or _artifact_path(summary, "consensus_ranking_csv"),
        "experiment_plan_csv": args.experiment_plan_csv or _artifact_path(summary, "experiment_plan_csv"),
        "experiment_plan_state_ledger_csv": args.experiment_plan_state_ledger_csv
        or _artifact_path(summary, "experiment_plan_state_ledger_csv"),
        "validation_labels_csv": args.validation_labels_csv
        or _artifact_path(summary, "experiment_validation_labels_csv"),
    }


def _source_table(df: pd.DataFrame | None, *, source_name: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if "nanobody_id" not in df.columns:
        raise ValueError(f"{source_name} must contain nanobody_id")

    rows: list[dict[str, Any]] = []
    source_priority = int(SOURCE_PRIORITIES.get(source_name, 99))
    for row_index, row in df.iterrows():
        nanobody_id = _clean_text(row.get("nanobody_id"))
        if not nanobody_id:
            continue
        label_value = _first_nonempty(row, SOURCE_LABEL_COLUMNS)
        result_value = _first_nonempty(row, SOURCE_RESULT_COLUMNS)
        status_value = _first_nonempty(row, SOURCE_STATUS_COLUMNS)
        inferred = _infer_evidence(
            label_value=label_value,
            result_value=result_value,
            status_value=status_value,
        )
        rows.append(
            {
                "nanobody_id": nanobody_id,
                "validation_label": inferred["validation_label"],
                "validation_label_ready": bool(inferred["validation_label_ready"]),
                "validation_label_source": source_name if bool(inferred["validation_label_ready"]) else "",
                "validation_category": inferred["validation_category"],
                "validation_reason": inferred["validation_reason"],
                "experiment_status": status_value,
                "experiment_result": result_value,
                "source_name": source_name,
                "source_priority": source_priority,
                "source_row_index": int(row_index),
                "status_priority": int(inferred["status_priority"]),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _select_best_evidence(sources: list[pd.DataFrame]) -> pd.DataFrame:
    available = [df for df in sources if df is not None and not df.empty]
    if not available:
        return pd.DataFrame(
            columns=[
                "nanobody_id",
                "validation_label",
                "validation_label_ready",
                "validation_label_source",
                "validation_category",
                "validation_reason",
                "experiment_status",
                "experiment_result",
                "source_name",
            ]
        )
    merged = pd.concat(available, ignore_index=True)
    merged["_ready_sort"] = merged["validation_label_ready"].astype(bool).astype(int)
    merged = merged.sort_values(
        by=["nanobody_id", "_ready_sort", "source_priority", "status_priority", "source_row_index"],
        ascending=[True, False, True, False, False],
        na_position="last",
    )
    best = merged.drop_duplicates(subset=["nanobody_id"], keep="first").drop(columns=["_ready_sort"])
    return best.reset_index(drop=True)


def _rank_series(consensus_df: pd.DataFrame) -> pd.Series:
    if "consensus_rank" in consensus_df.columns:
        ranks = pd.to_numeric(consensus_df["consensus_rank"], errors="coerce")
        if ranks.notna().any():
            return ranks
    return pd.Series(range(1, len(consensus_df) + 1), index=consensus_df.index, dtype="float64")


def _recommended_action(row: pd.Series) -> str:
    ready = bool(row.get("validation_label_ready"))
    in_top_k = bool(row.get("in_top_k"))
    category = _clean_text(row.get("validation_category"))
    if ready:
        return "Use this candidate as validation evidence; include it in validation feedback or retraining only if assay definition matches the target task."
    if category == "completed_needs_result":
        return "Fill experiment_result or validation_label; completed alone is not converted into a biological label."
    if category == "blocked_or_cancelled":
        return "Do not treat blocked/cancelled as a negative label; replace with an alternative candidate or retest when material is available."
    if category == "in_progress":
        return "Wait for the result, then update experiment_result or validation_label."
    if category == "inconclusive":
        return "Keep out of training labels unless the assay is repeated or resolved."
    if in_top_k:
        return "Prioritize experimental validation for this top-k candidate before using ranking as evidence."
    return "Keep as backlog evidence gap; validate after higher-ranked candidates or when diversity is needed."


def _validation_gap(row: pd.Series) -> str:
    if bool(row.get("validation_label_ready")):
        return "validated"
    category = _clean_text(row.get("validation_category"))
    if category:
        return category
    if bool(row.get("in_top_k")):
        return "top_k_unvalidated"
    return "not_recorded"


def _build_candidate_table(consensus_df: pd.DataFrame, evidence_df: pd.DataFrame, *, top_k: int) -> pd.DataFrame:
    if "nanobody_id" not in consensus_df.columns:
        raise ValueError("consensus_csv must contain nanobody_id")

    context_cols = [col for col in CONSENSUS_CONTEXT_COLUMNS if col in consensus_df.columns]
    work = consensus_df.loc[:, context_cols].copy()
    work["nanobody_id"] = work["nanobody_id"].map(_clean_text)
    work = work.loc[work["nanobody_id"].ne("")].copy()
    work["_rank"] = _rank_series(work)
    work = work.sort_values(by="_rank", ascending=True, na_position="last").reset_index(drop=True)
    work["in_top_k"] = work.index < int(max(top_k, 0))

    if evidence_df.empty:
        for col in [
            "validation_label",
            "validation_label_ready",
            "validation_label_source",
            "validation_category",
            "validation_reason",
            "experiment_status",
            "experiment_result",
            "source_name",
        ]:
            work[col] = False if col == "validation_label_ready" else ""
    else:
        keep_cols = [
            col
            for col in [
                "nanobody_id",
                "validation_label",
                "validation_label_ready",
                "validation_label_source",
                "validation_category",
                "validation_reason",
                "experiment_status",
                "experiment_result",
                "source_name",
            ]
            if col in evidence_df.columns
        ]
        work = work.merge(evidence_df.loc[:, keep_cols], on="nanobody_id", how="left")
        work["validation_label_ready"] = work["validation_label_ready"].fillna(False).astype(bool)
        for col in [
            "validation_label",
            "validation_label_source",
            "validation_category",
            "validation_reason",
            "experiment_status",
            "experiment_result",
            "source_name",
        ]:
            if col not in work.columns:
                work[col] = ""
            work[col] = work[col].map(_clean_text)

    work["validation_category"] = work["validation_category"].where(
        work["validation_category"].map(_clean_text).ne(""),
        "not_recorded",
    )
    work["validation_gap"] = work.apply(_validation_gap, axis=1)
    work["recommended_action"] = work.apply(_recommended_action, axis=1)
    return work.drop(columns=["_rank"], errors="ignore")


def _class_counts(candidate_df: pd.DataFrame) -> tuple[int, int]:
    labels = pd.to_numeric(
        candidate_df.loc[candidate_df["validation_label_ready"].astype(bool), "validation_label"],
        errors="coerce",
    )
    positive_count = int((labels >= 0.5).sum())
    negative_count = int((labels < 0.5).sum())
    return positive_count, negative_count


def _action_items(candidate_df: pd.DataFrame, summary: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    top_df = candidate_df.loc[candidate_df["in_top_k"].astype(bool)].copy()
    gaps = top_df.loc[~top_df["validation_label_ready"].astype(bool)].copy()
    if "consensus_rank" in gaps.columns:
        gaps["_rank_sort"] = pd.to_numeric(gaps["consensus_rank"], errors="coerce")
        gaps = gaps.sort_values("_rank_sort", ascending=True, na_position="last")
    for _, row in gaps.head(20).iterrows():
        category = _clean_text(row.get("validation_category"))
        action_type = "fill_completed_result" if category == "completed_needs_result" else "validate_top_candidate"
        rows.append(
            {
                "priority": len(rows) + 1,
                "action_type": action_type,
                "nanobody_id": _clean_text(row.get("nanobody_id")),
                "consensus_rank": _clean_text(row.get("consensus_rank")),
                "reason": _clean_text(row.get("validation_gap")) or "top-k lacks validation label",
                "recommended_action": _clean_text(row.get("recommended_action")),
            }
        )

    if int(summary.get("label_ready_count") or 0) > 0 and int(summary.get("label_class_count") or 0) < 2:
        rows.append(
            {
                "priority": len(rows) + 1,
                "action_type": "add_counterexamples",
                "nanobody_id": "",
                "consensus_rank": "",
                "reason": "validated labels contain only one class",
                "recommended_action": "Add likely negative and positive counterexamples before comparing or calibrating ranking quality.",
            }
        )
    if int(summary.get("completed_needs_result_count") or 0) > 0:
        rows.append(
            {
                "priority": len(rows) + 1,
                "action_type": "fill_missing_results",
                "nanobody_id": "",
                "consensus_rank": "",
                "reason": "some completed experiments lack interpretable result labels",
                "recommended_action": "Update experiment_result as positive/negative/inconclusive or set validation_label explicitly.",
            }
        )
    if not rows:
        rows.append(
            {
                "priority": 1,
                "action_type": "monitor",
                "nanobody_id": "",
                "consensus_rank": "",
                "reason": "no immediate validation evidence blocker",
                "recommended_action": "Keep updating the experiment ledger after each assay round.",
            }
        )
    return pd.DataFrame(rows, columns=ACTION_COLUMNS)


def _summary(
    candidate_df: pd.DataFrame,
    inputs: dict[str, str | None],
    *,
    top_k: int,
    min_compare_labels: int,
    min_calibration_labels: int,
) -> dict[str, Any]:
    candidate_count = int(len(candidate_df))
    label_ready_mask = candidate_df["validation_label_ready"].astype(bool) if not candidate_df.empty else pd.Series(dtype=bool)
    label_ready_count = int(label_ready_mask.sum()) if not candidate_df.empty else 0
    positive_count, negative_count = _class_counts(candidate_df) if not candidate_df.empty else (0, 0)
    label_class_count = int((1 if positive_count > 0 else 0) + (1 if negative_count > 0 else 0))
    top_df = candidate_df.loc[candidate_df["in_top_k"].astype(bool)].copy() if not candidate_df.empty else pd.DataFrame()
    top_k_count = int(len(top_df))
    top_k_validated_count = int(top_df["validation_label_ready"].astype(bool).sum()) if not top_df.empty else 0
    top_positive, top_negative = _class_counts(top_df) if not top_df.empty else (0, 0)
    top_k_unvalidated_count = int(top_k_count - top_k_validated_count)
    top_k_validation_coverage = float(top_k_validated_count / max(top_k_count, 1))
    gap_counts = (
        {str(k): int(v) for k, v in candidate_df["validation_gap"].value_counts(dropna=False).to_dict().items()}
        if not candidate_df.empty and "validation_gap" in candidate_df.columns
        else {}
    )
    category_counts = (
        {str(k): int(v) for k, v in candidate_df["validation_category"].value_counts(dropna=False).to_dict().items()}
        if not candidate_df.empty and "validation_category" in candidate_df.columns
        else {}
    )
    completed_needs_result_count = int(category_counts.get("completed_needs_result", 0))
    compare_ready = bool(label_ready_count >= int(min_compare_labels) and label_class_count >= 2)
    calibration_ready = bool(label_ready_count >= int(min_calibration_labels) and label_class_count >= 2)

    if candidate_count <= 0:
        audit_status = "FAIL"
    elif label_ready_count <= 0:
        audit_status = "WARN"
    elif label_class_count < 2:
        audit_status = "WARN"
    elif top_k_validated_count <= 0:
        audit_status = "WARN"
    elif label_ready_count < int(min_compare_labels) or top_k_validation_coverage < 0.30:
        audit_status = "WARN"
    else:
        audit_status = "PASS"

    next_actions: list[str] = []
    if candidate_count <= 0:
        next_actions.append("No consensus candidates were found; rerun consensus ranking before validation audit.")
    if label_ready_count <= 0:
        next_actions.append("Add explicit validation_label or experiment_result values to the experiment ledger; planned/completed alone is not enough.")
    if top_k_unvalidated_count > 0:
        next_actions.append(f"Validate or status-update {top_k_unvalidated_count} unvalidated candidates in the audited top-{top_k_count}.")
    if label_ready_count > 0 and label_class_count < 2:
        next_actions.append("Add both positive and negative examples before using validation feedback for model calibration.")
    if completed_needs_result_count > 0:
        next_actions.append("Fill experiment_result for completed experiments; completed status alone is intentionally not treated as a label.")
    if not next_actions:
        next_actions.append("Validation evidence is minimally usable; keep expanding balanced labels after each experiment round.")

    return {
        "built_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "audit_status": audit_status,
        "inputs": inputs,
        "top_k": int(top_k),
        "candidate_count": candidate_count,
        "label_ready_count": label_ready_count,
        "positive_label_count": positive_count,
        "negative_label_count": negative_count,
        "label_class_count": label_class_count,
        "top_k_count": top_k_count,
        "top_k_validated_count": top_k_validated_count,
        "top_k_positive_label_count": top_positive,
        "top_k_negative_label_count": top_negative,
        "top_k_unvalidated_count": top_k_unvalidated_count,
        "top_k_validation_coverage": top_k_validation_coverage,
        "completed_needs_result_count": completed_needs_result_count,
        "compare_ready": compare_ready,
        "calibration_ready": calibration_ready,
        "thresholds": {
            "min_compare_labels": int(min_compare_labels),
            "min_calibration_labels": int(min_calibration_labels),
            "top_k_min_coverage_for_pass": 0.30,
        },
        "gap_counts": gap_counts,
        "category_counts": category_counts,
        "next_actions": next_actions,
        "formula": {
            "label_source_priority": [
                "validation_labels_csv.validation_label",
                "experiment_plan_state_ledger_csv.validation_label",
                "experiment_plan_state_ledger_csv.experiment_result",
                "experiment_plan_csv.validation_label",
                "experiment_plan_csv.experiment_result",
            ],
            "status_safety_rule": "experiment_status alone never creates a biological positive or negative label",
            "compare_ready": "label_ready_count >= min_compare_labels and both positive and negative labels exist",
            "calibration_ready": "label_ready_count >= min_calibration_labels and both positive and negative labels exist",
        },
    }


def _md(value: Any) -> str:
    return _clean_text(value).replace("|", "\\|")


def _fmt_float(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return ""


def _report(summary: dict[str, Any], top_df: pd.DataFrame, action_df: pd.DataFrame) -> str:
    lines = [
        "# Validation Evidence Audit",
        "",
        "This audit checks whether current ranked candidates have enough real validation evidence. It does not change Rule, ML, or consensus scores.",
        "",
        "## Summary",
        "",
        f"- Audit status: `{summary['audit_status']}`",
        f"- Candidate count: `{summary['candidate_count']}`",
        f"- Label-ready candidates: `{summary['label_ready_count']}`",
        f"- Positive labels: `{summary['positive_label_count']}`",
        f"- Negative labels: `{summary['negative_label_count']}`",
        f"- Audited top-k: `{summary['top_k_count']}`",
        f"- Top-k validated: `{summary['top_k_validated_count']}`",
        f"- Top-k validation coverage: `{summary['top_k_validation_coverage']:.1%}`",
        f"- Compare ready: `{summary['compare_ready']}`",
        f"- Calibration ready: `{summary['calibration_ready']}`",
        "",
        "## Safety Rule",
        "",
        "- `completed` alone is not converted into a positive or negative label.",
        "- Only explicit `validation_label` or interpretable `experiment_result` values become validation evidence.",
        "- `blocked` / `cancelled` are treated as operational exclusions, not biological negative labels.",
        "",
        "## Next Actions",
        "",
    ]
    for action in summary.get("next_actions", []):
        lines.append(f"- {action}")

    lines.extend(["", "## Top-k Evidence", ""])
    if top_df.empty:
        lines.append("- No top-k candidates available.")
    else:
        cols = [
            col
            for col in [
                "nanobody_id",
                "consensus_rank",
                "consensus_score",
                "confidence_level",
                "validation_label_ready",
                "validation_label",
                "validation_gap",
                "recommended_action",
            ]
            if col in top_df.columns
        ]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, row in top_df.loc[:, cols].head(20).iterrows():
            values: list[str] = []
            for col in cols:
                value = row.get(col, "")
                if col == "consensus_score":
                    value = _fmt_float(value)
                values.append(_md(value))
            lines.append("| " + " | ".join(values) + " |")

    lines.extend(["", "## Action Items", ""])
    if action_df.empty:
        lines.append("- No action items.")
    else:
        cols = [col for col in ACTION_COLUMNS if col in action_df.columns]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, row in action_df.loc[:, cols].head(30).iterrows():
            lines.append("| " + " | ".join(_md(row.get(col, "")) for col in cols) + " |")

    lines.extend(
        [
            "",
            "## Input Files",
            "",
        ]
    )
    for key, value in summary.get("inputs", {}).items():
        lines.append(f"- {key}: `{value or ''}`")
    lines.append("")
    return "\n".join(lines)


def build_validation_evidence_audit(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = _resolve_inputs(args)
    consensus_df = _load_csv(inputs.get("consensus_csv"), required=True)
    assert consensus_df is not None
    validation_labels_df = _load_csv(inputs.get("validation_labels_csv"), required=False)
    ledger_df = _load_csv(inputs.get("experiment_plan_state_ledger_csv"), required=False)
    plan_df = _load_csv(inputs.get("experiment_plan_csv"), required=False)

    evidence_df = _select_best_evidence(
        [
            _source_table(validation_labels_df, source_name="validation_labels_csv"),
            _source_table(ledger_df, source_name="experiment_plan_state_ledger_csv"),
            _source_table(plan_df, source_name="experiment_plan_csv"),
        ]
    )
    candidate_df = _build_candidate_table(consensus_df, evidence_df, top_k=int(args.top_k))
    top_df = candidate_df.loc[candidate_df["in_top_k"].astype(bool)].copy()
    summary = _summary(
        candidate_df,
        inputs,
        top_k=int(args.top_k),
        min_compare_labels=int(args.min_compare_labels),
        min_calibration_labels=int(args.min_calibration_labels),
    )
    action_df = _action_items(candidate_df, summary)

    by_candidate_csv = out_dir / "validation_evidence_by_candidate.csv"
    topk_csv = out_dir / "validation_evidence_topk.csv"
    action_items_csv = out_dir / "validation_evidence_action_items.csv"
    summary_json = out_dir / "validation_evidence_summary.json"
    report_md = out_dir / "validation_evidence_report.md"

    ordered_cols = [col for col in EVIDENCE_COLUMNS if col in candidate_df.columns] + [
        col for col in candidate_df.columns if col not in EVIDENCE_COLUMNS
    ]
    candidate_df.loc[:, ordered_cols].to_csv(by_candidate_csv, index=False)
    top_df.loc[:, ordered_cols].to_csv(topk_csv, index=False)
    action_df.to_csv(action_items_csv, index=False)
    summary["outputs"] = {
        "validation_evidence_summary_json": str(summary_json),
        "validation_evidence_report_md": str(report_md),
        "validation_evidence_by_candidate_csv": str(by_candidate_csv),
        "validation_evidence_topk_csv": str(topk_csv),
        "validation_evidence_action_items_csv": str(action_items_csv),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    report_md.write_text(_report(summary, top_df, action_df), encoding="utf-8")
    return summary


def main() -> None:
    args = _build_parser().parse_args()
    summary = build_validation_evidence_audit(args)
    outputs = summary.get("outputs", {}) if isinstance(summary.get("outputs"), dict) else {}
    for path in outputs.values():
        if path:
            print(f"Saved: {path}")
    print(f"Audit status: {summary.get('audit_status')}")
    print(
        "Label-ready candidates: "
        f"{summary.get('label_ready_count')}/{summary.get('candidate_count')} "
        f"(top-k coverage {float(summary.get('top_k_validation_coverage') or 0):.1%})"
    )


if __name__ == "__main__":
    main()
