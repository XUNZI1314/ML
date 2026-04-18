"""Suggest next experiment priorities from consensus ranking outputs.

This is an active-learning style decision layer. It does not retrain the model;
it turns high-potential or high-uncertainty candidates into a review queue.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Suggest next nanobody validation experiments")
    parser.add_argument("--consensus_csv", required=True, help="Path to consensus_ranking.csv")
    parser.add_argument("--out_dir", default="experiment_suggestions", help="Output directory")
    parser.add_argument("--top_n", type=int, default=20, help="Number of candidates to include in the suggestion table")
    parser.add_argument("--high_score_threshold", type=float, default=0.70)
    parser.add_argument("--low_confidence_threshold", type=float, default=0.55)
    parser.add_argument("--qc_risk_threshold", type=float, default=0.40)
    parser.add_argument("--rank_delta_threshold", type=float, default=3.0)
    parser.add_argument(
        "--diversity_mode",
        choices=["auto", "profile", "column", "none"],
        default="auto",
        help="Apply lightweight diversity-aware ordering to the suggestion queue",
    )
    parser.add_argument(
        "--diversity_group_col",
        default="auto",
        help="Column used as diversity group; auto tries common cluster/family columns",
    )
    parser.add_argument(
        "--diversity_penalty",
        type=float,
        default=0.08,
        help="Priority penalty per already-selected candidate from the same diversity group",
    )
    parser.add_argument(
        "--diversity_max_per_group",
        type=int,
        default=2,
        help="Soft target before a diversity group receives an extra repeat warning; 0 disables this target",
    )
    parser.add_argument("--experiment_plan_budget", type=int, default=8, help="Max include_now candidates in the plan")
    parser.add_argument("--experiment_plan_validate_budget", type=int, default=4, help="Max validate_now candidates")
    parser.add_argument("--experiment_plan_review_budget", type=int, default=3, help="Max review_first candidates")
    parser.add_argument("--experiment_plan_standby_budget", type=int, default=3, help="Max standby candidates")
    parser.add_argument("--experiment_plan_group_quota", type=int, default=2, help="Max include_now candidates per diversity group; 0 disables quota")
    parser.add_argument(
        "--experiment_plan_override_csv",
        default=None,
        help=(
            "Optional CSV keyed by nanobody_id with plan_override/manual_plan_action, "
            "experiment_status, experiment_owner, experiment_cost and experiment_note columns"
        ),
    )
    return parser


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _clean_group_value(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "null", "na", "n/a"}:
        return ""
    return text


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null", "na", "n/a"}:
        return ""
    return text


def _first_nonempty(row: pd.Series, columns: list[str]) -> str:
    for col in columns:
        if col in row.index:
            text = _clean_text(row.get(col))
            if text:
                return text
    return ""


def _normalize_manual_plan_action(value: Any) -> str:
    text = _clean_text(value).lower().replace("-", "_").replace(" ", "_")
    if not text:
        return ""
    include_values = {"include", "include_now", "force_include", "lock", "locked", "must_include", "current_round"}
    exclude_values = {"exclude", "force_exclude", "skip", "remove", "reject", "blocked", "do_not_test"}
    standby_values = {"standby", "backup", "reserve", "waitlist", "wait_list"}
    defer_values = {"defer", "later", "later_round", "postpone", "hold"}
    completed_values = {"done", "complete", "completed", "success", "succeeded", "finished", "tested"}
    if text in include_values:
        return "include"
    if text in exclude_values:
        return "exclude"
    if text in standby_values:
        return "standby"
    if text in defer_values:
        return "defer"
    if text in completed_values:
        return "completed"
    return ""


def _status_implied_plan_action(value: Any) -> str:
    text = _clean_text(value).lower().replace("-", "_").replace(" ", "_")
    if not text:
        return ""
    if text in {"done", "complete", "completed", "success", "succeeded", "finished", "tested"}:
        return "completed"
    if text in {"failed", "cancelled", "canceled", "blocked", "not_available", "material_unavailable"}:
        return "exclude"
    if text in {"running", "in_progress", "started", "current", "ongoing"}:
        return "include"
    return ""


def load_experiment_plan_overrides(override_csv: str | Path | None) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if override_csv is None or not str(override_csv).strip():
        return {}, {"enabled": False, "source_csv": None, "row_count": 0}

    path = Path(override_csv).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"experiment_plan_override_csv not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    if "nanobody_id" not in df.columns:
        raise ValueError("experiment_plan_override_csv must contain a nanobody_id column")

    records: dict[str, dict[str, Any]] = {}
    duplicate_ids: list[str] = []
    for idx, row in df.iterrows():
        nanobody_id = _clean_text(row.get("nanobody_id"))
        if not nanobody_id:
            continue
        if nanobody_id in records:
            duplicate_ids.append(nanobody_id)
        raw_action = _first_nonempty(
            row,
            ["manual_plan_action", "plan_override", "plan_action", "override_action", "decision_override"],
        )
        records[nanobody_id] = {
            "manual_plan_action": _normalize_manual_plan_action(raw_action),
            "manual_plan_action_raw": raw_action,
            "manual_plan_reason": _first_nonempty(
                row,
                ["manual_plan_reason", "override_reason", "plan_reason", "reason"],
            ),
            "experiment_cost": _first_nonempty(row, ["experiment_cost", "cost", "budget_cost"]),
            "experiment_owner": _first_nonempty(row, ["experiment_owner", "owner", "assignee", "operator"]),
            "experiment_status": _first_nonempty(row, ["experiment_status", "status", "sample_status"]),
            "experiment_result": _first_nonempty(row, ["experiment_result", "result", "validation_result", "assay_result"]),
            "validation_label": _first_nonempty(row, ["validation_label", "experiment_label", "assay_label"]),
            "experiment_note": _first_nonempty(row, ["experiment_note", "note", "notes", "comment", "comments"]),
            "manual_plan_source_row": int(idx) + 2,
        }

    return records, {
        "enabled": True,
        "source_csv": str(path),
        "row_count": int(len(df)),
        "usable_override_count": int(len(records)),
        "duplicate_nanobody_ids": sorted(set(duplicate_ids)),
    }


def _choose_diversity_column(work: pd.DataFrame, requested_col: str) -> tuple[str | None, str]:
    requested = str(requested_col or "auto").strip()
    if requested and requested.lower() != "auto":
        if requested not in work.columns:
            raise ValueError(f"diversity_group_col not found in consensus_csv: {requested}")
        return requested, f"column:{requested}"

    candidates = [
        "diversity_group",
        "sequence_cluster",
        "cdr3_cluster",
        "cdr3_cluster_id",
        "nanobody_cluster",
        "nanobody_family",
        "family",
        "source_library",
        "library",
        "epitope_cluster",
        "predicted_epitope_cluster",
        "interface_cluster",
        "pocket_contact_cluster",
    ]
    for col in candidates:
        if col not in work.columns:
            continue
        values = work[col].map(_clean_group_value)
        non_empty = values[values.ne("")]
        if non_empty.empty:
            continue
        if non_empty.nunique(dropna=True) >= 2:
            return col, f"column:{col}"
    return None, "profile"


def _profile_group(
    row: pd.Series,
    *,
    high_score_threshold: float,
    low_confidence_threshold: float,
    qc_risk_threshold: float,
    rank_delta_threshold: float,
) -> str:
    consensus = _safe_float(row.get("consensus_score"), 0.0)
    confidence = _safe_float(row.get("confidence_score"), 0.0)
    qc_risk = _safe_float(row.get("qc_risk_score"), 0.0)
    abs_rank_delta = _safe_float(row.get("abs_rank_delta"), 0.0)
    tier = str(row.get("decision_tier") or "").strip().lower()
    flags = {
        item.strip()
        for item in str(row.get("review_reason_flags") or row.get("risk_flags") or "").split(";")
        if item.strip() and item.strip() != "none"
    }

    if "close_score_competition" in flags or _as_bool(row.get("close_score_competition_warning", False)):
        return "profile:close_decision"
    if abs_rank_delta >= float(rank_delta_threshold) or "rule_ml_rank_disagreement" in flags:
        return "profile:rule_ml_disagreement"
    if qc_risk >= float(qc_risk_threshold) or "pocket_overwide" in flags or "high_qc_risk" in flags:
        return "profile:qc_or_pocket_risk"
    if "conformer_instability" in flags or _as_bool(row.get("conformer_instability_warning", False)):
        return "profile:conformer_instability"
    if confidence < float(low_confidence_threshold):
        return "profile:low_confidence"
    if tier == "priority" and consensus >= float(high_score_threshold):
        return "profile:high_confidence_priority"
    if consensus >= float(high_score_threshold):
        return "profile:high_score_standard"
    return "profile:backup_or_exploratory"


def _infer_diversity_groups(
    work: pd.DataFrame,
    *,
    diversity_mode: str,
    diversity_group_col: str,
    high_score_threshold: float,
    low_confidence_threshold: float,
    qc_risk_threshold: float,
    rank_delta_threshold: float,
) -> tuple[pd.Series, str]:
    mode = str(diversity_mode or "auto").strip().lower()
    if mode == "none":
        return pd.Series("diversity_disabled", index=work.index), "none"

    if mode == "column":
        col, source = _choose_diversity_column(work, diversity_group_col)
        if col is None:
            raise ValueError("diversity_mode=column requires a valid diversity_group_col or an auto-detected group column.")
        groups = work[col].map(_clean_group_value).replace("", "ungrouped")
        return groups.astype(str), source

    if mode == "auto":
        col, source = _choose_diversity_column(work, diversity_group_col)
        if col is not None:
            groups = work[col].map(_clean_group_value).replace("", "ungrouped")
            return groups.astype(str), source

    groups = work.apply(
        lambda row: _profile_group(
            row,
            high_score_threshold=float(high_score_threshold),
            low_confidence_threshold=float(low_confidence_threshold),
            qc_risk_threshold=float(qc_risk_threshold),
            rank_delta_threshold=float(rank_delta_threshold),
        ),
        axis=1,
    )
    return groups.astype(str), "profile"


def _apply_diversity_ordering(
    work: pd.DataFrame,
    *,
    diversity_mode: str,
    diversity_group_col: str,
    diversity_penalty: float,
    diversity_max_per_group: int,
    high_score_threshold: float,
    low_confidence_threshold: float,
    qc_risk_threshold: float,
    rank_delta_threshold: float,
) -> pd.DataFrame:
    work = work.copy()
    groups, group_source = _infer_diversity_groups(
        work,
        diversity_mode=diversity_mode,
        diversity_group_col=diversity_group_col,
        high_score_threshold=high_score_threshold,
        low_confidence_threshold=low_confidence_threshold,
        qc_risk_threshold=qc_risk_threshold,
        rank_delta_threshold=rank_delta_threshold,
    )
    work["diversity_group"] = groups
    work["diversity_group_source"] = group_source
    work["raw_experiment_priority_score"] = work["experiment_priority_score"]

    tier_order = {"validate_now": 0, "review_first": 1, "backup": 2}
    work["_suggestion_tier_order"] = work["suggestion_tier"].map(tier_order).fillna(9)
    base_sorted = work.sort_values(
        by=["_suggestion_tier_order", "experiment_priority_score", "consensus_score", "confidence_score"],
        ascending=[True, False, False, False],
        na_position="last",
    ).copy()

    if str(diversity_mode or "auto").strip().lower() == "none" or base_sorted["diversity_group"].nunique(dropna=False) <= 1:
        base_sorted["diversity_seen_before"] = 0
        base_sorted["diversity_adjustment"] = 0.0
        base_sorted["diversity_adjusted_priority_score"] = base_sorted["experiment_priority_score"]
        base_sorted["diversity_note"] = (
            "diversity disabled"
            if str(diversity_mode or "").strip().lower() == "none"
            else "only one diversity group detected; raw priority order kept"
        )
        return base_sorted.drop(columns=["_suggestion_tier_order"]).reset_index(drop=True)

    selected_indices: list[Any] = []
    group_counts: dict[str, int] = {}
    seen_before_by_index: dict[Any, int] = {}
    adjustment_by_index: dict[Any, float] = {}
    adjusted_score_by_index: dict[Any, float] = {}
    note_by_index: dict[Any, str] = {}
    penalty_weight = max(0.0, float(diversity_penalty))
    soft_max = max(0, int(diversity_max_per_group))

    for tier_value in sorted(base_sorted["_suggestion_tier_order"].dropna().unique()):
        remaining = list(base_sorted[base_sorted["_suggestion_tier_order"].eq(tier_value)].index)
        while remaining:
            scored: list[tuple[float, float, Any, int, float]] = []
            for idx in remaining:
                row = base_sorted.loc[idx]
                group = str(row.get("diversity_group") or "ungrouped")
                seen = int(group_counts.get(group, 0))
                penalty = penalty_weight * seen
                if soft_max > 0 and seen >= soft_max:
                    penalty += penalty_weight * (seen - soft_max + 1)
                raw_score = _safe_float(row.get("experiment_priority_score"), 0.0)
                adjusted = raw_score - penalty
                scored.append((adjusted, raw_score, idx, seen, penalty))
            scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
            adjusted, raw_score, chosen_idx, seen, penalty = scored[0]
            group = str(base_sorted.loc[chosen_idx].get("diversity_group") or "ungrouped")
            selected_indices.append(chosen_idx)
            seen_before_by_index[chosen_idx] = seen
            adjustment_by_index[chosen_idx] = -float(penalty)
            adjusted_score_by_index[chosen_idx] = float(adjusted)
            if seen == 0:
                note = f"first candidate from group {group}"
            elif soft_max > 0 and seen >= soft_max:
                note = f"group {group} already selected {seen} times; soft diversity penalty applied"
            else:
                note = f"group {group} already selected {seen} times"
            note_by_index[chosen_idx] = note
            group_counts[group] = seen + 1
            remaining.remove(chosen_idx)

    ordered = base_sorted.loc[selected_indices].copy()
    ordered["diversity_seen_before"] = [seen_before_by_index.get(idx, 0) for idx in ordered.index]
    ordered["diversity_adjustment"] = [adjustment_by_index.get(idx, 0.0) for idx in ordered.index]
    ordered["diversity_adjusted_priority_score"] = [
        adjusted_score_by_index.get(idx, _safe_float(ordered.loc[idx].get("experiment_priority_score"), 0.0))
        for idx in ordered.index
    ]
    ordered["diversity_note"] = [note_by_index.get(idx, "") for idx in ordered.index]
    return ordered.drop(columns=["_suggestion_tier_order"]).reset_index(drop=True)


def _build_reason(row: pd.Series, *, high_score_threshold: float, low_confidence_threshold: float, qc_risk_threshold: float, rank_delta_threshold: float) -> tuple[str, str]:
    consensus = _safe_float(row.get("consensus_score"), 0.0)
    confidence = _safe_float(row.get("confidence_score"), 0.0)
    qc_risk = _safe_float(row.get("qc_risk_score"), 0.0)
    abs_rank_delta = _safe_float(row.get("abs_rank_delta"), 0.0)
    tier = str(row.get("decision_tier") or "").lower()

    reasons: list[str] = []
    detailed_reasons = str(row.get("low_confidence_reasons") or "").strip()
    review_flags = {
        item.strip()
        for item in str(row.get("review_reason_flags") or "").split(";")
        if item.strip() and item.strip() != "none"
    }
    if consensus >= high_score_threshold and confidence < low_confidence_threshold:
        reasons.append("高分但低可信")
    if abs_rank_delta >= rank_delta_threshold:
        reasons.append("Rule/ML 排名分歧大")
    if qc_risk >= qc_risk_threshold:
        reasons.append("QC 或 pocket overwide 风险较高")
    if "close_score_competition" in review_flags:
        reasons.append("与相邻候选分差很小")
    if "conformer_instability" in review_flags:
        reasons.append("构象间波动偏大")
    if "rule_ml_score_gap" in review_flags and "Rule/ML 排名分歧大" not in reasons:
        reasons.append("Rule/ML 分数分歧大")
    if tier == "priority" and confidence >= low_confidence_threshold:
        reasons.append("高优先级且可信度可接受")
    if consensus >= high_score_threshold and qc_risk < qc_risk_threshold:
        reasons.append("综合分高且 QC 风险可控")
    if not reasons:
        reasons.append("常规候选，可排在重点样本之后")

    if "高分但低可信" in reasons or "Rule/ML 排名分歧大" in reasons or "与相邻候选分差很小" in reasons:
        action = "优先人工复核结构与打分依据，再决定是否实验验证"
    elif "QC 或 pocket overwide 风险较高" in reasons:
        action = "先复核输入结构、pocket 定义和失败/warning 行"
    elif "构象间波动偏大" in reasons:
        action = "先复核不同 conformer 的 pose 稳定性，再决定是否实验验证"
    elif "高优先级且可信度可接受" in reasons or "综合分高且 QC 风险可控" in reasons:
        action = "优先进入下一轮实验验证"
    else:
        action = "暂缓，作为备用候选或等待更多数据"
    if detailed_reasons and detailed_reasons != "未发现明显低可信原因":
        reasons.append(f"细分原因: {detailed_reasons}")
    return "；".join(reasons), action


def build_next_experiment_suggestions(
    consensus_df: pd.DataFrame,
    *,
    top_n: int = 20,
    high_score_threshold: float = 0.70,
    low_confidence_threshold: float = 0.55,
    qc_risk_threshold: float = 0.40,
    rank_delta_threshold: float = 3.0,
    diversity_mode: str = "auto",
    diversity_group_col: str = "auto",
    diversity_penalty: float = 0.08,
    diversity_max_per_group: int = 2,
) -> pd.DataFrame:
    if consensus_df.empty:
        raise ValueError("consensus_csv is empty.")
    if "nanobody_id" not in consensus_df.columns:
        raise ValueError("consensus_csv is missing nanobody_id column.")

    work = consensus_df.copy()
    for col in ["consensus_score", "confidence_score", "qc_risk_score", "abs_rank_delta", "ml_score", "rule_score"]:
        if col in work.columns:
            work[col] = _to_numeric(work[col])
        else:
            work[col] = np.nan

    source_coverage = _to_numeric(work["source_coverage_score"]) if "source_coverage_score" in work.columns else pd.Series(1.0, index=work.index)
    score_component = work["consensus_score"].fillna(0.0).clip(0.0, 1.0)
    low_conf_component = (1.0 - work["confidence_score"].fillna(0.5).clip(0.0, 1.0))
    disagreement_component = (work["abs_rank_delta"].fillna(0.0) / max(float(rank_delta_threshold), 1.0)).clip(0.0, 1.0)
    qc_component = work["qc_risk_score"].fillna(0.0).clip(0.0, 1.0)
    missing_source_component = (1.0 - source_coverage.fillna(1.0).clip(0.0, 1.0))
    close_competition_component = (
        pd.to_numeric(work["close_score_competition_warning"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        if "close_score_competition_warning" in work.columns
        else pd.Series(0.0, index=work.index)
    )
    conformer_instability_component = (
        pd.to_numeric(work["conformer_instability_warning"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        if "conformer_instability_warning" in work.columns
        else pd.Series(0.0, index=work.index)
    )

    work["experiment_priority_score"] = (
        0.44 * score_component
        + 0.16 * low_conf_component
        + 0.14 * disagreement_component
        + 0.10 * qc_component
        + 0.06 * missing_source_component
        + 0.06 * close_competition_component
        + 0.04 * conformer_instability_component
    ).clip(0.0, 1.0)

    reasons: list[str] = []
    actions: list[str] = []
    for _, row in work.iterrows():
        reason, action = _build_reason(
            row,
            high_score_threshold=float(high_score_threshold),
            low_confidence_threshold=float(low_confidence_threshold),
            qc_risk_threshold=float(qc_risk_threshold),
            rank_delta_threshold=float(rank_delta_threshold),
        )
        reasons.append(reason)
        actions.append(action)
    work["suggestion_reason"] = reasons
    work["recommended_next_action"] = actions

    decision_tier = work["decision_tier"].fillna("").astype(str).str.lower() if "decision_tier" in work.columns else pd.Series("", index=work.index)
    validate_mask = (
        (work["experiment_priority_score"] >= 0.72)
        | (
            decision_tier.eq("priority")
            & work["consensus_score"].fillna(0.0).ge(float(high_score_threshold))
            & work["confidence_score"].fillna(0.0).ge(float(low_confidence_threshold))
            & work["qc_risk_score"].fillna(0.0).lt(float(qc_risk_threshold))
        )
    )
    review_mask = (
        work["experiment_priority_score"].ge(0.52)
        | work["consensus_score"].fillna(0.0).ge(float(high_score_threshold))
        | work["confidence_score"].fillna(1.0).lt(float(low_confidence_threshold))
        | work["qc_risk_score"].fillna(0.0).ge(float(qc_risk_threshold))
        | work["abs_rank_delta"].fillna(0.0).ge(float(rank_delta_threshold))
        | close_competition_component.gt(0)
        | conformer_instability_component.gt(0)
    )
    work["suggestion_tier"] = np.where(validate_mask, "validate_now", np.where(review_mask, "review_first", "backup"))

    work = _apply_diversity_ordering(
        work,
        diversity_mode=str(diversity_mode),
        diversity_group_col=str(diversity_group_col),
        diversity_penalty=float(diversity_penalty),
        diversity_max_per_group=int(diversity_max_per_group),
        high_score_threshold=float(high_score_threshold),
        low_confidence_threshold=float(low_confidence_threshold),
        qc_risk_threshold=float(qc_risk_threshold),
        rank_delta_threshold=float(rank_delta_threshold),
    )
    work.insert(0, "suggestion_rank", np.arange(1, len(work) + 1, dtype=int))
    if int(top_n) > 0:
        work = work.head(int(top_n)).copy()

    preferred = [
        "suggestion_rank",
        "nanobody_id",
        "suggestion_tier",
        "experiment_priority_score",
        "diversity_adjusted_priority_score",
        "diversity_group",
        "diversity_group_source",
        "diversity_seen_before",
        "diversity_adjustment",
        "diversity_note",
        "recommended_next_action",
        "suggestion_reason",
        "consensus_rank",
        "decision_tier",
        "confidence_level",
        "consensus_score",
        "confidence_score",
        "qc_risk_score",
        "abs_rank_delta",
        "ml_score",
        "rule_score",
        "risk_flags",
        "review_reason_flags",
        "low_confidence_reasons",
        "consensus_explanation",
    ]
    remaining = [col for col in work.columns if col not in preferred]
    return work[[col for col in preferred if col in work.columns] + remaining]


def _summary_payload(
    suggestions: pd.DataFrame,
    *,
    consensus_csv: Path,
    diversity_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    diversity_group_counts = (
        {str(k): int(v) for k, v in suggestions["diversity_group"].value_counts(dropna=False).to_dict().items()}
        if "diversity_group" in suggestions.columns
        else {}
    )
    diversity_source = (
        str(suggestions["diversity_group_source"].dropna().iloc[0])
        if "diversity_group_source" in suggestions.columns and not suggestions["diversity_group_source"].dropna().empty
        else "none"
    )
    return {
        "consensus_csv": str(consensus_csv),
        "suggestion_count": int(len(suggestions)),
        "tier_counts": {str(k): int(v) for k, v in suggestions["suggestion_tier"].value_counts(dropna=False).to_dict().items()},
        "mean_experiment_priority_score": float(_to_numeric(suggestions["experiment_priority_score"]).mean()),
        "mean_diversity_adjusted_priority_score": (
            float(_to_numeric(suggestions["diversity_adjusted_priority_score"]).mean())
            if "diversity_adjusted_priority_score" in suggestions.columns
            else None
        ),
        "diversity_group_source": diversity_source,
        "diversity_group_counts": diversity_group_counts,
        "diversity_repeat_count": (
            int(_to_numeric(suggestions["diversity_seen_before"]).gt(0).sum())
            if "diversity_seen_before" in suggestions.columns
            else 0
        ),
        "diversity_config": diversity_config or {},
        "top_suggestions": suggestions.head(10).replace({np.nan: None}).to_dict(orient="records"),
        "formula": {
            "experiment_priority_score": "0.44*consensus + 0.16*(1-confidence) + 0.14*rank_disagreement + 0.10*qc_risk + 0.06*missing_source + 0.06*close_competition + 0.04*conformer_instability",
            "diversity_adjusted_priority_score": "experiment_priority_score - diversity_penalty * already_selected_from_same_group, applied within each suggestion tier",
        },
    }


def build_experiment_plan(
    suggestions: pd.DataFrame,
    *,
    experiment_budget: int = 8,
    validate_budget: int = 4,
    review_budget: int = 3,
    standby_budget: int = 3,
    group_quota: int = 2,
    plan_overrides: dict[str, dict[str, Any]] | None = None,
    plan_override_info: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if suggestions.empty:
        raise ValueError("suggestions table is empty.")

    work = suggestions.copy()
    if "suggestion_rank" in work.columns:
        work["_plan_sort_rank"] = _to_numeric(work["suggestion_rank"]).fillna(np.inf)
        work = work.sort_values("_plan_sort_rank", ascending=True).drop(columns=["_plan_sort_rank"]).reset_index(drop=True)

    total_budget = max(0, int(experiment_budget))
    validate_limit = max(0, int(validate_budget))
    review_limit = max(0, int(review_budget))
    standby_limit = max(0, int(standby_budget))
    group_limit = max(0, int(group_quota))

    include_count = 0
    standby_count = 0
    include_tier_counts = {"validate_now": 0, "review_first": 0, "backup": 0}
    include_group_counts: dict[str, int] = {}
    overrides = plan_overrides or {}
    matched_override_ids: set[str] = set()

    decisions: list[str] = []
    phases: list[str] = []
    reasons: list[str] = []
    tier_count_before_values: list[int] = []
    group_count_before_values: list[int] = []
    manual_actions: list[str] = []
    manual_action_raw_values: list[str] = []
    manual_override_applied_values: list[bool] = []
    manual_reason_values: list[str] = []
    experiment_cost_values: list[str] = []
    experiment_owner_values: list[str] = []
    experiment_status_values: list[str] = []
    experiment_result_values: list[str] = []
    validation_label_values: list[str] = []
    experiment_note_values: list[str] = []
    manual_source_row_values: list[int | None] = []

    for _, row in work.iterrows():
        nanobody_id = _clean_text(row.get("nanobody_id"))
        override = overrides.get(nanobody_id, {}) if nanobody_id else {}
        if override:
            matched_override_ids.add(nanobody_id)

        manual_action = _clean_text(override.get("manual_plan_action"))
        experiment_status = _clean_text(override.get("experiment_status"))
        status_action = _status_implied_plan_action(experiment_status)
        effective_manual_action = manual_action or status_action
        manual_reason = _clean_text(override.get("manual_plan_reason"))
        manual_reason_suffix = f" Note: {manual_reason}" if manual_reason else ""

        tier = str(row.get("suggestion_tier") or "backup")
        group = str(row.get("diversity_group") or "ungrouped")
        tier_count_before = int(include_tier_counts.get(tier, 0))
        group_count_before = int(include_group_counts.get(group, 0))
        blockers: list[str] = []

        if effective_manual_action == "include":
            decision = "include_now"
            phase = "manual_include" if manual_action else "status_in_progress"
            reason = (
                "Manual/status override included this candidate in the current round; "
                "budget and quota blockers are intentionally bypassed."
                + manual_reason_suffix
            )
            include_count += 1
            include_tier_counts[tier] = tier_count_before + 1
            include_group_counts[group] = group_count_before + 1
        elif effective_manual_action == "exclude":
            decision = "defer"
            phase = "manual_excluded" if manual_action else "status_excluded"
            reason = "Manual/status override excluded this candidate from the current round." + manual_reason_suffix
        elif effective_manual_action == "standby":
            decision = "standby"
            phase = "manual_standby"
            reason = "Manual override placed this candidate in the standby pool." + manual_reason_suffix
            standby_count += 1
        elif effective_manual_action == "defer":
            decision = "defer"
            phase = "manual_defer"
            reason = "Manual override deferred this candidate to a later round." + manual_reason_suffix
        elif effective_manual_action == "completed":
            decision = "defer"
            phase = "already_completed"
            reason = "Experiment status indicates this candidate was already completed, so it is not scheduled again." + manual_reason_suffix
        else:
            if include_count >= total_budget:
                blockers.append(f"total experiment budget reached ({total_budget})")
            if tier == "validate_now" and tier_count_before >= validate_limit:
                blockers.append(f"validate_now budget reached ({validate_limit})")
            if tier == "review_first" and tier_count_before >= review_limit:
                blockers.append(f"review_first budget reached ({review_limit})")
            if tier == "backup":
                blockers.append("backup tier is kept as standby by default")
            if group_limit > 0 and group_count_before >= group_limit:
                blockers.append(f"diversity group quota reached for {group} ({group_limit})")

            if not blockers:
                decision = "include_now"
                phase = "current_round"
                reason = (
                    f"Included within total budget {include_count + 1}/{total_budget}; "
                    f"tier {tier} count {tier_count_before + 1}; "
                    f"group {group} count {group_count_before + 1}."
                )
                include_count += 1
                include_tier_counts[tier] = tier_count_before + 1
                include_group_counts[group] = group_count_before + 1
            elif standby_count < standby_limit:
                decision = "standby"
                phase = "standby_pool"
                reason = "Standby candidate; not included now because " + "; ".join(blockers) + "."
                standby_count += 1
            else:
                decision = "defer"
                phase = "later_round"
                reason = "Deferred because " + "; ".join(blockers) + "."

        decisions.append(decision)
        phases.append(phase)
        reasons.append(reason)
        tier_count_before_values.append(tier_count_before)
        group_count_before_values.append(group_count_before)
        manual_actions.append(effective_manual_action)
        manual_action_raw_values.append(_clean_text(override.get("manual_plan_action_raw")))
        manual_override_applied_values.append(bool(effective_manual_action))
        manual_reason_values.append(manual_reason)
        experiment_cost_values.append(_clean_text(override.get("experiment_cost")))
        experiment_owner_values.append(_clean_text(override.get("experiment_owner")))
        experiment_status_values.append(experiment_status)
        experiment_result_values.append(_clean_text(override.get("experiment_result")))
        validation_label_values.append(_clean_text(override.get("validation_label")))
        experiment_note_values.append(_clean_text(override.get("experiment_note")))
        source_row = override.get("manual_plan_source_row")
        manual_source_row_values.append(int(source_row) if source_row not in {None, ""} else None)

    work.insert(0, "plan_rank", np.arange(1, len(work) + 1, dtype=int))
    work.insert(1, "plan_decision", decisions)
    work.insert(2, "plan_phase", phases)
    work.insert(3, "plan_reason", reasons)
    work.insert(4, "plan_tier_count_before", tier_count_before_values)
    work.insert(5, "plan_group_count_before", group_count_before_values)
    work["manual_plan_action"] = manual_actions
    work["manual_plan_action_raw"] = manual_action_raw_values
    work["manual_plan_override_applied"] = manual_override_applied_values
    work["manual_plan_reason"] = manual_reason_values
    work["experiment_cost"] = experiment_cost_values
    work["experiment_owner"] = experiment_owner_values
    work["experiment_status"] = experiment_status_values
    work["experiment_result"] = experiment_result_values
    work["validation_label"] = validation_label_values
    work["experiment_note"] = experiment_note_values
    work["manual_plan_source_row"] = manual_source_row_values
    override_ids = set(overrides.keys())
    info = dict(plan_override_info or {})
    info.update(
        {
            "matched_override_count": int(len(matched_override_ids)),
            "unmatched_override_count": int(len(override_ids - matched_override_ids)),
            "unmatched_nanobody_ids": sorted(override_ids - matched_override_ids)[:50],
        }
    )
    work.attrs["manual_override_info"] = info
    return work


def _plan_summary_payload(
    experiment_plan: pd.DataFrame,
    *,
    budget_config: dict[str, Any],
) -> dict[str, Any]:
    include_now = experiment_plan[experiment_plan["plan_decision"].astype(str).eq("include_now")]
    standby = experiment_plan[experiment_plan["plan_decision"].astype(str).eq("standby")]
    manual_override_info = experiment_plan.attrs.get("manual_override_info", {})
    cost_total = None
    if "experiment_cost" in include_now.columns:
        cost_values = _to_numeric(include_now["experiment_cost"])
        if cost_values.notna().any():
            cost_total = float(cost_values.sum())
    return {
        "budget_config": budget_config,
        "manual_override_info": manual_override_info if isinstance(manual_override_info, dict) else {},
        "plan_count": int(len(experiment_plan)),
        "decision_counts": {
            str(k): int(v) for k, v in experiment_plan["plan_decision"].value_counts(dropna=False).to_dict().items()
        },
        "include_now_count": int(len(include_now)),
        "standby_count": int(len(standby)),
        "manual_override_applied_count": (
            int(experiment_plan["manual_plan_override_applied"].astype(bool).sum())
            if "manual_plan_override_applied" in experiment_plan.columns
            else 0
        ),
        "experiment_status_counts": {
            str(k): int(v)
            for k, v in experiment_plan["experiment_status"]
            .map(_clean_text)
            .replace("", np.nan)
            .dropna()
            .value_counts()
            .to_dict()
            .items()
        }
        if "experiment_status" in experiment_plan.columns
        else {},
        "include_now_experiment_cost_total": cost_total,
        "include_now_tier_counts": {
            str(k): int(v) for k, v in include_now["suggestion_tier"].value_counts(dropna=False).to_dict().items()
        }
        if "suggestion_tier" in include_now.columns
        else {},
        "include_now_diversity_group_counts": {
            str(k): int(v) for k, v in include_now["diversity_group"].value_counts(dropna=False).to_dict().items()
        }
        if "diversity_group" in include_now.columns
        else {},
        "include_now_candidates": include_now.head(50).replace({np.nan: None}).to_dict(orient="records"),
        "standby_candidates": standby.head(50).replace({np.nan: None}).to_dict(orient="records"),
    }


def _experiment_plan_report_text(experiment_plan: pd.DataFrame, summary: dict[str, Any]) -> str:
    def cell(value: Any) -> str:
        return str(value if value is not None else "").replace("\n", " ").replace("|", "/")

    budget = summary.get("budget_config") if isinstance(summary.get("budget_config"), dict) else {}
    manual_override_info = (
        summary.get("manual_override_info") if isinstance(summary.get("manual_override_info"), dict) else {}
    )
    lines = [
        "# Experiment Plan",
        "",
        f"- Include now: `{summary.get('include_now_count', 0)}`",
        f"- Standby: `{summary.get('standby_count', 0)}`",
        f"- Manual/status overrides applied: `{summary.get('manual_override_applied_count', 0)}`",
        f"- Decision counts: `{json.dumps(summary.get('decision_counts', {}), ensure_ascii=False)}`",
        f"- Budget: `{json.dumps(budget, ensure_ascii=False)}`",
        f"- Override source: `{manual_override_info.get('source_csv') or 'none'}`",
        "",
        "## Current Round",
        "",
        "| plan_rank | nanobody_id | tier | group | priority | owner | status | cost | decision | reason |",
        "|---:|---|---|---|---:|---|---|---:|---|---|",
    ]

    include_now = experiment_plan[experiment_plan["plan_decision"].astype(str).eq("include_now")]
    standby = experiment_plan[experiment_plan["plan_decision"].astype(str).eq("standby")]
    defer = experiment_plan[experiment_plan["plan_decision"].astype(str).eq("defer")]
    for _, row in include_now.iterrows():
        lines.append(
            "| "
            f"{int(row.get('plan_rank', 0))} | "
            f"{cell(row.get('nanobody_id', ''))} | "
            f"{cell(row.get('suggestion_tier', ''))} | "
            f"{cell(row.get('diversity_group', ''))} | "
            f"{_safe_float(row.get('diversity_adjusted_priority_score'), _safe_float(row.get('experiment_priority_score'), 0.0)):.4f} | "
            f"{cell(row.get('experiment_owner', ''))} | "
            f"{cell(row.get('experiment_status', ''))} | "
            f"{cell(row.get('experiment_cost', ''))} | "
            f"{cell(row.get('plan_decision', ''))} | "
            f"{cell(row.get('plan_reason', ''))} |"
        )

    lines.extend(
        [
            "",
            "## Standby Pool",
            "",
            "| plan_rank | nanobody_id | tier | group | owner | status | reason |",
            "|---:|---|---|---|---|---|---|",
        ]
    )
    for _, row in standby.iterrows():
        lines.append(
            "| "
            f"{int(row.get('plan_rank', 0))} | "
            f"{cell(row.get('nanobody_id', ''))} | "
            f"{cell(row.get('suggestion_tier', ''))} | "
            f"{cell(row.get('diversity_group', ''))} | "
            f"{cell(row.get('experiment_owner', ''))} | "
            f"{cell(row.get('experiment_status', ''))} | "
            f"{cell(row.get('plan_reason', ''))} |"
        )

    lines.extend(
        [
            "",
            "## Deferred Count",
            "",
            f"- Deferred candidates: `{len(defer)}`",
            "",
            "## Notes",
            "",
            "- `include_now` is the recommended current round within budget and diversity quota.",
            "- `standby` candidates are ready alternatives if an included candidate is rejected during manual review.",
            "- The plan is generated from ranking outputs only; it does not replace biological judgment or wet-lab feasibility checks.",
        ]
    )
    return "\n".join(lines)


def build_experiment_state_ledger(experiment_plan: pd.DataFrame) -> pd.DataFrame:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows: list[dict[str, Any]] = []
    for _, row in experiment_plan.iterrows():
        manual_action = _clean_text(row.get("manual_plan_action"))
        status = _clean_text(row.get("experiment_status"))
        if not status and _clean_text(row.get("plan_decision")) == "include_now":
            status = "planned"
        rows.append(
            {
                "nanobody_id": _clean_text(row.get("nanobody_id")),
                "plan_override": manual_action,
                "experiment_status": status,
                "experiment_owner": _clean_text(row.get("experiment_owner")),
                "experiment_cost": _clean_text(row.get("experiment_cost")),
                "experiment_result": _clean_text(row.get("experiment_result")),
                "validation_label": _clean_text(row.get("validation_label")),
                "experiment_note": _clean_text(row.get("experiment_note")),
                "manual_plan_reason": _clean_text(row.get("manual_plan_reason")),
                "last_plan_decision": _clean_text(row.get("plan_decision")),
                "last_plan_phase": _clean_text(row.get("plan_phase")),
                "last_plan_rank": row.get("plan_rank", ""),
                "suggestion_tier": _clean_text(row.get("suggestion_tier")),
                "diversity_group": _clean_text(row.get("diversity_group")),
                "experiment_priority_score": row.get("experiment_priority_score", ""),
                "diversity_adjusted_priority_score": row.get("diversity_adjusted_priority_score", ""),
                "generated_at": generated_at,
            }
        )
    return pd.DataFrame(rows)


def _report_text(suggestions: pd.DataFrame, summary: dict[str, Any]) -> str:
    def cell(value: Any) -> str:
        return str(value if value is not None else "").replace("\n", " ").replace("|", "/")

    lines = [
        "# Next Experiment Suggestions",
        "",
        f"- Suggestions: `{summary['suggestion_count']}`",
        f"- Mean priority score: `{summary['mean_experiment_priority_score']:.4f}`",
        f"- Mean diversity-adjusted priority score: `{_safe_float(summary.get('mean_diversity_adjusted_priority_score'), 0.0):.4f}`",
        f"- Tier counts: `{json.dumps(summary['tier_counts'], ensure_ascii=False)}`",
        f"- Diversity group source: `{summary.get('diversity_group_source', 'none')}`",
        f"- Diversity group counts: `{json.dumps(summary.get('diversity_group_counts', {}), ensure_ascii=False)}`",
        "",
        "## Top Suggestions",
        "",
        "| rank | nanobody_id | tier | raw priority | diversity priority | diversity group | action | reason |",
        "|---:|---|---|---:|---:|---|---|---|",
    ]
    for _, row in suggestions.head(20).iterrows():
        lines.append(
            "| "
            f"{int(row['suggestion_rank'])} | "
            f"{cell(row.get('nanobody_id', ''))} | "
            f"{cell(row.get('suggestion_tier', ''))} | "
            f"{_safe_float(row.get('experiment_priority_score'), 0.0):.4f} | "
            f"{_safe_float(row.get('diversity_adjusted_priority_score'), _safe_float(row.get('experiment_priority_score'), 0.0)):.4f} | "
            f"{cell(row.get('diversity_group', ''))} | "
            f"{cell(row.get('recommended_next_action', ''))} | "
            f"{cell(row.get('suggestion_reason', ''))} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `validate_now`: high combined potential or uncertainty, worth near-term validation or strict review.",
            "- `review_first`: promising but needs structure/QC/ranking review before wet-lab resources.",
            "- `backup`: keep as reserve unless additional evidence appears.",
            "- Diversity ordering is a soft queue-level adjustment. It does not change `experiment_priority_score`, Rule scores, ML scores or consensus scores.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = _build_parser().parse_args()
    consensus_csv = Path(args.consensus_csv).expanduser().resolve()
    if not consensus_csv.exists():
        raise FileNotFoundError(f"consensus_csv not found: {consensus_csv}")

    consensus_df = pd.read_csv(consensus_csv, low_memory=False)
    suggestions = build_next_experiment_suggestions(
        consensus_df,
        top_n=int(args.top_n),
        high_score_threshold=float(args.high_score_threshold),
        low_confidence_threshold=float(args.low_confidence_threshold),
        qc_risk_threshold=float(args.qc_risk_threshold),
        rank_delta_threshold=float(args.rank_delta_threshold),
        diversity_mode=str(args.diversity_mode),
        diversity_group_col=str(args.diversity_group_col),
        diversity_penalty=float(args.diversity_penalty),
        diversity_max_per_group=int(args.diversity_max_per_group),
    )

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    suggestions_csv = out_dir / "next_experiment_suggestions.csv"
    summary_json = out_dir / "next_experiment_suggestions_summary.json"
    report_md = out_dir / "next_experiment_suggestions_report.md"
    experiment_plan_csv = out_dir / "experiment_plan.csv"
    experiment_plan_summary_json = out_dir / "experiment_plan_summary.json"
    experiment_plan_md = out_dir / "experiment_plan.md"
    experiment_state_ledger_csv = out_dir / "experiment_plan_state_ledger.csv"

    suggestions.to_csv(suggestions_csv, index=False)
    summary = _summary_payload(
        suggestions,
        consensus_csv=consensus_csv,
        diversity_config={
            "diversity_mode": str(args.diversity_mode),
            "diversity_group_col": str(args.diversity_group_col),
            "diversity_penalty": float(args.diversity_penalty),
            "diversity_max_per_group": int(args.diversity_max_per_group),
        },
    )
    summary_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    report_md.write_text(_report_text(suggestions, summary), encoding="utf-8")

    budget_config = {
        "experiment_budget": int(args.experiment_plan_budget),
        "validate_budget": int(args.experiment_plan_validate_budget),
        "review_budget": int(args.experiment_plan_review_budget),
        "standby_budget": int(args.experiment_plan_standby_budget),
        "group_quota": int(args.experiment_plan_group_quota),
        "override_csv": str(Path(args.experiment_plan_override_csv).expanduser().resolve())
        if args.experiment_plan_override_csv
        else None,
    }
    plan_overrides, plan_override_info = load_experiment_plan_overrides(args.experiment_plan_override_csv)
    experiment_plan = build_experiment_plan(
        suggestions,
        experiment_budget=int(args.experiment_plan_budget),
        validate_budget=int(args.experiment_plan_validate_budget),
        review_budget=int(args.experiment_plan_review_budget),
        standby_budget=int(args.experiment_plan_standby_budget),
        group_quota=int(args.experiment_plan_group_quota),
        plan_overrides=plan_overrides,
        plan_override_info=plan_override_info,
    )
    experiment_plan.to_csv(experiment_plan_csv, index=False)
    plan_summary = _plan_summary_payload(experiment_plan, budget_config=budget_config)
    experiment_plan_summary_json.write_text(
        json.dumps(plan_summary, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    experiment_plan_md.write_text(_experiment_plan_report_text(experiment_plan, plan_summary), encoding="utf-8")
    experiment_state_ledger = build_experiment_state_ledger(experiment_plan)
    experiment_state_ledger.to_csv(experiment_state_ledger_csv, index=False)

    print(f"Saved: {suggestions_csv}")
    print(f"Saved: {summary_json}")
    print(f"Saved: {report_md}")
    print(f"Saved: {experiment_plan_csv}")
    print(f"Saved: {experiment_plan_summary_json}")
    print(f"Saved: {experiment_plan_md}")
    print(f"Saved: {experiment_state_ledger_csv}")


if __name__ == "__main__":
    main()
