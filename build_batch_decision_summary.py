"""Build a compact run-level decision summary from existing pipeline outputs.

This is a post-processing explanation layer. It reads quality gate, consensus
ranking, score cards and experiment plan artifacts; it does not change any
model, rule or consensus score.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ARTIFACT_FALLBACKS = {
    "quality_gate_summary_json": ("quality_gate", "quality_gate_summary.json"),
    "quality_gate_checks_csv": ("quality_gate", "quality_gate_checks.csv"),
    "quality_gate_report_md": ("quality_gate", "quality_gate_report.md"),
    "consensus_ranking_csv": ("consensus_outputs", "consensus_ranking.csv"),
    "score_explanation_cards_csv": ("score_explanation_cards", "score_explanation_cards.csv"),
    "score_explanation_cards_md": ("score_explanation_cards", "score_explanation_cards.md"),
    "score_explanation_cards_html": ("score_explanation_cards", "score_explanation_cards.html"),
    "candidate_report_index_html": ("candidate_report_cards", "index.html"),
    "candidate_comparison_report_md": ("candidate_comparisons", "candidate_comparison_report.md"),
    "experiment_suggestions_csv": ("experiment_suggestions", "next_experiment_suggestions.csv"),
    "experiment_plan_csv": ("experiment_suggestions", "experiment_plan.csv"),
    "experiment_plan_md": ("experiment_suggestions", "experiment_plan.md"),
    "validation_evidence_summary_json": ("validation_evidence_audit", "validation_evidence_summary.json"),
    "validation_evidence_report_md": ("validation_evidence_audit", "validation_evidence_report.md"),
    "validation_evidence_topk_csv": ("validation_evidence_audit", "validation_evidence_topk.csv"),
    "validation_evidence_action_items_csv": ("validation_evidence_audit", "validation_evidence_action_items.csv"),
    "run_provenance_card_md": ("provenance", "run_provenance_card.md"),
}

STATUS_ORDER = {"FAIL": 0, "WARN": 1, "PASS": 2}
RISK_STATUS_ORDER = {"FAIL": 0, "WARN": 1, "PASS": 2}
BENIGN_RISK_TOKENS = {
    "none",
    "nan",
    "null",
    "未发现明显低可信原因",
    "未发现明显高风险项，但仍建议人工检查结构和输入质量",
    "未发现明显高风险项，但仍建议人工检查结构和输入质量。",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a batch-level decision summary from existing ML artifacts")
    parser.add_argument("--summary_json", default=None, help="Path to recommended_pipeline_summary.json or smoke summary")
    parser.add_argument("--out_dir", default="batch_decision_summary", help="Output directory")
    parser.add_argument("--top_n", type=int, default=5, help="Top-N risk causes and rows to include")
    parser.add_argument("--quality_gate_summary_json", default=None)
    parser.add_argument("--quality_gate_checks_csv", default=None)
    parser.add_argument("--consensus_csv", default=None)
    parser.add_argument("--score_cards_csv", default=None)
    parser.add_argument("--experiment_suggestions_csv", default=None)
    parser.add_argument("--experiment_plan_csv", default=None)
    parser.add_argument("--validation_evidence_summary_json", default=None)
    parser.add_argument("--validation_evidence_topk_csv", default=None)
    parser.add_argument("--validation_evidence_action_items_csv", default=None)
    parser.add_argument("--feature_csv", default=None)
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


def _safe_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return default
    return number


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if path is None or not str(path).strip():
        return {}
    target = Path(path).expanduser()
    if not target.exists():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: str | Path | None) -> pd.DataFrame | None:
    if path is None or not str(path).strip():
        return None
    target = Path(path).expanduser()
    if not target.exists():
        return None
    try:
        return pd.read_csv(target, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(target, encoding="utf-8-sig", low_memory=False)


def _resolve_path(value: Any, *, base_dir: Path | None = None) -> Path | None:
    text = _clean_text(value)
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path.resolve()


def _summary_root(summary_path: Path | None, summary: dict[str, Any]) -> Path:
    out_dir = _resolve_path(summary.get("out_dir"), base_dir=summary_path.parent if summary_path else None)
    if out_dir is not None:
        return out_dir
    if summary_path is not None:
        return summary_path.parent
    return Path.cwd()


def _artifact_path(
    summary_path: Path | None,
    summary: dict[str, Any],
    key: str,
    *,
    override: str | Path | None = None,
) -> Path | None:
    if override is not None and str(override).strip():
        return _resolve_path(override, base_dir=Path.cwd())

    artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
    base_dir = summary_path.parent if summary_path is not None else Path.cwd()
    candidate = _resolve_path(artifacts.get(key), base_dir=base_dir)
    if candidate is not None and candidate.exists():
        return candidate

    fallback = ARTIFACT_FALLBACKS.get(key)
    if fallback:
        path = _summary_root(summary_path, summary).joinpath(*fallback).resolve()
        if path.exists():
            return path

    return candidate


def _split_tokens(value: Any) -> list[str]:
    text = _clean_text(value)
    if not text:
        return []
    tokens = [
        part.strip()
        for part in re.split(r"[;；|,\n]+", text)
        if part.strip()
    ]
    out: list[str] = []
    for token in tokens:
        normalized = token.strip()
        if normalized.lower() in BENIGN_RISK_TOKENS or normalized in BENIGN_RISK_TOKENS:
            continue
        if normalized not in out:
            out.append(normalized)
    return out


def _risk_tokens(row: pd.Series) -> list[str]:
    tokens: list[str] = []
    for column in [
        "main_risk_factors",
        "risk_flags",
        "review_reason_flags",
        "low_confidence_reasons",
    ]:
        if column in row.index:
            tokens.extend(_split_tokens(row.get(column)))
    return list(dict.fromkeys(tokens))


def _numeric_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def _ranked_candidates(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["_rank_sort"] = _numeric_series(work, "consensus_rank", default=float("inf"))
    work["_score_sort"] = _numeric_series(work, "consensus_score", default=-float("inf"))
    return work.sort_values(
        by=["_rank_sort", "_score_sort"],
        ascending=[True, False],
        na_position="last",
    ).drop(columns=["_rank_sort", "_score_sort"], errors="ignore")


def _candidate_base(score_cards: pd.DataFrame | None, consensus: pd.DataFrame | None) -> pd.DataFrame:
    if score_cards is not None and not score_cards.empty:
        base = score_cards.copy()
        if consensus is not None and not consensus.empty and "nanobody_id" in base.columns and "nanobody_id" in consensus.columns:
            missing_cols = [col for col in consensus.columns if col not in base.columns and col != "nanobody_id"]
            if missing_cols:
                base = base.merge(consensus[["nanobody_id"] + missing_cols], on="nanobody_id", how="left")
        return _ranked_candidates(base)
    if consensus is not None and not consensus.empty:
        return _ranked_candidates(consensus.copy())
    return pd.DataFrame()


def _candidate_card(section_key: str, section_label: str, row: pd.Series, source: str) -> dict[str, Any]:
    risks = _risk_tokens(row)
    evidence = (
        _clean_text(row.get("plain_language_summary"))
        or _clean_text(row.get("consensus_explanation"))
        or _clean_text(row.get("suggestion_reason"))
        or _clean_text(row.get("plan_reason"))
        or _clean_text(row.get("score_meaning"))
    )
    action = (
        _clean_text(row.get("recommended_action"))
        or _clean_text(row.get("recommended_next_action"))
        or _clean_text(row.get("plan_decision"))
        or "人工复核后再决定下一步。"
    )
    return {
        "section_key": section_key,
        "section_label": section_label,
        "source": source,
        "nanobody_id": _clean_text(row.get("nanobody_id")),
        "consensus_rank": _safe_int(row.get("consensus_rank"), 0),
        "consensus_score": _safe_float(row.get("consensus_score"), None),
        "confidence_level": _clean_text(row.get("confidence_level")),
        "confidence_score": _safe_float(row.get("confidence_score"), None),
        "decision_tier": _clean_text(row.get("decision_tier")),
        "qc_risk_score": _safe_float(row.get("qc_risk_score"), None),
        "risk_token_count": len(risks),
        "risk_summary": "；".join(risks[:5]),
        "recommended_action": action,
        "evidence": evidence,
    }


def _choose_best_candidate(base: pd.DataFrame) -> dict[str, Any] | None:
    if base.empty:
        return None
    return _candidate_card("best_candidate", "本批次最高综合排名候选", base.iloc[0], "score_or_consensus")


def _choose_most_stable_candidate(base: pd.DataFrame) -> dict[str, Any] | None:
    if base.empty:
        return None
    work = base.copy()
    confidence = _numeric_series(work, "confidence_score", default=0.5)
    score = _numeric_series(work, "consensus_score", default=0.0)
    qc = _numeric_series(work, "qc_risk_score", default=0.0)
    rank_penalty = _numeric_series(work, "consensus_rank", default=len(work) + 1) / max(float(len(work)), 1.0)
    risk_count = work.apply(lambda row: len(_risk_tokens(row)), axis=1).astype(float)
    work["_stability_score"] = (
        0.50 * confidence.clip(0.0, 1.0)
        + 0.30 * score.clip(0.0, 1.0)
        - 0.18 * qc.clip(0.0, 1.0)
        - 0.04 * risk_count.clip(0.0, 8.0)
        - 0.03 * rank_penalty.clip(0.0, 1.5)
    )
    row = work.sort_values("_stability_score", ascending=False, na_position="last").iloc[0]
    card = _candidate_card("most_stable_candidate", "当前证据最稳定候选", row, "score_or_consensus")
    card["stability_score"] = _safe_float(row.get("_stability_score"), None)
    return card


def _choose_highest_risk_candidate(base: pd.DataFrame) -> dict[str, Any] | None:
    if base.empty:
        return None
    work = base.copy()
    confidence = _numeric_series(work, "confidence_score", default=0.5)
    qc = _numeric_series(work, "qc_risk_score", default=0.0)
    risk_count = work.apply(lambda row: len(_risk_tokens(row)), axis=1).astype(float)
    review_bonus = work.get("decision_tier", pd.Series("", index=work.index)).astype(str).str.lower().eq("review").astype(float)
    work["_candidate_risk_score"] = (
        0.45 * qc.clip(0.0, 1.0)
        + 0.25 * (1.0 - confidence.clip(0.0, 1.0))
        + 0.08 * risk_count.clip(0.0, 8.0)
        + 0.12 * review_bonus
    )
    row = work.sort_values("_candidate_risk_score", ascending=False, na_position="last").iloc[0]
    card = _candidate_card("highest_risk_candidate", "最需要先复核的候选", row, "score_or_consensus")
    card["candidate_risk_score"] = _safe_float(row.get("_candidate_risk_score"), None)
    return card


def _choose_next_experiment_candidate(
    experiment_plan: pd.DataFrame | None,
    suggestions: pd.DataFrame | None,
    fallback: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if experiment_plan is not None and not experiment_plan.empty:
        work = experiment_plan.copy()
        if "plan_decision" in work.columns:
            include = work[work["plan_decision"].astype(str).str.lower().eq("include_now")].copy()
            if not include.empty:
                include["_rank"] = _numeric_series(include, "plan_rank", default=float("inf"))
                row = include.sort_values("_rank", ascending=True).iloc[0]
                return _candidate_card("next_experiment_candidate", "下一轮实验优先候选", row, "experiment_plan")
        work["_rank"] = _numeric_series(work, "plan_rank", default=float("inf"))
        row = work.sort_values("_rank", ascending=True).iloc[0]
        return _candidate_card("next_experiment_candidate", "下一轮实验优先候选", row, "experiment_plan")

    if suggestions is not None and not suggestions.empty:
        work = suggestions.copy()
        work["_rank"] = _numeric_series(work, "suggestion_rank", default=float("inf"))
        row = work.sort_values("_rank", ascending=True).iloc[0]
        return _candidate_card("next_experiment_candidate", "下一轮实验优先候选", row, "experiment_suggestions")

    if fallback is not None:
        out = dict(fallback)
        out["section_key"] = "next_experiment_candidate"
        out["section_label"] = "下一轮实验优先候选"
        out["source"] = "best_candidate_fallback"
        return out
    return None


def _top_quality_checks(checks: pd.DataFrame | None, top_n: int) -> list[dict[str, Any]]:
    if checks is None or checks.empty or "status" not in checks.columns:
        return []
    work = checks.copy()
    work["_status_order"] = work["status"].astype(str).str.upper().map(RISK_STATUS_ORDER).fillna(9)
    risky = work[~work["status"].astype(str).str.upper().eq("PASS")].copy()
    if risky.empty:
        return []
    columns = ["check_id", "status", "message", "recommended_action"]
    return (
        risky.sort_values("_status_order")
        .loc[:, [col for col in columns if col in risky.columns]]
        .head(int(top_n))
        .to_dict(orient="records")
    )


def _top_messages(feature_df: pd.DataFrame | None, column: str, top_n: int) -> list[dict[str, Any]]:
    if feature_df is None or feature_df.empty or column not in feature_df.columns:
        return []
    values = feature_df[column].map(_clean_text)
    values = values[values.ne("")]
    if values.empty:
        return []
    counts = values.value_counts(dropna=False).head(int(top_n))
    return [{"message": str(message), "count": int(count)} for message, count in counts.items()]


def _top_risky_input_rows(feature_df: pd.DataFrame | None, top_n: int) -> list[dict[str, Any]]:
    if feature_df is None or feature_df.empty:
        return []
    work = feature_df.copy()
    risk = pd.Series(0.0, index=work.index, dtype=float)
    if "status" in work.columns:
        risk += work["status"].astype(str).str.lower().eq("failed").astype(float) * 10.0
    if "error_message" in work.columns:
        risk += work["error_message"].map(_clean_text).ne("").astype(float) * 5.0
    if "warning_message" in work.columns:
        risk += work["warning_message"].map(_clean_text).ne("").astype(float) * 3.0
    for column in ["pocket_shape_overwide_proxy", "qc_risk_score"]:
        if column in work.columns:
            risk += pd.to_numeric(work[column], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    if float(risk.max()) <= 0.0:
        return []
    work["_row_risk_score"] = risk
    work["_source_row_number"] = work.index.astype(int) + 2
    columns = [
        "_source_row_number",
        "_row_risk_score",
        "nanobody_id",
        "conformer_id",
        "pose_id",
        "status",
        "error_message",
        "warning_message",
        "pocket_shape_overwide_proxy",
    ]
    return (
        work.sort_values("_row_risk_score", ascending=False)
        .loc[:, [col for col in columns if col in work.columns]]
        .head(int(top_n))
        .to_dict(orient="records")
    )


def _validation_evidence_summary(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload:
        return {
            "available": False,
            "audit_status": "UNKNOWN",
            "summary": "未找到验证证据审计结果。",
            "recommended_next_action": "如果要安排实验，先运行 build_validation_evidence_audit.py 或完整推荐 pipeline。",
        }

    audit_status = str(payload.get("audit_status") or "UNKNOWN").upper()
    label_ready_count = _safe_int(payload.get("label_ready_count"), 0)
    candidate_count = _safe_int(payload.get("candidate_count"), 0)
    top_k_count = _safe_int(payload.get("top_k_count"), _safe_int(payload.get("top_k"), 0))
    top_k_validated = _safe_int(payload.get("top_k_validated_count"), 0)
    coverage = _safe_float(payload.get("top_k_validation_coverage"), 0.0) or 0.0
    positive = _safe_int(payload.get("positive_label_count"), 0)
    negative = _safe_int(payload.get("negative_label_count"), 0)
    next_actions = payload.get("next_actions") if isinstance(payload.get("next_actions"), list) else []

    if audit_status == "PASS":
        summary = "当前真实验证证据达到最小可用标准。"
        recommended = "继续按实验 ledger 更新后续结果，并保持正负标签平衡。"
    elif label_ready_count <= 0:
        summary = "当前高排名候选还没有可用真实验证标签。"
        recommended = "实验前优先补 top 候选的 experiment_result 或 validation_label。"
    elif positive <= 0 or negative <= 0:
        summary = "当前真实验证标签只有单一类别，暂不适合做可靠对照或校准。"
        recommended = "补充正负两类样本，避免只用单侧证据解释模型。"
    elif top_k_validated <= 0:
        summary = "已有部分验证标签，但 top 候选仍缺少直接验证。"
        recommended = "优先验证当前 top-k 候选或把已有实验结果回填到 ledger。"
    else:
        summary = "当前有部分真实验证证据，但覆盖率仍偏低。"
        recommended = "继续补齐 top 候选验证，并在下一轮回灌标签后重新运行。"

    return {
        "available": True,
        "audit_status": audit_status,
        "summary": summary,
        "recommended_next_action": recommended,
        "candidate_count": candidate_count,
        "label_ready_count": label_ready_count,
        "positive_label_count": positive,
        "negative_label_count": negative,
        "top_k_count": top_k_count,
        "top_k_validated_count": top_k_validated,
        "top_k_unvalidated_count": _safe_int(payload.get("top_k_unvalidated_count"), max(top_k_count - top_k_validated, 0)),
        "top_k_validation_coverage": coverage,
        "compare_ready": bool(payload.get("compare_ready")),
        "calibration_ready": bool(payload.get("calibration_ready")),
        "next_actions": [_clean_text(item) for item in next_actions if _clean_text(item)][:5],
    }


def _top_validation_actions(action_items: pd.DataFrame | None, top_n: int) -> list[dict[str, Any]]:
    if action_items is None or action_items.empty:
        return []
    work = action_items.copy()
    if "priority" in work.columns:
        work["_priority"] = pd.to_numeric(work["priority"], errors="coerce").fillna(999999)
        work = work.sort_values("_priority", ascending=True, na_position="last")
    columns = ["priority", "action_type", "nanobody_id", "consensus_rank", "reason", "recommended_action"]
    return work.loc[:, [col for col in columns if col in work.columns]].head(int(top_n)).to_dict(orient="records")


def _build_batch_decision(
    quality_summary: dict[str, Any],
    next_candidate: dict[str, Any] | None,
    validation_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    status = str(quality_summary.get("overall_status") or "UNKNOWN").upper()
    quality_decision = _clean_text(quality_summary.get("decision"))
    next_id = _clean_text(next_candidate.get("nanobody_id")) if next_candidate else ""
    evidence = validation_evidence if isinstance(validation_evidence, dict) else {}
    evidence_status = str(evidence.get("audit_status") or "UNKNOWN").upper()
    evidence_action = _clean_text(evidence.get("recommended_next_action"))

    def with_evidence(action: str) -> str:
        if evidence_status in {"WARN", "FAIL"} and evidence_action:
            return f"{action} 验证证据审计提示：{evidence_action}"
        if evidence_status == "PASS":
            return f"{action} 真实验证证据已达到最小可用标准。"
        return action

    if status == "FAIL":
        return {
            "decision_tier": "fix_inputs_first",
            "summary": "当前批次存在阻塞性质量问题，不建议直接解读排名或安排实验。",
            "recommended_next_action": "先处理 Quality Gate 中的 FAIL 项，再重新运行 pipeline。",
            "quality_gate_decision": quality_decision,
        }
    if status == "WARN":
        action = "先复核 Quality Gate 的 WARN 项；若风险可接受，再查看候选排序和实验计划。"
        if next_id:
            action += f" 当前可先关注 `{next_id}`，但不要跳过风险复核。"
        return {
            "decision_tier": "review_before_experiment",
            "summary": "当前批次可用于初步查看，但需要先复核 warning、label 或 pocket 风险。",
            "recommended_next_action": with_evidence(action),
            "quality_gate_decision": quality_decision,
            "validation_evidence_status": evidence_status,
        }
    if status == "PASS":
        action = "可进入候选优先级解读。"
        if next_id:
            action += f" 下一轮优先关注 `{next_id}`。"
        return {
            "decision_tier": "ready_for_candidate_review",
            "summary": "当前批次通过基础质量门控，可以解读排序结果。",
            "recommended_next_action": with_evidence(action),
            "quality_gate_decision": quality_decision,
            "validation_evidence_status": evidence_status,
        }
    return {
        "decision_tier": "unknown_quality_status",
        "summary": "未找到完整 Quality Gate 结论，建议先确认 pipeline 是否完整跑完。",
        "recommended_next_action": with_evidence("先打开 recommended_pipeline_report.md 和 quality_gate_report.md 核对运行状态。"),
        "quality_gate_decision": quality_decision,
        "validation_evidence_status": evidence_status,
    }


def _existing_read_first_files(summary_path: Path | None, summary: dict[str, Any]) -> list[dict[str, str]]:
    order = [
        ("quality_gate_report_md", "先判断本次结果 PASS/WARN/FAIL"),
        ("score_explanation_cards_html", "快速阅读候选分数解释卡片"),
        ("score_explanation_cards_md", "不能打开 HTML 时阅读 Markdown 版分数解释"),
        ("consensus_ranking_csv", "查看完整共识排序和低可信原因"),
        ("candidate_report_index_html", "打开候选报告卡索引"),
        ("candidate_comparison_report_md", "理解相邻候选为什么排序不同"),
        ("experiment_plan_md", "查看下一轮实验计划单"),
        ("validation_evidence_report_md", "检查 top 候选真实验证覆盖和待补实验结果"),
        ("run_provenance_card_md", "复现实验参数、输入、依赖和输出 hash"),
    ]
    out: list[dict[str, str]] = []
    for key, reason in order:
        path = _artifact_path(summary_path, summary, key)
        if path is not None and path.exists():
            out.append({"artifact_key": key, "path": str(path), "why": reason})
    return out


def _clip_text(value: Any, max_chars: int = 220) -> str:
    text = _clean_text(value).replace("\n", " ")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _markdown_table(records: list[dict[str, Any]], columns: list[str], *, max_chars: int = 180) -> str:
    if not records:
        return "_无。_"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for record in records:
        values = []
        for column in columns:
            text = _clip_text(record.get(column, ""), max_chars=max_chars).replace("|", "\\|")
            values.append(text)
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, sep] + rows)


def _build_report(payload: dict[str, Any]) -> str:
    quality = payload.get("quality_gate") if isinstance(payload.get("quality_gate"), dict) else {}
    decision = payload.get("batch_decision") if isinstance(payload.get("batch_decision"), dict) else {}
    highlights = payload.get("candidate_highlights") if isinstance(payload.get("candidate_highlights"), dict) else {}
    cards = [card for card in highlights.values() if isinstance(card, dict)]
    risk_summary = payload.get("risk_summary") if isinstance(payload.get("risk_summary"), dict) else {}
    validation_evidence = (
        payload.get("validation_evidence")
        if isinstance(payload.get("validation_evidence"), dict)
        else {}
    )

    lines = [
        "# 本批次结论摘要",
        "",
        f"- Generated at: `{payload.get('generated_at', '')}`",
        f"- Source summary: `{payload.get('source_summary_json') or ''}`",
        f"- Quality gate: `{quality.get('overall_status', 'UNKNOWN')}`",
        f"- Validation evidence: `{validation_evidence.get('audit_status', 'UNKNOWN')}`",
        f"- Batch decision: {decision.get('summary', '')}",
        f"- Recommended next action: {decision.get('recommended_next_action', '')}",
        "",
        "## 候选结论卡",
        "",
        _markdown_table(
            cards,
            [
                "section_label",
                "nanobody_id",
                "consensus_rank",
                "consensus_score",
                "confidence_level",
                "qc_risk_score",
                "recommended_action",
                "risk_summary",
            ],
        ),
        "",
        "## 真实验证证据",
        "",
        f"- Summary: {validation_evidence.get('summary', '')}",
        f"- Label-ready: `{validation_evidence.get('label_ready_count', 0)}` / `{validation_evidence.get('candidate_count', 0)}`",
        f"- Positive / Negative labels: `{validation_evidence.get('positive_label_count', 0)}` / `{validation_evidence.get('negative_label_count', 0)}`",
        f"- Top-k validated: `{validation_evidence.get('top_k_validated_count', 0)}` / `{validation_evidence.get('top_k_count', 0)}`",
        f"- Top-k coverage: `{(_safe_float(validation_evidence.get('top_k_validation_coverage'), 0.0) or 0.0):.1%}`",
        f"- Compare ready: `{validation_evidence.get('compare_ready', False)}`",
        f"- Calibration ready: `{validation_evidence.get('calibration_ready', False)}`",
        "",
        "### 验证证据行动项",
        "",
        _markdown_table(
            risk_summary.get("validation_action_items", []) if isinstance(risk_summary.get("validation_action_items"), list) else [],
            ["priority", "action_type", "nanobody_id", "consensus_rank", "reason", "recommended_action"],
        ),
        "",
        "## 质量风险 Top 项",
        "",
        "### Quality Gate WARN/FAIL",
        "",
        _markdown_table(
            risk_summary.get("quality_checks", []) if isinstance(risk_summary.get("quality_checks"), list) else [],
            ["check_id", "status", "message", "recommended_action"],
        ),
        "",
        "### Warning Message Top-N",
        "",
        _markdown_table(
            risk_summary.get("warning_messages", []) if isinstance(risk_summary.get("warning_messages"), list) else [],
            ["message", "count"],
        ),
        "",
        "### Error Message Top-N",
        "",
        _markdown_table(
            risk_summary.get("error_messages", []) if isinstance(risk_summary.get("error_messages"), list) else [],
            ["message", "count"],
        ),
        "",
        "### 最高风险输入行",
        "",
        _markdown_table(
            risk_summary.get("risky_input_rows", []) if isinstance(risk_summary.get("risky_input_rows"), list) else [],
            ["_source_row_number", "_row_risk_score", "nanobody_id", "conformer_id", "pose_id", "status", "warning_message", "error_message"],
        ),
        "",
        "## 建议先打开的文件",
        "",
        _markdown_table(
            payload.get("read_first_files", []) if isinstance(payload.get("read_first_files"), list) else [],
            ["artifact_key", "path", "why"],
            max_chars=260,
        ),
        "",
        "## 解释边界",
        "",
        "- 本摘要只整合已有产物，不改变任何训练、Rule、ML 或 consensus 分数。",
        "- 如果 Quality Gate 是 `FAIL`，优先修复输入或 QC 问题，不要直接把排名当作候选结论。",
        "- 如果缺少真实 label，分数只能作为决策支持，不能解读成实验成功概率。",
        "",
    ]
    return "\n".join(lines)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    return value


def build_batch_decision_summary_outputs(
    *,
    summary_json: str | Path | None = None,
    out_dir: str | Path = "batch_decision_summary",
    top_n: int = 5,
    quality_gate_summary_json: str | Path | None = None,
    quality_gate_checks_csv: str | Path | None = None,
    consensus_csv: str | Path | None = None,
    score_cards_csv: str | Path | None = None,
    experiment_suggestions_csv: str | Path | None = None,
    experiment_plan_csv: str | Path | None = None,
    validation_evidence_summary_json: str | Path | None = None,
    validation_evidence_topk_csv: str | Path | None = None,
    validation_evidence_action_items_csv: str | Path | None = None,
    feature_csv: str | Path | None = None,
) -> dict[str, Any]:
    summary_path = _resolve_path(summary_json, base_dir=Path.cwd()) if summary_json else None
    summary = _read_json(summary_path) if summary_path is not None else {}
    if summary_path is None and not any([quality_gate_summary_json, consensus_csv, score_cards_csv, feature_csv]):
        raise ValueError("Provide --summary_json or direct artifact paths.")

    quality_summary_path = _artifact_path(
        summary_path,
        summary,
        "quality_gate_summary_json",
        override=quality_gate_summary_json,
    )
    quality_checks_path = _artifact_path(
        summary_path,
        summary,
        "quality_gate_checks_csv",
        override=quality_gate_checks_csv,
    )
    consensus_path = _artifact_path(summary_path, summary, "consensus_ranking_csv", override=consensus_csv)
    score_cards_path = _artifact_path(
        summary_path,
        summary,
        "score_explanation_cards_csv",
        override=score_cards_csv,
    )
    suggestions_path = _artifact_path(
        summary_path,
        summary,
        "experiment_suggestions_csv",
        override=experiment_suggestions_csv,
    )
    plan_path = _artifact_path(summary_path, summary, "experiment_plan_csv", override=experiment_plan_csv)
    validation_evidence_summary_path = _artifact_path(
        summary_path,
        summary,
        "validation_evidence_summary_json",
        override=validation_evidence_summary_json,
    )
    validation_evidence_topk_path = _artifact_path(
        summary_path,
        summary,
        "validation_evidence_topk_csv",
        override=validation_evidence_topk_csv,
    )
    validation_evidence_action_items_path = _artifact_path(
        summary_path,
        summary,
        "validation_evidence_action_items_csv",
        override=validation_evidence_action_items_csv,
    )

    quality_summary = _read_json(quality_summary_path)
    validation_evidence_payload = _read_json(validation_evidence_summary_path)
    validation_evidence = _validation_evidence_summary(validation_evidence_payload)
    feature_path = _resolve_path(feature_csv, base_dir=Path.cwd()) if feature_csv else None
    if feature_path is None:
        feature_path = _resolve_path(summary.get("feature_csv"), base_dir=summary_path.parent if summary_path else Path.cwd())
    if feature_path is None and quality_summary:
        feature_path = _resolve_path(quality_summary.get("feature_csv"), base_dir=Path.cwd())

    quality_checks = _read_csv(quality_checks_path)
    consensus = _read_csv(consensus_path)
    score_cards = _read_csv(score_cards_path)
    suggestions = _read_csv(suggestions_path)
    plan = _read_csv(plan_path)
    validation_action_items = _read_csv(validation_evidence_action_items_path)
    feature_df = _read_csv(feature_path)

    base = _candidate_base(score_cards, consensus)
    best = _choose_best_candidate(base)
    stable = _choose_most_stable_candidate(base)
    risky = _choose_highest_risk_candidate(base)
    next_experiment = _choose_next_experiment_candidate(plan, suggestions, best)
    batch_decision = _build_batch_decision(quality_summary, next_experiment, validation_evidence)

    highlights = {
        "best_candidate": best,
        "most_stable_candidate": stable,
        "highest_risk_candidate": risky,
        "next_experiment_candidate": next_experiment,
    }
    card_rows = [card for card in highlights.values() if isinstance(card, dict)]

    output_dir = Path(out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_out = output_dir / "batch_decision_summary.json"
    md_out = output_dir / "batch_decision_summary.md"
    cards_out = output_dir / "batch_decision_summary_cards.csv"

    risk_summary = {
        "quality_checks": _top_quality_checks(quality_checks, int(top_n)),
        "warning_messages": _top_messages(feature_df, "warning_message", int(top_n)),
        "error_messages": _top_messages(feature_df, "error_message", int(top_n)),
        "risky_input_rows": _top_risky_input_rows(feature_df, int(top_n)),
        "validation_action_items": _top_validation_actions(validation_action_items, int(top_n)),
    }

    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_summary_json": str(summary_path) if summary_path is not None else None,
        "artifact_inputs": {
            "quality_gate_summary_json": str(quality_summary_path) if quality_summary_path else None,
            "quality_gate_checks_csv": str(quality_checks_path) if quality_checks_path else None,
            "consensus_ranking_csv": str(consensus_path) if consensus_path else None,
            "score_explanation_cards_csv": str(score_cards_path) if score_cards_path else None,
            "experiment_suggestions_csv": str(suggestions_path) if suggestions_path else None,
            "experiment_plan_csv": str(plan_path) if plan_path else None,
            "validation_evidence_summary_json": str(validation_evidence_summary_path) if validation_evidence_summary_path else None,
            "validation_evidence_topk_csv": str(validation_evidence_topk_path) if validation_evidence_topk_path else None,
            "validation_evidence_action_items_csv": str(validation_evidence_action_items_path) if validation_evidence_action_items_path else None,
            "feature_csv": str(feature_path) if feature_path else None,
        },
        "quality_gate": {
            "overall_status": str(quality_summary.get("overall_status") or "UNKNOWN").upper(),
            "decision": quality_summary.get("decision"),
            "pass_count": quality_summary.get("pass_count"),
            "warn_count": quality_summary.get("warn_count"),
            "fail_count": quality_summary.get("fail_count"),
            "total_rows": quality_summary.get("total_rows"),
            "failed_rows": quality_summary.get("failed_rows"),
            "warning_rows": quality_summary.get("warning_rows"),
            "label": quality_summary.get("label"),
        },
        "batch_decision": batch_decision,
        "validation_evidence": validation_evidence,
        "candidate_count": int(len(base)),
        "candidate_highlights": highlights,
        "risk_summary": risk_summary,
        "read_first_files": _existing_read_first_files(summary_path, summary),
        "outputs": {
            "batch_decision_summary_json": str(summary_out),
            "batch_decision_summary_md": str(md_out),
            "batch_decision_summary_cards_csv": str(cards_out),
        },
    }

    pd.DataFrame(card_rows).to_csv(cards_out, index=False, encoding="utf-8")
    md_out.write_text(_build_report(_json_safe(payload)), encoding="utf-8")
    summary_out.write_text(json.dumps(_json_safe(payload), ensure_ascii=True, indent=2), encoding="utf-8")
    return _json_safe(payload)


def main() -> None:
    args = _build_parser().parse_args()
    payload = build_batch_decision_summary_outputs(
        summary_json=args.summary_json,
        out_dir=args.out_dir,
        top_n=int(args.top_n),
        quality_gate_summary_json=args.quality_gate_summary_json,
        quality_gate_checks_csv=args.quality_gate_checks_csv,
        consensus_csv=args.consensus_csv,
        score_cards_csv=args.score_cards_csv,
        experiment_suggestions_csv=args.experiment_suggestions_csv,
        experiment_plan_csv=args.experiment_plan_csv,
        validation_evidence_summary_json=args.validation_evidence_summary_json,
        validation_evidence_topk_csv=args.validation_evidence_topk_csv,
        validation_evidence_action_items_csv=args.validation_evidence_action_items_csv,
        feature_csv=args.feature_csv,
    )
    outputs = payload.get("outputs", {}) if isinstance(payload.get("outputs"), dict) else {}
    for key, value in outputs.items():
        print(f"Saved {key}: {value}")
    decision = payload.get("batch_decision", {}) if isinstance(payload.get("batch_decision"), dict) else {}
    print(f"Batch decision: {decision.get('decision_tier', 'unknown')}")


if __name__ == "__main__":
    main()
