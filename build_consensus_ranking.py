"""Build a conservative Rule + ML consensus nanobody ranking.

This script does not replace the existing ML or rule rankers. It merges their
nanobody-level outputs into a third report that highlights agreement, QC risk
and confidence. The goal is decision support, not a new trained model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ranking_common import compute_pocket_overwide_penalty


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Rule + ML consensus nanobody ranking")
    parser.add_argument("--rule_csv", required=True, help="Path to nanobody_rule_ranking.csv")
    parser.add_argument("--ml_csv", required=True, help="Path to nanobody_ranking.csv")
    parser.add_argument("--out_dir", default="consensus_outputs", help="Output directory")
    parser.add_argument("--feature_csv", default=None, help="Optional pose_features.csv for QC aggregation")
    parser.add_argument("--score_delta_scale", type=float, default=0.50, help="Score gap mapped to full disagreement")
    parser.add_argument("--overwide_threshold", type=float, default=0.55, help="Pocket overwide warning threshold")
    parser.add_argument("--close_score_delta", type=float, default=0.03, help="Consensus score gap treated as a near tie")
    parser.add_argument(
        "--conformer_std_threshold",
        type=float,
        default=0.15,
        help="Conformer score std threshold for instability warnings",
    )
    return parser


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _detect_score_column(df: pd.DataFrame, preferred: list[str]) -> str:
    for col in preferred:
        if col in df.columns:
            return col
    raise ValueError(f"No score column found. Tried: {preferred}")


def _first_existing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def _prepare_ranking_view(df: pd.DataFrame, *, prefix: str, score_col: str) -> pd.DataFrame:
    if "nanobody_id" not in df.columns:
        raise ValueError(f"{prefix} ranking is missing nanobody_id column.")
    work = df.copy()
    work[f"{prefix}_score"] = _to_numeric(work[score_col])
    if "rank" in work.columns:
        work[f"{prefix}_rank"] = _to_numeric(work["rank"])
    else:
        work[f"{prefix}_rank"] = work[f"{prefix}_score"].rank(method="min", ascending=False)

    useful_cols = [
        "nanobody_id",
        f"{prefix}_rank",
        f"{prefix}_score",
    ]
    extra_cols = _first_existing_columns(
        work,
        [
            "best_conformer",
            "best_pose_id",
            "best_pose_prob",
            "best_pose_rule_score",
            "mean_conformer_rule_score",
            "best_conformer_rule_score",
            "std_conformer_rule_score",
            "mean_conformer_score",
            "best_conformer_score",
            "std_conformer_score",
            "pocket_consistency_score",
            "mean_topk_pocket_hit_fraction",
            "mean_topk_catalytic_hit_fraction",
            "mean_topk_mouth_occlusion_score",
            "mean_topk_ligand_path_block_score",
            "mean_topk_pocket_shape_overwide_proxy",
            "pocket_overwide_penalty",
            "score_weight_pocket_overwide_penalty",
            "explanation",
        ],
    )
    rename_map = {col: f"{prefix}_{col}" for col in extra_cols}
    useful_cols.extend(extra_cols)
    return work.loc[:, useful_cols].rename(columns=rename_map)


def _aggregate_feature_qc(feature_csv: str | Path | None) -> pd.DataFrame:
    if not feature_csv:
        return pd.DataFrame()
    path = Path(feature_csv).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"feature_csv not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    if "nanobody_id" not in df.columns:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for nanobody_id, group in df.groupby("nanobody_id", dropna=False):
        row: dict[str, Any] = {
            "nanobody_id": str(nanobody_id),
            "qc_pose_row_count": int(len(group)),
        }
        if "status" in group.columns:
            status = group["status"].fillna("").astype(str).str.lower()
            row["qc_failed_pose_count"] = int(status.eq("failed").sum())
            row["qc_failed_pose_fraction"] = float(status.eq("failed").mean()) if len(status) > 0 else float("nan")
        else:
            row["qc_failed_pose_count"] = 0
            row["qc_failed_pose_fraction"] = 0.0

        if "warning_message" in group.columns:
            warning = group["warning_message"].fillna("").astype(str).str.strip().ne("")
            row["qc_warning_pose_count"] = int(warning.sum())
            row["qc_warning_pose_fraction"] = float(warning.mean()) if len(warning) > 0 else float("nan")
        else:
            row["qc_warning_pose_count"] = 0
            row["qc_warning_pose_fraction"] = 0.0

        if "pocket_shape_overwide_proxy" in group.columns:
            overwide = _to_numeric(group["pocket_shape_overwide_proxy"])
            finite = overwide[np.isfinite(overwide.to_numpy(dtype=np.float64))]
            row["qc_mean_pocket_shape_overwide_proxy"] = float(finite.mean()) if not finite.empty else float("nan")
            row["qc_max_pocket_shape_overwide_proxy"] = float(finite.max()) if not finite.empty else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _nanmean(values: list[Any], default: float = 0.0) -> float:
    arr = np.asarray([_safe_float(v) for v in values], dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float(default)
    return float(np.mean(finite))


def _build_risk_flags(row: pd.Series) -> str:
    flags: list[str] = []
    if bool(row.get("rule_ml_disagreement", False)):
        flags.append("rule_ml_disagreement")
    if bool(row.get("rule_ml_score_gap_warning", False)):
        flags.append("rule_ml_score_gap")
    if bool(row.get("pocket_overwide_warning", False)):
        flags.append("pocket_overwide")
    if bool(row.get("conformer_instability_warning", False)):
        flags.append("conformer_instability")
    if bool(row.get("close_score_competition_warning", False)):
        flags.append("close_score_competition")
    if _safe_float(row.get("qc_failed_pose_fraction"), 0.0) > 0:
        flags.append("failed_pose_rows")
    if _safe_float(row.get("qc_warning_pose_fraction"), 0.0) >= 0.25:
        flags.append("many_warning_rows")
    if _safe_float(row.get("source_coverage_score"), 0.0) < 1.0:
        flags.append("missing_one_ranker")
    return ";".join(flags) if flags else "none"


def _confidence_level(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.50:
        return "medium"
    return "low"


def _bool_text(value: bool) -> str:
    return "yes" if bool(value) else "no"


def _conformer_instability_score(row: pd.Series) -> float:
    values = [
        _safe_float(row.get("ml_std_conformer_score")),
        _safe_float(row.get("rule_std_conformer_rule_score")),
    ]
    finite = [value for value in values if np.isfinite(value)]
    return float(max(finite)) if finite else 0.0


def _build_review_reason_payload(row: pd.Series) -> tuple[str, str]:
    reasons: list[str] = []
    flags: list[str] = []

    if _safe_float(row.get("source_coverage_score"), 1.0) < 1.0:
        flags.append("missing_one_ranker")
        reasons.append("只被一条排序路线覆盖")
    if bool(row.get("rule_ml_disagreement", False)):
        flags.append("rule_ml_rank_disagreement")
        reasons.append(f"Rule/ML 排名差较大 ({_safe_float(row.get('abs_rank_delta'), 0.0):.0f})")
    if bool(row.get("rule_ml_score_gap_warning", False)):
        flags.append("rule_ml_score_gap")
        reasons.append(f"Rule/ML 分数差较大 ({_safe_float(row.get('ml_rule_score_gap'), 0.0):.3f})")
    if bool(row.get("low_rank_agreement_warning", False)):
        flags.append("low_rank_agreement")
        reasons.append(f"排名一致性偏低 ({_safe_float(row.get('rank_agreement_score'), 0.0):.3f})")
    if bool(row.get("low_score_alignment_warning", False)):
        flags.append("low_score_alignment")
        reasons.append(f"分数一致性偏低 ({_safe_float(row.get('score_alignment_score'), 0.0):.3f})")
    if bool(row.get("pocket_overwide_warning", False)):
        flags.append("pocket_overwide")
        reasons.append(f"pocket 可能偏宽 ({_safe_float(row.get('pocket_overwide_proxy_for_consensus'), 0.0):.3f})")
    if bool(row.get("conformer_instability_warning", False)):
        flags.append("conformer_instability")
        reasons.append(f"构象间分数波动偏大 ({_safe_float(row.get('conformer_instability_score'), 0.0):.3f})")
    if _safe_float(row.get("qc_failed_pose_fraction"), 0.0) > 0:
        flags.append("failed_pose_rows")
        reasons.append(f"存在失败 pose 行 ({_safe_float(row.get('qc_failed_pose_fraction'), 0.0):.1%})")
    if _safe_float(row.get("qc_warning_pose_fraction"), 0.0) >= 0.25:
        flags.append("many_warning_rows")
        reasons.append(f"warning pose 比例偏高 ({_safe_float(row.get('qc_warning_pose_fraction'), 0.0):.1%})")
    if _safe_float(row.get("qc_risk_score"), 0.0) >= 0.40:
        flags.append("high_qc_risk")
        reasons.append(f"综合 QC 风险偏高 ({_safe_float(row.get('qc_risk_score'), 0.0):.3f})")
    if bool(row.get("close_score_competition_warning", False)):
        flags.append("close_score_competition")
        reasons.append(f"与相邻候选分差很小 ({_safe_float(row.get('nearest_competitor_score_gap'), 0.0):.3f})")

    if not reasons:
        reasons.append("未发现明显低可信原因")
    if not flags:
        flags.append("none")
    return ";".join(dict.fromkeys(flags)), "；".join(dict.fromkeys(reasons))


def _decision_tier(row: pd.Series) -> str:
    consensus = _safe_float(row.get("consensus_score"), 0.0)
    confidence = _safe_float(row.get("confidence_score"), 0.0)
    qc_risk = _safe_float(row.get("qc_risk_score"), 0.0)
    if consensus >= 0.75 and confidence >= 0.65 and qc_risk < 0.40:
        return "priority"
    if consensus >= 0.65 or confidence < 0.50 or qc_risk >= 0.40:
        return "review"
    return "standard"


def _explain_row(row: pd.Series) -> str:
    parts: list[str] = []
    level = str(row.get("confidence_level") or "medium")
    tier = str(row.get("decision_tier") or "standard")
    if tier == "priority":
        parts.append("Rule 与 ML 共识较好，可作为优先候选复核")
    elif tier == "review":
        parts.append("建议人工复核后再决策")
    else:
        parts.append("可作为常规候选参考")

    if level == "high":
        parts.append("可信度高")
    elif level == "medium":
        parts.append("可信度中等")
    else:
        parts.append("可信度低")

    if bool(row.get("rule_ml_disagreement", False)):
        parts.append(f"Rule/ML 排名差 {int(_safe_float(row.get('abs_rank_delta'), 0.0))}")
    else:
        parts.append("Rule/ML 排名较一致")

    if bool(row.get("pocket_overwide_warning", False)):
        parts.append("pocket 可能偏宽")
    if _safe_float(row.get("qc_failed_pose_fraction"), 0.0) > 0:
        parts.append("存在失败 pose 行")
    reason_text = str(row.get("low_confidence_reasons") or "").strip()
    if reason_text and reason_text != "未发现明显低可信原因":
        parts.append(f"主要复核原因：{reason_text}")
    return "；".join(parts) + "。"


def _add_close_competition_fields(consensus: pd.DataFrame, close_score_delta: float) -> pd.DataFrame:
    if consensus.empty or "consensus_score" not in consensus.columns:
        return consensus
    work = consensus.copy()
    scores = _to_numeric(work["consensus_score"]).to_numpy(dtype=np.float64)
    previous_gaps: list[float] = []
    next_gaps: list[float] = []
    nearest_gaps: list[float] = []
    near_tie_counts: list[int] = []
    warnings: list[bool] = []
    threshold = max(float(close_score_delta), 0.0)
    for idx, score in enumerate(scores):
        prev_gap = float("nan")
        next_gap = float("nan")
        if idx > 0 and np.isfinite(score) and np.isfinite(scores[idx - 1]):
            prev_gap = float(scores[idx - 1] - score)
        if idx + 1 < len(scores) and np.isfinite(score) and np.isfinite(scores[idx + 1]):
            next_gap = float(score - scores[idx + 1])
        finite_gaps = [gap for gap in [prev_gap, next_gap] if np.isfinite(gap)]
        nearest = float(min(finite_gaps)) if finite_gaps else float("nan")
        if np.isfinite(score):
            near_count = int(np.sum(np.isfinite(scores) & (np.abs(scores - score) <= threshold))) - 1
        else:
            near_count = 0
        previous_gaps.append(prev_gap)
        next_gaps.append(next_gap)
        nearest_gaps.append(nearest)
        near_tie_counts.append(max(0, near_count))
        warnings.append(bool((np.isfinite(nearest) and nearest <= threshold) or near_count > 0))
    work["previous_consensus_score_gap"] = previous_gaps
    work["next_consensus_score_gap"] = next_gaps
    work["nearest_competitor_score_gap"] = nearest_gaps
    work["near_tie_candidate_count"] = near_tie_counts
    work["close_score_competition_warning"] = warnings
    return work


def _finalize_review_fields(consensus: pd.DataFrame) -> pd.DataFrame:
    work = consensus.copy()
    flags: list[str] = []
    reasons: list[str] = []
    for _, row in work.iterrows():
        flag_text, reason_text = _build_review_reason_payload(row)
        flags.append(flag_text)
        reasons.append(reason_text)
    work["review_reason_flags"] = flags
    work["low_confidence_reasons"] = reasons
    work["risk_flags"] = work.apply(_build_risk_flags, axis=1)
    work["decision_tier"] = work.apply(_decision_tier, axis=1)
    work["consensus_explanation"] = work.apply(_explain_row, axis=1)
    return work


def build_consensus_ranking(
    rule_df: pd.DataFrame,
    ml_df: pd.DataFrame,
    *,
    feature_qc_df: pd.DataFrame | None = None,
    score_delta_scale: float = 0.50,
    overwide_threshold: float = 0.55,
    close_score_delta: float = 0.03,
    conformer_std_threshold: float = 0.15,
) -> pd.DataFrame:
    rule_score_col = _detect_score_column(rule_df, ["final_rule_score", "final_score", "best_conformer_score"])
    ml_score_col = _detect_score_column(ml_df, ["final_score", "best_conformer_score"])

    rule_view = _prepare_ranking_view(rule_df, prefix="rule", score_col=rule_score_col)
    ml_view = _prepare_ranking_view(ml_df, prefix="ml", score_col=ml_score_col)
    merged = ml_view.merge(rule_view, on="nanobody_id", how="outer", indicator=True)
    merged["source_presence"] = merged["_merge"].astype(str)
    merged = merged.drop(columns=["_merge"])
    if merged.empty:
        raise ValueError("No nanobody rows found in rule or ML ranking files.")

    if feature_qc_df is not None and not feature_qc_df.empty:
        merged = merged.merge(feature_qc_df, on="nanobody_id", how="left")

    n = max(int(len(merged)), 1)
    delta_scale = max(float(score_delta_scale), 1e-6)

    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        out = row.to_dict()
        ml_score = _safe_float(row.get("ml_score"))
        rule_score = _safe_float(row.get("rule_score"))
        ml_rank = _safe_float(row.get("ml_rank"))
        rule_rank = _safe_float(row.get("rule_rank"))
        both_present = np.isfinite(ml_score) and np.isfinite(rule_score)

        score_mean = _nanmean([ml_score, rule_score], default=0.0)
        score_gap = abs(ml_score - rule_score) if both_present else float("nan")
        score_alignment = 1.0 - float(np.clip(score_gap / delta_scale, 0.0, 1.0)) if np.isfinite(score_gap) else 0.50

        if np.isfinite(ml_rank) and np.isfinite(rule_rank):
            abs_rank_delta = abs(rule_rank - ml_rank)
            rank_agreement = 1.0 - float(np.clip(abs_rank_delta / max(n - 1, 1), 0.0, 1.0))
        else:
            abs_rank_delta = float("nan")
            rank_agreement = 0.50

        overwide_values = [
            row.get("ml_mean_topk_pocket_shape_overwide_proxy"),
            row.get("rule_mean_topk_pocket_shape_overwide_proxy"),
            row.get("qc_mean_pocket_shape_overwide_proxy"),
            row.get("qc_max_pocket_shape_overwide_proxy"),
        ]
        overwide_proxy = _nanmean(overwide_values, default=0.0)
        overwide_penalty = compute_pocket_overwide_penalty(overwide_proxy, threshold=float(overwide_threshold))
        failed_fraction = _safe_float(row.get("qc_failed_pose_fraction"), 0.0)
        warning_fraction = _safe_float(row.get("qc_warning_pose_fraction"), 0.0)
        qc_risk = float(np.clip(max(overwide_penalty, failed_fraction, 0.5 * warning_fraction), 0.0, 1.0))
        conformer_instability = _conformer_instability_score(row)

        source_coverage = 1.0 if both_present else 0.50
        confidence_rank_agreement_component = 0.45 * rank_agreement
        confidence_score_alignment_component = 0.25 * score_alignment
        confidence_source_coverage_component = 0.20 * source_coverage
        confidence_qc_component = 0.10 * (1.0 - qc_risk)
        confidence = float(
            np.clip(
                confidence_rank_agreement_component
                + confidence_score_alignment_component
                + confidence_source_coverage_component
                + confidence_qc_component,
                0.0,
                1.0,
            )
        )
        consensus_score = float(
            np.clip(
                0.70 * score_mean
                + 0.20 * rank_agreement
                + 0.10 * score_alignment
                - 0.15 * qc_risk,
                0.0,
                1.0,
            )
        )

        disagreement_threshold = max(2.0, np.ceil(0.20 * n))
        out.update(
            {
                "ml_rule_score_mean": score_mean,
                "ml_rule_score_gap": score_gap,
                "score_alignment_score": score_alignment,
                "rank_delta": rule_rank - ml_rank if np.isfinite(ml_rank) and np.isfinite(rule_rank) else float("nan"),
                "abs_rank_delta": abs_rank_delta,
                "rank_agreement_score": rank_agreement,
                "source_coverage_score": source_coverage,
                "pocket_overwide_proxy_for_consensus": overwide_proxy,
                "pocket_overwide_warning": bool(overwide_proxy >= float(overwide_threshold)),
                "conformer_instability_score": conformer_instability,
                "conformer_instability_warning": bool(conformer_instability >= float(conformer_std_threshold)),
                "rule_ml_score_gap_warning": bool(np.isfinite(score_gap) and score_gap >= 0.5 * delta_scale),
                "low_rank_agreement_warning": bool(rank_agreement < 0.70),
                "low_score_alignment_warning": bool(score_alignment < 0.70),
                "confidence_rank_agreement_component": confidence_rank_agreement_component,
                "confidence_score_alignment_component": confidence_score_alignment_component,
                "confidence_source_coverage_component": confidence_source_coverage_component,
                "confidence_qc_component": confidence_qc_component,
                "qc_risk_score": qc_risk,
                "consensus_score": consensus_score,
                "confidence_score": confidence,
                "confidence_level": _confidence_level(confidence),
                "rule_ml_disagreement": bool(np.isfinite(abs_rank_delta) and abs_rank_delta >= disagreement_threshold),
            }
        )
        rows.append(out)

    consensus = pd.DataFrame(rows)
    consensus = consensus.sort_values(
        by=["consensus_score", "confidence_score", "ml_rule_score_mean"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    consensus.insert(0, "consensus_rank", np.arange(1, len(consensus) + 1, dtype=int))
    consensus = _add_close_competition_fields(consensus, close_score_delta=float(close_score_delta))
    consensus = _finalize_review_fields(consensus)

    preferred = [
        "consensus_rank",
        "nanobody_id",
        "decision_tier",
        "confidence_level",
        "consensus_score",
        "confidence_score",
        "ml_score",
        "rule_score",
        "ml_rank",
        "rule_rank",
        "abs_rank_delta",
        "rank_agreement_score",
        "score_alignment_score",
        "source_presence",
        "qc_risk_score",
        "review_reason_flags",
        "low_confidence_reasons",
        "risk_flags",
        "consensus_explanation",
    ]
    remaining = [col for col in consensus.columns if col not in preferred]
    return consensus[preferred + remaining]


def _build_summary(consensus: pd.DataFrame, *, rule_csv: Path, ml_csv: Path, feature_csv: Path | None) -> dict[str, Any]:
    tier_counts = consensus["decision_tier"].astype(str).value_counts(dropna=False).to_dict()
    confidence_counts = consensus["confidence_level"].astype(str).value_counts(dropna=False).to_dict()
    return {
        "rule_csv": str(rule_csv),
        "ml_csv": str(ml_csv),
        "feature_csv": None if feature_csv is None else str(feature_csv),
        "nanobody_count": int(len(consensus)),
        "decision_tier_counts": {str(k): int(v) for k, v in tier_counts.items()},
        "confidence_level_counts": {str(k): int(v) for k, v in confidence_counts.items()},
        "mean_consensus_score": float(_to_numeric(consensus["consensus_score"]).mean()),
        "mean_confidence_score": float(_to_numeric(consensus["confidence_score"]).mean()),
        "high_qc_risk_count": int((_to_numeric(consensus["qc_risk_score"]) >= 0.40).sum()),
        "rule_ml_disagreement_count": int(consensus["rule_ml_disagreement"].fillna(False).astype(bool).sum()),
        "close_score_competition_count": int(consensus["close_score_competition_warning"].fillna(False).astype(bool).sum())
        if "close_score_competition_warning" in consensus.columns
        else 0,
        "conformer_instability_count": int(consensus["conformer_instability_warning"].fillna(False).astype(bool).sum())
        if "conformer_instability_warning" in consensus.columns
        else 0,
        "review_reason_flag_counts": {
            str(flag): int(count)
            for flag, count in pd.Series(
                [
                    item
                    for text in consensus.get("review_reason_flags", pd.Series(dtype=object)).fillna("").astype(str)
                    for item in text.split(";")
                    if item and item != "none"
                ]
            ).value_counts(dropna=False).to_dict().items()
        },
        "top_consensus": consensus.head(10).replace({np.nan: None}).to_dict(orient="records"),
        "formula": {
            "consensus_score": "0.70*mean(ML,Rule) + 0.20*rank_agreement + 0.10*score_alignment - 0.15*qc_risk",
            "confidence_score": "0.45*rank_agreement + 0.25*score_alignment + 0.20*source_coverage + 0.10*(1-qc_risk)",
        },
    }


def _build_report(consensus: pd.DataFrame, summary: dict[str, Any]) -> str:
    lines = [
        "# Rule + ML Consensus Ranking",
        "",
        f"- Nanobodies: `{summary['nanobody_count']}`",
        f"- Mean consensus score: `{summary['mean_consensus_score']:.4f}`",
        f"- Mean confidence score: `{summary['mean_confidence_score']:.4f}`",
        f"- Rule/ML disagreement count: `{summary['rule_ml_disagreement_count']}`",
        f"- High QC risk count: `{summary['high_qc_risk_count']}`",
        f"- Close-score competition count: `{summary.get('close_score_competition_count', 0)}`",
        f"- Conformer instability count: `{summary.get('conformer_instability_count', 0)}`",
        "",
        "## Interpretation",
        "",
        "- `priority`: Rule and ML agree well, score is high, and QC risk is limited.",
        "- `review`: high potential or notable disagreement/QC risk; inspect before deciding.",
        "- `standard`: usable as regular ranking reference, but not top priority.",
        "",
        "## Top Consensus Candidates",
        "",
        "| consensus_rank | nanobody_id | tier | confidence | consensus_score | ml_score | rule_score | review_reasons |",
        "|---:|---|---|---|---:|---:|---:|---|",
    ]
    for _, row in consensus.head(20).iterrows():
        lines.append(
            "| "
            f"{int(row['consensus_rank'])} | "
            f"{row['nanobody_id']} | "
            f"{row['decision_tier']} | "
            f"{row['confidence_level']} | "
            f"{_safe_float(row.get('consensus_score'), 0.0):.4f} | "
            f"{_safe_float(row.get('ml_score'), 0.0):.4f} | "
            f"{_safe_float(row.get('rule_score'), 0.0):.4f} | "
            f"{row.get('low_confidence_reasons', '')} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = _build_parser().parse_args()
    rule_csv = Path(args.rule_csv).expanduser().resolve()
    ml_csv = Path(args.ml_csv).expanduser().resolve()
    if not rule_csv.exists():
        raise FileNotFoundError(f"rule_csv not found: {rule_csv}")
    if not ml_csv.exists():
        raise FileNotFoundError(f"ml_csv not found: {ml_csv}")

    feature_csv = Path(args.feature_csv).expanduser().resolve() if args.feature_csv else None
    rule_df = pd.read_csv(rule_csv, low_memory=False)
    ml_df = pd.read_csv(ml_csv, low_memory=False)
    feature_qc_df = _aggregate_feature_qc(feature_csv) if feature_csv is not None else pd.DataFrame()

    consensus = build_consensus_ranking(
        rule_df,
        ml_df,
        feature_qc_df=feature_qc_df,
        score_delta_scale=float(args.score_delta_scale),
        overwide_threshold=float(args.overwide_threshold),
        close_score_delta=float(args.close_score_delta),
        conformer_std_threshold=float(args.conformer_std_threshold),
    )
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    consensus_csv = out_dir / "consensus_ranking.csv"
    summary_json = out_dir / "consensus_summary.json"
    report_md = out_dir / "consensus_report.md"

    consensus.to_csv(consensus_csv, index=False)
    summary = _build_summary(consensus, rule_csv=rule_csv, ml_csv=ml_csv, feature_csv=feature_csv)
    summary_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    report_md.write_text(_build_report(consensus, summary), encoding="utf-8")

    print(f"Saved: {consensus_csv}")
    print(f"Saved: {summary_json}")
    print(f"Saved: {report_md}")


if __name__ == "__main__":
    main()
