"""Build pairwise candidate trade-off explanations from consensus ranking.

This post-processing step is intentionally lightweight: it does not retrain a
model and does not change any ranking score. It turns the existing consensus
ranking into reviewer-facing comparisons such as "why candidate A ranks ahead
of candidate B".
"""

from __future__ import annotations

import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


METRIC_SPECS: list[dict[str, str]] = [
    {"column": "consensus_score", "label": "共识综合分", "direction": "higher"},
    {"column": "confidence_score", "label": "可信度分", "direction": "higher"},
    {"column": "ml_score", "label": "ML 分数", "direction": "higher"},
    {"column": "rule_score", "label": "Rule 分数", "direction": "higher"},
    {"column": "rank_agreement_score", "label": "Rule/ML 排名一致性", "direction": "higher"},
    {"column": "score_alignment_score", "label": "Rule/ML 分数一致性", "direction": "higher"},
    {"column": "source_coverage_score", "label": "排序来源覆盖度", "direction": "higher"},
    {"column": "ml_mean_topk_pocket_hit_fraction", "label": "ML top-k pocket 覆盖", "direction": "higher"},
    {"column": "rule_mean_topk_pocket_hit_fraction", "label": "Rule top-k pocket 覆盖", "direction": "higher"},
    {"column": "ml_mean_topk_catalytic_hit_fraction", "label": "ML top-k catalytic 覆盖", "direction": "higher"},
    {"column": "rule_mean_topk_catalytic_hit_fraction", "label": "Rule top-k catalytic 覆盖", "direction": "higher"},
    {"column": "ml_mean_topk_mouth_occlusion_score", "label": "ML top-k 口部遮挡", "direction": "higher"},
    {"column": "rule_mean_topk_mouth_occlusion_score", "label": "Rule top-k 口部遮挡", "direction": "higher"},
    {"column": "ml_mean_topk_ligand_path_block_score", "label": "ML top-k 路径阻断", "direction": "higher"},
    {"column": "rule_mean_topk_ligand_path_block_score", "label": "Rule top-k 路径阻断", "direction": "higher"},
    {"column": "qc_risk_score", "label": "QC 风险", "direction": "lower"},
    {"column": "abs_rank_delta", "label": "Rule/ML 排名差", "direction": "lower"},
    {"column": "ml_rule_score_gap", "label": "Rule/ML 分数差", "direction": "lower"},
    {"column": "pocket_overwide_proxy_for_consensus", "label": "pocket 偏宽风险", "direction": "lower"},
    {"column": "ml_mean_topk_pocket_shape_overwide_proxy", "label": "ML top-k pocket 偏宽", "direction": "lower"},
    {"column": "rule_mean_topk_pocket_shape_overwide_proxy", "label": "Rule top-k pocket 偏宽", "direction": "lower"},
]


TRADEOFF_COLUMNS = [
    "consensus_rank",
    "nanobody_id",
    "decision_tier",
    "confidence_level",
    "consensus_score",
    "confidence_score",
    "ml_score",
    "rule_score",
    "rank_agreement_score",
    "qc_risk_score",
    "risk_flags",
]

GROUP_COL_CANDIDATES = [
    "diversity_group",
    "sequence_cluster",
    "nanobody_family",
    "epitope_cluster",
    "experiment_status",
    "decision_tier",
    "confidence_level",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build pairwise candidate comparison explanations.")
    parser.add_argument("--consensus_csv", required=True, help="Path to consensus_ranking.csv")
    parser.add_argument("--out_dir", default="candidate_comparisons", help="Output directory")
    parser.add_argument("--top_n", type=int, default=12, help="Only compare the top N consensus candidates")
    parser.add_argument(
        "--selected_nanobody_ids",
        default=None,
        help="Optional comma-separated nanobody IDs for custom comparison; uses consensus-rank order",
    )
    parser.add_argument(
        "--pair_mode",
        choices=["adjacent", "all", "top_vs_rest"],
        default="adjacent",
        help="Which pairwise comparisons to generate",
    )
    parser.add_argument(
        "--close_score_delta",
        type=float,
        default=0.03,
        help="Consensus score gap below this value is flagged as a close decision",
    )
    parser.add_argument(
        "--min_metric_delta",
        type=float,
        default=0.02,
        help="Minimum raw metric delta before a metric is used in text explanations",
    )
    parser.add_argument(
        "--group_col",
        default="auto",
        help=(
            "Column used for group-level candidate comparison summaries. "
            "Use 'auto' to pick diversity/family/status columns when available, or 'none' to disable."
        ),
    )
    return parser


def parse_selected_nanobody_ids(value: str | None) -> list[str]:
    if value is None:
        return []
    seen: set[str] = set()
    ids: list[str] = []
    for item in str(value).replace("\n", ",").replace(";", ",").split(","):
        text = item.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ids.append(text)
    return ids


def _to_numeric(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _fmt(value: Any, digits: int = 3) -> str:
    out = _to_numeric(value)
    if not np.isfinite(out):
        return ""
    if abs(out) >= 100:
        return f"{out:.1f}"
    return f"{out:.{digits}f}"


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


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


def _load_consensus(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"consensus_csv not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    if "nanobody_id" not in df.columns:
        raise ValueError("consensus_csv must contain nanobody_id.")
    if "consensus_rank" not in df.columns:
        if "consensus_score" not in df.columns:
            raise ValueError("consensus_csv must contain consensus_rank or consensus_score.")
        df = df.sort_values("consensus_score", ascending=False, na_position="last").reset_index(drop=True)
        df.insert(0, "consensus_rank", np.arange(1, len(df) + 1, dtype=int))
    df["consensus_rank"] = pd.to_numeric(df["consensus_rank"], errors="coerce")
    return df.sort_values("consensus_rank", ascending=True, na_position="last").reset_index(drop=True)


def _metric_edges(
    winner: pd.Series,
    runner_up: pd.Series,
    *,
    min_metric_delta: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    winner_edges: list[dict[str, Any]] = []
    runner_edges: list[dict[str, Any]] = []
    for spec in METRIC_SPECS:
        col = spec["column"]
        if col not in winner.index or col not in runner_up.index:
            continue
        winner_value = _to_numeric(winner.get(col))
        runner_value = _to_numeric(runner_up.get(col))
        if not np.isfinite(winner_value) or not np.isfinite(runner_value):
            continue
        if spec["direction"] == "higher":
            signed_delta = winner_value - runner_value
            winner_better = signed_delta > 0
        else:
            signed_delta = runner_value - winner_value
            winner_better = signed_delta > 0
        if abs(signed_delta) < float(min_metric_delta):
            continue
        edge = {
            "metric": col,
            "metric_label": spec["label"],
            "direction": spec["direction"],
            "winner_value": winner_value,
            "runner_up_value": runner_value,
            "absolute_delta": abs(signed_delta),
            "signed_delta_for_winner": signed_delta,
        }
        if winner_better:
            winner_edges.append(edge)
        else:
            runner_edges.append(edge)
    winner_edges.sort(key=lambda x: x["absolute_delta"], reverse=True)
    runner_edges.sort(key=lambda x: x["absolute_delta"], reverse=True)
    return winner_edges, runner_edges


def _edge_text(edge: dict[str, Any], *, subject: str) -> str:
    relation = "更高" if edge["direction"] == "higher" else "更低"
    if subject == "winner":
        value_a = _fmt(edge["winner_value"])
        value_b = _fmt(edge["runner_up_value"])
        return f"{edge['metric_label']} {relation} ({value_a} vs {value_b})"
    value_a = _fmt(edge["runner_up_value"])
    value_b = _fmt(edge["winner_value"])
    return f"{edge['metric_label']} {relation} ({value_a} vs {value_b})"


def _risk_text(row: pd.Series) -> list[str]:
    out: list[str] = []
    confidence = str(row.get("confidence_level") or "").strip().lower()
    if confidence == "low":
        out.append("可信度低")
    elif confidence == "medium":
        out.append("可信度中等")
    qc_risk = _to_numeric(row.get("qc_risk_score"), 0.0)
    if qc_risk >= 0.40:
        out.append(f"QC 风险偏高 ({_fmt(qc_risk)})")
    flags_value = row.get("risk_flags")
    flags = "" if pd.isna(flags_value) else str(flags_value).strip()
    if flags and flags.lower() not in {"none", "nan"}:
        out.append(f"风险标记: {flags}")
    if _as_bool(row.get("pocket_overwide_warning", False)):
        out.append("pocket 可能偏宽")
    return out


def _build_tradeoff_rows(df: pd.DataFrame, *, close_score_delta: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    work = df.reset_index(drop=True)
    for idx, row in work.iterrows():
        current_score = _to_numeric(row.get("consensus_score"))
        previous_gap = float("nan")
        next_gap = float("nan")
        if idx > 0:
            previous_gap = _to_numeric(work.loc[idx - 1].get("consensus_score")) - current_score
        if idx + 1 < len(work):
            next_gap = current_score - _to_numeric(work.loc[idx + 1].get("consensus_score"))

        strengths: list[str] = []
        if _to_numeric(row.get("consensus_score"), 0.0) >= 0.75:
            strengths.append("共识综合分高")
        if _to_numeric(row.get("confidence_score"), 0.0) >= 0.75:
            strengths.append("可信度高")
        if _to_numeric(row.get("rank_agreement_score"), 0.0) >= 0.80:
            strengths.append("Rule/ML 一致性好")
        if _to_numeric(row.get("qc_risk_score"), 1.0) < 0.25:
            strengths.append("QC 风险低")
        for col, label in [
            ("ml_mean_topk_pocket_hit_fraction", "ML pocket 覆盖较好"),
            ("rule_mean_topk_pocket_hit_fraction", "Rule pocket 覆盖较好"),
            ("ml_mean_topk_mouth_occlusion_score", "ML 口部遮挡较强"),
            ("rule_mean_topk_mouth_occlusion_score", "Rule 口部遮挡较强"),
            ("ml_mean_topk_ligand_path_block_score", "ML 路径阻断较强"),
            ("rule_mean_topk_ligand_path_block_score", "Rule 路径阻断较强"),
        ]:
            if col in row.index and _to_numeric(row.get(col), 0.0) >= 0.65:
                strengths.append(label)
        if not strengths:
            strengths.append("暂无明显单项优势，主要依赖综合排序")

        risks = _risk_text(row)
        if np.isfinite(next_gap) and next_gap < float(close_score_delta):
            risks.append(f"与下一名分差很小 ({_fmt(next_gap)})")
        std_values = [
            _to_numeric(row.get("ml_std_conformer_score")),
            _to_numeric(row.get("rule_std_conformer_rule_score")),
        ]
        finite_std = [x for x in std_values if np.isfinite(x)]
        if finite_std and max(finite_std) >= 0.15:
            risks.append(f"构象间波动偏大 (max std={_fmt(max(finite_std))})")
        if not risks:
            risks.append("未见明显复核风险")

        base = {col: row.get(col) for col in TRADEOFF_COLUMNS if col in row.index}
        base.update(
            {
                "previous_score_gap": previous_gap,
                "next_score_gap": next_gap,
                "close_decision_warning": bool(
                    (np.isfinite(next_gap) and next_gap < float(close_score_delta))
                    or (np.isfinite(previous_gap) and previous_gap < float(close_score_delta))
                ),
                "strength_summary": "；".join(dict.fromkeys(strengths)),
                "risk_summary": "；".join(dict.fromkeys(risks)),
            }
        )
        rows.append(base)
    return pd.DataFrame(rows)


def _select_pairs(df: pd.DataFrame, mode: str) -> list[tuple[int, int]]:
    n = len(df)
    if n < 2:
        return []
    if mode == "adjacent":
        return [(i, i + 1) for i in range(n - 1)]
    if mode == "top_vs_rest":
        return [(0, i) for i in range(1, n)]
    return list(itertools.combinations(range(n), 2))


def _filter_selected_candidates(consensus: pd.DataFrame, selected_ids: list[str]) -> tuple[pd.DataFrame, list[str]]:
    if not selected_ids:
        return consensus.copy(), []
    available = set(consensus["nanobody_id"].astype(str))
    missing = [item for item in selected_ids if item not in available]
    selected_set = set(selected_ids)
    out = consensus[consensus["nanobody_id"].astype(str).isin(selected_set)].copy()
    out = out.sort_values("consensus_rank", ascending=True, na_position="last").reset_index(drop=True)
    return out, missing


def _build_pairwise_rows(
    df: pd.DataFrame,
    *,
    pair_mode: str,
    close_score_delta: float,
    min_metric_delta: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i, j in _select_pairs(df, pair_mode):
        winner = df.iloc[i]
        runner = df.iloc[j]
        winner_edges, runner_edges = _metric_edges(winner, runner, min_metric_delta=min_metric_delta)
        winner_advantages = [_edge_text(edge, subject="winner") for edge in winner_edges[:4]]
        runner_counterpoints = [_edge_text(edge, subject="runner") for edge in runner_edges[:3]]

        winner_id = str(winner.get("nanobody_id"))
        runner_id = str(runner.get("nanobody_id"))
        score_gap = _to_numeric(winner.get("consensus_score")) - _to_numeric(runner.get("consensus_score"))
        is_close = bool(np.isfinite(score_gap) and score_gap < float(close_score_delta))

        if winner_advantages:
            reason = f"{winner_id} 排在 {runner_id} 前，主要因为" + "；".join(winner_advantages[:3])
        else:
            reason = f"{winner_id} 排在 {runner_id} 前，主要来自共识排序的综合分差"
        if runner_counterpoints:
            reason += "。但需要注意，" + runner_id + " 在" + "；".join(runner_counterpoints[:2]) + "。"
        else:
            reason += "。"
        if is_close:
            reason += f" 两者共识分差较小 ({_fmt(score_gap)})，建议一起进入人工复核。"
        winner_risks = _risk_text(winner)
        if winner_risks:
            reason += " 胜出候选仍需注意：" + "；".join(winner_risks[:2]) + "。"

        rows.append(
            {
                "pair_mode": pair_mode,
                "winner_nanobody_id": winner_id,
                "runner_up_nanobody_id": runner_id,
                "winner_consensus_rank": winner.get("consensus_rank"),
                "runner_up_consensus_rank": runner.get("consensus_rank"),
                "winner_consensus_score": winner.get("consensus_score"),
                "runner_up_consensus_score": runner.get("consensus_score"),
                "consensus_score_gap": score_gap,
                "is_close_decision": is_close,
                "winner_confidence_level": winner.get("confidence_level"),
                "runner_up_confidence_level": runner.get("confidence_level"),
                "winner_risk_flags": winner.get("risk_flags"),
                "runner_up_risk_flags": runner.get("risk_flags"),
                "winner_key_advantages": "；".join(winner_advantages) if winner_advantages else "综合排序领先",
                "runner_up_counterpoints": "；".join(runner_counterpoints) if runner_counterpoints else "none",
                "comparison_explanation": reason,
            }
        )
    return pd.DataFrame(rows)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _detect_group_col(df: pd.DataFrame, requested_group_col: str | None) -> str | None:
    requested = str(requested_group_col or "auto").strip()
    if not requested or requested.lower() == "none":
        return None
    if requested.lower() != "auto":
        return requested if requested in df.columns else None
    for column in GROUP_COL_CANDIDATES:
        if column not in df.columns:
            continue
        values = df[column].map(_clean_text)
        non_empty = values.loc[values.ne("")]
        if non_empty.empty:
            continue
        if non_empty.nunique(dropna=True) <= max(12, int(len(df) * 0.8)):
            return column
    return None


def _top_terms(values: pd.Series, *, max_terms: int = 4) -> str:
    counter: Counter[str] = Counter()
    for value in values.dropna().tolist():
        for item in str(value).replace("|", "；").split("；"):
            text = item.strip()
            if not text or text.lower() in {"none", "nan"}:
                continue
            counter[text] += 1
    if not counter:
        return ""
    return "；".join([term for term, _ in counter.most_common(max_terms)])


def _mean_numeric(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns:
        return float("nan")
    return float(pd.to_numeric(df[column], errors="coerce").mean())


def _build_group_summary_rows(
    top: pd.DataFrame,
    tradeoffs: pd.DataFrame,
    *,
    group_col: str | None,
) -> pd.DataFrame:
    if group_col is None or group_col not in top.columns or top.empty:
        return pd.DataFrame(
            columns=[
                "group_column",
                "group_value",
                "candidate_count",
                "best_consensus_rank",
                "best_nanobody_id",
                "mean_consensus_score",
                "mean_confidence_score",
                "mean_qc_risk_score",
                "low_confidence_count",
                "review_tier_count",
                "close_decision_count",
                "candidate_ids",
                "representative_strengths",
                "representative_risks",
                "group_summary_text",
            ]
        )

    work = top.copy()
    work["_group_value"] = work[group_col].map(_clean_text).replace("", "unknown")
    tradeoff_cols = [
        column
        for column in [
            "nanobody_id",
            "strength_summary",
            "risk_summary",
            "close_decision_warning",
        ]
        if column in tradeoffs.columns
    ]
    if tradeoff_cols:
        work = work.merge(tradeoffs.loc[:, tradeoff_cols], on="nanobody_id", how="left")

    rows: list[dict[str, Any]] = []
    for group_value, group in work.groupby("_group_value", sort=False, dropna=False):
        sorted_group = group.sort_values("consensus_rank", ascending=True, na_position="last")
        best = sorted_group.iloc[0]
        strengths = _top_terms(group.get("strength_summary", pd.Series(dtype=object)))
        risks = _top_terms(group.get("risk_summary", pd.Series(dtype=object)))
        candidate_ids = [str(item) for item in sorted_group["nanobody_id"].astype(str).tolist()]
        low_confidence_count = (
            int(group["confidence_level"].astype(str).str.lower().eq("low").sum())
            if "confidence_level" in group.columns
            else 0
        )
        review_tier_count = (
            int(group["decision_tier"].astype(str).str.lower().eq("review").sum())
            if "decision_tier" in group.columns
            else 0
        )
        close_count = (
            int(group["close_decision_warning"].fillna(False).astype(bool).sum())
            if "close_decision_warning" in group.columns
            else 0
        )
        best_id = str(best.get("nanobody_id") or "")
        summary_bits = [
            f"{group_col}={group_value}",
            f"共 {len(group)} 个候选",
            f"最高排名 {best_id} (rank {int(_to_numeric(best.get('consensus_rank'), 0))})",
        ]
        if strengths:
            summary_bits.append(f"共同优势: {strengths}")
        if risks:
            summary_bits.append(f"主要风险: {risks}")
        if close_count > 0:
            summary_bits.append(f"{close_count} 个候选存在 close decision 风险")

        rows.append(
            {
                "group_column": group_col,
                "group_value": str(group_value),
                "candidate_count": int(len(group)),
                "best_consensus_rank": best.get("consensus_rank"),
                "best_nanobody_id": best_id,
                "mean_consensus_score": _mean_numeric(group, "consensus_score"),
                "mean_confidence_score": _mean_numeric(group, "confidence_score"),
                "mean_qc_risk_score": _mean_numeric(group, "qc_risk_score"),
                "low_confidence_count": low_confidence_count,
                "review_tier_count": review_tier_count,
                "close_decision_count": close_count,
                "candidate_ids": ",".join(candidate_ids),
                "representative_strengths": strengths or "暂无明显共同优势",
                "representative_risks": risks or "未见明显共同风险",
                "group_summary_text": "；".join(summary_bits),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["best_consensus_rank", "candidate_count"],
            ascending=[True, False],
            na_position="last",
            kind="stable",
        ).reset_index(drop=True)
    return out


def _markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 20) -> list[str]:
    if df.empty:
        return ["_No rows._"]
    cols = [col for col in columns if col in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.head(max_rows).iterrows():
        values: list[str] = []
        for col in cols:
            value = row.get(col)
            if isinstance(value, float):
                values.append(_fmt(value))
            else:
                text = str(value) if pd.notna(value) else ""
                values.append(text.replace("\n", " ").replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _build_report(
    *,
    summary: dict[str, Any],
    tradeoffs: pd.DataFrame,
    pairwise: pd.DataFrame,
    group_summary: pd.DataFrame,
) -> str:
    lines = [
        "# Candidate Comparison Report",
        "",
        f"- Compared top candidates: `{summary['top_candidate_count']}`",
        f"- Pairwise comparisons: `{summary['pairwise_comparison_count']}`",
        f"- Close decisions: `{summary['close_decision_count']}`",
        "",
        "## How To Read",
        "",
        "- This report does not change the ranking. It explains existing consensus ranking decisions.",
        "- `winner_key_advantages` lists metrics that support the higher-ranked candidate.",
        "- `runner_up_counterpoints` lists metrics where the lower-ranked candidate is better.",
        "- `is_close_decision=True` means the score gap is small enough that both candidates should be reviewed together.",
        "",
        "## Top Candidate Trade-Offs",
        "",
    ]
    lines.extend(
        _markdown_table(
            tradeoffs,
            [
                "consensus_rank",
                "nanobody_id",
                "consensus_score",
                "confidence_level",
                "next_score_gap",
                "strength_summary",
                "risk_summary",
            ],
            max_rows=20,
        )
    )
    lines.extend(["", "## Pairwise Decision Explanations", ""])
    lines.extend(
        _markdown_table(
            pairwise,
            [
                "winner_nanobody_id",
                "runner_up_nanobody_id",
                "consensus_score_gap",
                "is_close_decision",
                "winner_key_advantages",
                "runner_up_counterpoints",
                "comparison_explanation",
            ],
            max_rows=30,
        )
    )
    lines.extend(["", "## Group-Level Comparison Summary", ""])
    lines.extend(
        _markdown_table(
            group_summary,
            [
                "group_column",
                "group_value",
                "candidate_count",
                "best_nanobody_id",
                "mean_consensus_score",
                "low_confidence_count",
                "close_decision_count",
                "representative_strengths",
                "representative_risks",
                "group_summary_text",
            ],
            max_rows=20,
        )
    )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `candidate_tradeoff_table.csv`",
            "- `candidate_pairwise_comparisons.csv`",
            "- `candidate_group_comparison_summary.csv`",
            "- `candidate_comparison_summary.json`",
        ]
    )
    return "\n".join(lines) + "\n"


def build_candidate_comparison_outputs(
    consensus: pd.DataFrame,
    *,
    top_n: int = 12,
    pair_mode: str = "adjacent",
    selected_nanobody_ids: list[str] | None = None,
    close_score_delta: float = 0.03,
    min_metric_delta: float = 0.02,
    group_col: str = "auto",
    consensus_csv: str | Path | None = None,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    selected_ids = selected_nanobody_ids or []
    if selected_ids:
        top, missing_selected_ids = _filter_selected_candidates(consensus, selected_ids)
        if len(top) < 2:
            raise ValueError("At least two selected nanobody IDs must be found in consensus ranking.")
    else:
        top = consensus.head(max(1, int(top_n))).copy()
        missing_selected_ids = []

    tradeoffs = _build_tradeoff_rows(top, close_score_delta=float(close_score_delta))
    pairwise = _build_pairwise_rows(
        top,
        pair_mode=str(pair_mode),
        close_score_delta=float(close_score_delta),
        min_metric_delta=float(min_metric_delta),
    )
    resolved_group_col = _detect_group_col(top, group_col)
    group_summary = _build_group_summary_rows(top, tradeoffs, group_col=resolved_group_col)
    summary = {
        "consensus_csv": None if consensus_csv is None else str(consensus_csv),
        "out_dir": None if out_dir is None else str(out_dir),
        "top_n": int(top_n),
        "pair_mode": str(pair_mode),
        "custom_selection_enabled": bool(selected_ids),
        "selected_nanobody_ids": selected_ids,
        "missing_selected_nanobody_ids": missing_selected_ids,
        "close_score_delta": float(close_score_delta),
        "min_metric_delta": float(min_metric_delta),
        "requested_group_col": str(group_col),
        "resolved_group_col": resolved_group_col,
        "top_candidate_count": int(len(top)),
        "pairwise_comparison_count": int(len(pairwise)),
        "group_summary_count": int(len(group_summary)),
        "close_decision_count": int(pairwise["is_close_decision"].sum()) if "is_close_decision" in pairwise.columns else 0,
        "low_confidence_top_candidate_count": int(
            top["confidence_level"].astype(str).str.lower().eq("low").sum()
        )
        if "confidence_level" in top.columns
        else 0,
        "review_or_low_confidence_count": int(
            (
                top.get("decision_tier", pd.Series(index=top.index, dtype=object)).astype(str).str.lower().eq("review")
                | top.get("confidence_level", pd.Series(index=top.index, dtype=object)).astype(str).str.lower().eq("low")
            ).sum()
        ),
    }
    report = _build_report(summary=summary, tradeoffs=tradeoffs, pairwise=pairwise, group_summary=group_summary)
    return {
        "selected_candidates": top,
        "tradeoffs": tradeoffs,
        "pairwise": pairwise,
        "group_summary": group_summary,
        "summary": summary,
        "report": report,
    }


def main() -> None:
    args = _build_parser().parse_args()
    consensus_csv = Path(args.consensus_csv).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    consensus = _load_consensus(consensus_csv)
    top_n = max(1, int(args.top_n))
    selected_ids = parse_selected_nanobody_ids(args.selected_nanobody_ids)
    result = build_candidate_comparison_outputs(
        consensus,
        top_n=top_n,
        pair_mode=str(args.pair_mode),
        close_score_delta=float(args.close_score_delta),
        min_metric_delta=float(args.min_metric_delta),
        group_col=str(args.group_col),
        selected_nanobody_ids=selected_ids,
        consensus_csv=consensus_csv,
        out_dir=out_dir,
    )
    tradeoffs = result["tradeoffs"]
    pairwise = result["pairwise"]
    group_summary = result["group_summary"]

    summary = dict(result["summary"])
    summary.update(
        {
        "outputs": {
            "candidate_tradeoff_table_csv": str(out_dir / "candidate_tradeoff_table.csv"),
            "candidate_pairwise_comparisons_csv": str(out_dir / "candidate_pairwise_comparisons.csv"),
            "candidate_group_comparison_summary_csv": str(out_dir / "candidate_group_comparison_summary.csv"),
            "candidate_comparison_summary_json": str(out_dir / "candidate_comparison_summary.json"),
            "candidate_comparison_report_md": str(out_dir / "candidate_comparison_report.md"),
        },
        }
    )

    outputs = {
        "tradeoffs": out_dir / "candidate_tradeoff_table.csv",
        "pairwise": out_dir / "candidate_pairwise_comparisons.csv",
        "group_summary": out_dir / "candidate_group_comparison_summary.csv",
        "summary": out_dir / "candidate_comparison_summary.json",
        "report": out_dir / "candidate_comparison_report.md",
    }
    tradeoffs.to_csv(outputs["tradeoffs"], index=False)
    pairwise.to_csv(outputs["pairwise"], index=False)
    group_summary.to_csv(outputs["group_summary"], index=False)
    outputs["summary"].write_text(json.dumps(_json_sanitize(summary), ensure_ascii=True, indent=2), encoding="utf-8")
    outputs["report"].write_text(result["report"], encoding="utf-8")

    for path in outputs.values():
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
