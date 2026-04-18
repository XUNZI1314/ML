"""Analyze how robust consensus rankings are to simple parameter changes.

This is a lightweight post-processing tool. It reads an existing
``consensus_ranking.csv`` and rescans scoring weights/QC penalty strengths
without rerunning feature construction, model training or docking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze ranking sensitivity from consensus_ranking.csv")
    parser.add_argument("--consensus_csv", required=True, help="Path to consensus_outputs/consensus_ranking.csv")
    parser.add_argument("--out_dir", default="parameter_sensitivity_outputs", help="Output directory")
    parser.add_argument("--top_n", type=int, default=5, help="Top-N boundary used for stability checks")
    parser.add_argument(
        "--ml_weight_grid",
        default="0.35,0.50,0.65",
        help="Comma-separated ML weights inside the Rule/ML score component",
    )
    parser.add_argument(
        "--rank_agreement_weight_grid",
        default="0.10,0.20,0.30",
        help="Comma-separated rank-agreement weights",
    )
    parser.add_argument(
        "--qc_penalty_weight_grid",
        default="0.00,0.15,0.30",
        help="Comma-separated QC risk penalty weights",
    )
    parser.add_argument(
        "--score_alignment_weight",
        type=float,
        default=0.10,
        help="Fixed score-alignment weight used in all scenarios",
    )
    parser.add_argument(
        "--sensitive_rank_span",
        type=float,
        default=3.0,
        help="Candidate is flagged sensitive when rank span is at least this value",
    )
    return parser


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _clip01(value: Any) -> float:
    return float(np.clip(_safe_float(value), 0.0, 1.0))


def _parse_grid(text: str, *, name: str) -> list[float]:
    values: list[float] = []
    for raw in str(text or "").split(","):
        item = raw.strip()
        if not item:
            continue
        try:
            value = float(item)
        except ValueError as exc:
            raise ValueError(f"Invalid {name} value {item!r}.") from exc
        if not np.isfinite(value):
            raise ValueError(f"Invalid {name} value {item!r}: not finite.")
        values.append(float(np.clip(value, 0.0, 1.0)))
    if not values:
        raise ValueError(f"{name} cannot be empty.")
    return sorted(set(values))


def _first_existing(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _load_consensus(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"consensus_csv not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    if "nanobody_id" not in df.columns:
        raise ValueError("consensus_csv must contain nanobody_id.")
    if "consensus_score" not in df.columns and ("ml_score" not in df.columns and "rule_score" not in df.columns):
        raise ValueError("consensus_csv must contain consensus_score or at least one of ml_score/rule_score.")
    return df.copy()


def _prepare_base(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if "consensus_rank" in work.columns:
        work["baseline_rank"] = _to_numeric(work["consensus_rank"])
    elif "rank" in work.columns:
        work["baseline_rank"] = _to_numeric(work["rank"])
    elif "consensus_score" in work.columns:
        work["baseline_rank"] = _to_numeric(work["consensus_score"]).rank(method="min", ascending=False)
    else:
        score_col = _first_existing(work, ["ml_score", "rule_score"])
        work["baseline_rank"] = _to_numeric(work[score_col]).rank(method="min", ascending=False)

    for col, default in [
        ("ml_score", np.nan),
        ("rule_score", np.nan),
        ("consensus_score", np.nan),
        ("rank_agreement_score", 0.50),
        ("score_alignment_score", 0.50),
        ("qc_risk_score", 0.0),
        ("confidence_score", np.nan),
    ]:
        if col in work.columns:
            work[col] = _to_numeric(work[col])
        else:
            work[col] = default

    return work


def _method_score(row: pd.Series, *, ml_weight: float) -> float:
    ml_score = _safe_float(row.get("ml_score"), default=float("nan"))
    rule_score = _safe_float(row.get("rule_score"), default=float("nan"))
    if np.isfinite(ml_score) and np.isfinite(rule_score):
        return _clip01(float(ml_weight) * ml_score + (1.0 - float(ml_weight)) * rule_score)
    if np.isfinite(ml_score):
        return _clip01(ml_score)
    if np.isfinite(rule_score):
        return _clip01(rule_score)
    return _clip01(row.get("consensus_score"))


def _build_scenarios(
    *,
    ml_weight_grid: list[float],
    rank_agreement_weight_grid: list[float],
    qc_penalty_weight_grid: list[float],
    score_alignment_weight: float,
) -> list[dict[str, float | str]]:
    scenarios: list[dict[str, float | str]] = []
    align_w = float(np.clip(score_alignment_weight, 0.0, 0.50))
    for ml_weight in ml_weight_grid:
        for rank_w in rank_agreement_weight_grid:
            for qc_w in qc_penalty_weight_grid:
                score_w = max(0.0, 1.0 - float(rank_w) - align_w)
                scenarios.append(
                    {
                        "scenario_name": f"ml{ml_weight:.2f}_rank{rank_w:.2f}_qc{qc_w:.2f}",
                        "ml_weight": float(ml_weight),
                        "rule_weight": float(1.0 - ml_weight),
                        "method_score_weight": float(score_w),
                        "rank_agreement_weight": float(rank_w),
                        "score_alignment_weight": float(align_w),
                        "qc_penalty_weight": float(qc_w),
                    }
                )
    return scenarios


def build_parameter_sensitivity(
    consensus_df: pd.DataFrame,
    *,
    top_n: int = 5,
    ml_weight_grid: list[float] | None = None,
    rank_agreement_weight_grid: list[float] | None = None,
    qc_penalty_weight_grid: list[float] | None = None,
    score_alignment_weight: float = 0.10,
    sensitive_rank_span: float = 3.0,
) -> dict[str, Any]:
    base = _prepare_base(consensus_df)
    top_n = max(1, int(top_n))
    scenario_specs = _build_scenarios(
        ml_weight_grid=ml_weight_grid or [0.35, 0.50, 0.65],
        rank_agreement_weight_grid=rank_agreement_weight_grid or [0.10, 0.20, 0.30],
        qc_penalty_weight_grid=qc_penalty_weight_grid or [0.0, 0.15, 0.30],
        score_alignment_weight=float(score_alignment_weight),
    )

    ranking_frames: list[pd.DataFrame] = []
    for scenario in scenario_specs:
        rows: list[dict[str, Any]] = []
        for _, row in base.iterrows():
            method_score = _method_score(row, ml_weight=float(scenario["ml_weight"]))
            score = (
                float(scenario["method_score_weight"]) * method_score
                + float(scenario["rank_agreement_weight"]) * _clip01(row.get("rank_agreement_score"))
                + float(scenario["score_alignment_weight"]) * _clip01(row.get("score_alignment_score"))
                - float(scenario["qc_penalty_weight"]) * _clip01(row.get("qc_risk_score"))
            )
            rows.append(
                {
                    **scenario,
                    "nanobody_id": row["nanobody_id"],
                    "sensitivity_score": _clip01(score),
                    "method_score_component": method_score,
                    "baseline_rank": _safe_float(row.get("baseline_rank"), default=float("nan")),
                    "baseline_consensus_score": _safe_float(row.get("consensus_score"), default=float("nan")),
                    "decision_tier": row.get("decision_tier"),
                    "confidence_level": row.get("confidence_level"),
                    "risk_flags": row.get("risk_flags"),
                }
            )
        frame = pd.DataFrame(rows)
        frame = frame.sort_values(
            by=["sensitivity_score", "method_score_component"],
            ascending=[False, False],
            na_position="last",
        ).reset_index(drop=True)
        frame.insert(0, "sensitivity_rank", np.arange(1, len(frame) + 1, dtype=int))
        frame["rank_delta_from_baseline"] = frame["sensitivity_rank"] - frame["baseline_rank"]
        ranking_frames.append(frame)

    scenario_rankings = pd.concat(ranking_frames, ignore_index=True) if ranking_frames else pd.DataFrame()

    baseline_top = set(
        base.sort_values("baseline_rank", ascending=True).head(top_n)["nanobody_id"].astype(str).tolist()
    )
    scenario_summary_rows: list[dict[str, Any]] = []
    for scenario_name, group in scenario_rankings.groupby("scenario_name", dropna=False):
        top_ids = set(group.sort_values("sensitivity_rank").head(top_n)["nanobody_id"].astype(str).tolist())
        union = baseline_top | top_ids
        jaccard = float(len(baseline_top & top_ids) / len(union)) if union else 1.0
        rank_corr_df = group.loc[:, ["baseline_rank", "sensitivity_rank"]].dropna()
        rank_spearman = (
            float(rank_corr_df["baseline_rank"].corr(rank_corr_df["sensitivity_rank"], method="spearman"))
            if rank_corr_df.shape[0] >= 2
            else float("nan")
        )
        first = group.iloc[0]
        scenario_summary_rows.append(
            {
                "scenario_name": scenario_name,
                "ml_weight": first.get("ml_weight"),
                "rule_weight": first.get("rule_weight"),
                "rank_agreement_weight": first.get("rank_agreement_weight"),
                "score_alignment_weight": first.get("score_alignment_weight"),
                "qc_penalty_weight": first.get("qc_penalty_weight"),
                "top_n_jaccard_vs_baseline": jaccard,
                "rank_spearman_vs_baseline": rank_spearman,
                "top_n_ids": ";".join(sorted(top_ids)),
            }
        )
    scenario_summary = pd.DataFrame(scenario_summary_rows).sort_values("scenario_name").reset_index(drop=True)

    candidate_rows: list[dict[str, Any]] = []
    for nanobody_id, group in scenario_rankings.groupby("nanobody_id", dropna=False):
        ranks = _to_numeric(group["sensitivity_rank"])
        scores = _to_numeric(group["sensitivity_score"])
        baseline_rank = _safe_float(group["baseline_rank"].iloc[0], default=float("nan"))
        best_rank = float(ranks.min())
        worst_rank = float(ranks.max())
        top_freq = float((ranks <= top_n).mean()) if len(ranks) else 0.0
        top_unstable = bool((baseline_rank <= top_n and worst_rank > top_n) or (baseline_rank > top_n and best_rank <= top_n))
        rank_span = float(worst_rank - best_rank)
        reasons: list[str] = []
        if rank_span >= float(sensitive_rank_span):
            reasons.append("rank_span_high")
        if top_unstable:
            reasons.append("top_n_boundary_unstable")
        score_span = float(scores.max() - scores.min()) if len(scores) else 0.0
        if score_span >= 0.10:
            reasons.append("score_span_high")
        candidate_rows.append(
            {
                "nanobody_id": nanobody_id,
                "baseline_rank": baseline_rank,
                "best_rank": best_rank,
                "worst_rank": worst_rank,
                "mean_rank": float(ranks.mean()),
                "rank_std": float(ranks.std(ddof=0)),
                "rank_span": rank_span,
                "best_score": float(scores.max()),
                "worst_score": float(scores.min()),
                "score_span": score_span,
                "top_n_frequency": top_freq,
                "top_n_unstable": top_unstable,
                "is_sensitive": bool(reasons),
                "sensitivity_reason": ";".join(reasons) if reasons else "stable",
                "decision_tier": group["decision_tier"].iloc[0],
                "confidence_level": group["confidence_level"].iloc[0],
                "risk_flags": group["risk_flags"].iloc[0],
            }
        )
    candidate_sensitivity = pd.DataFrame(candidate_rows).sort_values(
        by=["is_sensitive", "rank_span", "score_span"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    sensitive_candidates = candidate_sensitivity[candidate_sensitivity["is_sensitive"].astype(bool)].copy()

    summary = {
        "candidate_count": int(base.shape[0]),
        "scenario_count": int(len(scenario_specs)),
        "top_n": int(top_n),
        "sensitive_rank_span": float(sensitive_rank_span),
        "sensitive_candidate_count": int(sensitive_candidates.shape[0]),
        "top_n_unstable_count": int(candidate_sensitivity["top_n_unstable"].astype(bool).sum()) if not candidate_sensitivity.empty else 0,
        "max_rank_span": float(candidate_sensitivity["rank_span"].max()) if not candidate_sensitivity.empty else 0.0,
        "mean_rank_span": float(candidate_sensitivity["rank_span"].mean()) if not candidate_sensitivity.empty else 0.0,
        "min_top_n_jaccard_vs_baseline": float(scenario_summary["top_n_jaccard_vs_baseline"].min()) if not scenario_summary.empty else 1.0,
        "mean_top_n_jaccard_vs_baseline": float(scenario_summary["top_n_jaccard_vs_baseline"].mean()) if not scenario_summary.empty else 1.0,
        "most_sensitive_candidates": sensitive_candidates.head(10).replace({np.nan: None}).to_dict(orient="records"),
        "formula": "score = (1-rank_agreement_weight-score_alignment_weight)*(ml_weight*ML + rule_weight*Rule) + rank_agreement_weight*rank_agreement + score_alignment_weight*score_alignment - qc_penalty_weight*qc_risk",
    }

    return {
        "scenario_rankings": scenario_rankings,
        "scenario_summary": scenario_summary,
        "candidate_sensitivity": candidate_sensitivity,
        "sensitive_candidates": sensitive_candidates,
        "summary": summary,
    }


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


def _build_report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    sensitive = result["sensitive_candidates"]
    scenario_summary = result["scenario_summary"]
    lines = [
        "# Ranking Parameter Sensitivity Analysis",
        "",
        f"- Candidates: `{summary['candidate_count']}`",
        f"- Scenarios: `{summary['scenario_count']}`",
        f"- Top-N boundary: `{summary['top_n']}`",
        f"- Sensitive candidates: `{summary['sensitive_candidate_count']}`",
        f"- Top-N unstable candidates: `{summary['top_n_unstable_count']}`",
        f"- Max rank span: `{summary['max_rank_span']:.2f}`",
        f"- Mean Top-N Jaccard vs baseline: `{summary['mean_top_n_jaccard_vs_baseline']:.4f}`",
        "",
        "## Interpretation",
        "",
        "- Low rank span means the candidate is stable under the tested scoring weights.",
        "- `top_n_unstable=True` means the candidate can move across the Top-N decision boundary.",
        "- This report is a post-processing robustness check; it does not retrain the model.",
        "",
        "## Most Sensitive Candidates",
        "",
        "| nanobody_id | baseline_rank | best_rank | worst_rank | rank_span | top_n_frequency | reason |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in sensitive.head(20).iterrows():
        lines.append(
            "| "
            f"{row.get('nanobody_id')} | "
            f"{_safe_float(row.get('baseline_rank'), float('nan')):.0f} | "
            f"{_safe_float(row.get('best_rank'), float('nan')):.0f} | "
            f"{_safe_float(row.get('worst_rank'), float('nan')):.0f} | "
            f"{_safe_float(row.get('rank_span'), 0.0):.2f} | "
            f"{_safe_float(row.get('top_n_frequency'), 0.0):.2f} | "
            f"{row.get('sensitivity_reason')} |"
        )
    if sensitive.empty:
        lines.append("| none |  |  |  |  |  | no sensitive candidates under current thresholds |")

    lines.extend(
        [
            "",
            "## Least Stable Scenarios",
            "",
            "| scenario | top_n_jaccard_vs_baseline | rank_spearman_vs_baseline |",
            "|---|---:|---:|",
        ]
    )
    if not scenario_summary.empty:
        view = scenario_summary.sort_values(
            by=["top_n_jaccard_vs_baseline", "rank_spearman_vs_baseline"],
            ascending=[True, True],
            na_position="last",
        ).head(10)
        for _, row in view.iterrows():
            lines.append(
                "| "
                f"{row.get('scenario_name')} | "
                f"{_safe_float(row.get('top_n_jaccard_vs_baseline'), 0.0):.4f} | "
                f"{_safe_float(row.get('rank_spearman_vs_baseline'), float('nan')):.4f} |"
            )

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `scenario_rankings.csv`: full ranking under every tested parameter scenario.",
            "- `scenario_summary.csv`: Top-N overlap and rank correlation for each scenario.",
            "- `candidate_rank_sensitivity.csv`: per-candidate rank/score stability summary.",
            "- `sensitive_candidates.csv`: candidates flagged as sensitive.",
            "- `parameter_sensitivity_summary.json`: machine-readable summary.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = _build_parser().parse_args()
    consensus_csv = Path(args.consensus_csv).expanduser().resolve()
    consensus_df = _load_consensus(consensus_csv)
    result = build_parameter_sensitivity(
        consensus_df,
        top_n=int(args.top_n),
        ml_weight_grid=_parse_grid(args.ml_weight_grid, name="ml_weight_grid"),
        rank_agreement_weight_grid=_parse_grid(args.rank_agreement_weight_grid, name="rank_agreement_weight_grid"),
        qc_penalty_weight_grid=_parse_grid(args.qc_penalty_weight_grid, name="qc_penalty_weight_grid"),
        score_alignment_weight=float(args.score_alignment_weight),
        sensitive_rank_span=float(args.sensitive_rank_span),
    )

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    scenario_rankings_csv = out_dir / "scenario_rankings.csv"
    scenario_summary_csv = out_dir / "scenario_summary.csv"
    candidate_sensitivity_csv = out_dir / "candidate_rank_sensitivity.csv"
    sensitive_candidates_csv = out_dir / "sensitive_candidates.csv"
    summary_json = out_dir / "parameter_sensitivity_summary.json"
    report_md = out_dir / "parameter_sensitivity_report.md"

    result["scenario_rankings"].to_csv(scenario_rankings_csv, index=False)
    result["scenario_summary"].to_csv(scenario_summary_csv, index=False)
    result["candidate_sensitivity"].to_csv(candidate_sensitivity_csv, index=False)
    result["sensitive_candidates"].to_csv(sensitive_candidates_csv, index=False)
    summary_json.write_text(json.dumps(_json_sanitize(result["summary"]), ensure_ascii=True, indent=2), encoding="utf-8")
    report_md.write_text(_build_report(result), encoding="utf-8")

    for path in [
        scenario_rankings_csv,
        scenario_summary_csv,
        candidate_sensitivity_csv,
        sensitive_candidates_csv,
        summary_json,
        report_md,
    ]:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
