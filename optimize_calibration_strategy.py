"""Optimize calibration strategy from existing aggregation calibration trials.

This tool avoids re-running expensive calibration searches by re-scoring
existing trials under different strategy settings:
- rank_consistency_weight sweep
- selection_metric sweep
- optional baseline-anchored guard constraints

Outputs:
1) strategy_sweep_results.csv
2) recommended_strategy.json
3) recommended_strategy_report.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _to_float(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def _parse_float_list(text: str, default: list[float]) -> list[float]:
    raw = str(text).strip()
    if not raw:
        return list(default)
    out: list[float] = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        out.append(float(p))
    return out if out else list(default)


def _parse_str_list(text: str, default: list[str]) -> list[str]:
    raw = str(text).strip()
    if not raw:
        return list(default)
    out = [p.strip() for p in raw.split(",") if p.strip()]
    return out if out else list(default)


def _score_objective(pose_auc: float, nb_auc: float, rank_consistency: float, pose_w: float, nb_w: float, rank_w: float) -> float:
    p = pose_auc if np.isfinite(pose_auc) else 0.0
    n = nb_auc if np.isfinite(nb_auc) else 0.0
    r = rank_consistency if np.isfinite(rank_consistency) else 0.0
    r01 = float(np.clip((r + 1.0) * 0.5, 0.0, 1.0))
    return float(max(0.0, pose_w) * p + max(0.0, nb_w) * n + max(0.0, rank_w) * r01)


def _selection_value(row: pd.Series, selection_metric: str) -> float:
    metric = str(selection_metric).strip().lower()
    if metric == "nanobody_auc":
        return _to_float(row.get("nanobody_auc"))
    if metric == "rank_consistency":
        return _to_float(row.get("rank_consistency"))
    return _to_float(row.get("objective_recalc"))


def _pick_best(df: pd.DataFrame, selection_metric: str) -> pd.Series:
    if df.empty:
        raise ValueError("Cannot pick best from empty dataframe")

    work = df.copy()
    work["selection_primary"] = work.apply(lambda r: _selection_value(r, selection_metric), axis=1)
    work = work.sort_values(
        by=["selection_primary", "objective_recalc", "nanobody_auc", "pose_auc", "rank_consistency"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    return work.iloc[0]


def _build_markdown_report(result_df: pd.DataFrame, rec: dict[str, Any], constraints: dict[str, Any]) -> str:
    lines = [
        "# Calibration Strategy Sweep",
        "",
        "## Constraints",
        "",
        f"- min_nanobody_auc: {constraints.get('min_nanobody_auc')}",
        f"- min_rank_consistency: {constraints.get('min_rank_consistency')}",
        "",
        "## Recommended",
        "",
        f"- rank_consistency_weight: {rec.get('rank_consistency_weight')}",
        f"- selection_metric: {rec.get('selection_metric')}",
        f"- feasible: {rec.get('feasible')}",
        f"- used_constraint_fallback: {rec.get('used_constraint_fallback')}",
        f"- objective_recalc: {rec.get('objective_recalc')}",
        f"- pose_auc: {rec.get('pose_auc')}",
        f"- nanobody_auc: {rec.get('nanobody_auc')}",
        f"- rank_consistency: {rec.get('rank_consistency')}",
        "",
        "## Top Strategies",
        "",
        "| rank_weight | selection_metric | feasible | objective | pose_auc | nanobody_auc | rank_consistency |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]

    top = result_df.sort_values(by=["feasible", "objective_recalc"], ascending=[False, False]).head(10)
    for _, row in top.iterrows():
        lines.append(
            "| "
            f"{float(row['rank_consistency_weight']):.3f} | "
            f"{row['selection_metric']} | "
            f"{int(row['feasible'])} | "
            f"{float(row['objective_recalc']):.6f} | "
            f"{float(row['pose_auc']):.6f} | "
            f"{float(row['nanobody_auc']):.6f} | "
            f"{float(row['rank_consistency']):.6f} |"
        )

    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optimize calibration strategy from existing trials")
    parser.add_argument("--aggregation_trials_csv", required=True, help="Path to aggregation_calibration_trials.csv")
    parser.add_argument("--baseline_summary_json", default=None, help="Optional baseline ranking_comparison_summary.json")
    parser.add_argument("--out_dir", default="strategy_optimization", help="Output directory")

    parser.add_argument("--pose_weight", type=float, default=0.40)
    parser.add_argument("--nanobody_weight", type=float, default=0.60)
    parser.add_argument("--rank_weight_grid", default="0.00,0.05,0.10,0.20,0.30,0.40")
    parser.add_argument("--selection_metrics", default="objective,nanobody_auc,rank_consistency")

    parser.add_argument("--use_baseline_guard", action="store_true", help="Use baseline-derived constraints")
    parser.add_argument("--rank_guard_tolerance", type=float, default=0.005)
    parser.add_argument("--auc_guard_tolerance", type=float, default=0.0)

    parser.add_argument("--min_nanobody_auc", type=float, default=None)
    parser.add_argument("--min_rank_consistency", type=float, default=None)
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    trials_csv = Path(args.aggregation_trials_csv).expanduser()
    if not trials_csv.exists():
        raise FileNotFoundError(f"aggregation_trials_csv not found: {trials_csv}")

    trial_df = pd.read_csv(trials_csv, low_memory=False)
    required_cols = ["feature_trial_id", "w_mean", "w_best", "w_consistency", "w_std_penalty", "pose_auc", "nanobody_auc", "rank_consistency"]
    missing = [c for c in required_cols if c not in trial_df.columns]
    if missing:
        raise ValueError(f"aggregation_trials_csv missing columns: {missing}")

    rank_weight_grid = _parse_float_list(args.rank_weight_grid, default=[0.0, 0.1, 0.2, 0.3])
    selection_metrics = _parse_str_list(args.selection_metrics, default=["objective", "nanobody_auc", "rank_consistency"])

    min_nb_auc = args.min_nanobody_auc
    min_rank = args.min_rank_consistency
    baseline_payload: dict[str, Any] | None = None

    if args.use_baseline_guard:
        if not args.baseline_summary_json:
            raise ValueError("--use_baseline_guard requires --baseline_summary_json")
        baseline_path = Path(args.baseline_summary_json).expanduser()
        if not baseline_path.exists():
            raise FileNotFoundError(f"baseline_summary_json not found: {baseline_path}")
        baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))

        base_nb_auc = _to_float(baseline_payload.get("rule_auc"))
        base_rank = _to_float(baseline_payload.get("score_spearman"))

        if np.isfinite(base_nb_auc):
            min_nb_auc = max(0.0, base_nb_auc - max(0.0, float(args.auc_guard_tolerance)) - 1e-12)
        if np.isfinite(base_rank):
            min_rank = max(-1.0, base_rank - max(0.0, float(args.rank_guard_tolerance)) - 1e-12)

    min_nb = _to_float(min_nb_auc) if min_nb_auc is not None else float("nan")
    min_rank_c = _to_float(min_rank) if min_rank is not None else float("nan")

    rows: list[dict[str, Any]] = []
    pose_w = float(args.pose_weight)
    nb_w = float(args.nanobody_weight)

    for rank_w in rank_weight_grid:
        trial_work = trial_df.copy()
        trial_work["objective_recalc"] = trial_work.apply(
            lambda r: _score_objective(
                pose_auc=_to_float(r.get("pose_auc")),
                nb_auc=_to_float(r.get("nanobody_auc")),
                rank_consistency=_to_float(r.get("rank_consistency")),
                pose_w=pose_w,
                nb_w=nb_w,
                rank_w=float(rank_w),
            ),
            axis=1,
        )

        feasible_mask = pd.Series(np.ones((len(trial_work),), dtype=bool), index=trial_work.index)
        nb_auc_col = pd.to_numeric(trial_work["nanobody_auc"], errors="coerce")
        rank_col = pd.to_numeric(trial_work["rank_consistency"], errors="coerce")
        if np.isfinite(min_nb):
            feasible_mask &= nb_auc_col.to_numpy(dtype=np.float64) >= float(min_nb)
        if np.isfinite(min_rank_c):
            feasible_mask &= rank_col.to_numpy(dtype=np.float64) >= float(min_rank_c)

        trial_work["feasible"] = feasible_mask.astype(int)
        feasible_df = trial_work.loc[feasible_mask].copy()

        for selection in selection_metrics:
            used_fallback = False
            if len(feasible_df) > 0:
                best = _pick_best(feasible_df, selection_metric=selection)
            else:
                best = _pick_best(trial_work, selection_metric=selection)
                used_fallback = True

            rows.append(
                {
                    "rank_consistency_weight": float(rank_w),
                    "selection_metric": str(selection),
                    "feasible": int(0 if used_fallback else 1),
                    "used_constraint_fallback": int(1 if used_fallback else 0),
                    "objective_recalc": float(best["objective_recalc"]),
                    "pose_auc": float(best["pose_auc"]),
                    "nanobody_auc": float(best["nanobody_auc"]),
                    "rank_consistency": float(best["rank_consistency"]),
                    "feature_trial_id": int(best["feature_trial_id"]),
                    "w_mean": float(best["w_mean"]),
                    "w_best": float(best["w_best"]),
                    "w_consistency": float(best["w_consistency"]),
                    "w_std_penalty": float(best["w_std_penalty"]),
                }
            )

    if not rows:
        raise ValueError("No strategy rows generated.")

    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values(by=["feasible", "objective_recalc", "nanobody_auc", "rank_consistency"], ascending=[False, False, False, False]).reset_index(drop=True)

    rec_row = result_df.iloc[0]
    rec = {
        "rank_consistency_weight": float(rec_row["rank_consistency_weight"]),
        "selection_metric": str(rec_row["selection_metric"]),
        "feasible": bool(int(rec_row["feasible"]) == 1),
        "used_constraint_fallback": bool(int(rec_row["used_constraint_fallback"]) == 1),
        "objective_recalc": float(rec_row["objective_recalc"]),
        "pose_auc": float(rec_row["pose_auc"]),
        "nanobody_auc": float(rec_row["nanobody_auc"]),
        "rank_consistency": float(rec_row["rank_consistency"]),
        "feature_trial_id": int(rec_row["feature_trial_id"]),
        "w_mean": float(rec_row["w_mean"]),
        "w_best": float(rec_row["w_best"]),
        "w_consistency": float(rec_row["w_consistency"]),
        "w_std_penalty": float(rec_row["w_std_penalty"]),
    }

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "strategy_sweep_results.csv"
    rec_json = out_dir / "recommended_strategy.json"
    report_md = out_dir / "recommended_strategy_report.md"

    result_df.to_csv(csv_path, index=False)

    constraints = {
        "min_nanobody_auc": float(min_nb) if np.isfinite(min_nb) else None,
        "min_rank_consistency": float(min_rank_c) if np.isfinite(min_rank_c) else None,
    }
    payload = {
        "aggregation_trials_csv": str(trials_csv),
        "pose_weight": pose_w,
        "nanobody_weight": nb_w,
        "rank_weight_grid": rank_weight_grid,
        "selection_metrics": selection_metrics,
        "constraints": constraints,
        "baseline_guard_enabled": bool(args.use_baseline_guard),
        "baseline_summary": baseline_payload,
        "recommended": rec,
    }
    rec_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    report_md.write_text(_build_markdown_report(result_df=result_df, rec=rec, constraints=constraints), encoding="utf-8")

    print(f"Saved: {csv_path}")
    print(f"Saved: {rec_json}")
    print(f"Saved: {report_md}")
    print(
        "Recommended strategy: "
        f"rank_weight={rec['rank_consistency_weight']:.3f}, "
        f"selection_metric={rec['selection_metric']}, "
        f"feasible={rec['feasible']}"
    )


if __name__ == "__main__":
    main()
