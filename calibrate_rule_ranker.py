"""Calibrate rule-based ranking weights with supervision and optional ML consistency.

This script searches:
1) feature weights used by `build_rule_score`
2) aggregation weights used by `aggregate_rule_scores`

The objective can combine:
- pose-level AUC
- nanobody-level AUC
- optional ranking consistency to a reference ML ranking
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rule_ranker import (
    DEFAULT_RULE_FEATURE_SPECS,
    aggregate_rule_scores,
    build_rule_score,
    rank_nanobodies_by_rules,
    save_rule_outputs,
)


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute binary AUC without external dependencies."""
    y = np.asarray(y_true, dtype=np.float64)
    s = np.asarray(y_score, dtype=np.float64)

    mask = np.isfinite(y) & np.isfinite(s)
    y = y[mask]
    s = s[mask]
    if y.size == 0:
        return float("nan")

    y = (y >= 0.5).astype(np.int32)
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    if pos == 0 or neg == 0:
        return float("nan")

    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    auc = (np.sum(ranks[y == 1]) - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(xv) & np.isfinite(yv)
    xv = xv[mask]
    yv = yv[mask]
    if xv.size < 2:
        return float("nan")

    xc = xv - np.mean(xv)
    yc = yv - np.mean(yv)
    denom = np.sqrt(np.sum(xc * xc) * np.sum(yc * yc))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(xc * yc) / denom)


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    xr = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    yr = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    return _pearson_corr(xr, yr)


def _normalize_weights(raw: np.ndarray) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float64)
    arr = np.where(np.isfinite(arr), np.maximum(arr, 0.0), 0.0)
    total = float(np.sum(arr))
    if total <= 0.0:
        return np.full_like(arr, 1.0 / max(1, arr.size), dtype=np.float64)
    return arr / total


def _parse_float_list(value: str, default: list[float]) -> list[float]:
    text = str(value).strip()
    if not text:
        return list(default)

    out: list[float] = []
    for part in text.split(","):
        p = part.strip()
        if not p:
            continue
        out.append(float(p))
    return out if out else list(default)


def _derive_nanobody_labels(
    df: pd.DataFrame,
    label_col: str,
    threshold: float = 0.50,
) -> pd.DataFrame:
    work = df[["nanobody_id", label_col]].copy()
    work[label_col] = _to_numeric(work[label_col])
    work = work.loc[work[label_col].notna()].copy()
    if work.empty:
        raise ValueError(f"No valid values found in label column: {label_col}")

    grouped = work.groupby("nanobody_id", as_index=False)[label_col].mean()
    grouped = grouped.rename(columns={label_col: "nanobody_label_score"})

    thr = float(np.clip(threshold, 0.0, 1.0))
    grouped["nanobody_label"] = (grouped["nanobody_label_score"] >= thr).astype(float)

    # Fallback for degenerate all-0/all-1 labels.
    if grouped["nanobody_label"].nunique(dropna=True) < 2:
        median_score = float(grouped["nanobody_label_score"].median())
        grouped["nanobody_label"] = (grouped["nanobody_label_score"] > median_score).astype(float)

    if grouped["nanobody_label"].nunique(dropna=True) < 2:
        rank_pct = grouped["nanobody_label_score"].rank(method="average", pct=True)
        grouped["nanobody_label"] = (rank_pct >= 0.50).astype(float)

    if grouped["nanobody_label"].nunique(dropna=True) < 2:
        raise ValueError(
            "Failed to derive non-degenerate nanobody labels from pose labels. "
            "Please provide more diverse labels."
        )

    return grouped[["nanobody_id", "nanobody_label", "nanobody_label_score"]]


def _score_from_metrics(
    pose_auc: float,
    nanobody_auc: float,
    rank_consistency: float,
    pose_weight: float,
    nanobody_weight: float,
    rank_weight: float,
) -> float:
    p = float(pose_auc) if np.isfinite(pose_auc) else 0.0
    n = float(nanobody_auc) if np.isfinite(nanobody_auc) else 0.0
    r = float(rank_consistency) if np.isfinite(rank_consistency) else 0.0
    # Map correlation from [-1, 1] to [0, 1] so the objective is consistently additive.
    r01 = float(np.clip((r + 1.0) * 0.5, 0.0, 1.0))
    return float(
        max(0.0, pose_weight) * p
        + max(0.0, nanobody_weight) * n
        + max(0.0, rank_weight) * r01
    )


def _serialize_feature_spec(spec: dict[str, tuple[int, float]]) -> str:
    payload = {k: {"sign": int(v[0]), "weight": float(v[1])} for k, v in spec.items()}
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _build_feature_trials(
    available_specs: dict[str, tuple[int, float]],
    n_trials: int,
    random_seed: int,
    jitter_sigma: float,
) -> list[dict[str, tuple[int, float]]]:
    features = list(available_specs.keys())
    base = np.array([available_specs[f][1] for f in features], dtype=np.float64)
    base = _normalize_weights(base)

    rng = np.random.default_rng(int(random_seed))
    trials: list[dict[str, tuple[int, float]]] = []

    # Trial 0: baseline default weights
    trials.append({f: (int(available_specs[f][0]), float(base[i])) for i, f in enumerate(features)})

    n_extra = max(0, int(n_trials) - 1)
    alpha_base = np.maximum(base * max(2.0, float(len(features))), 0.2)
    sigma = max(0.01, float(jitter_sigma))

    for i in range(n_extra):
        if i % 2 == 0:
            noise = np.exp(rng.normal(loc=0.0, scale=sigma, size=len(features)))
            raw = base * noise
        else:
            raw = rng.dirichlet(alpha_base)

        w = _normalize_weights(raw)
        spec = {f: (int(available_specs[f][0]), float(w[idx])) for idx, f in enumerate(features)}
        trials.append(spec)

    return trials


@dataclass(slots=True)
class FeatureTrialResult:
    trial_id: int
    pose_auc: float
    feature_spec: dict[str, tuple[int, float]]
    pose_rule_df: pd.DataFrame


def _run_feature_search(
    pose_df: pd.DataFrame,
    label_col: str,
    feature_trials: list[dict[str, tuple[int, float]]],
    lower_q: float,
    upper_q: float,
) -> tuple[list[FeatureTrialResult], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    results: list[FeatureTrialResult] = []

    for trial_id, spec in enumerate(feature_trials):
        pose_rule_df, info = build_rule_score(
            pose_df,
            feature_specs=spec,
            lower_q=float(lower_q),
            upper_q=float(upper_q),
        )

        label = _to_numeric(pose_rule_df[label_col])
        score = _to_numeric(pose_rule_df["rule_blocking_score"])
        auc = _binary_auc(label.to_numpy(dtype=np.float64), score.to_numpy(dtype=np.float64))

        rows.append(
            {
                "feature_trial_id": int(trial_id),
                "pose_auc": float(auc),
                "feature_count": int(info.get("feature_count", len(spec))),
                "feature_spec_json": _serialize_feature_spec(spec),
            }
        )
        results.append(
            FeatureTrialResult(
                trial_id=int(trial_id),
                pose_auc=float(auc),
                feature_spec=spec,
                pose_rule_df=pose_rule_df,
            )
        )

    trial_df = pd.DataFrame(rows).sort_values(by=["pose_auc", "feature_trial_id"], ascending=[False, True]).reset_index(drop=True)
    return results, trial_df


def _run_aggregation_search(
    top_feature_results: list[FeatureTrialResult],
    nanobody_label_df: pd.DataFrame,
    ml_reference_df: pd.DataFrame | None,
    rank_consistency_metric: str,
    min_pose_auc: float | None,
    min_nanobody_auc: float | None,
    min_rank_consistency: float | None,
    selection_metric: str,
    top_k: int,
    conformer_geo_weight: float,
    pocket_overwide_penalty_weight: float,
    pocket_overwide_threshold: float,
    w_mean_grid: list[float],
    w_best_grid: list[float],
    w_consistency_grid: list[float],
    w_std_grid: list[float],
    consistency_hit_threshold: float,
    pose_weight: float,
    nanobody_weight: float,
    rank_weight: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    best_any: dict[str, Any] | None = None
    best_feasible: dict[str, Any] | None = None

    min_pose = float(min_pose_auc) if min_pose_auc is not None and np.isfinite(min_pose_auc) else None
    min_nb = float(min_nanobody_auc) if min_nanobody_auc is not None and np.isfinite(min_nanobody_auc) else None
    min_rank = float(min_rank_consistency) if min_rank_consistency is not None and np.isfinite(min_rank_consistency) else None

    def _metric_or_neg_inf(value: float) -> float:
        return float(value) if np.isfinite(value) else float("-inf")

    def _selection_primary(row: dict[str, Any]) -> float:
        metric = str(selection_metric).strip().lower()
        if metric == "nanobody_auc":
            return _metric_or_neg_inf(float(row.get("nanobody_auc", float("nan"))))
        if metric == "rank_consistency":
            return _metric_or_neg_inf(float(row.get("rank_consistency", float("nan"))))
        return _metric_or_neg_inf(float(row.get("objective", float("nan"))))

    def _is_better(row_a: dict[str, Any] | None, row_b: dict[str, Any]) -> bool:
        if row_a is None:
            return True

        key_a = (
            _selection_primary(row_a),
            _metric_or_neg_inf(float(row_a.get("objective", float("nan")))),
            _metric_or_neg_inf(float(row_a.get("nanobody_auc", float("nan")))),
            _metric_or_neg_inf(float(row_a.get("pose_auc", float("nan")))),
            _metric_or_neg_inf(float(row_a.get("rank_consistency", float("nan")))),
        )
        key_b = (
            _selection_primary(row_b),
            _metric_or_neg_inf(float(row_b.get("objective", float("nan")))),
            _metric_or_neg_inf(float(row_b.get("nanobody_auc", float("nan")))),
            _metric_or_neg_inf(float(row_b.get("pose_auc", float("nan")))),
            _metric_or_neg_inf(float(row_b.get("rank_consistency", float("nan")))),
        )
        return key_b > key_a

    for fr in top_feature_results:
        for w_mean in w_mean_grid:
            for w_best in w_best_grid:
                for w_consistency in w_consistency_grid:
                    for w_std in w_std_grid:
                        conformer_df, nanobody_df = aggregate_rule_scores(
                            fr.pose_rule_df,
                            top_k=int(top_k),
                            conformer_geo_weight=float(conformer_geo_weight),
                            pocket_overwide_penalty_weight=float(pocket_overwide_penalty_weight),
                            pocket_overwide_threshold=float(pocket_overwide_threshold),
                            mean_weight=float(w_mean),
                            best_weight=float(w_best),
                            consistency_weight=float(w_consistency),
                            std_penalty_weight=float(w_std),
                            consistency_hit_threshold=float(consistency_hit_threshold),
                        )
                        ranking_df = rank_nanobodies_by_rules(nanobody_df)

                        merged = ranking_df[["nanobody_id", "final_score"]].merge(
                            nanobody_label_df,
                            on="nanobody_id",
                            how="inner",
                        )
                        nb_auc = _binary_auc(
                            merged["nanobody_label"].to_numpy(dtype=np.float64),
                            merged["final_score"].to_numpy(dtype=np.float64),
                        )

                        rank_consistency = float("nan")
                        if ml_reference_df is not None:
                            merged_ml = ranking_df[["nanobody_id", "final_score"]].merge(
                                ml_reference_df,
                                on="nanobody_id",
                                how="inner",
                            )
                            if len(merged_ml) >= 2:
                                metric = str(rank_consistency_metric).strip().lower()
                                if metric == "rank_spearman":
                                    rule_rank = merged_ml["final_score"].rank(method="average", ascending=False).to_numpy(dtype=np.float64)
                                    ml_rank = merged_ml["ml_reference_score"].rank(method="average", ascending=False).to_numpy(dtype=np.float64)
                                    rank_consistency = _spearman_corr(rule_rank, ml_rank)
                                else:
                                    rank_consistency = _spearman_corr(
                                        merged_ml["final_score"].to_numpy(dtype=np.float64),
                                        merged_ml["ml_reference_score"].to_numpy(dtype=np.float64),
                                    )

                        objective = _score_from_metrics(
                            pose_auc=fr.pose_auc,
                            nanobody_auc=nb_auc,
                            rank_consistency=rank_consistency,
                            pose_weight=float(pose_weight),
                            nanobody_weight=float(nanobody_weight),
                            rank_weight=float(rank_weight),
                        )

                        meets_pose = True if min_pose is None else (np.isfinite(fr.pose_auc) and float(fr.pose_auc) >= min_pose)
                        meets_nb = True if min_nb is None else (np.isfinite(nb_auc) and float(nb_auc) >= min_nb)
                        meets_rank = True if min_rank is None else (np.isfinite(rank_consistency) and float(rank_consistency) >= min_rank)
                        feasible = bool(meets_pose and meets_nb and meets_rank)

                        row = {
                            "feature_trial_id": int(fr.trial_id),
                            "pose_auc": float(fr.pose_auc),
                            "nanobody_auc": float(nb_auc),
                            "rank_consistency": float(rank_consistency),
                            "objective": float(objective),
                            "selection_primary": float(_selection_primary({
                                "objective": objective,
                                "nanobody_auc": nb_auc,
                                "rank_consistency": rank_consistency,
                            })),
                            "feasible": int(feasible),
                            "w_mean": float(w_mean),
                            "w_best": float(w_best),
                            "w_consistency": float(w_consistency),
                            "w_std_penalty": float(w_std),
                            "pocket_overwide_penalty_weight": float(pocket_overwide_penalty_weight),
                            "pocket_overwide_threshold": float(pocket_overwide_threshold),
                        }
                        rows.append(row)

                        candidate = {
                            **row,
                            "feature_spec": fr.feature_spec,
                            "pose_rule_df": fr.pose_rule_df,
                            "conformer_df": conformer_df,
                            "ranking_df": ranking_df,
                        }

                        if _is_better(best_any, candidate):
                            best_any = candidate
                        if feasible and _is_better(best_feasible, candidate):
                            best_feasible = candidate

    if not rows or best_any is None:
        raise ValueError("No valid aggregation trial was produced.")

    best = best_feasible if best_feasible is not None else best_any
    best["used_constraint_fallback"] = bool(best_feasible is None)
    best["selection_metric"] = str(selection_metric)

    feasible_count = int(np.sum(pd.DataFrame(rows)["feasible"].to_numpy(dtype=np.int64)))
    best["feasible_count"] = feasible_count
    best["total_trial_count"] = int(len(rows))

    agg_df = pd.DataFrame(rows).sort_values(
        by=["feasible", "selection_primary", "objective", "nanobody_auc", "pose_auc", "rank_consistency"],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)
    return agg_df, best


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate rule_ranker weights with labels")
    parser.add_argument("--feature_csv", required=True, help="Path to pose_features.csv containing labels")
    parser.add_argument("--label_col", default="label", help="Pose label column (binary 0/1)")
    parser.add_argument(
        "--nanobody_label_threshold",
        type=float,
        default=0.50,
        help="Threshold on nanobody pose-label mean used to derive nanobody_label",
    )
    parser.add_argument("--out_dir", default="calibration_outputs", help="Output directory")

    parser.add_argument("--lower_q", type=float, default=0.01)
    parser.add_argument("--upper_q", type=float, default=0.99)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--conformer_geo_weight", type=float, default=0.15)
    parser.add_argument(
        "--pocket_overwide_penalty_weight",
        type=float,
        default=0.0,
        help="Optional fixed penalty weight for broad pocket definitions; default 0 keeps calibration unchanged",
    )
    parser.add_argument(
        "--pocket_overwide_threshold",
        type=float,
        default=0.55,
        help="Threshold for pocket_shape_overwide_proxy before optional penalty starts",
    )
    parser.add_argument("--consistency_hit_threshold", type=float, default=0.50)

    parser.add_argument("--n_feature_trials", type=int, default=40)
    parser.add_argument("--top_feature_trials_for_agg", type=int, default=8)
    parser.add_argument("--feature_jitter_sigma", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--pose_objective_weight", type=float, default=0.40)
    parser.add_argument("--nanobody_objective_weight", type=float, default=0.60)
    parser.add_argument(
        "--ml_ranking_csv",
        default=None,
        help="Optional nanobody ranking CSV from ML pipeline for consistency-aware calibration",
    )
    parser.add_argument(
        "--ml_score_col",
        default="final_score",
        help="Score column in ml_ranking_csv used for consistency matching",
    )
    parser.add_argument(
        "--rank_consistency_metric",
        choices=["score_spearman", "rank_spearman"],
        default="score_spearman",
        help="Consistency metric against ML reference ranking",
    )
    parser.add_argument(
        "--rank_consistency_weight",
        type=float,
        default=0.40,
        help="Weight of ML consistency term in objective (set to 0 to disable)",
    )
    parser.add_argument(
        "--selection_metric",
        choices=["objective", "nanobody_auc", "rank_consistency"],
        default="objective",
        help="Primary metric used to select best trial (with constraints applied first)",
    )
    parser.add_argument("--min_pose_auc", type=float, default=None, help="Optional minimum pose AUC constraint")
    parser.add_argument("--min_nanobody_auc", type=float, default=None, help="Optional minimum nanobody AUC constraint")
    parser.add_argument(
        "--min_rank_consistency",
        type=float,
        default=None,
        help="Optional minimum rank consistency constraint",
    )

    parser.add_argument("--w_mean_grid", default="0.45,0.50,0.55")
    parser.add_argument("--w_best_grid", default="0.20,0.25,0.30")
    parser.add_argument("--w_consistency_grid", default="0.15,0.20,0.25")
    parser.add_argument("--w_std_penalty_grid", default="0.10,0.15,0.20")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    feature_csv = Path(args.feature_csv).expanduser()
    if not feature_csv.exists():
        raise FileNotFoundError(f"feature_csv not found: {feature_csv}")

    pose_df = pd.read_csv(feature_csv, low_memory=False)
    if args.label_col not in pose_df.columns:
        raise ValueError(f"Label column not found: {args.label_col}")

    available_specs = {
        col: DEFAULT_RULE_FEATURE_SPECS[col]
        for col in DEFAULT_RULE_FEATURE_SPECS
        if col in pose_df.columns
    }
    if not available_specs:
        raise ValueError("No rule features from DEFAULT_RULE_FEATURE_SPECS are available in the feature CSV.")

    # Use one pass to align status filtering behavior before calibration.
    base_pose_rule_df, _ = build_rule_score(
        pose_df,
        feature_specs=available_specs,
        lower_q=float(args.lower_q),
        upper_q=float(args.upper_q),
    )

    label_valid = _to_numeric(base_pose_rule_df[args.label_col]).notna().sum()
    if int(label_valid) < 8:
        raise ValueError(
            f"Not enough valid labeled rows for calibration in column {args.label_col!r}. "
            f"Need >= 8, got {int(label_valid)}."
        )

    nanobody_label_df = _derive_nanobody_labels(
        base_pose_rule_df,
        args.label_col,
        threshold=float(args.nanobody_label_threshold),
    )

    ml_reference_df: pd.DataFrame | None = None
    if args.ml_ranking_csv:
        ml_csv = Path(args.ml_ranking_csv).expanduser()
        if not ml_csv.exists():
            raise FileNotFoundError(f"ml_ranking_csv not found: {ml_csv}")
        ml_df = pd.read_csv(ml_csv, low_memory=False)
        if "nanobody_id" not in ml_df.columns:
            raise ValueError("ml_ranking_csv must contain nanobody_id column")
        if args.ml_score_col not in ml_df.columns:
            raise ValueError(f"ml_ranking_csv missing score column: {args.ml_score_col}")

        ml_reference_df = ml_df[["nanobody_id", args.ml_score_col]].copy()
        ml_reference_df = ml_reference_df.rename(columns={args.ml_score_col: "ml_reference_score"})
        ml_reference_df["ml_reference_score"] = _to_numeric(ml_reference_df["ml_reference_score"])
        ml_reference_df = ml_reference_df.loc[ml_reference_df["ml_reference_score"].notna()].copy()
        ml_reference_df["nanobody_id"] = ml_reference_df["nanobody_id"].astype(str).str.strip()
        ml_reference_df = ml_reference_df.drop_duplicates(subset=["nanobody_id"], keep="first")
        if ml_reference_df.empty:
            raise ValueError("ml_ranking_csv has no valid rows after parsing ml_reference_score")

    requested_rank_weight = float(args.rank_consistency_weight)
    effective_rank_weight = requested_rank_weight
    if ml_reference_df is None and requested_rank_weight > 0.0:
        # Keep default behavior explicit: consistency term only applies when ML reference is provided.
        effective_rank_weight = 0.0
        print(
            "Warning: rank_consistency_weight is set but ml_ranking_csv is missing; "
            "falling back to effective rank_consistency_weight=0.0"
        )

    feature_trials = _build_feature_trials(
        available_specs=available_specs,
        n_trials=max(2, int(args.n_feature_trials)),
        random_seed=int(args.seed),
        jitter_sigma=float(args.feature_jitter_sigma),
    )

    feature_results, feature_trial_df = _run_feature_search(
        pose_df=base_pose_rule_df,
        label_col=str(args.label_col),
        feature_trials=feature_trials,
        lower_q=float(args.lower_q),
        upper_q=float(args.upper_q),
    )

    top_n = max(1, int(args.top_feature_trials_for_agg))
    top_feature_ids = feature_trial_df.head(top_n)["feature_trial_id"].astype(int).tolist()
    selected_feature_results = [fr for fr in feature_results if fr.trial_id in set(top_feature_ids)]

    w_mean_grid = _parse_float_list(args.w_mean_grid, default=[0.45, 0.50, 0.55])
    w_best_grid = _parse_float_list(args.w_best_grid, default=[0.20, 0.25, 0.30])
    w_consistency_grid = _parse_float_list(args.w_consistency_grid, default=[0.15, 0.20, 0.25])
    w_std_grid = _parse_float_list(args.w_std_penalty_grid, default=[0.10, 0.15, 0.20])

    agg_trial_df, best = _run_aggregation_search(
        top_feature_results=selected_feature_results,
        nanobody_label_df=nanobody_label_df,
        ml_reference_df=ml_reference_df,
        rank_consistency_metric=str(args.rank_consistency_metric),
        min_pose_auc=args.min_pose_auc,
        min_nanobody_auc=args.min_nanobody_auc,
        min_rank_consistency=args.min_rank_consistency,
        selection_metric=str(args.selection_metric),
        top_k=int(args.top_k),
        conformer_geo_weight=float(args.conformer_geo_weight),
        pocket_overwide_penalty_weight=float(args.pocket_overwide_penalty_weight),
        pocket_overwide_threshold=float(args.pocket_overwide_threshold),
        w_mean_grid=w_mean_grid,
        w_best_grid=w_best_grid,
        w_consistency_grid=w_consistency_grid,
        w_std_grid=w_std_grid,
        consistency_hit_threshold=float(args.consistency_hit_threshold),
        pose_weight=float(args.pose_objective_weight),
        nanobody_weight=float(args.nanobody_objective_weight),
        rank_weight=float(effective_rank_weight),
    )

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_trial_csv = out_dir / "feature_calibration_trials.csv"
    agg_trial_csv = out_dir / "aggregation_calibration_trials.csv"
    feature_trial_df.to_csv(feature_trial_csv, index=False)
    agg_trial_df.to_csv(agg_trial_csv, index=False)

    calibrated_out_dir = out_dir / "calibrated_rule_outputs"
    conformer_df, nanobody_df = aggregate_rule_scores(
        best["pose_rule_df"],
        top_k=int(args.top_k),
        conformer_geo_weight=float(args.conformer_geo_weight),
        pocket_overwide_penalty_weight=float(args.pocket_overwide_penalty_weight),
        pocket_overwide_threshold=float(args.pocket_overwide_threshold),
        mean_weight=float(best["w_mean"]),
        best_weight=float(best["w_best"]),
        consistency_weight=float(best["w_consistency"]),
        std_penalty_weight=float(best["w_std_penalty"]),
        consistency_hit_threshold=float(args.consistency_hit_threshold),
    )
    ranking_df = rank_nanobodies_by_rules(nanobody_df)
    pose_csv, conformer_csv, ranking_csv = save_rule_outputs(
        pose_rule_df=best["pose_rule_df"],
        conformer_rule_df=conformer_df,
        ranking_df=ranking_df,
        out_dir=calibrated_out_dir,
    )

    config_payload = {
        "input_feature_csv": str(feature_csv),
        "label_column": str(args.label_col),
        "feature_weights": {
            k: {"sign": int(v[0]), "weight": float(v[1])}
            for k, v in best["feature_spec"].items()
        },
        "aggregation_weights": {
            "w_mean": float(best["w_mean"]),
            "w_best": float(best["w_best"]),
            "w_consistency": float(best["w_consistency"]),
            "w_std_penalty": float(best["w_std_penalty"]),
            "conformer_geo_weight": float(args.conformer_geo_weight),
            "pocket_overwide_penalty_weight": float(args.pocket_overwide_penalty_weight),
            "pocket_overwide_threshold": float(args.pocket_overwide_threshold),
            "consistency_hit_threshold": float(args.consistency_hit_threshold),
            "top_k": int(args.top_k),
        },
        "metrics": {
            "best_pose_auc": float(best["pose_auc"]),
            "best_nanobody_auc": float(best["nanobody_auc"]),
            "best_rank_consistency": float(best.get("rank_consistency", float("nan"))),
            "objective": float(best["objective"]),
        },
        "objective_weights": {
            "pose_objective_weight": float(args.pose_objective_weight),
            "nanobody_objective_weight": float(args.nanobody_objective_weight),
            "rank_consistency_weight": float(effective_rank_weight),
            "requested_rank_consistency_weight": float(requested_rank_weight),
            "rank_consistency_metric": str(args.rank_consistency_metric),
            "selection_metric": str(args.selection_metric),
            "min_pose_auc": args.min_pose_auc,
            "min_nanobody_auc": args.min_nanobody_auc,
            "min_rank_consistency": args.min_rank_consistency,
        },
        "ml_reference": {
            "enabled": bool(ml_reference_df is not None),
            "ml_ranking_csv": str(Path(args.ml_ranking_csv).expanduser()) if args.ml_ranking_csv else None,
            "ml_score_col": str(args.ml_score_col),
            "n_rows": int(len(ml_reference_df)) if ml_reference_df is not None else 0,
        },
        "selection_summary": {
            "feasible_count": int(best.get("feasible_count", 0)),
            "total_trial_count": int(best.get("total_trial_count", 0)),
            "used_constraint_fallback": bool(best.get("used_constraint_fallback", False)),
        },
        "nanobody_labeling": {
            "mode": "mean_threshold_with_fallback",
            "threshold": float(args.nanobody_label_threshold),
            "positive_rate": float(nanobody_label_df["nanobody_label"].mean()),
        },
        "search": {
            "n_feature_trials": int(args.n_feature_trials),
            "top_feature_trials_for_agg": int(args.top_feature_trials_for_agg),
            "feature_jitter_sigma": float(args.feature_jitter_sigma),
            "seed": int(args.seed),
            "w_mean_grid": w_mean_grid,
            "w_best_grid": w_best_grid,
            "w_consistency_grid": w_consistency_grid,
            "w_std_penalty_grid": w_std_grid,
        },
        "artifacts": {
            "feature_trial_csv": str(feature_trial_csv),
            "aggregation_trial_csv": str(agg_trial_csv),
            "pose_rule_scores_csv": str(pose_csv),
            "conformer_rule_scores_csv": str(conformer_csv),
            "nanobody_rule_ranking_csv": str(ranking_csv),
        },
    }

    config_json = out_dir / "calibrated_rule_config.json"
    config_json.write_text(json.dumps(config_payload, ensure_ascii=True, indent=2), encoding="utf-8")

    print(f"Calibration done. Best objective={best['objective']:.4f}")
    print(
        "Best pose AUC="
        f"{best['pose_auc']:.4f}, best nanobody AUC={best['nanobody_auc']:.4f}, "
        f"best rank consistency={float(best.get('rank_consistency', float('nan'))):.4f}"
    )
    print(
        "Feasible trials="
        f"{int(best.get('feasible_count', 0))}/{int(best.get('total_trial_count', 0))}, "
        f"constraint_fallback={bool(best.get('used_constraint_fallback', False))}"
    )
    print(f"Saved: {feature_trial_csv}")
    print(f"Saved: {agg_trial_csv}")
    print(f"Saved: {config_json}")
    print(f"Saved calibrated rule outputs under: {calibrated_out_dir}")


if __name__ == "__main__":
    main()
