"""Rule-based pocket-blocking ranking from pose_features.csv.

This module provides a non-ML baseline to validate feature pipeline quality:
1) pose_rule_scores.csv
2) conformer_rule_scores.csv
3) nanobody_rule_ranking.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ranking_common import (
    CONSISTENCY_WEIGHTS,
    GEOMETRY_AUX_WEIGHTS,
    build_blocking_explanation,
    compute_consistency_score,
    compute_weighted_scaled_mean,
    safe_mean_if_exists,
    to_numeric as _to_numeric,
)


REQUIRED_ID_COLUMNS = ["nanobody_id", "conformer_id", "pose_id"]

# sign=+1 means larger is better; sign=-1 means smaller is better.
DEFAULT_RULE_FEATURE_SPECS: dict[str, tuple[int, float]] = {
    "pocket_hit_fraction": (+1, 0.20),
    "catalytic_hit_fraction": (+1, 0.14),
    "mouth_occlusion_score": (+1, 0.14),
    "mouth_axis_block_fraction": (+1, 0.06),
    "mouth_aperture_block_fraction": (+1, 0.06),
    "mouth_min_clearance": (-1, 0.03),
    "delta_pocket_occupancy_proxy": (+1, 0.12),
    "substrate_overlap_score": (+1, 0.10),
    "ligand_path_block_score": (+1, 0.06),
    "ligand_path_block_fraction": (+1, 0.05),
    "ligand_path_bottleneck_score": (+1, 0.04),
    "ligand_path_exit_block_fraction": (+1, 0.04),
    "ligand_path_min_clearance": (-1, 0.03),
    "min_distance_to_pocket": (-1, 0.08),
    "rsite_accuracy": (+1, 0.06),
    "mmgbsa": (-1, 0.05),
    "interface_dg": (-1, 0.03),
}

def _validate_and_clean_pose_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate minimal columns and normalize id/probability/status fields."""
    missing = [c for c in REQUIRED_ID_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required ID columns: {missing}")

    out = df.copy()
    for col in REQUIRED_ID_COLUMNS:
        out[col] = out[col].astype(str).str.strip()

    # Optional status filter if upstream provides it.
    if "status" in out.columns:
        ok_mask = out["status"].astype(str).str.lower().eq("ok")
        if ok_mask.any():
            out = out.loc[ok_mask].copy()

    if "pred_prob" in out.columns:
        out["pred_prob"] = _to_numeric(out["pred_prob"]).clip(0.0, 1.0)

    out = out.reset_index(drop=True)
    if out.empty:
        raise ValueError("Input table is empty after validation/filtering.")

    return out


def _robust_minmax_scale(values: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    """Robustly normalize a feature to [0, 1] by quantile clipping."""
    arr = _to_numeric(values).astype(float)
    out = pd.Series(np.nan, index=arr.index, dtype=float)

    finite_mask = np.isfinite(arr.to_numpy(dtype=np.float64))
    if not finite_mask.any():
        return out

    valid = arr.to_numpy(dtype=np.float64)[finite_mask]
    lq = float(np.clip(lower_q, 0.0, 1.0))
    uq = float(np.clip(upper_q, 0.0, 1.0))
    if lq >= uq:
        lq, uq = 0.0, 1.0

    lo = float(np.nanquantile(valid, lq))
    hi = float(np.nanquantile(valid, uq))
    denom = hi - lo

    if (not np.isfinite(denom)) or (denom < 1e-12):
        out.iloc[np.where(finite_mask)[0]] = 0.5
        return out

    scaled = ((arr - lo) / denom).clip(0.0, 1.0)
    out.iloc[np.where(finite_mask)[0]] = scaled.iloc[np.where(finite_mask)[0]]
    return out


def _top_component_summary(component_scores: dict[str, np.ndarray], row_idx: int, top_k: int = 3) -> str:
    """Create compact per-row component summary string."""
    parts: list[tuple[str, float]] = []
    for name, arr in component_scores.items():
        if row_idx >= arr.shape[0]:
            continue
        val = float(arr[row_idx])
        if np.isfinite(val):
            parts.append((name, val))

    if not parts:
        return ""

    parts.sort(key=lambda x: abs(x[1]), reverse=True)
    selected = parts[: max(1, int(top_k))]
    return ";".join([f"{name}={value:.3f}" for name, value in selected])


def build_rule_score(
    pose_df: pd.DataFrame,
    feature_specs: dict[str, tuple[int, float]] | None = None,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build pose-level rule_blocking_score from geometry and optional energy features.

    Rule formula (per pose, using available features only):
        aligned_f = robust_norm(f) if sign=+1 else 1 - robust_norm(f)
        rule_blocking_score = sum(w_f * aligned_f) / sum(w_f over finite aligned_f)
    """
    df = _validate_and_clean_pose_df(pose_df)
    specs = feature_specs or DEFAULT_RULE_FEATURE_SPECS

    out = df.copy()
    available = [c for c in specs if c in out.columns]
    if not available:
        raise ValueError("No rule features are available in pose table.")

    weighted_values: list[np.ndarray] = []
    weighted_masks: list[np.ndarray] = []
    component_scores: dict[str, np.ndarray] = {}
    used_features: list[str] = []

    for col in available:
        sign, weight = specs[col]
        w = float(max(weight, 0.0))
        if w <= 0.0:
            continue

        norm = _robust_minmax_scale(out[col], lower_q=lower_q, upper_q=upper_q)
        norm_np = norm.to_numpy(dtype=np.float64)
        if sign < 0:
            norm_np = 1.0 - norm_np

        valid = np.isfinite(norm_np)
        if not valid.any():
            continue

        contrib = np.where(valid, norm_np * w, 0.0)
        weighted_values.append(contrib)
        weighted_masks.append(np.where(valid, w, 0.0))
        component_scores[col] = contrib
        used_features.append(col)

    if not weighted_values:
        raise ValueError("All rule features are missing/invalid after normalization.")

    numerator = np.sum(np.vstack(weighted_values), axis=0)
    denominator = np.sum(np.vstack(weighted_masks), axis=0)
    score = np.where(denominator > 0.0, numerator / denominator, np.nan)

    finite_score = score[np.isfinite(score)]
    fallback = float(np.nanmedian(finite_score)) if finite_score.size > 0 else 0.5
    score = np.where(np.isfinite(score), score, fallback)
    score = np.clip(score, 0.0, 1.0)

    out["rule_blocking_score"] = score
    out["rule_used_feature_count"] = np.sum(np.vstack(weighted_masks) > 0.0, axis=0).astype(int)
    out["rule_components"] = [
        _top_component_summary(component_scores=component_scores, row_idx=i, top_k=3)
        for i in range(len(out))
    ]

    info = {
        "used_features": used_features,
        "feature_count": int(len(used_features)),
        "score_min": float(np.nanmin(score)) if len(score) > 0 else float("nan"),
        "score_p50": float(np.nanpercentile(score, 50)) if len(score) > 0 else float("nan"),
        "score_p90": float(np.nanpercentile(score, 90)) if len(score) > 0 else float("nan"),
        "score_max": float(np.nanmax(score)) if len(score) > 0 else float("nan"),
    }
    return out, info


def compute_pocket_consistency_score(
    conformer_group: pd.DataFrame,
    hit_threshold: float = 0.50,
) -> float:
    """Compute cross-conformer consistency score from geometric top-k means."""
    return compute_consistency_score(
        conformer_group,
        weights=CONSISTENCY_WEIGHTS,
        hit_threshold=hit_threshold,
    )


def build_explanation_text(row: pd.Series, std_ref: float = np.nan) -> str:
    """Build explanation text for nanobody rule ranking."""
    return build_blocking_explanation(
        row,
        std_ref=std_ref,
        fallback_reason="规则评分较高，建议进入结构复核",
    )


def aggregate_rule_scores(
    pose_rule_df: pd.DataFrame,
    top_k: int = 3,
    conformer_geo_weight: float = 0.15,
    mean_weight: float = 0.50,
    best_weight: float = 0.25,
    consistency_weight: float = 0.20,
    std_penalty_weight: float = 0.15,
    consistency_hit_threshold: float = 0.50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate pose rules into conformer and nanobody rule scores.

    Conformer score:
        conformer_rule_score = (1 - w_geo) * (0.7 * mean_topk_rule + 0.3 * best_pose_rule)
                            + w_geo * geo_aux_score

    Nanobody final score:
        final_score = w1 * mean_conformer_score
                    + w2 * best_conformer_score
                    + w3 * pocket_consistency_score
                    - w4 * std_conformer_score
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if pose_rule_df.empty:
        raise ValueError("pose_rule_df is empty.")
    if "rule_blocking_score" not in pose_rule_df.columns:
        raise ValueError("rule_blocking_score column is missing. Run build_rule_score first.")

    df = _validate_and_clean_pose_df(pose_rule_df)
    df["rule_blocking_score"] = _to_numeric(df["rule_blocking_score"]).clip(0.0, 1.0)
    df = df.loc[df["rule_blocking_score"].notna()].reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid rows after parsing rule_blocking_score.")

    # Global robust scaling for optional geometry auxiliary fusion.
    scaled_cols: dict[str, str] = {}
    for col in GEOMETRY_AUX_WEIGHTS:
        if col in df.columns:
            sc = f"__scaled_{col}"
            df[sc] = _robust_minmax_scale(df[col])
            scaled_cols[col] = sc

    conformer_rows: list[dict[str, Any]] = []
    for (nanobody_id, conformer_id), g in df.groupby(["nanobody_id", "conformer_id"], sort=False):
        g_sorted = g.sort_values(by="rule_blocking_score", ascending=False).reset_index(drop=True)
        k = min(int(top_k), len(g_sorted))
        top = g_sorted.iloc[:k]
        best = top.iloc[0]

        mean_topk_rule = safe_mean_if_exists(top, "rule_blocking_score")
        best_pose_rule = float(best["rule_blocking_score"])
        best_pose_prob = float(best.get("pred_prob", np.nan)) if "pred_prob" in best.index else float("nan")

        row: dict[str, Any] = {
            "nanobody_id": str(nanobody_id),
            "conformer_id": str(conformer_id),
            "num_poses": int(len(g_sorted)),
            "top_k_used": int(k),
            "topk_pose_ids": "|".join(map(str, top["pose_id"].tolist())),
            "best_pose_id": str(best["pose_id"]),
            "best_pose_prob": best_pose_prob,
            "best_pose_rule_score": best_pose_rule,
            "mean_topk_rule_blocking_score": mean_topk_rule,
            "mean_topk_pred_prob": safe_mean_if_exists(top, "pred_prob"),
        }

        for col in [
            "pocket_hit_fraction",
            "catalytic_hit_fraction",
            "mouth_occlusion_score",
            "mouth_axis_block_fraction",
            "mouth_aperture_block_fraction",
            "mouth_min_clearance",
            "substrate_overlap_score",
            "ligand_path_block_score",
            "ligand_path_block_fraction",
            "ligand_path_bottleneck_score",
            "ligand_path_exit_block_fraction",
            "ligand_path_min_clearance",
            "delta_pocket_occupancy_proxy",
            "rsite_accuracy",
            "min_distance_to_pocket",
            "mmgbsa",
            "interface_dg",
        ]:
            row[f"mean_topk_{col}"] = safe_mean_if_exists(top, col)

        geo_aux_score = compute_weighted_scaled_mean(
            top,
            scaled_columns=scaled_cols,
            weights=GEOMETRY_AUX_WEIGHTS,
        )

        # rule score stays dominant; geometry acts as auxiliary correction.
        w_geo = float(np.clip(conformer_geo_weight, 0.0, 0.40)) if np.isfinite(geo_aux_score) else 0.0
        base_rule = 0.70 * float(mean_topk_rule) + 0.30 * float(best_pose_rule)
        conformer_rule_score = (1.0 - w_geo) * base_rule + w_geo * (float(geo_aux_score) if np.isfinite(geo_aux_score) else 0.0)

        row["geo_aux_score"] = geo_aux_score
        row["score_weight_geo_aux"] = w_geo
        row["conformer_rule_score"] = float(np.clip(conformer_rule_score, 0.0, 1.0))

        # Aliases matching existing ML ranking naming.
        row["conformer_score"] = row["conformer_rule_score"]
        row["mean_topk_pocket_hit_fraction"] = row.get("mean_topk_pocket_hit_fraction", np.nan)
        row["mean_topk_catalytic_hit_fraction"] = row.get("mean_topk_catalytic_hit_fraction", np.nan)
        row["mean_topk_mouth_occlusion_score"] = row.get("mean_topk_mouth_occlusion_score", np.nan)
        row["mean_topk_mouth_aperture_block_fraction"] = row.get("mean_topk_mouth_aperture_block_fraction", np.nan)
        row["mean_topk_substrate_overlap_score"] = row.get("mean_topk_substrate_overlap_score", np.nan)
        row["mean_topk_ligand_path_exit_block_fraction"] = row.get("mean_topk_ligand_path_exit_block_fraction", np.nan)

        conformer_rows.append(row)

    conformer_df = pd.DataFrame(conformer_rows)
    if conformer_df.empty:
        raise ValueError("Failed to aggregate conformer rule scores.")

    conformer_df = conformer_df.sort_values(
        by=["nanobody_id", "conformer_rule_score", "best_pose_rule_score"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    w1 = float(max(mean_weight, 0.0))
    w2 = float(max(best_weight, 0.0))
    w3 = float(max(consistency_weight, 0.0))
    w4 = float(max(std_penalty_weight, 0.0))

    nanobody_rows: list[dict[str, Any]] = []
    for nanobody_id, g in conformer_df.groupby("nanobody_id", sort=False):
        g2 = g.sort_values(by=["conformer_rule_score", "best_pose_rule_score"], ascending=[False, False]).reset_index(drop=True)
        conf_scores = _to_numeric(g2["conformer_rule_score"])

        mean_conformer = float(conf_scores.mean()) if conf_scores.notna().any() else float("nan")
        std_conformer = float(conf_scores.std(ddof=0)) if conf_scores.notna().any() else float("nan")
        if not np.isfinite(std_conformer):
            std_conformer = 0.0

        best = g2.iloc[0]
        best_conformer_score = float(best["conformer_rule_score"])
        best_pose_rule = float(best["best_pose_rule_score"])

        pocket_consistency = compute_pocket_consistency_score(
            g2,
            hit_threshold=float(consistency_hit_threshold),
        )
        consistency_safe = float(pocket_consistency) if np.isfinite(pocket_consistency) else 0.0

        final_score = (
            w1 * mean_conformer
            + w2 * best_conformer_score
            + w3 * consistency_safe
            - w4 * std_conformer
        )

        row = {
            "nanobody_id": str(nanobody_id),
            "num_conformers": int(len(g2)),
            "best_conformer": str(best["conformer_id"]),
            "best_pose_id": str(best["best_pose_id"]),
            "best_pose_prob": float(best.get("best_pose_prob", np.nan)),
            "best_pose_rule_score": best_pose_rule,
            "mean_conformer_rule_score": mean_conformer,
            "best_conformer_rule_score": best_conformer_score,
            "std_conformer_rule_score": std_conformer,
            "pocket_consistency_score": pocket_consistency,
            "final_rule_score": float(final_score),
            "w_mean": w1,
            "w_best": w2,
            "w_consistency": w3,
            "w_std_penalty": w4,
        }

        for col in [
            "mean_topk_pocket_hit_fraction",
            "mean_topk_catalytic_hit_fraction",
            "mean_topk_mouth_occlusion_score",
            "mean_topk_mouth_axis_block_fraction",
            "mean_topk_mouth_aperture_block_fraction",
            "mean_topk_mouth_min_clearance",
            "mean_topk_substrate_overlap_score",
            "mean_topk_ligand_path_block_score",
            "mean_topk_ligand_path_block_fraction",
            "mean_topk_ligand_path_bottleneck_score",
            "mean_topk_ligand_path_exit_block_fraction",
            "mean_topk_ligand_path_min_clearance",
            "mean_topk_delta_pocket_occupancy_proxy",
            "mean_topk_rsite_accuracy",
        ]:
            row[col] = safe_mean_if_exists(g2, col)

        # Aliases with ML-style naming for downstream compatibility.
        row["mean_conformer_score"] = row["mean_conformer_rule_score"]
        row["best_conformer_score"] = row["best_conformer_rule_score"]
        row["std_conformer_score"] = row["std_conformer_rule_score"]
        row["final_score"] = row["final_rule_score"]

        nanobody_rows.append(row)

    nanobody_df = pd.DataFrame(nanobody_rows)
    if nanobody_df.empty:
        raise ValueError("Failed to aggregate nanobody rule scores.")

    std_ref = float(np.nanmedian(_to_numeric(nanobody_df["std_conformer_score"]).to_numpy(dtype=float))) if len(nanobody_df) > 0 else np.nan
    nanobody_df["explanation"] = nanobody_df.apply(lambda r: build_explanation_text(r, std_ref=std_ref), axis=1)

    return conformer_df, nanobody_df


def rank_nanobodies_by_rules(nanobody_rule_df: pd.DataFrame) -> pd.DataFrame:
    """Rank nanobodies by rule-based final score."""
    if nanobody_rule_df.empty:
        raise ValueError("nanobody_rule_df is empty.")

    score_col = "final_rule_score" if "final_rule_score" in nanobody_rule_df.columns else "final_score"
    if score_col not in nanobody_rule_df.columns:
        raise ValueError("No final score column found in nanobody_rule_df.")

    ranked = nanobody_rule_df.sort_values(
        by=[score_col, "best_conformer_rule_score", "best_pose_rule_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1, dtype=int))

    preferred_cols = [
        "rank",
        "nanobody_id",
        "num_conformers",
        "best_conformer",
        "best_pose_id",
        "best_pose_rule_score",
        "best_pose_prob",
        "mean_conformer_score",
        "best_conformer_score",
        "std_conformer_score",
        "pocket_consistency_score",
        "final_score",
        "explanation",
    ]
    remaining = [c for c in ranked.columns if c not in preferred_cols]
    return ranked[preferred_cols + remaining]


def save_rule_outputs(
    pose_rule_df: pd.DataFrame,
    conformer_rule_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    out_dir: str | Path = ".",
) -> tuple[str, str, str]:
    """Save all rule-ranking artifacts to CSV."""
    out_path = Path(out_dir).expanduser()
    out_path.mkdir(parents=True, exist_ok=True)

    pose_csv = out_path / "pose_rule_scores.csv"
    conformer_csv = out_path / "conformer_rule_scores.csv"
    ranking_csv = out_path / "nanobody_rule_ranking.csv"

    pose_rule_df.to_csv(pose_csv, index=False)
    conformer_rule_df.to_csv(conformer_csv, index=False)
    ranking_df.to_csv(ranking_csv, index=False)
    return str(pose_csv), str(conformer_csv), str(ranking_csv)


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for rule ranker."""
    parser = argparse.ArgumentParser(description="Rule-based pocket-blocking ranker")
    parser.add_argument("--feature_csv", required=True, help="Path to pose_features.csv")
    parser.add_argument("--out_dir", default="rule_outputs", help="Directory for output CSVs")

    parser.add_argument("--top_k", type=int, default=3, help="Top-k poses per conformer")
    parser.add_argument("--lower_q", type=float, default=0.01, help="Lower quantile for robust normalization")
    parser.add_argument("--upper_q", type=float, default=0.99, help="Upper quantile for robust normalization")

    parser.add_argument(
        "--conformer_geo_weight",
        type=float,
        default=0.15,
        help="Geometry auxiliary weight in conformer aggregation",
    )

    parser.add_argument("--w_mean", type=float, default=0.50, help="Weight for mean conformer score")
    parser.add_argument("--w_best", type=float, default=0.25, help="Weight for best conformer score")
    parser.add_argument("--w_consistency", type=float, default=0.20, help="Weight for pocket consistency score")
    parser.add_argument("--w_std_penalty", type=float, default=0.15, help="Penalty weight for score std")
    parser.add_argument(
        "--consistency_hit_threshold",
        type=float,
        default=0.50,
        help="Hit threshold used by pocket consistency computation",
    )
    return parser


def main() -> None:
    """CLI entry point."""
    args = _build_parser().parse_args()

    feature_csv = Path(args.feature_csv).expanduser()
    if not feature_csv.exists():
        raise FileNotFoundError(f"feature_csv not found: {feature_csv}")

    pose_df = pd.read_csv(feature_csv, low_memory=False)
    pose_rule_df, info = build_rule_score(
        pose_df,
        feature_specs=DEFAULT_RULE_FEATURE_SPECS,
        lower_q=float(args.lower_q),
        upper_q=float(args.upper_q),
    )

    conformer_rule_df, nanobody_rule_df = aggregate_rule_scores(
        pose_rule_df,
        top_k=int(args.top_k),
        conformer_geo_weight=float(args.conformer_geo_weight),
        mean_weight=float(args.w_mean),
        best_weight=float(args.w_best),
        consistency_weight=float(args.w_consistency),
        std_penalty_weight=float(args.w_std_penalty),
        consistency_hit_threshold=float(args.consistency_hit_threshold),
    )

    ranking_df = rank_nanobodies_by_rules(nanobody_rule_df)
    pose_csv, conformer_csv, ranking_csv = save_rule_outputs(
        pose_rule_df,
        conformer_rule_df,
        ranking_df,
        out_dir=args.out_dir,
    )

    print(f"Rule features used: {', '.join(info.get('used_features', []))}")
    print(f"Saved pose-level rule scores: {pose_csv}")
    print(f"Saved conformer-level rule scores: {conformer_csv}")
    print(f"Saved nanobody rule ranking: {ranking_csv}")
    print(f"Total nanobodies ranked: {len(ranking_df)}")


__all__ = [
    "build_rule_score",
    "aggregate_rule_scores",
    "rank_nanobodies_by_rules",
    "compute_pocket_consistency_score",
    "build_explanation_text",
    "save_rule_outputs",
]


if __name__ == "__main__":
    main()
