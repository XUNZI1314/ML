"""Aggregate pose predictions into conformer and nanobody rankings.

This module reads pose-level predictions and produces:
1) conformer_scores.csv
2) nanobody_ranking.csv
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
    apply_pocket_overwide_penalty,
    build_blocking_explanation,
    compute_consistency_score,
    compute_weighted_scaled_mean,
    safe_mean_if_exists,
    to_numeric as _to_numeric,
)


REQUIRED_COLUMNS = ["nanobody_id", "conformer_id", "pose_id", "pred_prob"]

GEOMETRY_COLUMNS = list(GEOMETRY_AUX_WEIGHTS)


def _robust_scale(values: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    """Robustly scale values into [0, 1] using quantile bounds."""
    arr = _to_numeric(values).astype(float)
    out = pd.Series(np.nan, index=arr.index, dtype=float)

    mask = np.isfinite(arr.to_numpy())
    if not mask.any():
        return out

    valid = arr.to_numpy()[mask]
    lq = float(np.clip(lower_q, 0.0, 1.0))
    uq = float(np.clip(upper_q, 0.0, 1.0))
    if lq >= uq:
        lq, uq = 0.0, 1.0

    lo = float(np.nanquantile(valid, lq))
    hi = float(np.nanquantile(valid, uq))
    denom = hi - lo

    if (not np.isfinite(denom)) or (denom < 1e-12):
        out.iloc[np.where(mask)[0]] = 0.5
        return out

    scaled = ((arr - lo) / denom).clip(0.0, 1.0)
    out.iloc[np.where(mask)[0]] = scaled.iloc[np.where(mask)[0]]
    return out


def _validate_pose_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate input prediction table and normalize key columns."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in pose prediction table: {missing}")

    out = df.copy()
    for col in ["nanobody_id", "conformer_id", "pose_id"]:
        out[col] = out[col].astype(str).str.strip()

    out["pred_prob"] = _to_numeric(out["pred_prob"]).clip(0.0, 1.0)
    out = out.loc[out["pred_prob"].notna()].reset_index(drop=True)
    if out.empty:
        raise ValueError("No valid rows left after parsing pred_prob.")

    # Optional status filtering when provided by upstream pipeline.
    if "status" in out.columns:
        mask_ok = out["status"].astype(str).str.lower().eq("ok")
        if mask_ok.any():
            out = out.loc[mask_ok].reset_index(drop=True)

    return out


def aggregate_conformer_scores(
    pose_df: pd.DataFrame,
    top_k: int = 3,
    blend_optional: bool = True,
    optional_weight: float = 0.15,
    pocket_overwide_penalty_weight: float = 0.0,
    pocket_overwide_threshold: float = 0.55,
) -> pd.DataFrame:
    """Aggregate pose-level predictions into conformer-level scores.

    For each (nanobody_id, conformer_id):
    1) sort poses by pred_prob descending
    2) take top-k poses
    3) compute top-k summary stats
    4) compute conformer_score with interpretable fusion:

    conformer_score = w_pred_mean * mean_topk_pred_prob
                    + w_pred_best * best_pose_prob
                    + w_geo * geo_aux_score

    where:
    - pred terms remain dominant (w_pred_mean + w_pred_best = 1 - w_geo)
    - geo_aux_score is weighted geometric evidence from
      pocket/catalytic/mouth/substrate signals (robust-scaled globally)
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    df = _validate_pose_df(pose_df)

    # Robust scaling keeps optional geometric columns comparable across proteins.
    scaled_cols: dict[str, str] = {}
    for col in GEOMETRY_COLUMNS:
        if col in df.columns:
            sc = f"__scaled_{col}"
            df[sc] = _robust_scale(df[col])
            scaled_cols[col] = sc

    # Additional optional columns for interpretation; not all participate in conformer_score.
    extra_optional_cols = [
        c
        for c in [
            "mouth_axis_block_fraction",
            "mouth_min_clearance",
            "ligand_path_block_score",
            "ligand_path_block_fraction",
            "ligand_path_bottleneck_score",
            "ligand_path_min_clearance",
            "ligand_path_candidate_count",
            "delta_pocket_occupancy_proxy",
            "pocket_block_volume_proxy",
            "pocket_shape_residue_count",
            "pocket_shape_overwide_proxy",
            "pocket_shape_tightness_proxy",
        ]
        if c in df.columns
    ]

    rows: list[dict[str, Any]] = []
    for (nanobody_id, conformer_id), g in df.groupby(["nanobody_id", "conformer_id"], sort=False):
        g_sorted = g.sort_values(by="pred_prob", ascending=False).reset_index(drop=True)
        k = min(int(top_k), len(g_sorted))
        top = g_sorted.iloc[:k]
        best_pose = top.iloc[0]

        mean_topk_pred_prob = safe_mean_if_exists(top, "pred_prob")
        best_pose_prob = float(best_pose["pred_prob"])

        row: dict[str, Any] = {
            "nanobody_id": str(nanobody_id),
            "conformer_id": str(conformer_id),
            "num_poses": int(len(g_sorted)),
            "top_k_used": int(k),
            "topk_pose_ids": "|".join(map(str, top["pose_id"].tolist())),
            "best_pose_id": str(best_pose["pose_id"]),
            "best_pose_prob": best_pose_prob,
            "mean_topk_pred_prob": mean_topk_pred_prob,
        }

        for col in GEOMETRY_COLUMNS:
            row[f"mean_topk_{col}"] = safe_mean_if_exists(top, col)

        for col in extra_optional_cols:
            row[f"mean_topk_{col}"] = safe_mean_if_exists(top, col)

        # Build geometric auxiliary score from scaled top-k means.
        geo_aux_score = compute_weighted_scaled_mean(
            top,
            scaled_columns=scaled_cols,
            weights=GEOMETRY_AUX_WEIGHTS,
        )

        # Preserve backward-compatible naming from first-round baseline.
        row["mean_pocket_hit_fraction"] = row.get("mean_topk_pocket_hit_fraction", np.nan)
        row["mean_catalytic_hit_fraction"] = row.get("mean_topk_catalytic_hit_fraction", np.nan)
        row["mean_mouth_occlusion_score"] = row.get("mean_topk_mouth_occlusion_score", np.nan)
        row["mean_substrate_overlap_score"] = row.get("mean_topk_substrate_overlap_score", np.nan)

        # Pred remains dominant; geometry acts as auxiliary correction.
        w_geo = float(np.clip(optional_weight, 0.0, 0.40)) if blend_optional and np.isfinite(geo_aux_score) else 0.0
        pred_mass = 1.0 - w_geo
        w_pred_mean = 0.70 * pred_mass
        w_pred_best = 0.30 * pred_mass

        conformer_score_raw = (
            w_pred_mean * float(mean_topk_pred_prob)
            + w_pred_best * float(best_pose_prob)
            + w_geo * (float(geo_aux_score) if np.isfinite(geo_aux_score) else 0.0)
        )
        conformer_score, overwide_penalty, overwide_weight = apply_pocket_overwide_penalty(
            conformer_score_raw,
            row.get("mean_topk_pocket_shape_overwide_proxy"),
            penalty_weight=float(pocket_overwide_penalty_weight),
            threshold=float(pocket_overwide_threshold),
        )

        row["optional_signal"] = geo_aux_score
        row["geo_aux_score"] = geo_aux_score
        row["score_weight_pred_mean"] = w_pred_mean
        row["score_weight_best_pose"] = w_pred_best
        row["score_weight_geo_aux"] = w_geo
        row["score_before_pocket_overwide_penalty"] = float(np.clip(conformer_score_raw, 0.0, 1.0))
        row["pocket_overwide_penalty"] = overwide_penalty
        row["score_weight_pocket_overwide_penalty"] = overwide_weight
        row["conformer_score"] = float(np.clip(conformer_score, 0.0, 1.0))
        rows.append(row)

    conformer_df = pd.DataFrame(rows)
    if conformer_df.empty:
        raise ValueError("Failed to aggregate any conformer score.")

    return conformer_df.sort_values(
        by=["nanobody_id", "conformer_score", "best_pose_prob"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def compute_pocket_consistency_score(
    conformer_group: pd.DataFrame,
    hit_threshold: float = 0.50,
) -> float:
    """Compute cross-conformer consistency for pocket-blocking related signals."""
    return compute_consistency_score(
        conformer_group,
        weights=CONSISTENCY_WEIGHTS,
        hit_threshold=hit_threshold,
    )


def build_explanation_text(row: pd.Series, std_ref: float = np.nan) -> str:
    """Build human-readable explanation from actual aggregated metrics."""
    return build_blocking_explanation(
        row,
        std_ref=std_ref,
        fallback_reason="综合评分较高，建议优先进入后续验证",
    )


def aggregate_nanobody_scores(
    conformer_df: pd.DataFrame,
    alpha: float = 0.15,
    mean_weight: float = 0.50,
    best_weight: float = 0.25,
    consistency_weight: float = 0.20,
    std_penalty_weight: float | None = None,
    consistency_hit_threshold: float = 0.50,
) -> pd.DataFrame:
    """Aggregate conformer-level scores into nanobody-level final_score.

    final_score = w1 * mean_conformer_score
                + w2 * best_conformer_score
                + w3 * pocket_consistency_score
                - w4 * std_conformer_score

    All weights are configurable; alpha is retained for backward compatibility
    and acts as default w4 when std_penalty_weight is not explicitly provided.
    """
    if conformer_df.empty:
        raise ValueError("conformer_df is empty.")

    required = {"nanobody_id", "conformer_id", "conformer_score", "best_pose_id", "best_pose_prob"}
    missing = [c for c in required if c not in conformer_df.columns]
    if missing:
        raise ValueError(f"Missing required conformer columns: {missing}")

    w1 = float(max(mean_weight, 0.0))
    w2 = float(max(best_weight, 0.0))
    w3 = float(max(consistency_weight, 0.0))
    w4 = float(max(alpha if std_penalty_weight is None else std_penalty_weight, 0.0))

    rows: list[dict[str, Any]] = []
    for nanobody_id, g in conformer_df.groupby("nanobody_id", sort=False):
        g2 = g.sort_values(by=["conformer_score", "best_pose_prob"], ascending=[False, False]).reset_index(drop=True)
        scores = _to_numeric(g2["conformer_score"])

        mean_score = float(scores.mean()) if scores.notna().any() else float("nan")
        std_score = float(scores.std(ddof=0)) if scores.notna().any() else float("nan")
        if not np.isfinite(std_score):
            std_score = 0.0

        best = g2.iloc[0]
        best_score = float(best["conformer_score"])
        consistency_score = compute_pocket_consistency_score(
            g2,
            hit_threshold=float(consistency_hit_threshold),
        )
        consistency_safe = float(consistency_score) if np.isfinite(consistency_score) else 0.0

        final_score = w1 * mean_score + w2 * best_score + w3 * consistency_safe - w4 * std_score

        row: dict[str, Any] = {
            "nanobody_id": str(nanobody_id),
            "num_conformers": int(len(g2)),
            "best_conformer": str(best["conformer_id"]),
            "best_pose_id": str(best["best_pose_id"]),
            "best_pose_prob": float(best["best_pose_prob"]),
            "mean_conformer_score": mean_score,
            "best_conformer_score": best_score,
            "std_conformer_score": std_score,
            "pocket_consistency_score": consistency_score,
            "final_score": float(final_score),
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
            "mean_topk_pocket_block_volume_proxy",
            "mean_topk_pocket_shape_residue_count",
            "mean_topk_pocket_shape_overwide_proxy",
            "mean_topk_pocket_shape_tightness_proxy",
            "pocket_overwide_penalty",
            "score_weight_pocket_overwide_penalty",
        ]:
            row[col] = safe_mean_if_exists(g2, col)

        # Backward-compatible aliases.
        row["mean_pocket_hit_fraction"] = row["mean_topk_pocket_hit_fraction"]
        row["mean_catalytic_hit_fraction"] = row["mean_topk_catalytic_hit_fraction"]
        row["mean_mouth_occlusion_score"] = row["mean_topk_mouth_occlusion_score"]
        row["mean_substrate_overlap_score"] = row["mean_topk_substrate_overlap_score"]

        rows.append(row)

    nb_df = pd.DataFrame(rows)
    std_ref = float(np.nanmedian(_to_numeric(nb_df["std_conformer_score"]).to_numpy(dtype=float))) if len(nb_df) > 0 else np.nan
    nb_df["explanation"] = nb_df.apply(lambda r: build_explanation_text(r, std_ref=std_ref), axis=1)
    return nb_df


def rank_nanobodies(nanobody_df: pd.DataFrame) -> pd.DataFrame:
    """Rank nanobodies by final_score descending and assign rank column."""
    if nanobody_df.empty:
        raise ValueError("nanobody_df is empty.")
    if "final_score" not in nanobody_df.columns:
        raise ValueError("final_score column is missing.")

    ranked = nanobody_df.sort_values(
        by=["final_score", "best_conformer_score", "best_pose_prob"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1, dtype=int))

    preferred_cols = [
        "rank",
        "nanobody_id",
        "num_conformers",
        "best_conformer",
        "best_pose_id",
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


def save_ranking_outputs(
    conformer_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    out_dir: str | Path = ".",
) -> tuple[str, str]:
    """Save conformer and nanobody ranking outputs to CSV files."""
    out_path = Path(out_dir).expanduser()
    out_path.mkdir(parents=True, exist_ok=True)

    conformer_csv = out_path / "conformer_scores.csv"
    ranking_csv = out_path / "nanobody_ranking.csv"

    conformer_df.to_csv(conformer_csv, index=False)
    ranking_df.to_csv(ranking_csv, index=False)
    return str(conformer_csv), str(ranking_csv)


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Aggregate and rank nanobody predictions")
    parser.add_argument("--pred_csv", required=True, help="Path to pose_predictions.csv")
    parser.add_argument("--out_dir", default=".", help="Directory to save ranking outputs")
    parser.add_argument("--top_k", type=int, default=3, help="Top-k poses per conformer")

    # Conformer fusion controls.
    parser.add_argument(
        "--optional_weight",
        type=float,
        default=0.15,
        help="Geometry auxiliary weight in conformer score (0~0.40 recommended)",
    )
    parser.add_argument(
        "--disable_optional_blend",
        action="store_true",
        help="Disable geometry auxiliary blending in conformer score",
    )
    parser.add_argument(
        "--pocket_overwide_penalty_weight",
        type=float,
        default=0.0,
        help="Optional penalty weight for broad pocket definitions; default 0 keeps ranking unchanged",
    )
    parser.add_argument(
        "--pocket_overwide_threshold",
        type=float,
        default=0.55,
        help="Threshold for pocket_shape_overwide_proxy before optional penalty starts",
    )

    # Final nanobody score controls.
    parser.add_argument("--alpha", type=float, default=0.15, help="Backward-compatible std penalty weight")
    parser.add_argument("--w_mean", type=float, default=0.50, help="w1 for mean_conformer_score")
    parser.add_argument("--w_best", type=float, default=0.25, help="w2 for best_conformer_score")
    parser.add_argument("--w_consistency", type=float, default=0.20, help="w3 for pocket_consistency_score")
    parser.add_argument(
        "--w_std_penalty",
        type=float,
        default=None,
        help="w4 for std_conformer_score (if omitted, fall back to --alpha)",
    )
    parser.add_argument(
        "--consistency_hit_threshold",
        type=float,
        default=0.50,
        help="Hit threshold used in pocket_consistency_score computation",
    )
    return parser


def main() -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    pred_csv = Path(args.pred_csv).expanduser()
    if not pred_csv.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {pred_csv}")

    pose_df = pd.read_csv(pred_csv, low_memory=False)

    conformer_df = aggregate_conformer_scores(
        pose_df,
        top_k=int(args.top_k),
        blend_optional=not bool(args.disable_optional_blend),
        optional_weight=float(args.optional_weight),
        pocket_overwide_penalty_weight=float(args.pocket_overwide_penalty_weight),
        pocket_overwide_threshold=float(args.pocket_overwide_threshold),
    )

    nanobody_df = aggregate_nanobody_scores(
        conformer_df,
        alpha=float(args.alpha),
        mean_weight=float(args.w_mean),
        best_weight=float(args.w_best),
        consistency_weight=float(args.w_consistency),
        std_penalty_weight=args.w_std_penalty,
        consistency_hit_threshold=float(args.consistency_hit_threshold),
    )
    ranking_df = rank_nanobodies(nanobody_df)

    conformer_csv, ranking_csv = save_ranking_outputs(conformer_df, ranking_df, out_dir=args.out_dir)
    print(f"Saved conformer scores: {conformer_csv}")
    print(f"Saved nanobody ranking: {ranking_csv}")
    print(f"Total nanobodies ranked: {len(ranking_df)}")


__all__ = [
    "safe_mean_if_exists",
    "compute_pocket_consistency_score",
    "build_explanation_text",
    "aggregate_conformer_scores",
    "aggregate_nanobody_scores",
    "rank_nanobodies",
    "save_ranking_outputs",
]


if __name__ == "__main__":
    main()
