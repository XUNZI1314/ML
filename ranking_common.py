"""Shared ranking helpers used by rule and ML aggregation pipelines."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


GEOMETRY_AUX_WEIGHTS: dict[str, float] = {
    "pocket_hit_fraction": 0.34,
    "catalytic_hit_fraction": 0.25,
    "mouth_occlusion_score": 0.16,
    "mouth_aperture_block_fraction": 0.13,
    "substrate_overlap_score": 0.07,
    "ligand_path_exit_block_fraction": 0.05,
}

CONSISTENCY_WEIGHTS: dict[str, float] = {
    "mean_topk_pocket_hit_fraction": 0.34,
    "mean_topk_catalytic_hit_fraction": 0.26,
    "mean_topk_mouth_occlusion_score": 0.16,
    "mean_topk_mouth_aperture_block_fraction": 0.14,
    "mean_topk_substrate_overlap_score": 0.05,
    "mean_topk_ligand_path_exit_block_fraction": 0.05,
}


def to_numeric(series: pd.Series) -> pd.Series:
    """Convert a Series to numeric with coercion."""
    return pd.to_numeric(series, errors="coerce")


def safe_mean_if_exists(df: pd.DataFrame, column: str) -> float:
    """Safely compute numeric mean for an optional column."""
    if column not in df.columns:
        return float("nan")
    vals = to_numeric(df[column])
    return float(vals.mean()) if vals.notna().any() else float("nan")


def compute_weighted_scaled_mean(
    df: pd.DataFrame,
    scaled_columns: Mapping[str, str],
    weights: Mapping[str, float],
) -> float:
    """Compute weighted mean from globally scaled optional columns."""
    numerator = 0.0
    denominator = 0.0
    for feature_name, weight in weights.items():
        scaled_col = scaled_columns.get(feature_name)
        if scaled_col is None:
            continue
        scaled_mean = safe_mean_if_exists(df, scaled_col)
        if not np.isfinite(scaled_mean):
            continue
        numerator += float(weight) * float(scaled_mean)
        denominator += float(weight)
    return float(numerator / denominator) if denominator > 0.0 else float("nan")


def compute_pocket_overwide_penalty(
    overwide_proxy: Any,
    threshold: float = 0.55,
) -> float:
    """Map a pocket overwide proxy to a normalized penalty severity.

    The returned value is only a severity in [0, 1]. Callers decide whether to
    apply it to scores. This keeps the default ranking behavior unchanged.
    """
    try:
        value = float(overwide_proxy)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(value):
        return 0.0

    try:
        threshold_value = float(threshold)
    except (TypeError, ValueError):
        threshold_value = 0.55
    if not np.isfinite(threshold_value):
        threshold_value = 0.55
    thr = float(np.clip(threshold_value, 0.0, 0.99))
    if value <= thr:
        return 0.0
    return float(np.clip((value - thr) / max(1.0 - thr, 1e-6), 0.0, 1.0))


def apply_pocket_overwide_penalty(
    score: Any,
    overwide_proxy: Any,
    penalty_weight: float = 0.0,
    threshold: float = 0.55,
) -> tuple[float, float, float]:
    """Apply optional pocket-overwide penalty to a score.

    Returns:
    - adjusted score
    - normalized penalty severity
    - sanitized penalty weight
    """
    try:
        base = float(score)
    except (TypeError, ValueError):
        base = float("nan")
    if not np.isfinite(base):
        base = 0.0

    try:
        weight_value = float(penalty_weight)
    except (TypeError, ValueError):
        weight_value = 0.0
    if not np.isfinite(weight_value):
        weight_value = 0.0
    weight = float(np.clip(weight_value, 0.0, 1.0))
    penalty = compute_pocket_overwide_penalty(overwide_proxy, threshold=threshold)
    adjusted = float(np.clip(base - weight * penalty, 0.0, 1.0))
    return adjusted, penalty, weight


def compute_consistency_score(
    conformer_group: pd.DataFrame,
    weights: Mapping[str, float] | None = None,
    hit_threshold: float = 0.50,
) -> float:
    """Compute cross-conformer consistency for pocket-blocking related signals."""
    if conformer_group.empty:
        return float("nan")

    weight_map = dict(CONSISTENCY_WEIGHTS if weights is None else weights)
    num = np.zeros((len(conformer_group),), dtype=np.float64)
    den = np.zeros((len(conformer_group),), dtype=np.float64)
    used_any = False

    for col, weight in weight_map.items():
        if col not in conformer_group.columns:
            continue
        values = to_numeric(conformer_group[col]).to_numpy(dtype=np.float64)
        finite = np.isfinite(values)
        if not finite.any():
            continue
        used_any = True
        clipped = np.where(finite, np.clip(values, 0.0, 1.0), 0.0)
        num += float(weight) * clipped
        den += np.where(finite, float(weight), 0.0)

    if (not used_any) or np.all(den <= 0.0):
        return float("nan")

    combined = np.where(den > 0.0, num / den, np.nan)
    finite_combined = combined[np.isfinite(combined)]
    if finite_combined.size == 0:
        return float("nan")

    mean_hit = float(np.nanmean(finite_combined))
    std_hit = float(np.nanstd(finite_combined, ddof=0))
    stable_hit_ratio = float(np.mean(finite_combined >= float(np.clip(hit_threshold, 0.0, 1.0))))

    consistency = 0.60 * mean_hit + 0.30 * stable_hit_ratio + 0.10 * (1.0 - std_hit)
    return float(np.clip(consistency, 0.0, 1.0))


def build_blocking_explanation(
    row: pd.Series,
    std_ref: float = np.nan,
    fallback_reason: str = "综合评分较高，建议优先进入后续验证",
) -> str:
    """Build human-readable explanation from aggregated blocking metrics."""
    reasons: list[str] = []

    num_conformers = int(row.get("num_conformers", 0))
    pocket_hit = float(row.get("mean_topk_pocket_hit_fraction", np.nan))
    catalytic_hit = float(row.get("mean_topk_catalytic_hit_fraction", np.nan))
    mouth_occ = float(row.get("mean_topk_mouth_occlusion_score", np.nan))
    mouth_axis_block = float(row.get("mean_topk_mouth_axis_block_fraction", np.nan))
    mouth_aperture_block = float(row.get("mean_topk_mouth_aperture_block_fraction", np.nan))
    mouth_min_clearance = float(row.get("mean_topk_mouth_min_clearance", np.nan))
    substrate_overlap = float(row.get("mean_topk_substrate_overlap_score", np.nan))
    ligand_path_block = float(row.get("mean_topk_ligand_path_block_score", np.nan))
    ligand_path_fraction = float(row.get("mean_topk_ligand_path_block_fraction", np.nan))
    ligand_path_bottleneck = float(row.get("mean_topk_ligand_path_bottleneck_score", np.nan))
    ligand_path_exit_block = float(row.get("mean_topk_ligand_path_exit_block_fraction", np.nan))
    ligand_path_clearance = float(row.get("mean_topk_ligand_path_min_clearance", np.nan))
    pocket_shape_overwide = float(row.get("mean_topk_pocket_shape_overwide_proxy", np.nan))
    pocket_shape_count = float(row.get("mean_topk_pocket_shape_residue_count", np.nan))
    consistency = float(row.get("pocket_consistency_score", np.nan))
    std_score = float(row.get("std_conformer_score", np.nan))
    best_conf = float(row.get("best_conformer_score", np.nan))

    if (
        num_conformers >= 2
        and np.isfinite(consistency)
        and consistency >= 0.62
        and np.isfinite(pocket_hit)
        and pocket_hit >= 0.48
    ):
        reasons.append(f"在 {num_conformers} 个构象中均保持较高 pocket 命中率")

    if np.isfinite(catalytic_hit) and catalytic_hit >= 0.42:
        reasons.append("对 catalytic residues 覆盖较强")

    if (
        (np.isfinite(mouth_occ) and mouth_occ >= 0.55)
        or (np.isfinite(mouth_axis_block) and mouth_axis_block >= 0.55)
        or (np.isfinite(mouth_aperture_block) and mouth_aperture_block >= 0.55)
        or (np.isfinite(mouth_min_clearance) and mouth_min_clearance <= 2.6)
    ):
        reasons.append("口袋口部轴向/孔径覆盖较强")

    if (
        (np.isfinite(ligand_path_block) and ligand_path_block >= 0.55)
        or (np.isfinite(substrate_overlap) and substrate_overlap >= 0.62)
    ):
        reasons.append("与 ligand template 冲突明显")

    if (
        (np.isfinite(ligand_path_fraction) and ligand_path_fraction >= 0.55)
        or (np.isfinite(ligand_path_bottleneck) and ligand_path_bottleneck >= 0.55)
        or (np.isfinite(ligand_path_exit_block) and ligand_path_exit_block >= 0.60)
        or (np.isfinite(ligand_path_clearance) and ligand_path_clearance <= 2.2)
    ):
        reasons.append("多候选配体通路出现连续阻断/瓶颈")

    stable_threshold = float(std_ref) if np.isfinite(std_ref) else 0.10
    if np.isfinite(std_score) and std_score <= stable_threshold:
        reasons.append("跨构象波动较低，稳定性较好")

    if np.isfinite(best_conf) and best_conf >= 0.82:
        reasons.append("存在高置信最佳构象")

    if np.isfinite(pocket_shape_overwide) and pocket_shape_overwide >= 0.55:
        if np.isfinite(pocket_shape_count):
            reasons.insert(0, f"pocket 定义偏宽，平均约 {pocket_shape_count:.1f} 个残基，建议复核口袋边界")
        else:
            reasons.insert(0, "pocket 定义偏宽，建议复核口袋边界")

    if not reasons:
        reasons.append(str(fallback_reason))

    return "；".join(reasons[:4])


__all__ = [
    "GEOMETRY_AUX_WEIGHTS",
    "CONSISTENCY_WEIGHTS",
    "to_numeric",
    "safe_mean_if_exists",
    "compute_pocket_overwide_penalty",
    "apply_pocket_overwide_penalty",
    "compute_weighted_scaled_mean",
    "compute_consistency_score",
    "build_blocking_explanation",
]
