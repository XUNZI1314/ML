"""Pseudo-label generation and PyTorch training for pose-level tabular model.

This module trains a simple MLP to predict whether a pose is more likely to be
pocket-blocking binding, using either real labels or generated pseudo labels.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


ID_COLUMNS = ["nanobody_id", "conformer_id", "pose_id"]
PATH_COLUMNS = ["pdb_path", "pocket_file", "catalytic_file", "ligand_file"]
TEXT_STATUS_COLUMNS = [
    "status",
    "error_message",
    "warning_message",
    "split_mode",
    "geometry_debug_summary",
]


def build_feature_direction_map() -> dict[str, tuple[int, float]]:
    """Return pseudo-label direction/weight map.

    sign=+1 means larger is better; sign=-1 means smaller is better.
    """
    return {
        "pocket_hit_fraction": (+1, 0.16),
        "catalytic_hit_fraction": (+1, 0.14),
        "mouth_occlusion_score": (+1, 0.12),
        "mouth_axis_block_fraction": (+1, 0.05),
        "mouth_aperture_block_fraction": (+1, 0.05),
        "mouth_min_clearance": (-1, 0.03),
        "delta_pocket_occupancy_proxy": (+1, 0.11),
        "substrate_overlap_score": (+1, 0.11),
        "ligand_path_block_score": (+1, 0.06),
        "ligand_path_block_fraction": (+1, 0.05),
        "ligand_path_bottleneck_score": (+1, 0.04),
        "ligand_path_exit_block_fraction": (+1, 0.04),
        "ligand_path_min_clearance": (-1, 0.03),
        "min_distance_to_pocket": (-1, 0.08),
        "rsite_accuracy": (+1, 0.06),
        "hdock_score": (-1, 0.06),
        "mmgbsa": (-1, 0.05),
        "interface_dg": (-1, 0.03),
    }


class PoseDataset(Dataset):
    """Simple tabular dataset with hard labels and optional soft targets."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        soft_y: np.ndarray | None = None,
        soft_mask: np.ndarray | None = None,
    ) -> None:
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

        if soft_y is None:
            soft_y = np.full((x.shape[0],), np.nan, dtype=np.float32)
        if soft_mask is None:
            soft_mask = np.zeros((x.shape[0],), dtype=bool)

        self.soft_y = torch.as_tensor(soft_y, dtype=torch.float32)
        self.soft_mask = torch.as_tensor(soft_mask.astype(np.float32), dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx], self.soft_y[idx], self.soft_mask[idx]


@dataclass(slots=True)
class PreparedData:
    """Container for prepared tensors and feature transformation stats."""

    train_dataset: PoseDataset
    val_dataset: PoseDataset
    feature_columns: list[str]
    fill_values: dict[str, float]
    feature_mean: dict[str, float]
    feature_std: dict[str, float]
    train_index: np.ndarray
    val_index: np.ndarray


def _is_missing(value: Any) -> bool:
    """Return True for missing-like value."""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _to_numeric_series(series: pd.Series) -> pd.Series:
    """Convert a pandas Series to numeric with coercion."""
    return pd.to_numeric(series, errors="coerce")


def _robust_minmax_scale(values: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    """Robust 0~1 scaling with quantile clipping."""
    arr = _to_numeric_series(values).astype(float)
    out = pd.Series(np.nan, index=arr.index, dtype=float)

    finite_mask = np.isfinite(arr.to_numpy())
    if not finite_mask.any():
        return out

    valid = arr.to_numpy()[finite_mask]
    lq = float(np.clip(lower_q, 0.0, 1.0))
    uq = float(np.clip(upper_q, 0.0, 1.0))
    if lq >= uq:
        lq, uq = 0.0, 1.0

    lower = float(np.nanquantile(valid, lq))
    upper = float(np.nanquantile(valid, uq))

    denom = upper - lower
    if (not np.isfinite(denom)) or (denom < 1e-12):
        out.iloc[np.where(finite_mask)[0]] = 0.5
        return out

    scaled = (arr - lower) / denom
    scaled = scaled.clip(0.0, 1.0)
    out.iloc[np.where(finite_mask)[0]] = scaled.iloc[np.where(finite_mask)[0]]
    return out


def filter_low_quality_features(
    df: pd.DataFrame,
    feature_columns: list[str],
    max_missing_ratio: float = 0.80,
    near_constant_ratio: float = 0.995,
) -> tuple[list[str], dict[str, Any]]:
    """Filter high-missing and near-constant features.

    Returns filtered feature list plus a diagnostics dictionary.
    """
    max_missing = float(np.clip(max_missing_ratio, 0.0, 1.0))
    near_const = float(np.clip(near_constant_ratio, 0.0, 1.0))

    kept: list[str] = []
    removed_high_missing: list[str] = []
    removed_near_constant: list[str] = []
    missing_ratio_map: dict[str, float] = {}

    for col in feature_columns:
        if col not in df.columns:
            continue

        series = _to_numeric_series(df[col])
        ratio = float(series.isna().mean())
        missing_ratio_map[col] = ratio
        if ratio > max_missing:
            removed_high_missing.append(col)
            continue

        finite = series[np.isfinite(series.to_numpy(dtype=np.float64))]
        if finite.empty:
            removed_high_missing.append(col)
            continue

        if finite.shape[0] <= 1:
            removed_near_constant.append(col)
            continue

        value_counts = finite.value_counts(normalize=True, dropna=True)
        top_ratio = float(value_counts.iloc[0]) if not value_counts.empty else 1.0
        std_value = float(np.nanstd(finite.to_numpy(dtype=np.float64)))
        if top_ratio >= near_const or std_value <= 1e-12:
            removed_near_constant.append(col)
            continue

        kept.append(col)

    diagnostics = {
        "initial_feature_count": int(len(feature_columns)),
        "kept_feature_count": int(len(kept)),
        "removed_high_missing": sorted(removed_high_missing),
        "removed_near_constant": sorted(removed_near_constant),
        "missing_ratio": missing_ratio_map,
    }
    return kept, diagnostics


def summarize_pseudo_label_distribution(
    pseudo_score: np.ndarray,
    pseudo_label: np.ndarray,
) -> dict[str, Any]:
    """Summarize pseudo-score and pseudo-label distribution."""
    score = np.asarray(pseudo_score, dtype=np.float64)
    label = np.asarray(pseudo_label, dtype=np.float64)
    finite = score[np.isfinite(score)]

    return {
        "n_rows": int(score.shape[0]),
        "pseudo_positive_rate": float(np.nanmean(label)) if label.size > 0 else float("nan"),
        "pseudo_score_min": float(np.nanmin(finite)) if finite.size > 0 else float("nan"),
        "pseudo_score_p10": float(np.nanpercentile(finite, 10)) if finite.size > 0 else float("nan"),
        "pseudo_score_p50": float(np.nanpercentile(finite, 50)) if finite.size > 0 else float("nan"),
        "pseudo_score_p90": float(np.nanpercentile(finite, 90)) if finite.size > 0 else float("nan"),
        "pseudo_score_max": float(np.nanmax(finite)) if finite.size > 0 else float("nan"),
    }


def _top_component_summary(
    component_scores: dict[str, np.ndarray],
    row_idx: int,
    top_k: int = 3,
) -> str:
    """Build compact pseudo-component summary string for one row."""
    if not component_scores:
        return ""

    parts: list[tuple[str, float]] = []
    for name, arr in component_scores.items():
        if row_idx >= arr.shape[0]:
            continue
        value = float(arr[row_idx])
        if np.isfinite(value):
            parts.append((name, value))

    if not parts:
        return ""

    parts.sort(key=lambda x: abs(x[1]), reverse=True)
    selected = parts[: max(1, int(top_k))]
    return ";".join([f"{name}={value:.3f}" for name, value in selected])


def select_feature_columns(
    df: pd.DataFrame,
    exclude_columns: list[str] | None = None,
    min_non_nan_ratio: float = 0.05,
    max_missing_ratio: float = 0.80,
    near_constant_ratio: float = 0.995,
) -> list[str]:
    """Select usable numeric feature columns for tabular model.

    The function auto-detects numeric-like columns and excludes id/status/label
    columns and columns with too few valid values.
    """
    excluded = {
        *ID_COLUMNS,
        *PATH_COLUMNS,
        *TEXT_STATUS_COLUMNS,
        "label",
        "pseudo_label",
        "pseudo_score",
        "pseudo_rank",
        "pseudo_components",
        "_row_index",
    }
    if exclude_columns:
        excluded.update(exclude_columns)

    feature_cols: list[str] = []
    n_rows = max(1, len(df))
    min_count = max(1, int(np.ceil(min_non_nan_ratio * n_rows)))

    for col in df.columns:
        if col in excluded:
            continue

        s = _to_numeric_series(df[col])
        non_nan = int(s.notna().sum())
        if non_nan < min_count:
            continue

        # Keep columns with at least some variance after coercion.
        if non_nan > 1 and float(s.std(skipna=True)) <= 0.0:
            continue

        feature_cols.append(col)

    filtered, diagnostics = filter_low_quality_features(
        df=df,
        feature_columns=feature_cols,
        max_missing_ratio=max_missing_ratio,
        near_constant_ratio=near_constant_ratio,
    )

    if not filtered:
        raise ValueError("No valid numeric feature columns were selected.")
    _ = diagnostics  # reserved for future logging hooks
    return filtered


def build_pseudo_labels(
    df: pd.DataFrame,
    top_fraction: float = 0.25,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    threshold_mode: str = "top_fraction",
    threshold_value: float | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build pseudo_score and pseudo_label from available geometry/energy columns.

    Pseudo label procedure:
    1) Detect available columns.
    2) Align direction (higher score means better pocket-blocking probability).
    3) Robust min-max normalize into [0, 1].
    4) Weighted sum to get pseudo_score.
    5) Apply configured threshold mode to produce pseudo_label.
    """
    feature_spec = build_feature_direction_map()

    out = df.copy()
    available = [c for c in feature_spec if c in out.columns]
    if not available:
        raise ValueError("No pseudo-label source columns are available.")

    weighted_values: list[np.ndarray] = []
    weighted_masks: list[np.ndarray] = []
    used_cols: list[str] = []
    component_scores: dict[str, np.ndarray] = {}

    for col in available:
        sign, weight = feature_spec[col]
        norm = _robust_minmax_scale(out[col], lower_q=lower_q, upper_q=upper_q)
        norm_np = norm.to_numpy(dtype=np.float64)

        # Direction alignment.
        if sign < 0:
            norm_np = 1.0 - norm_np

        valid_mask = np.isfinite(norm_np)
        if not valid_mask.any():
            continue

        contrib = np.where(valid_mask, norm_np * weight, 0.0)
        weighted_values.append(contrib)
        weighted_masks.append(np.where(valid_mask, weight, 0.0))
        used_cols.append(col)
        component_scores[col] = contrib

    if not used_cols:
        raise ValueError("All pseudo-label source columns are empty/invalid.")

    numerator = np.sum(np.vstack(weighted_values), axis=0)
    denominator = np.sum(np.vstack(weighted_masks), axis=0)
    pseudo_score = np.where(denominator > 0.0, numerator / denominator, np.nan)

    # Fill rows without usable source values by global median to avoid drop.
    finite_mask = np.isfinite(pseudo_score)
    if finite_mask.any():
        fallback_score = float(np.nanmedian(pseudo_score[finite_mask]))
    else:
        fallback_score = 0.5
    pseudo_score = np.where(np.isfinite(pseudo_score), pseudo_score, fallback_score)
    pseudo_score = np.clip(pseudo_score, 0.0, 1.0)

    mode = str(threshold_mode).strip().lower()
    rank_pct = pd.Series(pseudo_score).rank(method="average", pct=True).to_numpy(dtype=np.float64)
    pseudo_rank = rank_pct

    if mode == "fixed":
        fixed_thr = 0.6 if threshold_value is None else float(threshold_value)
        fixed_thr = float(np.clip(fixed_thr, 0.0, 1.0))
        pseudo_label = (pseudo_score >= fixed_thr).astype(np.float64)
        threshold_info = {"mode": "fixed", "value": fixed_thr}
    elif mode == "quantile":
        q = 0.75 if threshold_value is None else float(threshold_value)
        q = float(np.clip(q, 0.0, 1.0))
        quantile_thr = float(np.nanquantile(pseudo_score, q))
        pseudo_label = (pseudo_score >= quantile_thr).astype(np.float64)
        threshold_info = {"mode": "quantile", "quantile": q, "score_threshold": quantile_thr}
    else:
        frac = float(np.clip(top_fraction if threshold_value is None else float(threshold_value), 0.0, 1.0))
        if frac <= 0.0:
            pseudo_label = np.zeros_like(pseudo_score, dtype=np.float64)
        elif frac >= 1.0:
            pseudo_label = np.ones_like(pseudo_score, dtype=np.float64)
        else:
            cutoff = 1.0 - frac
            pseudo_label = (rank_pct >= cutoff).astype(np.float64)
        threshold_info = {"mode": "top_fraction", "value": frac}

    out["pseudo_score"] = pseudo_score
    out["pseudo_label"] = pseudo_label
    out["pseudo_rank"] = pseudo_rank

    component_summary = [
        _top_component_summary(component_scores=component_scores, row_idx=i, top_k=3)
        for i in range(len(out))
    ]
    out["pseudo_components"] = component_summary

    distribution = summarize_pseudo_label_distribution(pseudo_score, pseudo_label)

    info = {
        "used_columns": used_cols,
        "threshold": threshold_info,
        "pseudo_positive_rate": float(np.mean(pseudo_label)) if len(pseudo_label) > 0 else 0.0,
        "distribution": distribution,
    }
    return out, info


def split_by_nanobody_group(
    df: pd.DataFrame,
    group_col: str = "nanobody_id",
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split train/val by nanobody_id group to avoid leakage.

    Returns:
        train_idx, val_idx as row-index arrays (0-based positional indexes).
    """
    if len(df) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    frac = float(np.clip(val_ratio, 0.0, 0.9))
    rng = np.random.default_rng(int(seed))

    if group_col not in df.columns:
        perm = rng.permutation(len(df))
        n_val = max(1, int(round(len(df) * frac))) if len(df) > 1 else 0
        val_idx = np.sort(perm[:n_val])
        train_idx = np.sort(perm[n_val:])
        return train_idx, val_idx

    groups = df[group_col].astype(str)
    unique_groups = groups.dropna().unique().tolist()

    if len(unique_groups) >= 2:
        rng.shuffle(unique_groups)
        n_val_groups = max(1, int(round(len(unique_groups) * frac)))
        n_val_groups = min(n_val_groups, len(unique_groups) - 1)
        val_groups = set(unique_groups[:n_val_groups])

        val_mask = groups.isin(val_groups).to_numpy()
        val_idx = np.where(val_mask)[0]
        train_idx = np.where(~val_mask)[0]
    else:
        # Fallback to row-wise split when grouping cannot be formed.
        perm = rng.permutation(len(df))
        n_val = max(1, int(round(len(df) * frac))) if len(df) > 1 else 0
        val_idx = np.sort(perm[:n_val])
        train_idx = np.sort(perm[n_val:])

    if len(train_idx) == 0 and len(val_idx) > 0:
        train_idx = val_idx.copy()
    if len(val_idx) == 0 and len(train_idx) > 1:
        val_idx = train_idx[-1:]
        train_idx = train_idx[:-1]

    return np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int)


def _standardize_with_train_stats(
    feature_df: pd.DataFrame,
    train_idx: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, float], dict[str, float]]:
    """Fill NaN with train medians and standardize by train mean/std."""
    if len(feature_df) == 0:
        return feature_df.copy(), {}, {}, {}

    train_df = feature_df.iloc[train_idx] if len(train_idx) > 0 else feature_df

    fill_values = train_df.median(skipna=True).to_dict()
    filled = feature_df.copy()
    for col, med in fill_values.items():
        if not np.isfinite(med):
            med = 0.0
            fill_values[col] = med
        filled[col] = filled[col].fillna(med)

    mean = filled.iloc[train_idx].mean(axis=0, skipna=True).to_dict() if len(train_idx) > 0 else filled.mean(axis=0, skipna=True).to_dict()
    std = filled.iloc[train_idx].std(axis=0, skipna=True).to_dict() if len(train_idx) > 0 else filled.std(axis=0, skipna=True).to_dict()

    scaled = filled.copy()
    for col in scaled.columns:
        m = float(mean.get(col, 0.0))
        s = float(std.get(col, 1.0))
        if (not np.isfinite(s)) or (s < 1e-12):
            s = 1.0
            std[col] = s
        scaled[col] = (scaled[col] - m) / s

    return scaled, {k: float(v) for k, v in fill_values.items()}, {k: float(v) for k, v in mean.items()}, {k: float(v) for k, v in std.items()}


def prepare_datasets(
    df: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    soft_target_column: str | None = "pseudo_score",
    val_ratio: float = 0.2,
    group_col: str = "nanobody_id",
    seed: int = 42,
) -> PreparedData:
    """Prepare train/val datasets with group-wise split and robust preprocessing."""
    if len(feature_columns) == 0:
        raise ValueError("feature_columns is empty.")
    if label_column not in df.columns:
        raise ValueError(f"Label column {label_column!r} is missing.")

    work_df = df.reset_index(drop=True).copy()
    y_all = _to_numeric_series(work_df[label_column])
    valid_label_mask = y_all.notna().to_numpy()
    if valid_label_mask.sum() < 2:
        raise ValueError("Not enough valid labels for training.")

    trainable_df = work_df.loc[valid_label_mask].reset_index(drop=True)
    y = _to_numeric_series(trainable_df[label_column]).astype(float)
    y = y.clip(0.0, 1.0).to_numpy(dtype=np.float32)

    x_raw = trainable_df[feature_columns].apply(pd.to_numeric, errors="coerce")

    tr_idx, va_idx = split_by_nanobody_group(
        trainable_df,
        group_col=group_col,
        val_ratio=val_ratio,
        seed=seed,
    )

    x_scaled, fill_values, feat_mean, feat_std = _standardize_with_train_stats(x_raw, tr_idx)
    x_np = x_scaled.to_numpy(dtype=np.float32)

    soft_y_np: np.ndarray | None = None
    soft_mask_np: np.ndarray | None = None
    if soft_target_column and (soft_target_column in trainable_df.columns):
        soft_y = _to_numeric_series(trainable_df[soft_target_column]).to_numpy(dtype=np.float32)
        soft_mask = np.isfinite(soft_y)
        soft_y_np = np.where(soft_mask, np.clip(soft_y, 0.0, 1.0), np.nan).astype(np.float32)
        soft_mask_np = soft_mask.astype(bool)

    train_dataset = PoseDataset(
        x=x_np[tr_idx],
        y=y[tr_idx],
        soft_y=soft_y_np[tr_idx] if soft_y_np is not None else None,
        soft_mask=soft_mask_np[tr_idx] if soft_mask_np is not None else None,
    )
    val_dataset = PoseDataset(
        x=x_np[va_idx] if len(va_idx) > 0 else x_np[tr_idx],
        y=y[va_idx] if len(va_idx) > 0 else y[tr_idx],
        soft_y=soft_y_np[va_idx] if (soft_y_np is not None and len(va_idx) > 0) else (soft_y_np[tr_idx] if soft_y_np is not None else None),
        soft_mask=soft_mask_np[va_idx] if (soft_mask_np is not None and len(va_idx) > 0) else (soft_mask_np[tr_idx] if soft_mask_np is not None else None),
    )

    return PreparedData(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        feature_columns=list(feature_columns),
        fill_values=fill_values,
        feature_mean=feat_mean,
        feature_std=feat_std,
        train_index=tr_idx,
        val_index=va_idx,
    )


class PoseMLP(nn.Module):
    """Simple MLP for pose-level tabular features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (128, 64),
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")

        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            h_int = int(h)
            if h_int <= 0:
                continue
            layers.append(nn.Linear(prev, h_int))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(float(np.clip(dropout, 0.0, 0.9))))
            prev = h_int
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute binary AUC without external dependencies."""
    y = y_true.astype(int)
    s = y_score.astype(float)

    pos_mask = y == 1
    neg_mask = y == 0
    n_pos = int(np.sum(pos_mask))
    n_neg = int(np.sum(neg_mask))
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    auc = (np.sum(ranks[pos_mask]) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _binary_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Compute accuracy/precision/recall/F1 using a fixed threshold."""
    if y_true.size == 0:
        return {
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
        }

    y_bin = np.asarray(y_true, dtype=np.float32) >= 0.5
    p_bin = np.asarray(y_prob, dtype=np.float32) >= float(threshold)

    tp = float(np.sum(np.logical_and(p_bin, y_bin)))
    fp = float(np.sum(np.logical_and(p_bin, np.logical_not(y_bin))))
    fn = float(np.sum(np.logical_and(np.logical_not(p_bin), y_bin)))
    tn = float(np.sum(np.logical_and(np.logical_not(p_bin), np.logical_not(y_bin))))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1.0)
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    """Evaluate model on one dataloader and return loss + classification metrics."""
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for x, y, _, _ in data_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)

            if criterion is not None:
                loss = criterion(logits, y)
                batch_size = int(x.shape[0])
                total_loss += float(loss.item()) * batch_size
                total_count += batch_size

            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(y.detach().cpu().numpy())

    logits_np = np.concatenate(all_logits, axis=0) if all_logits else np.empty((0,), dtype=np.float32)
    labels_np = np.concatenate(all_labels, axis=0) if all_labels else np.empty((0,), dtype=np.float32)

    probs = 1.0 / (1.0 + np.exp(-logits_np)) if logits_np.size > 0 else np.empty((0,), dtype=np.float32)
    preds = (probs >= 0.5).astype(np.float32)

    if labels_np.size > 0:
        class_metrics = _binary_classification_metrics(labels_np, probs, threshold=0.5)
        auc = _binary_auc(labels_np, probs)
    else:
        class_metrics = {
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
        }
        auc = float("nan")

    avg_loss = float(total_loss / total_count) if total_count > 0 else float("nan")
    return {
        "loss": avg_loss,
        "auc": auc,
        "accuracy": float(class_metrics["accuracy"]),
        "acc": float(class_metrics["accuracy"]),
        "precision": float(class_metrics["precision"]),
        "recall": float(class_metrics["recall"]),
        "f1": float(class_metrics["f1"]),
    }


def train_model(
    model: nn.Module,
    prepared: PreparedData,
    out_dir: str | Path,
    epochs: int = 80,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    soft_target_weight: float = 0.25,
    early_stopping_patience: int = 12,
    min_delta: float = 1e-4,
    grad_clip_norm: float = 1.0,
    scheduler_patience: int = 5,
    scheduler_factor: float = 0.5,
    device: str | torch.device | None = None,
) -> tuple[nn.Module, pd.DataFrame, dict[str, Any]]:
    """Train MLP with BCEWithLogits and optional soft-target auxiliary loss."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    train_loader = DataLoader(prepared.train_dataset, batch_size=max(1, int(batch_size)), shuffle=True)
    val_loader = DataLoader(prepared.val_dataset, batch_size=max(1, int(batch_size)), shuffle=False)

    y_train = prepared.train_dataset.y.detach().cpu().numpy().astype(float)
    pos_count = float(np.sum(y_train == 1.0))
    neg_count = float(np.sum(y_train == 0.0))
    if pos_count > 0 and neg_count > 0:
        pos_weight_value = neg_count / max(pos_count, 1.0)
    else:
        pos_weight_value = 1.0

    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    mse_criterion = nn.MSELoss(reduction="mean")

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(np.clip(scheduler_factor, 0.1, 0.95)),
        patience=max(1, int(scheduler_patience)),
        threshold=max(0.0, float(min_delta) * 0.5),
    )

    best_val_loss = np.inf
    best_epoch = -1
    no_improve_epochs = 0
    patience = max(1, int(early_stopping_patience))
    log_rows: list[dict[str, Any]] = []

    for epoch in range(1, int(max(1, epochs)) + 1):
        model.train()
        epoch_loss = 0.0
        hard_loss_sum = 0.0
        soft_loss_sum = 0.0
        soft_count = 0
        seen = 0

        for x, y, soft_y, soft_mask in train_loader:
            x = x.to(device)
            y = y.to(device)
            soft_y = soft_y.to(device)
            soft_mask = soft_mask.to(device)

            logits = model(x)
            hard_loss = bce_criterion(logits, y)

            if float(soft_target_weight) > 0.0:
                valid_soft = soft_mask > 0.5
                if torch.any(valid_soft):
                    prob = torch.sigmoid(logits[valid_soft])
                    soft_loss = mse_criterion(prob, soft_y[valid_soft])
                else:
                    soft_loss = torch.tensor(0.0, device=device)
            else:
                soft_loss = torch.tensor(0.0, device=device)

            loss = hard_loss + float(max(0.0, soft_target_weight)) * soft_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(grad_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
            optimizer.step()

            batch_n = int(x.shape[0])
            epoch_loss += float(loss.item()) * batch_n
            hard_loss_sum += float(hard_loss.item()) * batch_n
            valid_count = int(torch.sum(soft_mask > 0.5).item())
            soft_count += valid_count
            if valid_count > 0:
                soft_loss_sum += float(soft_loss.item()) * valid_count
            seen += batch_n

        train_total_loss = float(epoch_loss / seen) if seen > 0 else float("nan")
        train_hard_loss = float(hard_loss_sum / seen) if seen > 0 else float("nan")
        train_soft_loss = float(soft_loss_sum / soft_count) if soft_count > 0 else float("nan")

        train_metrics = evaluate_model(model, train_loader, criterion=bce_criterion, device=device)
        val_metrics = evaluate_model(model, val_loader, criterion=bce_criterion, device=device)
        val_loss_for_sched = float(val_metrics["loss"])
        if np.isfinite(val_loss_for_sched):
            scheduler.step(val_loss_for_sched)

        current_lr = float(optimizer.param_groups[0].get("lr", lr))

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_total_loss,
            "train_total_loss": train_total_loss,
            "train_hard_loss": train_hard_loss,
            "train_soft_loss": train_soft_loss,
            "train_auc": float(train_metrics["auc"]),
            "train_accuracy": float(train_metrics["accuracy"]),
            "train_precision": float(train_metrics["precision"]),
            "train_recall": float(train_metrics["recall"]),
            "train_f1": float(train_metrics["f1"]),
            "val_loss": float(val_metrics["loss"]),
            "val_auc": float(val_metrics["auc"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_precision": float(val_metrics["precision"]),
            "val_recall": float(val_metrics["recall"]),
            "val_f1": float(val_metrics["f1"]),
            "soft_supervision_fraction": float(soft_count / seen) if seen > 0 else 0.0,
            "pos_weight": float(pos_weight_value),
        }
        log_rows.append(row)

        current_val = float(val_metrics["loss"])
        improved = np.isfinite(current_val) and (current_val + float(min_delta) < best_val_loss)
        if improved:
            best_val_loss = current_val
            best_epoch = epoch
            no_improve_epochs = 0
            ckpt = {
                "model_state_dict": model.state_dict(),
                "input_dim": int(prepared.train_dataset.x.shape[1]),
                "best_epoch": int(best_epoch),
                "best_val_loss": float(best_val_loss),
                "feature_columns": prepared.feature_columns,
                "fill_values": prepared.fill_values,
                "feature_mean": prepared.feature_mean,
                "feature_std": prepared.feature_std,
            }
            torch.save(ckpt, out_path / "best_model.pt")
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            break

    train_log = pd.DataFrame(log_rows)
    train_log.to_csv(out_path / "train_log.csv", index=False)

    # Load best checkpoint back for downstream prediction consistency.
    best_ckpt_path = out_path / "best_model.pt"
    if best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    summary = {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss) if np.isfinite(best_val_loss) else float("nan"),
        "stopped_early": bool(no_improve_epochs >= patience),
        "epochs_ran": int(len(log_rows)),
        "train_size": int(len(prepared.train_dataset)),
        "val_size": int(len(prepared.val_dataset)),
        "pos_weight": float(pos_weight_value),
    }
    return model, train_log, summary


def _prepare_inference_features(
    df: pd.DataFrame,
    feature_columns: list[str],
    fill_values: dict[str, float],
    feature_mean: dict[str, float],
    feature_std: dict[str, float],
) -> pd.DataFrame:
    """Prepare standardized feature matrix for inference and interpretation."""
    x = df[feature_columns].apply(pd.to_numeric, errors="coerce").copy()
    for col in feature_columns:
        fill_v = float(fill_values.get(col, 0.0))
        mean_v = float(feature_mean.get(col, 0.0))
        std_v = float(feature_std.get(col, 1.0))
        if (not np.isfinite(std_v)) or (std_v < 1e-12):
            std_v = 1.0
        x[col] = x[col].fillna(fill_v)
        x[col] = (x[col] - mean_v) / std_v
    return x


def _summarize_prediction_contributions(
    model: nn.Module,
    x_std: pd.DataFrame,
    feature_columns: list[str],
    top_k: int = 3,
) -> list[str]:
    """Approximate per-row top feature contributions using first-layer sensitivity."""
    n_rows = int(len(x_std))
    if n_rows == 0 or len(feature_columns) == 0:
        return [""] * n_rows

    feature_weight = np.ones((len(feature_columns),), dtype=np.float64)
    try:
        first_linear = None
        if hasattr(model, "net") and isinstance(model.net, nn.Sequential):
            for layer in model.net:
                if isinstance(layer, nn.Linear):
                    first_linear = layer
                    break
        if first_linear is not None:
            w = first_linear.weight.detach().cpu().numpy().astype(np.float64)
            if w.ndim == 2 and w.shape[1] == len(feature_columns):
                feature_weight = np.mean(np.abs(w), axis=0)
    except Exception:
        pass

    x_np = x_std.to_numpy(dtype=np.float64)
    contrib = x_np * feature_weight.reshape(1, -1)

    summaries: list[str] = []
    k = max(1, int(top_k))
    for i in range(n_rows):
        row = contrib[i]
        finite_mask = np.isfinite(row)
        if not finite_mask.any():
            summaries.append("")
            continue

        abs_row = np.where(finite_mask, np.abs(row), -np.inf)
        top_idx = np.argsort(abs_row)[-k:][::-1]
        parts: list[str] = []
        for idx in top_idx:
            if idx < 0 or idx >= len(feature_columns):
                continue
            if not np.isfinite(row[idx]):
                continue
            parts.append(f"{feature_columns[idx]}={row[idx]:+.3f}")
        summaries.append(";".join(parts))

    return summaries


def predict_pose_prob(
    model: nn.Module,
    df: pd.DataFrame,
    feature_columns: list[str],
    fill_values: dict[str, float],
    feature_mean: dict[str, float],
    feature_std: dict[str, float],
    batch_size: int = 512,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Predict pose probability for each row of the input DataFrame."""
    if len(df) == 0:
        return np.empty((0,), dtype=np.float32)

    x = _prepare_inference_features(
        df=df,
        feature_columns=feature_columns,
        fill_values=fill_values,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )

    x_np = x.to_numpy(dtype=np.float32)
    x_tensor = torch.as_tensor(x_np, dtype=torch.float32)

    model.eval()
    model.to(device)
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x_tensor.shape[0], max(1, int(batch_size))):
            end = min(start + max(1, int(batch_size)), x_tensor.shape[0])
            logits = model(x_tensor[start:end].to(device))
            p = torch.sigmoid(logits).detach().cpu().numpy()
            probs.append(p)

    return np.concatenate(probs, axis=0).astype(np.float32)


def predict_pose_prob_with_logits(
    model: nn.Module,
    df: pd.DataFrame,
    feature_columns: list[str],
    fill_values: dict[str, float],
    feature_mean: dict[str, float],
    feature_std: dict[str, float],
    batch_size: int = 512,
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Predict probabilities and raw logits for each pose."""
    if len(df) == 0:
        empty = np.empty((0,), dtype=np.float32)
        return empty, empty

    x = _prepare_inference_features(
        df=df,
        feature_columns=feature_columns,
        fill_values=fill_values,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    x_np = x.to_numpy(dtype=np.float32)
    x_tensor = torch.as_tensor(x_np, dtype=torch.float32)

    model.eval()
    model.to(device)
    probs: list[np.ndarray] = []
    logits_all: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x_tensor.shape[0], max(1, int(batch_size))):
            end = min(start + max(1, int(batch_size)), x_tensor.shape[0])
            logits = model(x_tensor[start:end].to(device))
            logits_np = logits.detach().cpu().numpy()
            p = (1.0 / (1.0 + np.exp(-logits_np))).astype(np.float32)
            logits_all.append(logits_np)
            probs.append(p)

    return np.concatenate(probs, axis=0).astype(np.float32), np.concatenate(logits_all, axis=0).astype(np.float32)


def save_training_artifacts(
    out_dir: str | Path,
    prepared: PreparedData,
    train_log: pd.DataFrame,
    summary: dict[str, Any],
    pseudo_info: dict[str, Any],
    mode: str,
) -> dict[str, str]:
    """Save training artifacts for reproducibility and debugging."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_log_path = out_path / "train_log.csv"
    feature_json_path = out_path / "feature_columns.json"
    summary_json_path = out_path / "training_summary.json"
    summary_csv_path = out_path / "training_summary.csv"

    train_log.to_csv(train_log_path, index=False)

    feature_payload = {
        "feature_columns": list(prepared.feature_columns),
        "fill_values": {k: float(v) for k, v in prepared.fill_values.items()},
        "feature_mean": {k: float(v) for k, v in prepared.feature_mean.items()},
        "feature_std": {k: float(v) for k, v in prepared.feature_std.items()},
    }
    with feature_json_path.open("w", encoding="utf-8") as f:
        json.dump(feature_payload, f, ensure_ascii=True, indent=2)

    summary_payload = {
        "mode": str(mode),
        "n_rows_train": int(len(prepared.train_dataset)),
        "n_rows_val": int(len(prepared.val_dataset)),
        "n_features": int(len(prepared.feature_columns)),
        "summary": summary,
        "pseudo": pseudo_info,
    }
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=True, indent=2)

    summary_row = {
        "mode": str(mode),
        "n_rows": int(len(prepared.train_dataset) + len(prepared.val_dataset)),
        "n_features": int(len(prepared.feature_columns)),
        "best_epoch": int(summary.get("best_epoch", -1)),
        "best_val_loss": float(summary.get("best_val_loss", np.nan)),
        "epochs_ran": int(summary.get("epochs_ran", 0)),
        "stopped_early": bool(summary.get("stopped_early", False)),
        "pseudo_positive_rate": float(pseudo_info.get("pseudo_positive_rate", np.nan)),
        "pseudo_columns_used": ",".join(map(str, pseudo_info.get("used_columns", []))),
    }
    pd.DataFrame([summary_row]).to_csv(summary_csv_path, index=False)

    return {
        "train_log_csv": str(train_log_path),
        "feature_columns_json": str(feature_json_path),
        "training_summary_json": str(summary_json_path),
        "training_summary_csv": str(summary_csv_path),
    }


def _build_parser() -> argparse.ArgumentParser:
    """Build minimal CLI parser for training module."""
    parser = argparse.ArgumentParser(description="Train pose model from pose_features.csv")
    parser.add_argument("--feature_csv", required=True, help="Path to pose_features.csv")
    parser.add_argument("--out_dir", default="model_outputs", help="Output directory")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--soft_target_weight", type=float, default=0.25)
    parser.add_argument("--top_pseudo_fraction", type=float, default=0.25)
    parser.add_argument(
        "--pseudo_threshold_mode",
        choices=["top_fraction", "quantile", "fixed"],
        default="top_fraction",
        help="Pseudo-label threshold mode.",
    )
    parser.add_argument(
        "--pseudo_threshold_value",
        type=float,
        default=None,
        help="Threshold value interpreted by pseudo_threshold_mode.",
    )
    parser.add_argument("--feature_min_non_nan_ratio", type=float, default=0.05)
    parser.add_argument("--feature_max_missing_ratio", type=float, default=0.80)
    parser.add_argument("--feature_near_constant_ratio", type=float, default=0.995)
    parser.add_argument("--early_stopping_patience", type=int, default=12)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--lr_scheduler_patience", type=int, default=5)
    parser.add_argument("--lr_scheduler_factor", type=float, default=0.5)
    return parser


def main() -> None:
    """Minimal training entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    feature_csv = Path(args.feature_csv).expanduser()
    if not feature_csv.exists():
        raise FileNotFoundError(f"Feature CSV not found: {feature_csv}")

    df = pd.read_csv(feature_csv, low_memory=False)
    if df.empty:
        raise ValueError("Input feature CSV is empty.")

    if "status" in df.columns:
        ok_df = df[df["status"].astype(str).str.lower() == "ok"].copy()
        if len(ok_df) > 0:
            df = ok_df.reset_index(drop=True)

    has_real_label = "label" in df.columns and _to_numeric_series(df["label"]).notna().any()

    pseudo_info: dict[str, Any] = {
        "used_columns": [],
        "pseudo_positive_rate": float("nan"),
        "threshold": {"mode": "none"},
    }
    if has_real_label:
        # Real-label mode: label used as hard target.
        df_for_train = df.copy()
        label_column = "label"

        # Try to build pseudo score for optional soft supervision when possible.
        try:
            df_for_train, pseudo_info = build_pseudo_labels(
                df_for_train,
                top_fraction=float(args.top_pseudo_fraction),
                threshold_mode=str(args.pseudo_threshold_mode),
                threshold_value=args.pseudo_threshold_value,
            )
            soft_column = "pseudo_score"
        except Exception:
            soft_column = None
    else:
        # Pseudo-label mode: pseudo_label becomes hard target.
        df_for_train, pseudo_info = build_pseudo_labels(
            df.copy(),
            top_fraction=float(args.top_pseudo_fraction),
            threshold_mode=str(args.pseudo_threshold_mode),
            threshold_value=args.pseudo_threshold_value,
        )
        label_column = "pseudo_label"
        soft_column = "pseudo_score"

    feature_columns = select_feature_columns(
        df_for_train,
        min_non_nan_ratio=float(args.feature_min_non_nan_ratio),
        max_missing_ratio=float(args.feature_max_missing_ratio),
        near_constant_ratio=float(args.feature_near_constant_ratio),
    )
    prepared = prepare_datasets(
        df=df_for_train,
        feature_columns=feature_columns,
        label_column=label_column,
        soft_target_column=soft_column,
        val_ratio=float(args.val_ratio),
        group_col="nanobody_id",
        seed=int(args.seed),
    )

    model = PoseMLP(input_dim=len(feature_columns), hidden_dims=(128, 64), dropout=0.2)

    out_dir = Path(args.out_dir).expanduser()
    model, train_log, summary = train_model(
        model=model,
        prepared=prepared,
        out_dir=out_dir,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        soft_target_weight=float(args.soft_target_weight),
        early_stopping_patience=int(args.early_stopping_patience),
        min_delta=float(args.min_delta),
        grad_clip_norm=float(args.grad_clip_norm),
        scheduler_patience=int(args.lr_scheduler_patience),
        scheduler_factor=float(args.lr_scheduler_factor),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    pred_prob, pred_logit = predict_pose_prob_with_logits(
        model=model,
        df=df_for_train,
        feature_columns=prepared.feature_columns,
        fill_values=prepared.fill_values,
        feature_mean=prepared.feature_mean,
        feature_std=prepared.feature_std,
        batch_size=max(256, int(args.batch_size)),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    x_infer = _prepare_inference_features(
        df=df_for_train,
        feature_columns=prepared.feature_columns,
        fill_values=prepared.fill_values,
        feature_mean=prepared.feature_mean,
        feature_std=prepared.feature_std,
    )
    contribution_summary = _summarize_prediction_contributions(
        model=model,
        x_std=x_infer,
        feature_columns=prepared.feature_columns,
        top_k=3,
    )

    pred_df = pd.DataFrame(
        {
            "nanobody_id": df_for_train.get("nanobody_id"),
            "conformer_id": df_for_train.get("conformer_id"),
            "pose_id": df_for_train.get("pose_id"),
            "pred_prob": pred_prob,
            "pred_logit": pred_logit,
            "top_contributing_features": contribution_summary,
        }
    )

    # Keep key geometry/path features for downstream aggregation and explanation.
    passthrough_optional_cols = [
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
        "ligand_path_candidate_count",
        "delta_pocket_occupancy_proxy",
        "pocket_block_volume_proxy",
        "min_distance_to_pocket",
        "rsite_accuracy",
    ]
    for col in passthrough_optional_cols:
        if col in df_for_train.columns and col not in pred_df.columns:
            pred_df[col] = _to_numeric_series(df_for_train[col])

    if "label" in df_for_train.columns:
        pred_df["label"] = _to_numeric_series(df_for_train["label"])
    if "pseudo_label" in df_for_train.columns:
        pred_df["pseudo_label"] = _to_numeric_series(df_for_train["pseudo_label"])
    if "pseudo_score" in df_for_train.columns:
        pred_df["pseudo_score"] = _to_numeric_series(df_for_train["pseudo_score"])
    if "pseudo_rank" in df_for_train.columns:
        pred_df["pseudo_rank"] = _to_numeric_series(df_for_train["pseudo_rank"])
    if "pseudo_components" in df_for_train.columns:
        pred_df["pseudo_components"] = df_for_train["pseudo_components"].astype(str)

    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_dir / "pose_predictions.csv", index=False)

    artifacts = save_training_artifacts(
        out_dir=out_dir,
        prepared=prepared,
        train_log=train_log,
        summary=summary,
        pseudo_info=pseudo_info,
        mode="real_label" if has_real_label else "pseudo_label",
    )

    print(
        f"Training done. rows={len(df_for_train)}, features={len(feature_columns)}, "
        f"best_epoch={summary.get('best_epoch', -1)}"
    )
    print(f"Saved: {out_dir / 'best_model.pt'}")
    print(f"Saved: {out_dir / 'pose_predictions.csv'}")
    print(f"Saved: {artifacts['train_log_csv']}")
    print(f"Saved: {artifacts['feature_columns_json']}")
    print(f"Saved: {artifacts['training_summary_json']}")
    print(f"Saved: {artifacts['training_summary_csv']}")


__all__ = [
    "build_feature_direction_map",
    "filter_low_quality_features",
    "summarize_pseudo_label_distribution",
    "select_feature_columns",
    "build_pseudo_labels",
    "split_by_nanobody_group",
    "prepare_datasets",
    "PoseMLP",
    "train_model",
    "evaluate_model",
    "predict_pose_prob",
    "predict_pose_prob_with_logits",
    "save_training_artifacts",
]


if __name__ == "__main__":
    main()
