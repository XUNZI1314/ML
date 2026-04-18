from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from rank_nanobodies import (
    aggregate_conformer_scores,
    aggregate_nanobody_scores,
    rank_nanobodies,
)
from rule_ranker import (
    aggregate_rule_scores,
    build_rule_score,
    rank_nanobodies_by_rules,
)
from train_pose_model import (
    PoseMLP,
    _binary_auc,
    _binary_classification_metrics,
    build_feature_direction_map,
    build_pseudo_labels,
    predict_pose_prob_with_logits,
    prepare_datasets,
    save_training_artifacts,
    select_feature_columns,
    train_model,
)


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _brier_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(np.mean((y_score.astype(float) - y_true.astype(float)) ** 2))


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.allclose(np.nanstd(x), 0.0) or np.allclose(np.nanstd(y), 0.0):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    x_rank = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    y_rank = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    return _pearson_corr(x_rank, y_rank)


def _build_reliability_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int,
) -> pd.DataFrame:
    if y_true.size == 0 or y_score.size == 0:
        return pd.DataFrame(
            columns=[
                "bin_index",
                "bin_left",
                "bin_right",
                "count",
                "mean_score",
                "positive_rate",
                "abs_gap",
            ]
        )

    bins = max(2, int(n_bins))
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, Any]] = []
    clipped = np.clip(y_score.astype(float), 0.0, 1.0)
    labels = y_true.astype(float)
    for idx in range(bins):
        left = float(edges[idx])
        right = float(edges[idx + 1])
        if idx == bins - 1:
            mask = (clipped >= left) & (clipped <= right)
        else:
            mask = (clipped >= left) & (clipped < right)
        count = int(np.sum(mask))
        if count > 0:
            mean_score = float(np.mean(clipped[mask]))
            positive_rate = float(np.mean(labels[mask]))
            abs_gap = float(abs(mean_score - positive_rate))
        else:
            mean_score = float("nan")
            positive_rate = float("nan")
            abs_gap = float("nan")
        rows.append(
            {
                "bin_index": idx,
                "bin_left": left,
                "bin_right": right,
                "count": count,
                "mean_score": mean_score,
                "positive_rate": positive_rate,
                "abs_gap": abs_gap,
            }
        )
    return pd.DataFrame(rows)


def _reliability_ece(curve_df: pd.DataFrame) -> float:
    if curve_df.empty or "count" not in curve_df.columns:
        return float("nan")
    counts = _to_numeric(curve_df["count"]).fillna(0.0).to_numpy(dtype=float)
    gaps = _to_numeric(curve_df["abs_gap"]).fillna(0.0).to_numpy(dtype=float)
    total = float(np.sum(counts))
    if total <= 0.0:
        return float("nan")
    return float(np.sum((counts / total) * gaps))


def _summarize_scalar(series: pd.Series) -> dict[str, float]:
    values = _to_numeric(series).to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=0)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def _build_feature_proxy_benchmark(
    df: pd.DataFrame,
    fold_col: str = "cv_fold",
) -> pd.DataFrame:
    feature_spec = build_feature_direction_map()
    if "label" not in df.columns:
        return pd.DataFrame()

    score_source = "label_score" if "label_score" in df.columns else "label"
    scopes: list[Any] = sorted(pd.unique(df[fold_col]).tolist()) if fold_col in df.columns else []
    scopes.append("all")

    rows: list[dict[str, Any]] = []
    for scope in scopes:
        scope_df = df if scope == "all" else df.loc[df[fold_col] == scope]
        y_true = _to_numeric(scope_df["label"]).clip(0.0, 1.0)
        y_score_ref = _to_numeric(scope_df[score_source]).clip(0.0, 1.0)
        for feature, (sign, weight) in feature_spec.items():
            if feature not in scope_df.columns:
                continue
            x = _to_numeric(scope_df[feature])
            mask = x.notna() & y_true.notna() & y_score_ref.notna()
            if int(mask.sum()) < 2:
                continue

            feat = x.loc[mask].to_numpy(dtype=float)
            y = y_true.loc[mask].to_numpy(dtype=float)
            y_ref = y_score_ref.loc[mask].to_numpy(dtype=float)
            aligned = feat if sign >= 0 else -feat
            pos_mask = y >= 0.5
            neg_mask = y < 0.5
            rows.append(
                {
                    "cv_scope": scope,
                    "feature": feature,
                    "direction_sign": int(sign),
                    "default_weight": float(weight),
                    "n_rows": int(mask.sum()),
                    "proxy_auc": float(_binary_auc(y, aligned)),
                    "proxy_rank_spearman": float(_spearman_corr(aligned, y_ref)),
                    "feature_mean_all": float(np.nanmean(feat)),
                    "feature_mean_pos": float(np.nanmean(feat[pos_mask])) if pos_mask.any() else float("nan"),
                    "feature_mean_neg": float(np.nanmean(feat[neg_mask])) if neg_mask.any() else float("nan"),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        by=["cv_scope", "proxy_auc", "proxy_rank_spearman", "default_weight"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)


def _snake_fold_order(n_groups: int, n_folds: int) -> list[int]:
    order: list[int] = []
    while len(order) < n_groups:
        order.extend(list(range(n_folds)))
        if len(order) >= n_groups:
            break
        if n_folds > 1:
            order.extend(list(range(n_folds - 1, -1, -1)))
    return order[:n_groups]


def _build_group_folds(
    df: pd.DataFrame,
    group_col: str,
    label_col: str,
    n_folds: int,
    seed: int,
) -> list[list[str]]:
    if group_col not in df.columns:
        raise ValueError(f"Missing group column: {group_col}")
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    work = df[[group_col, label_col]].copy()
    work[label_col] = _to_numeric(work[label_col]).clip(0.0, 1.0)
    work = work.loc[work[group_col].notna() & work[label_col].notna()].reset_index(drop=True)
    if work.empty:
        raise ValueError("No valid rows available for grouped fold split.")

    grouped = (
        work.groupby(group_col, dropna=False)[label_col]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "label_mean", "count": "row_count"})
    )
    rng = np.random.default_rng(int(seed))
    grouped["tie_breaker"] = rng.random(len(grouped))
    grouped = grouped.sort_values(
        by=["label_mean", "row_count", "tie_breaker"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    folds = max(2, min(int(n_folds), int(len(grouped))))
    assignment = _snake_fold_order(n_groups=len(grouped), n_folds=folds)
    fold_groups: list[list[str]] = [[] for _ in range(folds)]
    for row_idx, fold_idx in enumerate(assignment):
        fold_groups[fold_idx].append(str(grouped.iloc[row_idx][group_col]))
    return [groups for groups in fold_groups if groups]


def _derive_nanobody_labels_from_df(df: pd.DataFrame) -> pd.DataFrame:
    if "nanobody_id" not in df.columns:
        raise ValueError("Missing nanobody_id column.")
    if "label" not in df.columns:
        raise ValueError("Missing label column.")

    work = df.copy()
    work["label"] = _to_numeric(work["label"]).clip(0.0, 1.0)
    if "label_score" in work.columns:
        work["label_score"] = _to_numeric(work["label_score"]).clip(0.0, 1.0)
    else:
        work["label_score"] = work["label"]

    grouped = (
        work.groupby("nanobody_id", dropna=False)
        .agg(
            pose_label_positive_rate=("label", "mean"),
            nanobody_label=("label", "max"),
            label_score=("label_score", "mean"),
            n_pose_rows=("label", "count"),
        )
        .reset_index()
    )
    grouped["nanobody_label"] = (grouped["nanobody_label"].fillna(0.0) >= 0.5).astype(int)
    return grouped


def _resolve_device(device_arg: str | None) -> str:
    if device_arg:
        device = str(device_arg).strip().lower()
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _prepare_ml_training_frame(
    train_df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, str | None, dict[str, Any]]:
    pseudo_info: dict[str, Any] = {
        "used_columns": [],
        "pseudo_positive_rate": float("nan"),
        "threshold": {"mode": "none"},
    }
    try:
        df_with_pseudo, pseudo_info = build_pseudo_labels(
            train_df.copy(),
            top_fraction=float(args.top_pseudo_fraction),
            threshold_mode=str(args.pseudo_threshold_mode),
            threshold_value=args.pseudo_threshold_value,
        )
        return df_with_pseudo, "pseudo_score", pseudo_info
    except Exception as exc:
        pseudo_info["warning"] = str(exc)
        return train_df.copy(), None, pseudo_info


def _build_pose_prediction_frame(
    test_df: pd.DataFrame,
    pred_prob: np.ndarray,
    pred_logit: np.ndarray,
    fold_id: int,
) -> pd.DataFrame:
    out = test_df.reset_index(drop=True).copy()
    out["cv_fold"] = int(fold_id)
    out["pred_prob"] = np.clip(np.asarray(pred_prob, dtype=float), 0.0, 1.0)
    out["pred_logit"] = np.asarray(pred_logit, dtype=float)
    out["label"] = _to_numeric(out["label"]).clip(0.0, 1.0)
    return out


def _build_pose_metrics(
    pose_pred_df: pd.DataFrame,
    reliability_bins: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    y_true = _to_numeric(pose_pred_df["label"]).fillna(0.0).to_numpy(dtype=float)
    y_score = _to_numeric(pose_pred_df["pred_prob"]).fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
    cls = _binary_classification_metrics(y_true, y_score, threshold=0.5)
    curve_df = _build_reliability_curve(y_true=y_true, y_score=y_score, n_bins=reliability_bins)
    metrics = {
        "pose_auc": float(_binary_auc(y_true, y_score)),
        "pose_brier": float(_brier_score(y_true, y_score)),
        "pose_ece": float(_reliability_ece(curve_df)),
        "pose_accuracy": float(cls["accuracy"]),
        "pose_precision": float(cls["precision"]),
        "pose_recall": float(cls["recall"]),
        "pose_f1": float(cls["f1"]),
    }
    return metrics, curve_df


def _build_nanobody_method_metrics(
    ranked_df: pd.DataFrame,
    score_col: str,
    label_col: str,
    label_score_col: str,
    reliability_bins: int,
    prefix: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    work = ranked_df.copy()
    y_true = _to_numeric(work[label_col]).fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
    y_score = _to_numeric(work[score_col]).fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
    label_score = _to_numeric(work[label_score_col]).fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)

    curve_df = _build_reliability_curve(y_true=y_true, y_score=y_score, n_bins=reliability_bins)
    metrics = {
        f"{prefix}_auc": float(_binary_auc(y_true, y_score)),
        f"{prefix}_brier": float(_brier_score(y_true, y_score)),
        f"{prefix}_ece": float(_reliability_ece(curve_df)),
        f"{prefix}_rank_spearman": float(_spearman_corr(y_score, label_score)),
    }
    return metrics, curve_df


def _merge_rule_predictions(
    pose_rule_df: pd.DataFrame,
    pose_pred_df: pd.DataFrame,
) -> pd.DataFrame:
    key_cols = [c for c in ["nanobody_id", "conformer_id", "pose_id"] if c in pose_rule_df.columns and c in pose_pred_df.columns]
    if len(key_cols) < 3:
        return pose_rule_df
    score_cols = pose_pred_df[key_cols + ["pred_prob", "pred_logit"]].copy()
    merged = pose_rule_df.merge(score_cols, on=key_cols, how="left")
    return merged


def _run_single_fold(
    df: pd.DataFrame,
    test_groups: list[str],
    fold_id: int,
    out_dir: Path,
    args: argparse.Namespace,
    device: str,
) -> dict[str, Any]:
    test_mask = df["nanobody_id"].astype(str).isin(test_groups)
    train_df = df.loc[~test_mask].reset_index(drop=True)
    test_df = df.loc[test_mask].reset_index(drop=True)
    if train_df.empty or test_df.empty:
        raise ValueError(f"Fold {fold_id} produced an empty train/test split.")

    fold_dir = out_dir / "folds" / f"fold_{fold_id:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_for_model, soft_column, pseudo_info = _prepare_ml_training_frame(train_df=train_df, args=args)
    feature_columns = select_feature_columns(
        train_for_model,
        min_non_nan_ratio=float(args.feature_min_non_nan_ratio),
        max_missing_ratio=float(args.feature_max_missing_ratio),
        near_constant_ratio=float(args.feature_near_constant_ratio),
    )
    prepared = prepare_datasets(
        train_for_model,
        feature_columns=feature_columns,
        label_column="label",
        soft_target_column=soft_column,
        val_ratio=float(args.val_ratio),
        group_col="nanobody_id",
        seed=int(args.seed) + int(fold_id),
    )
    model = PoseMLP(input_dim=len(feature_columns))
    model, train_log, train_summary = train_model(
        model=model,
        prepared=prepared,
        out_dir=fold_dir,
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
        device=device,
    )
    save_training_artifacts(
        out_dir=fold_dir,
        prepared=prepared,
        train_log=train_log,
        summary=train_summary,
        pseudo_info=pseudo_info,
        mode="real_label_cv",
    )

    pred_prob, pred_logit = predict_pose_prob_with_logits(
        model=model,
        df=test_df,
        feature_columns=prepared.feature_columns,
        fill_values=prepared.fill_values,
        feature_mean=prepared.feature_mean,
        feature_std=prepared.feature_std,
        batch_size=int(args.batch_size),
        device=device,
    )
    pose_pred_df = _build_pose_prediction_frame(test_df=test_df, pred_prob=pred_prob, pred_logit=pred_logit, fold_id=fold_id)
    pose_pred_df.to_csv(fold_dir / "pose_cv_predictions.csv", index=False)

    pose_metrics, pose_curve_df = _build_pose_metrics(
        pose_pred_df=pose_pred_df,
        reliability_bins=int(args.reliability_bins),
    )
    pose_curve_df.insert(0, "cv_fold", int(fold_id))
    pose_curve_df.to_csv(fold_dir / "pose_reliability_curve.csv", index=False)

    conformer_df = aggregate_conformer_scores(
        pose_pred_df,
        top_k=int(args.top_k),
        blend_optional=bool(args.blend_optional),
        optional_weight=float(args.optional_weight),
    )
    ml_nanobody_df = aggregate_nanobody_scores(
        conformer_df,
        alpha=float(args.alpha),
        mean_weight=float(args.w_mean),
        best_weight=float(args.w_best),
        consistency_weight=float(args.w_consistency),
        std_penalty_weight=float(args.w_std_penalty),
        consistency_hit_threshold=float(args.consistency_hit_threshold),
    )
    ml_ranking_df = rank_nanobodies(ml_nanobody_df)

    pose_rule_df, rule_info = build_rule_score(test_df.copy())
    pose_rule_df = _merge_rule_predictions(pose_rule_df=pose_rule_df, pose_pred_df=pose_pred_df)
    rule_conformer_df, rule_nanobody_df = aggregate_rule_scores(
        pose_rule_df,
        top_k=int(args.top_k),
        conformer_geo_weight=float(args.optional_weight),
        mean_weight=float(args.w_mean),
        best_weight=float(args.w_best),
        consistency_weight=float(args.w_consistency),
        std_penalty_weight=float(args.w_std_penalty),
        consistency_hit_threshold=float(args.consistency_hit_threshold),
    )
    rule_ranking_df = rank_nanobodies_by_rules(rule_nanobody_df)

    truth_df = _derive_nanobody_labels_from_df(test_df)
    ml_eval_df = ml_ranking_df.merge(truth_df, on="nanobody_id", how="left")
    rule_eval_df = rule_ranking_df.merge(truth_df, on="nanobody_id", how="left")
    ml_eval_df.insert(0, "cv_fold", int(fold_id))
    rule_eval_df.insert(0, "cv_fold", int(fold_id))
    ml_eval_df.to_csv(fold_dir / "nanobody_ranking_ml.csv", index=False)
    rule_eval_df.to_csv(fold_dir / "nanobody_ranking_rule.csv", index=False)

    ml_metrics, ml_curve_df = _build_nanobody_method_metrics(
        ranked_df=ml_eval_df,
        score_col="final_score",
        label_col="nanobody_label",
        label_score_col="label_score",
        reliability_bins=int(args.reliability_bins),
        prefix="ml_nanobody",
    )
    rule_metrics, rule_curve_df = _build_nanobody_method_metrics(
        ranked_df=rule_eval_df,
        score_col="final_score",
        label_col="nanobody_label",
        label_score_col="label_score",
        reliability_bins=int(args.reliability_bins),
        prefix="rule_nanobody",
    )
    ml_curve_df.insert(0, "method", "ml")
    ml_curve_df.insert(0, "cv_fold", int(fold_id))
    rule_curve_df.insert(0, "method", "rule")
    rule_curve_df.insert(0, "cv_fold", int(fold_id))
    pd.concat([ml_curve_df, rule_curve_df], ignore_index=True).to_csv(
        fold_dir / "nanobody_reliability_curve.csv",
        index=False,
    )

    fold_metrics = {
        "cv_fold": int(fold_id),
        "n_train_rows": int(len(train_df)),
        "n_test_rows": int(len(test_df)),
        "n_test_nanobodies": int(truth_df["nanobody_id"].nunique()),
        "n_features": int(len(feature_columns)),
        "best_epoch": int(train_summary.get("best_epoch", -1)),
        "best_val_loss": float(train_summary.get("best_val_loss", np.nan)),
        "epochs_ran": int(train_summary.get("epochs_ran", 0)),
        "pseudo_feature_count": int(len(pseudo_info.get("used_columns", []))),
        "rule_feature_count": int(rule_info.get("feature_count", 0)),
        **pose_metrics,
        **ml_metrics,
        **rule_metrics,
    }
    fold_metrics["nanobody_auc_delta_ml_minus_rule"] = float(
        fold_metrics["ml_nanobody_auc"] - fold_metrics["rule_nanobody_auc"]
        if np.isfinite(fold_metrics["ml_nanobody_auc"]) and np.isfinite(fold_metrics["rule_nanobody_auc"])
        else np.nan
    )

    return {
        "fold_metrics": fold_metrics,
        "pose_predictions": pose_pred_df,
        "pose_curve": pose_curve_df,
        "ml_eval": ml_eval_df,
        "rule_eval": rule_eval_df,
        "ml_curve": ml_curve_df,
        "rule_curve": rule_curve_df,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run grouped cross-validation benchmark for pose ML pipeline.")
    parser.add_argument("--feature_csv", required=True, help="Path to pose_features.csv with real labels.")
    parser.add_argument("--out_dir", default="benchmark_outputs", help="Output directory")
    parser.add_argument("--folds", type=int, default=5, help="Grouped CV fold count")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="cpu / cuda / auto")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--soft_target_weight", type=float, default=0.25)
    parser.add_argument("--top_pseudo_fraction", type=float, default=0.25)
    parser.add_argument(
        "--pseudo_threshold_mode",
        choices=["top_fraction", "quantile", "fixed"],
        default="top_fraction",
    )
    parser.add_argument("--pseudo_threshold_value", type=float, default=None)
    parser.add_argument("--feature_min_non_nan_ratio", type=float, default=0.05)
    parser.add_argument("--feature_max_missing_ratio", type=float, default=0.80)
    parser.add_argument("--feature_near_constant_ratio", type=float, default=0.995)
    parser.add_argument("--early_stopping_patience", type=int, default=12)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--lr_scheduler_patience", type=int, default=5)
    parser.add_argument("--lr_scheduler_factor", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--blend_optional", action="store_true", help="Blend optional geometry signal in conformer aggregation")
    parser.add_argument("--optional_weight", type=float, default=0.15)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--w_mean", type=float, default=0.50)
    parser.add_argument("--w_best", type=float, default=0.25)
    parser.add_argument("--w_consistency", type=float, default=0.20)
    parser.add_argument("--w_std_penalty", type=float, default=0.15)
    parser.add_argument("--consistency_hit_threshold", type=float, default=0.50)
    parser.add_argument("--reliability_bins", type=int, default=10)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    feature_csv = Path(args.feature_csv).expanduser().resolve()
    if not feature_csv.exists():
        raise FileNotFoundError(f"Feature CSV not found: {feature_csv}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(feature_csv, low_memory=False)
    if df.empty:
        raise ValueError("Input feature CSV is empty.")
    required_cols = {"nanobody_id", "conformer_id", "pose_id", "label"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for benchmark: {missing}")

    if "status" in df.columns:
        ok_df = df[df["status"].astype(str).str.lower() == "ok"].copy()
        if len(ok_df) > 0:
            df = ok_df.reset_index(drop=True)

    df["label"] = _to_numeric(df["label"]).clip(0.0, 1.0)
    df = df.loc[df["label"].notna()].reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid labeled rows left after preprocessing.")
    if int(df["label"].nunique()) < 2:
        raise ValueError("Benchmark requires at least two label classes.")

    device = _resolve_device(args.device)
    fold_groups = _build_group_folds(
        df=df,
        group_col="nanobody_id",
        label_col="label",
        n_folds=int(args.folds),
        seed=int(args.seed),
    )

    fold_results: list[dict[str, Any]] = []
    for fold_id, test_groups in enumerate(fold_groups, start=1):
        fold_results.append(
            _run_single_fold(
                df=df,
                test_groups=test_groups,
                fold_id=fold_id,
                out_dir=out_dir,
                args=args,
                device=device,
            )
        )

    fold_metrics_df = pd.DataFrame([item["fold_metrics"] for item in fold_results])
    fold_metrics_df.to_csv(out_dir / "fold_metrics.csv", index=False)

    pose_predictions_df = pd.concat([item["pose_predictions"] for item in fold_results], ignore_index=True)
    pose_predictions_df.to_csv(out_dir / "pose_cv_predictions.csv", index=False)
    proxy_benchmark_df = _build_feature_proxy_benchmark(pose_predictions_df)
    if not proxy_benchmark_df.empty:
        proxy_benchmark_df.to_csv(out_dir / "geometry_proxy_benchmark.csv", index=False)

    pose_curve_all = _build_reliability_curve(
        y_true=_to_numeric(pose_predictions_df["label"]).fillna(0.0).to_numpy(dtype=float),
        y_score=_to_numeric(pose_predictions_df["pred_prob"]).fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float),
        n_bins=int(args.reliability_bins),
    )
    pose_curve_all.insert(0, "cv_fold", "all")
    pose_curve_df = pd.concat(
        [pd.concat([item["pose_curve"]], ignore_index=True) for item in fold_results] + [pose_curve_all],
        ignore_index=True,
    )
    pose_curve_df.to_csv(out_dir / "pose_reliability_curve.csv", index=False)

    ml_eval_df = pd.concat([item["ml_eval"] for item in fold_results], ignore_index=True)
    rule_eval_df = pd.concat([item["rule_eval"] for item in fold_results], ignore_index=True)

    ml_curve_all = _build_reliability_curve(
        y_true=_to_numeric(ml_eval_df["nanobody_label"]).fillna(0.0).to_numpy(dtype=float),
        y_score=_to_numeric(ml_eval_df["final_score"]).fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float),
        n_bins=int(args.reliability_bins),
    )
    ml_curve_all.insert(0, "method", "ml")
    ml_curve_all.insert(0, "cv_fold", "all")
    rule_curve_all = _build_reliability_curve(
        y_true=_to_numeric(rule_eval_df["nanobody_label"]).fillna(0.0).to_numpy(dtype=float),
        y_score=_to_numeric(rule_eval_df["final_score"]).fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float),
        n_bins=int(args.reliability_bins),
    )
    rule_curve_all.insert(0, "method", "rule")
    rule_curve_all.insert(0, "cv_fold", "all")

    nanobody_curve_df = pd.concat(
        [item["ml_curve"] for item in fold_results]
        + [item["rule_curve"] for item in fold_results]
        + [ml_curve_all, rule_curve_all],
        ignore_index=True,
    )
    nanobody_curve_df.to_csv(out_dir / "nanobody_reliability_curve.csv", index=False)

    nanobody_benchmark_df = (
        ml_eval_df.rename(
            columns={
                "rank": "ml_rank",
                "final_score": "ml_final_score",
                "explanation": "ml_explanation",
            }
        )
        .merge(
            rule_eval_df.rename(
                columns={
                    "rank": "rule_rank",
                    "final_score": "rule_final_score",
                    "explanation": "rule_explanation",
                }
            )[
                [
                    "cv_fold",
                    "nanobody_id",
                    "rule_rank",
                    "rule_final_score",
                    "rule_explanation",
                ]
            ],
            on=["cv_fold", "nanobody_id"],
            how="outer",
        )
    )
    nanobody_benchmark_df["score_delta_ml_minus_rule"] = (
        _to_numeric(nanobody_benchmark_df["ml_final_score"]).fillna(np.nan)
        - _to_numeric(nanobody_benchmark_df["rule_final_score"]).fillna(np.nan)
    )
    nanobody_benchmark_df.to_csv(out_dir / "nanobody_benchmark_table.csv", index=False)

    overall_pose_metrics, _ = _build_pose_metrics(
        pose_pred_df=pose_predictions_df,
        reliability_bins=int(args.reliability_bins),
    )
    overall_ml_metrics, _ = _build_nanobody_method_metrics(
        ranked_df=ml_eval_df,
        score_col="final_score",
        label_col="nanobody_label",
        label_score_col="label_score",
        reliability_bins=int(args.reliability_bins),
        prefix="ml_nanobody",
    )
    overall_rule_metrics, _ = _build_nanobody_method_metrics(
        ranked_df=rule_eval_df,
        score_col="final_score",
        label_col="nanobody_label",
        label_score_col="label_score",
        reliability_bins=int(args.reliability_bins),
        prefix="rule_nanobody",
    )

    summary_payload = {
        "feature_csv": str(feature_csv),
        "out_dir": str(out_dir),
        "fold_count": int(len(fold_groups)),
        "device": str(device),
        "rows": {
            "pose_rows": int(len(pose_predictions_df)),
            "nanobody_rows": int(len(nanobody_benchmark_df)),
            "unique_nanobodies": int(pose_predictions_df["nanobody_id"].astype(str).nunique()),
        },
        "settings": vars(args),
        "overall_pose_metrics": overall_pose_metrics,
        "overall_ml_nanobody_metrics": overall_ml_metrics,
        "overall_rule_nanobody_metrics": overall_rule_metrics,
        "geometry_proxy_summary": (
            proxy_benchmark_df.loc[proxy_benchmark_df["cv_scope"].astype(str) == "all", ["feature", "proxy_auc", "proxy_rank_spearman"]]
            .head(10)
            .to_dict(orient="records")
            if not proxy_benchmark_df.empty
            else []
        ),
        "fold_metric_summary": {
            column: _summarize_scalar(fold_metrics_df[column])
            for column in fold_metrics_df.columns
            if column != "cv_fold"
        },
    }
    with (out_dir / "benchmark_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=True, indent=2)

    report_lines = [
        "# Pose Benchmark Report",
        "",
        f"- feature_csv: `{feature_csv}`",
        f"- out_dir: `{out_dir}`",
        f"- folds: `{len(fold_groups)}`",
        f"- device: `{device}`",
        "",
        "## Overall Metrics",
        "",
        f"- pose_auc: `{overall_pose_metrics['pose_auc']:.4f}`",
        f"- pose_brier: `{overall_pose_metrics['pose_brier']:.4f}`",
        f"- pose_ece: `{overall_pose_metrics['pose_ece']:.4f}`",
        f"- ml_nanobody_auc: `{overall_ml_metrics['ml_nanobody_auc']:.4f}`",
        f"- rule_nanobody_auc: `{overall_rule_metrics['rule_nanobody_auc']:.4f}`",
        f"- ml_nanobody_rank_spearman: `{overall_ml_metrics['ml_nanobody_rank_spearman']:.4f}`",
        f"- rule_nanobody_rank_spearman: `{overall_rule_metrics['rule_nanobody_rank_spearman']:.4f}`",
        "",
        "## Outputs",
        "",
        "- `fold_metrics.csv`",
        "- `pose_cv_predictions.csv`",
        "- `nanobody_benchmark_table.csv`",
        "- `pose_reliability_curve.csv`",
        "- `nanobody_reliability_curve.csv`",
        "- `geometry_proxy_benchmark.csv`",
        "- `benchmark_summary.json`",
    ]
    (out_dir / "benchmark_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Saved: {out_dir / 'fold_metrics.csv'}")
    print(f"Saved: {out_dir / 'pose_cv_predictions.csv'}")
    print(f"Saved: {out_dir / 'nanobody_benchmark_table.csv'}")
    if not proxy_benchmark_df.empty:
        print(f"Saved: {out_dir / 'geometry_proxy_benchmark.csv'}")
    print(f"Saved: {out_dir / 'benchmark_summary.json'}")
    print(f"Saved: {out_dir / 'benchmark_report.md'}")


if __name__ == "__main__":
    main()
