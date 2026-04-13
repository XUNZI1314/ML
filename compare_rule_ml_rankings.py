"""Compare rule-based and ML-based nanobody rankings.

Outputs:
1) ranking_comparison_table.csv
2) ranking_comparison_summary.json
3) ranking_comparison_report.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _detect_score_column(df: pd.DataFrame, preferred: list[str]) -> str:
    for col in preferred:
        if col in df.columns:
            return col
    raise ValueError(f"No score column found. Tried: {preferred}")


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(xv) & np.isfinite(yv)
    xv = xv[mask]
    yv = yv[mask]
    if xv.size < 2:
        return float("nan")

    x_center = xv - np.mean(xv)
    y_center = yv - np.mean(yv)
    denom = np.sqrt(np.sum(x_center * x_center) * np.sum(y_center * y_center))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(x_center * y_center) / denom)


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    xr = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    yr = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    return _pearson_corr(xr, yr)


def _kendall_tau_approx(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Kendall tau-a style correlation (ignores tie correction)."""
    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(xv) & np.isfinite(yv)
    xv = xv[mask]
    yv = yv[mask]

    n = int(xv.size)
    if n < 2:
        return float("nan")

    concordant = 0
    discordant = 0

    for i in range(n - 1):
        dx = xv[i + 1 :] - xv[i]
        dy = yv[i + 1 :] - yv[i]
        sign = np.sign(dx) * np.sign(dy)
        concordant += int(np.sum(sign > 0))
        discordant += int(np.sum(sign < 0))

    total = concordant + discordant
    if total == 0:
        return float("nan")
    return float((concordant - discordant) / total)


def _topk_overlap(ids_a: list[str], ids_b: list[str], k: int) -> dict[str, Any]:
    k_eff = max(1, min(int(k), len(ids_a), len(ids_b)))
    set_a = set(ids_a[:k_eff])
    set_b = set(ids_b[:k_eff])
    inter = set_a.intersection(set_b)
    union = set_a.union(set_b)
    return {
        "k": int(k_eff),
        "overlap_count": int(len(inter)),
        "overlap_ratio": float(len(inter) / k_eff) if k_eff > 0 else float("nan"),
        "jaccard": float(len(inter) / len(union)) if len(union) > 0 else float("nan"),
    }


def _binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
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


def _derive_nanobody_labels_from_pose(
    pose_feature_csv: Path,
    label_col: str,
    threshold: float = 0.50,
) -> pd.DataFrame:
    pose_df = pd.read_csv(pose_feature_csv, low_memory=False)
    required = ["nanobody_id", label_col]
    missing = [c for c in required if c not in pose_df.columns]
    if missing:
        raise ValueError(f"pose_feature_csv is missing columns: {missing}")

    work = pose_df[["nanobody_id", label_col]].copy()
    work[label_col] = _to_numeric(work[label_col])
    work = work.loc[work[label_col].notna()].copy()
    if work.empty:
        raise ValueError(f"No valid labels found in pose_feature_csv.{label_col}")

    grouped = work.groupby("nanobody_id", as_index=False)[label_col].mean()
    grouped = grouped.rename(columns={label_col: "nanobody_label_score"})

    thr = float(np.clip(threshold, 0.0, 1.0))
    grouped["nanobody_label"] = (grouped["nanobody_label_score"] >= thr).astype(float)

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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare rule ranking and ML ranking outputs")
    parser.add_argument("--rule_csv", required=True, help="Path to nanobody_rule_ranking.csv")
    parser.add_argument("--ml_csv", required=True, help="Path to nanobody_ranking.csv")
    parser.add_argument("--out_dir", default="comparison_outputs", help="Output directory")
    parser.add_argument("--topk_list", default="5,10,20", help="Comma-separated top-k cutoffs")

    parser.add_argument(
        "--pose_feature_csv",
        default=None,
        help="Optional pose_features.csv path for deriving nanobody labels",
    )
    parser.add_argument("--label_col", default="label", help="Label column in pose_feature_csv")
    parser.add_argument(
        "--nanobody_label_threshold",
        type=float,
        default=0.50,
        help="Threshold on nanobody pose-label mean used to derive nanobody_label",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    rule_csv = Path(args.rule_csv).expanduser()
    ml_csv = Path(args.ml_csv).expanduser()
    if not rule_csv.exists():
        raise FileNotFoundError(f"rule_csv not found: {rule_csv}")
    if not ml_csv.exists():
        raise FileNotFoundError(f"ml_csv not found: {ml_csv}")

    rule_df = pd.read_csv(rule_csv, low_memory=False)
    ml_df = pd.read_csv(ml_csv, low_memory=False)
    if "nanobody_id" not in rule_df.columns or "nanobody_id" not in ml_df.columns:
        raise ValueError("Both CSVs must contain nanobody_id column.")

    rule_score_col = _detect_score_column(rule_df, preferred=["final_score", "final_rule_score", "best_conformer_score"])
    ml_score_col = _detect_score_column(ml_df, preferred=["final_score", "best_conformer_score"])

    rule_view = rule_df[["nanobody_id", rule_score_col]].copy().rename(columns={rule_score_col: "rule_score"})
    ml_view = ml_df[["nanobody_id", ml_score_col]].copy().rename(columns={ml_score_col: "ml_score"})
    merged = rule_view.merge(ml_view, on="nanobody_id", how="inner")
    if merged.empty:
        raise ValueError("No common nanobody_id values between the two ranking files.")

    merged["rule_rank"] = merged["rule_score"].rank(method="min", ascending=False).astype(int)
    merged["ml_rank"] = merged["ml_score"].rank(method="min", ascending=False).astype(int)
    merged["rank_delta"] = merged["rule_rank"] - merged["ml_rank"]
    merged["abs_rank_delta"] = merged["rank_delta"].abs()

    merged = merged.sort_values(by=["abs_rank_delta", "rule_rank"], ascending=[False, True]).reset_index(drop=True)

    score_spearman = _spearman_corr(
        merged["rule_score"].to_numpy(dtype=np.float64),
        merged["ml_score"].to_numpy(dtype=np.float64),
    )
    rank_spearman = _spearman_corr(
        merged["rule_rank"].to_numpy(dtype=np.float64),
        merged["ml_rank"].to_numpy(dtype=np.float64),
    )
    rank_kendall = _kendall_tau_approx(
        merged["rule_rank"].to_numpy(dtype=np.float64),
        merged["ml_rank"].to_numpy(dtype=np.float64),
    )

    rule_sorted_ids = (
        rule_view.sort_values(by="rule_score", ascending=False)["nanobody_id"].astype(str).tolist()
    )
    ml_sorted_ids = (
        ml_view.sort_values(by="ml_score", ascending=False)["nanobody_id"].astype(str).tolist()
    )

    topk_values = []
    for item in str(args.topk_list).split(","):
        text = item.strip()
        if not text:
            continue
        topk_values.append(int(text))
    if not topk_values:
        topk_values = [5, 10, 20]

    topk_overlap = [_topk_overlap(rule_sorted_ids, ml_sorted_ids, k) for k in topk_values]

    summary: dict[str, Any] = {
        "rule_csv": str(rule_csv),
        "ml_csv": str(ml_csv),
        "rule_score_column": rule_score_col,
        "ml_score_column": ml_score_col,
        "n_common_nanobodies": int(len(merged)),
        "score_spearman": float(score_spearman),
        "rank_spearman": float(rank_spearman),
        "rank_kendall_tau": float(rank_kendall),
        "topk_overlap": topk_overlap,
    }

    if args.pose_feature_csv:
        pose_csv = Path(args.pose_feature_csv).expanduser()
        if not pose_csv.exists():
            raise FileNotFoundError(f"pose_feature_csv not found: {pose_csv}")
        label_df = _derive_nanobody_labels_from_pose(
            pose_csv,
            label_col=str(args.label_col),
            threshold=float(args.nanobody_label_threshold),
        )
        with_label = merged.merge(label_df, on="nanobody_id", how="inner")
        summary["n_nanobodies_with_label"] = int(len(with_label))
        summary["nanobody_labeling"] = {
            "mode": "mean_threshold_with_fallback",
            "threshold": float(args.nanobody_label_threshold),
            "positive_rate": float(with_label["nanobody_label"].mean()) if len(with_label) > 0 else float("nan"),
        }
        summary["rule_auc"] = _binary_auc(
            with_label["nanobody_label"].to_numpy(dtype=np.float64),
            with_label["rule_score"].to_numpy(dtype=np.float64),
        )
        summary["ml_auc"] = _binary_auc(
            with_label["nanobody_label"].to_numpy(dtype=np.float64),
            with_label["ml_score"].to_numpy(dtype=np.float64),
        )

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    table_csv = out_dir / "ranking_comparison_table.csv"
    summary_json = out_dir / "ranking_comparison_summary.json"
    report_md = out_dir / "ranking_comparison_report.md"

    merged.to_csv(table_csv, index=False)
    summary_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")

    report_lines = [
        "# Rule vs ML Ranking Comparison",
        "",
        f"- Common nanobodies: {summary['n_common_nanobodies']}",
        f"- Spearman(score): {summary['score_spearman']:.4f}",
        f"- Spearman(rank): {summary['rank_spearman']:.4f}",
        f"- Kendall tau(rank): {summary['rank_kendall_tau']:.4f}",
        "",
        "## Top-K Overlap",
        "",
        "| K | Overlap Count | Overlap Ratio | Jaccard |",
        "|---:|---:|---:|---:|",
    ]
    for item in topk_overlap:
        report_lines.append(
            f"| {item['k']} | {item['overlap_count']} | {item['overlap_ratio']:.4f} | {item['jaccard']:.4f} |"
        )

    if "rule_auc" in summary and "ml_auc" in summary:
        report_lines.extend(
            [
                "",
                "## Label-Based AUC",
                "",
                f"- Rule AUC: {float(summary['rule_auc']):.4f}",
                f"- ML AUC: {float(summary['ml_auc']):.4f}",
            ]
        )

    report_lines.extend(
        [
            "",
            "## Most Divergent Samples (Top 15 by |rank delta|)",
            "",
            "| nanobody_id | rule_rank | ml_rank | rank_delta | rule_score | ml_score |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in merged.head(15).iterrows():
        report_lines.append(
            "| "
            f"{row['nanobody_id']} | "
            f"{int(row['rule_rank'])} | "
            f"{int(row['ml_rank'])} | "
            f"{int(row['rank_delta'])} | "
            f"{float(row['rule_score']):.4f} | "
            f"{float(row['ml_score']):.4f} |"
        )

    report_md.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Comparison done. Common nanobodies={len(merged)}")
    print(f"Saved: {table_csv}")
    print(f"Saved: {summary_json}")
    print(f"Saved: {report_md}")


if __name__ == "__main__":
    main()
