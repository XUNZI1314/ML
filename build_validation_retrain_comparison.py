from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


METRIC_DIRECTIONS = {
    "label_valid_count": "higher",
    "label_class_count": "higher",
    "baseline_rank_spearman": "higher",
    "baseline_rule_auc": "higher",
    "calibrated_rank_spearman": "higher",
    "calibrated_rule_auc": "higher",
    "best_val_loss": "lower",
    "failed_rows": "lower",
    "warning_rows": "lower",
    "top_k_overlap_fraction": "context",
    "validated_top_k_count": "higher",
}

RANK_CANDIDATES = ["consensus_rank", "rank", "nanobody_rank", "final_rank"]
SCORE_CANDIDATES = ["consensus_score", "final_score", "ml_score", "rule_score", "best_conformer_score"]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _to_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _resolve_summary_path(path_text: str | Path) -> Path:
    path = Path(path_text).expanduser().resolve()
    if path.is_dir():
        path = path / "recommended_pipeline_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"summary not found: {path}")
    return path


def _out_dir_from_summary(summary: dict[str, Any], summary_path: Path) -> Path:
    out_dir_text = str(summary.get("out_dir") or "").strip()
    if out_dir_text:
        return Path(out_dir_text).expanduser().resolve()
    return summary_path.parent.resolve()


def _resolve_artifact_path(
    summary: dict[str, Any],
    summary_path: Path,
    artifact_key: str,
    fallback_parts: tuple[str, ...],
) -> Path | None:
    artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
    artifact_text = str(artifacts.get(artifact_key) or "").strip()
    candidates: list[Path] = []
    if artifact_text:
        artifact_path = Path(artifact_text).expanduser()
        if not artifact_path.is_absolute():
            artifact_path = summary_path.parent / artifact_path
        candidates.append(artifact_path.resolve())

    out_dir = _out_dir_from_summary(summary, summary_path)
    if fallback_parts:
        candidates.append(out_dir.joinpath(*fallback_parts).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else None


def _load_training_summary(summary: dict[str, Any], summary_path: Path) -> dict[str, Any]:
    path = _resolve_artifact_path(summary, summary_path, "training_summary_json", ("model_outputs", "training_summary.json"))
    return _read_json(path) if path is not None else {}


def _load_feature_qc_summary(summary: dict[str, Any], summary_path: Path) -> dict[str, Any]:
    artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
    feature_qc_text = str(artifacts.get("feature_qc_json") or "").strip()
    if feature_qc_text:
        path = Path(feature_qc_text).expanduser()
        if not path.is_absolute():
            path = summary_path.parent / path
        return _read_json(path.resolve())
    return {}


def _extract_run_metrics(summary: dict[str, Any], summary_path: Path) -> dict[str, Any]:
    comparison = summary.get("comparison") if isinstance(summary.get("comparison"), dict) else {}
    baseline = comparison.get("baseline_rule_vs_ml") if isinstance(comparison.get("baseline_rule_vs_ml"), dict) else {}
    calibrated = comparison.get("calibrated_rule_vs_ml") if isinstance(comparison.get("calibrated_rule_vs_ml"), dict) else {}

    training_payload = _load_training_summary(summary, summary_path)
    training_summary = (
        training_payload.get("summary")
        if isinstance(training_payload, dict) and isinstance(training_payload.get("summary"), dict)
        else {}
    )
    qc_payload = _load_feature_qc_summary(summary, summary_path)
    processing_summary = (
        qc_payload.get("processing_summary")
        if isinstance(qc_payload, dict) and isinstance(qc_payload.get("processing_summary"), dict)
        else {}
    )

    return {
        "summary_path": str(summary_path),
        "out_dir": str(_out_dir_from_summary(summary, summary_path)),
        "start_mode": summary.get("start_mode"),
        "feature_csv": summary.get("feature_csv"),
        "label_col": summary.get("label_col"),
        "label_valid_count": summary.get("label_valid_count"),
        "label_class_count": summary.get("label_class_count"),
        "calibration_possible": summary.get("calibration_possible"),
        "baseline_rank_spearman": baseline.get("rank_spearman"),
        "baseline_rule_auc": baseline.get("rule_auc"),
        "calibrated_rank_spearman": calibrated.get("rank_spearman"),
        "calibrated_rule_auc": calibrated.get("rule_auc"),
        "training_mode": training_payload.get("mode") if isinstance(training_payload, dict) else None,
        "train_rows": training_payload.get("n_rows_train") if isinstance(training_payload, dict) else None,
        "val_rows": training_payload.get("n_rows_val") if isinstance(training_payload, dict) else None,
        "feature_count": training_payload.get("n_features") if isinstance(training_payload, dict) else None,
        "best_epoch": training_summary.get("best_epoch"),
        "best_val_loss": training_summary.get("best_val_loss"),
        "failed_rows": processing_summary.get("failed_rows"),
        "warning_rows": processing_summary.get("rows_with_warning_message"),
    }


def _detect_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def _load_consensus_ranking(summary: dict[str, Any], summary_path: Path) -> pd.DataFrame:
    path = _resolve_artifact_path(
        summary,
        summary_path,
        "consensus_ranking_csv",
        ("consensus_outputs", "consensus_ranking.csv"),
    )
    df = _load_csv(path)
    if df.empty or "nanobody_id" not in df.columns:
        return pd.DataFrame(columns=["nanobody_id", "rank", "score"])

    work = df.copy()
    rank_col = _detect_column(work, RANK_CANDIDATES)
    score_col = _detect_column(work, SCORE_CANDIDATES)
    if rank_col is not None:
        work["_rank"] = pd.to_numeric(work[rank_col], errors="coerce")
    else:
        work["_rank"] = float("nan")
    if score_col is not None:
        work["_score"] = pd.to_numeric(work[score_col], errors="coerce")
    else:
        work["_score"] = float("nan")

    if work["_rank"].notna().any():
        work = work.sort_values(["_rank"], ascending=True, na_position="last", kind="stable")
    elif work["_score"].notna().any():
        work = work.sort_values(["_score"], ascending=False, na_position="last", kind="stable")
    work = work.reset_index(drop=True)
    if not work["_rank"].notna().any():
        work["_rank"] = work.index + 1

    return pd.DataFrame(
        {
            "nanobody_id": work["nanobody_id"].map(_clean_text),
            "rank": pd.to_numeric(work["_rank"], errors="coerce"),
            "score": pd.to_numeric(work["_score"], errors="coerce"),
        }
    )


def _load_validation_labels(path_text: str | Path | None) -> pd.DataFrame:
    if path_text is None or not str(path_text).strip():
        return pd.DataFrame(columns=["nanobody_id", "validation_label"])
    path = Path(path_text).expanduser().resolve()
    df = _load_csv(path)
    if df.empty or "nanobody_id" not in df.columns:
        return pd.DataFrame(columns=["nanobody_id", "validation_label"])
    label_col = "validation_label" if "validation_label" in df.columns else None
    if label_col is None:
        for candidate in ["experiment_label", "label"]:
            if candidate in df.columns:
                label_col = candidate
                break
    if label_col is None:
        return pd.DataFrame(columns=["nanobody_id", "validation_label"])
    return df.loc[:, ["nanobody_id", label_col]].rename(columns={label_col: "validation_label"}).drop_duplicates(
        subset=["nanobody_id"],
        keep="last",
    )


def _top_ids(df: pd.DataFrame, top_k: int) -> list[str]:
    if df.empty:
        return []
    return [str(item) for item in df.sort_values("rank", ascending=True).head(int(top_k))["nanobody_id"].tolist()]


def _build_rank_delta_table(
    before_rank: pd.DataFrame,
    after_rank: pd.DataFrame,
    *,
    top_k: int,
    validation_labels: pd.DataFrame,
) -> pd.DataFrame:
    before = before_rank.rename(columns={"rank": "before_rank", "score": "before_score"})
    after = after_rank.rename(columns={"rank": "after_rank", "score": "after_score"})
    merged = before.merge(after, on="nanobody_id", how="outer")
    if not validation_labels.empty:
        merged = merged.merge(validation_labels, on="nanobody_id", how="left")

    merged["rank_delta"] = pd.to_numeric(merged["after_rank"], errors="coerce") - pd.to_numeric(
        merged["before_rank"],
        errors="coerce",
    )
    merged["rank_improvement"] = -merged["rank_delta"]
    merged["score_delta"] = pd.to_numeric(merged["after_score"], errors="coerce") - pd.to_numeric(
        merged["before_score"],
        errors="coerce",
    )
    merged["in_before_top_k"] = pd.to_numeric(merged["before_rank"], errors="coerce").le(int(top_k))
    merged["in_after_top_k"] = pd.to_numeric(merged["after_rank"], errors="coerce").le(int(top_k))
    merged["entered_top_k"] = (~merged["in_before_top_k"]) & merged["in_after_top_k"]
    merged["left_top_k"] = merged["in_before_top_k"] & (~merged["in_after_top_k"])

    sort_cols = ["entered_top_k", "rank_improvement", "after_rank", "before_rank"]
    merged = merged.sort_values(sort_cols, ascending=[False, False, True, True], na_position="last", kind="stable")
    return merged.reset_index(drop=True)


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    number = _to_float(value)
    if number is not None:
        if abs(number - round(number)) < 1e-9:
            return str(int(round(number)))
        return f"{number:.4f}"
    return str(value)


def _metric_interpretation(metric: str, delta: float | None) -> str:
    direction = METRIC_DIRECTIONS.get(metric, "context")
    if delta is None:
        return "not comparable"
    if direction == "context":
        return "context metric"
    if abs(delta) < 1e-12:
        return "unchanged"
    improved = delta > 0 if direction == "higher" else delta < 0
    return "improved" if improved else "regressed"


def _build_metric_table(
    before_metrics: dict[str, Any],
    after_metrics: dict[str, Any],
    *,
    top_k_overlap: dict[str, Any],
    validated_top_k_count: int | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_keys = [
        "label_valid_count",
        "label_class_count",
        "baseline_rank_spearman",
        "baseline_rule_auc",
        "calibrated_rank_spearman",
        "calibrated_rule_auc",
        "best_val_loss",
        "failed_rows",
        "warning_rows",
    ]
    for metric in metric_keys:
        before_value = before_metrics.get(metric)
        after_value = after_metrics.get(metric)
        before_num = _to_float(before_value)
        after_num = _to_float(after_value)
        delta = after_num - before_num if before_num is not None and after_num is not None else None
        rows.append(
            {
                "metric": metric,
                "before_value": before_value,
                "after_value": after_value,
                "delta": delta,
                "direction": METRIC_DIRECTIONS.get(metric, "context"),
                "interpretation": _metric_interpretation(metric, delta),
            }
        )

    rows.append(
        {
            "metric": "top_k_overlap_fraction",
            "before_value": None,
            "after_value": top_k_overlap.get("fraction"),
            "delta": None,
            "direction": "context",
            "interpretation": f"{top_k_overlap.get('overlap_count', 0)} shared candidates in top {top_k_overlap.get('k')}",
        }
    )
    if validated_top_k_count is not None:
        rows.append(
            {
                "metric": "validated_top_k_count",
                "before_value": None,
                "after_value": validated_top_k_count,
                "delta": None,
                "direction": "higher",
                "interpretation": "validated candidates in after top-k",
            }
        )
    return pd.DataFrame(rows)


def _markdown_table(df: pd.DataFrame, columns: list[str], *, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    rows = df.loc[:, [column for column in columns if column in df.columns]].head(max_rows).copy()
    if rows.empty:
        return "_No rows._"
    header = "| " + " | ".join(rows.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(rows.columns)) + " |"
    body = []
    for _, row in rows.iterrows():
        body.append("| " + " | ".join(_format_value(row.get(column)) for column in rows.columns) + " |")
    return "\n".join([header, sep] + body)


def _build_report(
    *,
    summary: dict[str, Any],
    metric_df: pd.DataFrame,
    rank_delta_df: pd.DataFrame,
    before_metrics: dict[str, Any],
    after_metrics: dict[str, Any],
    top_k: int,
) -> str:
    lines = [
        "# Validation Retrain Comparison Report",
        "",
        "This report compares a baseline run with a validation-label retrain run.",
        "",
        "## Inputs",
        "",
        f"- Before summary: `{before_metrics.get('summary_path')}`",
        f"- After summary: `{after_metrics.get('summary_path')}`",
        f"- Before label column: `{before_metrics.get('label_col')}`",
        f"- After label column: `{after_metrics.get('label_col')}`",
        f"- Top-k used for overlap: `{top_k}`",
        "",
        "## Key Metrics",
        "",
        _markdown_table(
            metric_df,
            ["metric", "before_value", "after_value", "delta", "direction", "interpretation"],
            max_rows=50,
        ),
        "",
        "## Top Rank Changes",
        "",
        _markdown_table(
            rank_delta_df,
            [
                "nanobody_id",
                "before_rank",
                "after_rank",
                "rank_delta",
                "rank_improvement",
                "before_score",
                "after_score",
                "score_delta",
                "validation_label",
                "entered_top_k",
                "left_top_k",
            ],
            max_rows=30,
        ),
        "",
        "## Interpretation Notes",
        "",
        "- `rank_delta = after_rank - before_rank`; negative values mean the candidate moved up.",
        "- `rank_improvement = -rank_delta`; positive values mean the candidate improved.",
        "- This report is a regression/decision-support check, not proof of biological improvement.",
        "- Prefer judging retrain quality with held-out or newly completed validation data.",
        "",
        "## Summary JSON",
        "",
        "```json",
        json.dumps(summary, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    return "\n".join(lines)


def build_validation_retrain_comparison(
    *,
    before_summary: str | Path,
    after_summary: str | Path,
    out_dir: str | Path,
    validation_labels_csv: str | Path | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    before_summary_path = _resolve_summary_path(before_summary)
    after_summary_path = _resolve_summary_path(after_summary)
    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    before_summary_payload = _read_json(before_summary_path)
    after_summary_payload = _read_json(after_summary_path)
    before_metrics = _extract_run_metrics(before_summary_payload, before_summary_path)
    after_metrics = _extract_run_metrics(after_summary_payload, after_summary_path)

    before_rank = _load_consensus_ranking(before_summary_payload, before_summary_path)
    after_rank = _load_consensus_ranking(after_summary_payload, after_summary_path)
    validation_labels = _load_validation_labels(validation_labels_csv)
    rank_delta_df = _build_rank_delta_table(before_rank, after_rank, top_k=top_k, validation_labels=validation_labels)

    before_top = set(_top_ids(before_rank, top_k))
    after_top = set(_top_ids(after_rank, top_k))
    overlap_count = len(before_top.intersection(after_top))
    union_count = len(before_top.union(after_top))
    top_k_overlap = {
        "k": int(top_k),
        "before_top_k_count": int(len(before_top)),
        "after_top_k_count": int(len(after_top)),
        "overlap_count": int(overlap_count),
        "fraction": float(overlap_count / max(1, min(len(before_top), len(after_top)))) if before_top and after_top else None,
        "jaccard": float(overlap_count / union_count) if union_count else None,
    }

    validated_top_k_count: int | None = None
    if not validation_labels.empty and not rank_delta_df.empty:
        label_ready = rank_delta_df["validation_label"].notna() if "validation_label" in rank_delta_df.columns else pd.Series(False)
        validated_top_k_count = int((label_ready & rank_delta_df["in_after_top_k"].astype(bool)).sum())

    metric_df = _build_metric_table(
        before_metrics,
        after_metrics,
        top_k_overlap=top_k_overlap,
        validated_top_k_count=validated_top_k_count,
    )

    metric_csv = out_path / "validation_retrain_metric_comparison.csv"
    rank_delta_csv = out_path / "validation_retrain_candidate_rank_delta.csv"
    summary_json = out_path / "validation_retrain_comparison_summary.json"
    report_md = out_path / "validation_retrain_comparison_report.md"
    metric_df.to_csv(metric_csv, index=False)
    rank_delta_df.to_csv(rank_delta_csv, index=False)

    summary = {
        "before_summary": str(before_summary_path),
        "after_summary": str(after_summary_path),
        "validation_labels_csv": str(Path(validation_labels_csv).expanduser().resolve())
        if validation_labels_csv is not None and str(validation_labels_csv).strip()
        else None,
        "top_k": int(top_k),
        "top_k_overlap": top_k_overlap,
        "validated_top_k_count": validated_top_k_count,
        "before_metrics": before_metrics,
        "after_metrics": after_metrics,
        "outputs": {
            "metric_comparison_csv": str(metric_csv),
            "candidate_rank_delta_csv": str(rank_delta_csv),
            "summary_json": str(summary_json),
            "report_md": str(report_md),
        },
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    report_md.write_text(
        _build_report(
            summary=summary,
            metric_df=metric_df,
            rank_delta_df=rank_delta_df,
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            top_k=top_k,
        ),
        encoding="utf-8",
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare a baseline run with a validation-label retrain run")
    parser.add_argument("--before_summary", required=True, help="Before-run recommended_pipeline_summary.json or output dir")
    parser.add_argument("--after_summary", required=True, help="After-run recommended_pipeline_summary.json or output dir")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--validation_labels_csv", default=None, help="Optional experiment_validation_labels.csv")
    parser.add_argument("--top_k", type=int, default=10, help="Top-k used for rank overlap analysis")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    summary = build_validation_retrain_comparison(
        before_summary=args.before_summary,
        after_summary=args.after_summary,
        out_dir=args.out_dir,
        validation_labels_csv=args.validation_labels_csv,
        top_k=int(args.top_k),
    )
    outputs = summary.get("outputs") if isinstance(summary.get("outputs"), dict) else {}
    for key in ["metric_comparison_csv", "candidate_rank_delta_csv", "summary_json", "report_md"]:
        if outputs.get(key):
            print(f"Saved: {outputs[key]}")


if __name__ == "__main__":
    main()
