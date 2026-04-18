"""Build PASS/WARN/FAIL quality gate reports for ML pipeline runs.

The quality gate is a post-processing layer. It reads existing feature/QC
artifacts and makes a conservative run-level decision; it does not change model
training, Rule scoring, ML scoring or ranking outputs.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STATUS_ORDER = {"PASS": 0, "WARN": 1, "FAIL": 2}
DEFAULT_WARNING_FRACTION_THRESHOLD = 0.20
DEFAULT_OVERWIDE_FRACTION_THRESHOLD = 0.20
DEFAULT_OVERWIDE_MAX_THRESHOLD = 0.80


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if path is None or not str(path).strip():
        return {}
    json_path = Path(path).expanduser().resolve()
    if not json_path.exists():
        return {}
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: str | Path | None) -> pd.DataFrame | None:
    if path is None or not str(path).strip():
        return None
    csv_path = Path(path).expanduser().resolve()
    if not csv_path.exists():
        return None
    try:
        return pd.read_csv(csv_path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _status_max(statuses: list[str]) -> str:
    if not statuses:
        return "PASS"
    return max((str(status).upper() for status in statuses), key=lambda item: STATUS_ORDER.get(item, 0))


def _row(
    *,
    check_id: str,
    status: str,
    metric: str,
    value: Any,
    threshold: Any,
    message: str,
    recommended_action: str,
) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "status": str(status).upper(),
        "metric": metric,
        "value": value,
        "threshold": threshold,
        "message": message,
        "recommended_action": recommended_action,
    }


def _processing_summary_from_feature_df(feature_df: pd.DataFrame | None) -> dict[str, Any]:
    if feature_df is None:
        return {}
    total = int(len(feature_df))
    status_series = feature_df.get("status", pd.Series([], dtype=object))
    failed_rows = int(status_series.astype(str).str.lower().eq("failed").sum()) if total > 0 else 0
    ok_rows = int(status_series.astype(str).str.lower().eq("ok").sum()) if total > 0 else 0
    warning_rows = 0
    if "warning_message" in feature_df.columns and total > 0:
        warning_rows = int(feature_df["warning_message"].fillna("").astype(str).str.strip().ne("").sum())
    return {
        "total_rows": total,
        "ok_rows": ok_rows,
        "failed_rows": failed_rows,
        "rows_with_warning_message": warning_rows,
    }


def _feature_qc_from_feature_df(feature_df: pd.DataFrame | None, *, pocket_overwide_threshold: float) -> dict[str, Any]:
    if feature_df is None:
        return {}
    missing_ratio = {str(column): float(feature_df[column].isna().mean()) for column in feature_df.columns}
    all_empty = sorted([column for column, ratio in missing_ratio.items() if ratio >= 1.0])
    pocket_shape_qc: dict[str, Any] = {}
    if "pocket_shape_overwide_proxy" in feature_df.columns:
        overwide = pd.to_numeric(feature_df["pocket_shape_overwide_proxy"], errors="coerce")
        finite = overwide[np.isfinite(overwide.to_numpy(dtype=np.float64))]
        if not finite.empty:
            high_mask = overwide >= float(pocket_overwide_threshold)
            pocket_shape_qc = {
                "overwide_threshold": float(pocket_overwide_threshold),
                "finite_row_count": int(len(finite)),
                "high_overwide_row_count": int(high_mask.fillna(False).sum()),
                "high_overwide_row_fraction": float(high_mask.fillna(False).sum() / max(int(len(finite)), 1)),
                "overwide_proxy_p95": float(np.nanpercentile(finite.to_numpy(dtype=np.float64), 95)),
                "overwide_proxy_max": float(np.nanmax(finite.to_numpy(dtype=np.float64))),
            }
    return {
        "row_count": int(len(feature_df)),
        "column_count": int(feature_df.shape[1]),
        "all_empty_columns": all_empty,
        "missing_ratio": missing_ratio,
        "pocket_shape_qc": pocket_shape_qc,
    }


def _label_summary(feature_df: pd.DataFrame | None, label_col: str) -> dict[str, Any]:
    if feature_df is None:
        return {
            "label_status": "unknown",
            "label_valid_count": 0,
            "label_class_count": 0,
            "message": "没有 feature_csv，无法判断 label 覆盖。",
        }
    if label_col not in feature_df.columns:
        return {
            "label_status": "missing_label_column",
            "label_valid_count": 0,
            "label_class_count": 0,
            "message": f"未发现 `{label_col}` 列；当前只能按无真实标签/弱监督结果解释。",
        }
    labels = pd.to_numeric(feature_df[label_col], errors="coerce").dropna()
    valid_count = int(len(labels))
    class_count = int(labels.nunique()) if valid_count else 0
    if valid_count >= 8 and class_count >= 2:
        status = "usable_for_calibration"
        message = f"`{label_col}` 有 {valid_count} 条有效值、{class_count} 个类别，可支持对照和校准。"
    elif valid_count > 0 and class_count >= 2:
        status = "usable_for_compare_only"
        message = f"`{label_col}` 有 {valid_count} 条有效值，但不足 8 条，适合对照，不适合稳定校准。"
    elif valid_count > 0:
        status = "degenerate_label"
        message = f"`{label_col}` 有值但类别不足，不能作为可靠监督信号。"
    else:
        status = "no_valid_label"
        message = f"`{label_col}` 没有有效数值。"
    return {
        "label_status": status,
        "label_valid_count": valid_count,
        "label_class_count": class_count,
        "message": message,
    }


def build_quality_gate(
    *,
    feature_csv: str | Path | None = None,
    feature_qc_json: str | Path | None = None,
    label_col: str = "label",
    warning_fraction_threshold: float = DEFAULT_WARNING_FRACTION_THRESHOLD,
    pocket_overwide_fraction_threshold: float = DEFAULT_OVERWIDE_FRACTION_THRESHOLD,
    pocket_overwide_max_threshold: float = DEFAULT_OVERWIDE_MAX_THRESHOLD,
    pocket_overwide_threshold: float = 0.55,
) -> dict[str, Any]:
    feature_df = _read_csv(feature_csv)
    qc_payload = _read_json(feature_qc_json)
    processing_summary = (
        qc_payload.get("processing_summary")
        if isinstance(qc_payload.get("processing_summary"), dict)
        else _processing_summary_from_feature_df(feature_df)
    )
    feature_qc = (
        qc_payload.get("feature_qc")
        if isinstance(qc_payload.get("feature_qc"), dict)
        else _feature_qc_from_feature_df(feature_df, pocket_overwide_threshold=pocket_overwide_threshold)
    )
    label = _label_summary(feature_df, str(label_col))

    checks: list[dict[str, Any]] = []
    total_rows = _safe_int(processing_summary.get("total_rows"), 0)
    failed_rows = _safe_int(processing_summary.get("failed_rows"), 0)
    warning_rows = _safe_int(processing_summary.get("rows_with_warning_message"), 0)
    warning_fraction = float(warning_rows / max(total_rows, 1))

    checks.append(
        _row(
            check_id="row_count",
            status="FAIL" if total_rows <= 0 else "PASS",
            metric="total_rows",
            value=total_rows,
            threshold="> 0",
            message="没有可用行。" if total_rows <= 0 else "存在可用输入行。",
            recommended_action="检查 input_csv / feature_csv 是否为空。" if total_rows <= 0 else "无需处理。",
        )
    )
    checks.append(
        _row(
            check_id="failed_rows",
            status="FAIL" if failed_rows > 0 else "PASS",
            metric="failed_rows",
            value=failed_rows,
            threshold="== 0",
            message=f"发现 {failed_rows} 行 feature 构建失败。" if failed_rows > 0 else "没有 failed 行。",
            recommended_action="优先打开 failed 行，检查 PDB 路径、链拆分、pocket/catalytic/ligand 文件。" if failed_rows > 0 else "无需处理。",
        )
    )
    warning_status = "WARN" if warning_rows > 0 or warning_fraction >= float(warning_fraction_threshold) else "PASS"
    checks.append(
        _row(
            check_id="warning_rows",
            status=warning_status,
            metric="warning_row_fraction",
            value=round(warning_fraction, 6),
            threshold=f"< {float(warning_fraction_threshold):.2f} and count == 0 preferred",
            message=f"有 {warning_rows} 行包含 warning，占比 {warning_fraction:.1%}。" if warning_rows > 0 else "没有 warning 行。",
            recommended_action="查看 warning_message，重点复核 pocket/catalytic 匹配和可选几何输入。" if warning_rows > 0 else "无需处理。",
        )
    )

    all_empty_columns = feature_qc.get("all_empty_columns") if isinstance(feature_qc.get("all_empty_columns"), list) else []
    non_id_empty = [str(col) for col in all_empty_columns if str(col) not in {"error_message", "warning_message"}]
    checks.append(
        _row(
            check_id="all_empty_columns",
            status="WARN" if non_id_empty else "PASS",
            metric="all_empty_column_count",
            value=len(non_id_empty),
            threshold="== 0 preferred",
            message=("存在全空列: " + ", ".join(non_id_empty[:12])) if non_id_empty else "没有需要关注的全空列。",
            recommended_action="确认这些列是否应由输入文件或可选几何模板生成。" if non_id_empty else "无需处理。",
        )
    )

    pocket_shape_qc = feature_qc.get("pocket_shape_qc") if isinstance(feature_qc.get("pocket_shape_qc"), dict) else {}
    high_overwide_fraction = _safe_float(pocket_shape_qc.get("high_overwide_row_fraction"), 0.0)
    overwide_max = _safe_float(pocket_shape_qc.get("overwide_proxy_max"), 0.0)
    overwide_warn = (
        high_overwide_fraction >= float(pocket_overwide_fraction_threshold)
        or overwide_max >= float(pocket_overwide_max_threshold)
    )
    checks.append(
        _row(
            check_id="pocket_shape_overwide",
            status="WARN" if overwide_warn else "PASS",
            metric="high_overwide_row_fraction / overwide_proxy_max",
            value=f"{high_overwide_fraction:.6f} / {overwide_max:.6f}",
            threshold=f"fraction < {float(pocket_overwide_fraction_threshold):.2f}; max < {float(pocket_overwide_max_threshold):.2f}",
            message=(
                f"pocket overwide 风险偏高：高 overwide 占比 {high_overwide_fraction:.1%}，max={overwide_max:.3f}。"
                if overwide_warn
                else "pocket shape overwide 风险未超过阈值。"
            ),
            recommended_action="复核 pocket residue 边界，必要时比较 manual/P2Rank/fpocket/method consensus。" if overwide_warn else "无需处理。",
        )
    )

    label_status = str(label.get("label_status") or "unknown")
    label_warn = label_status not in {"usable_for_calibration"}
    checks.append(
        _row(
            check_id="label_coverage",
            status="WARN" if label_warn else "PASS",
            metric="label_status",
            value=label_status,
            threshold="usable_for_calibration preferred",
            message=str(label.get("message") or ""),
            recommended_action=(
                "结果只能按无真实标签/弱监督排序解释；如需正式校准，请补真实 validation_label 或 experiment_result。"
                if label_warn
                else "可继续查看 compare/calibration 输出。"
            ),
        )
    )

    status = _status_max([row["status"] for row in checks])
    fail_count = sum(1 for row in checks if row["status"] == "FAIL")
    warn_count = sum(1 for row in checks if row["status"] == "WARN")
    pass_count = sum(1 for row in checks if row["status"] == "PASS")
    if status == "FAIL":
        decision = "当前运行存在阻塞性质量问题，不建议直接解读排名。"
    elif status == "WARN":
        decision = "当前运行可用于初步查看，但需要先复核 warning / label / pocket 风险。"
    else:
        decision = "当前运行通过基础质量门控，可进入排序结果解读。"

    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "overall_status": status,
        "decision": decision,
        "pass_count": pass_count,
        "warn_count": warn_count,
        "fail_count": fail_count,
        "total_rows": total_rows,
        "failed_rows": failed_rows,
        "warning_rows": warning_rows,
        "warning_row_fraction": warning_fraction,
        "label": label,
        "feature_csv": None if feature_csv is None else str(Path(feature_csv).expanduser().resolve()),
        "feature_qc_json": None if feature_qc_json is None else str(Path(feature_qc_json).expanduser().resolve()),
        "thresholds": {
            "warning_fraction_threshold": float(warning_fraction_threshold),
            "pocket_overwide_fraction_threshold": float(pocket_overwide_fraction_threshold),
            "pocket_overwide_max_threshold": float(pocket_overwide_max_threshold),
            "pocket_overwide_threshold": float(pocket_overwide_threshold),
        },
    }
    return {
        "summary": summary,
        "checks": checks,
    }


def _markdown_table(checks: list[dict[str, Any]]) -> str:
    if not checks:
        return "_No checks._"
    columns = ["check_id", "status", "metric", "value", "threshold", "message", "recommended_action"]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in checks:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def _build_report(summary: dict[str, Any], checks: list[dict[str, Any]]) -> str:
    return "\n".join(
        [
            "# Quality Gate Report",
            "",
            f"- Generated at: `{summary.get('generated_at', '')}`",
            f"- Overall status: `{summary.get('overall_status', '')}`",
            f"- Decision: {summary.get('decision', '')}",
            f"- PASS / WARN / FAIL: `{summary.get('pass_count', 0)}` / `{summary.get('warn_count', 0)}` / `{summary.get('fail_count', 0)}`",
            f"- Rows: `{summary.get('total_rows', 0)}`",
            f"- Failed rows: `{summary.get('failed_rows', 0)}`",
            f"- Warning rows: `{summary.get('warning_rows', 0)}`",
            "",
            "## Checks",
            "",
            _markdown_table(checks),
            "",
        ]
    )


def build_quality_gate_outputs(
    *,
    feature_csv: str | Path | None,
    feature_qc_json: str | Path | None = None,
    out_dir: str | Path = "quality_gate",
    label_col: str = "label",
    warning_fraction_threshold: float = DEFAULT_WARNING_FRACTION_THRESHOLD,
    pocket_overwide_fraction_threshold: float = DEFAULT_OVERWIDE_FRACTION_THRESHOLD,
    pocket_overwide_max_threshold: float = DEFAULT_OVERWIDE_MAX_THRESHOLD,
    pocket_overwide_threshold: float = 0.55,
) -> dict[str, Any]:
    result = build_quality_gate(
        feature_csv=feature_csv,
        feature_qc_json=feature_qc_json,
        label_col=label_col,
        warning_fraction_threshold=warning_fraction_threshold,
        pocket_overwide_fraction_threshold=pocket_overwide_fraction_threshold,
        pocket_overwide_max_threshold=pocket_overwide_max_threshold,
        pocket_overwide_threshold=pocket_overwide_threshold,
    )
    output_dir = Path(out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = output_dir / "quality_gate_summary.json"
    checks_csv = output_dir / "quality_gate_checks.csv"
    report_md = output_dir / "quality_gate_report.md"
    checks_df = pd.DataFrame(result["checks"])
    outputs = {
        "quality_gate_summary_json": str(summary_json),
        "quality_gate_checks_csv": str(checks_csv),
        "quality_gate_report_md": str(report_md),
    }
    summary = dict(result["summary"])
    summary["outputs"] = outputs
    summary_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    checks_df.to_csv(checks_csv, index=False)
    report_md.write_text(_build_report(summary, result["checks"]), encoding="utf-8")
    return {
        "summary": summary,
        "checks": result["checks"],
        "outputs": outputs,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build PASS/WARN/FAIL quality gate reports.")
    parser.add_argument("--feature_csv", required=True, help="Path to pose_features.csv")
    parser.add_argument("--feature_qc_json", default=None, help="Optional feature_qc.json")
    parser.add_argument("--out_dir", default="quality_gate")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--warning_fraction_threshold", type=float, default=DEFAULT_WARNING_FRACTION_THRESHOLD)
    parser.add_argument("--pocket_overwide_fraction_threshold", type=float, default=DEFAULT_OVERWIDE_FRACTION_THRESHOLD)
    parser.add_argument("--pocket_overwide_max_threshold", type=float, default=DEFAULT_OVERWIDE_MAX_THRESHOLD)
    parser.add_argument("--pocket_overwide_threshold", type=float, default=0.55)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = build_quality_gate_outputs(
        feature_csv=args.feature_csv,
        feature_qc_json=args.feature_qc_json,
        out_dir=args.out_dir,
        label_col=args.label_col,
        warning_fraction_threshold=float(args.warning_fraction_threshold),
        pocket_overwide_fraction_threshold=float(args.pocket_overwide_fraction_threshold),
        pocket_overwide_max_threshold=float(args.pocket_overwide_max_threshold),
        pocket_overwide_threshold=float(args.pocket_overwide_threshold),
    )
    for key, value in result["outputs"].items():
        print(f"Saved {key}: {value}")
    summary = result["summary"]
    print(
        "Quality gate: "
        f"{summary.get('overall_status')} "
        f"(PASS={summary.get('pass_count')}, WARN={summary.get('warn_count')}, FAIL={summary.get('fail_count')})"
    )


if __name__ == "__main__":
    main()
