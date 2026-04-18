"""Build a cross-run experiment state ledger from local ML app outputs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


APP_RUN_METADATA_NAME = "app_run_metadata.json"
SUMMARY_RELATIVE_PATH = Path("outputs") / "recommended_pipeline_summary.json"
DEFAULT_LEDGER_RELATIVE_PATH = Path("outputs") / "experiment_suggestions" / "experiment_plan_state_ledger.csv"
DEFAULT_EDITED_OVERRIDE_RELATIVE_PATH = Path("outputs") / "experiment_suggestions" / "experiment_plan_override_edited.csv"

OUTPUT_COLUMNS = [
    "nanobody_id",
    "plan_override",
    "experiment_status",
    "experiment_result",
    "validation_label",
    "experiment_owner",
    "experiment_cost",
    "experiment_note",
    "manual_plan_reason",
    "last_plan_decision",
    "last_plan_phase",
    "last_plan_rank",
    "suggestion_tier",
    "diversity_group",
    "experiment_priority_score",
    "diversity_adjusted_priority_score",
    "latest_source_kind",
    "latest_source_run_name",
    "latest_source_path",
    "latest_source_mtime",
    "latest_run_started_at",
    "latest_run_status",
    "source_record_count",
]


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null", "na", "n/a"}:
        return ""
    return text


def _normalize_plan_override(value: Any) -> str:
    text = _clean_text(value).lower().replace("-", "_").replace(" ", "_")
    if text in {"include", "include_now", "force_include", "lock", "locked", "must_include", "current_round"}:
        return "include"
    if text in {"exclude", "force_exclude", "skip", "remove", "reject", "blocked", "do_not_test"}:
        return "exclude"
    if text in {"standby", "backup", "reserve", "waitlist", "wait_list"}:
        return "standby"
    if text in {"defer", "later", "later_round", "postpone", "hold"}:
        return "defer"
    return ""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _timestamp_text(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    except OSError:
        return ""


def _resolve_artifact_path(summary: dict[str, Any], artifact_key: str, fallback: Path) -> Path:
    artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
    value = artifacts.get(artifact_key)
    if value:
        return Path(str(value)).expanduser().resolve()
    return fallback


def _read_source_table(path: Path, *, source_kind: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    if df.empty or "nanobody_id" not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    rename_map = {
        "owner": "experiment_owner",
        "assignee": "experiment_owner",
        "status": "experiment_status",
        "sample_status": "experiment_status",
        "result": "experiment_result",
        "validation_result": "experiment_result",
        "assay_result": "experiment_result",
        "experiment_label": "validation_label",
        "assay_label": "validation_label",
        "cost": "experiment_cost",
        "budget_cost": "experiment_cost",
        "note": "experiment_note",
        "notes": "experiment_note",
        "comment": "experiment_note",
        "comments": "experiment_note",
        "manual_plan_action": "plan_override",
    }
    for old, new in rename_map.items():
        if old in work.columns and new not in work.columns:
            work[new] = work[old]

    for col in OUTPUT_COLUMNS:
        if col not in work.columns:
            work[col] = ""

    work = work[OUTPUT_COLUMNS].copy()
    for col in work.columns:
        work[col] = work[col].map(_clean_text)
    work["plan_override"] = work["plan_override"].map(_normalize_plan_override)
    work["latest_source_kind"] = source_kind
    return work[work["nanobody_id"].astype(str).str.strip().ne("")].reset_index(drop=True)


def _collect_run_source_rows(run_root: Path) -> pd.DataFrame:
    metadata = _read_json(run_root / APP_RUN_METADATA_NAME)
    summary = _read_json(run_root / SUMMARY_RELATIVE_PATH)
    run_name = _clean_text(metadata.get("run_name")) or run_root.name
    run_started_at = _clean_text(metadata.get("started_at") or metadata.get("created_at"))
    run_status = _clean_text(metadata.get("status") or summary.get("status"))

    source_paths = [
        (
            "state_ledger",
            _resolve_artifact_path(
                summary,
                "experiment_plan_state_ledger_csv",
                run_root / DEFAULT_LEDGER_RELATIVE_PATH,
            ),
        ),
        ("edited_override", run_root / DEFAULT_EDITED_OVERRIDE_RELATIVE_PATH),
    ]

    tables: list[pd.DataFrame] = []
    for source_kind, source_path in source_paths:
        source_path = source_path.expanduser().resolve()
        table = _read_source_table(source_path, source_kind=source_kind)
        if table.empty:
            continue
        table["latest_source_run_name"] = run_name
        table["latest_source_path"] = str(source_path)
        table["latest_source_mtime"] = _timestamp_text(source_path)
        table["latest_run_started_at"] = run_started_at
        table["latest_run_status"] = run_status
        tables.append(table)

    if not tables:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return pd.concat(tables, ignore_index=True)


def _discover_run_roots(root: Path) -> list[Path]:
    if not root.exists():
        return []
    run_roots: list[Path] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if child.name.startswith("_") or child.name in {"parameter_templates", "compare_exports"}:
            continue
        if (child / APP_RUN_METADATA_NAME).exists() or (child / SUMMARY_RELATIVE_PATH).exists():
            run_roots.append(child)
    return sorted(run_roots, key=lambda p: p.stat().st_mtime if p.exists() else 0.0)


def build_global_experiment_ledger(local_app_runs: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    root = Path(local_app_runs).expanduser().resolve()
    history_tables = [_collect_run_source_rows(run_root) for run_root in _discover_run_roots(root)]
    history = (
        pd.concat([item for item in history_tables if not item.empty], ignore_index=True)
        if any(not item.empty for item in history_tables)
        else pd.DataFrame(columns=OUTPUT_COLUMNS)
    )
    if history.empty:
        summary = {
            "local_app_runs": str(root),
            "history_row_count": 0,
            "latest_row_count": 0,
            "unique_nanobody_count": 0,
            "status_counts": {},
            "source_kind_counts": {},
            "built_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        return history.copy(), history.copy(), summary

    history["_run_sort"] = pd.to_datetime(history["latest_run_started_at"], errors="coerce")
    history["_run_sort"] = history["_run_sort"].fillna(pd.Timestamp.min)
    history["_source_sort"] = pd.to_datetime(history["latest_source_mtime"], errors="coerce")
    history["_source_sort"] = history["_source_sort"].fillna(pd.Timestamp.min)
    history["_source_priority"] = history["latest_source_kind"].map({"state_ledger": 0, "edited_override": 1}).fillna(0)
    history = history.sort_values(
        ["nanobody_id", "_run_sort", "_source_sort", "_source_priority"],
        ascending=True,
    ).reset_index(drop=True)
    counts = history.groupby("nanobody_id", dropna=False).size().rename("source_record_count").reset_index()
    latest = history.groupby("nanobody_id", dropna=False).tail(1).copy()
    latest = latest.drop(columns=["source_record_count"], errors="ignore").merge(counts, on="nanobody_id", how="left")
    latest = latest.drop(columns=["_run_sort", "_source_sort", "_source_priority"], errors="ignore")
    history = history.drop(columns=["_run_sort", "_source_sort", "_source_priority"], errors="ignore")

    latest = latest[OUTPUT_COLUMNS].sort_values(["experiment_status", "nanobody_id"], ascending=True).reset_index(drop=True)
    history = history[OUTPUT_COLUMNS].reset_index(drop=True)
    summary = {
        "local_app_runs": str(root),
        "history_row_count": int(len(history)),
        "latest_row_count": int(len(latest)),
        "unique_nanobody_count": int(latest["nanobody_id"].nunique(dropna=True)),
        "status_counts": {str(k): int(v) for k, v in latest["experiment_status"].value_counts(dropna=False).to_dict().items()},
        "source_kind_counts": {str(k): int(v) for k, v in history["latest_source_kind"].value_counts(dropna=False).to_dict().items()},
        "built_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    return latest, history, summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a global experiment state ledger from local_app_runs")
    parser.add_argument("--local_app_runs", default="local_app_runs", help="Root directory containing local app run folders")
    parser.add_argument("--out_dir", default=None, help="Output directory; default writes into local_app_runs")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    latest, history, summary = build_global_experiment_ledger(args.local_app_runs)
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else Path(args.local_app_runs).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    latest_csv = out_dir / "experiment_state_ledger_global.csv"
    history_csv = out_dir / "experiment_state_ledger_global_history.csv"
    summary_json = out_dir / "experiment_state_ledger_global_summary.json"
    latest.to_csv(latest_csv, index=False)
    history.to_csv(history_csv, index=False)
    summary_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"Saved: {latest_csv}")
    print(f"Saved: {history_csv}")
    print(f"Saved: {summary_json}")


if __name__ == "__main__":
    main()
