"""Shared helpers for pipeline orchestration scripts."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class CommandResult:
    """Compact execution record for one subprocess command."""

    name: str
    command: list[str]
    returncode: int
    elapsed_sec: float
    stdout_tail: str
    stderr_tail: str


def run_command(name: str, command: list[str], cwd: Path) -> CommandResult:
    """Run one command and capture a compact summary."""
    start = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed = time.perf_counter() - start

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    out_tail = "\n".join(stdout.splitlines()[-12:])
    err_tail = "\n".join(stderr.splitlines()[-12:])

    result = CommandResult(
        name=name,
        command=command,
        returncode=int(proc.returncode),
        elapsed_sec=float(elapsed),
        stdout_tail=out_tail,
        stderr_tail=err_tail,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed [{name}] rc={proc.returncode}\n"
            f"CMD: {' '.join(command)}\n"
            f"STDOUT:\n{out_tail}\n"
            f"STDERR:\n{err_tail}"
        )
    return result


def ensure_files_exist(paths: list[Path]) -> None:
    """Ensure all expected output files exist."""
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Expected output files not found: {missing}")


def serialize_command_results(commands: list[CommandResult]) -> list[dict[str, Any]]:
    """Convert command results to JSON-serializable dictionaries."""
    return [
        {
            "name": c.name,
            "command": c.command,
            "returncode": c.returncode,
            "elapsed_sec": c.elapsed_sec,
            "stdout_tail": c.stdout_tail,
            "stderr_tail": c.stderr_tail,
        }
        for c in commands
    ]


def safe_float(value: Any) -> float | None:
    """Parse finite float values; invalid inputs return None."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def load_strategy_seed(strategy_json: Path) -> dict[str, Any]:
    """Load strategy seed fields from recommended_strategy.json."""
    payload = json.loads(strategy_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("strategy json must be an object")

    rec = payload.get("recommended")
    rec_obj = rec if isinstance(rec, dict) else {}
    constraints_obj = payload.get("constraints") if isinstance(payload.get("constraints"), dict) else {}

    allowed_selection = {"objective", "nanobody_auc", "rank_consistency"}
    selection_metric = str(rec_obj.get("selection_metric", "")).strip().lower()
    if selection_metric not in allowed_selection:
        selection_metric = None

    return {
        "rank_consistency_weight": safe_float(rec_obj.get("rank_consistency_weight")),
        "selection_metric": selection_metric,
        "min_nanobody_auc": safe_float(constraints_obj.get("min_nanobody_auc")),
        "min_rank_consistency": safe_float(constraints_obj.get("min_rank_consistency")),
    }


def resolve_calibration_strategy(
    *,
    default_rank_consistency_weight: float,
    default_selection_metric: str,
    strategy_seed_payload: dict[str, Any] | None,
    strategy_seed_path: Path | None,
    baseline_summary: dict[str, Any] | None,
    enable_baseline_guard: bool,
    rank_guard_tolerance: float,
    auc_guard_tolerance: float,
) -> dict[str, Any]:
    """Merge CLI defaults, optional seed strategy, and baseline guards."""
    rank_weight = float(default_rank_consistency_weight)
    selection_metric = str(default_selection_metric)
    min_rank_consistency = (
        None if strategy_seed_payload is None else safe_float(strategy_seed_payload.get("min_rank_consistency"))
    )
    min_nanobody_auc = (
        None if strategy_seed_payload is None else safe_float(strategy_seed_payload.get("min_nanobody_auc"))
    )

    if strategy_seed_payload is not None:
        seeded_rank_weight = safe_float(strategy_seed_payload.get("rank_consistency_weight"))
        if seeded_rank_weight is not None:
            rank_weight = float(seeded_rank_weight)
        seeded_selection_metric = str(strategy_seed_payload.get("selection_metric", "")).strip()
        if seeded_selection_metric:
            selection_metric = seeded_selection_metric

    if enable_baseline_guard and baseline_summary is not None:
        baseline_rank = safe_float(baseline_summary.get("score_spearman"))
        baseline_rule_auc = safe_float(baseline_summary.get("rule_auc"))
        rank_tol = max(0.0, float(rank_guard_tolerance))
        auc_tol = max(0.0, float(auc_guard_tolerance))

        if baseline_rank is not None:
            derived_min_rank = max(-1.0, float(baseline_rank) - rank_tol - 1e-12)
            if min_rank_consistency is None:
                min_rank_consistency = float(derived_min_rank)
            else:
                min_rank_consistency = max(float(min_rank_consistency), float(derived_min_rank))
        if baseline_rule_auc is not None:
            derived_min_nb_auc = max(0.0, float(baseline_rule_auc) - auc_tol - 1e-12)
            if min_nanobody_auc is None:
                min_nanobody_auc = float(derived_min_nb_auc)
            else:
                min_nanobody_auc = max(float(min_nanobody_auc), float(derived_min_nb_auc))

    summary = {
        "enabled": bool(strategy_seed_payload is not None),
        "source_json": str(strategy_seed_path) if strategy_seed_path is not None else None,
        "applied_rank_consistency_weight": float(rank_weight),
        "applied_selection_metric": str(selection_metric),
        "applied_min_rank_consistency": min_rank_consistency,
        "applied_min_nanobody_auc": min_nanobody_auc,
    }
    return {
        "rank_consistency_weight": float(rank_weight),
        "selection_metric": str(selection_metric),
        "min_rank_consistency": min_rank_consistency,
        "min_nanobody_auc": min_nanobody_auc,
        "summary": summary,
    }


def resolve_strategy_seed_path(
    repo_root: Path,
    strategy_seed_json: str | None,
    auto_seed_path: Path,
    disable_auto_seed: bool,
) -> Path | None:
    """Resolve explicit or auto-discovered recommended strategy seed path."""
    if strategy_seed_json:
        path = Path(str(strategy_seed_json)).expanduser()
        if not path.is_absolute():
            path = (repo_root / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"strategy_seed_json not found: {path}")
        return path

    if disable_auto_seed:
        return None
    return auto_seed_path if auto_seed_path.exists() else None


def _format_metric(value: Any, digits: int = 4) -> str:
    numeric = safe_float(value)
    if numeric is None:
        return "N/A"
    return f"{float(numeric):.{digits}f}"


def _extract_metric_delta(improvement_summary: dict[str, Any] | None, metric_key: str) -> float | None:
    if not isinstance(improvement_summary, dict):
        return None
    metrics = improvement_summary.get("metrics")
    if not isinstance(metrics, list):
        return None
    for item in metrics:
        if not isinstance(item, dict):
            continue
        if str(item.get("metric_key")) != str(metric_key):
            continue
        return safe_float(item.get("delta"))
    return None


def build_execution_report(
    *,
    title: str,
    start_mode: str | None = None,
    input_csv: str | None = None,
    feature_csv: str | None = None,
    label_summary: dict[str, Any] | None = None,
    synthetic_data: dict[str, Any] | None = None,
    strategy_seed_summary: dict[str, Any] | None = None,
    baseline_summary: dict[str, Any] | None = None,
    calibrated_summary: dict[str, Any] | None = None,
    improvement_summary: dict[str, Any] | None = None,
    strategy_summary: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
    commands: list[CommandResult] | None = None,
    notes: list[str] | None = None,
) -> str:
    """Build a compact Markdown report for orchestration outputs."""
    lines = [f"# {title}", ""]

    overview_lines: list[str] = []
    if start_mode:
        overview_lines.append(f"- Start mode: `{start_mode}`")
    if input_csv:
        overview_lines.append(f"- Input CSV: `{input_csv}`")
    if feature_csv:
        overview_lines.append(f"- Feature CSV: `{feature_csv}`")
    if synthetic_data:
        rows = synthetic_data.get("rows")
        nanobodies = synthetic_data.get("nanobodies")
        conformers = synthetic_data.get("conformers")
        positive_rate = synthetic_data.get("positive_rate")
        overview_lines.append(
            "- Synthetic data: "
            f"rows={rows}, nanobodies={nanobodies}, conformers={conformers}, positive_rate={_format_metric(positive_rate)}"
        )
    if label_summary:
        overview_lines.append(
            "- Labels: "
            f"valid={int(label_summary.get('label_valid_count', 0))}, "
            f"classes={int(label_summary.get('label_class_count', 0))}, "
            f"compare={bool(label_summary.get('label_compare_possible', False))}, "
            f"calibration={bool(label_summary.get('calibration_possible', False))}"
        )
    if overview_lines:
        lines.extend(overview_lines)
        lines.append("")

    if notes:
        lines.extend(["## Notes", ""])
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")

    metric_lines: list[str] = []
    if baseline_summary is not None:
        metric_lines.extend(
            [
                "- Baseline score Spearman: "
                f"{_format_metric(baseline_summary.get('score_spearman'))}",
                "- Baseline rank Spearman: "
                f"{_format_metric(baseline_summary.get('rank_spearman'))}",
                f"- Baseline rule AUC: {_format_metric(baseline_summary.get('rule_auc'))}",
                f"- Baseline ML AUC: {_format_metric(baseline_summary.get('ml_auc'))}",
            ]
        )
    if calibrated_summary is not None:
        metric_lines.extend(
            [
                "- Calibrated score Spearman: "
                f"{_format_metric(calibrated_summary.get('score_spearman'))}",
                "- Calibrated rank Spearman: "
                f"{_format_metric(calibrated_summary.get('rank_spearman'))}",
                f"- Calibrated rule AUC: {_format_metric(calibrated_summary.get('rule_auc'))}",
                f"- Calibrated ML AUC: {_format_metric(calibrated_summary.get('ml_auc'))}",
            ]
        )
    rule_auc_delta = _extract_metric_delta(improvement_summary, "rule_auc")
    if rule_auc_delta is not None:
        metric_lines.append(f"- Calibrated minus baseline rule AUC: {float(rule_auc_delta):+.4f}")
    if metric_lines:
        lines.extend(["## Key Metrics", ""])
        lines.extend(metric_lines)
        lines.append("")

    strategy_lines: list[str] = []
    if strategy_seed_summary:
        strategy_lines.append(
            "- Seed strategy: "
            f"enabled={bool(strategy_seed_summary.get('enabled', False))}, "
            f"selection_metric={strategy_seed_summary.get('applied_selection_metric')}, "
            f"rank_weight={_format_metric(strategy_seed_summary.get('applied_rank_consistency_weight'))}"
        )
        if strategy_seed_summary.get("source_json"):
            strategy_lines.append(f"- Seed source: `{strategy_seed_summary['source_json']}`")
        if strategy_seed_summary.get("applied_min_rank_consistency") is not None:
            strategy_lines.append(
                "- Seed min rank consistency: "
                f"{_format_metric(strategy_seed_summary.get('applied_min_rank_consistency'))}"
            )
        if strategy_seed_summary.get("applied_min_nanobody_auc") is not None:
            strategy_lines.append(
                "- Seed min nanobody AUC: "
                f"{_format_metric(strategy_seed_summary.get('applied_min_nanobody_auc'))}"
            )
    if isinstance(strategy_summary, dict):
        recommended = strategy_summary.get("recommended") if isinstance(strategy_summary.get("recommended"), dict) else {}
        constraints = strategy_summary.get("constraints") if isinstance(strategy_summary.get("constraints"), dict) else {}
        if recommended:
            strategy_lines.append(
                "- Recommended strategy: "
                f"selection_metric={recommended.get('selection_metric')}, "
                f"rank_weight={_format_metric(recommended.get('rank_consistency_weight'))}, "
                f"nanobody_auc={_format_metric(recommended.get('nanobody_auc'))}, "
                f"rank_consistency={_format_metric(recommended.get('rank_consistency'))}, "
                f"feasible={recommended.get('feasible')}"
            )
        if constraints:
            strategy_lines.append(
                "- Strategy constraints: "
                f"min_nanobody_auc={_format_metric(constraints.get('min_nanobody_auc'))}, "
                f"min_rank_consistency={_format_metric(constraints.get('min_rank_consistency'))}"
            )
    if strategy_lines:
        lines.extend(["## Strategy", ""])
        lines.extend(strategy_lines)
        lines.append("")

    if artifacts:
        lines.extend(["## Artifacts", ""])
        for key, value in artifacts.items():
            if value is None:
                continue
            lines.append(f"- {key}: `{value}`")
        lines.append("")

    if commands:
        lines.extend(["## Commands", "", "| Step | RC | Sec |", "|---|---:|---:|"])
        for item in commands:
            lines.append(
                f"| {item.name} | {int(item.returncode)} | {float(item.elapsed_sec):.2f} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


__all__ = [
    "CommandResult",
    "build_execution_report",
    "run_command",
    "ensure_files_exist",
    "serialize_command_results",
    "safe_float",
    "load_strategy_seed",
    "resolve_calibration_strategy",
    "resolve_strategy_seed_path",
]
