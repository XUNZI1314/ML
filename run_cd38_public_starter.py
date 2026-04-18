"""Run a lightweight public CD38 benchmark starter workflow.

This script intentionally does not run external pocket finders. It refreshes
the public CD38 benchmark artifacts that can be generated locally from the
existing repository inputs, then writes a compact handoff report.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local public CD38 starter reports.")
    parser.add_argument("--out_dir", default="benchmarks/cd38/public_starter", help="Starter summary output directory.")
    parser.add_argument("--python_executable", default=None, help="Python executable for child commands.")
    parser.add_argument("--continue_on_error", action="store_true", help="Continue remaining steps after a command fails.")
    parser.add_argument(
        "--skip_external_package_refresh",
        action="store_true",
        help="Skip prepare_cd38_external_tool_inputs.py and preflight refresh.",
    )
    parser.add_argument(
        "--p2rank_root",
        default="benchmarks/cd38/results",
        help="Local P2Rank result root scanned for existing *_predictions.csv files.",
    )
    parser.add_argument(
        "--rank_by_pdb",
        default="3ROP=2,4OGW=1",
        help="P2Rank active-site rank overrides used during readiness refresh.",
    )
    return parser


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _run_command(command: list[str], *, cwd: Path) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=str(cwd), check=False)
    return {
        "name": Path(command[1]).stem if len(command) > 1 else Path(command[0]).stem,
        "command": command,
        "returncode": int(completed.returncode),
        "status": "success" if int(completed.returncode) == 0 else "failed",
    }


def _load_json(path: str | Path) -> dict[str, Any]:
    target = Path(path).expanduser()
    if not target.exists():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: str | Path) -> pd.DataFrame:
    target = Path(path).expanduser()
    if not target.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(target, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(target, encoding="utf-8-sig", low_memory=False)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _artifact_exists(path: str | Path) -> bool:
    return Path(path).expanduser().exists()


def _summarize_panel(panel_csv: Path) -> dict[str, Any]:
    df = _read_csv(panel_csv)
    if df.empty:
        return {"row_count": 0, "method_counts": {}, "pdb_counts": {}}
    method_counts = df["method"].fillna("unknown").astype(str).value_counts().to_dict() if "method" in df.columns else {}
    pdb_counts = df["pdb_id"].fillna("unknown").astype(str).value_counts().to_dict() if "pdb_id" in df.columns else {}
    return {
        "row_count": int(len(df)),
        "method_counts": {str(k): int(v) for k, v in method_counts.items()},
        "pdb_counts": {str(k): int(v) for k, v in pdb_counts.items()},
    }


def _summarize_proxy(summary_json: Path) -> dict[str, Any]:
    payload = _load_json(summary_json)
    if not payload:
        return {}
    return {
        "recommendation": payload.get("recommendation"),
        "benchmark_row_count": payload.get("benchmark_row_count"),
        "structure_count": payload.get("structure_count"),
        "method_count": payload.get("method_count"),
        "default_penalty_recommended": payload.get("default_penalty_recommended"),
    }


def _summarize_readiness(summary_json: Path) -> dict[str, Any]:
    payload = _load_json(summary_json)
    if not payload:
        return {}
    expansion = payload.get("expansion_plan") if isinstance(payload.get("expansion_plan"), dict) else {}
    p2rank = payload.get("p2rank") if isinstance(payload.get("p2rank"), dict) else {}
    fpocket = payload.get("fpocket") if isinstance(payload.get("fpocket"), dict) else {}
    return {
        "missing_or_pending_count": expansion.get("missing_or_pending_count"),
        "status_counts": expansion.get("status_counts", {}),
        "p2rank_status": p2rank.get("status"),
        "p2rank_manifest_row_count": p2rank.get("manifest_row_count"),
        "fpocket_status": fpocket.get("status"),
        "fpocket_manifest_row_count": fpocket.get("manifest_row_count"),
        "next_actions": payload.get("next_actions", []),
    }


def _summarize_action_plan(action_json: Path) -> dict[str, Any]:
    payload = _load_json(action_json)
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    return {
        "overall_status": summary.get("overall_status"),
        "action_count": summary.get("action_count"),
        "benchmark_gap_action_count": summary.get("benchmark_gap_action_count"),
        "missing_benchmark_gap_action_count": summary.get("missing_benchmark_gap_action_count"),
        "p2rank_tool_available": summary.get("p2rank_tool_available"),
        "fpocket_tool_available": summary.get("fpocket_tool_available"),
    }


def _build_report(summary: dict[str, Any]) -> str:
    artifacts = summary["artifacts"]
    command_rows = summary["commands"]
    readiness = summary.get("readiness", {})
    action_plan = summary.get("action_plan", {})
    next_actions = readiness.get("next_actions") or [
        "Review the CD38 external benchmark action plan and fill missing fpocket/P2Rank outputs when available."
    ]

    lines = [
        "# CD38 Public Structure Starter",
        "",
        "This report refreshes the locally runnable CD38 benchmark starter. It uses public PDB structures already available in the repository and does not run external pocket finders.",
        "",
        "## What This Proves",
        "",
        "- Public CD38 PDB inputs are present and reusable for benchmark checks.",
        "- Existing ligand-contact and P2Rank baseline results can be summarized into one panel.",
        "- The current pocket-shape proxy calibration can be refreshed without changing model scores.",
        "- Missing external fpocket/P2Rank outputs are converted into an explicit action plan and next-run script.",
        "",
        "## Current Status",
        "",
        f"- Panel rows: `{summary['panel'].get('row_count', 0)}`",
        f"- Panel methods: `{summary['panel'].get('method_counts', {})}`",
        f"- Readiness missing/pending pairs: `{readiness.get('missing_or_pending_count', '')}`",
        f"- Action plan status: `{action_plan.get('overall_status', '')}`",
        f"- Missing benchmark gap actions: `{action_plan.get('missing_benchmark_gap_action_count', '')}`",
        "",
        "## Next Actions",
        "",
    ]
    for action in next_actions:
        lines.append(f"- {action}")

    lines.extend(
        [
            "",
            "## Key Outputs",
            "",
        ]
    )
    for label, path in artifacts.items():
        lines.append(f"- {label}: `{path}`")

    lines.extend(
        [
            "",
            "## Commands Run",
            "",
            "| Step | Status | Return Code |",
            "| --- | --- | ---: |",
        ]
    )
    for row in command_rows:
        lines.append(f"| {row.get('name')} | {row.get('status')} | {row.get('returncode')} |")

    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This starter is a public-structure benchmark helper. It is not a nanobody screening run and does not replace real experimental validation.",
            "",
        ]
    )
    return "\n".join(lines)


def run_cd38_public_starter(
    *,
    out_dir: str | Path = "benchmarks/cd38/public_starter",
    python_executable: str | None = None,
    continue_on_error: bool = False,
    skip_external_package_refresh: bool = False,
    p2rank_root: str | Path = "benchmarks/cd38/results",
    rank_by_pdb: str = "3ROP=2,4OGW=1",
) -> dict[str, Any]:
    repo_root = _repo_root()
    py = str(python_executable or sys.executable)
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    commands: list[list[str]] = [
        [py, "summarize_cd38_benchmarks.py"],
        [py, "inspect_cd38_ligand_candidates.py"],
        [py, "analyze_cd38_pocket_parameter_sensitivity.py"],
        [py, "build_cd38_proxy_calibration_report.py"],
    ]
    if not skip_external_package_refresh:
        commands.extend(
            [
                [py, "prepare_cd38_external_tool_inputs.py"],
                [py, "check_cd38_external_tool_environment.py"],
            ]
        )
    commands.extend(
        [
            [
                py,
                "refresh_cd38_benchmark_readiness.py",
                "--p2rank_root",
                str(p2rank_root),
                "--rank_by_pdb",
                str(rank_by_pdb),
                "--continue_on_error",
            ],
            [py, "build_cd38_external_benchmark_action_plan.py"],
            [py, "build_cd38_external_tool_runbook.py"],
        ]
    )

    command_records: list[dict[str, Any]] = []
    for command in commands:
        record = _run_command(command, cwd=repo_root)
        command_records.append(record)
        if record["returncode"] != 0 and not continue_on_error:
            break

    panel_csv = repo_root / "benchmarks" / "cd38" / "results" / "cd38_benchmark_panel.csv"
    panel_md = repo_root / "benchmarks" / "cd38" / "results" / "cd38_benchmark_panel.md"
    ligand_report = repo_root / "benchmarks" / "cd38" / "ligand_candidates" / "cd38_ligand_candidate_report.md"
    sensitivity_report = repo_root / "benchmarks" / "cd38" / "parameter_sensitivity" / "cd38_pocket_parameter_sensitivity_report.md"
    proxy_summary = repo_root / "benchmarks" / "cd38" / "proxy_calibration" / "cd38_proxy_calibration_summary.json"
    proxy_report = repo_root / "benchmarks" / "cd38" / "proxy_calibration" / "cd38_proxy_calibration_report.md"
    readiness_summary = repo_root / "benchmarks" / "cd38" / "readiness" / "cd38_benchmark_readiness_summary.json"
    readiness_report = repo_root / "benchmarks" / "cd38" / "readiness" / "cd38_benchmark_readiness_report.md"
    action_plan_json = repo_root / "benchmarks" / "cd38" / "action_plan" / "cd38_external_benchmark_action_plan.json"
    action_plan_md = repo_root / "benchmarks" / "cd38" / "action_plan" / "cd38_external_benchmark_action_plan.md"
    next_run_md = repo_root / "benchmarks" / "cd38" / "external_tool_inputs" / "cd38_external_tool_next_run.md"
    next_run_ps1 = repo_root / "benchmarks" / "cd38" / "external_tool_inputs" / "run_cd38_external_next_benchmark.ps1"
    preflight_report = repo_root / "benchmarks" / "cd38" / "external_tool_inputs" / "preflight" / "cd38_external_tool_preflight_report.md"

    artifacts = {
        "panel_csv": str(panel_csv),
        "panel_md": str(panel_md),
        "ligand_candidate_report_md": str(ligand_report),
        "parameter_sensitivity_report_md": str(sensitivity_report),
        "proxy_calibration_report_md": str(proxy_report),
        "readiness_report_md": str(readiness_report),
        "external_tool_preflight_report_md": str(preflight_report),
        "action_plan_md": str(action_plan_md),
        "external_tool_next_run_md": str(next_run_md),
        "external_tool_next_run_ps1": str(next_run_ps1),
    }
    summary: dict[str, Any] = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "repo_root": str(repo_root),
        "out_dir": str(out),
        "commands": command_records,
        "artifacts": artifacts,
        "artifact_presence": {key: _artifact_exists(path) for key, path in artifacts.items()},
        "panel": _summarize_panel(panel_csv),
        "proxy_calibration": _summarize_proxy(proxy_summary),
        "readiness": _summarize_readiness(readiness_summary),
        "action_plan": _summarize_action_plan(action_plan_json),
    }
    summary["failed_command_count"] = sum(1 for row in command_records if row.get("status") != "success")
    summary["completed_command_count"] = sum(1 for row in command_records if row.get("status") == "success")

    summary_json = out / "cd38_public_starter_summary.json"
    report_md = out / "cd38_public_starter_report.md"
    summary["outputs"] = {
        "summary_json": str(summary_json),
        "report_md": str(report_md),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    report_md.write_text(_build_report(summary), encoding="utf-8")

    print(f"CD38 public starter summary: {summary_json}")
    print(f"CD38 public starter report: {report_md}")
    print(f"Commands successful: {summary['completed_command_count']}/{len(command_records)}")
    print(f"Panel rows: {_safe_int(summary['panel'].get('row_count'))}")
    return summary


def main() -> None:
    args = _build_parser().parse_args()
    run_cd38_public_starter(
        out_dir=args.out_dir,
        python_executable=args.python_executable,
        continue_on_error=bool(args.continue_on_error),
        skip_external_package_refresh=bool(args.skip_external_package_refresh),
        p2rank_root=args.p2rank_root,
        rank_by_pdb=str(args.rank_by_pdb),
    )


if __name__ == "__main__":
    main()
