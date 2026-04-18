from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refresh CD38 benchmark readiness reports across ligand scan, expansion plan, P2Rank, and fpocket."
    )
    parser.add_argument(
        "--structure_targets_csv",
        default="benchmarks/cd38/cd38_structure_targets.csv",
        help="CD38 target structures CSV.",
    )
    parser.add_argument(
        "--manifest_csv",
        default="benchmarks/cd38/cd38_benchmark_manifest.csv",
        help="Current CD38 benchmark manifest.",
    )
    parser.add_argument("--results_root", default="benchmarks/cd38/results", help="CD38 benchmark results root.")
    parser.add_argument(
        "--truth_file",
        default="benchmarks/cd38/cd38_active_site_truth.txt",
        help="Ground-truth residue definition file.",
    )
    parser.add_argument("--out_dir", default="benchmarks/cd38/readiness", help="Consolidated readiness output directory.")
    parser.add_argument(
        "--ligand_candidates_out_dir",
        default="benchmarks/cd38/ligand_candidates",
        help="Output directory for inspect_cd38_ligand_candidates.py.",
    )
    parser.add_argument(
        "--expansion_plan_out_dir",
        default="benchmarks/cd38/expansion_plan",
        help="Output directory for build_cd38_benchmark_expansion_plan.py.",
    )
    parser.add_argument("--p2rank_root", nargs="*", default=None, help="Optional P2Rank output roots to scan.")
    parser.add_argument("--fpocket_root", nargs="*", default=None, help="Optional fpocket output roots to scan.")
    parser.add_argument("--chain_filter", default="A", help="Chain filter for external pocket-tool import.")
    parser.add_argument("--near_threshold", type=float, default=4.5, help="Near-hit threshold in Angstrom.")
    parser.add_argument(
        "--rank_by_pdb",
        default=None,
        help="Optional P2Rank rank overrides passed to prepare_cd38_p2rank_panel.py, e.g. 3ROP=2,4OGW=1.",
    )
    parser.add_argument(
        "--run_discovered",
        action="store_true",
        help="Pass --run to P2Rank/fpocket discovery scripts after manifest generation.",
    )
    parser.add_argument("--skip_ligand_scan", action="store_true", help="Skip inspect_cd38_ligand_candidates.py.")
    parser.add_argument("--skip_expansion_plan", action="store_true", help="Skip build_cd38_benchmark_expansion_plan.py.")
    parser.add_argument("--python_executable", default=None, help="Python executable used for child commands.")
    parser.add_argument("--continue_on_error", action="store_true", help="Continue refreshing remaining reports after a failure.")
    return parser


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _run_command(command: list[str], cwd: Path) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=str(cwd), check=False)
    return {
        "command": command,
        "returncode": int(completed.returncode),
        "status": "success" if int(completed.returncode) == 0 else "failed",
    }


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"load_error": str(exc), "path": str(path)}


def _as_roots(values: list[str] | None) -> list[str]:
    return [str(value).strip() for value in (values or []) if str(value).strip()]


def _make_ligand_command(args: argparse.Namespace, py: str, out_dir: Path) -> tuple[list[str], Path]:
    ligand_out_dir = Path(args.ligand_candidates_out_dir).expanduser()
    summary_json = ligand_out_dir / "cd38_ligand_candidate_summary.json"
    command = [
        py,
        "inspect_cd38_ligand_candidates.py",
        "--structure_targets_csv",
        str(args.structure_targets_csv),
        "--truth_file",
        str(args.truth_file),
        "--out_dir",
        str(ligand_out_dir),
        "--distance_threshold",
        str(float(args.near_threshold)),
    ]
    return command, summary_json


def _make_expansion_command(args: argparse.Namespace, py: str) -> tuple[list[str], Path]:
    expansion_out_dir = Path(args.expansion_plan_out_dir).expanduser()
    summary_json = expansion_out_dir / "cd38_benchmark_expansion_summary.json"
    command = [
        py,
        "build_cd38_benchmark_expansion_plan.py",
        "--structure_targets_csv",
        str(args.structure_targets_csv),
        "--manifest_csv",
        str(args.manifest_csv),
        "--results_root",
        str(args.results_root),
        "--out_dir",
        str(expansion_out_dir),
        "--near_threshold",
        str(float(args.near_threshold)),
    ]
    return command, summary_json


def _make_p2rank_command(args: argparse.Namespace, py: str, out_dir: Path, roots: list[str]) -> tuple[list[str], Path]:
    manifest_out = out_dir / "p2rank_discovered_manifest.csv"
    summary_json = out_dir / "p2rank_discovered_manifest.csv.summary.json"
    report_md = out_dir / "p2rank_discovered_manifest.csv.report.md"
    command = [
        py,
        "prepare_cd38_p2rank_panel.py",
        "--p2rank_root",
        *roots,
        "--manifest_out",
        str(manifest_out),
        "--summary_json",
        str(summary_json),
        "--report_md",
        str(report_md),
        "--results_root",
        str(args.results_root),
        "--truth_file",
        str(args.truth_file),
        "--chain_filter",
        str(args.chain_filter),
        "--near_threshold",
        str(float(args.near_threshold)),
    ]
    if args.rank_by_pdb:
        command.extend(["--rank_by_pdb", str(args.rank_by_pdb)])
    if bool(args.run_discovered):
        command.append("--run")
    return command, summary_json


def _make_fpocket_command(args: argparse.Namespace, py: str, out_dir: Path, roots: list[str]) -> tuple[list[str], Path]:
    manifest_out = out_dir / "fpocket_discovered_manifest.csv"
    summary_json = out_dir / "fpocket_discovered_manifest.csv.summary.json"
    report_md = out_dir / "fpocket_discovered_manifest.csv.report.md"
    command = [
        py,
        "prepare_cd38_fpocket_panel.py",
        "--fpocket_root",
        *roots,
        "--manifest_out",
        str(manifest_out),
        "--summary_json",
        str(summary_json),
        "--report_md",
        str(report_md),
        "--results_root",
        str(args.results_root),
        "--truth_file",
        str(args.truth_file),
        "--chain_filter",
        str(args.chain_filter),
        "--near_threshold",
        str(float(args.near_threshold)),
    ]
    if bool(args.run_discovered):
        command.append("--run")
    return command, summary_json


def _short_tool_summary(tool: str, payload: dict[str, Any] | None, roots: list[str] | None = None) -> dict[str, Any]:
    if payload is None:
        return {
            "tool": tool,
            "status": "not_run",
            "roots": roots or [],
        }
    if tool == "ligand_candidates":
        return {
            "tool": tool,
            "status": "ready",
            "structure_count": payload.get("structure_count"),
            "candidate_count": payload.get("candidate_count"),
            "recommended_count": payload.get("recommended_count"),
            "structures_without_recommended_ligand": payload.get("structures_without_recommended_ligand", []),
        }
    if tool == "expansion_plan":
        return {
            "tool": tool,
            "status": "ready",
            "target_structure_count": payload.get("target_structure_count"),
            "target_method_count": payload.get("target_method_count"),
            "missing_or_pending_count": payload.get("missing_or_pending_count"),
            "status_counts": payload.get("status_counts", {}),
        }
    if tool == "p2rank":
        return {
            "tool": tool,
            "status": "ready" if payload.get("ready_for_manifest_run") else "needs_external_output",
            "roots": roots or [],
            "discovered_prediction_file_count": payload.get("discovered_prediction_file_count", 0),
            "manifest_row_count": payload.get("manifest_row_count", 0),
            "skipped_count": payload.get("skipped_count", 0),
            "recommended_next_action": payload.get("recommended_next_action", ""),
        }
    if tool == "fpocket":
        return {
            "tool": tool,
            "status": "ready" if payload.get("ready_for_manifest_run") else "needs_external_output",
            "roots": roots or [],
            "discovered_pocket_file_count": payload.get("discovered_pocket_file_count", 0),
            "manifest_row_count": payload.get("manifest_row_count", 0),
            "skipped_count": payload.get("skipped_count", 0),
            "recommended_next_action": payload.get("recommended_next_action", ""),
        }
    return {"tool": tool, "status": "ready", "payload": payload}


def _build_next_actions(summary: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    expansion = summary.get("expansion_plan") or {}
    status_counts = dict(expansion.get("status_counts") or {})
    p2rank = summary.get("p2rank") or {}
    fpocket = summary.get("fpocket") or {}
    needs_external_outputs = bool(status_counts.get("needs_p2rank_output") or status_counts.get("needs_fpocket_output"))
    has_external_roots = bool(p2rank.get("roots") or fpocket.get("roots"))
    if needs_external_outputs and not has_external_roots:
        actions.append(
            "If no external outputs exist yet, run python prepare_cd38_external_tool_inputs.py, then run check_cd38_external_tool_environment.py or external_tool_inputs/check_external_tool_environment.ps1."
        )
    if p2rank.get("manifest_row_count"):
        actions.append("Review the P2Rank readiness report; rerun with --run_discovered or run the generated P2Rank manifest when ready.")
    elif p2rank.get("roots"):
        actions.append("Check P2Rank readiness report; no runnable P2Rank manifest rows were created.")
    elif status_counts.get("needs_p2rank_output"):
        actions.append("Run external P2Rank for structures listed as needs_p2rank_output, then rerun with --p2rank_root.")

    if fpocket.get("manifest_row_count"):
        actions.append("Review the fpocket readiness report; rerun with --run_discovered or run the generated fpocket manifest when ready.")
    elif fpocket.get("roots"):
        actions.append("Check fpocket readiness report; no runnable fpocket manifest rows were created.")
    elif status_counts.get("needs_fpocket_output"):
        actions.append("Run external fpocket for structures listed as needs_fpocket_output, then rerun with --fpocket_root.")

    if not actions:
        actions.append("No immediate readiness blockers were detected in the refreshed reports.")
    return actions


def _build_report(summary: dict[str, Any]) -> str:
    next_actions = summary.get("next_actions", [])
    lines = [
        "# CD38 Benchmark Readiness Refresh",
        "",
        "This report consolidates ligand suitability, benchmark expansion gaps, and optional external P2Rank/fpocket discovery checks.",
        "",
        "## Summary",
        "",
        f"- Ligand scan status: `{summary['ligand_candidates']['status']}`",
        f"- Expansion plan status: `{summary['expansion_plan']['status']}`",
        f"- P2Rank discovery status: `{summary['p2rank']['status']}`",
        f"- fpocket discovery status: `{summary['fpocket']['status']}`",
        f"- Missing or pending structure-method pairs: `{summary['expansion_plan'].get('missing_or_pending_count', '')}`",
        "",
        "## Next Actions",
        "",
    ]
    for action in next_actions:
        lines.append(f"- {action}")

    lines.extend(
        [
            "",
            "## Detail Files",
            "",
            f"- Readiness summary JSON: `{summary['outputs']['summary_json']}`",
            f"- Expansion missing actions CSV: `{summary['outputs']['expansion_missing_actions_csv']}`",
            f"- Ligand candidate report: `{summary['outputs']['ligand_candidate_report_md']}`",
            f"- P2Rank readiness report: `{summary['outputs']['p2rank_report_md']}`",
            f"- fpocket readiness report: `{summary['outputs']['fpocket_report_md']}`",
            "",
            "## Tool Status",
            "",
            "| Tool | Status | Key Count | Notes |",
            "| --- | --- | ---: | --- |",
        ]
    )
    ligand = summary["ligand_candidates"]
    lines.append(
        f"| ligand_candidates | {ligand.get('status')} | {ligand.get('recommended_count', '')} | recommended ligand candidates |"
    )
    expansion = summary["expansion_plan"]
    lines.append(
        f"| expansion_plan | {expansion.get('status')} | {expansion.get('missing_or_pending_count', '')} | missing or pending pairs |"
    )
    p2rank = summary["p2rank"]
    lines.append(
        f"| p2rank | {p2rank.get('status')} | {p2rank.get('manifest_row_count', '')} | runnable manifest rows |"
    )
    fpocket = summary["fpocket"]
    lines.append(
        f"| fpocket | {fpocket.get('status')} | {fpocket.get('manifest_row_count', '')} | runnable manifest rows |"
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = _build_parser().parse_args()
    repo_root = _repo_root()
    py = str(args.python_executable or sys.executable)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    command_records: list[dict[str, Any]] = []
    payloads: dict[str, dict[str, Any] | None] = {
        "ligand_candidates": None,
        "expansion_plan": None,
        "p2rank": None,
        "fpocket": None,
    }

    tasks: list[tuple[str, list[str], Path]] = []
    if not bool(args.skip_ligand_scan):
        command, summary_path = _make_ligand_command(args, py, out_dir)
        tasks.append(("ligand_candidates", command, summary_path))
    if not bool(args.skip_expansion_plan):
        command, summary_path = _make_expansion_command(args, py)
        tasks.append(("expansion_plan", command, summary_path))

    p2rank_roots = _as_roots(args.p2rank_root)
    fpocket_roots = _as_roots(args.fpocket_root)
    if p2rank_roots:
        command, summary_path = _make_p2rank_command(args, py, out_dir, p2rank_roots)
        tasks.append(("p2rank", command, summary_path))
    if fpocket_roots:
        command, summary_path = _make_fpocket_command(args, py, out_dir, fpocket_roots)
        tasks.append(("fpocket", command, summary_path))

    for tool, command, summary_path in tasks:
        record = _run_command(command, repo_root)
        record["tool"] = tool
        record["summary_path"] = str(summary_path)
        command_records.append(record)
        if record["returncode"] != 0 and not bool(args.continue_on_error):
            payloads[tool] = _load_json(summary_path)
            break
        payloads[tool] = _load_json(summary_path)

    outputs = {
        "summary_json": str(out_dir / "cd38_benchmark_readiness_summary.json"),
        "report_md": str(out_dir / "cd38_benchmark_readiness_report.md"),
        "commands_json": str(out_dir / "cd38_benchmark_readiness_commands.json"),
        "expansion_missing_actions_csv": str(Path(args.expansion_plan_out_dir).expanduser() / "cd38_benchmark_missing_actions.csv"),
        "ligand_candidate_report_md": str(Path(args.ligand_candidates_out_dir).expanduser() / "cd38_ligand_candidate_report.md"),
        "p2rank_report_md": str(out_dir / "p2rank_discovered_manifest.csv.report.md"),
        "fpocket_report_md": str(out_dir / "fpocket_discovered_manifest.csv.report.md"),
    }
    summary: dict[str, Any] = {
        "structure_targets_csv": str(Path(args.structure_targets_csv).expanduser().resolve()),
        "manifest_csv": str(Path(args.manifest_csv).expanduser().resolve()),
        "results_root": str(Path(args.results_root).expanduser().resolve()),
        "truth_file": str(Path(args.truth_file).expanduser().resolve()),
        "out_dir": str(out_dir),
        "run_discovered": bool(args.run_discovered),
        "ligand_candidates": _short_tool_summary("ligand_candidates", payloads["ligand_candidates"]),
        "expansion_plan": _short_tool_summary("expansion_plan", payloads["expansion_plan"]),
        "p2rank": _short_tool_summary("p2rank", payloads["p2rank"], roots=p2rank_roots),
        "fpocket": _short_tool_summary("fpocket", payloads["fpocket"], roots=fpocket_roots),
        "commands": command_records,
        "outputs": outputs,
    }
    summary["next_actions"] = _build_next_actions(summary)

    summary_json = Path(outputs["summary_json"])
    report_md = Path(outputs["report_md"])
    commands_json = Path(outputs["commands_json"])
    summary_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    commands_json.write_text(json.dumps(command_records, ensure_ascii=True, indent=2), encoding="utf-8")
    report_md.write_text(_build_report(summary), encoding="utf-8")

    print(f"Saved: {summary_json}")
    print(f"Saved: {commands_json}")
    print(f"Saved: {report_md}")
    print(f"Next actions: {len(summary['next_actions'])}")
    for action in summary["next_actions"]:
        print(f"- {action}")


if __name__ == "__main__":
    main()
