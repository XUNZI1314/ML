from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run an end-to-end CD38 external-tool workflow self-test without using real external outputs. "
            "The test validates packaging, return-package gate behavior, and synthetic fixture blocking."
        )
    )
    parser.add_argument(
        "--package_dir",
        default="benchmarks/cd38/external_tool_inputs",
        help="CD38 external_tool_inputs package to test.",
    )
    parser.add_argument(
        "--work_dir",
        default="smoke_test_outputs/cd38_external_workflow_selftest",
        help="Output directory for the workflow self-test.",
    )
    parser.add_argument("--python_executable", default=None, help="Python executable for child commands.")
    parser.add_argument(
        "--skip_public_starter",
        action="store_true",
        help="Skip the optional public starter refresh step.",
    )
    return parser


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _resolve(path: str | Path, repo_root: Path) -> Path:
    target = Path(path).expanduser()
    if not target.is_absolute():
        target = repo_root / target
    return target.resolve()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"load_error": str(exc), "path": str(path)}
    return payload if isinstance(payload, dict) else {"payload": payload}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _run(command: list[str], *, cwd: Path) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return {
        "command": command,
        "returncode": int(completed.returncode),
        "status": "success" if int(completed.returncode) == 0 else "failed",
        "stdout": str(completed.stdout or ""),
        "stderr": str(completed.stderr or ""),
    }


def _step_result(name: str, record: dict[str, Any], *, passed: bool, detail: str) -> dict[str, Any]:
    return {
        "step": name,
        "passed": bool(passed),
        "detail": detail,
        "returncode": record.get("returncode"),
        "status": record.get("status"),
        "command": record.get("command"),
        "stdout_tail": "\n".join(str(record.get("stdout") or "").splitlines()[-20:]),
        "stderr_tail": "\n".join(str(record.get("stderr") or "").splitlines()[-20:]),
    }


def _build_report(summary: dict[str, Any]) -> str:
    lines = [
        "# CD38 External Workflow Self-Test",
        "",
        "This self-test checks the local CD38 external-tool workflow wiring without using real external-tool outputs.",
        "",
        "## Result",
        "",
        f"- Overall status: `{summary.get('overall_status')}`",
        f"- Passed steps: `{summary.get('passed_step_count')}/{summary.get('step_count')}`",
        f"- Transfer zip: `{summary.get('transfer_zip')}`",
        f"- Original transfer gate status: `{summary.get('original_transfer_gate_status')}`",
        f"- Synthetic fixture gate status: `{summary.get('synthetic_fixture_gate_status')}`",
        "",
        "## Steps",
        "",
        "| Step | Passed | Detail |",
        "| --- | --- | --- |",
    ]
    for step in summary.get("steps", []):
        detail = str(step.get("detail") or "").replace("|", "\\|")
        lines.append(f"| {step.get('step')} | {step.get('passed')} | {detail} |")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This test proves packaging and safety-gate behavior only. It does not run P2Rank/fpocket and does not create scientific CD38 benchmark evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def run_selftest(
    *,
    package_dir: str | Path = "benchmarks/cd38/external_tool_inputs",
    work_dir: str | Path = "smoke_test_outputs/cd38_external_workflow_selftest",
    python_executable: str | None = None,
    skip_public_starter: bool = False,
) -> dict[str, Any]:
    repo_root = _repo_root()
    py = str(python_executable or sys.executable)
    package = _resolve(package_dir, repo_root)
    work = _resolve(work_dir, repo_root)
    work.mkdir(parents=True, exist_ok=True)
    transfer_dir = work / "transfer"
    transfer_zip = transfer_dir / "cd38_external_tool_inputs_transfer.zip"
    finalize_out = work / "finalize_original_transfer"
    return_selftest_dir = work / "return_import_selftest"

    steps: list[dict[str, Any]] = []

    package_command = [
        py,
        "package_cd38_external_tool_inputs.py",
        "--package_dir",
        str(package),
        "--out_dir",
        str(transfer_dir),
    ]
    package_record = _run(package_command, cwd=repo_root)
    transfer_summary = _load_json(transfer_dir / "cd38_external_tool_inputs_transfer_summary.json")
    package_passed = (
        int(package_record["returncode"]) == 0
        and transfer_zip.exists()
        and int(transfer_summary.get("file_count") or 0) > 0
    )
    steps.append(
        _step_result(
            "package_transfer_zip",
            package_record,
            passed=package_passed,
            detail=f"transfer_zip_exists={transfer_zip.exists()}, file_count={transfer_summary.get('file_count')}",
        )
    )

    gate_command = [
        py,
        "finalize_cd38_external_benchmark.py",
        "--package_dir",
        str(package),
        "--out_dir",
        str(finalize_out),
        "--import_source",
        str(transfer_zip),
        "--continue_on_error",
        "--strict_import_gate",
    ]
    gate_record = _run(gate_command, cwd=repo_root)
    gate_summary = _load_json(finalize_out / "cd38_external_benchmark_finalize_summary.json")
    original_gate_status = str(gate_summary.get("import_gate_status") or "")
    gate_passed = int(gate_record["returncode"]) != 0 and original_gate_status == "FAIL_INPUT_PACKAGE"
    steps.append(
        _step_result(
            "original_transfer_strict_gate_blocks",
            gate_record,
            passed=gate_passed,
            detail=f"returncode={gate_record.get('returncode')}, gate_status={original_gate_status}",
        )
    )

    return_command = [
        py,
        "selftest_cd38_return_import_workflow.py",
        "--package_dir",
        str(package),
        "--work_dir",
        str(return_selftest_dir),
    ]
    return_record = _run(return_command, cwd=repo_root)
    return_summary = _load_json(return_selftest_dir / "cd38_return_import_selftest_summary.json")
    importer_summary = (
        return_summary.get("importer_summary")
        if isinstance(return_summary.get("importer_summary"), dict)
        else {}
    )
    coverage = importer_summary.get("coverage") if isinstance(importer_summary.get("coverage"), dict) else {}
    fixture_gate = (
        return_summary.get("gate_summary")
        if isinstance(return_summary.get("gate_summary"), dict)
        else {}
    )
    synthetic_gate_status = str(fixture_gate.get("gate_status") or "")
    return_passed = (
        int(return_record["returncode"]) == 0
        and str(return_summary.get("status") or "") == "pass"
        and int(coverage.get("ready_expected_output_count") or 0) == int(coverage.get("expected_output_count") or -1)
        and synthetic_gate_status == "FAIL_SYNTHETIC_FIXTURE"
    )
    steps.append(
        _step_result(
            "synthetic_return_fixture_selftest",
            return_record,
            passed=return_passed,
            detail=(
                f"selftest_status={return_summary.get('status')}, "
                f"coverage={coverage.get('ready_expected_output_count')}/{coverage.get('expected_output_count')}, "
                f"gate_status={synthetic_gate_status}"
            ),
        )
    )

    public_summary: dict[str, Any] = {}
    if skip_public_starter:
        steps.append(
            {
                "step": "public_starter_refresh",
                "passed": True,
                "detail": "skipped_by_flag",
                "returncode": "",
                "status": "skipped",
                "command": [],
                "stdout_tail": "",
                "stderr_tail": "",
            }
        )
    else:
        public_command = [py, "run_cd38_public_starter.py", "--continue_on_error"]
        public_record = _run(public_command, cwd=repo_root)
        public_summary = _load_json(repo_root / "benchmarks" / "cd38" / "public_starter" / "cd38_public_starter_summary.json")
        panel = public_summary.get("panel") if isinstance(public_summary.get("panel"), dict) else {}
        public_passed = int(public_record["returncode"]) == 0 and int(public_summary.get("failed_command_count") or 0) == 0
        steps.append(
            _step_result(
                "public_starter_refresh",
                public_record,
                passed=public_passed,
                detail=f"failed_command_count={public_summary.get('failed_command_count')}, panel_rows={panel.get('row_count')}",
            )
        )

    passed_count = sum(1 for step in steps if bool(step.get("passed")))
    overall_status = "pass" if passed_count == len(steps) else "fail"
    summary: dict[str, Any] = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "overall_status": overall_status,
        "step_count": int(len(steps)),
        "passed_step_count": int(passed_count),
        "package_dir": str(package),
        "work_dir": str(work),
        "transfer_zip": str(transfer_zip),
        "original_transfer_gate_status": original_gate_status,
        "synthetic_fixture_gate_status": synthetic_gate_status,
        "steps": steps,
        "artifacts": {
            "summary_json": str(work / "cd38_external_workflow_selftest_summary.json"),
            "report_md": str(work / "cd38_external_workflow_selftest_report.md"),
            "transfer_zip": str(transfer_zip),
            "original_transfer_finalize_summary": str(finalize_out / "cd38_external_benchmark_finalize_summary.json"),
            "return_import_selftest_summary": str(return_selftest_dir / "cd38_return_import_selftest_summary.json"),
            "public_starter_summary": str(repo_root / "benchmarks" / "cd38" / "public_starter" / "cd38_public_starter_summary.json"),
        },
    }
    summary_json = Path(summary["artifacts"]["summary_json"])
    report_md = Path(summary["artifacts"]["report_md"])
    _write_json(summary_json, summary)
    report_md.write_text(_build_report(summary), encoding="utf-8")
    print(f"Saved: {summary_json}")
    print(f"Saved: {report_md}")
    print(f"Overall status: {overall_status}")
    if overall_status != "pass":
        raise SystemExit("CD38 external workflow self-test failed.")
    return summary


def main() -> None:
    args = _build_parser().parse_args()
    run_selftest(
        package_dir=args.package_dir,
        work_dir=args.work_dir,
        python_executable=args.python_executable,
        skip_public_starter=bool(args.skip_public_starter),
    )


if __name__ == "__main__":
    main()
