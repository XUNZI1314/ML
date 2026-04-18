from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a synthetic CD38 external-tool return package and dry-run the importer. "
            "This is a path/coverage smoke test only; it must not be used as benchmark evidence."
        )
    )
    parser.add_argument(
        "--package_dir",
        default="benchmarks/cd38/external_tool_inputs",
        help="Local CD38 external_tool_inputs package with expected return manifest.",
    )
    parser.add_argument(
        "--work_dir",
        default="smoke_test_outputs/cd38_return_import_selftest",
        help="Directory for the synthetic return fixture and self-test reports.",
    )
    parser.add_argument("--python_executable", default=None, help="Python executable for the importer subprocess.")
    return parser


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _write_p2rank_fixture(path: Path, pdb_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "name,rank,score,probability,residue_ids",
                f"pocket1,1,1.0,0.90,A_125 A_127 A_129 A_174 A_206",
                f"pocket2,2,0.5,0.50,A_37 A_38 A_39 A_40",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _pdb_atom_line(serial: int, atom_name: str, resname: str, chain: str, resseq: int, x: float) -> str:
    return (
        f"ATOM  {serial:5d} {atom_name:<4s} {resname:>3s} {chain:1s}"
        f"{resseq:4d}    {x:8.3f}{0.0:8.3f}{0.0:8.3f}{1.00:6.2f}{20.00:6.2f}           C"
    )


def _write_fpocket_fixture(path: Path, pdb_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    residue_numbers = [125, 127, 129, 174, 206]
    lines = [
        "REMARK Synthetic fpocket smoke-test fixture. Do not use as benchmark evidence.",
    ]
    serial = 1
    for index, resseq in enumerate(residue_numbers):
        lines.append(_pdb_atom_line(serial, "CA", "ALA", "A", resseq, float(index)))
        serial += 1
    lines.extend(["TER", "END", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _materialize_expected_file(content_root: Path, row: dict[str, str]) -> Path | None:
    pdb_id = str(row.get("pdb_id") or "").strip().upper()
    method = str(row.get("method") or "").strip().lower()
    expected_path = str(row.get("expected_return_path") or "").strip().replace("\\", "/")
    if not pdb_id or not method or not expected_path:
        return None

    if method == "p2rank":
        relative_path = Path(expected_path)
        target = content_root / relative_path
        _write_p2rank_fixture(target, pdb_id)
        return target
    if method == "fpocket":
        concrete_path = expected_path.replace("pocket*_atm.pdb", "pocket1_atm.pdb")
        relative_path = Path(concrete_path)
        target = content_root / relative_path
        _write_fpocket_fixture(target, pdb_id)
        return target
    return None


def _write_fixture_zip(source_root: Path, out_zip: Path) -> None:
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(source_root.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(source_root.parent).as_posix())


def _build_report(summary: dict[str, Any]) -> str:
    importer = summary.get("importer_summary") if isinstance(summary.get("importer_summary"), dict) else {}
    coverage = importer.get("coverage") if isinstance(importer.get("coverage"), dict) else {}
    lines = [
        "# CD38 Return Import Workflow Self-Test",
        "",
        "This report uses a synthetic returned package to test path detection and expected-output coverage only.",
        "",
        "## Result",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Fixture zip: `{summary.get('fixture_zip')}`",
        f"- Synthetic files: `{summary.get('synthetic_file_count')}`",
        f"- Importer return code: `{summary.get('importer_returncode')}`",
        f"- Candidate output files: `{importer.get('candidate_file_count', '')}`",
        f"- Expected outputs ready: `{coverage.get('ready_expected_output_count', '')}/{coverage.get('expected_output_count', '')}`",
        f"- Missing expected outputs: `{coverage.get('missing_expected_output_count', '')}`",
        f"- Source diagnosis: `{importer.get('source_diagnosis', '')}`",
        f"- Gate return code: `{summary.get('gate_returncode')}`",
        f"- Gate status: `{(summary.get('gate_summary') or {}).get('gate_status', '')}`",
        "",
        "## Boundary",
        "",
        "The generated P2Rank/fpocket files are synthetic fixtures. They prove only that the returned-package importer can find the expected paths. The gate is expected to block this fixture as `FAIL_SYNTHETIC_FIXTURE`. These files are not real P2Rank/fpocket outputs and must not be used as CD38 benchmark evidence.",
        "",
    ]
    if summary.get("failure_reason"):
        lines.extend(["## Failure Reason", "", str(summary["failure_reason"]), ""])
    return "\n".join(lines)


def run_selftest(
    *,
    package_dir: str | Path = "benchmarks/cd38/external_tool_inputs",
    work_dir: str | Path = "smoke_test_outputs/cd38_return_import_selftest",
    python_executable: str | None = None,
) -> dict[str, Any]:
    repo_root = _repo_root()
    py = str(python_executable or sys.executable)
    package = Path(package_dir).expanduser()
    if not package.is_absolute():
        package = repo_root / package
    package = package.resolve()
    work = Path(work_dir).expanduser()
    if not work.is_absolute():
        work = repo_root / work
    work = work.resolve()
    expected_csv = package / "cd38_external_tool_expected_returns.csv"
    if not expected_csv.exists():
        raise FileNotFoundError(f"Expected return manifest not found: {expected_csv}")

    fixture_root = work / "returned_external_tool_inputs" / "external_tool_inputs"
    fixture_zip = work / "returned_external_tool_inputs_smoke_fixture.zip"
    import_out = work / "import_dry_run"
    fixture_root.mkdir(parents=True, exist_ok=True)
    expected_rows = _read_csv(expected_csv)
    written_files = [
        path
        for row in expected_rows
        if (path := _materialize_expected_file(fixture_root, row)) is not None
    ]
    (fixture_root / "SMOKE_FIXTURE_DO_NOT_USE_FOR_BENCHMARK.txt").write_text(
        "Synthetic returned-package fixture for importer smoke testing only.\n",
        encoding="utf-8",
    )
    _write_fixture_zip(fixture_root, fixture_zip)

    command = [
        py,
        "import_cd38_external_tool_outputs.py",
        "--source",
        str(fixture_zip),
        "--package_dir",
        str(package),
        "--out_dir",
        str(import_out),
        "--dry_run",
    ]
    completed = subprocess.run(
        command,
        cwd=str(repo_root),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    importer_summary_path = import_out / "cd38_external_tool_output_import_summary.json"
    importer_summary: dict[str, Any] = {}
    if importer_summary_path.exists():
        importer_summary = json.loads(importer_summary_path.read_text(encoding="utf-8"))

    gate_out = work / "import_gate"
    gate_command = [
        py,
        "build_cd38_return_package_gate.py",
        "--import_summary_json",
        str(importer_summary_path),
        "--scan_csv",
        str(import_out / "cd38_external_tool_output_import_scan.csv"),
        "--out_dir",
        str(gate_out),
    ]
    gate_completed = subprocess.run(
        gate_command,
        cwd=str(repo_root),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    gate_summary_path = gate_out / "cd38_return_package_gate_summary.json"
    gate_summary: dict[str, Any] = {}
    if gate_summary_path.exists():
        gate_summary = json.loads(gate_summary_path.read_text(encoding="utf-8"))

    coverage = importer_summary.get("coverage") if isinstance(importer_summary.get("coverage"), dict) else {}
    expected_count = int(coverage.get("expected_output_count") or 0)
    ready_count = int(coverage.get("ready_expected_output_count") or 0)
    candidate_count = int(importer_summary.get("candidate_file_count") or 0)
    gate_status = str(gate_summary.get("gate_status") or "")
    status = "pass"
    failure_reason = ""
    if int(completed.returncode) != 0:
        status = "fail"
        failure_reason = f"Importer returned non-zero exit code: {completed.returncode}"
    elif int(gate_completed.returncode) != 0:
        status = "fail"
        failure_reason = f"Gate returned non-zero exit code: {gate_completed.returncode}"
    elif expected_count <= 0 or ready_count != expected_count:
        status = "fail"
        failure_reason = f"Expected coverage was not complete: ready={ready_count}, expected={expected_count}"
    elif candidate_count < expected_count:
        status = "fail"
        failure_reason = f"Candidate output count too low: candidates={candidate_count}, expected={expected_count}"
    elif gate_status != "FAIL_SYNTHETIC_FIXTURE":
        status = "fail"
        failure_reason = f"Gate did not block the synthetic fixture as expected: {gate_status}"

    summary: dict[str, Any] = {
        "status": status,
        "package_dir": str(package),
        "work_dir": str(work),
        "fixture_root": str(fixture_root),
        "fixture_zip": str(fixture_zip),
        "synthetic_file_count": int(len(written_files)),
        "expected_manifest_rows": int(len(expected_rows)),
        "importer_command": command,
        "importer_returncode": int(completed.returncode),
        "importer_stdout": str(completed.stdout or ""),
        "importer_stderr": str(completed.stderr or ""),
        "importer_summary": importer_summary,
        "gate_command": gate_command,
        "gate_returncode": int(gate_completed.returncode),
        "gate_stdout": str(gate_completed.stdout or ""),
        "gate_stderr": str(gate_completed.stderr or ""),
        "gate_summary": gate_summary,
        "failure_reason": failure_reason,
        "outputs": {
            "summary_json": str(work / "cd38_return_import_selftest_summary.json"),
            "report_md": str(work / "cd38_return_import_selftest_report.md"),
            "fixture_zip": str(fixture_zip),
            "import_summary_json": str(importer_summary_path),
            "import_report_md": str(import_out / "cd38_external_tool_output_import_report.md"),
            "import_coverage_csv": str(import_out / "cd38_external_tool_output_import_coverage.csv"),
            "gate_summary_json": str(gate_summary_path),
            "gate_report_md": str(gate_out / "cd38_return_package_gate_report.md"),
        },
    }
    summary_json = Path(summary["outputs"]["summary_json"])
    report_md = Path(summary["outputs"]["report_md"])
    _write_json(summary_json, summary)
    report_md.write_text(_build_report(summary), encoding="utf-8")

    print(f"Saved: {summary_json}")
    print(f"Saved: {report_md}")
    print(f"Fixture zip: {fixture_zip}")
    print(f"Status: {status}")
    if status != "pass":
        raise SystemExit(failure_reason or "CD38 return import workflow self-test failed.")
    return summary


def main() -> None:
    args = _build_parser().parse_args()
    run_selftest(
        package_dir=args.package_dir,
        work_dir=args.work_dir,
        python_executable=args.python_executable,
    )


if __name__ == "__main__":
    main()
