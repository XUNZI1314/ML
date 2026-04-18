from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate whether the latest CD38 returned external-tool package is safe to import/finalize."
    )
    parser.add_argument(
        "--import_summary_json",
        default="benchmarks/cd38/external_tool_inputs/imported_outputs/cd38_external_tool_output_import_summary.json",
        help="Summary JSON written by import_cd38_external_tool_outputs.py.",
    )
    parser.add_argument(
        "--scan_csv",
        default="benchmarks/cd38/external_tool_inputs/imported_outputs/cd38_external_tool_output_import_scan.csv",
        help="Scan CSV written by import_cd38_external_tool_outputs.py.",
    )
    parser.add_argument(
        "--action_plan_csv",
        default="benchmarks/cd38/action_plan/cd38_external_benchmark_action_plan.csv",
        help="Optional consolidated action plan CSV for context.",
    )
    parser.add_argument(
        "--out_dir",
        default="benchmarks/cd38/external_tool_inputs/import_gate",
        help="Output directory for gate JSON/Markdown/CSV.",
    )
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when the gate status is not PASS.")
    return parser


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _resolve(path: str | Path) -> Path:
    target = Path(path).expanduser()
    if not target.is_absolute():
        target = _repo_root() / target
    return target.resolve()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"load_error": str(exc), "path": str(path)}
    return payload if isinstance(payload, dict) else {"payload": payload}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _safe_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _is_truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes", "y"}


def _has_synthetic_fixture_marker(import_summary: dict[str, Any], scan_rows: list[dict[str, str]]) -> bool:
    source = str(import_summary.get("source") or "").lower()
    if "smoke_fixture" in source:
        return True
    for row in scan_rows:
        combined = " ".join(
            [
                str(row.get("archive_or_source_path") or ""),
                str(row.get("package_relative_path") or ""),
                str(row.get("reason") or ""),
            ]
        ).lower()
        if "smoke_fixture" in combined or "synthetic" in combined or "do_not_use_for_benchmark" in combined:
            return True
    return False


def _action_plan_context(action_plan_rows: list[dict[str, str]]) -> dict[str, Any]:
    benchmark_rows = [row for row in action_plan_rows if _is_truthy(row.get("benchmark_gap"))]
    missing_rows = [
        row
        for row in benchmark_rows
        if str(row.get("coverage_status") or row.get("current_status") or "").strip().lower()
        not in {"ready_from_return", "complete", "ready", "imported"}
    ]
    priority_missing = [
        f"{row.get('pdb_id')}:{row.get('method')}"
        for row in missing_rows
        if str(row.get("pdb_id") or "").strip() and str(row.get("method") or "").strip()
    ]
    return {
        "action_plan_rows": int(len(action_plan_rows)),
        "benchmark_gap_rows": int(len(benchmark_rows)),
        "benchmark_gap_rows_still_missing": int(len(missing_rows)),
        "benchmark_gap_missing_keys": priority_missing,
    }


def _decide_gate(import_summary: dict[str, Any], scan_rows: list[dict[str, str]]) -> tuple[str, str, list[str]]:
    if not import_summary:
        return (
            "FAIL_NO_IMPORT_SUMMARY",
            "No import summary exists. Run a dry-run import or finalize with --import_source first.",
            ["Run `python import_cd38_external_tool_outputs.py --source <returned_zip_or_dir> --dry_run`."],
        )
    if import_summary.get("load_error"):
        return (
            "FAIL_IMPORT_SUMMARY_UNREADABLE",
            f"Import summary could not be read: {import_summary.get('load_error')}",
            ["Regenerate the import summary with a dry-run import."],
        )

    source_diagnosis = str(import_summary.get("source_diagnosis") or "")
    coverage = import_summary.get("coverage") if isinstance(import_summary.get("coverage"), dict) else {}
    candidate_count = _safe_int(import_summary.get("candidate_file_count"))
    expected_count = _safe_int(coverage.get("expected_output_count"))
    ready_count = _safe_int(coverage.get("ready_expected_output_count"))
    missing_count = _safe_int(coverage.get("missing_expected_output_count"))
    unexpected_count = _safe_int(coverage.get("unexpected_output_count"))
    imported_count = _safe_int(import_summary.get("imported_count"))
    dry_run_count = _safe_int(import_summary.get("dry_run_count"))

    if _has_synthetic_fixture_marker(import_summary, scan_rows):
        return (
            "FAIL_SYNTHETIC_FIXTURE",
            "The returned source is a synthetic smoke-test fixture. It proves path handling only and must not be imported as benchmark evidence.",
            [
                "Do not run benchmark import on this source.",
                "Use a real returned package generated by P2Rank/fpocket on the transfer zip.",
            ],
        )
    if source_diagnosis == "input_package_without_external_outputs":
        return (
            "FAIL_INPUT_PACKAGE",
            "The source looks like the original transfer/input package, not a completed return package.",
            [
                "Run P2Rank/fpocket on the transfer package first.",
                "Return `p2rank_outputs/` and `fpocket_runs/*/*_out/`, then dry-run again.",
            ],
        )
    if candidate_count <= 0:
        return (
            "FAIL_NO_CANDIDATE_OUTPUTS",
            "No importable P2Rank/fpocket output files were detected.",
            [
                "Open the import scan CSV and inspect ignored files.",
                "Confirm the returned source contains `p2rank_outputs/` or `fpocket_runs/*/*_out/`.",
            ],
        )
    if expected_count <= 0:
        return (
            "FAIL_NO_EXPECTED_RETURN_MANIFEST",
            "Expected return manifest was not available, so coverage cannot be evaluated.",
            ["Regenerate CD38 external tool inputs, then dry-run the returned package again."],
        )
    if ready_count < expected_count or missing_count > 0:
        missing = coverage.get("missing_expected_outputs") if isinstance(coverage.get("missing_expected_outputs"), list) else []
        return (
            "WARN_PARTIAL_RETURN",
            f"The returned package is importable but incomplete: ready={ready_count}, expected={expected_count}.",
            [
                "Do not treat the benchmark as complete yet.",
                f"Missing expected outputs: {', '.join(str(item) for item in missing) if missing else 'see coverage CSV'}.",
                "Use the repair plan CSV to rerun only missing PDB-method outputs.",
            ],
        )
    if unexpected_count > 0:
        return (
            "WARN_UNEXPECTED_OUTPUTS",
            "All expected outputs are present, but unexpected returned outputs were also found.",
            [
                "Review unexpected outputs before importing into benchmark panel.",
                "If they are harmless extra files, proceed with finalize after review.",
            ],
        )
    if imported_count > 0:
        return (
            "PASS_IMPORTED_READY_FOR_FINALIZE",
            "Expected outputs have been imported into the local package and are ready for readiness/finalize checks.",
            ["Run or review `python finalize_cd38_external_benchmark.py --run_discovered --run_sensitivity`."],
        )
    if dry_run_count > 0:
        return (
            "PASS_READY_FOR_IMPORT",
            "Dry-run found all expected external outputs. The returned package is ready to import.",
            ["Run finalize without `--import_dry_run`, ideally with `--run_discovered --run_sensitivity`."],
        )
    return (
        "PASS_READY_FOR_REVIEW",
        "All expected outputs are accounted for; review import manifest before final benchmark import.",
        ["Review the import manifest, then run finalize with `--run_discovered`."],
    )


def _build_report(summary: dict[str, Any]) -> str:
    coverage = summary.get("coverage") if isinstance(summary.get("coverage"), dict) else {}
    context = summary.get("action_plan_context") if isinstance(summary.get("action_plan_context"), dict) else {}
    lines = [
        "# CD38 Return Package Gate",
        "",
        "This gate evaluates whether the latest returned P2Rank/fpocket package is safe to import into the CD38 benchmark workflow.",
        "",
        "## Gate Decision",
        "",
        f"- Gate status: `{summary.get('gate_status')}`",
        f"- Decision: {summary.get('decision_message')}",
        f"- Source diagnosis: `{summary.get('source_diagnosis')}`",
        f"- Candidate files: `{summary.get('candidate_file_count')}`",
        f"- Imported files: `{summary.get('imported_count')}`",
        f"- Dry-run files: `{summary.get('dry_run_count')}`",
        f"- Expected coverage: `{coverage.get('ready_expected_output_count', '')}/{coverage.get('expected_output_count', '')}`",
        f"- Missing expected outputs: `{coverage.get('missing_expected_output_count', '')}`",
        f"- Unexpected outputs: `{coverage.get('unexpected_output_count', '')}`",
        f"- Synthetic fixture detected: `{summary.get('synthetic_fixture_detected')}`",
        "",
        "## Recommended Actions",
        "",
    ]
    for action in summary.get("recommended_actions", []):
        lines.append(f"- {action}")
    lines.extend(
        [
            "",
            "## Action Plan Context",
            "",
            f"- Action plan rows: `{context.get('action_plan_rows', '')}`",
            f"- Benchmark gap rows: `{context.get('benchmark_gap_rows', '')}`",
            f"- Benchmark gap rows still missing: `{context.get('benchmark_gap_rows_still_missing', '')}`",
            "",
            "## Files",
            "",
            f"- Import summary: `{summary.get('import_summary_json')}`",
            f"- Import scan CSV: `{summary.get('scan_csv')}`",
            f"- Gate summary JSON: `{summary.get('outputs', {}).get('summary_json', '')}`",
            f"- Gate report: `{summary.get('outputs', {}).get('report_md', '')}`",
            f"- Gate decision CSV: `{summary.get('outputs', {}).get('decision_csv', '')}`",
            "",
            "## Boundary",
            "",
            "This gate checks returned package shape, coverage, and obvious misuse. It does not validate scientific pocket accuracy. Real P2Rank/fpocket outputs still need benchmark scoring and manual review.",
            "",
        ]
    )
    return "\n".join(lines)


def build_gate(
    *,
    import_summary_json: str | Path = "benchmarks/cd38/external_tool_inputs/imported_outputs/cd38_external_tool_output_import_summary.json",
    scan_csv: str | Path = "benchmarks/cd38/external_tool_inputs/imported_outputs/cd38_external_tool_output_import_scan.csv",
    action_plan_csv: str | Path = "benchmarks/cd38/action_plan/cd38_external_benchmark_action_plan.csv",
    out_dir: str | Path = "benchmarks/cd38/external_tool_inputs/import_gate",
) -> dict[str, Any]:
    import_summary_path = _resolve(import_summary_json)
    scan_path = _resolve(scan_csv)
    action_plan_path = _resolve(action_plan_csv)
    out = _resolve(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    import_summary = _load_json(import_summary_path)
    scan_rows = _read_csv(scan_path)
    action_plan_rows = _read_csv(action_plan_path)
    gate_status, decision_message, recommended_actions = _decide_gate(import_summary, scan_rows)
    coverage = import_summary.get("coverage") if isinstance(import_summary.get("coverage"), dict) else {}
    synthetic_fixture_detected = _has_synthetic_fixture_marker(import_summary, scan_rows)

    summary: dict[str, Any] = {
        "gate_status": gate_status,
        "decision_message": decision_message,
        "recommended_actions": recommended_actions,
        "import_summary_json": str(import_summary_path),
        "scan_csv": str(scan_path),
        "action_plan_csv": str(action_plan_path),
        "source": str(import_summary.get("source") or ""),
        "source_diagnosis": str(import_summary.get("source_diagnosis") or ""),
        "candidate_file_count": _safe_int(import_summary.get("candidate_file_count")),
        "imported_count": _safe_int(import_summary.get("imported_count")),
        "dry_run_count": _safe_int(import_summary.get("dry_run_count")),
        "coverage": coverage,
        "synthetic_fixture_detected": bool(synthetic_fixture_detected),
        "action_plan_context": _action_plan_context(action_plan_rows),
    }
    summary["outputs"] = {
        "summary_json": str(out / "cd38_return_package_gate_summary.json"),
        "report_md": str(out / "cd38_return_package_gate_report.md"),
        "decision_csv": str(out / "cd38_return_package_gate_decision.csv"),
    }

    Path(summary["outputs"]["summary_json"]).write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    Path(summary["outputs"]["report_md"]).write_text(_build_report(summary), encoding="utf-8")
    _write_csv(
        Path(summary["outputs"]["decision_csv"]),
        [
            {
                "gate_status": summary["gate_status"],
                "source_diagnosis": summary["source_diagnosis"],
                "candidate_file_count": summary["candidate_file_count"],
                "ready_expected_output_count": coverage.get("ready_expected_output_count", ""),
                "expected_output_count": coverage.get("expected_output_count", ""),
                "missing_expected_output_count": coverage.get("missing_expected_output_count", ""),
                "synthetic_fixture_detected": summary["synthetic_fixture_detected"],
                "decision_message": summary["decision_message"],
            }
        ],
        [
            "gate_status",
            "source_diagnosis",
            "candidate_file_count",
            "ready_expected_output_count",
            "expected_output_count",
            "missing_expected_output_count",
            "synthetic_fixture_detected",
            "decision_message",
        ],
    )
    return summary


def main() -> None:
    args = _build_parser().parse_args()
    summary = build_gate(
        import_summary_json=args.import_summary_json,
        scan_csv=args.scan_csv,
        action_plan_csv=args.action_plan_csv,
        out_dir=args.out_dir,
    )
    print(f"Saved: {summary['outputs']['summary_json']}")
    print(f"Saved: {summary['outputs']['report_md']}")
    print(f"Gate status: {summary['gate_status']}")
    if args.strict and not str(summary["gate_status"]).startswith("PASS"):
        raise SystemExit(str(summary["gate_status"]))


if __name__ == "__main__":
    main()
