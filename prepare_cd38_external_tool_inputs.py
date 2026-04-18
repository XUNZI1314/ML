from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any
from urllib.request import urlopen


TARGET_COLUMNS = [
    "enabled",
    "priority",
    "pdb_id",
    "protein_chain",
    "ligand_chain",
    "ligand_resnames",
    "ligand_resseqs",
    "desired_methods",
    "p2rank_predictions_csv",
    "fpocket_root",
    "notes",
]


MANIFEST_COLUMNS = [
    "enabled",
    "priority",
    "pdb_id",
    "protein_chain",
    "desired_methods",
    "source_pdb_path",
    "tool_input_pdb_path",
    "p2rank_output_dir",
    "expected_p2rank_predictions_csv",
    "fpocket_run_dir",
    "fpocket_run_pdb_path",
    "expected_fpocket_output_dir",
    "expected_fpocket_pockets_dir",
    "recommended_p2rank_command",
    "recommended_fpocket_command",
    "notes",
]

EXPECTED_RETURN_COLUMNS = [
    "pdb_id",
    "method",
    "expected_return_path",
    "required_pattern",
    "generated_by",
    "coverage_key",
    "notes",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare CD38 PDB inputs and command templates for running external P2Rank/fpocket tools."
    )
    parser.add_argument(
        "--structure_targets_csv",
        default="benchmarks/cd38/cd38_structure_targets.csv",
        help="CD38 target structures CSV.",
    )
    parser.add_argument("--inputs_dir", default="benchmarks/cd38/inputs", help="Local cache for downloaded PDB files.")
    parser.add_argument(
        "--out_dir",
        default="benchmarks/cd38/external_tool_inputs",
        help="Output directory for external-tool input package.",
    )
    parser.add_argument("--chain_filter", default="A", help="Default chain filter used in follow-up readiness commands.")
    parser.add_argument(
        "--rank_by_pdb",
        default="3ROP=2,4OGW=1",
        help="Optional P2Rank rank overrides for follow-up readiness command.",
    )
    parser.add_argument("--force_download", action="store_true", help="Redownload PDB files even if cached locally.")
    parser.add_argument("--skip_download", action="store_true", help="Do not download missing PDB files.")
    return parser


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


def _truthy(value: Any, default: bool = True) -> bool:
    text = str(value or "").strip().lower()
    if not text or text == "default":
        return default
    return text in {"1", "true", "yes", "y", "on"}


def _split_csv(value: Any) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def _normalize_methods(value: Any) -> list[str]:
    methods: list[str] = []
    for method in _split_csv(value):
        normalized = method.strip().lower().replace("-", "_")
        if normalized and normalized not in methods:
            methods.append(normalized)
    return methods


def _path_for_script(path: Path) -> str:
    return str(path).replace("/", "\\")


def _download_pdb(pdb_id: str, inputs_dir: Path, *, force_download: bool, skip_download: bool) -> tuple[Path | None, str]:
    code = str(pdb_id).strip().upper()
    if len(code) != 4 or not code[0].isdigit():
        return None, f"invalid_pdb_id:{pdb_id}"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    out_path = inputs_dir / f"{code}.pdb"
    if out_path.exists() and not force_download:
        return out_path, "cached"
    if skip_download:
        return (out_path if out_path.exists() else None), "missing_download_skipped"
    url = f"https://files.rcsb.org/download/{code}.pdb"
    with urlopen(url, timeout=30) as resp:
        out_path.write_bytes(resp.read())
    return out_path, "downloaded"


def _copy_input_pdb(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == target.resolve():
        return "same_path"
    shutil.copy2(source, target)
    return "copied"


def _build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    targets_csv = Path(args.structure_targets_csv).expanduser().resolve()
    inputs_dir = Path(args.inputs_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    pdb_input_dir = out_dir / "pdbs"
    p2rank_output_root = out_dir / "p2rank_outputs"
    fpocket_run_root = out_dir / "fpocket_runs"
    targets = [row for row in _read_csv(targets_csv) if _truthy(row.get("enabled"), default=True)]

    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for target in targets:
        pdb_id = str(target.get("pdb_id") or "").strip().upper()
        if not pdb_id:
            records.append({"status": "skipped", "reason": "missing_pdb_id", "target": target})
            continue
        methods = _normalize_methods(target.get("desired_methods"))
        source_pdb, source_status = _download_pdb(
            pdb_id,
            inputs_dir,
            force_download=bool(args.force_download),
            skip_download=bool(args.skip_download),
        )
        if source_pdb is None or not source_pdb.exists():
            records.append({"status": "skipped", "reason": source_status, "pdb_id": pdb_id})
            continue

        tool_input_pdb = pdb_input_dir / f"{pdb_id}.pdb"
        copy_status = _copy_input_pdb(source_pdb, tool_input_pdb)

        p2rank_output_dir = p2rank_output_root / pdb_id
        expected_p2rank_predictions = p2rank_output_dir / f"{pdb_id}.pdb_predictions.csv"
        fpocket_run_dir = fpocket_run_root / pdb_id
        fpocket_run_pdb = fpocket_run_dir / f"{pdb_id}.pdb"
        expected_fpocket_output_dir = fpocket_run_dir / f"{pdb_id}_out"
        expected_fpocket_pockets_dir = expected_fpocket_output_dir / "pockets"
        fpocket_run_dir.mkdir(parents=True, exist_ok=True)
        _copy_input_pdb(source_pdb, fpocket_run_pdb)

        p2rank_command = (
            f'prank predict -f "{_path_for_script(tool_input_pdb)}" -o "{_path_for_script(p2rank_output_dir)}"'
            if "p2rank" in methods
            else ""
        )
        fpocket_command = (
            f'Push-Location "{_path_for_script(fpocket_run_dir)}"; fpocket -f "{pdb_id}.pdb"; Pop-Location'
            if "fpocket" in methods
            else ""
        )
        rows.append(
            {
                "enabled": "true",
                "priority": str(target.get("priority") or ""),
                "pdb_id": pdb_id,
                "protein_chain": str(target.get("protein_chain") or "A").strip(),
                "desired_methods": ",".join(methods),
                "source_pdb_path": str(source_pdb),
                "tool_input_pdb_path": str(tool_input_pdb),
                "p2rank_output_dir": str(p2rank_output_dir) if "p2rank" in methods else "",
                "expected_p2rank_predictions_csv": str(expected_p2rank_predictions) if "p2rank" in methods else "",
                "fpocket_run_dir": str(fpocket_run_dir) if "fpocket" in methods else "",
                "fpocket_run_pdb_path": str(fpocket_run_pdb) if "fpocket" in methods else "",
                "expected_fpocket_output_dir": str(expected_fpocket_output_dir) if "fpocket" in methods else "",
                "expected_fpocket_pockets_dir": str(expected_fpocket_pockets_dir) if "fpocket" in methods else "",
                "recommended_p2rank_command": p2rank_command,
                "recommended_fpocket_command": fpocket_command,
                "notes": str(target.get("notes") or ""),
            }
        )
        records.append(
            {
                "status": "prepared",
                "pdb_id": pdb_id,
                "source_status": source_status,
                "copy_status": copy_status,
                "methods": methods,
            }
        )
    return rows, records


def _build_script_header(title: str) -> list[str]:
    return [
        "# Generated by prepare_cd38_external_tool_inputs.py",
        f"# {title}",
        "$ErrorActionPreference = 'Stop'",
        "$PackageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path",
        "",
    ]


def _build_bash_header(title: str) -> list[str]:
    return [
        "#!/usr/bin/env bash",
        "# Generated by prepare_cd38_external_tool_inputs.py",
        f"# {title}",
        "set -euo pipefail",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        "",
    ]


def _write_executable_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")
    try:
        path.chmod(0o755)
    except OSError:
        pass


def _write_p2rank_script(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = _build_script_header("Run external P2Rank commands for CD38 target structures.")
    for row in rows:
        if not str(row.get("recommended_p2rank_command") or "").strip():
            continue
        pdb_id = str(row["pdb_id"])
        lines.extend(
            [
                f"# {pdb_id}",
                f'$PdbPath = Join-Path $PackageRoot "pdbs\\{pdb_id}.pdb"',
                f'$OutDir = Join-Path $PackageRoot "p2rank_outputs\\{pdb_id}"',
                "New-Item -ItemType Directory -Force -Path $OutDir | Out-Null",
                "prank predict -f $PdbPath -o $OutDir",
                "",
            ]
        )
    if len(lines) <= 4:
        lines.append("# No P2Rank target commands were generated.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_p2rank_bash_script(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = _build_bash_header("Run external P2Rank commands for CD38 target structures.")
    lines.extend(
        [
            'if ! command -v prank >/dev/null 2>&1; then',
            '  echo "Missing P2Rank executable: prank. Install P2Rank or add it to PATH." >&2',
            "  exit 127",
            "fi",
            "",
        ]
    )
    for row in rows:
        if not str(row.get("recommended_p2rank_command") or "").strip():
            continue
        pdb_id = str(row["pdb_id"])
        lines.extend(
            [
                f"# {pdb_id}",
                f'PDB_PATH="$SCRIPT_DIR/pdbs/{pdb_id}.pdb"',
                f'OUT_DIR="$SCRIPT_DIR/p2rank_outputs/{pdb_id}"',
                'mkdir -p "$OUT_DIR"',
                'prank predict -f "$PDB_PATH" -o "$OUT_DIR"',
                "",
            ]
        )
    if len(lines) <= 10:
        lines.append("# No P2Rank target commands were generated.")
    _write_executable_text(path, "\n".join(lines) + "\n")


def _write_fpocket_script(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = _build_script_header("Run external fpocket commands for CD38 target structures.")
    for row in rows:
        if not str(row.get("recommended_fpocket_command") or "").strip():
            continue
        pdb_id = str(row["pdb_id"])
        lines.extend(
            [
                f"# {pdb_id}",
                f'$RunDir = Join-Path $PackageRoot "fpocket_runs\\{pdb_id}"',
                "Push-Location $RunDir",
                f'fpocket -f "{pdb_id}.pdb"',
                "Pop-Location",
                "",
            ]
        )
    if len(lines) <= 4:
        lines.append("# No fpocket target commands were generated.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_fpocket_bash_script(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = _build_bash_header("Run external fpocket commands for CD38 target structures.")
    lines.extend(
        [
            'if ! command -v fpocket >/dev/null 2>&1; then',
            '  echo "Missing fpocket executable. Install fpocket or add it to PATH." >&2',
            "  exit 127",
            "fi",
            "",
        ]
    )
    for row in rows:
        if not str(row.get("recommended_fpocket_command") or "").strip():
            continue
        pdb_id = str(row["pdb_id"])
        lines.extend(
            [
                f"# {pdb_id}",
                f'RUN_DIR="$SCRIPT_DIR/fpocket_runs/{pdb_id}"',
                f'(cd "$RUN_DIR" && fpocket -f "{pdb_id}.pdb")',
                "",
            ]
        )
    if len(lines) <= 10:
        lines.append("# No fpocket target commands were generated.")
    _write_executable_text(path, "\n".join(lines) + "\n")


def _write_refresh_script(path: Path, args: argparse.Namespace) -> None:
    command = (
        "python refresh_cd38_benchmark_readiness.py "
        "--p2rank_root $P2RankRoot "
        "--fpocket_root $FpocketRoot "
        f"--chain_filter {args.chain_filter}"
    )
    if str(args.rank_by_pdb or "").strip():
        command += f' --rank_by_pdb "{args.rank_by_pdb}"'
    lines = _build_script_header("Refresh CD38 readiness after external tools finish.")
    lines.extend(
        [
            '$RepoRoot = Resolve-Path (Join-Path $PackageRoot "..\\..\\..")',
            '$P2RankRoot = Join-Path $PackageRoot "p2rank_outputs"',
            '$FpocketRoot = Join-Path $PackageRoot "fpocket_runs"',
            "Push-Location $RepoRoot",
            "try {",
            f"  {command}",
            "} finally {",
            "  Pop-Location",
            "}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_preflight_script(path: Path) -> None:
    lines = _build_script_header("Check local P2Rank/fpocket availability and expected CD38 external-tool outputs.")
    lines.extend(
        [
            '$RepoRoot = Resolve-Path (Join-Path $PackageRoot "..\\..\\..")',
            "Push-Location $RepoRoot",
            "try {",
            "  python check_cd38_external_tool_environment.py --package_dir $PackageRoot",
            "} finally {",
            "  Pop-Location",
            "}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_finalize_script(path: Path) -> None:
    lines = [
        "# Generated by prepare_cd38_external_tool_inputs.py",
        "# Finalize CD38 external P2Rank/fpocket benchmark import.",
        "param(",
        "  [string]$ImportSource = '',",
        "  [switch]$RunDiscovered,",
        "  [switch]$RunSensitivity,",
        "  [switch]$ImportOverwrite,",
        "  [switch]$ImportDryRun",
        ")",
        "",
        "$ErrorActionPreference = 'Stop'",
        "$PackageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path",
        "",
    ]
    lines.extend(
        [
            '$RepoRoot = Resolve-Path (Join-Path $PackageRoot "..\\..\\..")',
            "Push-Location $RepoRoot",
            "try {",
            "  $Command = @('finalize_cd38_external_benchmark.py', '--package_dir', $PackageRoot)",
            "  if ($ImportSource) { $Command += @('--import_source', $ImportSource) }",
            "  if ($RunDiscovered) { $Command += '--run_discovered' }",
            "  if ($RunSensitivity) { $Command += '--run_sensitivity' }",
            "  if ($ImportOverwrite) { $Command += '--import_overwrite' }",
            "  if ($ImportDryRun) { $Command += '--import_dry_run' }",
            "  python @Command",
            "} finally {",
            "  Pop-Location",
            "}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _build_expected_return_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expected_rows: list[dict[str, Any]] = []
    for row in rows:
        pdb_id = str(row.get("pdb_id") or "").strip().upper()
        methods = _normalize_methods(row.get("desired_methods"))
        if "p2rank" in methods:
            expected_rows.append(
                {
                    "pdb_id": pdb_id,
                    "method": "p2rank",
                    "expected_return_path": f"p2rank_outputs/{pdb_id}/{pdb_id}.pdb_predictions.csv",
                    "required_pattern": f"p2rank_outputs/{pdb_id}/*_predictions.csv",
                    "generated_by": "run_p2rank_templates.ps1 or run_p2rank_templates.sh",
                    "coverage_key": f"{pdb_id}:p2rank",
                    "notes": "Required for P2Rank benchmark import coverage.",
                }
            )
        if "fpocket" in methods:
            expected_rows.append(
                {
                    "pdb_id": pdb_id,
                    "method": "fpocket",
                    "expected_return_path": f"fpocket_runs/{pdb_id}/{pdb_id}_out/pockets/pocket*_atm.pdb",
                    "required_pattern": f"fpocket_runs/{pdb_id}/{pdb_id}_out/pockets/pocket*_atm.pdb",
                    "generated_by": "run_fpocket_templates.ps1 or run_fpocket_templates.sh",
                    "coverage_key": f"{pdb_id}:fpocket",
                    "notes": "At least one pocket*_atm.pdb is required for fpocket benchmark import coverage.",
                }
            )
    return expected_rows


def _build_return_checklist(expected_rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# CD38 External Tool Return Checklist",
        "",
        "Use this checklist after running P2Rank/fpocket on another machine. The returned zip or folder should contain the output paths below.",
        "",
        "## Expected Returned Outputs",
        "",
        "| PDB ID | Method | Expected return path | Generated by |",
        "| --- | --- | --- | --- |",
    ]
    for row in expected_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_escape(row.get("pdb_id", "")),
                    _md_escape(row.get("method", "")),
                    _md_escape(row.get("expected_return_path", "")),
                    _md_escape(row.get("generated_by", "")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Recommended Return Workflow",
            "",
            "1. Run the P2Rank/fpocket template scripts on the external machine.",
            "2. Confirm the expected paths above exist inside this package folder.",
            "3. Zip the whole package folder or copy back `p2rank_outputs/` and `fpocket_runs/*/*_out/`.",
            "4. In the original project, run:",
            "",
            "```bash",
            "python import_cd38_external_tool_outputs.py --source <returned_zip_or_dir> --dry_run",
            "```",
            "",
            "5. Check `cd38_external_tool_output_import_coverage.csv`; all expected `coverage_key` values should be ready before `--run_discovered`.",
            "",
            "## Coverage Keys",
            "",
        ]
    )
    for row in expected_rows:
        lines.append(f"- `{row['coverage_key']}` -> `{row['expected_return_path']}`")
    lines.extend(
        [
            "",
            "## Package Files",
            "",
            "- Expected return manifest: `cd38_external_tool_expected_returns.csv`",
            "- Input manifest: `cd38_external_tool_input_manifest.csv`",
            "",
        ]
    )
    return "\n".join(lines)


def _md_escape(value: Any) -> str:
    return str(value).replace("|", "\\|")


def _package_rel_path(value: Any, package_dir: Any) -> str:
    if not value:
        return ""
    try:
        path = Path(str(value)).resolve()
        base = Path(str(package_dir)).resolve()
        return path.relative_to(base).as_posix()
    except (OSError, ValueError):
        return str(value).replace("\\", "/")


def _build_report(rows: list[dict[str, Any]], records: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# CD38 External Tool Input Package",
        "",
        "This package contains PDB inputs and command templates for running external P2Rank/fpocket tools before importing their outputs into the CD38 benchmark.",
        "",
        "## Summary",
        "",
        f"- Prepared structures: `{summary['prepared_structure_count']}`",
        f"- P2Rank targets: `{summary['p2rank_target_count']}`",
        f"- fpocket targets: `{summary['fpocket_target_count']}`",
        "- Package directory: `external_tool_inputs/`",
        "",
        "## Files",
        "",
        "- Input manifest: `cd38_external_tool_input_manifest.csv`",
        "- Expected return manifest: `cd38_external_tool_expected_returns.csv`",
        "- Return checklist: `cd38_external_tool_return_checklist.md`",
        "- P2Rank PowerShell template: `run_p2rank_templates.ps1`",
        "- fpocket PowerShell template: `run_fpocket_templates.ps1`",
        "- P2Rank Bash template: `run_p2rank_templates.sh`",
        "- fpocket Bash template: `run_fpocket_templates.sh`",
        "- Environment/output preflight template: `check_external_tool_environment.ps1`",
        "- Readiness refresh template: `refresh_readiness_after_external_tools.ps1`",
        "- Finalize workflow template: `finalize_external_benchmark.ps1`",
        "",
        "## Recommended Flow",
        "",
        "1. Run `check_external_tool_environment.ps1` to confirm whether P2Rank/fpocket commands and expected outputs are available.",
        "   The preflight step resolves current package-local paths first, so the package can be moved to WSL/Linux or another machine without being trapped by old absolute paths in the manifest.",
        "2. Run or adapt `run_p2rank_templates.ps1` on a machine where P2Rank is installed.",
        "3. Run or adapt `run_fpocket_templates.ps1` on a machine where fpocket is installed.",
        "4. On Linux/WSL, use `run_p2rank_templates.sh` and `run_fpocket_templates.sh` instead.",
        "5. Before zipping/copying back, open `cd38_external_tool_return_checklist.md` and confirm the expected returned output paths exist.",
        "6. Copy the generated output folders back into this package if they were run elsewhere; if you received a whole returned directory or zip, use `finalize_cd38_external_benchmark.py --import_source <returned_zip_or_dir>`.",
        "7. Run `check_external_tool_environment.ps1` again to confirm expected outputs are now present.",
        "8. Run `finalize_external_benchmark.ps1` for a one-command preflight + readiness summary.",
        "9. If readiness reports look correct, rerun `finalize_cd38_external_benchmark.py --run_discovered` to import discovered rows; if the returned package is already confirmed, use `finalize_cd38_external_benchmark.py --import_source <returned_zip_or_dir> --run_discovered`.",
        "",
        "## Targets",
        "",
        "| PDB ID | Methods | PDB Input | Expected P2Rank CSV | Expected fpocket pockets dir |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_escape(row.get("pdb_id", "")),
                    _md_escape(row.get("desired_methods", "")),
                    _md_escape(_package_rel_path(row.get("tool_input_pdb_path", ""), summary.get("out_dir", ""))),
                    _md_escape(
                        _package_rel_path(row.get("expected_p2rank_predictions_csv", ""), summary.get("out_dir", ""))
                    ),
                    _md_escape(
                        _package_rel_path(row.get("expected_fpocket_pockets_dir", ""), summary.get("out_dir", ""))
                    ),
                ]
            )
            + " |"
        )
    skipped = [record for record in records if record.get("status") != "prepared"]
    if skipped:
        lines.extend(["", "## Skipped Targets", ""])
        for record in skipped:
            lines.append(f"- `{record.get('pdb_id', '')}`: {record.get('reason', '')}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = _build_parser().parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, records = _build_rows(args)
    input_manifest_csv = out_dir / "cd38_external_tool_input_manifest.csv"
    expected_return_manifest_csv = out_dir / "cd38_external_tool_expected_returns.csv"
    return_checklist_md = out_dir / "cd38_external_tool_return_checklist.md"
    run_p2rank_ps1 = out_dir / "run_p2rank_templates.ps1"
    run_fpocket_ps1 = out_dir / "run_fpocket_templates.ps1"
    run_p2rank_sh = out_dir / "run_p2rank_templates.sh"
    run_fpocket_sh = out_dir / "run_fpocket_templates.sh"
    check_environment_ps1 = out_dir / "check_external_tool_environment.ps1"
    refresh_readiness_ps1 = out_dir / "refresh_readiness_after_external_tools.ps1"
    finalize_benchmark_ps1 = out_dir / "finalize_external_benchmark.ps1"
    summary_json = out_dir / "cd38_external_tool_inputs_summary.json"
    report_md = out_dir / "cd38_external_tool_inputs.md"

    _write_csv(input_manifest_csv, rows, MANIFEST_COLUMNS)
    expected_return_rows = _build_expected_return_rows(rows)
    _write_csv(expected_return_manifest_csv, expected_return_rows, EXPECTED_RETURN_COLUMNS)
    _write_p2rank_script(run_p2rank_ps1, rows)
    _write_fpocket_script(run_fpocket_ps1, rows)
    _write_p2rank_bash_script(run_p2rank_sh, rows)
    _write_fpocket_bash_script(run_fpocket_sh, rows)
    _write_preflight_script(check_environment_ps1)
    _write_refresh_script(refresh_readiness_ps1, args)
    _write_finalize_script(finalize_benchmark_ps1)

    summary = {
        "structure_targets_csv": str(Path(args.structure_targets_csv).expanduser().resolve()),
        "inputs_dir": str(Path(args.inputs_dir).expanduser().resolve()),
        "out_dir": str(out_dir),
        "prepared_structure_count": int(len(rows)),
        "p2rank_target_count": int(sum(1 for row in rows if str(row.get("recommended_p2rank_command") or "").strip())),
        "fpocket_target_count": int(sum(1 for row in rows if str(row.get("recommended_fpocket_command") or "").strip())),
        "expected_return_count": int(len(expected_return_rows)),
        "records": records,
        "input_manifest_csv": str(input_manifest_csv),
        "expected_return_manifest_csv": str(expected_return_manifest_csv),
        "return_checklist_md": str(return_checklist_md),
        "run_p2rank_ps1": str(run_p2rank_ps1),
        "run_fpocket_ps1": str(run_fpocket_ps1),
        "run_p2rank_sh": str(run_p2rank_sh),
        "run_fpocket_sh": str(run_fpocket_sh),
        "check_environment_ps1": str(check_environment_ps1),
        "refresh_readiness_ps1": str(refresh_readiness_ps1),
        "finalize_benchmark_ps1": str(finalize_benchmark_ps1),
        "summary_json": str(summary_json),
        "report_md": str(report_md),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    return_checklist_md.write_text(_build_return_checklist(expected_return_rows, summary), encoding="utf-8")
    report_md.write_text(_build_report(rows, records, summary), encoding="utf-8")

    print(f"Saved: {input_manifest_csv}")
    print(f"Saved: {expected_return_manifest_csv}")
    print(f"Saved: {return_checklist_md}")
    print(f"Saved: {run_p2rank_ps1}")
    print(f"Saved: {run_fpocket_ps1}")
    print(f"Saved: {run_p2rank_sh}")
    print(f"Saved: {run_fpocket_sh}")
    print(f"Saved: {check_environment_ps1}")
    print(f"Saved: {refresh_readiness_ps1}")
    print(f"Saved: {finalize_benchmark_ps1}")
    print(f"Saved: {summary_json}")
    print(f"Saved: {report_md}")
    print(f"Prepared structures: {len(rows)}")


if __name__ == "__main__":
    main()
