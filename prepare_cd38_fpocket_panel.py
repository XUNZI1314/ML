from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


MANIFEST_COLUMNS = [
    "enabled",
    "result_name",
    "method",
    "rcsb_pdb_id",
    "pdb_path",
    "predictions_csv",
    "fpocket_pocket_pdb",
    "predicted_pocket_file",
    "protein_chain",
    "ligand_chain",
    "ligand_resnames",
    "ligand_resseqs",
    "distance_threshold",
    "include_neighbor_shell",
    "near_threshold",
    "rank",
    "name",
    "chain_filter",
    "top_n",
    "min_probability",
    "include_hetatm",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Discover fpocket pocket*_atm.pdb files and build a CD38 benchmark manifest."
    )
    parser.add_argument(
        "--fpocket_root",
        nargs="+",
        required=True,
        help="One or more fpocket output roots. The script scans **/pockets/pocket*_atm.pdb and **/pocket*_atm.pdb.",
    )
    parser.add_argument(
        "--manifest_out",
        default="benchmarks/cd38/fpocket_discovered_manifest.csv",
        help="Output manifest CSV for run_cd38_benchmark_manifest.py",
    )
    parser.add_argument("--results_root", default="benchmarks/cd38/results", help="Benchmark result root")
    parser.add_argument(
        "--truth_file",
        default="benchmarks/cd38/cd38_active_site_truth.txt",
        help="Ground-truth residue definition file used when --run is set",
    )
    parser.add_argument("--rcsb_pdb_id", default=None, help="Use one fixed PDB ID for all discovered pockets")
    parser.add_argument("--chain_filter", default="A", help="Protein chain to keep when extracting residues")
    parser.add_argument("--near_threshold", type=float, default=4.5, help="Near-hit threshold for benchmark")
    parser.add_argument("--include_hetatm", action="store_true", help="Also parse HETATM records from fpocket PDBs")
    parser.add_argument(
        "--max_pockets_per_structure",
        type=int,
        default=None,
        help="Optional cap after sorting pockets by pocket number within each inferred PDB ID",
    )
    parser.add_argument("--name_prefix", default=None, help="Optional prefix for result_name values")
    parser.add_argument(
        "--summary_json",
        default=None,
        help="Optional discovery summary JSON. Defaults to <manifest_out>.summary.json",
    )
    parser.add_argument(
        "--report_md",
        default=None,
        help="Optional human-readable readiness report. Defaults to <manifest_out>.report.md",
    )
    parser.add_argument("--run", action="store_true", help="Run the generated fpocket manifest immediately")
    parser.add_argument("--force", action="store_true", help="Pass --force to run_cd38_benchmark_manifest.py")
    parser.add_argument("--dry_run", action="store_true", help="Pass --dry_run to run_cd38_benchmark_manifest.py")
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Pass --continue_on_error to run_cd38_benchmark_manifest.py",
    )
    parser.add_argument("--skip_summary", action="store_true", help="Pass --skip_summary to run_cd38_benchmark_manifest.py")
    parser.add_argument("--python_executable", default=None, help="Python executable for child commands")
    return parser


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _path_for_manifest(path: Path, repo_root: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(repo_root))
    except ValueError:
        return str(resolved)


def _discover_pockets(roots: list[str]) -> list[Path]:
    seen: set[Path] = set()
    pockets: list[Path] = []
    patterns = ["**/pockets/pocket*_atm.pdb", "**/pocket*_atm.pdb"]
    for raw_root in roots:
        root = Path(raw_root).expanduser().resolve()
        if not root.exists() or not root.is_dir():
            continue
        for pattern in patterns:
            for path in root.glob(pattern):
                resolved = path.resolve()
                if resolved.is_file() and resolved not in seen:
                    seen.add(resolved)
                    pockets.append(resolved)
    return sorted(pockets, key=lambda p: str(p).lower())


def _summarize_roots(roots: list[str]) -> list[dict[str, Any]]:
    patterns = ["**/pockets/pocket*_atm.pdb", "**/pocket*_atm.pdb"]
    summaries: list[dict[str, Any]] = []
    for raw_root in roots:
        root = Path(raw_root).expanduser().resolve()
        matched: set[Path] = set()
        if root.exists() and root.is_dir():
            for pattern in patterns:
                for path in root.glob(pattern):
                    resolved = path.resolve()
                    if resolved.is_file():
                        matched.add(resolved)
        summaries.append(
            {
                "input": str(raw_root),
                "resolved": str(root),
                "exists": bool(root.exists()),
                "is_dir": bool(root.is_dir()),
                "pocket_file_count": int(len(matched)),
            }
        )
    return summaries


def _infer_pdb_id(path: Path, explicit: str | None) -> str:
    if explicit:
        return str(explicit).strip().upper()
    pattern = re.compile(r"(?:^|[^A-Za-z0-9])([0-9][0-9A-Za-z]{3})(?:_out|_fpocket|[^A-Za-z0-9]|$)")
    for part in reversed(path.parts):
        match = pattern.search(part)
        if match:
            return match.group(1).upper()
    return ""


def _pocket_number(path: Path) -> int | None:
    match = re.search(r"pocket(\d+)_atm\.pdb$", path.name, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def _safe_name(text: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(text).strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "fpocket"


def _make_result_name(
    *,
    pdb_id: str,
    pocket_path: Path,
    chain_filter: str,
    name_prefix: str | None,
    used: Counter[str],
) -> str:
    pocket_no = _pocket_number(pocket_path)
    pocket_label = f"pocket{pocket_no}" if pocket_no is not None else _safe_name(pocket_path.stem)
    parts = []
    if name_prefix:
        parts.append(_safe_name(name_prefix))
    parts.extend([_safe_name(pdb_id), "fpocket", _safe_name(pocket_label)])
    if chain_filter:
        parts.append(f"chain{_safe_name(chain_filter)}")
    base = "_".join(part for part in parts if part)
    used[base] += 1
    return base if used[base] == 1 else f"{base}_{used[base]}"


def _build_rows(args: argparse.Namespace, repo_root: Path, pockets: list[Path]) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for pocket_path in pockets:
        pdb_id = _infer_pdb_id(pocket_path, args.rcsb_pdb_id)
        if not pdb_id:
            records.append(
                {
                    "status": "skipped",
                    "reason": "could_not_infer_pdb_id",
                    "fpocket_pocket_pdb": str(pocket_path),
                }
            )
            continue
        candidates.append(
            {
                "pdb_id": pdb_id,
                "pocket_number": _pocket_number(pocket_path),
                "fpocket_pocket_pdb": pocket_path,
            }
        )

    candidates.sort(key=lambda item: (item["pdb_id"], item["pocket_number"] or 10**9, str(item["fpocket_pocket_pdb"]).lower()))
    if args.max_pockets_per_structure is not None:
        capped: list[dict[str, Any]] = []
        for pdb_id in sorted({item["pdb_id"] for item in candidates}):
            subset = [item for item in candidates if item["pdb_id"] == pdb_id]
            capped.extend(subset[: max(0, int(args.max_pockets_per_structure))])
        candidates = capped

    used_names: Counter[str] = Counter()
    rows: list[dict[str, str]] = []
    for item in candidates:
        pdb_id = str(item["pdb_id"])
        pocket_path = Path(item["fpocket_pocket_pdb"])
        result_name = _make_result_name(
            pdb_id=pdb_id,
            pocket_path=pocket_path,
            chain_filter=str(args.chain_filter or "").strip(),
            name_prefix=args.name_prefix,
            used=used_names,
        )
        row = {col: "" for col in MANIFEST_COLUMNS}
        row.update(
            {
                "enabled": "true",
                "result_name": result_name,
                "method": "fpocket",
                "rcsb_pdb_id": pdb_id,
                "fpocket_pocket_pdb": _path_for_manifest(pocket_path, repo_root),
                "near_threshold": str(float(args.near_threshold)),
                "chain_filter": str(args.chain_filter or "").strip(),
                "include_hetatm": "true" if bool(args.include_hetatm) else "false",
            }
        )
        rows.append(row)
        records.append(
            {
                "status": "manifest_row",
                "result_name": result_name,
                "rcsb_pdb_id": pdb_id,
                "pocket_number": item["pocket_number"],
                "fpocket_pocket_pdb": str(pocket_path),
            }
        )
    return rows, records


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _md_escape(value: Any) -> str:
    return str(value).replace("|", "\\|")


def _format_command(command: list[str]) -> str:
    formatted: list[str] = []
    for item in command:
        text = str(item)
        if re.search(r"\s", text):
            formatted.append(f'"{text}"')
        else:
            formatted.append(text)
    return " ".join(formatted)


def _build_recommended_command(
    *,
    args: argparse.Namespace,
    manifest_out: Path,
    summary_json: Path | None = None,
    report_md: Path,
    include_run: bool,
) -> list[str]:
    command = ["python", "prepare_cd38_fpocket_panel.py"]
    command.append("--fpocket_root")
    command.extend(str(root) for root in args.fpocket_root)
    command.extend(["--manifest_out", str(manifest_out)])
    if summary_json is not None:
        command.extend(["--summary_json", str(summary_json)])
    command.extend(["--report_md", str(report_md)])
    command.extend(["--results_root", str(args.results_root)])
    command.extend(["--truth_file", str(args.truth_file)])
    if args.rcsb_pdb_id:
        command.extend(["--rcsb_pdb_id", str(args.rcsb_pdb_id)])
    if args.chain_filter:
        command.extend(["--chain_filter", str(args.chain_filter)])
    command.extend(["--near_threshold", str(float(args.near_threshold))])
    if args.name_prefix:
        command.extend(["--name_prefix", str(args.name_prefix)])
    if bool(args.include_hetatm):
        command.append("--include_hetatm")
    if args.max_pockets_per_structure is not None:
        command.extend(["--max_pockets_per_structure", str(args.max_pockets_per_structure)])
    if bool(args.force):
        command.append("--force")
    if bool(args.dry_run):
        command.append("--dry_run")
    if bool(args.continue_on_error):
        command.append("--continue_on_error")
    if bool(args.skip_summary):
        command.append("--skip_summary")
    if args.python_executable:
        command.extend(["--python_executable", str(args.python_executable)])
    if include_run:
        command.append("--run")
    return command


def _recommended_next_action(rows: list[dict[str, str]], records: list[dict[str, Any]]) -> str:
    if rows:
        return "Manifest rows were created; run the generated manifest or rerun this script with --run."
    skipped_missing_pdb = [record for record in records if record.get("reason") == "could_not_infer_pdb_id"]
    if skipped_missing_pdb:
        return "Pocket files were found but PDB IDs could not be inferred; rerun with --rcsb_pdb_id or rename fpocket output folders to include the 4-character PDB ID."
    return "No fpocket pocket*_atm.pdb files were found; place real fpocket outputs under the scanned root and rerun discovery."


def _build_readiness_report(
    *,
    args: argparse.Namespace,
    manifest_out: Path,
    summary_json: Path,
    report_md: Path,
    root_statuses: list[dict[str, Any]],
    rows: list[dict[str, str]],
    records: list[dict[str, Any]],
    per_pdb_id: Counter[str],
    discovered_pocket_file_count: int,
) -> str:
    skipped = [record for record in records if record.get("status") == "skipped"]
    command_without_run = _build_recommended_command(
        args=args,
        manifest_out=manifest_out,
        summary_json=summary_json,
        report_md=report_md,
        include_run=False,
    )
    command_with_run = _build_recommended_command(
        args=args,
        manifest_out=manifest_out,
        summary_json=summary_json,
        report_md=report_md,
        include_run=True,
    )

    lines = [
        "# CD38 fpocket Benchmark Readiness Report",
        "",
        "This report checks whether real `fpocket` output folders are ready to be converted into the CD38 benchmark manifest.",
        "",
        "## Summary",
        "",
        f"- Manifest CSV: `{manifest_out}`",
        f"- Summary JSON: `{summary_json}`",
        f"- Report MD: `{report_md}`",
        f"- Discovered pocket files: `{discovered_pocket_file_count}`",
        f"- Runnable manifest rows: `{len(rows)}`",
        f"- Skipped files: `{len(skipped)}`",
        f"- Chain filter: `{str(args.chain_filter or '').strip()}`",
        f"- Near-hit threshold: `{float(args.near_threshold)}`",
        f"- Include HETATM: `{bool(args.include_hetatm)}`",
        f"- Max pockets per structure: `{args.max_pockets_per_structure if args.max_pockets_per_structure is not None else 'not capped'}`",
        f"- Recommended next action: {_recommended_next_action(rows, records)}",
        "",
        "## Scanned Roots",
        "",
        "| Input | Resolved Path | Exists | Directory | pocket*_atm.pdb files |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for item in root_statuses:
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_escape(item.get("input", "")),
                    _md_escape(item.get("resolved", "")),
                    str(bool(item.get("exists"))),
                    str(bool(item.get("is_dir"))),
                    str(int(item.get("pocket_file_count", 0))),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Manifest Rows By PDB ID", ""])
    if per_pdb_id:
        lines.extend(["| PDB ID | Manifest rows |", "| --- | ---: |"])
        for pdb_id, count in sorted(per_pdb_id.items()):
            lines.append(f"| {_md_escape(pdb_id)} | {int(count)} |")
    else:
        lines.append("No runnable fpocket manifest rows were generated.")

    lines.extend(["", "## Skipped Files", ""])
    if skipped:
        lines.extend(["| Reason | fpocket pocket PDB |", "| --- | --- |"])
        for record in skipped[:50]:
            lines.append(
                f"| {_md_escape(record.get('reason', 'unknown'))} | {_md_escape(record.get('fpocket_pocket_pdb', ''))} |"
            )
        if len(skipped) > 50:
            lines.append(f"| more | {len(skipped) - 50} additional skipped files omitted from this report |")
    else:
        lines.append("No discovered pocket files were skipped.")

    lines.extend(
        [
            "",
            "## Commands",
            "",
            "Regenerate manifest and report only:",
            "",
            "```powershell",
            _format_command(command_without_run),
            "```",
            "",
            "Regenerate manifest, then run the fpocket benchmark panel:",
            "",
            "```powershell",
            _format_command(command_with_run),
            "```",
            "",
            "After benchmark results exist, update the panel summary and parameter sensitivity tables:",
            "",
            "```powershell",
            "python summarize_cd38_benchmarks.py --results_root benchmarks\\cd38\\results",
            _format_command(
                [
                    "python",
                    "analyze_cd38_pocket_parameter_sensitivity.py",
                    "--manifest_csv",
                    str(manifest_out),
                    "--results_root",
                    "benchmarks\\cd38\\results",
                    "--truth_file",
                    "benchmarks\\cd38\\cd38_active_site_truth.txt",
                    "--out_dir",
                    "benchmarks\\cd38\\parameter_sensitivity",
                ]
            ),
            "```",
            "",
            "## Interpretation",
            "",
            "- `Runnable manifest rows > 0` means the directory structure is usable for the current benchmark pipeline.",
            "- `Skipped files > 0` usually means the PDB ID was not present in the fpocket output path; use `--rcsb_pdb_id` for single-structure batches.",
            "- This script does not run external `fpocket`; it only imports existing `pocket*_atm.pdb` files into the local benchmark workflow.",
            "",
        ]
    )
    return "\n".join(lines)


def _run_manifest(args: argparse.Namespace, repo_root: Path, manifest_out: Path) -> dict[str, Any]:
    py = str(args.python_executable or sys.executable)
    command = [
        py,
        str(repo_root / "run_cd38_benchmark_manifest.py"),
        "--manifest_csv",
        str(manifest_out),
        "--results_root",
        str(Path(args.results_root).expanduser().resolve()),
        "--truth_file",
        str(Path(args.truth_file).expanduser().resolve()),
        "--methods",
        "fpocket",
    ]
    if bool(args.force):
        command.append("--force")
    if bool(args.dry_run):
        command.append("--dry_run")
    if bool(args.continue_on_error):
        command.append("--continue_on_error")
    if bool(args.skip_summary):
        command.append("--skip_summary")
    completed = subprocess.run(command, cwd=str(repo_root), check=False)
    return {"command": command, "returncode": int(completed.returncode)}


def main() -> None:
    args = _build_parser().parse_args()
    repo_root = _repo_root()
    manifest_out = Path(args.manifest_out).expanduser().resolve()
    summary_json = (
        Path(args.summary_json).expanduser().resolve()
        if args.summary_json
        else manifest_out.with_suffix(manifest_out.suffix + ".summary.json")
    )
    report_md = (
        Path(args.report_md).expanduser().resolve()
        if args.report_md
        else manifest_out.with_suffix(manifest_out.suffix + ".report.md")
    )

    root_statuses = _summarize_roots(list(args.fpocket_root))
    pockets = _discover_pockets(list(args.fpocket_root))
    rows, records = _build_rows(args, repo_root, pockets)
    _write_manifest(manifest_out, rows)

    run_payload: dict[str, Any] | None = None
    run_failed = False
    if bool(args.run) and rows:
        run_payload = _run_manifest(args, repo_root, manifest_out)
        run_failed = int(run_payload["returncode"]) != 0

    per_pdb_id = Counter(row["rcsb_pdb_id"] for row in rows)
    summary = {
        "fpocket_roots": [str(Path(root).expanduser().resolve()) for root in args.fpocket_root],
        "manifest_out": str(manifest_out),
        "report_md": str(report_md),
        "results_root": str(Path(args.results_root).expanduser().resolve()),
        "truth_file": str(Path(args.truth_file).expanduser().resolve()),
        "root_statuses": root_statuses,
        "discovered_pocket_file_count": int(len(pockets)),
        "manifest_row_count": int(len(rows)),
        "skipped_count": int(sum(1 for record in records if record.get("status") == "skipped")),
        "per_pdb_id_counts": dict(sorted(per_pdb_id.items())),
        "ready_for_manifest_run": bool(rows),
        "recommended_next_action": _recommended_next_action(rows, records),
        "recommended_manifest_command": _build_recommended_command(
            args=args,
            manifest_out=manifest_out,
            summary_json=summary_json,
            report_md=report_md,
            include_run=False,
        ),
        "recommended_run_command": _build_recommended_command(
            args=args,
            manifest_out=manifest_out,
            summary_json=summary_json,
            report_md=report_md,
            include_run=True,
        ),
        "chain_filter": str(args.chain_filter or "").strip(),
        "near_threshold": float(args.near_threshold),
        "include_hetatm": bool(args.include_hetatm),
        "max_pockets_per_structure": args.max_pockets_per_structure,
        "run_requested": bool(args.run),
        "run": run_payload,
        "records": records,
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(
        _build_readiness_report(
            args=args,
            manifest_out=manifest_out,
            summary_json=summary_json,
            report_md=report_md,
            root_statuses=root_statuses,
            rows=rows,
            records=records,
            per_pdb_id=per_pdb_id,
            discovered_pocket_file_count=len(pockets),
        ),
        encoding="utf-8",
    )

    print(f"Saved: {manifest_out}")
    print(f"Saved: {summary_json}")
    print(f"Saved: {report_md}")
    print(f"Discovered pocket files: {len(pockets)}")
    print(f"Manifest rows: {len(rows)}")
    if not rows:
        print("No runnable fpocket rows were created. Pass --rcsb_pdb_id if PDB IDs cannot be inferred from paths.")
    if run_failed and run_payload is not None:
        raise subprocess.CalledProcessError(run_payload["returncode"], run_payload["command"])


if __name__ == "__main__":
    main()
