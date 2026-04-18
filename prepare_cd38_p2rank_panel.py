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
        description="Discover P2Rank *_predictions.csv files and build a CD38 benchmark manifest."
    )
    parser.add_argument(
        "--p2rank_root",
        nargs="+",
        required=True,
        help="One or more P2Rank output roots. The script scans **/*_predictions.csv and **/*predictions.csv.",
    )
    parser.add_argument(
        "--manifest_out",
        default="benchmarks/cd38/p2rank_discovered_manifest.csv",
        help="Output manifest CSV for run_cd38_benchmark_manifest.py",
    )
    parser.add_argument("--results_root", default="benchmarks/cd38/results", help="Benchmark result root")
    parser.add_argument(
        "--truth_file",
        default="benchmarks/cd38/cd38_active_site_truth.txt",
        help="Ground-truth residue definition file used when --run is set",
    )
    parser.add_argument("--rcsb_pdb_id", default=None, help="Use one fixed PDB ID for all discovered prediction files")
    parser.add_argument("--chain_filter", default="A", help="Protein chain to keep when extracting residues")
    parser.add_argument("--near_threshold", type=float, default=4.5, help="Near-hit threshold for benchmark")
    parser.add_argument("--rank", type=int, default=None, help="Use one fixed P2Rank rank for all discovered files")
    parser.add_argument(
        "--rank_by_pdb",
        default=None,
        help="Comma-separated rank overrides, e.g. 3ROP=2,4OGW=1. Overrides --rank for matching PDB IDs.",
    )
    parser.add_argument("--top_n", type=int, default=1, help="Top-N pockets to merge when rank is not specified")
    parser.add_argument("--min_probability", type=float, default=None, help="Optional P2Rank probability cutoff")
    parser.add_argument(
        "--max_files_per_structure",
        type=int,
        default=None,
        help="Optional cap after sorting prediction files within each inferred PDB ID",
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
    parser.add_argument("--run", action="store_true", help="Run the generated P2Rank manifest immediately")
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


def _discover_predictions(roots: list[str]) -> list[Path]:
    seen: set[Path] = set()
    predictions: list[Path] = []
    patterns = ["**/*_predictions.csv", "**/*predictions.csv", "**/predictions.csv"]
    for raw_root in roots:
        root = Path(raw_root).expanduser().resolve()
        if not root.exists() or not root.is_dir():
            continue
        for pattern in patterns:
            for path in root.glob(pattern):
                resolved = path.resolve()
                if resolved.is_file() and resolved not in seen:
                    seen.add(resolved)
                    predictions.append(resolved)
    return sorted(predictions, key=lambda p: str(p).lower())


def _summarize_roots(roots: list[str]) -> list[dict[str, Any]]:
    patterns = ["**/*_predictions.csv", "**/*predictions.csv", "**/predictions.csv"]
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
                "prediction_file_count": int(len(matched)),
            }
        )
    return summaries


def _infer_pdb_id(path: Path, explicit: str | None) -> str:
    if explicit:
        return str(explicit).strip().upper()
    filename_patterns = [
        re.compile(r"^([0-9][0-9A-Za-z]{3})(?:\.pdb)?_predictions\.csv$", flags=re.IGNORECASE),
        re.compile(r"^([0-9][0-9A-Za-z]{3})[_\-.].*predictions\.csv$", flags=re.IGNORECASE),
    ]
    for pattern in filename_patterns:
        match = pattern.search(path.name)
        if match:
            return match.group(1).upper()
    part_pattern = re.compile(r"(?:^|[^A-Za-z0-9])([0-9][0-9A-Za-z]{3})(?:_p2rank|_out|[^A-Za-z0-9]|$)")
    for part in reversed(path.parts):
        match = part_pattern.search(part)
        if match:
            return match.group(1).upper()
    return ""


def _safe_name(text: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(text).strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "p2rank"


def _parse_rank_by_pdb(value: str | None) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for item in str(value or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid --rank_by_pdb item: {item!r}. Expected PDB=rank.")
        pdb_id, rank_text = item.split("=", 1)
        mapping[str(pdb_id).strip().upper()] = int(rank_text.strip())
    return mapping


def _inspect_predictions_csv(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, skipinitialspace=True)
            fieldnames = [str(col).strip() for col in (reader.fieldnames or [])]
            rows = [dict(row) for row in reader]
    except Exception as exc:
        return {
            "valid": False,
            "reason": "could_not_read_csv",
            "error": str(exc),
            "prediction_file": str(path),
        }

    normalized = {col.strip() for col in fieldnames}
    missing = sorted({"rank", "residue_ids"} - normalized)
    if missing:
        return {
            "valid": False,
            "reason": "missing_required_columns",
            "missing_columns": missing,
            "columns": fieldnames,
            "prediction_file": str(path),
        }

    ranks: list[int] = []
    names: list[str] = []
    for row in rows:
        rank_value = str(row.get("rank") or "").strip()
        if rank_value:
            try:
                ranks.append(int(float(rank_value)))
            except ValueError:
                pass
        name = str(row.get("name") or "").strip()
        if name:
            names.append(name)
    return {
        "valid": True,
        "reason": "",
        "prediction_file": str(path),
        "columns": fieldnames,
        "pocket_count": int(len(rows)),
        "ranks": sorted(set(ranks)),
        "names": sorted(set(names)),
        "has_probability": "probability" in normalized,
    }


def _make_result_name(
    *,
    pdb_id: str,
    predictions_csv: Path,
    rank: int | None,
    top_n: int,
    chain_filter: str,
    name_prefix: str | None,
    used: Counter[str],
) -> str:
    parts: list[str] = []
    if name_prefix:
        parts.append(_safe_name(name_prefix))
    parts.extend([_safe_name(pdb_id), "p2rank"])
    parts.append(f"rank{rank}" if rank is not None else f"top{max(1, int(top_n))}")
    if chain_filter:
        parts.append(f"chain{_safe_name(chain_filter)}")
    base = "_".join(part for part in parts if part)
    if not base:
        base = _safe_name(predictions_csv.stem)
    used[base] += 1
    return base if used[base] == 1 else f"{base}_{used[base]}"


def _build_rows(
    args: argparse.Namespace,
    repo_root: Path,
    predictions: list[Path],
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    rank_by_pdb = _parse_rank_by_pdb(args.rank_by_pdb)
    records: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for predictions_csv in predictions:
        pdb_id = _infer_pdb_id(predictions_csv, args.rcsb_pdb_id)
        inspect_payload = _inspect_predictions_csv(predictions_csv)
        if not pdb_id:
            records.append(
                {
                    "status": "skipped",
                    "reason": "could_not_infer_pdb_id",
                    "predictions_csv": str(predictions_csv),
                    "inspection": inspect_payload,
                }
            )
            continue
        if not bool(inspect_payload.get("valid")):
            records.append(
                {
                    "status": "skipped",
                    "reason": str(inspect_payload.get("reason") or "invalid_predictions_csv"),
                    "rcsb_pdb_id": pdb_id,
                    "predictions_csv": str(predictions_csv),
                    "inspection": inspect_payload,
                }
            )
            continue
        candidates.append(
            {
                "pdb_id": pdb_id,
                "predictions_csv": predictions_csv,
                "inspection": inspect_payload,
            }
        )

    candidates.sort(key=lambda item: (item["pdb_id"], str(item["predictions_csv"]).lower()))
    if args.max_files_per_structure is not None:
        capped: list[dict[str, Any]] = []
        for pdb_id in sorted({item["pdb_id"] for item in candidates}):
            subset = [item for item in candidates if item["pdb_id"] == pdb_id]
            capped.extend(subset[: max(0, int(args.max_files_per_structure))])
        candidates = capped

    used_names: Counter[str] = Counter()
    rows: list[dict[str, str]] = []
    for item in candidates:
        pdb_id = str(item["pdb_id"])
        predictions_csv = Path(item["predictions_csv"])
        selected_rank = rank_by_pdb.get(pdb_id, args.rank)
        result_name = _make_result_name(
            pdb_id=pdb_id,
            predictions_csv=predictions_csv,
            rank=selected_rank,
            top_n=int(args.top_n),
            chain_filter=str(args.chain_filter or "").strip(),
            name_prefix=args.name_prefix,
            used=used_names,
        )
        row = {col: "" for col in MANIFEST_COLUMNS}
        row.update(
            {
                "enabled": "true",
                "result_name": result_name,
                "method": "p2rank",
                "rcsb_pdb_id": pdb_id,
                "predictions_csv": _path_for_manifest(predictions_csv, repo_root),
                "near_threshold": str(float(args.near_threshold)),
                "rank": str(int(selected_rank)) if selected_rank is not None else "",
                "chain_filter": str(args.chain_filter or "").strip(),
                "top_n": "" if selected_rank is not None else str(max(1, int(args.top_n))),
                "min_probability": "" if args.min_probability is None else str(float(args.min_probability)),
            }
        )
        rows.append(row)
        records.append(
            {
                "status": "manifest_row",
                "result_name": result_name,
                "rcsb_pdb_id": pdb_id,
                "rank": selected_rank,
                "top_n": None if selected_rank is not None else max(1, int(args.top_n)),
                "predictions_csv": str(predictions_csv),
                "inspection": item["inspection"],
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
        formatted.append(f'"{text}"' if re.search(r"\s", text) else text)
    return " ".join(formatted)


def _build_recommended_command(
    *,
    args: argparse.Namespace,
    manifest_out: Path,
    summary_json: Path | None = None,
    report_md: Path,
    include_run: bool,
) -> list[str]:
    command = ["python", "prepare_cd38_p2rank_panel.py", "--p2rank_root"]
    command.extend(str(root) for root in args.p2rank_root)
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
    if args.rank is not None:
        command.extend(["--rank", str(int(args.rank))])
    if args.rank_by_pdb:
        command.extend(["--rank_by_pdb", str(args.rank_by_pdb)])
    command.extend(["--top_n", str(max(1, int(args.top_n)))])
    if args.min_probability is not None:
        command.extend(["--min_probability", str(float(args.min_probability))])
    if args.max_files_per_structure is not None:
        command.extend(["--max_files_per_structure", str(args.max_files_per_structure)])
    if args.name_prefix:
        command.extend(["--name_prefix", str(args.name_prefix)])
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
    if any(record.get("reason") == "could_not_infer_pdb_id" for record in records):
        return "Prediction files were found but PDB IDs could not be inferred; rerun with --rcsb_pdb_id or rename files to include the 4-character PDB ID."
    if any(record.get("reason") == "missing_required_columns" for record in records):
        return "Prediction files were found but do not look like P2Rank predictions.csv files; check for rank and residue_ids columns."
    return "No P2Rank prediction files were found; run P2Rank externally and rerun discovery."


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
    discovered_prediction_file_count: int,
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
        "# CD38 P2Rank Benchmark Readiness Report",
        "",
        "This report checks whether real `P2Rank` prediction CSV files are ready to be converted into the CD38 benchmark manifest.",
        "",
        "## Summary",
        "",
        f"- Manifest CSV: `{manifest_out}`",
        f"- Summary JSON: `{summary_json}`",
        f"- Report MD: `{report_md}`",
        f"- Discovered prediction files: `{discovered_prediction_file_count}`",
        f"- Runnable manifest rows: `{len(rows)}`",
        f"- Skipped files: `{len(skipped)}`",
        f"- Chain filter: `{str(args.chain_filter or '').strip()}`",
        f"- Near-hit threshold: `{float(args.near_threshold)}`",
        f"- Fixed rank: `{args.rank if args.rank is not None else 'not fixed'}`",
        f"- Rank overrides: `{args.rank_by_pdb or ''}`",
        f"- Top-N when rank is not fixed: `{max(1, int(args.top_n))}`",
        f"- Recommended next action: {_recommended_next_action(rows, records)}",
        "",
        "## Scanned Roots",
        "",
        "| Input | Resolved Path | Exists | Directory | prediction CSV files |",
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
                    str(int(item.get("prediction_file_count", 0))),
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
        lines.append("No runnable P2Rank manifest rows were generated.")

    lines.extend(["", "## Skipped Files", ""])
    if skipped:
        lines.extend(["| Reason | Prediction CSV | Details |", "| --- | --- | --- |"])
        for record in skipped[:50]:
            inspection = dict(record.get("inspection") or {})
            detail = inspection.get("missing_columns") or inspection.get("error") or ""
            lines.append(
                f"| {_md_escape(record.get('reason', 'unknown'))} | {_md_escape(record.get('predictions_csv', ''))} | {_md_escape(detail)} |"
            )
        if len(skipped) > 50:
            lines.append(f"| more | {len(skipped) - 50} additional skipped files omitted from this report |  |")
    else:
        lines.append("No discovered prediction files were skipped.")

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
            "Regenerate manifest, then run the P2Rank benchmark panel:",
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
            "- `Runnable manifest rows > 0` means the directory structure and CSV schema are usable for the current benchmark pipeline.",
            "- `Skipped files > 0` usually means the PDB ID was not present in the filename/path or the CSV lacks P2Rank columns.",
            "- This script does not run external `P2Rank`; it only imports existing `*_predictions.csv` files into the local benchmark workflow.",
            "- Use `--rank_by_pdb 3ROP=2,4OGW=1` when chain-specific active pockets are not P2Rank global rank 1.",
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
        "p2rank",
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

    root_statuses = _summarize_roots(list(args.p2rank_root))
    prediction_files = _discover_predictions(list(args.p2rank_root))
    rows, records = _build_rows(args, repo_root, prediction_files)
    _write_manifest(manifest_out, rows)

    run_payload: dict[str, Any] | None = None
    run_failed = False
    if bool(args.run) and rows:
        run_payload = _run_manifest(args, repo_root, manifest_out)
        run_failed = int(run_payload["returncode"]) != 0

    per_pdb_id = Counter(row["rcsb_pdb_id"] for row in rows)
    summary = {
        "p2rank_roots": [str(Path(root).expanduser().resolve()) for root in args.p2rank_root],
        "manifest_out": str(manifest_out),
        "report_md": str(report_md),
        "results_root": str(Path(args.results_root).expanduser().resolve()),
        "truth_file": str(Path(args.truth_file).expanduser().resolve()),
        "root_statuses": root_statuses,
        "discovered_prediction_file_count": int(len(prediction_files)),
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
        "rank": args.rank,
        "rank_by_pdb": _parse_rank_by_pdb(args.rank_by_pdb),
        "top_n": int(args.top_n),
        "min_probability": args.min_probability,
        "max_files_per_structure": args.max_files_per_structure,
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
            discovered_prediction_file_count=len(prediction_files),
        ),
        encoding="utf-8",
    )

    print(f"Saved: {manifest_out}")
    print(f"Saved: {summary_json}")
    print(f"Saved: {report_md}")
    print(f"Discovered prediction files: {len(prediction_files)}")
    print(f"Manifest rows: {len(rows)}")
    if not rows:
        print("No runnable P2Rank rows were created. Pass --rcsb_pdb_id if PDB IDs cannot be inferred from paths.")
    if run_failed and run_payload is not None:
        raise subprocess.CalledProcessError(run_payload["returncode"], run_payload["command"])


if __name__ == "__main__":
    main()
