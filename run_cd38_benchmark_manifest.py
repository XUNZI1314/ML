from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CD38 pocket benchmarks from a CSV manifest.")
    parser.add_argument(
        "--manifest_csv",
        default="benchmarks/cd38/cd38_benchmark_manifest.csv",
        help="CSV manifest describing benchmark jobs",
    )
    parser.add_argument("--results_root", default="benchmarks/cd38/results", help="Output root for result folders")
    parser.add_argument(
        "--truth_file",
        default="benchmarks/cd38/cd38_active_site_truth.txt",
        help="Ground-truth residue definition file",
    )
    parser.add_argument("--only", default=None, help="Comma-separated result_name values to run")
    parser.add_argument("--methods", default=None, help="Comma-separated methods to run, e.g. ligand_contact,p2rank")
    parser.add_argument("--force", action="store_true", help="Rerun rows even if cd38_pocket_accuracy_summary.json exists")
    parser.add_argument("--dry_run", action="store_true", help="Print and record commands without executing them")
    parser.add_argument("--continue_on_error", action="store_true", help="Continue remaining rows after a failed command")
    parser.add_argument("--skip_summary", action="store_true", help="Do not run summarize_cd38_benchmarks.py after jobs")
    parser.add_argument("--python_executable", default=None, help="Python executable used for child commands")
    parser.add_argument("--run_summary_json", default=None, help="Optional path for manifest run summary JSON")
    return parser


def _cell(row: pd.Series, key: str, default: str = "") -> str:
    value = row.get(key, default)
    if pd.isna(value):
        return default
    return str(value).strip()


def _truthy(value: Any, default: bool = True) -> bool:
    if value is None or pd.isna(value):
        return default
    text = str(value).strip().lower()
    if text in {"", "default"}:
        return default
    return text in {"1", "true", "yes", "y", "on"}


def _append_arg(command: list[str], flag: str, value: Any) -> None:
    if value is None or pd.isna(value):
        return
    text = str(value).strip()
    if not text:
        return
    command.extend([flag, text])


def _append_float_arg(command: list[str], flag: str, value: Any) -> None:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return
    command.extend([flag, str(float(value))])


def _append_int_arg(command: list[str], flag: str, value: Any) -> None:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return
    command.extend([flag, str(int(float(value)))])


def _resolve_path(value: Any, repo_root: Path) -> Path | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def _require_source(command: list[str], row: pd.Series, repo_root: Path) -> None:
    pdb_path = _resolve_path(row.get("pdb_path"), repo_root)
    rcsb_pdb_id = _cell(row, "rcsb_pdb_id")
    if pdb_path is not None:
        command.extend(["--pdb_path", str(pdb_path)])
    elif rcsb_pdb_id:
        command.extend(["--rcsb_pdb_id", rcsb_pdb_id.upper()])
    else:
        raise ValueError("Each benchmark row must provide pdb_path or rcsb_pdb_id.")


def _build_row_command(
    row: pd.Series,
    *,
    repo_root: Path,
    py: str,
    out_dir: Path,
    truth_file: Path,
) -> list[str]:
    method = _cell(row, "method").lower()
    if method == "ligand_contact":
        command = [
            py,
            str(repo_root / "run_cd38_ligand_contact_benchmark.py"),
            "--out_dir",
            str(out_dir),
            "--truth_file",
            str(truth_file),
        ]
        _require_source(command, row, repo_root)
        _append_arg(command, "--protein_chain", _cell(row, "protein_chain"))
        _append_arg(command, "--ligand_chain", _cell(row, "ligand_chain"))
        _append_arg(command, "--ligand_resnames", _cell(row, "ligand_resnames"))
        _append_arg(command, "--ligand_resseqs", _cell(row, "ligand_resseqs"))
        _append_float_arg(command, "--distance_threshold", row.get("distance_threshold"))
        _append_float_arg(command, "--include_neighbor_shell", row.get("include_neighbor_shell"))
        _append_float_arg(command, "--near_threshold", row.get("near_threshold"))
        return command

    if method == "p2rank":
        predictions_csv = _resolve_path(row.get("predictions_csv"), repo_root)
        if predictions_csv is None:
            raise ValueError("p2rank rows require predictions_csv.")
        command = [
            py,
            str(repo_root / "run_cd38_p2rank_benchmark.py"),
            "--predictions_csv",
            str(predictions_csv),
            "--out_dir",
            str(out_dir),
            "--truth_file",
            str(truth_file),
        ]
        _require_source(command, row, repo_root)
        _append_int_arg(command, "--rank", row.get("rank"))
        _append_arg(command, "--name", _cell(row, "name"))
        _append_arg(command, "--chain_filter", _cell(row, "chain_filter"))
        _append_int_arg(command, "--top_n", row.get("top_n"))
        _append_float_arg(command, "--min_probability", row.get("min_probability"))
        _append_float_arg(command, "--near_threshold", row.get("near_threshold"))
        return command

    if method == "fpocket":
        fpocket_pocket_pdb = _resolve_path(row.get("fpocket_pocket_pdb"), repo_root)
        if fpocket_pocket_pdb is None:
            raise ValueError("fpocket rows require fpocket_pocket_pdb.")
        command = [
            py,
            str(repo_root / "run_cd38_fpocket_benchmark.py"),
            "--fpocket_pocket_pdb",
            str(fpocket_pocket_pdb),
            "--out_dir",
            str(out_dir),
            "--truth_file",
            str(truth_file),
        ]
        _require_source(command, row, repo_root)
        _append_arg(command, "--chain_filter", _cell(row, "chain_filter"))
        _append_float_arg(command, "--near_threshold", row.get("near_threshold"))
        if _truthy(row.get("include_hetatm"), default=False):
            command.append("--include_hetatm")
        return command

    if method in {"residue_file", "manual"}:
        predicted_pocket_file = _resolve_path(row.get("predicted_pocket_file"), repo_root)
        if predicted_pocket_file is None:
            raise ValueError("residue_file rows require predicted_pocket_file.")
        command = [
            py,
            str(repo_root / "benchmark_cd38_pocket_accuracy.py"),
            "--truth_file",
            str(truth_file),
            "--predicted_pocket_file",
            str(predicted_pocket_file),
            "--out_dir",
            str(out_dir),
        ]
        _require_source(command, row, repo_root)
        _append_float_arg(command, "--near_threshold", row.get("near_threshold"))
        return command

    raise ValueError(f"Unsupported benchmark method: {method!r}")


def _run_command(command: list[str], cwd: Path, *, dry_run: bool) -> int:
    print(" ".join(command))
    if dry_run:
        return 0
    completed = subprocess.run(command, cwd=str(cwd), check=False)
    return int(completed.returncode)


def main() -> None:
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parent
    manifest_csv = Path(args.manifest_csv).expanduser().resolve()
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_csv}")

    results_root = Path(args.results_root).expanduser().resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    truth_file = Path(args.truth_file).expanduser().resolve()
    if not truth_file.exists():
        raise FileNotFoundError(f"Truth file not found: {truth_file}")

    only = {x.strip() for x in str(args.only or "").split(",") if x.strip()}
    methods = {x.strip().lower() for x in str(args.methods or "").split(",") if x.strip()}
    py = str(args.python_executable or sys.executable)
    df = pd.read_csv(manifest_csv, keep_default_na=False)
    records: list[dict[str, Any]] = []

    for idx, row in df.iterrows():
        result_name = _cell(row, "result_name")
        method = _cell(row, "method").lower()
        if not result_name:
            records.append({"row_index": int(idx), "status": "failed", "error": "missing result_name"})
            if not args.continue_on_error:
                raise ValueError(f"Row {idx} is missing result_name.")
            continue
        if not _truthy(row.get("enabled"), default=True):
            records.append({"row_index": int(idx), "result_name": result_name, "method": method, "status": "disabled"})
            continue
        if only and result_name not in only:
            records.append({"row_index": int(idx), "result_name": result_name, "method": method, "status": "filtered"})
            continue
        if methods and method not in methods:
            records.append({"row_index": int(idx), "result_name": result_name, "method": method, "status": "filtered"})
            continue

        out_dir = results_root / result_name
        summary_path = out_dir / "cd38_pocket_accuracy_summary.json"
        if summary_path.exists() and not bool(args.force):
            records.append(
                {
                    "row_index": int(idx),
                    "result_name": result_name,
                    "method": method,
                    "status": "skipped_existing",
                    "out_dir": str(out_dir),
                }
            )
            continue

        try:
            command = _build_row_command(row, repo_root=repo_root, py=py, out_dir=out_dir, truth_file=truth_file)
            returncode = _run_command(command, repo_root, dry_run=bool(args.dry_run))
            status = "dry_run" if bool(args.dry_run) else ("success" if returncode == 0 else "failed")
            records.append(
                {
                    "row_index": int(idx),
                    "result_name": result_name,
                    "method": method,
                    "status": status,
                    "returncode": int(returncode),
                    "out_dir": str(out_dir),
                    "command": command,
                }
            )
            if returncode != 0 and not bool(args.continue_on_error):
                raise subprocess.CalledProcessError(returncode, command)
        except Exception as exc:
            records.append(
                {
                    "row_index": int(idx),
                    "result_name": result_name,
                    "method": method,
                    "status": "failed",
                    "out_dir": str(out_dir),
                    "error": str(exc),
                }
            )
            if not bool(args.continue_on_error):
                raise

    if not bool(args.skip_summary) and not bool(args.dry_run):
        summarize_cmd = [
            py,
            str(repo_root / "summarize_cd38_benchmarks.py"),
            "--results_root",
            str(results_root),
        ]
        returncode = _run_command(summarize_cmd, repo_root, dry_run=False)
        records.append(
            {
                "row_index": None,
                "result_name": "summary",
                "method": "summarize_cd38_benchmarks",
                "status": "success" if returncode == 0 else "failed",
                "returncode": int(returncode),
                "command": summarize_cmd,
            }
        )
        if returncode != 0 and not bool(args.continue_on_error):
            raise subprocess.CalledProcessError(returncode, summarize_cmd)

    payload = {
        "manifest_csv": str(manifest_csv),
        "results_root": str(results_root),
        "truth_file": str(truth_file),
        "dry_run": bool(args.dry_run),
        "force": bool(args.force),
        "records": records,
        "status_counts": pd.Series([r.get("status") for r in records]).value_counts(dropna=False).to_dict(),
    }
    run_summary_json = (
        Path(args.run_summary_json).expanduser().resolve()
        if args.run_summary_json
        else results_root / "manifest_run_summary.json"
    )
    run_summary_json.parent.mkdir(parents=True, exist_ok=True)
    run_summary_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"Saved: {run_summary_json}")


if __name__ == "__main__":
    main()
