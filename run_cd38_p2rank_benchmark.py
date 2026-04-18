from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the CD38 pocket benchmark directly from one P2Rank predictions.csv file."
    )
    parser.add_argument("--predictions_csv", required=True, help="Path to <protein>_predictions.csv from P2Rank")
    parser.add_argument("--out_dir", required=True, help="Output directory for extracted pocket and benchmark tables")
    parser.add_argument("--pdb_path", default=None, help="Optional local PDB path")
    parser.add_argument("--rcsb_pdb_id", default=None, help="Optional RCSB PDB ID, e.g. 3ROP")
    parser.add_argument(
        "--truth_file",
        default="benchmarks/cd38/cd38_active_site_truth.txt",
        help="Ground-truth residue definition file",
    )
    parser.add_argument("--rank", type=int, default=None, help="Specific P2Rank pocket rank to benchmark")
    parser.add_argument("--name", default=None, help="Specific P2Rank pocket name, e.g. pocket2")
    parser.add_argument("--chain_filter", default=None, help="Optional chain filter before benchmarking")
    parser.add_argument("--top_n", type=int, default=1, help="Merge top N pockets when rank/name is not provided")
    parser.add_argument("--min_probability", type=float, default=None, help="Optional minimum P2Rank probability")
    parser.add_argument("--near_threshold", type=float, default=4.5, help="Residue-neighbor threshold in Angstrom")
    return parser


def _run_command(command: list[str], workdir: Path) -> None:
    subprocess.run(command, cwd=str(workdir), check=True)


def _build_run_manifest(
    args: argparse.Namespace,
    out_dir: Path,
    pocket_file: Path,
    original_predictions_csv: Path,
    local_predictions_csv: Path,
) -> dict[str, Any]:
    return {
        "method": "p2rank",
        "predictions_csv": str(local_predictions_csv),
        "original_predictions_csv": str(original_predictions_csv),
        "out_dir": str(out_dir),
        "pdb_path": str(Path(args.pdb_path).expanduser().resolve()) if args.pdb_path else None,
        "rcsb_pdb_id": str(args.rcsb_pdb_id).strip().upper() if args.rcsb_pdb_id else None,
        "truth_file": str(Path(args.truth_file).expanduser().resolve()),
        "rank": int(args.rank) if args.rank is not None else None,
        "name": str(args.name).strip() if args.name else None,
        "chain_filter": str(args.chain_filter).strip() if args.chain_filter else None,
        "top_n": int(args.top_n),
        "min_probability": float(args.min_probability) if args.min_probability is not None else None,
        "near_threshold": float(args.near_threshold),
        "predicted_pocket_file": str(pocket_file),
    }


def main() -> None:
    args = _build_parser().parse_args()
    if not args.pdb_path and not args.rcsb_pdb_id:
        raise ValueError("Either --pdb_path or --rcsb_pdb_id must be provided.")

    repo_root = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    original_predictions_csv = Path(args.predictions_csv).expanduser().resolve()
    local_predictions_csv = out_dir / "source_predictions.csv"
    if original_predictions_csv != local_predictions_csv:
        shutil.copy2(original_predictions_csv, local_predictions_csv)

    pocket_file = out_dir / "predicted_pocket.txt"
    selection_summary = out_dir / "p2rank_selection_summary.json"

    extract_cmd = [
        sys.executable,
        str(repo_root / "extract_p2rank_pocket_residues.py"),
        "--predictions_csv",
        str(local_predictions_csv),
        "--out_file",
        str(pocket_file),
        "--summary_json",
        str(selection_summary),
        "--top_n",
        str(max(1, int(args.top_n))),
    ]
    if args.rank is not None:
        extract_cmd.extend(["--rank", str(int(args.rank))])
    if args.name:
        extract_cmd.extend(["--name", str(args.name).strip()])
    if args.chain_filter:
        extract_cmd.extend(["--chain_filter", str(args.chain_filter).strip()])
    if args.min_probability is not None:
        extract_cmd.extend(["--min_probability", str(float(args.min_probability))])
    _run_command(extract_cmd, repo_root)

    benchmark_cmd = [
        sys.executable,
        str(repo_root / "benchmark_cd38_pocket_accuracy.py"),
        "--truth_file",
        str(Path(args.truth_file).expanduser().resolve()),
        "--predicted_pocket_file",
        str(pocket_file),
        "--out_dir",
        str(out_dir),
        "--near_threshold",
        str(float(args.near_threshold)),
    ]
    if args.pdb_path:
        benchmark_cmd.extend(["--pdb_path", str(Path(args.pdb_path).expanduser().resolve())])
    else:
        benchmark_cmd.extend(["--rcsb_pdb_id", str(args.rcsb_pdb_id).strip().upper()])
    _run_command(benchmark_cmd, repo_root)

    manifest = _build_run_manifest(
        args,
        out_dir=out_dir,
        pocket_file=pocket_file,
        original_predictions_csv=original_predictions_csv,
        local_predictions_csv=local_predictions_csv,
    )
    manifest_path = out_dir / "run_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=True, indent=2)

    print(f"Saved benchmark outputs to: {out_dir}")


if __name__ == "__main__":
    main()
