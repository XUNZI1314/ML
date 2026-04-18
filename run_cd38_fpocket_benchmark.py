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
        description="Run the CD38 pocket benchmark directly from one fpocket pocket atom PDB file."
    )
    parser.add_argument("--fpocket_pocket_pdb", required=True, help="fpocket pocket atom PDB, e.g. pocket1_atm.pdb")
    parser.add_argument("--out_dir", required=True, help="Output directory for extracted pocket and benchmark tables")
    parser.add_argument("--pdb_path", default=None, help="Optional local CD38 structure PDB path")
    parser.add_argument("--rcsb_pdb_id", default=None, help="Optional RCSB PDB ID, e.g. 3ROP")
    parser.add_argument(
        "--truth_file",
        default="benchmarks/cd38/cd38_active_site_truth.txt",
        help="Ground-truth residue definition file",
    )
    parser.add_argument("--chain_filter", default=None, help="Optional chain filter before benchmarking")
    parser.add_argument("--near_threshold", type=float, default=4.5, help="Residue-neighbor threshold in Angstrom")
    parser.add_argument("--include_hetatm", action="store_true", help="Also parse HETATM records from the fpocket PDB")
    return parser


def _run_command(command: list[str], workdir: Path) -> None:
    subprocess.run(command, cwd=str(workdir), check=True)


def _build_run_manifest(
    args: argparse.Namespace,
    out_dir: Path,
    pocket_file: Path,
    original_fpocket_pocket_pdb: Path,
    local_fpocket_pocket_pdb: Path,
) -> dict[str, Any]:
    return {
        "method": "fpocket",
        "fpocket_pocket_pdb": str(local_fpocket_pocket_pdb),
        "original_fpocket_pocket_pdb": str(original_fpocket_pocket_pdb),
        "out_dir": str(out_dir),
        "pdb_path": str(Path(args.pdb_path).expanduser().resolve()) if args.pdb_path else None,
        "rcsb_pdb_id": str(args.rcsb_pdb_id).strip().upper() if args.rcsb_pdb_id else None,
        "truth_file": str(Path(args.truth_file).expanduser().resolve()),
        "chain_filter": str(args.chain_filter).strip() if args.chain_filter else None,
        "near_threshold": float(args.near_threshold),
        "include_hetatm": bool(args.include_hetatm),
        "predicted_pocket_file": str(pocket_file),
    }


def main() -> None:
    args = _build_parser().parse_args()
    if not args.pdb_path and not args.rcsb_pdb_id:
        raise ValueError("Either --pdb_path or --rcsb_pdb_id must be provided.")

    repo_root = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    original_fpocket_pocket_pdb = Path(args.fpocket_pocket_pdb).expanduser().resolve()
    if not original_fpocket_pocket_pdb.exists():
        raise FileNotFoundError(f"fpocket pocket PDB not found: {original_fpocket_pocket_pdb}")
    local_fpocket_pocket_pdb = out_dir / "source_fpocket_pocket.pdb"
    if original_fpocket_pocket_pdb != local_fpocket_pocket_pdb:
        shutil.copy2(original_fpocket_pocket_pdb, local_fpocket_pocket_pdb)

    pocket_file = out_dir / "predicted_pocket.txt"
    extraction_summary = out_dir / "fpocket_selection_summary.json"

    extract_cmd = [
        sys.executable,
        str(repo_root / "extract_fpocket_pocket_residues.py"),
        "--pocket_pdb",
        str(local_fpocket_pocket_pdb),
        "--out_file",
        str(pocket_file),
        "--summary_json",
        str(extraction_summary),
    ]
    if args.chain_filter:
        extract_cmd.extend(["--chain_filter", str(args.chain_filter).strip()])
    if bool(args.include_hetatm):
        extract_cmd.append("--include_hetatm")
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
        original_fpocket_pocket_pdb=original_fpocket_pocket_pdb,
        local_fpocket_pocket_pdb=local_fpocket_pocket_pdb,
    )
    manifest_path = out_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")

    print(f"Saved benchmark outputs to: {out_dir}")


if __name__ == "__main__":
    main()
