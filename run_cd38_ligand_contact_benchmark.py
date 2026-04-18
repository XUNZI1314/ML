from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.request import urlopen


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the CD38 pocket benchmark directly from one ligand-contact baseline."
    )
    parser.add_argument("--pdb_path", default=None, help="Optional local PDB path")
    parser.add_argument("--rcsb_pdb_id", default=None, help="Optional RCSB PDB ID, e.g. 3ROP")
    parser.add_argument("--out_dir", required=True, help="Output directory for extracted pocket and benchmark tables")
    parser.add_argument(
        "--truth_file",
        default="benchmarks/cd38/cd38_active_site_truth.txt",
        help="Ground-truth residue definition file",
    )
    parser.add_argument("--protein_chain", required=True, help="Protein chain to benchmark, e.g. A")
    parser.add_argument("--ligand_chain", default=None, help="Ligand chain filter")
    parser.add_argument("--ligand_resnames", default=None, help="Comma-separated ligand residue names")
    parser.add_argument("--ligand_resseqs", default=None, help="Comma-separated ligand residue numbers")
    parser.add_argument("--distance_threshold", type=float, default=4.5, help="Ligand-contact threshold in Angstrom")
    parser.add_argument("--include_neighbor_shell", type=float, default=0.0, help="Optional extra shell")
    parser.add_argument("--near_threshold", type=float, default=4.5, help="Residue-neighbor threshold in Angstrom")
    return parser


def _download_rcsb_pdb(pdb_id: str, out_dir: Path) -> Path:
    pdb_code = str(pdb_id).strip().upper()
    if len(pdb_code) < 4:
        raise ValueError(f"Invalid PDB ID: {pdb_id!r}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pdb_code}.pdb"
    url = f"https://files.rcsb.org/download/{pdb_code}.pdb"
    with urlopen(url, timeout=30) as resp:
        payload = resp.read()
    out_path.write_bytes(payload)
    return out_path


def _run_command(command: list[str], workdir: Path) -> None:
    subprocess.run(command, cwd=str(workdir), check=True)


def _build_run_manifest(
    args: argparse.Namespace,
    out_dir: Path,
    pdb_path: Path,
    pocket_file: Path,
) -> dict[str, Any]:
    return {
        "method": "ligand_contact",
        "out_dir": str(out_dir),
        "pdb_path": str(pdb_path),
        "rcsb_pdb_id": str(args.rcsb_pdb_id).strip().upper() if args.rcsb_pdb_id else None,
        "truth_file": str(Path(args.truth_file).expanduser().resolve()),
        "protein_chain": str(args.protein_chain).strip(),
        "ligand_chain": str(args.ligand_chain).strip() if args.ligand_chain else None,
        "ligand_resnames": [x.strip().upper() for x in str(args.ligand_resnames or "").split(",") if x.strip()],
        "ligand_resseqs": [int(x.strip()) for x in str(args.ligand_resseqs or "").split(",") if x.strip()],
        "distance_threshold": float(args.distance_threshold),
        "include_neighbor_shell": float(args.include_neighbor_shell),
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

    inputs_dir = out_dir / "inputs"
    if args.pdb_path:
        source_pdb = Path(args.pdb_path).expanduser().resolve()
        if not source_pdb.exists():
            raise FileNotFoundError(f"PDB not found: {source_pdb}")
        pdb_path = inputs_dir / source_pdb.name
        inputs_dir.mkdir(parents=True, exist_ok=True)
        if source_pdb != pdb_path:
            shutil.copy2(source_pdb, pdb_path)
    else:
        pdb_path = _download_rcsb_pdb(str(args.rcsb_pdb_id).strip().upper(), inputs_dir)

    pocket_file = out_dir / "predicted_pocket.txt"
    selection_summary = out_dir / "ligand_contact_selection_summary.json"

    derive_cmd = [
        sys.executable,
        str(repo_root / "derive_ligand_contact_pocket.py"),
        "--pdb_path",
        str(pdb_path),
        "--out_file",
        str(pocket_file),
        "--summary_json",
        str(selection_summary),
        "--protein_chain",
        str(args.protein_chain).strip(),
        "--distance_threshold",
        str(float(args.distance_threshold)),
        "--include_neighbor_shell",
        str(float(args.include_neighbor_shell)),
    ]
    if args.ligand_chain:
        derive_cmd.extend(["--ligand_chain", str(args.ligand_chain).strip()])
    if args.ligand_resnames:
        derive_cmd.extend(["--ligand_resnames", str(args.ligand_resnames).strip()])
    if args.ligand_resseqs:
        derive_cmd.extend(["--ligand_resseqs", str(args.ligand_resseqs).strip()])
    _run_command(derive_cmd, repo_root)

    benchmark_cmd = [
        sys.executable,
        str(repo_root / "benchmark_cd38_pocket_accuracy.py"),
        "--pdb_path",
        str(pdb_path),
        "--truth_file",
        str(Path(args.truth_file).expanduser().resolve()),
        "--predicted_pocket_file",
        str(pocket_file),
        "--out_dir",
        str(out_dir),
        "--near_threshold",
        str(float(args.near_threshold)),
    ]
    _run_command(benchmark_cmd, repo_root)

    manifest = _build_run_manifest(args, out_dir=out_dir, pdb_path=pdb_path, pocket_file=pocket_file)
    manifest_path = out_dir / "run_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=True, indent=2)

    print(f"Saved benchmark outputs to: {out_dir}")


if __name__ == "__main__":
    main()
