from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure

from pdb_parser import load_complex_pdb
from pocket_io import normalize_residue_key


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Derive pocket residues from ligand-contact geometry in one structure.")
    parser.add_argument("--pdb_path", required=True, help="Path to one PDB structure")
    parser.add_argument("--out_file", required=True, help="Output residue file")
    parser.add_argument("--summary_json", default=None, help="Optional summary JSON output path")
    parser.add_argument("--protein_chain", default=None, help="Restrict protein residues to one chain")
    parser.add_argument("--ligand_chain", default=None, help="Restrict ligand residues to one chain")
    parser.add_argument("--ligand_resnames", default=None, help="Comma-separated ligand residue names, e.g. NMN or 50A,NCA")
    parser.add_argument("--ligand_resseqs", default=None, help="Comma-separated ligand residue numbers, e.g. 301,302")
    parser.add_argument("--distance_threshold", type=float, default=4.5, help="Ligand-contact threshold in Angstrom")
    parser.add_argument("--include_neighbor_shell", type=float, default=0.0, help="Optional extra shell above distance_threshold")
    return parser


def _iter_residues(entity: Any) -> list[Residue]:
    if isinstance(entity, Residue):
        return [entity]
    if isinstance(entity, Chain):
        return list(entity.get_residues())
    if isinstance(entity, Model):
        return list(entity.get_residues())
    if isinstance(entity, Structure):
        models = list(entity.get_models())
        if not models:
            return []
        return list(models[0].get_residues())
    if hasattr(entity, "get_residues"):
        return list(entity.get_residues())
    raise TypeError("Unsupported structure entity.")


def _coords_from_residue(residue: Residue) -> np.ndarray:
    coords: list[np.ndarray] = []
    for atom in residue.get_atoms():
        try:
            coord = np.asarray(atom.get_coord(), dtype=np.float64).reshape(3)
        except Exception:
            continue
        if np.isfinite(coord).all():
            coords.append(coord)
    if not coords:
        return np.empty((0, 3), dtype=np.float64)
    return np.vstack(coords).astype(np.float64, copy=False)


def _global_min_distance(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    if coords_a.shape[0] == 0 or coords_b.shape[0] == 0:
        return float("nan")
    diff = coords_a[:, None, :] - coords_b[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff, optimize=True)
    return float(np.sqrt(np.min(d2)))


def _residue_chain_id(residue: Residue) -> str:
    return str(residue.get_parent().id).strip() or "_"


def _canonical_key(residue: Residue) -> str:
    _, resseq, icode = residue.id
    return normalize_residue_key(_residue_chain_id(residue), int(resseq), str(icode).strip())


def _protein_like_residue(residue: Residue) -> bool:
    hetflag = str(residue.id[0]).strip()
    return hetflag == ""


def main() -> None:
    args = _build_parser().parse_args()
    pdb_path = Path(args.pdb_path).expanduser().resolve()
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB not found: {pdb_path}")

    structure = load_complex_pdb(str(pdb_path))
    residues = _iter_residues(structure)

    protein_chain = str(args.protein_chain).strip() if args.protein_chain else None
    ligand_chain = str(args.ligand_chain).strip() if args.ligand_chain else None
    ligand_resnames = {x.strip().upper() for x in str(args.ligand_resnames or "").split(",") if x.strip()}
    ligand_resseqs = {int(x.strip()) for x in str(args.ligand_resseqs or "").split(",") if x.strip()}
    threshold = max(float(args.distance_threshold), 0.0)
    shell = max(float(args.include_neighbor_shell), 0.0)

    ligand_residues: list[Residue] = []
    protein_residues: list[Residue] = []
    for residue in residues:
        chain_id = _residue_chain_id(residue)
        _, resseq, _ = residue.id
        resname = str(residue.get_resname()).strip().upper()
        if _protein_like_residue(residue):
            if protein_chain and chain_id != protein_chain:
                continue
            protein_residues.append(residue)
            continue

        if ligand_chain and chain_id != ligand_chain:
            continue
        if ligand_resnames and resname not in ligand_resnames:
            continue
        if ligand_resseqs and int(resseq) not in ligand_resseqs:
            continue
        ligand_residues.append(residue)

    if not ligand_residues:
        raise ValueError("No ligand residues matched the provided filters.")
    if not protein_residues:
        raise ValueError("No protein residues matched the provided protein-chain filter.")

    ligand_coords = [coords for coords in (_coords_from_residue(res) for res in ligand_residues) if coords.shape[0] > 0]
    if not ligand_coords:
        raise ValueError("Matched ligand residues do not contain valid coordinates.")
    ligand_coord_block = np.vstack(ligand_coords).astype(np.float64, copy=False)

    hit_rows: list[dict[str, Any]] = []
    residue_keys: list[str] = []
    for residue in protein_residues:
        coords = _coords_from_residue(residue)
        distance = _global_min_distance(coords, ligand_coord_block)
        if not np.isfinite(distance):
            continue
        if distance <= threshold + shell:
            key = _canonical_key(residue)
            _, resseq, icode = residue.id
            residue_keys.append(key)
            hit_rows.append(
                {
                    "residue_key": key,
                    "chain_id": _residue_chain_id(residue),
                    "resseq": int(resseq),
                    "icode": str(icode).strip(),
                    "resname": str(residue.get_resname()).strip(),
                    "min_distance_to_ligand": float(distance),
                    "within_core_threshold": bool(distance <= threshold),
                }
            )

    residue_keys = sorted(set(residue_keys), key=lambda x: (x.split(":")[0], int(x.split(":")[1]), x.split(":")[2] if x.count(":") >= 2 else ""))
    out_file = Path(args.out_file).expanduser().resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(residue_keys) + ("\n" if residue_keys else ""), encoding="utf-8")

    summary = {
        "pdb_path": str(pdb_path),
        "protein_chain": protein_chain,
        "ligand_chain": ligand_chain,
        "ligand_resnames": sorted(ligand_resnames),
        "ligand_resseqs": sorted(ligand_resseqs),
        "distance_threshold": threshold,
        "include_neighbor_shell": shell,
        "ligand_residue_count": int(len(ligand_residues)),
        "predicted_pocket_residue_count": int(len(residue_keys)),
        "predicted_residue_keys": residue_keys,
    }

    if args.summary_json:
        summary_path = Path(args.summary_json).expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=True, indent=2)

    print(f"Saved: {out_file}")
    if args.summary_json:
        print(f"Saved: {Path(args.summary_json).expanduser().resolve()}")
    print(f"Predicted pocket residues: {len(residue_keys)}")


if __name__ == "__main__":
    main()
