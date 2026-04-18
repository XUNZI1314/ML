from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pocket_io import normalize_residue_key


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract residue IDs from one fpocket pocket atom PDB file.")
    parser.add_argument("--pocket_pdb", required=True, help="fpocket pocket atom PDB, e.g. pocket1_atm.pdb")
    parser.add_argument("--out_file", required=True, help="Output residue file")
    parser.add_argument("--summary_json", default=None, help="Optional extraction summary JSON")
    parser.add_argument("--chain_filter", default=None, help="Optional chain ID to keep")
    parser.add_argument(
        "--include_hetatm",
        action="store_true",
        help="Also parse HETATM records. By default only ATOM records are used.",
    )
    return parser


def _parse_pdb_residue_key(line: str) -> str:
    if len(line) < 27:
        raise ValueError(f"PDB atom line is too short: {line!r}")
    chain_id = line[21].strip() or "_"
    resseq_text = line[22:26].strip()
    if not resseq_text:
        raise ValueError(f"Missing residue number in PDB atom line: {line!r}")
    resseq = int(resseq_text)
    icode = line[26].strip()
    return normalize_residue_key(chain_id, resseq, icode)


def extract_fpocket_residue_keys(
    pocket_pdb: str | Path,
    *,
    chain_filter: str | None = None,
    include_hetatm: bool = False,
) -> tuple[list[str], dict[str, Any]]:
    path = Path(pocket_pdb).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"fpocket pocket PDB not found: {path}")

    chain = str(chain_filter).strip() if chain_filter else None
    keys: set[str] = set()
    atom_record_count = 0
    skipped_chain_count = 0
    skipped_record_count = 0

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        record = line[:6].strip().upper()
        if record != "ATOM" and not (include_hetatm and record == "HETATM"):
            skipped_record_count += 1
            continue
        atom_record_count += 1
        key = _parse_pdb_residue_key(line)
        if chain and key.split(":", 1)[0] != chain:
            skipped_chain_count += 1
            continue
        keys.add(key)

    residue_keys = sorted(
        keys,
        key=lambda x: (
            x.split(":")[0],
            int(x.split(":")[1]),
            x.split(":")[2] if x.count(":") >= 2 else "",
        ),
    )
    summary = {
        "pocket_pdb": str(path),
        "chain_filter": chain,
        "include_hetatm": bool(include_hetatm),
        "atom_record_count": int(atom_record_count),
        "skipped_chain_count": int(skipped_chain_count),
        "skipped_record_count": int(skipped_record_count),
        "residue_count": int(len(residue_keys)),
        "residue_keys": residue_keys,
    }
    return residue_keys, summary


def main() -> None:
    args = _build_parser().parse_args()
    residue_keys, summary = extract_fpocket_residue_keys(
        args.pocket_pdb,
        chain_filter=args.chain_filter,
        include_hetatm=bool(args.include_hetatm),
    )

    out_file = Path(args.out_file).expanduser().resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(residue_keys) + ("\n" if residue_keys else ""), encoding="utf-8")

    if args.summary_json:
        summary_path = Path(args.summary_json).expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"Saved: {summary_path}")
    print(f"Saved: {out_file}")
    print(f"Residues extracted: {len(residue_keys)}")


if __name__ == "__main__":
    main()
