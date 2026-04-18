from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import numpy as np

from pocket_io import load_residue_set, normalize_residue_key


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


CANDIDATE_COLUMNS = [
    "pdb_id",
    "chain_id",
    "resname",
    "resseq",
    "icode",
    "residue_label",
    "category",
    "atom_count",
    "element_count",
    "min_distance_to_truth",
    "truth_contacts_within_threshold",
    "protein_contacts_within_threshold",
    "recommended_for_ligand_contact",
    "recommendation_reason",
]


COMMON_SOLVENTS = {
    "HOH",
    "WAT",
    "DOD",
    "SOL",
    "EDO",
    "GOL",
    "PEG",
    "MPD",
    "DMS",
    "SO4",
    "PO4",
    "ACT",
    "ACE",
}


COMMON_IONS = {
    "AL",
    "BA",
    "BR",
    "CA",
    "CD",
    "CL",
    "CO",
    "CS",
    "CU",
    "FE",
    "IOD",
    "K",
    "LI",
    "MG",
    "MN",
    "NA",
    "NI",
    "RB",
    "SR",
    "ZN",
}


COMMON_GLYCANS = {
    "NAG",
    "BMA",
    "MAN",
    "FUC",
    "GAL",
    "GLC",
    "SIA",
    "NDG",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect CD38 PDB HETATM records and suggest ligand metadata for ligand-contact benchmarks."
    )
    parser.add_argument(
        "--structure_targets_csv",
        default="benchmarks/cd38/cd38_structure_targets.csv",
        help="CD38 target structures CSV.",
    )
    parser.add_argument(
        "--truth_file",
        default="benchmarks/cd38/cd38_active_site_truth.txt",
        help="Truth residue definition file used to score active-site proximity.",
    )
    parser.add_argument("--inputs_dir", default="benchmarks/cd38/inputs", help="Where downloaded PDB files are cached.")
    parser.add_argument("--out_dir", default="benchmarks/cd38/ligand_candidates", help="Output directory.")
    parser.add_argument("--distance_threshold", type=float, default=4.5, help="Truth/contact threshold in Angstrom.")
    parser.add_argument("--force_download", action="store_true", help="Redownload PDB files even if cached locally.")
    parser.add_argument(
        "--suggested_targets_csv",
        default=None,
        help="Optional suggested target CSV. Defaults to <out_dir>/cd38_structure_targets_suggested.csv.",
    )
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


def _download_pdb(pdb_id: str, inputs_dir: Path, *, force_download: bool) -> Path:
    code = str(pdb_id).strip().upper()
    if len(code) != 4 or not code[0].isdigit():
        raise ValueError(f"Invalid PDB ID: {pdb_id!r}")
    inputs_dir.mkdir(parents=True, exist_ok=True)
    out_path = inputs_dir / f"{code}.pdb"
    if out_path.exists() and not force_download:
        return out_path
    url = f"https://files.rcsb.org/download/{code}.pdb"
    with urlopen(url, timeout=30) as resp:
        out_path.write_bytes(resp.read())
    return out_path


def _parse_pdb_atoms(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    protein_atoms: list[dict[str, Any]] = []
    het_atoms: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        record = line[:6].strip()
        if record not in {"ATOM", "HETATM"}:
            continue
        try:
            coord = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=np.float64)
        except ValueError:
            continue
        if not np.isfinite(coord).all():
            continue
        chain_id = line[21].strip() or "_"
        resname = line[17:20].strip().upper()
        resseq_text = line[22:26].strip()
        if not resseq_text:
            continue
        try:
            resseq = int(resseq_text)
        except ValueError:
            continue
        atom = {
            "record": record,
            "atom_name": line[12:16].strip(),
            "element": (line[76:78].strip() or line[12:16].strip()[:1]).upper(),
            "chain_id": chain_id,
            "resname": resname,
            "resseq": resseq,
            "icode": line[26].strip(),
            "coord": coord,
        }
        if record == "ATOM":
            atom["residue_key"] = normalize_residue_key(chain_id, resseq, atom["icode"])
            protein_atoms.append(atom)
        else:
            het_atoms.append(atom)
    return protein_atoms, het_atoms


def _category_for_residue(resname: str, atom_count: int) -> str:
    name = str(resname or "").strip().upper()
    if name in COMMON_SOLVENTS:
        return "solvent_or_buffer"
    if name in COMMON_IONS or atom_count <= 1:
        return "ion_or_single_atom"
    if name in COMMON_GLYCANS:
        return "glycan"
    return "ligand_like"


def _min_distance(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    if coords_a.size == 0 or coords_b.size == 0:
        return float("nan")
    diff = coords_a[:, None, :] - coords_b[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff, optimize=True)
    return float(np.sqrt(np.min(d2)))


def _contact_count(ligand_coords: np.ndarray, residue_coords: dict[str, list[np.ndarray]], threshold: float) -> int:
    count = 0
    for coords in residue_coords.values():
        block = np.vstack(coords).astype(np.float64, copy=False)
        distance = _min_distance(ligand_coords, block)
        if math.isfinite(distance) and distance <= threshold:
            count += 1
    return count


def _inspect_one(
    *,
    pdb_id: str,
    pdb_path: Path,
    truth_keys: set[str],
    protein_chain: str,
    threshold: float,
) -> list[dict[str, Any]]:
    protein_atoms, het_atoms = _parse_pdb_atoms(pdb_path)
    target_protein_atoms = [
        atom
        for atom in protein_atoms
        if not protein_chain or str(atom["chain_id"]).strip() == protein_chain
    ]
    truth_coords: dict[str, list[np.ndarray]] = defaultdict(list)
    protein_coords: dict[str, list[np.ndarray]] = defaultdict(list)
    for atom in target_protein_atoms:
        residue_key = str(atom["residue_key"])
        protein_coords[residue_key].append(atom["coord"])
        if residue_key in truth_keys:
            truth_coords[residue_key].append(atom["coord"])

    grouped: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for atom in het_atoms:
        if protein_chain and str(atom["chain_id"]).strip() != protein_chain:
            continue
        grouped[(str(atom["chain_id"]), str(atom["resname"]), int(atom["resseq"]), str(atom["icode"]))].append(atom)

    rows: list[dict[str, Any]] = []
    truth_block = np.vstack([coord for coords in truth_coords.values() for coord in coords]).astype(np.float64, copy=False) if truth_coords else np.empty((0, 3), dtype=np.float64)
    for (chain_id, resname, resseq, icode), atoms in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][2], item[0][1], item[0][3])):
        ligand_coords = np.vstack([atom["coord"] for atom in atoms]).astype(np.float64, copy=False)
        elements = sorted({str(atom["element"]) for atom in atoms if str(atom["element"]).strip()})
        category = _category_for_residue(resname, len(atoms))
        min_truth_distance = _min_distance(ligand_coords, truth_block)
        truth_contacts = _contact_count(ligand_coords, truth_coords, threshold)
        protein_contacts = _contact_count(ligand_coords, protein_coords, threshold)
        recommended = bool(
            category == "ligand_like"
            and len(atoms) >= 3
            and truth_contacts > 0
            and math.isfinite(min_truth_distance)
        )
        if recommended:
            reason = "ligand-like residue contacts known CD38 truth residues"
        elif category != "ligand_like":
            reason = f"excluded as {category}"
        elif not truth_contacts:
            reason = "ligand-like residue does not contact known CD38 truth residues"
        else:
            reason = "not enough ligand atoms for a robust contact baseline"
        rows.append(
            {
                "pdb_id": pdb_id,
                "chain_id": chain_id,
                "resname": resname,
                "resseq": int(resseq),
                "icode": icode,
                "residue_label": f"{chain_id}:{resname}:{resseq}{icode}",
                "category": category,
                "atom_count": int(len(atoms)),
                "element_count": int(len(elements)),
                "min_distance_to_truth": min_truth_distance,
                "truth_contacts_within_threshold": int(truth_contacts),
                "protein_contacts_within_threshold": int(protein_contacts),
                "recommended_for_ligand_contact": recommended,
                "recommendation_reason": reason,
            }
        )
    rows.sort(
        key=lambda row: (
            not bool(row["recommended_for_ligand_contact"]),
            float(row["min_distance_to_truth"]) if math.isfinite(float(row["min_distance_to_truth"])) else 10**9,
            -int(row["truth_contacts_within_threshold"]),
            str(row["residue_label"]),
        )
    )
    return rows


def _format_float(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return f"{number:.3f}"


def _suggest_target_rows(targets: list[dict[str, str]], candidate_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    recommended_by_pdb: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        if bool(row.get("recommended_for_ligand_contact")):
            recommended_by_pdb[str(row["pdb_id"]).upper()].append(row)

    suggested: list[dict[str, str]] = []
    for target in targets:
        row = {col: str(target.get(col) or "") for col in TARGET_COLUMNS}
        pdb_id = str(row.get("pdb_id") or "").strip().upper()
        methods = _normalize_methods(row.get("desired_methods"))
        recommended = recommended_by_pdb.get(pdb_id, [])
        if "ligand_contact" in methods and not recommended:
            methods = [method for method in methods if method != "ligand_contact"]
            note = row.get("notes", "").strip()
            suffix = "No active-site ligand candidate was detected; ligand_contact removed from suggested methods."
            row["notes"] = f"{note} {suffix}".strip()
        elif recommended and not row.get("ligand_resnames"):
            names = []
            chains = []
            resseqs = []
            for item in recommended:
                if str(item["resname"]) not in names:
                    names.append(str(item["resname"]))
                if str(item["chain_id"]) not in chains:
                    chains.append(str(item["chain_id"]))
                resseqs.append(str(item["resseq"]))
            row["ligand_chain"] = ",".join(chains)
            row["ligand_resnames"] = ",".join(names)
            row["ligand_resseqs"] = ",".join(resseqs)
        row["desired_methods"] = ",".join(methods)
        suggested.append(row)
    return suggested


def _build_report(candidate_rows: list[dict[str, Any]], suggested_targets: list[dict[str, str]], summary: dict[str, Any]) -> str:
    lines = [
        "# CD38 Ligand Candidate Inspection",
        "",
        "This report scans HETATM residues and flags ligand-like residues that contact the known CD38 truth residues.",
        "",
        "## Summary",
        "",
        f"- Structures inspected: `{summary['structure_count']}`",
        f"- HETATM residue candidates: `{summary['candidate_count']}`",
        f"- Recommended ligand-contact candidates: `{summary['recommended_count']}`",
        f"- Structures without active-site ligand candidates: `{', '.join(summary['structures_without_recommended_ligand']) or 'none'}`",
        "",
        "## Recommended Ligand Candidates",
        "",
    ]
    recommended = [row for row in candidate_rows if bool(row.get("recommended_for_ligand_contact"))]
    if recommended:
        headers = [
            "pdb_id",
            "residue_label",
            "atom_count",
            "min_distance_to_truth",
            "truth_contacts_within_threshold",
            "recommendation_reason",
        ]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in recommended:
            values = []
            for header in headers:
                value = row.get(header, "")
                values.append(_format_float(value) if header == "min_distance_to_truth" else str(value))
            lines.append("| " + " | ".join(values) + " |")
    else:
        lines.append("No active-site ligand-like HETATM residues were detected.")

    lines.extend(["", "## Suggested Target Rows", ""])
    lines.append("| pdb_id | desired_methods | ligand_chain | ligand_resnames | ligand_resseqs | notes |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in suggested_targets:
        lines.append(
            "| "
            + " | ".join(
                str(row.get(col, "")).replace("|", "\\|")
                for col in ["pdb_id", "desired_methods", "ligand_chain", "ligand_resnames", "ligand_resseqs", "notes"]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `recommended_for_ligand_contact=True` requires a ligand-like HETATM residue with contact to known CD38 truth residues.",
            "- Water, common ions, buffers, and glycans are listed in the CSV but not recommended as ligand-contact baselines.",
            "- If a structure has no recommended ligand candidate, use it for P2Rank/fpocket pocket detection rather than ligand-contact baseline.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = _build_parser().parse_args()
    targets_csv = Path(args.structure_targets_csv).expanduser().resolve()
    truth_file = Path(args.truth_file).expanduser().resolve()
    inputs_dir = Path(args.inputs_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    suggested_targets_csv = (
        Path(args.suggested_targets_csv).expanduser().resolve()
        if args.suggested_targets_csv
        else out_dir / "cd38_structure_targets_suggested.csv"
    )

    targets = [row for row in _read_csv(targets_csv) if _truthy(row.get("enabled"), default=True)]
    if not targets:
        raise FileNotFoundError(f"No enabled target rows found: {targets_csv}")
    truth_keys = load_residue_set(str(truth_file))
    threshold = float(args.distance_threshold)

    candidate_rows: list[dict[str, Any]] = []
    inspected_pdb_ids: list[str] = []
    errors: list[dict[str, str]] = []
    for target in targets:
        pdb_id = str(target.get("pdb_id") or "").strip().upper()
        if not pdb_id:
            continue
        try:
            pdb_path = _download_pdb(pdb_id, inputs_dir, force_download=bool(args.force_download))
            inspected_pdb_ids.append(pdb_id)
            candidate_rows.extend(
                _inspect_one(
                    pdb_id=pdb_id,
                    pdb_path=pdb_path,
                    truth_keys=truth_keys,
                    protein_chain=str(target.get("protein_chain") or "A").strip(),
                    threshold=threshold,
                )
            )
        except Exception as exc:
            errors.append({"pdb_id": pdb_id, "error": str(exc)})

    suggested_targets = _suggest_target_rows(targets, candidate_rows)
    recommended_count = sum(1 for row in candidate_rows if bool(row.get("recommended_for_ligand_contact")))
    recommended_pdbs = {str(row["pdb_id"]).upper() for row in candidate_rows if bool(row.get("recommended_for_ligand_contact"))}
    structures_without = sorted(set(inspected_pdb_ids) - recommended_pdbs)
    summary = {
        "structure_targets_csv": str(targets_csv),
        "truth_file": str(truth_file),
        "inputs_dir": str(inputs_dir),
        "out_dir": str(out_dir),
        "distance_threshold": threshold,
        "structure_count": len(set(inspected_pdb_ids)),
        "candidate_count": len(candidate_rows),
        "recommended_count": recommended_count,
        "recommended_by_pdb": dict(
            sorted(
                Counter(str(row["pdb_id"]).upper() for row in candidate_rows if bool(row.get("recommended_for_ligand_contact"))).items()
            )
        ),
        "structures_without_recommended_ligand": structures_without,
        "errors": errors,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    candidates_csv = out_dir / "cd38_ligand_candidates.csv"
    recommended_csv = out_dir / "cd38_recommended_ligand_candidates.csv"
    summary_json = out_dir / "cd38_ligand_candidate_summary.json"
    report_md = out_dir / "cd38_ligand_candidate_report.md"
    _write_csv(candidates_csv, candidate_rows, CANDIDATE_COLUMNS)
    _write_csv(
        recommended_csv,
        [row for row in candidate_rows if bool(row.get("recommended_for_ligand_contact"))],
        CANDIDATE_COLUMNS,
    )
    _write_csv(suggested_targets_csv, suggested_targets, TARGET_COLUMNS)
    summary_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    report_md.write_text(_build_report(candidate_rows, suggested_targets, summary), encoding="utf-8")

    print(f"Saved: {candidates_csv}")
    print(f"Saved: {recommended_csv}")
    print(f"Saved: {suggested_targets_csv}")
    print(f"Saved: {summary_json}")
    print(f"Saved: {report_md}")
    print(f"Structures inspected: {summary['structure_count']}")
    print(f"Recommended ligand candidates: {summary['recommended_count']}")


if __name__ == "__main__":
    main()
