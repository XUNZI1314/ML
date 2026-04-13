"""PDB parsing utilities for nanobody-antigen complex preprocessing.

This module focuses on robust structure loading and splitting for downstream
geometry feature extraction. It intentionally does not include any training
or model logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Atom import Atom, DisorderedAtom
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure


_CHAIN_ID_POOL = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


@dataclass(frozen=True, slots=True)
class AtomInfo:
    """A single atom record extracted from a parsed structure.

    Attributes:
        serial_number: Atom serial number from the PDB record.
        atom_name: Atom name, for example "CA".
        element: Chemical element symbol.
        coord: Atom coordinate in Angstrom, shape (3,).
        occupancy: Occupancy value if present.
        bfactor: B-factor value if present.
        altloc: Alternate location code.
        chain_id: Chain identifier.
        resseq: Residue sequence number.
        icode: Insertion code.
        resname: Residue name.
        is_hetatm: Whether this atom belongs to HETATM residue.
    """

    serial_number: int | None
    atom_name: str
    element: str
    coord: tuple[float, float, float]
    occupancy: float | None
    bfactor: float | None
    altloc: str
    chain_id: str
    resseq: int
    icode: str
    resname: str
    is_hetatm: bool


@dataclass(frozen=True, slots=True)
class ResidueInfo:
    """Summary information for one residue.

    Attributes:
        uid: Unique residue identifier.
        chain_id: Chain identifier.
        resseq: Residue sequence number.
        icode: Insertion code.
        resname: Residue name.
        hetflag: Biopython hetero flag (" " for standard amino acid residues).
        atom_count: Number of non-hydrogen atoms included in this residue.
        heavy_atom_count: Number of heavy atoms (same as atom_count here).
    """

    uid: str
    chain_id: str
    resseq: int
    icode: str
    resname: str
    hetflag: str
    atom_count: int
    heavy_atom_count: int


@dataclass(frozen=True, slots=True)
class ChainInfo:
    """Summary information for one chain."""

    chain_id: str
    residue_count: int
    atom_count: int
    heavy_atom_count: int
    het_residue_count: int = 0
    disordered_atom_count: int = 0
    empty_residue_count: int = 0


@dataclass(frozen=True, slots=True)
class StructureValidationResult:
    """Validation summary for a parsed structure or model."""

    structure_id: str
    model_count: int
    chain_count: int
    residue_count: int
    atom_count: int
    heavy_atom_count: int
    chain_infos: tuple[ChainInfo, ...]
    issues: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ParsedComplexResult:
    """Parsed complex metadata attached to the loaded structure."""

    source_path: str
    structure: Structure
    validation: StructureValidationResult
    split_hint: str
    ter_block_count: int
    synthetic_chain_ids: tuple[str, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SplitInferenceResult:
    """Auto-split inference result for antigen/nanobody chain assignment."""

    antigen_chain_ids: tuple[str, ...]
    nanobody_chain_ids: tuple[str, ...]
    split_mode: str
    detail: str = ""


@dataclass(slots=True)
class EntityView:
    """A selected subset of a model, represented by chain IDs.

    Attributes:
        label: Human-readable label, e.g. "antigen" or "nanobody".
        model: Biopython model object that contains all selected chains.
        chain_ids: Tuple of chain IDs included in this view.
        chain_infos: Cached chain-level summaries.
    """

    label: str
    model: Model
    chain_ids: tuple[str, ...]
    chain_infos: tuple[ChainInfo, ...]
    source_mode: str = ""
    source_detail: str = ""


@dataclass(slots=True)
class ComplexSplitResult:
    """Result object that stores antigen/nanobody split outputs."""

    antigen: EntityView
    nanobody: EntityView
    method: str
    split_mode: str = ""
    source_detail: str = ""

    def __post_init__(self) -> None:
        """Keep method and split_mode aligned for backward compatibility."""
        if not self.split_mode:
            self.split_mode = self.method
        if not self.method:
            self.method = self.split_mode


def _record_name(line: str) -> str:
    """Return uppercase PDB record name from one line."""
    return line[:6].strip().upper()


def _replace_chain_id(line: str, chain_id: str) -> str:
    """Replace chain ID field (column 22, 1-based) in a PDB line."""
    if len(chain_id) != 1:
        raise ValueError(f"chain_id must be one character, got {chain_id!r}")

    base = line.rstrip("\r\n")
    if len(base) < 22:
        base = base.ljust(22)
    return f"{base[:21]}{chain_id}{base[22:]}"


def _extract_first_model_lines(lines: list[str]) -> list[str]:
    """Extract lines corresponding to the first MODEL block.

    If no MODEL records exist, the original lines are returned.
    """
    selected: list[str] = []
    saw_model = False
    in_first_model = False

    for line in lines:
        rec = _record_name(line)

        if rec == "MODEL":
            if not saw_model:
                saw_model = True
                in_first_model = True
            else:
                break
            continue

        if rec == "ENDMDL":
            if in_first_model:
                break
            continue

        if saw_model and not in_first_model:
            continue

        selected.append(line.rstrip("\r\n"))

    return selected


def _normalize_chain_ids_if_needed(lines: list[str]) -> tuple[list[str], dict[str, Any]]:
    """Assign synthetic chain IDs from TER blocks when chain IDs are missing.

    In addition to explicit TER records, this helper will also split chainless
    fragments when residue numbering visibly resets, which is common in real
    PDB exports that omit chain identifiers.
    """
    atom_like_lines = [line for line in lines if _record_name(line) in {"ATOM", "HETATM"}]
    if not atom_like_lines:
        raise ValueError("No ATOM/HETATM records found in the first MODEL.")

    has_explicit_chain = any(
        _parse_pdb_atom_metadata(line)["chain_id"] for line in atom_like_lines if len(line) >= 27
    )
    if has_explicit_chain:
        return lines, {
            "chain_id_mode": "original",
            "split_hint": "auto_chain",
            "ter_block_count": 0,
            "synthetic_chain_ids": (),
        }

    blocks = extract_ter_blocks_from_pdb_text(lines)
    if not blocks:
        raise ValueError("Unable to derive chain blocks from chainless PDB text.")

    if len(blocks) > len(_CHAIN_ID_POOL):
        raise ValueError(
            "Too many inferred blocks without chain IDs; "
            f"max supported is {len(_CHAIN_ID_POOL)}."
        )

    normalized: list[str] = []
    generated_chain_ids: list[str] = []
    for idx, block in enumerate(blocks):
        chain_id = _CHAIN_ID_POOL[idx]
        generated_chain_ids.append(chain_id)
        for line in block:
            normalized.append(_replace_chain_id(line, chain_id))
        if idx < len(blocks) - 1:
            normalized.append("TER")

    split_hint = "ter_fallback" if len(blocks) >= 2 or any(_record_name(line) == "TER" for line in lines) else "auto_chain"
    return normalized, {
        "chain_id_mode": "synthetic",
        "split_hint": split_hint,
        "ter_block_count": len(blocks),
        "synthetic_chain_ids": tuple(generated_chain_ids),
    }


def _get_first_model(structure: Structure | Model) -> Model:
    """Return first model from a Structure or pass through a Model."""
    if isinstance(structure, Model):
        return structure
    if not isinstance(structure, Structure):
        raise TypeError(
            "structure must be Bio.PDB.Structure.Structure or Bio.PDB.Model.Model"
        )

    models = list(structure.get_models())
    if not models:
        raise ValueError("No model found in structure.")
    return models[0]


def _chain_label(chain_id: str) -> str:
    """Pretty label for chain ID, using '_' for blank chain IDs."""
    return chain_id if str(chain_id).strip() else "_"


def _is_hetero_residue(residue: Residue) -> bool:
    """Return True if residue comes from HETATM/WAT record."""
    hetflag = str(residue.id[0]).strip()
    return bool(hetflag)


def _infer_element(atom: Atom) -> str:
    """Infer element symbol from atom object, robust to malformed records."""
    element = str(getattr(atom, "element", "") or "").strip().upper()
    if element:
        return element

    atom_name = atom.get_name().strip().upper()
    letters = "".join(ch for ch in atom_name if ch.isalpha())
    if not letters:
        return ""
    return letters[0]


def _is_hydrogen(atom: Atom) -> bool:
    """Return True if atom is hydrogen (or deuterium)."""
    element = _infer_element(atom)
    if element in {"H", "D"}:
        return True

    name = atom.get_name().strip().upper()
    return name.startswith("H") or name.startswith("D")


def _safe_int(text: Any, default: int = -1) -> int:
    """Convert text to int with a fallback value."""
    try:
        return int(str(text).strip())
    except Exception:
        return int(default)


def _normalize_icode_token(icode: Any) -> str:
    """Normalize insertion code to a printable token."""
    text = str(icode).strip()
    if text in {"", " ", "_", "-"}:
        return "_"
    return text[:1]


def _format_residue_uid(
    chain_id: Any,
    resseq: Any,
    icode: Any,
    resname: Any,
    include_hetflag: bool = False,
    hetflag: Any = "",
) -> str:
    """Build a strict residue UID string.

    Format:
        chain_id:resseq:icode:resname

    If include_hetflag is True, a final suffix is appended to avoid ambiguity
    for HETATM residues:
        chain_id:resseq:icode:resname:hetflag
    """
    chain_token = _chain_label(str(chain_id))
    resseq_token = str(_safe_int(resseq, default=-1))
    icode_token = _normalize_icode_token(icode)
    resname_token = str(resname).strip().upper() or "UNK"

    uid = f"{chain_token}:{resseq_token}:{icode_token}:{resname_token}"
    if include_hetflag:
        hetflag_token = str(hetflag).strip().upper() or "ATOM"
        uid = f"{uid}:{hetflag_token}"
    return uid


def _parse_pdb_atom_metadata(line: str) -> dict[str, Any]:
    """Parse the key metadata fields from a PDB ATOM/HETATM line."""
    record = _record_name(line)
    if record not in {"ATOM", "HETATM"}:
        raise ValueError(f"Unsupported PDB record type: {record!r}")

    base = line.rstrip("\r\n")
    serial = _safe_int(base[6:11], default=-1)
    atom_name = base[12:16].strip()
    altloc = base[16:17].strip()
    resname = base[17:20].strip() or "UNK"
    chain_id = base[21:22].strip()
    resseq = _safe_int(base[22:26], default=-1)
    icode = base[26:27].strip()
    return {
        "record": record,
        "serial": serial,
        "atom_name": atom_name,
        "altloc": altloc,
        "resname": resname,
        "chain_id": chain_id,
        "resseq": resseq,
        "icode": icode,
        "is_hetatm": record == "HETATM",
    }


def _atom_altloc_rank(atom: Atom) -> tuple[int, float, float, int]:
    """Ranking key for altloc selection.

    Prefer blank altloc first; if blanks are unavailable, prefer higher occupancy.
    Ties are broken by lower B-factor and then by serial number.
    """
    altloc = str(atom.get_altloc()).strip()
    blank_bonus = 1 if not altloc else 0

    occupancy = atom.get_occupancy()
    occ_value = float(occupancy) if occupancy is not None and np.isfinite(float(occupancy)) else -1.0

    bfactor = atom.get_bfactor()
    bfactor_value = float(bfactor) if bfactor is not None and np.isfinite(float(bfactor)) else 1e9

    try:
        serial = int(atom.get_serial_number())
    except Exception:
        serial = -1

    return blank_bonus, occ_value, -bfactor_value, -serial


def select_best_altloc_atoms(residue: Residue) -> list[Atom]:
    """Select one atom per altloc site, preferring blank or higher-occupancy atoms."""
    if not isinstance(residue, Residue):
        raise TypeError("residue must be a Bio.PDB.Residue.Residue instance")

    selected: list[Atom] = []
    residue_items = list(getattr(residue, "child_list", residue.get_list()))
    for item in residue_items:
        if isinstance(item, DisorderedAtom) or hasattr(item, "disordered_get_list"):
            children = list(item.disordered_get_list()) if hasattr(item, "disordered_get_list") else []
            if not children:
                continue
            best_child = sorted(children, key=_atom_altloc_rank, reverse=True)[0]
            selected.append(best_child)
        else:
            selected.append(item)

    return selected


def extract_ter_blocks_from_pdb_text(pdb_text: Iterable[str] | str) -> list[list[str]]:
    """Split normalized PDB text into blocks using TER records and residue resets.

    This helper is intentionally conservative:
    - It respects explicit TER records.
    - If chain IDs are blank, a decreasing residue number is treated as a block boundary.
    - Non-coordinate records are ignored.
    """
    if isinstance(pdb_text, str):
        lines = pdb_text.splitlines()
    else:
        lines = list(pdb_text)

    blocks: list[list[str]] = []
    current_block: list[str] = []
    last_blank_chain_resseq: int | None = None

    for raw_line in lines:
        record = _record_name(raw_line)
        if record == "TER":
            if current_block:
                blocks.append(current_block)
                current_block = []
            last_blank_chain_resseq = None
            continue

        if record not in {"ATOM", "HETATM"}:
            continue

        meta = _parse_pdb_atom_metadata(raw_line)
        if not meta["chain_id"] and current_block:
            if last_blank_chain_resseq is not None and meta["resseq"] < last_blank_chain_resseq:
                blocks.append(current_block)
                current_block = []
                last_blank_chain_resseq = None

        current_block.append(raw_line.rstrip("\r\n"))
        if not meta["chain_id"]:
            last_blank_chain_resseq = meta["resseq"]

    if current_block:
        blocks.append(current_block)

    return blocks


def summarize_chain_contents(
    entity: Any,
    heavy_only: bool = True,
    include_hetatm: bool = False,
) -> tuple[ChainInfo, ...]:
    """Summarize chain-level residue and atom contents for a structure-like entity."""
    chains = list(_iter_selected_chains(entity))
    summaries: list[ChainInfo] = []

    for chain in chains:
        residue_count = 0
        het_residue_count = 0
        empty_residue_count = 0
        atom_count = 0
        heavy_atom_count = 0
        disordered_atom_count = 0

        for residue in chain.get_residues():
            is_het = _is_hetero_residue(residue)
            if is_het:
                het_residue_count += 1
                if not include_hetatm:
                    continue

            selected_atoms = select_best_altloc_atoms(residue)
            if not selected_atoms:
                empty_residue_count += 1
                continue

            residue_has_valid_atom = False
            residue_atom_count = 0
            for atom in selected_atoms:
                if hasattr(atom, "disordered_get_list"):
                    disordered_atom_count += 1
                if heavy_only and _is_hydrogen(atom):
                    continue
                try:
                    coord = np.asarray(atom.get_coord(), dtype=np.float64).reshape(3)
                except Exception:
                    continue
                if not np.isfinite(coord).all():
                    continue

                residue_atom_count += 1
                residue_has_valid_atom = True

            if residue_has_valid_atom:
                residue_count += 1
                atom_count += residue_atom_count
                heavy_atom_count += residue_atom_count
            else:
                empty_residue_count += 1

        summaries.append(
            ChainInfo(
                chain_id=str(chain.id),
                residue_count=residue_count,
                atom_count=atom_count,
                heavy_atom_count=heavy_atom_count,
                het_residue_count=het_residue_count,
                disordered_atom_count=disordered_atom_count,
                empty_residue_count=empty_residue_count,
            )
        )

    return tuple(summaries)


def validate_structure(entity: Any, strict: bool = True) -> StructureValidationResult:
    """Validate a parsed structure/model and return a concise summary.

    When ``strict=True``, empty structures, empty chains, and atom-less inputs
    raise ValueError with a clear message.
    """
    model = _get_first_model(entity)
    chain_infos = summarize_chain_contents(model, heavy_only=True, include_hetatm=False)

    model_count = 1
    chain_count = len(chain_infos)
    residue_count = int(sum(info.residue_count for info in chain_infos))
    atom_count = int(sum(info.atom_count for info in chain_infos))
    heavy_atom_count = int(sum(info.heavy_atom_count for info in chain_infos))

    issues: list[str] = []
    if chain_count == 0:
        issues.append("No chains found in the first MODEL.")
    if residue_count == 0:
        issues.append("No standard residues found in the first MODEL.")
    if heavy_atom_count == 0:
        issues.append("No valid heavy atoms found in the first MODEL.")

    empty_chain_ids = [info.chain_id for info in chain_infos if info.residue_count == 0]
    if empty_chain_ids:
        issues.append(f"Empty chains detected: {', '.join(empty_chain_ids)}")

    result = StructureValidationResult(
        structure_id=str(getattr(entity, "id", "unknown")),
        model_count=model_count,
        chain_count=chain_count,
        residue_count=residue_count,
        atom_count=atom_count,
        heavy_atom_count=heavy_atom_count,
        chain_infos=chain_infos,
        issues=tuple(issues),
    )

    if strict and issues:
        raise ValueError(
            f"Invalid PDB structure {result.structure_id!r}: "
            + "; ".join(issues)
        )

    return result


def infer_split_by_chain_size_or_composition(
    chain_infos: Iterable[ChainInfo],
    chain_order: Iterable[str] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...], str, str]:
    """Infer antigen/nanobody chains from chain sizes when user does not provide IDs.

    Heuristic:
    - For two chains, the smaller chain is treated as nanobody.
    - For more than two chains, a single small outlier chain can be treated as nanobody
      if it is clearly smaller than the next chain and within a plausible VHH size range.
    """
    infos = [info for info in chain_infos if info.residue_count > 0]
    if len(infos) < 2:
        raise ValueError("Need at least two non-empty chains to infer antigen/nanobody split.")

    order = list(chain_order or [])
    order_index = {cid: idx for idx, cid in enumerate(order)}

    def sort_key(info: ChainInfo) -> tuple[int, int, int, int, str]:
        return (
            info.residue_count,
            info.heavy_atom_count,
            info.atom_count,
            order_index.get(info.chain_id, 10_000),
            info.chain_id,
        )

    ranked = sorted(infos, key=sort_key)
    smallest = ranked[0]
    second = ranked[1]

    if len(ranked) == 2:
        nanobody_ids = (smallest.chain_id,)
        antigen_ids = (ranked[1].chain_id,)
        detail = f"Two chains detected; smaller chain {smallest.chain_id} selected as nanobody."
        return antigen_ids, nanobody_ids, "auto_chain", detail

    plausible_vhh = smallest.residue_count <= 220
    clearly_smaller = smallest.residue_count <= max(1, int(round(second.residue_count * 0.75)))
    if plausible_vhh and clearly_smaller:
        nanobody_ids = (smallest.chain_id,)
        antigen_ids = tuple(info.chain_id for info in infos if info.chain_id != smallest.chain_id)
        detail = (
            f"Auto split selected {smallest.chain_id} as nanobody using chain-size heuristic; "
            f"residue_count={smallest.residue_count}, next_smallest={second.residue_count}."
        )
        return antigen_ids, nanobody_ids, "auto_chain", detail

    raise ValueError(
        "Unable to infer antigen/nanobody split automatically from chain size. "
        f"Candidates: {', '.join(f'{info.chain_id}({info.residue_count})' for info in ranked)}"
    )


def _safe_int(text: Any, default: int = -1) -> int:
    """Convert text to int with a fallback value."""
    try:
        return int(str(text).strip())
    except Exception:
        return int(default)


def _normalize_icode_token(icode: Any) -> str:
    """Normalize insertion code to a printable token."""
    text = str(icode).strip()
    if text in {"", " ", "_", "-"}:
        return "_"
    return text[:1]


def _format_residue_uid(
    chain_id: Any,
    resseq: Any,
    icode: Any,
    resname: Any,
    include_hetflag: bool = False,
    hetflag: Any = "",
) -> str:
    """Build a strict residue UID string.

    Format:
        chain_id:resseq:icode:resname

    If include_hetflag is True, a final suffix is appended to avoid ambiguity
    for HETATM residues:
        chain_id:resseq:icode:resname:hetflag
    """
    chain_token = _chain_label(str(chain_id))
    resseq_token = str(_safe_int(resseq, default=-1))
    icode_token = _normalize_icode_token(icode)
    resname_token = str(resname).strip().upper() or "UNK"

    uid = f"{chain_token}:{resseq_token}:{icode_token}:{resname_token}"
    if include_hetflag:
        hetflag_token = str(hetflag).strip().upper() or "ATOM"
        uid = f"{uid}:{hetflag_token}"
    return uid


def _parse_pdb_atom_metadata(line: str) -> dict[str, Any]:
    """Parse the key metadata fields from a PDB ATOM/HETATM line."""
    record = _record_name(line)
    if record not in {"ATOM", "HETATM"}:
        raise ValueError(f"Unsupported PDB record type: {record!r}")

    base = line.rstrip("\r\n")
    serial = _safe_int(base[6:11], default=-1)
    atom_name = base[12:16].strip()
    altloc = base[16:17].strip()
    resname = base[17:20].strip() or "UNK"
    chain_id = base[21:22].strip()
    resseq = _safe_int(base[22:26], default=-1)
    icode = base[26:27].strip()
    return {
        "record": record,
        "serial": serial,
        "atom_name": atom_name,
        "altloc": altloc,
        "resname": resname,
        "chain_id": chain_id,
        "resseq": resseq,
        "icode": icode,
        "is_hetatm": record == "HETATM",
    }


def _atom_altloc_rank(atom: Atom) -> tuple[int, float, float, int]:
    """Ranking key for altloc selection.

    Prefer blank altloc first; if blanks are unavailable, prefer higher occupancy.
    Ties are broken by lower B-factor and then by serial number.
    """
    altloc = str(atom.get_altloc()).strip()
    blank_bonus = 1 if not altloc else 0

    occupancy = atom.get_occupancy()
    occ_value = float(occupancy) if occupancy is not None and np.isfinite(float(occupancy)) else -1.0

    bfactor = atom.get_bfactor()
    bfactor_value = float(bfactor) if bfactor is not None and np.isfinite(float(bfactor)) else 1e9

    try:
        serial = int(atom.get_serial_number())
    except Exception:
        serial = -1

    return blank_bonus, occ_value, -bfactor_value, -serial


def select_best_altloc_atoms(residue: Residue) -> list[Atom]:
    """Select one atom per altloc site, preferring blank or higher-occupancy atoms."""
    if not isinstance(residue, Residue):
        raise TypeError("residue must be a Bio.PDB.Residue.Residue instance")

    selected: list[Atom] = []
    residue_items = list(getattr(residue, "child_list", residue.get_list()))
    for item in residue_items:
        if isinstance(item, DisorderedAtom) or hasattr(item, "disordered_get_list"):
            children = list(item.disordered_get_list()) if hasattr(item, "disordered_get_list") else []
            if not children:
                continue
            best_child = sorted(children, key=_atom_altloc_rank, reverse=True)[0]
            selected.append(best_child)
        else:
            selected.append(item)

    return selected


def extract_ter_blocks_from_pdb_text(pdb_text: Iterable[str] | str) -> list[list[str]]:
    """Split normalized PDB text into blocks using TER records and residue resets.

    This helper is intentionally conservative:
    - It respects explicit TER records.
    - If chain IDs are blank, a decreasing residue number is treated as a block boundary.
    - Non-coordinate records are ignored.
    """
    if isinstance(pdb_text, str):
        lines = pdb_text.splitlines()
    else:
        lines = list(pdb_text)

    blocks: list[list[str]] = []
    current_block: list[str] = []
    last_blank_chain_resseq: int | None = None
    saw_ter = False

    for raw_line in lines:
        record = _record_name(raw_line)
        if record == "TER":
            saw_ter = True
            if current_block:
                blocks.append(current_block)
                current_block = []
            last_blank_chain_resseq = None
            continue

        if record not in {"ATOM", "HETATM"}:
            continue

        meta = _parse_pdb_atom_metadata(raw_line)
        if not meta["chain_id"] and current_block:
            if last_blank_chain_resseq is not None and meta["resseq"] < last_blank_chain_resseq:
                blocks.append(current_block)
                current_block = []
                last_blank_chain_resseq = None

        current_block.append(raw_line.rstrip("\r\n"))
        if not meta["chain_id"]:
            last_blank_chain_resseq = meta["resseq"]

    if current_block:
        blocks.append(current_block)

    # Return a single block when no coordinate records were found.
    if not blocks and saw_ter:
        return []
    return blocks


def summarize_chain_contents(
    entity: Any,
    heavy_only: bool = True,
    include_hetatm: bool = False,
) -> tuple[ChainInfo, ...]:
    """Summarize chain-level residue and atom contents for a structure-like entity."""
    chains = list(_iter_selected_chains(entity))
    summaries: list[ChainInfo] = []

    for chain in chains:
        residue_count = 0
        het_residue_count = 0
        empty_residue_count = 0
        atom_count = 0
        heavy_atom_count = 0
        disordered_atom_count = 0

        for residue in chain.get_residues():
            is_het = _is_hetero_residue(residue)
            if is_het:
                het_residue_count += 1
                if not include_hetatm:
                    continue

            selected_atoms = select_best_altloc_atoms(residue)
            if not selected_atoms:
                empty_residue_count += 1
                continue

            residue_has_heavy_atom = False
            residue_atom_count = 0
            for atom in selected_atoms:
                if hasattr(atom, "disordered_get_list"):
                    disordered_atom_count += 1
                if heavy_only and _is_hydrogen(atom):
                    continue
                try:
                    coord = np.asarray(atom.get_coord(), dtype=np.float64).reshape(3)
                except Exception:
                    continue
                if not np.isfinite(coord).all():
                    continue

                residue_atom_count += 1
                residue_has_heavy_atom = True

            if residue_has_heavy_atom:
                residue_count += 1
                atom_count += residue_atom_count
                heavy_atom_count += residue_atom_count
            else:
                empty_residue_count += 1

        summaries.append(
            ChainInfo(
                chain_id=str(chain.id),
                residue_count=residue_count,
                atom_count=atom_count,
                heavy_atom_count=heavy_atom_count,
                het_residue_count=het_residue_count,
                disordered_atom_count=disordered_atom_count,
                empty_residue_count=empty_residue_count,
            )
        )

    return tuple(summaries)


def validate_structure(entity: Any, strict: bool = True) -> StructureValidationResult:
    """Validate a parsed structure/model and return a concise summary.

    When ``strict=True``, empty structures, empty chains, and atom-less inputs
    raise ValueError with a clear message.
    """
    model = _get_first_model(entity)
    chain_infos = summarize_chain_contents(model, heavy_only=True, include_hetatm=False)

    model_count = 1 if isinstance(entity, (Structure, Model)) else 1
    chain_count = len(chain_infos)
    residue_count = int(sum(info.residue_count for info in chain_infos))
    atom_count = int(sum(info.atom_count for info in chain_infos))
    heavy_atom_count = int(sum(info.heavy_atom_count for info in chain_infos))

    issues: list[str] = []
    if chain_count == 0:
        issues.append("No chains found in the first MODEL.")
    if residue_count == 0:
        issues.append("No standard residues found in the first MODEL.")
    if heavy_atom_count == 0:
        issues.append("No valid heavy atoms found in the first MODEL.")

    empty_chain_ids = [info.chain_id for info in chain_infos if info.residue_count == 0]
    if empty_chain_ids:
        issues.append(f"Empty chains detected: {', '.join(empty_chain_ids)}")

    result = StructureValidationResult(
        structure_id=str(getattr(entity, "id", "unknown")),
        model_count=model_count,
        chain_count=chain_count,
        residue_count=residue_count,
        atom_count=atom_count,
        heavy_atom_count=heavy_atom_count,
        chain_infos=chain_infos,
        issues=tuple(issues),
    )

    if strict and issues:
        raise ValueError(
            f"Invalid PDB structure {result.structure_id!r}: "
            + "; ".join(issues)
        )

    return result


def infer_split_by_chain_size_or_composition(
    chain_infos: Iterable[ChainInfo],
    chain_order: Iterable[str] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...], str, str]:
    """Infer antigen/nanobody chains from chain sizes when user does not provide IDs.

    Heuristic:
    - For two chains, the smaller chain is treated as nanobody.
    - For more than two chains, a single small outlier chain can be treated as nanobody
      if it is clearly smaller than the next chain and within a plausible VHH size range.
    """
    infos = [info for info in chain_infos if info.residue_count > 0]
    if len(infos) < 2:
        raise ValueError("Need at least two non-empty chains to infer antigen/nanobody split.")

    order = list(chain_order or [])
    order_index = {cid: idx for idx, cid in enumerate(order)}

    def sort_key(info: ChainInfo) -> tuple[int, int, int, int, str]:
        return (
            info.residue_count,
            info.heavy_atom_count,
            info.atom_count,
            order_index.get(info.chain_id, 10_000),
            info.chain_id,
        )

    ranked = sorted(infos, key=sort_key)
    smallest = ranked[0]
    second = ranked[1]

    if len(ranked) == 2:
        nanobody_ids = (smallest.chain_id,)
        antigen_ids = (ranked[1].chain_id,)
        detail = f"Two chains detected; smaller chain {smallest.chain_id} selected as nanobody."
        return antigen_ids, nanobody_ids, "auto_chain", detail

    plausible_vhh = smallest.residue_count <= 220
    clearly_smaller = smallest.residue_count <= max(1, int(round(second.residue_count * 0.75)))
    if plausible_vhh and clearly_smaller:
        nanobody_ids = (smallest.chain_id,)
        antigen_ids = tuple(info.chain_id for info in infos if info.chain_id != smallest.chain_id)
        detail = (
            f"Auto split selected {smallest.chain_id} as nanobody using chain-size heuristic; "
            f"residue_count={smallest.residue_count}, next_smallest={second.residue_count}."
        )
        return antigen_ids, nanobody_ids, "auto_chain", detail

    raise ValueError(
        "Unable to infer antigen/nanobody split automatically from chain size. "
        f"Candidates: {', '.join(f'{info.chain_id}({info.residue_count})' for info in ranked)}"
    )


def _iter_selected_chains(entity: Any) -> Iterator[Chain]:
    """Yield selected chains from supported entity types."""
    if isinstance(entity, EntityView):
        for chain_id in entity.chain_ids:
            if chain_id not in entity.model:
                raise ValueError(
                    f"Chain {chain_id!r} is missing from model for entity {entity.label!r}."
                )
            yield entity.model[chain_id]
        return

    if isinstance(entity, Chain):
        yield entity
        return

    if isinstance(entity, Residue):
        parent_chain = entity.get_parent()
        if not isinstance(parent_chain, Chain):
            raise ValueError("Residue parent chain is invalid.")
        yield parent_chain
        return

    if isinstance(entity, Model):
        yield from entity.get_chains()
        return

    if isinstance(entity, Structure):
        model = _get_first_model(entity)
        yield from model.get_chains()
        return

    raise TypeError(
        "Unsupported entity type. Expected EntityView, Structure, Model, Chain or Residue."
    )


def _iter_selected_residues(entity: Any) -> Iterator[Residue]:
    """Yield residues from supported entity types."""
    if isinstance(entity, Residue):
        yield entity
        return

    for chain in _iter_selected_chains(entity):
        yield from chain.get_residues()


def _summarize_chains(model: Model, chain_ids: Iterable[str]) -> tuple[ChainInfo, ...]:
    """Build chain-level summary objects for selected chain IDs."""
    selected_chain_ids = tuple(str(chain_id) for chain_id in chain_ids)
    for chain_id in selected_chain_ids:
        if chain_id not in model:
            raise ValueError(f"Chain {chain_id!r} not found in model.")

    selection = EntityView(
        label="selected",
        model=model,
        chain_ids=selected_chain_ids,
        chain_infos=(),
    )
    return summarize_chain_contents(selection, heavy_only=True, include_hetatm=False)


def _normalize_chain_selection(chain_value: str | Iterable[str] | None) -> list[str] | None:
    """Normalize user chain selection argument to list[str]."""
    if chain_value is None:
        return None

    if isinstance(chain_value, str):
        if not chain_value:
            raise ValueError("Chain ID string cannot be empty.")
        return [chain_value]

    try:
        values = list(chain_value)
    except TypeError as exc:
        raise TypeError("chain selection must be str, iterable[str], or None") from exc

    if not values:
        raise ValueError("chain selection cannot be an empty iterable.")

    normalized = []
    for item in values:
        if item is None:
            raise ValueError("chain selection contains None.")
        text = str(item)
        if not text:
            raise ValueError("chain selection contains an empty ID.")
        normalized.append(text)
    return normalized


def parse_complex_pdb(pdb_path: str) -> ParsedComplexResult:
    """Parse a complex PDB and return the structure plus validation metadata."""
    path = Path(pdb_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"PDB file not found: {path}")
    if not path.is_file():
        raise ValueError(f"PDB path is not a file: {path}")

    try:
        raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        raise ValueError(f"Failed to read PDB text from {path}: {exc}") from exc

    first_model_lines = _extract_first_model_lines(raw_lines)
    if not first_model_lines:
        raise ValueError(f"No usable coordinate records found in first MODEL of {path}.")

    normalized_lines, metadata = _normalize_chain_ids_if_needed(first_model_lines)

    parser = PDBParser(PERMISSIVE=True, QUIET=True)
    try:
        structure = parser.get_structure(path.stem, StringIO("\n".join(normalized_lines) + "\n"))
    except Exception as exc:
        raise ValueError(f"Failed to parse PDB structure from {path}: {exc}") from exc

    validation = validate_structure(structure, strict=True)
    parsed = ParsedComplexResult(
        source_path=str(path),
        structure=structure,
        validation=validation,
        split_hint=str(metadata.get("split_hint", "auto_chain")),
        ter_block_count=int(metadata.get("ter_block_count", 0)),
        synthetic_chain_ids=tuple(metadata.get("synthetic_chain_ids", ())),
        warnings=(),
    )

    structure.xtra["source_pdb_path"] = str(path)
    structure.xtra["chain_id_mode"] = metadata["chain_id_mode"]
    structure.xtra["split_hint"] = metadata.get("split_hint", "auto_chain")
    structure.xtra["ter_block_count"] = int(metadata.get("ter_block_count", 0))
    structure.xtra["synthetic_chain_ids"] = tuple(metadata.get("synthetic_chain_ids", ()))
    structure.xtra["validation_result"] = validation
    structure.xtra["parsed_complex_result"] = parsed
    return parsed


def load_complex_pdb(pdb_path: str) -> Structure:
    """Load a complex PDB and keep only the first MODEL.

    This is a compatibility wrapper that returns the Biopython Structure while
    the richer :func:`parse_complex_pdb` helper keeps the parsed metadata.
    """
    return parse_complex_pdb(pdb_path).structure


def split_antigen_nanobody(
    structure: Structure | Model,
    antigen_chain: str | Iterable[str] | None = None,
    nanobody_chain: str | Iterable[str] | None = None,
    fallback_by_ter: bool = True,
) -> ComplexSplitResult:
    """Split parsed complex into antigen and nanobody entities.

    Priority of split strategy:
    1) Explicit user-provided chain IDs.
    2) TER fallback metadata generated by ``load_complex_pdb``.
    3) If exactly two chains exist, use chain order in file.

    Args:
        structure: Biopython Structure or Model.
        antigen_chain: Chain ID(s) for antigen.
        nanobody_chain: Chain ID(s) for nanobody.
        fallback_by_ter: Whether to use TER-based fallback when IDs are not provided.

    Returns:
        A ``ComplexSplitResult`` containing antigen and nanobody views.

    Raises:
        ValueError: On ambiguous or invalid split configuration.
    """
    model = _get_first_model(structure)
    chains = list(model.get_chains())
    if not chains:
        raise ValueError("No chain found in first MODEL.")

    chain_map = {str(chain.id): chain for chain in chains}
    available_ids = list(chain_map.keys())
    chain_infos = summarize_chain_contents(model, heavy_only=True, include_hetatm=False)

    antigen_ids = _normalize_chain_selection(antigen_chain)
    nanobody_ids = _normalize_chain_selection(nanobody_chain)
    split_mode = ""
    source_detail = ""

    if antigen_ids is not None or nanobody_ids is not None:
        if antigen_ids is None:
            assert nanobody_ids is not None
            antigen_ids = [cid for cid in available_ids if cid not in nanobody_ids]
        if nanobody_ids is None:
            assert antigen_ids is not None
            nanobody_ids = [cid for cid in available_ids if cid not in antigen_ids]

        assert antigen_ids is not None
        assert nanobody_ids is not None

        if not antigen_ids:
            raise ValueError("No antigen chain left after applying nanobody_chain selection.")
        if not nanobody_ids:
            raise ValueError("No nanobody chain left after applying antigen_chain selection.")

        overlap = set(antigen_ids) & set(nanobody_ids)
        if overlap:
            overlap_txt = ", ".join(sorted(_chain_label(x) for x in overlap))
            raise ValueError(f"Antigen and nanobody chains overlap: {overlap_txt}")

        unknown = [cid for cid in (antigen_ids + nanobody_ids) if cid not in chain_map]
        if unknown:
            unknown_txt = ", ".join(sorted(_chain_label(x) for x in unknown))
            available_txt = ", ".join(_chain_label(x) for x in available_ids)
            raise ValueError(
                f"Unknown chain ID(s): {unknown_txt}. Available chain IDs: {available_txt}"
            )

        split_mode = "explicit_chain"
        source_detail = (
            f"antigen={','.join(antigen_ids)}; nanobody={','.join(nanobody_ids)}"
        )
    else:
        chain_id_mode = str(getattr(structure, "xtra", {}).get("chain_id_mode", "original"))
        synthetic_chain_ids = tuple(getattr(structure, "xtra", {}).get("synthetic_chain_ids", ()))
        ter_block_count = int(getattr(structure, "xtra", {}).get("ter_block_count", 0))

        if fallback_by_ter and chain_id_mode == "synthetic":
            if len(synthetic_chain_ids) == 2:
                antigen_ids = [synthetic_chain_ids[0]]
                nanobody_ids = [synthetic_chain_ids[1]]
                source_detail = (
                    f"synthetic_blocks={','.join(synthetic_chain_ids)}; ter_block_count={ter_block_count}"
                )
            else:
                antigen_ids, nanobody_ids, split_mode, source_detail = infer_split_by_chain_size_or_composition(
                    chain_infos,
                    chain_order=available_ids,
                )
            split_mode = "ter_fallback"
        else:
            antigen_ids, nanobody_ids, split_mode, source_detail = infer_split_by_chain_size_or_composition(
                chain_infos,
                chain_order=available_ids,
            )

    assert antigen_ids is not None
    assert nanobody_ids is not None

    antigen_view = EntityView(
        label="antigen",
        model=model,
        chain_ids=tuple(antigen_ids),
        chain_infos=_summarize_chains(model, antigen_ids),
        source_mode=split_mode,
        source_detail=source_detail,
    )
    nanobody_view = EntityView(
        label="nanobody",
        model=model,
        chain_ids=tuple(nanobody_ids),
        chain_infos=_summarize_chains(model, nanobody_ids),
        source_mode=split_mode,
        source_detail=source_detail,
    )

    return ComplexSplitResult(
        antigen=antigen_view,
        nanobody=nanobody_view,
        method=split_mode,
        split_mode=split_mode,
        source_detail=source_detail,
    )


def extract_atoms_from_entity(
    entity: Any,
    heavy_only: bool = True,
    include_hetatm: bool = False,
) -> list[AtomInfo]:
    """Extract atom records from an entity.

    Args:
        entity: EntityView, Structure, Model, Chain, or Residue.
        heavy_only: If True, only heavy atoms are returned.
        include_hetatm: If False (default), atoms from HETATM residues are ignored.

    Returns:
        List of ``AtomInfo`` records.
    """
    atoms: list[AtomInfo] = []

    for residue in _iter_selected_residues(entity):
        is_hetatm = _is_hetero_residue(residue)
        if is_hetatm and not include_hetatm:
            continue

        chain = residue.get_parent()
        if not isinstance(chain, Chain):
            continue

        resseq = int(residue.id[1])
        icode = str(residue.id[2]).strip()
        resname = str(residue.get_resname()).strip()

        for atom in residue.get_atoms():
            if heavy_only and _is_hydrogen(atom):
                continue

            try:
                coord = np.asarray(atom.get_coord(), dtype=np.float64).reshape(3)
            except Exception:
                # Missing or malformed atom coordinates are skipped safely.
                continue

            if not np.isfinite(coord).all():
                continue

            serial_number: int | None
            try:
                serial_number = int(atom.get_serial_number())
            except Exception:
                serial_number = None

            occupancy = atom.get_occupancy()
            occupancy_value = float(occupancy) if occupancy is not None else None
            bfactor_value = float(atom.get_bfactor()) if atom.get_bfactor() is not None else None
            altloc = str(atom.get_altloc()).strip()
            element = _infer_element(atom)

            atoms.append(
                AtomInfo(
                    serial_number=serial_number,
                    atom_name=str(atom.get_name()).strip(),
                    element=element,
                    coord=(float(coord[0]), float(coord[1]), float(coord[2])),
                    occupancy=occupancy_value,
                    bfactor=bfactor_value,
                    altloc=altloc,
                    chain_id=str(chain.id),
                    resseq=resseq,
                    icode=icode,
                    resname=resname,
                    is_hetatm=is_hetatm,
                )
            )

    return atoms


def extract_residues_from_entity(entity: Any) -> list[ResidueInfo]:
    """Extract residue records from an entity.

    Standard behavior ignores HETATM residues by default, matching the atom
    extraction default and common protein-only workflows.

    Args:
        entity: EntityView, Structure, Model, Chain, or Residue.

    Returns:
        List of ``ResidueInfo``.
    """
    residues: list[ResidueInfo] = []

    for residue in _iter_selected_residues(entity):
        hetflag = str(residue.id[0]).strip()
        if hetflag:
            continue

        chain = residue.get_parent()
        if not isinstance(chain, Chain):
            continue

        heavy_count = 0
        for atom in residue.get_atoms():
            if _is_hydrogen(atom):
                continue
            heavy_count += 1

        residues.append(
            ResidueInfo(
                uid=get_residue_uid(residue),
                chain_id=str(chain.id),
                resseq=int(residue.id[1]),
                icode=str(residue.id[2]).strip(),
                resname=str(residue.get_resname()).strip(),
                hetflag=hetflag,
                atom_count=heavy_count,
                heavy_atom_count=heavy_count,
            )
        )

    return residues


def get_residue_uid(residue: Residue) -> str:
    """Build a residue UID using chain + residue number + insertion code + name.

    UID format:
        ``{chain}:{resseq}:{icode}:{resname}:{hetflag}``

    Examples:
        ``A:42:_:TYR:ATOM``
        ``B:100:A:SER:ATOM``

    Args:
        residue: Biopython residue object.

    Returns:
        Stable residue UID string.
    """
    if not isinstance(residue, Residue):
        raise TypeError("residue must be a Bio.PDB.Residue.Residue instance")

    chain = residue.get_parent()
    chain_id = str(chain.id) if isinstance(chain, Chain) else " "
    chain_token = _chain_label(chain_id)

    hetflag_raw = str(residue.id[0]).strip()
    hetflag = hetflag_raw if hetflag_raw else "ATOM"

    resseq = int(residue.id[1])
    icode_raw = str(residue.id[2]).strip()
    icode = icode_raw if icode_raw else "_"

    resname = str(residue.get_resname()).strip() or "UNK"
    return f"{chain_token}:{resseq}:{icode}:{resname}:{hetflag}"


def _coords_to_array(coords: Any, name: str) -> np.ndarray:
    """Convert coordinate-like input to a clean (N, 3) float array."""
    if isinstance(coords, np.ndarray):
        arr = np.asarray(coords, dtype=np.float64)
    elif isinstance(coords, list) and coords and isinstance(coords[0], AtomInfo):
        arr = np.asarray([atom.coord for atom in coords], dtype=np.float64)
    else:
        arr = np.asarray(coords, dtype=np.float64)

    if arr.ndim == 1:
        if arr.size != 3:
            raise ValueError(f"{name} must contain 3 values or shape (N, 3), got shape {arr.shape}")
        arr = arr.reshape(1, 3)

    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {arr.shape}")

    valid_mask = np.isfinite(arr).all(axis=1)
    arr = arr[valid_mask]
    if arr.size == 0:
        raise ValueError(f"{name} has no valid finite coordinates.")
    return arr


def compute_centroid(coords: Any) -> np.ndarray:
    """Compute centroid for a set of 3D coordinates.

    Args:
        coords: Array-like with shape (N, 3), single point (3,), or list[AtomInfo].

    Returns:
        A numpy array with shape (3,) representing centroid.
    """
    arr = _coords_to_array(coords, name="coords")
    return np.mean(arr, axis=0, dtype=np.float64)


def pairwise_min_distance(coords_a: Any, coords_b: Any) -> float:
    """Compute minimum Euclidean distance between two coordinate sets.

    Args:
        coords_a: Array-like with shape (N, 3) or list[AtomInfo].
        coords_b: Array-like with shape (M, 3) or list[AtomInfo].

    Returns:
        Minimum pairwise distance as float.
    """
    a = _coords_to_array(coords_a, name="coords_a")
    b = _coords_to_array(coords_b, name="coords_b")

    min_d2 = np.inf
    chunk_size = 1024

    for start in range(0, a.shape[0], chunk_size):
        end = min(start + chunk_size, a.shape[0])
        block = a[start:end]
        diff = block[:, None, :] - b[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff, optimize=True)
        block_min = float(np.min(d2))
        if block_min < min_d2:
            min_d2 = block_min

    return float(np.sqrt(min_d2))


__all__ = [
    "AtomInfo",
    "ResidueInfo",
    "ChainInfo",
    "EntityView",
    "ComplexSplitResult",
    "load_complex_pdb",
    "split_antigen_nanobody",
    "extract_atoms_from_entity",
    "extract_residues_from_entity",
    "get_residue_uid",
    "compute_centroid",
    "pairwise_min_distance",
]
