from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure

from extract_fpocket_pocket_residues import extract_fpocket_residue_keys
from pdb_parser import load_complex_pdb
from pocket_io import (
    extract_residue_atoms,
    load_ligand_template_pdb,
    load_residue_set,
    match_residues_in_structure,
    normalize_residue_key,
    parse_residue_token_or_range,
    parse_fpocket_output,
    parse_p2rank_output,
)


EVIDENCE_COLUMNS = [
    "residue_key",
    "evidence_type",
    "source",
    "confidence",
    "method",
    "structure_id",
    "source_path",
    "weight",
    "distance_angstrom",
    "note",
]

SUPPORT_COLUMNS = [
    "residue_key",
    "chain_id",
    "resseq",
    "icode",
    "resname",
    "present_in_structure",
    "weighted_support_score",
    "evidence_count",
    "method_count",
    "high_confidence_source_count",
    "max_weight",
    "min_distance_angstrom",
    "evidence_types",
    "methods",
    "sources",
    "is_curated_candidate",
    "needs_review",
    "review_reason",
]

SOURCE_TEMPLATE_COLUMNS = [
    "residue_key",
    "evidence_role",
    "source_kind",
    "paper_title",
    "pmid",
    "doi",
    "uniprot_id",
    "uniprot_feature",
    "mcsa_id",
    "pdb_id",
    "source_url",
    "source_sentence",
    "evidence_level",
    "ai_model",
    "ai_prompt_id",
    "ai_extraction_confidence",
    "curator",
    "review_status",
    "manual_note",
]

SOURCE_AUDIT_COLUMNS = [
    *SOURCE_TEMPLATE_COLUMNS,
    "source_table_path",
    "source_residue_file",
    "has_traceable_source",
    "audit_issue",
]

TRACEABLE_SOURCE_FIELDS = (
    "paper_title",
    "pmid",
    "doi",
    "uniprot_id",
    "uniprot_feature",
    "mcsa_id",
    "pdb_id",
    "source_sentence",
)

EXTERNAL_POCKET_SOURCES = {"p2rank", "fpocket"}
TRUSTED_CURATED_EVIDENCE_TYPES = {
    "manual_pocket",
    "literature_residue",
    "catalytic_core",
    "ligand_contact",
}

SOURCE_TABLE_ALIASES = {
    "residue_key": ("residue_key", "residue", "residue_id", "site", "residue_site"),
    "chain_id": ("chain_id", "chain", "chainid"),
    "resseq": ("resseq", "residue_number", "resnum", "res_no", "position", "aa_position"),
    "icode": ("icode", "insertion_code", "ins_code"),
    "evidence_role": ("evidence_role", "role", "evidence_type"),
    "source_kind": ("source_kind", "source_category", "source_type", "source"),
    "paper_title": ("paper_title", "paper", "title", "article_title"),
    "pmid": ("pmid", "pubmed", "pubmed_id"),
    "doi": ("doi",),
    "uniprot_id": ("uniprot_id", "uniprot", "accession"),
    "uniprot_feature": ("uniprot_feature", "feature", "annotation"),
    "mcsa_id": ("mcsa_id", "m_csa", "mcsa"),
    "pdb_id": ("pdb_id", "pdb"),
    "source_url": ("source_url", "url", "link"),
    "source_sentence": ("source_sentence", "evidence_sentence", "sentence", "quote", "excerpt"),
    "evidence_level": ("evidence_level", "level", "confidence"),
    "ai_model": ("ai_model", "model", "llm_model"),
    "ai_prompt_id": ("ai_prompt_id", "prompt_id", "prompt_version"),
    "ai_extraction_confidence": ("ai_extraction_confidence", "ai_confidence", "extraction_confidence"),
    "curator": ("curator", "reviewer", "annotator"),
    "review_status": ("review_status", "status", "manual_status"),
    "manual_note": ("manual_note", "note", "notes", "comment", "comments", "remark"),
}


@dataclass(frozen=True, slots=True)
class EvidenceRow:
    residue_key: str
    evidence_type: str
    source: str
    confidence: str
    method: str
    structure_id: str
    source_path: str
    weight: float
    distance_angstrom: float | None = None
    note: str = ""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build residue-level pocket evidence by combining manual, literature, "
            "catalytic-anchor, ligand-contact, P2Rank, fpocket and AI-prior inputs."
        )
    )
    parser.add_argument("--pdb_path", required=True, help="Representative PDB structure.")
    parser.add_argument("--out_dir", default="pocket_evidence_outputs", help="Output directory.")
    parser.add_argument("--antigen_chain", default="B", help="Antigen/protein chain to evaluate. Default: B.")
    parser.add_argument(
        "--include_non_antigen_residues",
        action="store_true",
        help="Keep evidence residues outside antigen_chain. Default: filter them out.",
    )
    parser.add_argument("--manual_pocket_file", default=None, help="Manual/rsite pocket residue file.")
    parser.add_argument("--literature_file", default=None, help="Literature-curated functional/pocket residue file.")
    parser.add_argument("--catalytic_file", default=None, help="Catalytic anchor residue file.")
    parser.add_argument(
        "--literature_source_table",
        default=None,
        help=(
            "Optional CSV/TSV source audit table for literature_file. "
            "Columns may include residue_key, paper_title, pmid, doi, uniprot_id, mcsa_id, source_sentence, review_status."
        ),
    )
    parser.add_argument(
        "--catalytic_source_table",
        default=None,
        help=(
            "Optional CSV/TSV source audit table for catalytic_file. "
            "Columns may include residue_key, paper_title, pmid, doi, uniprot_id, mcsa_id, source_sentence, review_status."
        ),
    )
    parser.add_argument(
        "--ai_pocket_file",
        default=None,
        help="Optional AI/LLM-assisted pocket residue prior file. This is evidence only, not an API call.",
    )
    parser.add_argument(
        "--ai_source_table",
        "--ai_prior_source_table",
        dest="ai_source_table",
        default=None,
        help=(
            "Optional CSV/TSV offline audit table for AI prior residues. "
            "Rows should keep source_sentence, evidence_level and review_status; AI prior is never treated as ground truth."
        ),
    )
    parser.add_argument(
        "--p2rank_file",
        default=None,
        help="P2Rank predictions.csv, directory, or normalized residue-list file.",
    )
    parser.add_argument("--p2rank_top_n", type=int, default=1, help="Merge top N P2Rank pockets when using predictions.csv.")
    parser.add_argument("--p2rank_rank", type=int, default=None, help="Select one P2Rank rank from predictions.csv.")
    parser.add_argument("--p2rank_name", default=None, help="Select one P2Rank pocket name from predictions.csv.")
    parser.add_argument("--p2rank_min_probability", type=float, default=None, help="Optional P2Rank probability cutoff.")
    parser.add_argument(
        "--fpocket_file",
        default=None,
        help="fpocket pocket atom PDB, output directory, or normalized residue-list file.",
    )
    parser.add_argument(
        "--ligand_file",
        default=None,
        help="Ligand/template PDB in the same coordinate frame as pdb_path.",
    )
    parser.add_argument(
        "--ligand_contact_threshold",
        type=float,
        default=4.5,
        help="Ligand-contact distance threshold in Angstrom.",
    )
    parser.add_argument(
        "--anchor_shell_radii",
        default="4,6,8",
        help="Comma-separated catalytic-anchor shell radii in Angstrom, e.g. 4,6,8.",
    )
    parser.add_argument(
        "--curated_min_support",
        type=float,
        default=1.20,
        help="Minimum weighted support score for a candidate without high-confidence evidence.",
    )
    parser.add_argument(
        "--external_overwide_max_residue_count",
        type=int,
        default=35,
        help="P2Rank/fpocket residue count above this absolute threshold is treated as overwide.",
    )
    parser.add_argument(
        "--external_overwide_max_fraction",
        type=float,
        default=0.18,
        help="P2Rank/fpocket residue fraction above this structure-relative threshold is treated as overwide.",
    )
    parser.add_argument(
        "--disable_external_precision_guard",
        action="store_true",
        help="Disable the guard that keeps overwide P2Rank/fpocket edge residues out of curated pocket candidates.",
    )
    return parser


def _sort_residue_key(key: str) -> tuple[str, int, str]:
    parts = [part.strip() for part in str(key).split(":")]
    chain = parts[0] if len(parts) >= 1 else ""
    try:
        resseq = int(parts[1]) if len(parts) >= 2 else 0
    except ValueError:
        resseq = 0
    icode = parts[2] if len(parts) >= 3 else ""
    return chain, resseq, icode


def _key_chain(key: str) -> str:
    return str(key).split(":", 1)[0].strip()


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_sanitize(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if np.isfinite(out) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _parse_float_list(text: str) -> list[float]:
    values: list[float] = []
    for token in str(text or "").split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if value <= 0:
            raise ValueError("Distance radii must be positive.")
        values.append(value)
    if not values:
        raise ValueError("At least one distance radius is required.")
    return sorted(set(values))


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


def _is_protein_residue(residue: Residue) -> bool:
    return str(residue.id[0]).strip() == ""


def _residue_chain_id(residue: Residue) -> str:
    parent = residue.get_parent()
    if isinstance(parent, Chain):
        return str(parent.id).strip() or "_"
    return "_"


def _canonical_key(residue: Residue) -> str:
    _, resseq, icode = residue.id
    return normalize_residue_key(_residue_chain_id(residue), int(resseq), str(icode).strip())


def _residue_metadata(residue: Residue) -> dict[str, Any]:
    _, resseq, icode = residue.id
    return {
        "chain_id": _residue_chain_id(residue),
        "resseq": int(resseq),
        "icode": str(icode).strip(),
        "resname": str(residue.get_resname()).strip(),
    }


def _residue_coord_index(
    structure: Structure,
    *,
    antigen_chain: str | None,
    include_non_antigen_residues: bool,
) -> tuple[dict[str, Residue], dict[str, np.ndarray]]:
    residues: dict[str, Residue] = {}
    coords: dict[str, np.ndarray] = {}
    for residue in _iter_residues(structure):
        if not _is_protein_residue(residue):
            continue
        key = _canonical_key(residue)
        if antigen_chain and not include_non_antigen_residues and _key_chain(key) != antigen_chain:
            continue
        atom_coords = extract_residue_atoms(residue)
        residues[key] = residue
        coords[key] = atom_coords
    return residues, coords


def _min_distance(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    if coords_a.shape[0] == 0 or coords_b.shape[0] == 0:
        return float("nan")
    diff = coords_a[:, None, :] - coords_b[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff, optimize=True)
    return float(np.sqrt(np.min(d2)))


def _filter_keys(
    keys: Iterable[str],
    *,
    antigen_chain: str | None,
    include_non_antigen_residues: bool,
) -> tuple[list[str], list[str]]:
    kept: list[str] = []
    skipped: list[str] = []
    for key in sorted(set(keys), key=_sort_residue_key):
        if antigen_chain and not include_non_antigen_residues and _key_chain(key) != antigen_chain:
            skipped.append(key)
        else:
            kept.append(key)
    return kept, skipped


def _add_residue_set_evidence(
    rows: list[EvidenceRow],
    residue_keys: Iterable[str],
    *,
    evidence_type: str,
    source: str,
    confidence: str,
    method: str,
    structure_id: str,
    source_path: str,
    weight: float,
    note: str,
    antigen_chain: str | None,
    include_non_antigen_residues: bool,
) -> dict[str, Any]:
    unique_keys = set(residue_keys)
    kept, skipped = _filter_keys(
        unique_keys,
        antigen_chain=antigen_chain,
        include_non_antigen_residues=include_non_antigen_residues,
    )
    for key in kept:
        rows.append(
            EvidenceRow(
                residue_key=key,
                evidence_type=evidence_type,
                source=source,
                confidence=confidence,
                method=method,
                structure_id=structure_id,
                source_path=source_path,
                weight=float(weight),
                note=note,
            )
        )
    return {
        "evidence_type": evidence_type,
        "source": source,
        "source_path": source_path,
        "input_residue_count": int(len(unique_keys)),
        "kept_residue_count": int(len(kept)),
        "skipped_non_antigen_count": int(len(skipped)),
        "skipped_non_antigen_residues": skipped[:50],
    }


def _load_residue_file(path_text: str | None) -> set[str]:
    if not path_text:
        return set()
    path = Path(path_text).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Residue evidence file not found: {path}")
    return set(load_residue_set(str(path)))


def _normalize_source_column(column: str) -> str:
    text = str(column).strip().lstrip("\ufeff").lower()
    normalized = "".join(ch if ch.isalnum() else "_" for ch in text)
    return "_".join(part for part in normalized.split("_") if part)


def _canonical_source_cells(raw: dict[str, Any]) -> dict[str, str]:
    normalized_raw = {_normalize_source_column(key): "" if value is None else str(value).strip() for key, value in raw.items()}
    cells: dict[str, str] = {column: "" for column in SOURCE_TEMPLATE_COLUMNS}
    cells.update({"chain_id": "", "resseq": "", "icode": ""})
    for canonical_name, aliases in SOURCE_TABLE_ALIASES.items():
        for alias in aliases:
            value = normalized_raw.get(_normalize_source_column(alias), "")
            if value:
                cells[canonical_name] = value
                break
    return cells


def _infer_chain_from_residue_token(token: str) -> str | None:
    text = str(token or "").strip().replace("：", ":")
    if ":" in text:
        first = text.split(":", 1)[0].strip()
        if first and not first.lstrip("-").isdigit():
            return first
    parts = [part for part in text.split() if part]
    if len(parts) >= 2 and not parts[0].lstrip("-").isdigit():
        return parts[0]
    return None


def _source_residue_keys_from_cells(cells: dict[str, str]) -> list[str]:
    residue_text = str(cells.get("residue_key") or "").strip()
    keys: set[str] = set()
    if residue_text:
        chunks: list[str] = []
        for raw_line in residue_text.replace(";", "\n").replace("|", "\n").splitlines():
            chunks.extend(segment.strip() for segment in raw_line.split(",") if segment.strip())
        default_chain: str | None = None
        for chunk in chunks:
            default_chain = _infer_chain_from_residue_token(chunk) or default_chain
            for key in parse_residue_token_or_range(chunk, default_chain=default_chain):
                keys.add(key)
        return sorted(keys, key=_sort_residue_key)

    chain_id = str(cells.get("chain_id") or "").strip()
    resseq = str(cells.get("resseq") or "").strip()
    icode = str(cells.get("icode") or "").strip()
    if chain_id and resseq:
        for key in parse_residue_token_or_range(f"{chain_id}:{resseq}", default_chain=chain_id):
            if icode and ":" not in key.split(":", 2)[-1]:
                chain, key_resseq, _ = _sort_residue_key(key)
                keys.add(normalize_residue_key(chain, key_resseq, icode))
            else:
                keys.add(key)
    return sorted(keys, key=_sort_residue_key)


def _load_source_audit_table(path_text: str | None) -> tuple[dict[str, list[dict[str, str]]], dict[str, Any], list[str]]:
    if not path_text:
        return {}, {"path": None, "row_count": 0, "mapped_row_count": 0, "mapped_residue_count": 0}, []

    path = Path(path_text).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Source audit table not found: {path}")

    sep = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)
    if len(df.columns) == 1 and sep == ",":
        # A TSV saved with .csv is common during manual curation.
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)

    mapped: dict[str, list[dict[str, str]]] = defaultdict(list)
    invalid_rows: list[int] = []
    for row_index, raw_row in df.iterrows():
        cells = _canonical_source_cells({str(col): raw_row[col] for col in df.columns})
        try:
            residue_keys = _source_residue_keys_from_cells(cells)
        except ValueError:
            residue_keys = []
        if not residue_keys:
            invalid_rows.append(int(row_index) + 2)
            continue
        normalized_row = {column: str(cells.get(column, "") or "").strip() for column in SOURCE_TEMPLATE_COLUMNS}
        for residue_key in residue_keys:
            item = dict(normalized_row)
            item["residue_key"] = residue_key
            mapped[residue_key].append(item)

    warnings: list[str] = []
    if invalid_rows:
        warnings.append(
            f"Source audit table {path} has rows without parseable residue_key/chain/resseq: {invalid_rows[:20]}"
        )
    summary = {
        "path": str(path),
        "row_count": int(len(df)),
        "mapped_row_count": int(sum(len(items) for items in mapped.values())),
        "mapped_residue_count": int(len(mapped)),
        "invalid_row_count": int(len(invalid_rows)),
        "invalid_rows": invalid_rows[:50],
    }
    return dict(mapped), summary, warnings


def _blank_source_row(*, residue_key: str, evidence_role: str) -> dict[str, str]:
    row = {column: "" for column in SOURCE_TEMPLATE_COLUMNS}
    row["residue_key"] = residue_key
    row["evidence_role"] = evidence_role
    row["review_status"] = "unreviewed"
    return row


def _source_row_has_traceable_reference(row: dict[str, Any]) -> bool:
    return any(str(row.get(field, "") or "").strip() for field in TRACEABLE_SOURCE_FIELDS)


def _build_source_audit(
    source_specs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[str]]:
    audit_rows: list[dict[str, Any]] = []
    template_rows: list[dict[str, Any]] = []
    table_summaries: list[dict[str, Any]] = []
    warnings: list[str] = []

    for spec in source_specs:
        evidence_role = str(spec.get("evidence_role") or "").strip()
        residue_keys = sorted(set(spec.get("residue_keys") or []), key=_sort_residue_key)
        source_residue_file = str(spec.get("source_residue_file") or "")
        source_table_path = str(spec.get("source_table") or "").strip()
        source_map, table_summary, table_warnings = _load_source_audit_table(source_table_path or None)
        table_summary["evidence_role"] = evidence_role
        table_summaries.append(table_summary)
        warnings.extend(table_warnings)

        for residue_key in residue_keys:
            matched_rows = source_map.get(residue_key) or []
            if not matched_rows:
                matched_rows = [_blank_source_row(residue_key=residue_key, evidence_role=evidence_role)]

            for matched_row in matched_rows:
                row = {column: str(matched_row.get(column, "") or "").strip() for column in SOURCE_TEMPLATE_COLUMNS}
                row["residue_key"] = residue_key
                if not row["evidence_role"]:
                    row["evidence_role"] = evidence_role
                if evidence_role == "ai_prior" and not row["source_kind"]:
                    row["source_kind"] = "ai_extraction"
                if not row["review_status"]:
                    row["review_status"] = "unreviewed"

                issues: list[str] = []
                if not source_table_path:
                    issues.append("missing_source_table")
                elif not source_map.get(residue_key):
                    issues.append("missing_source_row")

                has_traceable = _source_row_has_traceable_reference(row)
                if not has_traceable:
                    issues.append("missing_traceable_source")

                review_status = row["review_status"].strip().lower()
                if review_status in {"rejected", "exclude", "excluded", "false"}:
                    issues.append("rejected_source")
                elif review_status not in {"confirmed", "reviewed", "accepted", "curated"}:
                    issues.append("unreviewed_source")
                if evidence_role == "ai_prior":
                    if not str(row.get("source_sentence", "") or "").strip():
                        issues.append("ai_missing_source_sentence")
                    if not str(row.get("evidence_level", "") or "").strip():
                        issues.append("ai_missing_evidence_level")
                    if not str(row.get("review_status", "") or "").strip():
                        issues.append("ai_missing_human_review_status")
                    issues.append("ai_prior_not_ground_truth")

                audit_row = {
                    **row,
                    "source_table_path": source_table_path,
                    "source_residue_file": source_residue_file,
                    "has_traceable_source": bool(has_traceable),
                    "audit_issue": ";".join(sorted(set(issues))),
                }
                audit_rows.append(audit_row)
                template_rows.append({column: row.get(column, "") for column in SOURCE_TEMPLATE_COLUMNS})

    missing_source_table_count = sum("missing_source_table" in str(row.get("audit_issue", "")) for row in audit_rows)
    missing_source_row_count = sum("missing_source_row" in str(row.get("audit_issue", "")) for row in audit_rows)
    missing_traceable_count = sum("missing_traceable_source" in str(row.get("audit_issue", "")) for row in audit_rows)
    unreviewed_count = sum("unreviewed_source" in str(row.get("audit_issue", "")) for row in audit_rows)
    rejected_count = sum("rejected_source" in str(row.get("audit_issue", "")) for row in audit_rows)
    ai_prior_count = sum(str(row.get("evidence_role", "")) == "ai_prior" for row in audit_rows)
    ai_missing_sentence_count = sum("ai_missing_source_sentence" in str(row.get("audit_issue", "")) for row in audit_rows)
    ai_missing_level_count = sum("ai_missing_evidence_level" in str(row.get("audit_issue", "")) for row in audit_rows)
    ai_unreviewed_count = sum(
        str(row.get("evidence_role", "")) == "ai_prior"
        and "unreviewed_source" in str(row.get("audit_issue", ""))
        for row in audit_rows
    )

    if missing_traceable_count:
        warnings.append(
            "Source audit: "
            f"{missing_traceable_count} source rows lack PMID/DOI/UniProt/M-CSA/PDB/title/source sentence."
        )
    if unreviewed_count:
        warnings.append(f"Source audit: {unreviewed_count} source rows are not marked confirmed/reviewed.")
    if ai_prior_count:
        warnings.append(
            f"AI prior audit: {ai_prior_count} rows are evidence-only residue priors and are never treated as ground truth."
        )
    if ai_missing_sentence_count or ai_missing_level_count:
        warnings.append(
            "AI prior audit: "
            f"{ai_missing_sentence_count} rows lack source_sentence; "
            f"{ai_missing_level_count} rows lack evidence_level."
        )

    summary = {
        "enabled": bool(source_specs),
        "audit_row_count": int(len(audit_rows)),
        "template_row_count": int(len(template_rows)),
        "traceable_row_count": int(sum(bool(row.get("has_traceable_source")) for row in audit_rows)),
        "missing_source_table_count": int(missing_source_table_count),
        "missing_source_row_count": int(missing_source_row_count),
        "missing_traceable_source_count": int(missing_traceable_count),
        "unreviewed_source_count": int(unreviewed_count),
        "rejected_source_count": int(rejected_count),
        "ai_prior_row_count": int(ai_prior_count),
        "ai_missing_source_sentence_count": int(ai_missing_sentence_count),
        "ai_missing_evidence_level_count": int(ai_missing_level_count),
        "ai_unreviewed_source_count": int(ai_unreviewed_count),
        "table_summaries": table_summaries,
    }
    return audit_rows, template_rows, summary, warnings


def _parse_p2rank_residue_token(token: str) -> str:
    text = str(token).strip()
    if not text:
        raise ValueError("Empty P2Rank residue token.")
    if "_" not in text:
        raise ValueError(f"Unsupported P2Rank residue token: {token!r}")
    parts = [part for part in text.split("_") if part != ""]
    if len(parts) < 2:
        raise ValueError(f"Unsupported P2Rank residue token: {token!r}")
    chain_id = parts[0]
    resseq = int(parts[1])
    icode = parts[2] if len(parts) >= 3 and len(parts[2]) == 1 and not parts[2].isdigit() else ""
    return normalize_residue_key(chain_id, resseq, icode)


def _load_p2rank_keys(
    path_text: str,
    *,
    top_n: int,
    rank: int | None,
    name: str | None,
    min_probability: float | None,
) -> tuple[set[str], dict[str, Any]]:
    path = Path(path_text).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"P2Rank file not found: {path}")

    if path.is_file():
        try:
            df = pd.read_csv(path, skipinitialspace=True)
            df.columns = [str(col).strip() for col in df.columns]
        except Exception:
            df = pd.DataFrame()
        if {"rank", "residue_ids"}.issubset(set(df.columns)):
            work = df.copy()
            if min_probability is not None and "probability" in work.columns:
                probs = pd.to_numeric(work["probability"], errors="coerce")
                work = work.loc[probs >= float(min_probability)].reset_index(drop=True)
            if rank is not None:
                ranks = pd.to_numeric(work["rank"], errors="coerce")
                work = work.loc[ranks == int(rank)].reset_index(drop=True)
            if name:
                work = work.loc[work["name"].astype(str).str.strip() == str(name).strip()].reset_index(drop=True)
            if work.empty:
                return set(), {
                    "source_path": str(path),
                    "parser": "p2rank_predictions_csv",
                    "selected_pockets": [],
                    "warning": "No P2Rank pockets left after filtering.",
                }
            if rank is None and not name:
                work = work.sort_values(by="rank", ascending=True).head(max(1, int(top_n))).reset_index(drop=True)

            residue_keys: set[str] = set()
            selected_pockets: list[dict[str, Any]] = []
            for _, row in work.iterrows():
                pocket_keys: list[str] = []
                for token in str(row["residue_ids"]).split():
                    key = _parse_p2rank_residue_token(token)
                    residue_keys.add(key)
                    pocket_keys.append(key)
                selected_pockets.append(
                    {
                        "name": str(row.get("name", "")).strip(),
                        "rank": int(pd.to_numeric(pd.Series([row.get("rank")]), errors="coerce").fillna(-1).iloc[0]),
                        "score": float(pd.to_numeric(pd.Series([row.get("score")]), errors="coerce").iloc[0])
                        if "score" in row.index
                        else None,
                        "probability": float(
                            pd.to_numeric(pd.Series([row.get("probability")]), errors="coerce").iloc[0]
                        )
                        if "probability" in row.index
                        else None,
                        "residue_count": int(len(set(pocket_keys))),
                    }
                )
            return residue_keys, {
                "source_path": str(path),
                "parser": "p2rank_predictions_csv",
                "selected_pockets": selected_pockets,
            }

    residue_keys = set(parse_p2rank_output(str(path)))
    return residue_keys, {
        "source_path": str(path),
        "parser": "generic_p2rank_or_residue_list",
        "selected_pockets": [],
    }


def _looks_like_pdb(path: Path) -> bool:
    if path.suffix.lower() != ".pdb":
        return False
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[:200]:
            if line[:6].strip().upper() in {"ATOM", "HETATM"}:
                return True
    except Exception:
        return False
    return False


def _load_fpocket_keys(path_text: str, *, antigen_chain: str | None) -> tuple[set[str], dict[str, Any]]:
    path = Path(path_text).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"fpocket file not found: {path}")
    if path.is_file() and _looks_like_pdb(path):
        keys, summary = extract_fpocket_residue_keys(path, chain_filter=antigen_chain, include_hetatm=False)
        summary["parser"] = "fpocket_pocket_pdb"
        return set(keys), summary
    keys = set(parse_fpocket_output(str(path)))
    return keys, {
        "source_path": str(path),
        "parser": "generic_fpocket_or_residue_list",
    }


def _add_ligand_contact_evidence(
    rows: list[EvidenceRow],
    *,
    ligand_file: str,
    residue_coords: dict[str, np.ndarray],
    structure_id: str,
    threshold: float,
) -> dict[str, Any]:
    ligand = load_ligand_template_pdb(ligand_file)
    hit_count = 0
    distances: list[float] = []
    for key in sorted(residue_coords, key=_sort_residue_key):
        distance = _min_distance(residue_coords[key], ligand.coordinates)
        if not np.isfinite(distance) or distance > threshold:
            continue
        hit_count += 1
        distances.append(float(distance))
        rows.append(
            EvidenceRow(
                residue_key=key,
                evidence_type="ligand_contact",
                source="ligand_template",
                confidence="high",
                method="structure_distance",
                structure_id=structure_id,
                source_path=str(Path(ligand_file).expanduser().resolve()),
                weight=0.75,
                distance_angstrom=float(distance),
                note=f"Residue heavy atom within {threshold:g} A of ligand/template.",
            )
        )
    return {
        "evidence_type": "ligand_contact",
        "source_path": str(Path(ligand_file).expanduser().resolve()),
        "ligand_atom_count": int(ligand.coordinates.shape[0]),
        "distance_threshold_angstrom": float(threshold),
        "kept_residue_count": int(hit_count),
        "min_distance_angstrom": float(min(distances)) if distances else None,
    }


def _add_catalytic_anchor_evidence(
    rows: list[EvidenceRow],
    *,
    catalytic_file: str,
    catalytic_keys: set[str],
    structure: Structure,
    residue_coords: dict[str, np.ndarray],
    structure_id: str,
    antigen_chain: str | None,
    include_non_antigen_residues: bool,
    shell_radii: list[float],
) -> dict[str, Any]:
    source_path = str(Path(catalytic_file).expanduser().resolve())
    core_summary = _add_residue_set_evidence(
        rows,
        catalytic_keys,
        evidence_type="catalytic_core",
        source="catalytic_or_literature_anchor",
        confidence="high",
        method="curated_residue_file",
        structure_id=structure_id,
        source_path=source_path,
        weight=0.90,
        note="Catalytic/literature anchor residue.",
        antigen_chain=antigen_chain,
        include_non_antigen_residues=include_non_antigen_residues,
    )

    kept_core_keys, _ = _filter_keys(
        catalytic_keys,
        antigen_chain=antigen_chain,
        include_non_antigen_residues=include_non_antigen_residues,
    )
    match = match_residues_in_structure(structure, kept_core_keys, allow_loose_match=True)
    anchor_coords = match.atom_coordinates
    if anchor_coords.shape[0] == 0:
        return {
            **core_summary,
            "matched_anchor_count": 0,
            "anchor_shell_count": 0,
            "warnings": ["No catalytic anchors matched structure coordinates; shell evidence was skipped."],
        }

    radii = sorted(shell_radii)
    shell_weights = _shell_weights(radii)
    shell_count = 0
    shell_distance_values: list[float] = []
    kept_core_set = set(kept_core_keys)
    for key in sorted(residue_coords, key=_sort_residue_key):
        if key in kept_core_set:
            continue
        distance = _min_distance(residue_coords[key], anchor_coords)
        if not np.isfinite(distance):
            continue
        shell_radius = next((radius for radius in radii if distance <= radius), None)
        if shell_radius is None:
            continue
        evidence_type = f"catalytic_anchor_shell_{shell_radius:g}A"
        rows.append(
            EvidenceRow(
                residue_key=key,
                evidence_type=evidence_type,
                source="catalytic_anchor_shell",
                confidence="medium" if shell_radius <= 6.0 else "low",
                method="structure_distance",
                structure_id=structure_id,
                source_path=source_path,
                weight=float(shell_weights[shell_radius]),
                distance_angstrom=float(distance),
                note=f"Nearest heavy atom is within {shell_radius:g} A of a catalytic/literature anchor.",
            )
        )
        shell_count += 1
        shell_distance_values.append(float(distance))

    warnings: list[str] = []
    if len(kept_core_keys) > 12:
        warnings.append(
            "Catalytic/anchor file contains more than 12 residues; it may be a broad pocket file rather than true catalytic anchors."
        )
    if shell_count > max(25, len(residue_coords) * 0.25):
        warnings.append("Catalytic anchor shell is broad; inspect candidate_curated_pocket.txt before treating it as precise.")

    return {
        **core_summary,
        "matched_anchor_count": int(match.summary.get("matched_key_count", 0)),
        "unmatched_anchor_count": int(match.summary.get("unmatched_key_count", 0)),
        "anchor_shell_count": int(shell_count),
        "shell_radii_angstrom": radii,
        "min_shell_distance_angstrom": float(min(shell_distance_values)) if shell_distance_values else None,
        "warnings": warnings,
    }


def _shell_weights(radii: list[float]) -> dict[float, float]:
    weights: dict[float, float] = {}
    for idx, radius in enumerate(sorted(radii)):
        if idx == 0:
            weights[radius] = 0.65
        elif idx == 1:
            weights[radius] = 0.45
        else:
            weights[radius] = 0.25
    return weights


def _build_external_precision_guard(
    evidence_rows: list[EvidenceRow],
    *,
    structure_residue_count: int,
    max_residue_count: int,
    max_fraction: float,
    enabled: bool,
) -> tuple[set[str], dict[str, Any], list[str]]:
    threshold_count = max(int(max_residue_count), int(np.ceil(max(0.0, float(max_fraction)) * structure_residue_count)))
    source_to_keys: dict[str, set[str]] = defaultdict(set)
    grouped: dict[str, list[EvidenceRow]] = defaultdict(list)
    for row in evidence_rows:
        grouped[row.residue_key].append(row)
        source = str(row.source or "").strip().lower()
        if source in EXTERNAL_POCKET_SOURCES:
            source_to_keys[source].add(row.residue_key)

    source_counts = {source: int(len(keys)) for source, keys in sorted(source_to_keys.items())}
    overwide_sources = sorted([source for source, keys in source_to_keys.items() if len(keys) > threshold_count])
    guarded_keys: set[str] = set()
    if enabled and overwide_sources:
        overwide_set = set(overwide_sources)
        for residue_key, rows in grouped.items():
            has_overwide_external = any(str(row.source or "").strip().lower() in overwide_set for row in rows)
            if not has_overwide_external:
                continue
            has_trusted_curated = any(row.evidence_type in TRUSTED_CURATED_EVIDENCE_TYPES for row in rows)
            if not has_trusted_curated:
                guarded_keys.add(residue_key)

    warnings: list[str] = []
    if enabled and overwide_sources:
        warnings.append(
            "External precision guard: "
            f"{', '.join(overwide_sources)} predicted more than {threshold_count} residues; "
            f"{len(guarded_keys)} low-confidence edge residues were kept in review instead of curated candidates."
        )

    summary = {
        "enabled": bool(enabled),
        "structure_residue_count": int(structure_residue_count),
        "max_residue_count": int(max_residue_count),
        "max_fraction": float(max_fraction),
        "threshold_count": int(threshold_count),
        "source_residue_counts": source_counts,
        "overwide_sources": [
            {
                "source": source,
                "residue_count": int(source_counts.get(source, 0)),
                "threshold_count": int(threshold_count),
            }
            for source in overwide_sources
        ],
        "guarded_residue_count": int(len(guarded_keys)),
        "guarded_residues_preview": sorted(guarded_keys, key=_sort_residue_key)[:200],
    }
    return guarded_keys, summary, warnings


def _aggregate_support(
    evidence_rows: list[EvidenceRow],
    *,
    structure_residues: dict[str, Residue],
    curated_min_support: float,
    precision_guarded_keys: set[str] | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[EvidenceRow]] = defaultdict(list)
    for row in evidence_rows:
        grouped[row.residue_key].append(row)

    support_rows: list[dict[str, Any]] = []
    guarded_keys = precision_guarded_keys or set()
    for residue_key in sorted(grouped, key=_sort_residue_key):
        rows = grouped[residue_key]
        methods = sorted({row.method for row in rows if row.method})
        evidence_types = sorted({row.evidence_type for row in rows if row.evidence_type})
        sources = sorted({row.source for row in rows if row.source})
        weighted_support = float(sum(float(row.weight) for row in rows))
        non_ai_rows = [row for row in rows if row.evidence_type != "ai_prior"]
        non_ai_weighted_support = float(sum(float(row.weight) for row in non_ai_rows))
        high_conf_count = sum(1 for row in rows if row.confidence.lower() == "high" or row.weight >= 0.75)
        distances = [float(row.distance_angstrom) for row in rows if row.distance_angstrom is not None and np.isfinite(row.distance_angstrom)]
        max_weight = float(max(float(row.weight) for row in rows)) if rows else 0.0
        method_count = len(methods)
        non_ai_method_count = len({row.method for row in non_ai_rows if row.method})

        is_curated = bool(
            high_conf_count >= 1
            or non_ai_method_count >= 2
            or non_ai_weighted_support >= float(curated_min_support)
        )
        review_reasons: list[str] = []
        if high_conf_count == 0 and method_count == 1:
            review_reasons.append("single_low_or_medium_confidence_method")
        if any(row.evidence_type == "ai_prior" for row in rows) and high_conf_count == 0:
            review_reasons.append("ai_prior_requires_review")
        if evidence_types and all(item.startswith("catalytic_anchor_shell_") for item in evidence_types):
            review_reasons.append("anchor_shell_only")
        if evidence_types == ["ai_prior"]:
            review_reasons.append("ai_prior_only")
        if residue_key in guarded_keys:
            review_reasons.append("external_overwide_guard")
            is_curated = False
        if not is_curated:
            review_reasons.append("below_curated_support_threshold")
        needs_review = bool(review_reasons)

        residue = structure_residues.get(residue_key)
        if residue is not None:
            meta = _residue_metadata(residue)
            present = True
        else:
            chain, resseq, icode = _sort_residue_key(residue_key)
            meta = {"chain_id": chain, "resseq": resseq, "icode": icode, "resname": ""}
            present = False
            review_reasons.append("not_found_in_structure")
            needs_review = True

        support_rows.append(
            {
                "residue_key": residue_key,
                **meta,
                "present_in_structure": bool(present),
                "weighted_support_score": round(weighted_support, 6),
                "evidence_count": int(len(rows)),
                "method_count": int(method_count),
                "high_confidence_source_count": int(high_conf_count),
                "max_weight": round(max_weight, 6),
                "min_distance_angstrom": round(float(min(distances)), 6) if distances else "",
                "evidence_types": ";".join(evidence_types),
                "methods": ";".join(methods),
                "sources": ";".join(sources),
                "is_curated_candidate": bool(is_curated),
                "needs_review": bool(needs_review),
                "review_reason": ";".join(sorted(set(review_reasons))),
            }
        )
    return sorted(
        support_rows,
        key=lambda row: (
            -float(row["weighted_support_score"]),
            _sort_residue_key(str(row["residue_key"])),
        ),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(_json_sanitize(row))


def _write_residue_list(path: Path, residue_keys: Iterable[str]) -> None:
    keys = sorted(set(residue_keys), key=_sort_residue_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(keys) + ("\n" if keys else ""), encoding="utf-8")


def _write_report(
    path: Path,
    *,
    summary: dict[str, Any],
    support_rows: list[dict[str, Any]],
    evidence_counts: Counter[str],
) -> None:
    top_rows = support_rows[:30]
    lines: list[str] = []
    lines.append("# Pocket Evidence Report")
    lines.append("")
    lines.append("This report combines residue-level pocket evidence. It does not train or change the ranking model.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- PDB: `{summary['pdb_path']}`")
    lines.append(f"- Antigen chain: `{summary['antigen_chain']}`")
    lines.append(f"- Evidence rows: {summary['evidence_row_count']}")
    lines.append(f"- Supported residues: {summary['supported_residue_count']}")
    lines.append(f"- Curated candidates: {summary['curated_candidate_count']}")
    lines.append(f"- Review residues: {summary['review_residue_count']}")
    lines.append("")
    lines.append("## Evidence Counts")
    lines.append("")
    if evidence_counts:
        for name, count in sorted(evidence_counts.items()):
            lines.append(f"- `{name}`: {count}")
    else:
        lines.append("- No evidence rows were generated.")
    warnings = summary.get("warnings") or []
    if warnings:
        lines.append("")
        lines.append("## Warnings")
        lines.append("")
        for warning in warnings:
            lines.append(f"- {warning}")
    source_audit = summary.get("source_audit") if isinstance(summary.get("source_audit"), dict) else {}
    if source_audit and source_audit.get("enabled"):
        lines.append("")
        lines.append("## Literature/Catalytic Source Audit")
        lines.append("")
        lines.append(f"- Audit rows: {source_audit.get('audit_row_count', 0)}")
        lines.append(f"- Traceable rows: {source_audit.get('traceable_row_count', 0)}")
        lines.append(f"- Missing source table rows: {source_audit.get('missing_source_table_count', 0)}")
        lines.append(f"- Missing residue source rows: {source_audit.get('missing_source_row_count', 0)}")
        lines.append(f"- Missing traceable source rows: {source_audit.get('missing_traceable_source_count', 0)}")
        lines.append(f"- Unreviewed source rows: {source_audit.get('unreviewed_source_count', 0)}")
        lines.append(f"- AI prior audit rows: {source_audit.get('ai_prior_row_count', 0)}")
        lines.append(f"- AI missing source sentence rows: {source_audit.get('ai_missing_source_sentence_count', 0)}")
        lines.append(f"- AI missing evidence level rows: {source_audit.get('ai_missing_evidence_level_count', 0)}")
        lines.append("")
        lines.append(
            "Use `evidence_source_template.csv` to fill paper/PMID/DOI/UniProt/M-CSA/source sentence and review status, "
            "then pass it back with `--literature_source_table` or `--catalytic_source_table`."
        )
        if int(source_audit.get("ai_prior_row_count", 0) or 0) > 0:
            lines.append(
                "AI prior rows must keep source_sentence, evidence_level and human review status. "
                "They remain evidence-only priors and do not become ground truth unless converted to manual/literature evidence."
            )
    precision_guard = (
        summary.get("external_precision_guard")
        if isinstance(summary.get("external_precision_guard"), dict)
        else {}
    )
    if precision_guard and precision_guard.get("enabled"):
        lines.append("")
        lines.append("## External Precision Guard")
        lines.append("")
        lines.append(f"- Threshold count: {precision_guard.get('threshold_count', 0)}")
        lines.append(f"- Source residue counts: `{precision_guard.get('source_residue_counts', {})}`")
        overwide_sources = precision_guard.get("overwide_sources") or []
        if overwide_sources:
            labels = [
                f"{item.get('source')}={item.get('residue_count')}"
                for item in overwide_sources
                if isinstance(item, dict)
            ]
            lines.append(f"- Overwide sources: `{', '.join(labels)}`")
        else:
            lines.append("- Overwide sources: none")
        lines.append(f"- Guarded residues: {precision_guard.get('guarded_residue_count', 0)}")
        lines.append(
            "Guarded residues remain in `pocket_residue_support.csv` and `review_residues.txt`, "
            "but are excluded from `candidate_curated_pocket.txt` unless supported by trusted curated evidence."
        )
    lines.append("")
    lines.append("## Top Supported Residues")
    lines.append("")
    if top_rows:
        lines.append("| residue | resname | support | evidence | methods | review |")
        lines.append("|---|---:|---:|---:|---|---|")
        for row in top_rows:
            lines.append(
                "| "
                f"{row['residue_key']} | "
                f"{row['resname']} | "
                f"{row['weighted_support_score']} | "
                f"{row['evidence_count']} | "
                f"{row['methods']} | "
                f"{row['review_reason']} |"
            )
    else:
        lines.append("No residue support rows were generated.")
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    for label, output_path in summary.get("output_files", {}).items():
        lines.append(f"- `{label}`: `{output_path}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_pocket_evidence(args: argparse.Namespace) -> dict[str, Any]:
    pdb_path = Path(args.pdb_path).expanduser().resolve()
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB not found: {pdb_path}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    antigen_chain = str(args.antigen_chain).strip() if args.antigen_chain else None
    structure = load_complex_pdb(str(pdb_path))
    structure_id = pdb_path.stem
    shell_radii = _parse_float_list(args.anchor_shell_radii)
    residue_map, residue_coords = _residue_coord_index(
        structure,
        antigen_chain=antigen_chain,
        include_non_antigen_residues=bool(args.include_non_antigen_residues),
    )
    if not residue_map:
        raise ValueError("No protein residues were available after applying antigen-chain filtering.")

    evidence_rows: list[EvidenceRow] = []
    source_summaries: list[dict[str, Any]] = []
    source_audit_specs: list[dict[str, Any]] = []
    warnings: list[str] = []

    if args.manual_pocket_file:
        keys = _load_residue_file(args.manual_pocket_file)
        source_summaries.append(
            _add_residue_set_evidence(
                evidence_rows,
                keys,
                evidence_type="manual_pocket",
                source="manual_or_rsite",
                confidence="high",
                method="curated_residue_file",
                structure_id=structure_id,
                source_path=str(Path(args.manual_pocket_file).expanduser().resolve()),
                weight=1.00,
                note="Manual/rsite pocket residue.",
                antigen_chain=antigen_chain,
                include_non_antigen_residues=bool(args.include_non_antigen_residues),
            )
        )

    if args.literature_file:
        keys = _load_residue_file(args.literature_file)
        kept_keys, _ = _filter_keys(
            keys,
            antigen_chain=antigen_chain,
            include_non_antigen_residues=bool(args.include_non_antigen_residues),
        )
        source_audit_specs.append(
            {
                "evidence_role": "literature",
                "residue_keys": kept_keys,
                "source_residue_file": str(Path(args.literature_file).expanduser().resolve()),
                "source_table": getattr(args, "literature_source_table", None),
            }
        )
        source_summaries.append(
            _add_residue_set_evidence(
                evidence_rows,
                keys,
                evidence_type="literature_residue",
                source="literature_curated",
                confidence="high",
                method="curated_residue_file",
                structure_id=structure_id,
                source_path=str(Path(args.literature_file).expanduser().resolve()),
                weight=0.95,
                note="Literature-curated functional/pocket residue.",
                antigen_chain=antigen_chain,
                include_non_antigen_residues=bool(args.include_non_antigen_residues),
            )
        )

    if args.catalytic_file:
        catalytic_keys = _load_residue_file(args.catalytic_file)
        kept_keys, _ = _filter_keys(
            catalytic_keys,
            antigen_chain=antigen_chain,
            include_non_antigen_residues=bool(args.include_non_antigen_residues),
        )
        source_audit_specs.append(
            {
                "evidence_role": "catalytic",
                "residue_keys": kept_keys,
                "source_residue_file": str(Path(args.catalytic_file).expanduser().resolve()),
                "source_table": getattr(args, "catalytic_source_table", None),
            }
        )
        catalytic_summary = _add_catalytic_anchor_evidence(
            evidence_rows,
            catalytic_file=args.catalytic_file,
            catalytic_keys=catalytic_keys,
            structure=structure,
            residue_coords=residue_coords,
            structure_id=structure_id,
            antigen_chain=antigen_chain,
            include_non_antigen_residues=bool(args.include_non_antigen_residues),
            shell_radii=shell_radii,
        )
        source_summaries.append(catalytic_summary)
        warnings.extend(catalytic_summary.get("warnings") or [])

    if args.ai_pocket_file:
        keys = _load_residue_file(args.ai_pocket_file)
        kept_keys, _ = _filter_keys(
            keys,
            antigen_chain=antigen_chain,
            include_non_antigen_residues=bool(args.include_non_antigen_residues),
        )
        source_audit_specs.append(
            {
                "evidence_role": "ai_prior",
                "residue_keys": kept_keys,
                "source_residue_file": str(Path(args.ai_pocket_file).expanduser().resolve()),
                "source_table": getattr(args, "ai_source_table", None),
            }
        )
        source_summaries.append(
            _add_residue_set_evidence(
                evidence_rows,
                keys,
                evidence_type="ai_prior",
                source="ai_assisted_prior",
                confidence="medium",
                method="ai_or_llm_extracted_residue_file",
                structure_id=structure_id,
                source_path=str(Path(args.ai_pocket_file).expanduser().resolve()),
                weight=0.45,
                note="AI-assisted prior; use as supporting evidence, not ground truth.",
                antigen_chain=antigen_chain,
                include_non_antigen_residues=bool(args.include_non_antigen_residues),
            )
        )

    if args.p2rank_file:
        p2rank_keys, p2rank_summary = _load_p2rank_keys(
            args.p2rank_file,
            top_n=max(1, int(args.p2rank_top_n)),
            rank=args.p2rank_rank,
            name=args.p2rank_name,
            min_probability=args.p2rank_min_probability,
        )
        p2rank_summary.update(
            _add_residue_set_evidence(
                evidence_rows,
                p2rank_keys,
                evidence_type="p2rank_prediction",
                source="p2rank",
                confidence="medium",
                method="external_pocket_tool",
                structure_id=structure_id,
                source_path=str(Path(args.p2rank_file).expanduser().resolve()),
                weight=0.55,
                note="P2Rank predicted pocket residue.",
                antigen_chain=antigen_chain,
                include_non_antigen_residues=bool(args.include_non_antigen_residues),
            )
        )
        source_summaries.append(p2rank_summary)

    if args.fpocket_file:
        fpocket_keys, fpocket_summary = _load_fpocket_keys(args.fpocket_file, antigen_chain=antigen_chain)
        fpocket_summary.update(
            _add_residue_set_evidence(
                evidence_rows,
                fpocket_keys,
                evidence_type="fpocket_prediction",
                source="fpocket",
                confidence="medium",
                method="external_pocket_tool",
                structure_id=structure_id,
                source_path=str(Path(args.fpocket_file).expanduser().resolve()),
                weight=0.50,
                note="fpocket predicted pocket residue.",
                antigen_chain=antigen_chain,
                include_non_antigen_residues=bool(args.include_non_antigen_residues),
            )
        )
        source_summaries.append(fpocket_summary)

    if args.ligand_file:
        source_summaries.append(
            _add_ligand_contact_evidence(
                evidence_rows,
                ligand_file=args.ligand_file,
                residue_coords=residue_coords,
                structure_id=structure_id,
                threshold=max(0.0, float(args.ligand_contact_threshold)),
            )
        )

    precision_guarded_keys, precision_guard_summary, precision_guard_warnings = _build_external_precision_guard(
        evidence_rows,
        structure_residue_count=len(residue_map),
        max_residue_count=max(1, int(getattr(args, "external_overwide_max_residue_count", 35))),
        max_fraction=max(0.0, float(getattr(args, "external_overwide_max_fraction", 0.18))),
        enabled=not bool(getattr(args, "disable_external_precision_guard", False)),
    )
    warnings.extend(precision_guard_warnings)

    evidence_dict_rows = [asdict(row) for row in evidence_rows]
    support_rows = _aggregate_support(
        evidence_rows,
        structure_residues=residue_map,
        curated_min_support=float(args.curated_min_support),
        precision_guarded_keys=precision_guarded_keys,
    )
    curated_keys = [row["residue_key"] for row in support_rows if row["is_curated_candidate"]]
    review_keys = [row["residue_key"] for row in support_rows if row["needs_review"]]
    evidence_counts = Counter(row.evidence_type for row in evidence_rows)
    source_audit_rows, source_template_rows, source_audit_summary, source_audit_warnings = _build_source_audit(
        source_audit_specs
    )
    warnings.extend(source_audit_warnings)

    output_files = {
        "pocket_evidence_csv": str(out_dir / "pocket_evidence.csv"),
        "pocket_residue_support_csv": str(out_dir / "pocket_residue_support.csv"),
        "candidate_curated_pocket_txt": str(out_dir / "candidate_curated_pocket.txt"),
        "review_residues_txt": str(out_dir / "review_residues.txt"),
        "evidence_source_audit_csv": str(out_dir / "evidence_source_audit.csv"),
        "evidence_source_template_csv": str(out_dir / "evidence_source_template.csv"),
        "ai_prior_audit_csv": str(out_dir / "ai_prior_audit.csv"),
        "ai_prior_template_csv": str(out_dir / "ai_prior_template.csv"),
        "summary_json": str(out_dir / "pocket_evidence_summary.json"),
        "report_md": str(out_dir / "POCKET_EVIDENCE_REPORT.md"),
    }

    summary = {
        "pdb_path": str(pdb_path),
        "structure_id": structure_id,
        "antigen_chain": antigen_chain,
        "include_non_antigen_residues": bool(args.include_non_antigen_residues),
        "structure_residue_count": int(len(residue_map)),
        "evidence_row_count": int(len(evidence_rows)),
        "supported_residue_count": int(len(support_rows)),
        "curated_candidate_count": int(len(curated_keys)),
        "review_residue_count": int(len(review_keys)),
        "curated_min_support": float(args.curated_min_support),
        "anchor_shell_radii_angstrom": shell_radii,
        "evidence_counts": dict(evidence_counts),
        "source_summaries": source_summaries,
        "source_audit": source_audit_summary,
        "external_precision_guard": precision_guard_summary,
        "warnings": sorted(set(warnings)),
        "output_files": output_files,
    }

    _write_csv(out_dir / "pocket_evidence.csv", evidence_dict_rows, EVIDENCE_COLUMNS)
    _write_csv(out_dir / "pocket_residue_support.csv", support_rows, SUPPORT_COLUMNS)
    _write_csv(out_dir / "evidence_source_audit.csv", source_audit_rows, SOURCE_AUDIT_COLUMNS)
    _write_csv(out_dir / "evidence_source_template.csv", source_template_rows, SOURCE_TEMPLATE_COLUMNS)
    _write_csv(
        out_dir / "ai_prior_audit.csv",
        [row for row in source_audit_rows if row.get("evidence_role") == "ai_prior"],
        SOURCE_AUDIT_COLUMNS,
    )
    _write_csv(
        out_dir / "ai_prior_template.csv",
        [row for row in source_template_rows if row.get("evidence_role") == "ai_prior"],
        SOURCE_TEMPLATE_COLUMNS,
    )
    _write_residue_list(out_dir / "candidate_curated_pocket.txt", curated_keys)
    _write_residue_list(out_dir / "review_residues.txt", review_keys)
    (out_dir / "pocket_evidence_summary.json").write_text(
        json.dumps(_json_sanitize(summary), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    _write_report(
        out_dir / "POCKET_EVIDENCE_REPORT.md",
        summary=summary,
        support_rows=support_rows,
        evidence_counts=evidence_counts,
    )

    return summary


def main() -> None:
    args = _build_parser().parse_args()
    summary = build_pocket_evidence(args)
    print(f"Saved: {summary['output_files']['pocket_evidence_csv']}")
    print(f"Saved: {summary['output_files']['pocket_residue_support_csv']}")
    print(f"Saved: {summary['output_files']['candidate_curated_pocket_txt']}")
    print(f"Saved: {summary['output_files']['report_md']}")
    print(
        "Pocket evidence: "
        f"{summary['evidence_row_count']} rows, "
        f"{summary['curated_candidate_count']} curated candidates, "
        f"{summary['review_residue_count']} review residues."
    )
    for warning in summary.get("warnings") or []:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
