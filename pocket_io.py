"""Pocket and catalytic residue I/O utilities.

This module provides robust parsers for pocket/catalytic residue definition
files and optional ligand-template PDB files. It is designed as a reusable
preprocessing component and does not include any model or training logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import StringIO
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol
import re

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Atom import Atom
from Bio.PDB.Atom import DisorderedAtom
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure


_RESSEQ_ICODE_RE = re.compile(r"^(?P<resseq>[+-]?\d+)(?P<icode>[A-Za-z]?)$")
_RESSEQ_RANGE_RE = re.compile(r"^(?P<start>[+-]?\d+)\s*-\s*(?P<end>[+-]?\d+)$")
_BLANK_CHAIN_TOKEN = "_"
_EXTERNAL_OUTPUT_SUFFIXES = frozenset({".txt", ".csv", ".tsv", ".json", ".res", ".lst", ".list", ".out", ".dat"})
_EXTERNAL_TEXT_COMMENT_PREFIXES = ("#", ";", "//")
_CHAIN_RES_INLINE_RE = re.compile(
    r"(?P<chain>[A-Za-z0-9_])\s*[:\s]\s*(?P<resseq>[+-]?\d+)(?P<icode_a>[A-Za-z]?)(?:\s*[:\s]\s*(?P<icode_b>[A-Za-z]))?"
)


@dataclass(frozen=True, slots=True)
class ResidueKeySet:
    """Normalized residue key set from a text file.

    Attributes:
        source_path: Source file path.
        residue_keys: Normalized residue keys, e.g. "A:45" or "A:45:B".
    """

    source_path: str
    residue_keys: frozenset[str]


@dataclass(frozen=True, slots=True)
class LigandAtom:
    """Atom information extracted from a ligand/template PDB."""

    serial_number: int | None
    atom_name: str
    element: str
    coord: tuple[float, float, float]
    residue_name: str
    chain_id: str
    resseq: int
    icode: str


@dataclass(slots=True)
class LigandTemplate:
    """Ligand/template coordinates container.

    Attributes:
        source_path: Ligand template PDB file path.
        atoms: Parsed atom records.
        coordinates: Atom coordinates, shape (N, 3).
    """

    source_path: str
    atoms: tuple[LigandAtom, ...]
    coordinates: np.ndarray
    centroid: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class PocketSource:
    """Source metadata for pocket/catalytic definitions.

    Attributes:
        source_name: Source identifier, e.g. "manual", "pykvfinder", "fpocket", "p2rank".
        source_path: Optional source file path.
        label: Semantic label, e.g. "pocket" or "catalytic".
        metadata: Extra source metadata reserved for future integrations.
    """

    source_name: str
    source_path: str | None = None
    label: str = "pocket"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ResidueMatchDetail:
    """One requested residue key and its matching resolution details."""

    requested_key: str
    matched_key: str | None
    mode: str
    warning: str = ""


@dataclass(slots=True)
class PocketResidueMatchResult:
    """Standardized residue matching result for pocket/catalytic definitions."""

    requested_keys: tuple[str, ...]
    matched_keys: tuple[str, ...]
    unmatched_keys: tuple[str, ...]
    key_to_residues: dict[str, tuple[Residue, ...]]
    matched_residues: tuple[Residue, ...]
    residue_centroids: dict[str, tuple[float, float, float]]
    atom_coordinates: np.ndarray
    details: tuple[ResidueMatchDetail, ...] = ()
    warnings: tuple[str, ...] = ()
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ResidueMatchResult:
    """Mapping result between text residue keys and structure residues."""

    requested_keys: frozenset[str]
    matched_keys: frozenset[str]
    missing_keys: frozenset[str]
    key_to_residues: dict[str, tuple[Residue, ...]]
    matched_residues: tuple[Residue, ...]
    unmatched_keys: frozenset[str] = field(default_factory=frozenset)
    warnings: tuple[str, ...] = ()
    details: tuple[ResidueMatchDetail, ...] = ()
    residue_centroids: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    atom_coordinates: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float64)
    )
    summary: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.unmatched_keys and self.missing_keys:
            self.unmatched_keys = frozenset(self.missing_keys)


@dataclass(slots=True)
class PocketDefinitionData:
    """Unified pocket/catalytic/ligand definition payload."""

    pocket: ResidueKeySet | None
    catalytic: ResidueKeySet | None
    ligand_template: LigandTemplate | None


@dataclass(slots=True)
class PocketDefinition:
    """Unified pocket/catalytic definition with matched structural payload."""

    source: PocketSource
    residue_keys: frozenset[str]
    match_result: PocketResidueMatchResult | None = None
    matched_residues: tuple[Residue, ...] = ()
    atom_coordinates: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float64)
    )
    residue_centroids: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StructureResidueIndex:
    """Prebuilt indexes for robust residue-key matching against structures."""

    residues: tuple[Residue, ...]
    exact_key_index: dict[str, tuple[Residue, ...]]
    chain_resseq_index: dict[tuple[str, int], tuple[Residue, ...]]
    chain_resseq_icode_index: dict[tuple[str, int, str], tuple[Residue, ...]]
    chainless_resseq_index: dict[int, tuple[Residue, ...]]
    chainless_resseq_icode_index: dict[tuple[int, str], tuple[Residue, ...]]
    chain_ids: tuple[str, ...]


class ExternalPocketToolAdapter(Protocol):
    """Interface reserved for future external pocket-tool integrations."""

    tool_name: str

    def parse_output(self, output_path: str) -> set[str]:
        """Parse tool output and return normalized residue keys."""


def _normalize_chain_token(chain_id: Any) -> str:
    """Normalize chain ID to text token.

    Blank chains are represented as "_" to avoid ambiguous empty strings in
    downstream key-based matching.
    """
    text = str(chain_id).strip()
    if not text:
        return _BLANK_CHAIN_TOKEN
    if ":" in text:
        raise ValueError(f"Invalid chain ID {chain_id!r}: ':' is not allowed.")
    return text


def _normalize_icode(icode: Any) -> str:
    """Normalize insertion code to empty string or one-character code."""
    text = str(icode).strip()
    if text in {"", "_", "-"}:
        return ""
    if len(text) != 1:
        raise ValueError(f"Invalid insertion code {icode!r}: must be one character.")
    return text


def _parse_resseq_icode(raw: str, explicit_icode: str = "") -> tuple[int, str]:
    """Parse residue number and insertion code from textual token."""
    raw = raw.strip()
    if not raw:
        raise ValueError("Residue number is empty.")

    if explicit_icode:
        try:
            resseq = int(raw)
        except ValueError as exc:
            raise ValueError(f"Invalid residue number {raw!r}.") from exc
        return resseq, _normalize_icode(explicit_icode)

    match = _RESSEQ_ICODE_RE.match(raw)
    if not match:
        raise ValueError(
            f"Invalid residue token {raw!r}. Expected formats like '45' or '45A'."
        )

    resseq = int(match.group("resseq"))
    icode = _normalize_icode(match.group("icode"))
    return resseq, icode


def _normalize_residue_definition_text(value: Any) -> str:
    """Normalize common text variants in residue definition files."""
    return (
        str(value)
        .replace("\ufeff", "")
        .replace("：", ":")
        .replace("，", ",")
        .replace("–", "-")
        .replace("—", "-")
        .strip()
    )


def _expand_residue_range_token(raw: str) -> list[int] | None:
    """Expand a plain integer residue range token such as ``37-40``."""
    text = str(raw).strip()
    match = _RESSEQ_RANGE_RE.match(text)
    if not match:
        return None
    start = int(match.group("start"))
    end = int(match.group("end"))
    if start > end:
        raise ValueError(f"Invalid residue range {raw!r}: start must be <= end.")
    return list(range(start, end + 1))


def _looks_like_chain_token(raw: str) -> bool:
    text = str(raw).strip()
    return bool(text) and bool(re.fullmatch(r"[A-Za-z0-9_]+", text))


def _parse_residue_chain_reversed_token(token: str) -> tuple[str, ...] | None:
    """Parse residue:chain tokens such as ``147:B`` or ``82-88:B``."""
    text = _normalize_residue_definition_text(token)
    parts = [p.strip() for p in text.split(":") if p.strip()]
    if len(parts) != 2:
        return None

    res_token, chain = parts
    if not _looks_like_chain_token(chain):
        return None

    expanded = _expand_residue_range_token(res_token)
    if expanded is not None:
        return tuple(normalize_residue_key(chain, resseq) for resseq in expanded)

    try:
        resseq, icode = _parse_resseq_icode(res_token)
    except ValueError:
        return None
    return (normalize_residue_key(chain, resseq, icode),)


def normalize_residue_key(chain_id: Any, resseq: Any, icode: Any = "") -> str:
    """Normalize residue identity to a canonical key.

    Canonical key format:
    - Without insertion code: ``{chain}:{resseq}``
    - With insertion code: ``{chain}:{resseq}:{icode}``

    Examples:
    - ``A:45``
    - ``B:100:A``

    Args:
        chain_id: Chain identifier.
        resseq: Residue sequence number.
        icode: Insertion code (optional).

    Returns:
        Canonical residue key.
    """
    chain = _normalize_chain_token(chain_id)
    try:
        resseq_int = int(str(resseq).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid residue number {resseq!r}.") from exc

    icode_norm = _normalize_icode(icode)
    if icode_norm:
        return f"{chain}:{resseq_int}:{icode_norm}"
    return f"{chain}:{resseq_int}"


def parse_residue_token(token: str, default_chain: str | None = None) -> str:
    """Parse one residue token and return canonical residue key.

    Supported token formats include:
    - "A:45"
    - "45:A"
    - "A:45A"
    - "A:45:A"
    - "A 45"
    - "A 45A"
    - "45" or "45A" when ``default_chain`` is provided

    Use ``parse_residue_token_or_range`` when range syntax such as
    ``A:37-40`` should be accepted.
    """
    text = _normalize_residue_definition_text(token)
    if not text:
        raise ValueError("Empty residue entry.")

    if ":" in text:
        reversed_keys = _parse_residue_chain_reversed_token(text)
        if reversed_keys is not None and len(reversed_keys) == 1:
            return reversed_keys[0]
        parts = [p.strip() for p in text.split(":") if p.strip()]
        if len(parts) == 2:
            chain, res_token = parts
            resseq, icode = _parse_resseq_icode(res_token)
            return normalize_residue_key(chain, resseq, icode)
        if len(parts) == 3:
            chain, res_token, icode_token = parts
            resseq, icode = _parse_resseq_icode(res_token, explicit_icode=icode_token)
            return normalize_residue_key(chain, resseq, icode)
        raise ValueError(
            f"Invalid residue entry {text!r}. Use 'A:45', '45:A', 'A:45A', 'A:45:A', "
            "'A 45', or parse ranges via parse_residue_token_or_range."
        )

    ws_parts = [p for p in text.split() if p]
    if len(ws_parts) == 2:
        chain, res_token = ws_parts
        resseq, icode = _parse_resseq_icode(res_token)
        return normalize_residue_key(chain, resseq, icode)
    if len(ws_parts) > 2:
        raise ValueError(
            f"Invalid residue token {text!r}. Too many whitespace-separated fields."
        )

    if default_chain is None:
        raise ValueError(
            f"Residue entry {text!r} has no chain ID. "
            "Use formats like 'A:45', 'A 45', 'A:37-40', or provide a chain prefix earlier on the line."
        )

    resseq, icode = _parse_resseq_icode(text)
    return normalize_residue_key(default_chain, resseq, icode)


def parse_residue_token_or_range(token: str, default_chain: str | None = None) -> tuple[str, ...]:
    """Parse one residue token, expanding integer ranges when present.

    Supported range formats include ``A:37-40``, ``A 37-40`` and ``37-40``
    when ``default_chain`` is provided. ``37-40:A`` is also accepted for
    residue:chain legacy files. Ranges use inclusive endpoints and do not
    support insertion codes.
    """
    text = _normalize_residue_definition_text(token)
    if not text:
        raise ValueError("Empty residue entry.")

    if ":" in text:
        reversed_keys = _parse_residue_chain_reversed_token(text)
        if reversed_keys is not None:
            return reversed_keys
        parts = [p.strip() for p in text.split(":") if p.strip()]
        if len(parts) == 2:
            chain, res_token = parts
            expanded = _expand_residue_range_token(res_token)
            if expanded is not None:
                return tuple(normalize_residue_key(chain, resseq) for resseq in expanded)
        if len(parts) == 3:
            _, res_token, icode_token = parts
            if _expand_residue_range_token(res_token) is not None or "-" in str(icode_token):
                raise ValueError("Residue ranges do not support insertion-code syntax.")

    ws_parts = [p for p in text.split() if p]
    if len(ws_parts) == 2:
        chain, res_token = ws_parts
        expanded = _expand_residue_range_token(res_token)
        if expanded is not None:
            return tuple(normalize_residue_key(chain, resseq) for resseq in expanded)
    elif len(ws_parts) == 1 and default_chain is not None:
        expanded = _expand_residue_range_token(ws_parts[0])
        if expanded is not None:
            return tuple(normalize_residue_key(default_chain, resseq) for resseq in expanded)

    return (parse_residue_token(text, default_chain=default_chain),)


def _parse_residue_entry(token: str, default_chain: str | None = None) -> str:
    """Backward-compatible wrapper for legacy internal calls."""
    return parse_residue_token(token, default_chain=default_chain)


def _split_line_tokens(line: str) -> list[str]:
    """Split one cleaned definition line into residue-like tokens.

    Handles lines such as:
    - "A:45,67,89"
    - "A:37-40"
    - "A 45"
    - "A 45 67 89"
    - "A:45 67 89"
    """
    line = _normalize_residue_definition_text(line)
    segments = [seg.strip() for seg in line.split(",") if seg.strip()]
    if not segments:
        return []

    if len(segments) == 1:
        one = segments[0]
        ws_parts = [p for p in one.split() if p]
        if len(ws_parts) >= 2 and ":" not in ws_parts[0]:
            chain = ws_parts[0]
            return [f"{chain} {token}" for token in ws_parts[1:]]
        if len(ws_parts) >= 2 and ":" in ws_parts[0]:
            return [ws_parts[0], *ws_parts[1:]]

    return segments


def load_residue_set(file_path: str) -> set[str]:
    """Load and normalize residue keys from pocket/catalytic definition file.

    Supported formats:
    - Format A (one residue per line):
      ``A:45``
    - Format B (multiple residues in one line):
      ``A:45,67,89,102``
    - Format C (inclusive ranges):
      ``A:37-40`` is equivalent to ``A:37,A:38,A:39,A:40``.
    - Format D: blank lines and comment lines starting with ``#``.

    Args:
        file_path: Path to residue definition text file.

    Returns:
        Normalized residue key set.

    Raises:
        FileNotFoundError: File not found.
        ValueError: File contains malformed residue entries.
    """
    path = Path(file_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Residue definition file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Residue definition path is not a file: {path}")

    residue_keys: set[str] = set()

    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith(";") or line.startswith("//"):
            continue

        # Allow inline comments while keeping format C behavior.
        if "#" in line:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue

        tokens = _split_line_tokens(line)
        if not tokens:
            continue

        default_chain: str | None = None
        for idx, token in enumerate(tokens):
            try:
                keys = parse_residue_token_or_range(token, default_chain=default_chain)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid residue format at {path}:{line_no}: {token!r}. {exc}"
                ) from exc

            residue_keys.update(keys)

            if idx == 0:
                default_chain = keys[0].split(":", 1)[0]

    return residue_keys


def _iter_structure_residues(structure_or_residues: Any) -> Iterator[Residue]:
    """Yield residues from supported structure-like input."""
    if isinstance(structure_or_residues, Residue):
        yield structure_or_residues
        return

    if isinstance(structure_or_residues, Chain):
        yield from structure_or_residues.get_residues()
        return

    if isinstance(structure_or_residues, Model):
        yield from structure_or_residues.get_residues()
        return

    if isinstance(structure_or_residues, Structure):
        models = list(structure_or_residues.get_models())
        if not models:
            raise ValueError("Structure contains no MODEL records.")
        yield from models[0].get_residues()
        return

    if isinstance(structure_or_residues, Iterable):
        for item in structure_or_residues:
            if not isinstance(item, Residue):
                raise TypeError(
                    "When passing an iterable, every item must be Bio.PDB.Residue.Residue."
                )
            yield item
        return

    raise TypeError(
        "Unsupported input type for residue matching. "
        "Use Structure, Model, Chain, Residue, or iterable of Residue."
    )


def _residue_to_key(residue: Residue) -> str:
    """Convert a structure residue object to canonical key."""
    parent_chain = residue.get_parent()
    chain_id = str(parent_chain.id) if isinstance(parent_chain, Chain) else _BLANK_CHAIN_TOKEN
    _, resseq, icode = residue.id
    return normalize_residue_key(chain_id=chain_id, resseq=resseq, icode=icode)


def _parse_normalized_residue_key(residue_key: str) -> tuple[str, int, str]:
    """Parse canonical residue key back to components."""
    parts = [p.strip() for p in str(residue_key).split(":")]
    if len(parts) == 2:
        chain, resseq_text = parts
        return _normalize_chain_token(chain), int(resseq_text), ""
    if len(parts) == 3:
        chain, resseq_text, icode = parts
        return _normalize_chain_token(chain), int(resseq_text), _normalize_icode(icode)
    raise ValueError(f"Invalid normalized residue key: {residue_key!r}")


def _to_tuple_index(raw: dict[Any, list[Residue]]) -> dict[Any, tuple[Residue, ...]]:
    """Freeze list-based indexes into tuple-based indexes."""
    return {k: tuple(v) for k, v in raw.items()}


def build_structure_residue_index(
    structure_or_residues: Any,
    include_hetatm: bool = False,
) -> StructureResidueIndex:
    """Build robust residue indexes from a structure-like input."""
    residues: list[Residue] = []
    exact_key_index_raw: dict[str, list[Residue]] = {}
    chain_resseq_index_raw: dict[tuple[str, int], list[Residue]] = {}
    chain_resseq_icode_index_raw: dict[tuple[str, int, str], list[Residue]] = {}
    chainless_resseq_index_raw: dict[int, list[Residue]] = {}
    chainless_resseq_icode_index_raw: dict[tuple[int, str], list[Residue]] = {}
    chain_id_order: list[str] = []

    for residue in _iter_structure_residues(structure_or_residues):
        hetflag = str(residue.id[0]).strip()
        if hetflag and not include_hetatm:
            continue

        key = _residue_to_key(residue)
        chain_id, resseq, icode = _parse_normalized_residue_key(key)
        residues.append(residue)

        if chain_id not in chain_id_order:
            chain_id_order.append(chain_id)

        exact_key_index_raw.setdefault(key, []).append(residue)
        chain_resseq_index_raw.setdefault((chain_id, resseq), []).append(residue)
        chain_resseq_icode_index_raw.setdefault((chain_id, resseq, icode), []).append(residue)
        chainless_resseq_index_raw.setdefault(resseq, []).append(residue)
        chainless_resseq_icode_index_raw.setdefault((resseq, icode), []).append(residue)

    return StructureResidueIndex(
        residues=tuple(residues),
        exact_key_index=_to_tuple_index(exact_key_index_raw),
        chain_resseq_index=_to_tuple_index(chain_resseq_index_raw),
        chain_resseq_icode_index=_to_tuple_index(chain_resseq_icode_index_raw),
        chainless_resseq_index=_to_tuple_index(chainless_resseq_index_raw),
        chainless_resseq_icode_index=_to_tuple_index(chainless_resseq_icode_index_raw),
        chain_ids=tuple(chain_id_order),
    )


def match_one_residue_key(
    residue_key: str,
    residue_index: StructureResidueIndex,
    allow_loose_match: bool = True,
    max_resseq_offset: int = 1,
) -> tuple[tuple[Residue, ...], ResidueMatchDetail]:
    """Match one residue key against a prebuilt structure index.

    Matching strategy:
    1) exact key (chain + resseq + icode)
    2) same chain + resseq (ignore icode) when unique
    3) missing/blank chain fallback by global unique residue id
    4) optional small residue-number offset fallback within the same chain
    """
    requested_key = parse_residue_token(residue_key, default_chain=None)
    chain_id, resseq, icode = _parse_normalized_residue_key(requested_key)

    exact_hits = residue_index.exact_key_index.get(requested_key, ())
    if exact_hits:
        return exact_hits, ResidueMatchDetail(
            requested_key=requested_key,
            matched_key=requested_key,
            mode="exact",
            warning="",
        )

    warnings: list[str] = []

    if allow_loose_match:
        same_chain_hits = residue_index.chain_resseq_index.get((chain_id, resseq), ())
        if len(same_chain_hits) == 1:
            matched_key = _residue_to_key(same_chain_hits[0])
            warning = (
                f"{requested_key}: exact match failed; matched by same chain+resseq "
                f"(requested icode={icode or '_'}, matched {matched_key})."
            )
            return same_chain_hits, ResidueMatchDetail(
                requested_key=requested_key,
                matched_key=matched_key,
                mode="same_chain_ignore_icode",
                warning=warning,
            )
        if len(same_chain_hits) > 1:
            warnings.append(
                f"{requested_key}: ambiguous same-chain fallback with {len(same_chain_hits)} candidates."
            )

        chain_is_missing = chain_id == _BLANK_CHAIN_TOKEN or chain_id not in residue_index.chain_ids
        if chain_is_missing:
            chainless_exact_hits = residue_index.chainless_resseq_icode_index.get((resseq, icode), ())
            if len(chainless_exact_hits) == 1:
                matched_key = _residue_to_key(chainless_exact_hits[0])
                warning = (
                    f"{requested_key}: chain missing/unavailable; matched by unique global resseq+icode "
                    f"to {matched_key}."
                )
                return chainless_exact_hits, ResidueMatchDetail(
                    requested_key=requested_key,
                    matched_key=matched_key,
                    mode="chain_fallback_exact",
                    warning=warning,
                )

            chainless_hits = residue_index.chainless_resseq_index.get(resseq, ())
            if len(chainless_hits) == 1:
                matched_key = _residue_to_key(chainless_hits[0])
                warning = (
                    f"{requested_key}: chain missing/unavailable; matched by unique global resseq "
                    f"to {matched_key}."
                )
                return chainless_hits, ResidueMatchDetail(
                    requested_key=requested_key,
                    matched_key=matched_key,
                    mode="chain_fallback_resseq",
                    warning=warning,
                )
            if len(chainless_hits) > 1:
                warnings.append(
                    f"{requested_key}: ambiguous chain fallback by resseq with {len(chainless_hits)} candidates."
                )

        if max_resseq_offset > 0:
            for offset in range(1, int(max_resseq_offset) + 1):
                for direction in (-1, 1):
                    candidate_resseq = resseq + direction * offset
                    candidate_hits = residue_index.chain_resseq_icode_index.get(
                        (chain_id, candidate_resseq, icode),
                        (),
                    )
                    if len(candidate_hits) == 1:
                        matched_key = _residue_to_key(candidate_hits[0])
                        warning = (
                            f"{requested_key}: matched by chain+offset fallback (offset={direction * offset}) "
                            f"to {matched_key}."
                        )
                        return candidate_hits, ResidueMatchDetail(
                            requested_key=requested_key,
                            matched_key=matched_key,
                            mode="chain_offset",
                            warning=warning,
                        )

                    candidate_hits_no_icode = residue_index.chain_resseq_index.get(
                        (chain_id, candidate_resseq),
                        (),
                    )
                    if len(candidate_hits_no_icode) == 1:
                        matched_key = _residue_to_key(candidate_hits_no_icode[0])
                        warning = (
                            f"{requested_key}: matched by chain+offset fallback (ignore icode, "
                            f"offset={direction * offset}) to {matched_key}."
                        )
                        return candidate_hits_no_icode, ResidueMatchDetail(
                            requested_key=requested_key,
                            matched_key=matched_key,
                            mode="chain_offset_ignore_icode",
                            warning=warning,
                        )

    unmatched_warning = (
        "; ".join(warnings)
        if warnings
        else f"{requested_key}: no residue matched by exact or configured loose rules."
    )
    return (), ResidueMatchDetail(
        requested_key=requested_key,
        matched_key=None,
        mode="unmatched",
        warning=unmatched_warning,
    )


def match_residues_in_structure(
    structure_or_residues: Any,
    residue_set: Iterable[str],
    allow_loose_match: bool = True,
    max_resseq_offset: int = 1,
) -> ResidueMatchResult:
    """Match text residue keys against real residues in a structure.

    Notes:
    - Matching is performed on canonical keys from ``normalize_residue_key``.
    - HETATM residues are ignored by default because pocket/catalytic files
      usually describe protein residues.

    Args:
        structure_or_residues: Structure/Model/Chain/Residue or iterable of Residue.
        residue_set: Residue keys loaded from text file.

    Returns:
        ``ResidueMatchResult`` with matched/missing details.
    """
    if residue_set is None:
        raise ValueError("residue_set cannot be None.")

    requested: set[str] = set()
    for raw_key in residue_set:
        requested.add(parse_residue_token(str(raw_key), default_chain=None))

    residue_index = build_structure_residue_index(
        structure_or_residues=structure_or_residues,
        include_hetatm=False,
    )

    details: list[ResidueMatchDetail] = []
    warnings: list[str] = []
    key_to_residues: dict[str, tuple[Residue, ...]] = {}
    matched_residue_list: list[Residue] = []
    matched_keys: set[str] = set()

    for key in sorted(requested):
        residues, detail = match_one_residue_key(
            residue_key=key,
            residue_index=residue_index,
            allow_loose_match=allow_loose_match,
            max_resseq_offset=max_resseq_offset,
        )
        details.append(detail)
        if detail.warning:
            warnings.append(detail.warning)
        if residues:
            key_to_residues[key] = residues
            matched_residue_list.extend(residues)
            matched_keys.add(key)

    missing_keys = requested - matched_keys

    unique_residue_ids: set[int] = set()
    unique_residues: list[Residue] = []
    for residue in matched_residue_list:
        rid = id(residue)
        if rid in unique_residue_ids:
            continue
        unique_residue_ids.add(rid)
        unique_residues.append(residue)

    residue_centroids: dict[str, tuple[float, float, float]] = {}
    coord_chunks: list[np.ndarray] = []
    matched_chain_ids: set[str] = set()
    for key, residues in key_to_residues.items():
        residue_coords: list[np.ndarray] = []
        for residue in residues:
            try:
                coords = extract_residue_atoms(residue)
            except Exception:
                coords = np.empty((0, 3), dtype=np.float64)
            if coords.shape[0] > 0:
                residue_coords.append(coords)
                coord_chunks.append(coords)

            parent_chain = residue.get_parent()
            if isinstance(parent_chain, Chain):
                chain_text = str(parent_chain.id).strip() or _BLANK_CHAIN_TOKEN
                matched_chain_ids.add(chain_text)

        if residue_coords:
            merged = np.vstack(residue_coords).astype(np.float64, copy=False)
            center = np.mean(merged, axis=0, dtype=np.float64)
            residue_centroids[key] = (float(center[0]), float(center[1]), float(center[2]))

    atom_coordinates = (
        np.vstack(coord_chunks).astype(np.float64, copy=False)
        if coord_chunks
        else np.empty((0, 3), dtype=np.float64)
    )

    summary = {
        "requested_residue_count": int(len(requested)),
        "matched_key_count": int(len(matched_keys)),
        "unmatched_key_count": int(len(missing_keys)),
        "matched_residue_count": int(len(unique_residues)),
        "matched_chain_count": int(len(matched_chain_ids)),
        "is_cross_chain_pocket": bool(len(matched_chain_ids) > 1),
        "match_warning_count": int(len(warnings)),
    }

    return ResidueMatchResult(
        requested_keys=frozenset(requested),
        matched_keys=frozenset(matched_keys),
        missing_keys=frozenset(missing_keys),
        key_to_residues=key_to_residues,
        matched_residues=tuple(unique_residues),
        unmatched_keys=frozenset(missing_keys),
        warnings=tuple(warnings),
        details=tuple(details),
        residue_centroids=residue_centroids,
        atom_coordinates=atom_coordinates,
        summary=summary,
    )


def _infer_element(atom: Atom) -> str:
    """Infer atom element symbol robustly for malformed records."""
    element = str(getattr(atom, "element", "") or "").strip().upper()
    if element:
        return element

    name = atom.get_name().strip().upper()
    letters = "".join(ch for ch in name if ch.isalpha())
    if not letters:
        return ""
    return letters[0]


def _is_hydrogen(atom: Atom) -> bool:
    """Return True when atom is hydrogen/deuterium."""
    element = _infer_element(atom)
    if element in {"H", "D"}:
        return True
    atom_name = atom.get_name().strip().upper()
    return atom_name.startswith("H") or atom_name.startswith("D")


def _record_name(line: str) -> str:
    """Return uppercase PDB record name from one line."""
    return line[:6].strip().upper()


def _extract_first_model_lines(lines: list[str]) -> list[str]:
    """Extract first MODEL block from PDB text.

    If no explicit MODEL record exists, the input lines are returned as-is.
    """
    selected: list[str] = []
    saw_model = False
    in_first_model = False

    for raw_line in lines:
        record = _record_name(raw_line)
        if record == "MODEL":
            if not saw_model:
                saw_model = True
                in_first_model = True
            else:
                break
            continue

        if record == "ENDMDL":
            if in_first_model:
                break
            continue

        if saw_model and not in_first_model:
            continue

        selected.append(raw_line.rstrip("\r\n"))

    return selected


def _atom_altloc_rank(atom: Atom) -> tuple[int, float, float, int]:
    """Ranking key for selecting the best altloc atom candidate."""
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


def _select_best_altloc_atoms(residue: Residue) -> list[Atom]:
    """Return one selected atom per residue atom name, resolving altlocs safely."""
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


def load_ligand_template_pdb(ligand_pdb_path: str) -> LigandTemplate:
    """Load ligand/template PDB and extract atom coordinates.

    The parser uses the first MODEL when multiple MODEL blocks exist.
    Both ATOM and HETATM records are accepted.

    Args:
        ligand_pdb_path: Ligand/template PDB path.

    Returns:
        ``LigandTemplate`` data object.

    Raises:
        FileNotFoundError: File not found.
        ValueError: File has no valid atom coordinates.
    """
    path = Path(ligand_pdb_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Ligand template PDB not found: {path}")
    if not path.is_file():
        raise ValueError(f"Ligand template path is not a file: {path}")

    raw_text = path.read_text(encoding="utf-8", errors="replace")
    if not raw_text.strip():
        raise ValueError(f"Ligand template file is empty: {path}")

    raw_lines = raw_text.splitlines()
    first_model_lines = _extract_first_model_lines(raw_lines)
    if not first_model_lines:
        raise ValueError(f"No usable records found in ligand template file: {path}")

    atom_like_lines = [line for line in first_model_lines if _record_name(line) in {"ATOM", "HETATM"}]
    if not atom_like_lines:
        raise ValueError(f"No ATOM/HETATM records found in ligand template file: {path}")

    parser = PDBParser(PERMISSIVE=True, QUIET=True)
    try:
        structure = parser.get_structure(path.stem, StringIO("\n".join(first_model_lines) + "\n"))
    except Exception as exc:
        raise ValueError(f"Failed to parse ligand template PDB {path}: {exc}") from exc

    models = list(structure.get_models())
    if not models:
        raise ValueError(f"No MODEL found in ligand template file: {path}")

    model = models[0]
    atoms: list[LigandAtom] = []
    coords: list[tuple[float, float, float]] = []

    for residue in model.get_residues():
        parent_chain = residue.get_parent()
        chain_id = str(parent_chain.id) if isinstance(parent_chain, Chain) else _BLANK_CHAIN_TOKEN
        resname = str(residue.get_resname()).strip() or "UNK"
        resseq = int(residue.id[1])
        icode = str(residue.id[2]).strip()

        for atom in _select_best_altloc_atoms(residue):
            # Keep ligand template compact by default: heavy atoms only.
            if _is_hydrogen(atom):
                continue

            try:
                coord = np.asarray(atom.get_coord(), dtype=np.float64).reshape(3)
            except Exception:
                continue

            if not np.isfinite(coord).all():
                continue

            try:
                serial = int(atom.get_serial_number())
            except Exception:
                serial = None

            atom_info = LigandAtom(
                serial_number=serial,
                atom_name=str(atom.get_name()).strip(),
                element=_infer_element(atom),
                coord=(float(coord[0]), float(coord[1]), float(coord[2])),
                residue_name=resname,
                chain_id=chain_id if chain_id.strip() else _BLANK_CHAIN_TOKEN,
                resseq=resseq,
                icode=icode,
            )
            atoms.append(atom_info)
            coords.append(atom_info.coord)

    if not coords:
        raise ValueError(f"No valid heavy-atom coordinates found in ligand template: {path}")

    coord_array = np.asarray(coords, dtype=np.float64)
    centroid = np.mean(coord_array, axis=0, dtype=np.float64)
    return LigandTemplate(
        source_path=str(path),
        atoms=tuple(atoms),
        coordinates=coord_array,
        centroid=centroid,
    )


def extract_residue_atoms(residue: Residue) -> np.ndarray:
    """Extract heavy-atom coordinates from one residue.

    Missing/malformed atom coordinates are skipped instead of crashing.

    Args:
        residue: Biopython residue object.

    Returns:
        Array of shape (N, 3). Returns empty array when no valid heavy atom exists.
    """
    if not isinstance(residue, Residue):
        raise TypeError("residue must be a Bio.PDB.Residue.Residue instance")

    coords: list[tuple[float, float, float]] = []
    for atom in _select_best_altloc_atoms(residue):
        if _is_hydrogen(atom):
            continue

        try:
            xyz = np.asarray(atom.get_coord(), dtype=np.float64).reshape(3)
        except Exception:
            continue

        if not np.isfinite(xyz).all():
            continue
        coords.append((float(xyz[0]), float(xyz[1]), float(xyz[2])))

    if not coords:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(coords, dtype=np.float64)


def extract_residue_centroid(residue: Residue) -> np.ndarray:
    """Compute centroid of heavy atoms for one residue.

    Args:
        residue: Biopython residue object.

    Returns:
        Centroid coordinates with shape (3,).

    Raises:
        ValueError: Residue has no valid heavy-atom coordinates.
    """
    coords = extract_residue_atoms(residue)
    if coords.shape[0] == 0:
        try:
            res_key = _residue_to_key(residue)
        except Exception:
            res_key = "<unknown residue>"
        raise ValueError(f"Cannot compute centroid: residue {res_key} has no valid heavy atoms.")

    return np.mean(coords, axis=0, dtype=np.float64)


def _to_standard_match_result(match: ResidueMatchResult) -> PocketResidueMatchResult:
    """Convert backward-compatible ResidueMatchResult to standardized result object."""
    return PocketResidueMatchResult(
        requested_keys=tuple(sorted(match.requested_keys)),
        matched_keys=tuple(sorted(match.matched_keys)),
        unmatched_keys=tuple(sorted(match.unmatched_keys or match.missing_keys)),
        key_to_residues=dict(match.key_to_residues),
        matched_residues=tuple(match.matched_residues),
        residue_centroids=dict(match.residue_centroids),
        atom_coordinates=np.asarray(match.atom_coordinates, dtype=np.float64),
        details=tuple(match.details),
        warnings=tuple(match.warnings),
        summary=dict(match.summary),
    )


def summarize_pocket_definition(
    definition: PocketDefinition,
    ligand_template: LigandTemplate | None = None,
) -> dict[str, Any]:
    """Build explainable summary statistics for one pocket/catalytic definition."""
    residue_keys = set(definition.residue_keys)
    key_chains = {key.split(":", 1)[0] for key in residue_keys if ":" in key}

    matched_residues = list(definition.matched_residues)
    matched_chain_ids: set[str] = set()
    for residue in matched_residues:
        parent = residue.get_parent()
        if isinstance(parent, Chain):
            matched_chain_ids.add(str(parent.id).strip() or _BLANK_CHAIN_TOKEN)

    unmatched_count = 0
    matched_key_count = 0
    warning_count = len(definition.warnings)
    if definition.match_result is not None:
        unmatched_count = len(definition.match_result.unmatched_keys)
        matched_key_count = len(definition.match_result.matched_keys)
        warning_count = len(definition.match_result.warnings)

    if matched_chain_ids:
        is_cross_chain = len(matched_chain_ids) > 1
    else:
        is_cross_chain = len(key_chains) > 1

    ligand_atom_count = 0
    if ligand_template is not None:
        try:
            ligand_atom_count = int(np.asarray(ligand_template.coordinates).shape[0])
        except Exception:
            ligand_atom_count = 0

    summary = {
        "label": definition.source.label,
        "source_name": definition.source.source_name,
        "source_path": definition.source.source_path,
        "defined_residue_count": int(len(residue_keys)),
        "matched_key_count": int(matched_key_count),
        "matched_residue_count": int(len(matched_residues)),
        "unmatched_key_count": int(unmatched_count),
        "is_cross_chain_pocket": bool(is_cross_chain),
        "matched_chain_count": int(len(matched_chain_ids)),
        "warning_count": int(warning_count),
        "ligand_atom_count": int(ligand_atom_count),
    }
    return summary


def make_pocket_definition_from_manual_file(
    file_path: str,
    structure_or_residues: Any = None,
    label: str = "pocket",
    allow_loose_match: bool = True,
    max_resseq_offset: int = 1,
) -> PocketDefinition:
    """Build standardized pocket/catalytic definition from a manual residue file."""
    path = Path(file_path).expanduser()
    residue_keys = load_residue_set(str(path))
    source = PocketSource(
        source_name="manual",
        source_path=str(path),
        label=str(label).strip() or "pocket",
        metadata={},
    )

    if structure_or_residues is None:
        definition = PocketDefinition(
            source=source,
            residue_keys=frozenset(residue_keys),
        )
        definition.summary = summarize_pocket_definition(definition)
        return definition

    match = match_residues_in_structure(
        structure_or_residues=structure_or_residues,
        residue_set=residue_keys,
        allow_loose_match=allow_loose_match,
        max_resseq_offset=max_resseq_offset,
    )
    standard_match = _to_standard_match_result(match)

    definition = PocketDefinition(
        source=source,
        residue_keys=frozenset(residue_keys),
        match_result=standard_match,
        matched_residues=tuple(standard_match.matched_residues),
        atom_coordinates=np.asarray(standard_match.atom_coordinates, dtype=np.float64),
        residue_centroids=dict(standard_match.residue_centroids),
        warnings=tuple(standard_match.warnings),
    )
    definition.summary = summarize_pocket_definition(definition)
    return definition


def _iter_external_output_files(path: Path, preferred_markers: tuple[str, ...]) -> list[Path]:
    """Collect candidate files from an external pocket-tool output path."""
    if path.is_file():
        return [path]

    if not path.is_dir():
        return []

    preferred = tuple(m.strip().lower() for m in preferred_markers if m and m.strip())
    scored: list[tuple[int, str, Path]] = []
    for fp in path.rglob("*"):
        if not fp.is_file():
            continue
        suffix = fp.suffix.lower()
        if suffix not in _EXTERNAL_OUTPUT_SUFFIXES:
            continue

        name = fp.name.lower()
        score = 0
        if suffix == ".json":
            score += 1
        if "residue" in name:
            score += 2
        if any(marker in name for marker in preferred):
            score += 3
        scored.append((score, str(fp).lower(), fp))

    scored.sort(key=lambda x: (-x[0], x[1]))
    return [item[2] for item in scored]


def _extract_residue_keys_from_text(text: str) -> set[str]:
    """Extract normalized residue keys from loosely formatted text output."""
    residue_keys: set[str] = set()
    lines = str(text).splitlines()

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith(_EXTERNAL_TEXT_COMMENT_PREFIXES):
            continue

        cleaned_line = line.replace("|", ",").replace(";", ",")
        tokens = _split_line_tokens(cleaned_line)
        default_chain: str | None = None

        for token in tokens:
            tok = token.strip().strip("[]{}()")
            if not tok:
                continue

            try:
                keys = parse_residue_token_or_range(tok, default_chain=default_chain)
                residue_keys.update(keys)
                default_chain = keys[0].split(":", 1)[0]
                continue
            except ValueError:
                pass

            for match in _CHAIN_RES_INLINE_RE.finditer(tok):
                chain = match.group("chain")
                resseq = int(match.group("resseq"))
                icode = (match.group("icode_b") or match.group("icode_a") or "").strip()
                try:
                    key = normalize_residue_key(chain, resseq, icode)
                except ValueError:
                    continue
                residue_keys.add(key)
                default_chain = chain

    return residue_keys


def _extract_residue_keys_from_json(payload: Any, default_chain: str | None = None) -> set[str]:
    """Extract normalized residue keys from nested JSON-like payloads."""
    residue_keys: set[str] = set()

    if isinstance(payload, dict):
        chain_keys = ("chain", "chain_id", "chainId", "chainID")
        resseq_keys = ("resseq", "resseq_id", "residue_number", "residueNumber", "resid", "resnum")
        icode_keys = ("icode", "ins_code", "insertion_code", "insertionCode")
        residue_key_fields = (
            "residue_key",
            "residueKey",
            "residue_id",
            "residueId",
            "residue",
        )

        chain_value: str | None = default_chain
        for key in chain_keys:
            if key in payload and payload[key] is not None:
                chain_value = str(payload[key]).strip() or default_chain
                break

        for key in residue_key_fields:
            if key not in payload:
                continue
            value = payload[key]
            if isinstance(value, str):
                try:
                    residue_keys.update(parse_residue_token_or_range(value, default_chain=chain_value))
                    continue
                except ValueError:
                    residue_keys.update(_extract_residue_keys_from_text(value))

        if chain_value is not None:
            for key in resseq_keys:
                if key not in payload:
                    continue
                try:
                    resseq_raw = payload[key]
                    icode_raw = ""
                    for ik in icode_keys:
                        if ik in payload and payload[ik] is not None:
                            icode_raw = str(payload[ik]).strip()
                            break
                    residue_keys.add(normalize_residue_key(chain_value, int(resseq_raw), icode_raw))
                    break
                except Exception:
                    continue

        for value in payload.values():
            residue_keys.update(_extract_residue_keys_from_json(value, default_chain=chain_value))
        return residue_keys

    if isinstance(payload, (list, tuple, set)):
        for item in payload:
            residue_keys.update(_extract_residue_keys_from_json(item, default_chain=default_chain))
        return residue_keys

    if isinstance(payload, str):
        try:
            residue_keys.update(parse_residue_token_or_range(payload, default_chain=default_chain))
            return residue_keys
        except ValueError:
            return _extract_residue_keys_from_text(payload)

    return residue_keys


def _parse_external_pocket_output(
    output_path: str,
    tool_name: str,
    preferred_name_markers: tuple[str, ...],
) -> set[str]:
    """Parse external pocket-tool output and return normalized residue keys."""
    path = Path(output_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{tool_name} output not found: {path}")

    candidate_files = _iter_external_output_files(path, preferred_markers=preferred_name_markers)
    if not candidate_files:
        raise ValueError(
            f"{tool_name} output path contains no parsable files. "
            f"Supported suffixes: {sorted(_EXTERNAL_OUTPUT_SUFFIXES)}"
        )

    residue_keys: set[str] = set()
    for fp in candidate_files:
        parsed: set[str] = set()

        # Prefer strict manual parser first for explicit residue list files.
        try:
            parsed = load_residue_set(str(fp))
        except Exception:
            parsed = set()

        if not parsed and fp.suffix.lower() == ".json":
            try:
                payload = json.loads(fp.read_text(encoding="utf-8", errors="replace"))
                parsed = _extract_residue_keys_from_json(payload)
            except Exception:
                parsed = set()

        if not parsed:
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
                parsed = _extract_residue_keys_from_text(text)
            except Exception:
                parsed = set()

        if parsed:
            residue_keys.update(parsed)

    if residue_keys:
        return residue_keys

    raise ValueError(
        f"No valid residue keys could be parsed from {tool_name} output: {path}. "
        "Expected entries like 'A:45', 'A 45', or JSON fields with chain/residue identifiers."
    )


def parse_pykvfinder_output(output_path: str) -> set[str]:
    """Parse pyKVFinder output and return normalized residue keys.

    Supports file or directory input. The parser accepts common text/JSON outputs
    and normalizes residue identifiers to canonical keys like ``A:45``.
    """
    return _parse_external_pocket_output(
        output_path=output_path,
        tool_name="pyKVFinder",
        preferred_name_markers=("pykvfinder", "kvfinder", "pocket", "residue"),
    )


def parse_fpocket_output(output_path: str) -> set[str]:
    """Parse fpocket output and return normalized residue keys."""
    return _parse_external_pocket_output(
        output_path=output_path,
        tool_name="fpocket",
        preferred_name_markers=("fpocket", "pocket", "residue"),
    )


def parse_p2rank_output(output_path: str) -> set[str]:
    """Parse P2Rank output and return normalized residue keys."""
    return _parse_external_pocket_output(
        output_path=output_path,
        tool_name="P2Rank",
        preferred_name_markers=("p2rank", "rank", "pocket", "residue"),
    )


def build_pocket_definition_data(
    pocket_file: str | None,
    catalytic_file: str | None,
    ligand_template_pdb: str | None = None,
) -> PocketDefinitionData:
    """Build unified pocket-definition payload from optional inputs.

    Args:
        pocket_file: Pocket residue definition file path.
        catalytic_file: Catalytic residue definition file path.
        ligand_template_pdb: Optional ligand template PDB path.

    Returns:
        ``PocketDefinitionData`` containing parsed inputs.
    """
    pocket_data: ResidueKeySet | None = None
    catalytic_data: ResidueKeySet | None = None
    ligand_data: LigandTemplate | None = None

    if pocket_file:
        pocket_keys = load_residue_set(pocket_file)
        pocket_data = ResidueKeySet(
            source_path=str(Path(pocket_file).expanduser()),
            residue_keys=frozenset(pocket_keys),
        )

    if catalytic_file:
        catalytic_keys = load_residue_set(catalytic_file)
        catalytic_data = ResidueKeySet(
            source_path=str(Path(catalytic_file).expanduser()),
            residue_keys=frozenset(catalytic_keys),
        )

    if ligand_template_pdb:
        ligand_data = load_ligand_template_pdb(ligand_template_pdb)

    return PocketDefinitionData(
        pocket=pocket_data,
        catalytic=catalytic_data,
        ligand_template=ligand_data,
    )


__all__ = [
    "ResidueKeySet",
    "LigandAtom",
    "LigandTemplate",
    "PocketSource",
    "ResidueMatchDetail",
    "PocketResidueMatchResult",
    "ResidueMatchResult",
    "StructureResidueIndex",
    "PocketDefinitionData",
    "PocketDefinition",
    "ExternalPocketToolAdapter",
    "parse_residue_token",
    "parse_residue_token_or_range",
    "load_residue_set",
    "normalize_residue_key",
    "build_structure_residue_index",
    "match_one_residue_key",
    "match_residues_in_structure",
    "load_ligand_template_pdb",
    "extract_residue_atoms",
    "extract_residue_centroid",
    "summarize_pocket_definition",
    "make_pocket_definition_from_manual_file",
    "parse_pykvfinder_output",
    "parse_fpocket_output",
    "parse_p2rank_output",
    "build_pocket_definition_data",
]
