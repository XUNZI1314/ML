"""Export residue-level annotations and interface summaries for one complex.

This script is intentionally lightweight. It reuses the existing structure
parsing and residue-matching modules, then writes viewer-friendly JSON/CSV
artifacts without changing the ranking or training pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pdb_parser import (
    AtomInfo,
    ResidueInfo,
    extract_atoms_from_entity,
    extract_residues_from_entity,
    get_residue_uid,
    load_complex_pdb,
    split_antigen_nanobody,
)
from pocket_io import (
    ResidueMatchResult,
    load_residue_set,
    match_residues_in_structure,
    normalize_residue_key,
    parse_residue_token_or_range,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export residue/interface annotations for one complex")
    parser.add_argument("--pdb_path", required=True, help="Path to one complex PDB file")
    parser.add_argument("--out_dir", default="structure_annotation_outputs", help="Output directory")
    parser.add_argument("--antigen_chain", default=None, help="Optional antigen chain ID(s)")
    parser.add_argument("--nanobody_chain", default=None, help="Optional nanobody chain ID(s)")
    parser.add_argument("--pocket_file", default=None, help="Optional pocket residue definition file")
    parser.add_argument("--catalytic_file", default=None, help="Optional catalytic residue definition file")
    parser.add_argument(
        "--key_residues",
        default=None,
        help="Optional inline user key residues, e.g. 'A:45,A:46', 'A:45 46 47', or 'A:37-40'",
    )
    parser.add_argument("--key_residue_file", default=None, help="Optional user key residue definition file")
    parser.add_argument(
        "--key_residue_default_chain",
        default=None,
        help="Default chain used when inline key residues omit chain IDs, e.g. '45 46 47'",
    )
    parser.add_argument(
        "--interface_threshold",
        type=float,
        default=4.5,
        help="Interface residue threshold in Angstrom (default: 4.5)",
    )
    parser.add_argument(
        "--pocket_neighbor_threshold",
        type=float,
        default=4.5,
        help="Threshold used to mark residues as near pocket region (default: 4.5)",
    )
    return parser


def _entity_to_residue_objects(entity: Any) -> list[Any]:
    chain_ids = tuple(getattr(entity, "chain_ids", ()))
    model = getattr(entity, "model", None)
    if model is None or not chain_ids:
        raise TypeError("Entity does not expose model/chain_ids for residue extraction.")

    residues: list[Any] = []
    for chain_id in chain_ids:
        chain = model[chain_id]
        residues.extend(list(chain.get_residues()))
    return residues


def _resolve_optional_path(path_like: str | None) -> Path | None:
    if path_like is None:
        return None
    text = str(path_like).strip()
    if not text:
        return None
    return Path(text).expanduser().resolve()


def _residue_ident_from_info(info: ResidueInfo) -> tuple[str, int, str, str]:
    return (str(info.chain_id), int(info.resseq), str(info.icode), str(info.resname))


def _residue_ident_from_atom(atom: AtomInfo) -> tuple[str, int, str, str]:
    return (str(atom.chain_id), int(atom.resseq), str(atom.icode), str(atom.resname))


def _build_residue_atom_map(atoms: list[AtomInfo]) -> dict[tuple[str, int, str, str], np.ndarray]:
    grouped: dict[tuple[str, int, str, str], list[np.ndarray]] = {}
    for atom in atoms:
        ident = _residue_ident_from_atom(atom)
        grouped.setdefault(ident, []).append(np.asarray(atom.coord, dtype=np.float64).reshape(1, 3))
    return {
        ident: np.vstack(parts).astype(np.float64, copy=False)
        for ident, parts in grouped.items()
        if parts
    }


def _global_min_distance(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    if coords_a.shape[0] == 0 or coords_b.shape[0] == 0:
        return float("nan")

    diff = coords_a[:, None, :] - coords_b[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff, optimize=True)
    return float(np.sqrt(np.min(d2)))


def _distance_to_point(coords: np.ndarray, point: np.ndarray | None) -> float:
    if coords.shape[0] == 0 or point is None:
        return float("nan")
    arr = np.asarray(point, dtype=np.float64).reshape(1, 3)
    return _global_min_distance(coords, arr)


def _coords_from_atom_infos(atoms: list[AtomInfo]) -> np.ndarray:
    if not atoms:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray([atom.coord for atom in atoms], dtype=np.float64).reshape(-1, 3)


def _mean_coord(coords: np.ndarray) -> np.ndarray | None:
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] == 0:
        return None
    finite = coords[np.isfinite(coords).all(axis=1)]
    if finite.shape[0] == 0:
        return None
    return np.mean(finite, axis=0, dtype=np.float64)


def _match_optional_residues(entity: Any, path_like: str | None) -> tuple[Path | None, set[str] | None, ResidueMatchResult | None]:
    path = _resolve_optional_path(path_like)
    if path is None:
        return None, None, None
    if not path.exists():
        raise FileNotFoundError(f"Residue definition file not found: {path}")
    keys = load_residue_set(str(path))
    try:
        match = match_residues_in_structure(entity, keys)
    except TypeError:
        match = match_residues_in_structure(_entity_to_residue_objects(entity), keys)
    return path, keys, match


def _split_inline_residue_tokens(line: str, default_chain: str | None = None) -> list[str]:
    segments = [seg.strip() for seg in str(line).split(",") if seg.strip()]
    if not segments:
        return []

    if len(segments) == 1:
        one = segments[0]
        ws_parts = [p for p in one.split() if p]
        if len(ws_parts) >= 2 and ":" not in ws_parts[0]:
            if default_chain is not None:
                all_can_use_default = True
                for token in ws_parts:
                    try:
                        parse_residue_token_or_range(token, default_chain=default_chain)
                    except ValueError:
                        all_can_use_default = False
                        break
                if all_can_use_default:
                    return ws_parts

            chain = ws_parts[0]
            return [f"{chain} {token}" for token in ws_parts[1:]]
        if len(ws_parts) >= 2 and ":" in ws_parts[0]:
            return [ws_parts[0], *ws_parts[1:]]

    return segments


def _parse_inline_residue_keys(text: str | None, default_chain: str | None = None) -> set[str]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return set()

    residue_keys: set[str] = set()
    running_default_chain = str(default_chain).strip() if default_chain else None
    for raw_line in cleaned.replace(";", "\n").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "#" in line:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue

        tokens = _split_inline_residue_tokens(line, default_chain=running_default_chain)
        if not tokens:
            continue

        line_default_chain = running_default_chain
        for idx, token in enumerate(tokens):
            keys = parse_residue_token_or_range(token, default_chain=line_default_chain)
            residue_keys.update(keys)
            if idx == 0:
                line_default_chain = keys[0].split(":", 1)[0]
                if running_default_chain is None:
                    running_default_chain = line_default_chain

    return residue_keys


def _match_residue_definition(
    entity: Any,
    *,
    path_like: str | None = None,
    inline_text: str | None = None,
    default_chain: str | None = None,
) -> tuple[Path | None, set[str] | None, ResidueMatchResult | None, str | None]:
    path = _resolve_optional_path(path_like)
    key_set: set[str] = set()
    inline_payload = str(inline_text).strip() if inline_text is not None and str(inline_text).strip() else None

    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"Residue definition file not found: {path}")
        key_set.update(load_residue_set(str(path)))
    if inline_payload is not None:
        key_set.update(_parse_inline_residue_keys(inline_payload, default_chain=default_chain))

    if not key_set:
        return path, None, None, inline_payload

    try:
        match = match_residues_in_structure(entity, key_set)
    except TypeError:
        match = match_residues_in_structure(_entity_to_residue_objects(entity), key_set)
    return path, key_set, match, inline_payload


def _match_to_uid_set(match: ResidueMatchResult | None) -> set[str]:
    if match is None:
        return set()
    return {get_residue_uid(residue) for residue in match.matched_residues}


def _match_to_coords(match: ResidueMatchResult | None) -> np.ndarray:
    if match is None:
        return np.empty((0, 3), dtype=np.float64)
    coords = np.asarray(match.atom_coordinates, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        return np.empty((0, 3), dtype=np.float64)
    finite = coords[np.isfinite(coords).all(axis=1)]
    return finite.astype(np.float64, copy=False)


def _serialize_match_result(
    match: ResidueMatchResult | None,
    source_path: Path | None,
    inline_text: str | None = None,
) -> dict[str, Any]:
    if match is None:
        return {
            "source_path": str(source_path) if source_path is not None else None,
            "inline_text": inline_text,
            "requested_keys": [],
            "matched_keys": [],
            "unmatched_keys": [],
            "matched_residue_uids": [],
            "centroid": None,
            "residue_centroids": {},
            "warnings": [],
            "summary": {},
        }
    matched_coords = _match_to_coords(match)
    centroid = _mean_coord(matched_coords)
    return {
        "source_path": str(source_path) if source_path is not None else None,
        "inline_text": inline_text,
        "requested_keys": sorted(str(x) for x in match.requested_keys),
        "matched_keys": sorted(str(x) for x in match.matched_keys),
        "unmatched_keys": sorted(str(x) for x in match.unmatched_keys),
        "matched_residue_uids": sorted(get_residue_uid(residue) for residue in match.matched_residues),
        "centroid": None if centroid is None else [float(x) for x in centroid.tolist()],
        "residue_centroids": {
            str(key): [float(v[0]), float(v[1]), float(v[2])]
            for key, v in sorted(match.residue_centroids.items())
        },
        "warnings": [str(x) for x in match.warnings],
        "summary": dict(match.summary),
    }


def _to_jsonable_scalar(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        val = float(value)
        return val if np.isfinite(val) else None
    return value


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return _to_jsonable_scalar(value)


def _to_jsonable_row(row: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _to_jsonable_scalar(value) for key, value in row.items()}


def _finite_weighted_average(items: list[tuple[str, float, float]]) -> float | None:
    total_weight = 0.0
    total_value = 0.0
    for _, value, weight in items:
        val = float(value)
        w = float(weight)
        if not np.isfinite(val) or not np.isfinite(w) or w <= 0.0:
            continue
        total_weight += w
        total_value += val * w
    if total_weight <= 1e-12:
        return None
    return float(total_value / total_weight)


def _collect_annotations(
    *,
    entity_label: str,
    residues: list[ResidueInfo],
    residue_atom_map: dict[tuple[str, int, str, str], np.ndarray],
    opposite_coords: np.ndarray,
    pocket_coords: np.ndarray,
    pocket_center: np.ndarray | None,
    pocket_uid_set: set[str],
    catalytic_uid_set: set[str],
    key_uid_set: set[str],
    interface_threshold: float,
    pocket_neighbor_threshold: float,
    partner_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for residue in residues:
        ident = _residue_ident_from_info(residue)
        coords = residue_atom_map.get(ident, np.empty((0, 3), dtype=np.float64))
        residue_key = normalize_residue_key(residue.chain_id, residue.resseq, residue.icode)
        partner = dict(partner_lookup.get(str(residue.uid), {}))

        min_to_opposite = _global_min_distance(coords, opposite_coords)
        min_to_pocket = _global_min_distance(coords, pocket_coords)
        min_to_pocket_center = _distance_to_point(coords, pocket_center)

        row = {
            "entity_label": str(entity_label),
            "uid": str(residue.uid),
            "residue_key": str(residue_key),
            "chain_id": str(residue.chain_id),
            "resseq": int(residue.resseq),
            "icode": str(residue.icode),
            "resname": str(residue.resname),
            "atom_count": int(residue.atom_count),
            "is_pocket": bool(residue.uid in pocket_uid_set),
            "is_catalytic": bool(residue.uid in catalytic_uid_set),
            "is_user_key_residue": bool(residue.uid in key_uid_set),
            "is_interface": bool(np.isfinite(min_to_opposite) and min_to_opposite <= interface_threshold),
            "is_pocket_neighbor": bool(np.isfinite(min_to_pocket) and min_to_pocket <= pocket_neighbor_threshold),
            "min_distance_to_opposite_entity": min_to_opposite,
            "min_distance_to_pocket": min_to_pocket,
            "min_distance_to_pocket_center": min_to_pocket_center,
            "min_distance_to_nanobody": min_to_opposite if entity_label == "antigen" else float("nan"),
            "min_distance_to_antigen": min_to_opposite if entity_label == "nanobody" else float("nan"),
            "nearest_opposite_residue_uid": partner.get("nearest_opposite_residue_uid"),
            "nearest_opposite_residue_key": partner.get("nearest_opposite_residue_key"),
            "nearest_opposite_chain_id": partner.get("nearest_opposite_chain_id"),
            "nearest_opposite_resname": partner.get("nearest_opposite_resname"),
            "nearest_opposite_resseq": partner.get("nearest_opposite_resseq"),
            "nearest_opposite_is_interface_contact": partner.get("nearest_opposite_is_interface_contact"),
        }
        rows.append(row)
    return rows


def _summarize_annotations(
    annotation_df: pd.DataFrame,
    pocket_uid_set: set[str],
    interface_threshold: float,
    split_mode: str,
    source_detail: str,
    interface_pair_count: int,
    key_uid_set: set[str],
) -> dict[str, Any]:
    antigen_df = annotation_df.loc[annotation_df["entity_label"] == "antigen"].reset_index(drop=True)
    nanobody_df = annotation_df.loc[annotation_df["entity_label"] == "nanobody"].reset_index(drop=True)

    antigen_interface_df = antigen_df.loc[antigen_df["is_interface"]].reset_index(drop=True)
    nanobody_interface_df = nanobody_df.loc[nanobody_df["is_interface"]].reset_index(drop=True)
    pocket_df = antigen_df.loc[antigen_df["is_pocket"]].reset_index(drop=True)
    pocket_interface_df = pocket_df.loc[pocket_df["is_interface"]].reset_index(drop=True)
    nanobody_pocket_neighbor_df = nanobody_df.loc[nanobody_df["is_pocket_neighbor"]].reset_index(drop=True)

    pocket_count = int(len(pocket_df))
    pocket_interface_count = int(len(pocket_interface_df))
    nanobody_pocket_neighbor_count = int(len(nanobody_pocket_neighbor_df))

    pocket_contact_coverage = (
        float(pocket_interface_count / pocket_count)
        if pocket_count > 0
        else float("nan")
    )
    mean_pocket_contact_closeness = float("nan")
    if pocket_count > 0:
        pocket_min_dist = pd.to_numeric(pocket_df["min_distance_to_nanobody"], errors="coerce").to_numpy(dtype=np.float64)
        finite = pocket_min_dist[np.isfinite(pocket_min_dist)]
        if finite.size > 0 and interface_threshold > 0:
            closeness = np.clip(1.0 - (finite / float(interface_threshold)), 0.0, 1.0)
            mean_pocket_contact_closeness = float(np.mean(closeness))

    nanobody_pocket_contact_ratio = (
        float(nanobody_pocket_neighbor_count / max(int(len(nanobody_df)), 1))
        if len(nanobody_df) > 0
        else float("nan")
    )

    return {
        "split_mode": str(split_mode),
        "split_source_detail": str(source_detail),
        "total_residue_count": int(len(annotation_df)),
        "antigen_residue_count": int(len(antigen_df)),
        "nanobody_residue_count": int(len(nanobody_df)),
        "pocket_residue_count": int(len(pocket_uid_set)),
        "key_residue_count": int(len(key_uid_set)),
        "antigen_key_residue_count": int(antigen_df["is_user_key_residue"].sum()) if "is_user_key_residue" in antigen_df.columns else 0,
        "nanobody_key_residue_count": int(nanobody_df["is_user_key_residue"].sum()) if "is_user_key_residue" in nanobody_df.columns else 0,
        "antigen_interface_count": int(len(antigen_interface_df)),
        "nanobody_interface_count": int(len(nanobody_interface_df)),
        "interface_pair_count": int(interface_pair_count),
        "pocket_interface_overlap_count": pocket_interface_count,
        "pocket_contact_coverage": pocket_contact_coverage,
        "mean_pocket_contact_closeness": mean_pocket_contact_closeness,
        "nanobody_pocket_neighbor_count": nanobody_pocket_neighbor_count,
        "nanobody_pocket_contact_ratio": nanobody_pocket_contact_ratio,
    }


def _build_residue_payloads(
    residues: list[ResidueInfo],
    residue_atom_map: dict[tuple[str, int, str, str], np.ndarray],
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for residue in residues:
        ident = _residue_ident_from_info(residue)
        coords = residue_atom_map.get(ident, np.empty((0, 3), dtype=np.float64))
        payloads.append(
            {
                "uid": str(residue.uid),
                "residue_key": normalize_residue_key(residue.chain_id, residue.resseq, residue.icode),
                "chain_id": str(residue.chain_id),
                "resseq": int(residue.resseq),
                "icode": str(residue.icode),
                "resname": str(residue.resname),
                "coords": coords,
            }
        )
    return payloads


def _compute_interface_pair_details(
    antigen_residues: list[ResidueInfo],
    nanobody_residues: list[ResidueInfo],
    antigen_atom_map: dict[tuple[str, int, str, str], np.ndarray],
    nanobody_atom_map: dict[tuple[str, int, str, str], np.ndarray],
    interface_threshold: float,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    pair_columns = [
        "antigen_uid",
        "antigen_residue_key",
        "antigen_chain_id",
        "antigen_resname",
        "antigen_resseq",
        "nanobody_uid",
        "nanobody_residue_key",
        "nanobody_chain_id",
        "nanobody_resname",
        "nanobody_resseq",
        "min_atom_distance",
    ]
    antigen_payloads = _build_residue_payloads(antigen_residues, antigen_atom_map)
    nanobody_payloads = _build_residue_payloads(nanobody_residues, nanobody_atom_map)

    antigen_partner_lookup: dict[str, dict[str, Any]] = {}
    nanobody_partner_lookup: dict[str, dict[str, Any]] = {}
    nanobody_best_dist: dict[str, float] = {item["uid"]: float("inf") for item in nanobody_payloads}
    pair_rows: list[dict[str, Any]] = []

    for antigen_item in antigen_payloads:
        best_dist = float("inf")
        best_partner: dict[str, Any] | None = None
        for nanobody_item in nanobody_payloads:
            dist = _global_min_distance(antigen_item["coords"], nanobody_item["coords"])
            if not np.isfinite(dist):
                continue

            if dist < best_dist:
                best_dist = dist
                best_partner = nanobody_item

            if dist < nanobody_best_dist[nanobody_item["uid"]]:
                nanobody_best_dist[nanobody_item["uid"]] = dist
                nanobody_partner_lookup[nanobody_item["uid"]] = {
                    "nearest_opposite_residue_uid": antigen_item["uid"],
                    "nearest_opposite_residue_key": antigen_item["residue_key"],
                    "nearest_opposite_chain_id": antigen_item["chain_id"],
                    "nearest_opposite_resname": antigen_item["resname"],
                    "nearest_opposite_resseq": antigen_item["resseq"],
                    "nearest_opposite_is_interface_contact": bool(dist <= interface_threshold),
                }

            if dist <= interface_threshold:
                pair_rows.append(
                    {
                        "antigen_uid": antigen_item["uid"],
                        "antigen_residue_key": antigen_item["residue_key"],
                        "antigen_chain_id": antigen_item["chain_id"],
                        "antigen_resname": antigen_item["resname"],
                        "antigen_resseq": antigen_item["resseq"],
                        "nanobody_uid": nanobody_item["uid"],
                        "nanobody_residue_key": nanobody_item["residue_key"],
                        "nanobody_chain_id": nanobody_item["chain_id"],
                        "nanobody_resname": nanobody_item["resname"],
                        "nanobody_resseq": nanobody_item["resseq"],
                        "min_atom_distance": float(dist),
                    }
                )

        antigen_partner_lookup[antigen_item["uid"]] = {
            "nearest_opposite_residue_uid": None if best_partner is None else best_partner["uid"],
            "nearest_opposite_residue_key": None if best_partner is None else best_partner["residue_key"],
            "nearest_opposite_chain_id": None if best_partner is None else best_partner["chain_id"],
            "nearest_opposite_resname": None if best_partner is None else best_partner["resname"],
            "nearest_opposite_resseq": None if best_partner is None else best_partner["resseq"],
            "nearest_opposite_is_interface_contact": bool(np.isfinite(best_dist) and best_dist <= interface_threshold),
        }

    interface_pair_df = pd.DataFrame(pair_rows, columns=pair_columns)
    if not interface_pair_df.empty:
        interface_pair_df = interface_pair_df.sort_values(
            by=["min_atom_distance", "antigen_chain_id", "antigen_resseq", "nanobody_chain_id", "nanobody_resseq"],
            ascending=[True, True, True, True, True],
        ).reset_index(drop=True)

    return interface_pair_df, antigen_partner_lookup, nanobody_partner_lookup


def _build_pocket_payload(
    *,
    annotation_df: pd.DataFrame,
    pocket_match: ResidueMatchResult | None,
    pocket_path: Path | None,
    catalytic_uid_set: set[str],
    key_uid_set: set[str],
    interface_pair_df: pd.DataFrame,
) -> dict[str, Any]:
    pocket_df = annotation_df.loc[annotation_df["is_pocket"]].reset_index(drop=True)
    if pocket_df.empty:
        return {
            "pocket_count": 0,
            "pockets": [],
        }

    pocket_residue_uids = set(pocket_df["uid"].astype(str).tolist())
    pocket_pair_df = interface_pair_df.loc[
        interface_pair_df["antigen_uid"].astype(str).isin(pocket_residue_uids)
    ].reset_index(drop=True) if not interface_pair_df.empty else interface_pair_df.copy()

    residue_rows = []
    for _, row in pocket_df.sort_values(by=["chain_id", "resseq", "icode", "resname"]).iterrows():
        residue_rows.append(
            {
                "uid": str(row["uid"]),
                "residue_key": str(row["residue_key"]),
                "chain_id": str(row["chain_id"]),
                "resseq": int(row["resseq"]),
                "icode": str(row["icode"]),
                "resname": str(row["resname"]),
                "is_interface": bool(row["is_interface"]),
                "is_catalytic": bool(row["uid"] in catalytic_uid_set),
                "is_user_key_residue": bool(row["uid"] in key_uid_set),
                "min_distance_to_nanobody": _to_jsonable_scalar(row["min_distance_to_nanobody"]),
                "nearest_opposite_residue_uid": row.get("nearest_opposite_residue_uid"),
                "nearest_opposite_residue_key": row.get("nearest_opposite_residue_key"),
            }
        )

    min_distance = pd.to_numeric(pocket_df["min_distance_to_nanobody"], errors="coerce").to_numpy(dtype=np.float64)
    finite_min_distance = min_distance[np.isfinite(min_distance)]
    mean_distance = float(np.mean(finite_min_distance)) if finite_min_distance.size > 0 else float("nan")
    min_distance_value = float(np.min(finite_min_distance)) if finite_min_distance.size > 0 else float("nan")

    centroid = None
    if pocket_match is not None:
        coords = _match_to_coords(pocket_match)
        center = _mean_coord(coords)
        if center is not None:
            centroid = [float(x) for x in center.tolist()]

    source_type = "none"
    if pocket_path is not None:
        source_type = "file"
    elif pocket_match is not None:
        source_type = "inline"

    pocket_payload = {
        "pocket_id": "pocket_1",
        "label": "Pocket 1",
        "source_type": source_type,
        "source_path": str(pocket_path) if pocket_path is not None else None,
        "residue_count": int(len(pocket_df)),
        "interface_overlap_count": int(pocket_df["is_interface"].sum()),
        "interface_overlap_ratio": float(pocket_df["is_interface"].mean()) if len(pocket_df) > 0 else None,
        "catalytic_overlap_count": int(sum(uid in catalytic_uid_set for uid in pocket_residue_uids)),
        "key_residue_overlap_count": int(sum(uid in key_uid_set for uid in pocket_residue_uids)),
        "contact_pair_count": int(pocket_pair_df.shape[0]),
        "centroid": centroid,
        "min_distance_to_nanobody": min_distance_value if np.isfinite(min_distance_value) else None,
        "mean_distance_to_nanobody": mean_distance if np.isfinite(mean_distance) else None,
        "matched_residue_uids": sorted(pocket_residue_uids),
        "matched_keys": [] if pocket_match is None else sorted(str(x) for x in pocket_match.matched_keys),
        "unmatched_keys": [] if pocket_match is None else sorted(str(x) for x in pocket_match.unmatched_keys),
        "residues": residue_rows,
    }
    return {
        "pocket_count": 1,
        "pockets": [pocket_payload],
    }


def _build_blocking_summary(
    *,
    annotation_df: pd.DataFrame,
    summary: dict[str, Any],
    interface_pair_df: pd.DataFrame,
    interface_threshold: float,
) -> dict[str, Any]:
    pocket_df = annotation_df.loc[annotation_df["is_pocket"]].reset_index(drop=True)
    if pocket_df.empty:
        return {
            "available": False,
            "reason": "No pocket residues matched; blocking summary is unavailable.",
            "components": {},
            "weights": {},
            "blocking_score_simple": None,
            "blocking_confidence_band": "unavailable",
            "explanation": "No pocket residues were defined or matched in the current structure.",
        }

    pocket_count = int(len(pocket_df))
    pocket_contact_count = int(pocket_df["is_interface"].sum())
    pocket_contact_coverage = float(summary.get("pocket_contact_coverage", float("nan")))
    nanobody_support = float(summary.get("nanobody_pocket_contact_ratio", float("nan")))

    pocket_pair_df = interface_pair_df.loc[
        interface_pair_df["antigen_uid"].astype(str).isin(set(pocket_df["uid"].astype(str).tolist()))
    ].reset_index(drop=True) if not interface_pair_df.empty else interface_pair_df.copy()
    contact_pair_count = int(pocket_pair_df.shape[0])
    pair_density = float(np.clip(contact_pair_count / max(pocket_count, 1), 0.0, 1.0))

    min_dist = pd.to_numeric(pocket_df["min_distance_to_nanobody"], errors="coerce").to_numpy(dtype=np.float64)
    finite_min_dist = min_dist[np.isfinite(min_dist)]
    min_distance = float(np.min(finite_min_dist)) if finite_min_dist.size > 0 else float("nan")
    min_distance_closeness = float(np.clip(1.0 - (min_distance / max(interface_threshold, 1e-6)), 0.0, 1.0)) if np.isfinite(min_distance) else float("nan")

    components = {
        "pocket_contact_coverage": pocket_contact_coverage,
        "contact_pair_density": pair_density,
        "min_distance_closeness": min_distance_closeness,
        "nanobody_pocket_support": nanobody_support,
    }
    weights = {
        "pocket_contact_coverage": 0.40,
        "contact_pair_density": 0.25,
        "min_distance_closeness": 0.20,
        "nanobody_pocket_support": 0.15,
    }
    score = _finite_weighted_average(
        [(name, value, weights[name]) for name, value in components.items()]
    )

    if score is None:
        band = "unavailable"
    elif score >= 0.70:
        band = "high"
    elif score >= 0.45:
        band = "medium"
    else:
        band = "low"

    explanation_parts = [
        f"Pocket contact coverage={pocket_contact_count}/{pocket_count}",
        f"interface contact pairs={contact_pair_count}",
    ]
    if np.isfinite(min_distance):
        explanation_parts.append(f"min pocket-to-nanobody distance={min_distance:.2f}A")
    if score is not None:
        explanation_parts.append(f"blocking_score_simple={score:.4f} ({band})")

    return {
        "available": True,
        "components": _to_jsonable(components),
        "weights": _to_jsonable(weights),
        "pocket_contact_count": pocket_contact_count,
        "pocket_residue_count": pocket_count,
        "contact_pair_count": contact_pair_count,
        "min_pocket_to_nanobody_distance": min_distance if np.isfinite(min_distance) else None,
        "blocking_score_simple": score,
        "blocking_confidence_band": band,
        "explanation": "; ".join(explanation_parts),
    }


def main() -> None:
    args = _build_parser().parse_args()

    pdb_path = Path(args.pdb_path).expanduser().resolve()
    if not pdb_path.exists():
        raise FileNotFoundError(f"pdb_path not found: {pdb_path}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    structure = load_complex_pdb(str(pdb_path))
    split = split_antigen_nanobody(
        structure,
        antigen_chain=args.antigen_chain,
        nanobody_chain=args.nanobody_chain,
        fallback_by_ter=True,
    )

    antigen_residues = extract_residues_from_entity(split.antigen)
    nanobody_residues = extract_residues_from_entity(split.nanobody)
    antigen_atoms = extract_atoms_from_entity(split.antigen, heavy_only=True)
    nanobody_atoms = extract_atoms_from_entity(split.nanobody, heavy_only=True)

    antigen_atom_map = _build_residue_atom_map(antigen_atoms)
    nanobody_atom_map = _build_residue_atom_map(nanobody_atoms)
    antigen_coords = _coords_from_atom_infos(antigen_atoms)
    nanobody_coords = _coords_from_atom_infos(nanobody_atoms)

    pocket_path, _, pocket_match = _match_optional_residues(split.antigen, args.pocket_file)
    catalytic_path, _, catalytic_match = _match_optional_residues(split.antigen, args.catalytic_file)
    key_path, _, key_match, key_inline = _match_residue_definition(
        structure,
        path_like=args.key_residue_file,
        inline_text=args.key_residues,
        default_chain=args.key_residue_default_chain,
    )

    pocket_uid_set = _match_to_uid_set(pocket_match)
    catalytic_uid_set = _match_to_uid_set(catalytic_match)
    key_uid_set = _match_to_uid_set(key_match)
    pocket_coords = _match_to_coords(pocket_match)
    pocket_center = _mean_coord(pocket_coords)

    interface_threshold = max(float(args.interface_threshold), 0.0)
    pocket_neighbor_threshold = max(float(args.pocket_neighbor_threshold), 0.0)
    interface_pair_df, antigen_partner_lookup, nanobody_partner_lookup = _compute_interface_pair_details(
        antigen_residues=antigen_residues,
        nanobody_residues=nanobody_residues,
        antigen_atom_map=antigen_atom_map,
        nanobody_atom_map=nanobody_atom_map,
        interface_threshold=interface_threshold,
    )

    antigen_rows = _collect_annotations(
        entity_label="antigen",
        residues=antigen_residues,
        residue_atom_map=antigen_atom_map,
        opposite_coords=nanobody_coords,
        pocket_coords=pocket_coords,
        pocket_center=pocket_center,
        pocket_uid_set=pocket_uid_set,
        catalytic_uid_set=catalytic_uid_set,
        key_uid_set=key_uid_set,
        interface_threshold=interface_threshold,
        pocket_neighbor_threshold=pocket_neighbor_threshold,
        partner_lookup=antigen_partner_lookup,
    )
    nanobody_rows = _collect_annotations(
        entity_label="nanobody",
        residues=nanobody_residues,
        residue_atom_map=nanobody_atom_map,
        opposite_coords=antigen_coords,
        pocket_coords=pocket_coords,
        pocket_center=pocket_center,
        pocket_uid_set=pocket_uid_set,
        catalytic_uid_set=catalytic_uid_set,
        key_uid_set=key_uid_set,
        interface_threshold=interface_threshold,
        pocket_neighbor_threshold=pocket_neighbor_threshold,
        partner_lookup=nanobody_partner_lookup,
    )

    annotation_df = pd.DataFrame(antigen_rows + nanobody_rows)
    if annotation_df.empty:
        raise ValueError("No residue annotations generated.")

    annotation_df = annotation_df.sort_values(
        by=["entity_label", "chain_id", "resseq", "icode", "resname"],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)

    interface_df = annotation_df.loc[annotation_df["is_interface"]].reset_index(drop=True)
    summary = _summarize_annotations(
        annotation_df=annotation_df,
        pocket_uid_set=pocket_uid_set,
        interface_threshold=interface_threshold,
        split_mode=str(getattr(split, "split_mode", "") or getattr(split, "method", "")),
        source_detail=str(getattr(split, "source_detail", "")),
        interface_pair_count=int(interface_pair_df.shape[0]),
        key_uid_set=key_uid_set,
    )
    key_df = annotation_df.loc[annotation_df["is_user_key_residue"]].reset_index(drop=True)
    pocket_payload = _build_pocket_payload(
        annotation_df=annotation_df,
        pocket_match=pocket_match,
        pocket_path=pocket_path,
        catalytic_uid_set=catalytic_uid_set,
        key_uid_set=key_uid_set,
        interface_pair_df=interface_pair_df,
    )
    blocking_summary = _build_blocking_summary(
        annotation_df=annotation_df,
        summary=summary,
        interface_pair_df=interface_pair_df,
        interface_threshold=interface_threshold,
    )

    bundle = {
        "input": {
            "pdb_path": str(pdb_path),
            "antigen_chain": args.antigen_chain,
            "nanobody_chain": args.nanobody_chain,
            "key_residues": args.key_residues,
            "key_residue_file": args.key_residue_file,
            "key_residue_default_chain": args.key_residue_default_chain,
        },
        "thresholds": {
            "interface_threshold": interface_threshold,
            "pocket_neighbor_threshold": pocket_neighbor_threshold,
        },
        "split": {
            "split_mode": summary["split_mode"],
            "source_detail": summary["split_source_detail"],
            "antigen_chain_ids": list(getattr(split.antigen, "chain_ids", ())),
            "nanobody_chain_ids": list(getattr(split.nanobody, "chain_ids", ())),
        },
        "pocket_definition": _serialize_match_result(pocket_match, pocket_path),
        "catalytic_definition": _serialize_match_result(catalytic_match, catalytic_path),
        "key_residue_definition": _serialize_match_result(key_match, key_path, inline_text=key_inline),
        "summary": _to_jsonable(summary),
        "pockets": _to_jsonable(pocket_payload),
        "blocking_summary": _to_jsonable(blocking_summary),
        "interface": {
            "antigen_residue_uids": sorted(
                interface_df.loc[interface_df["entity_label"] == "antigen", "uid"].astype(str).tolist()
            ),
            "nanobody_residue_uids": sorted(
                interface_df.loc[interface_df["entity_label"] == "nanobody", "uid"].astype(str).tolist()
            ),
            "pairs": [_to_jsonable_row(row) for row in interface_pair_df.to_dict(orient="records")],
        },
        "key_residues": {
            "antigen_residue_uids": sorted(
                key_df.loc[key_df["entity_label"] == "antigen", "uid"].astype(str).tolist()
            ),
            "nanobody_residue_uids": sorted(
                key_df.loc[key_df["entity_label"] == "nanobody", "uid"].astype(str).tolist()
            ),
        },
        "residue_annotations": [_to_jsonable_row(row) for row in annotation_df.to_dict(orient="records")],
    }

    annotation_csv = out_dir / "residue_annotations.csv"
    annotation_json = out_dir / "residue_annotations.json"
    interface_csv = out_dir / "interface_residues.csv"
    interface_pair_csv = out_dir / "interface_pairs.csv"
    key_csv = out_dir / "key_residues.csv"
    summary_json = out_dir / "structure_annotation_summary.json"
    pocket_json = out_dir / "pocket_payload.json"
    blocking_json = out_dir / "blocking_summary.json"
    bundle_json = out_dir / "analysis_bundle.json"

    annotation_df.to_csv(annotation_csv, index=False)
    interface_df.to_csv(interface_csv, index=False)
    interface_pair_df.to_csv(interface_pair_csv, index=False)
    key_df.to_csv(key_csv, index=False)
    annotation_json.write_text(
        json.dumps(bundle["residue_annotations"], ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    summary_json.write_text(
        json.dumps(_to_jsonable_row(summary), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    pocket_json.write_text(json.dumps(_to_jsonable(pocket_payload), ensure_ascii=True, indent=2), encoding="utf-8")
    blocking_json.write_text(json.dumps(_to_jsonable(blocking_summary), ensure_ascii=True, indent=2), encoding="utf-8")
    bundle_json.write_text(json.dumps(_to_jsonable(bundle), ensure_ascii=True, indent=2), encoding="utf-8")

    print(f"Saved: {annotation_csv}")
    print(f"Saved: {annotation_json}")
    print(f"Saved: {interface_csv}")
    print(f"Saved: {interface_pair_csv}")
    print(f"Saved: {key_csv}")
    print(f"Saved: {summary_json}")
    print(f"Saved: {pocket_json}")
    print(f"Saved: {blocking_json}")
    print(f"Saved: {bundle_json}")
    print(
        "Interface residues: "
        f"antigen={summary['antigen_interface_count']}, "
        f"nanobody={summary['nanobody_interface_count']}"
    )
    print(f"Interface pairs: {summary['interface_pair_count']}")
    print(f"User key residues: {summary['key_residue_count']}")
    if np.isfinite(float(summary.get("pocket_contact_coverage", float("nan")))):
        print(f"Pocket contact coverage: {float(summary['pocket_contact_coverage']):.4f}")
    if blocking_summary.get("blocking_score_simple") is not None:
        print(
            "Simple blocking score: "
            f"{float(blocking_summary['blocking_score_simple']):.4f} "
            f"({blocking_summary['blocking_confidence_band']})"
        )


if __name__ == "__main__":
    main()
