"""Geometry feature extraction for nanobody pocket-blocking screening.

This module computes geometric features only. It does not include training or
model-related logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, pi
from typing import Any, Iterable

import numpy as np
from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure

try:
    from pocket_io import (
        ResidueKeySet,
        ResidueMatchResult,
        match_residues_in_structure,
        normalize_residue_key,
    )
except Exception:  # pragma: no cover - optional coupling fallback
    ResidueKeySet = None  # type: ignore[assignment]
    ResidueMatchResult = None  # type: ignore[assignment]
    match_residues_in_structure = None  # type: ignore[assignment]
    normalize_residue_key = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class ResidueGeometry:
    """Geometry cache for one residue."""

    key: str
    atom_coords: np.ndarray
    centroid: np.ndarray


@dataclass(slots=True)
class GeometryFeatureConfig:
    """Threshold configuration for geometry feature extraction."""

    atom_contact_threshold: float = 4.5
    catalytic_contact_threshold: float = 4.5
    substrate_clash_threshold: float = 2.8
    mouth_residue_fraction: float = 0.30
    interface_contact_threshold: float = 6.0
    ligand_path_radius: float = 3.0
    catalytic_block_min_hits: int = 1


DEFAULT_CATALYTIC_ANCHOR_SHELL_RADII = (4.0, 6.0, 8.0)


def _is_hydrogen(atom_like: Any) -> bool:
    """Return True if the atom-like object is a hydrogen/deuterium atom."""
    if hasattr(atom_like, "element"):
        element = str(getattr(atom_like, "element", "") or "").strip().upper()
        if element in {"H", "D"}:
            return True

    name = ""
    if hasattr(atom_like, "get_name"):
        try:
            name = str(atom_like.get_name()).strip().upper()
        except Exception:
            name = ""
    elif hasattr(atom_like, "atom_name"):
        name = str(getattr(atom_like, "atom_name")).strip().upper()

    return name.startswith("H") or name.startswith("D")


def _coord_from_item(item: Any) -> np.ndarray | None:
    """Extract a single (3,) coordinate from various atom-like objects."""
    try:
        if isinstance(item, np.ndarray):
            arr = np.asarray(item, dtype=np.float64)
            if arr.ndim == 1 and arr.size == 3:
                if np.isfinite(arr).all():
                    return arr
                return None
            return None

        if isinstance(item, (tuple, list)) and len(item) == 3:
            arr = np.asarray(item, dtype=np.float64)
            if np.isfinite(arr).all():
                return arr
            return None

        if hasattr(item, "get_coord"):
            arr = np.asarray(item.get_coord(), dtype=np.float64).reshape(3)
            if np.isfinite(arr).all():
                return arr
            return None

        if hasattr(item, "coord"):
            arr = np.asarray(getattr(item, "coord"), dtype=np.float64).reshape(3)
            if np.isfinite(arr).all():
                return arr
            return None

        if isinstance(item, dict) and "coord" in item:
            arr = np.asarray(item["coord"], dtype=np.float64).reshape(3)
            if np.isfinite(arr).all():
                return arr
            return None
    except Exception:
        return None

    return None


def _iter_atom_like(data: Any) -> Iterable[Any]:
    """Yield atom-like objects recursively from flexible inputs."""
    if data is None:
        return

    if isinstance(data, np.ndarray):
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim == 1 and arr.size == 3:
            yield arr
            return
        if arr.ndim == 2 and arr.shape[1] == 3:
            for row in arr:
                yield row
            return
        raise ValueError(f"Coordinate array must be shape (3,) or (N,3), got {arr.shape}")

    if isinstance(data, Residue):
        for atom in data.get_atoms():
            yield atom
        return

    if isinstance(data, (Structure, Model, Chain)):
        for atom in data.get_atoms():
            yield atom
        return

    if hasattr(data, "get_atoms") and callable(getattr(data, "get_atoms")):
        for atom in data.get_atoms():
            yield atom
        return

    if isinstance(data, (str, bytes)):
        raise TypeError("String input is not valid atom/coordinate data.")

    if isinstance(data, Iterable):
        for item in data:
            if isinstance(item, (Residue, Structure, Model, Chain)) or hasattr(item, "get_atoms"):
                for atom in _iter_atom_like(item):
                    yield atom
            else:
                yield item
        return

    yield data


def _coords_from_atoms(data: Any, heavy_only: bool = True) -> np.ndarray:
    """Convert flexible atom-like input to clean coordinate array (N, 3)."""
    coords: list[np.ndarray] = []
    for atom_like in _iter_atom_like(data):
        if heavy_only and _is_hydrogen(atom_like):
            continue
        coord = _coord_from_item(atom_like)
        if coord is not None:
            coords.append(coord)

    if not coords:
        return np.empty((0, 3), dtype=np.float64)
    return np.vstack(coords).astype(np.float64, copy=False)


def _safe_residue_key(residue: Residue) -> str:
    """Create canonical residue key including chain/resseq/insertion code."""
    parent = residue.get_parent()
    chain_id = str(parent.id) if isinstance(parent, Chain) else "_"
    _, resseq, icode = residue.id
    icode_text = str(icode).strip()

    if callable(normalize_residue_key):
        try:
            return normalize_residue_key(chain_id, int(resseq), icode_text)
        except Exception:
            pass

    if icode_text:
        return f"{chain_id}:{int(resseq)}:{icode_text}"
    return f"{chain_id}:{int(resseq)}"


def _iter_residues(data: Any) -> list[Residue]:
    """Resolve flexible residue-like input into residue objects."""
    if data is None:
        return []
    if isinstance(data, Residue):
        return [data]
    if isinstance(data, Chain):
        return list(data.get_residues())
    if isinstance(data, Model):
        return list(data.get_residues())
    if isinstance(data, Structure):
        models = list(data.get_models())
        if not models:
            return []
        return list(models[0].get_residues())

    if ResidueMatchResult is not None and isinstance(data, ResidueMatchResult):
        return list(data.matched_residues)

    if isinstance(data, Iterable) and not isinstance(data, (str, bytes)):
        residues: list[Residue] = []
        for item in data:
            if isinstance(item, Residue):
                residues.append(item)
            elif isinstance(item, Chain):
                residues.extend(list(item.get_residues()))
            elif isinstance(item, Model):
                residues.extend(list(item.get_residues()))
            elif isinstance(item, Structure):
                models = list(item.get_models())
                if models:
                    residues.extend(list(models[0].get_residues()))
        return residues

    return []


def _looks_like_residue_key_set(data: Any) -> bool:
    """Return True when data looks like a set/list of residue key strings."""
    if data is None:
        return False
    if ResidueKeySet is not None and isinstance(data, ResidueKeySet):
        return True
    if isinstance(data, (set, tuple, list, frozenset)) and data:
        return all(isinstance(x, str) for x in data)
    return False


def _extract_key_set(data: Any) -> set[str]:
    """Extract string key set from residue-key-like input."""
    if data is None:
        return set()
    if ResidueKeySet is not None and isinstance(data, ResidueKeySet):
        return set(data.residue_keys)
    if isinstance(data, (set, tuple, list, frozenset)):
        return {str(x) for x in data}
    return set()


def _resolve_residue_geometries(
    residues_or_keys: Any,
    residue_reference: Any = None,
) -> list[ResidueGeometry]:
    """Resolve residues (or residue keys) into residue geometry cache objects."""
    residues: list[Residue] = []

    if _looks_like_residue_key_set(residues_or_keys):
        keys = _extract_key_set(residues_or_keys)
        if not keys:
            return []
        if residue_reference is not None and callable(match_residues_in_structure):
            try:
                match = match_residues_in_structure(residue_reference, keys)
                residues = list(match.matched_residues)
            except Exception:
                residues = []
    else:
        residues = _iter_residues(residues_or_keys)

    geometries: list[ResidueGeometry] = []
    for residue in residues:
        hetflag = str(residue.id[0]).strip()
        if hetflag:
            # Skip hetero residues for protein pocket geometry features.
            continue
        atom_coords = _coords_from_atoms(residue, heavy_only=True)
        if atom_coords.shape[0] == 0:
            continue
        centroid = np.mean(atom_coords, axis=0, dtype=np.float64)
        key = _safe_residue_key(residue)
        geometries.append(ResidueGeometry(key=key, atom_coords=atom_coords, centroid=centroid))

    return geometries


def _concat_residue_atoms(residue_geometries: list[ResidueGeometry]) -> np.ndarray:
    """Concatenate atom coordinates from residue geometry objects."""
    if not residue_geometries:
        return np.empty((0, 3), dtype=np.float64)
    parts = [r.atom_coords for r in residue_geometries if r.atom_coords.shape[0] > 0]
    if not parts:
        return np.empty((0, 3), dtype=np.float64)
    return np.vstack(parts).astype(np.float64, copy=False)


def _rowwise_min_distance(coords_a: np.ndarray, coords_b: np.ndarray) -> np.ndarray:
    """Return per-row minimum Euclidean distance from A to B."""
    if coords_a.shape[0] == 0:
        return np.empty((0,), dtype=np.float64)
    if coords_b.shape[0] == 0:
        return np.full((coords_a.shape[0],), np.nan, dtype=np.float64)

    mins = np.full((coords_a.shape[0],), np.inf, dtype=np.float64)
    chunk = 1024
    for start in range(0, coords_a.shape[0], chunk):
        end = min(start + chunk, coords_a.shape[0])
        block = coords_a[start:end]
        diff = block[:, None, :] - coords_b[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff, optimize=True)
        mins[start:end] = np.sqrt(np.min(d2, axis=1))
    return mins


def _global_min_distance(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """Return global minimum distance between two coordinate sets."""
    row_mins = _rowwise_min_distance(coords_a, coords_b)
    if row_mins.size == 0 or np.isnan(row_mins).all():
        return float("nan")
    return float(np.nanmin(row_mins))


def _nan() -> float:
    """Return float NaN helper."""
    return float("nan")


def _radius_label(radius: float) -> str:
    """Convert an Angstrom radius to a stable feature-name token."""
    value = float(radius)
    if abs(value - round(value)) <= 1e-9:
        return f"{int(round(value))}A"
    return f"{value:g}".replace(".", "p").replace("-", "m") + "A"


def _residue_key_sort_key(key: str) -> tuple[str, int, str]:
    """Sort residue keys by chain, residue number, then raw text."""
    text = str(key)
    parts = text.split(":")
    chain = parts[0] if parts else ""
    number = 0
    if len(parts) > 1:
        token = parts[1]
        if token.lstrip("-").isdigit():
            number = int(token)
    return chain, number, text


def _safe_unit_vector(vector: np.ndarray) -> np.ndarray | None:
    """Normalize a 3D vector; return None when the vector is degenerate."""
    arr = np.asarray(vector, dtype=np.float64).reshape(3)
    if not np.isfinite(arr).all():
        return None
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        return None
    return arr / norm


def _weighted_centroid(points: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray | None:
    """Return weighted centroid of finite 3D points."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
        return None

    finite_mask = np.isfinite(pts).all(axis=1)
    if not finite_mask.any():
        return None

    pts = pts[finite_mask]
    if weights is None:
        return np.mean(pts, axis=0, dtype=np.float64)

    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.size != finite_mask.size:
        return np.mean(pts, axis=0, dtype=np.float64)

    w = w[finite_mask]
    w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)
    if float(np.sum(w)) <= 1e-12:
        return np.mean(pts, axis=0, dtype=np.float64)
    return np.average(pts, axis=0, weights=w)


def _build_orthonormal_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build two unit vectors orthogonal to the provided axis."""
    axis_unit = _safe_unit_vector(axis)
    if axis_unit is None:
        axis_unit = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    reference = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(axis_unit, reference))) >= 0.85:
        reference = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    basis_u = np.cross(axis_unit, reference)
    basis_u = _safe_unit_vector(basis_u)
    if basis_u is None:
        basis_u = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    basis_v = np.cross(axis_unit, basis_u)
    basis_v = _safe_unit_vector(basis_v)
    if basis_v is None:
        basis_v = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    return basis_u, basis_v


def _deduplicate_points(points: np.ndarray, min_distance: float = 0.8) -> np.ndarray:
    """Drop near-duplicate sample points while preserving order."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)

    threshold = max(float(min_distance), 1e-6)
    kept: list[np.ndarray] = []
    for point in pts:
        if not np.isfinite(point).all():
            continue
        if not kept:
            kept.append(point)
            continue
        prev = np.vstack(kept).astype(np.float64, copy=False)
        dist = np.linalg.norm(prev - point[None, :], axis=1)
        if np.all(dist > threshold):
            kept.append(point)

    if not kept:
        return np.empty((0, 3), dtype=np.float64)
    return np.vstack(kept).astype(np.float64, copy=False)


def _estimate_point_coverage(
    points: np.ndarray,
    reference_atoms: Any,
    occupancy_radius: float,
) -> tuple[float, float]:
    """Estimate binary coverage fraction and minimum clearance for sample points."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
        return 0.0, _nan()

    ref_coords = _coords_from_atoms(reference_atoms, heavy_only=True)
    if ref_coords.shape[0] == 0:
        return 0.0, _nan()

    min_dist = _rowwise_min_distance(pts, ref_coords)
    finite = np.isfinite(min_dist)
    if not finite.any():
        return 0.0, _nan()

    radius = max(float(occupancy_radius), 0.0)
    coverage = float(np.mean(min_dist[finite] <= radius))
    min_clearance = float(np.nanmin(min_dist[finite]))
    return coverage, min_clearance


def compute_residue_min_distances(
    residue_geometries: list[ResidueGeometry],
    reference_atoms: Any,
    distance_level: str = "atom",
) -> np.ndarray:
    """Compute minimum distances from residues to a reference atom set.

    distance_level:
    - "atom": residue heavy atoms -> reference heavy atoms
    - "centroid": residue centroid -> reference heavy atoms
    """
    if not residue_geometries:
        return np.empty((0,), dtype=np.float64)

    level = str(distance_level).strip().lower()
    if level not in {"atom", "centroid"}:
        raise ValueError("distance_level must be 'atom' or 'centroid'.")

    ref_coords = _coords_from_atoms(reference_atoms, heavy_only=True)
    if ref_coords.shape[0] == 0:
        return np.full((len(residue_geometries),), np.nan, dtype=np.float64)

    out = np.full((len(residue_geometries),), np.nan, dtype=np.float64)
    for idx, geo in enumerate(residue_geometries):
        if level == "atom":
            source = geo.atom_coords
        else:
            source = np.asarray(geo.centroid, dtype=np.float64).reshape(1, 3)
        out[idx] = _global_min_distance(source, ref_coords)
    return out


def estimate_local_exposure_proxy(
    residue_centroid: np.ndarray,
    antigen_atoms: Any,
    neighborhood_radius: float = 8.0,
    k_neighbors: int = 24,
) -> float:
    """Estimate local residue exposure from nearby antigen-atom crowding.

    This is an approximate geometric proxy, not a solvent-accessibility
    calculation. Higher value means the residue is more exposed.
    """
    center = np.asarray(residue_centroid, dtype=np.float64).reshape(3)
    if not np.isfinite(center).all():
        return 0.0

    if isinstance(antigen_atoms, np.ndarray):
        arr = np.asarray(antigen_atoms, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] == 3:
            ag_coords = arr[np.isfinite(arr).all(axis=1)]
        elif arr.ndim == 1 and arr.size == 3:
            ag_coords = arr.reshape(1, 3)
        else:
            ag_coords = np.empty((0, 3), dtype=np.float64)
    else:
        ag_coords = _coords_from_atoms(antigen_atoms, heavy_only=True)

    if ag_coords.shape[0] == 0:
        return 1.0

    dist = np.linalg.norm(ag_coords - center[None, :], axis=1)
    dist = dist[np.isfinite(dist)]
    if dist.size == 0:
        return 1.0

    radius = max(float(neighborhood_radius), 1e-6)
    density_count = float(np.sum(dist <= radius))
    local_density = density_count / max(radius ** 3, 1e-6)

    k = int(np.clip(int(k_neighbors), 4, max(4, dist.size)))
    knn = np.partition(dist, k - 1)[:k]
    knn_mean = float(np.mean(knn)) if knn.size > 0 else radius

    density_term = 1.0 / (1.0 + local_density * 32.0)
    knn_term = float(np.clip(knn_mean / radius, 0.0, 1.0))
    exposure = 0.55 * knn_term + 0.45 * density_term
    return float(np.clip(exposure, 0.0, 1.0))


def _resolve_pocket_geo(
    pocket_residues: Any,
    residue_reference: Any = None,
) -> list[ResidueGeometry]:
    """Resolve pocket-like input to residue geometry list."""
    if isinstance(pocket_residues, list) and pocket_residues and isinstance(pocket_residues[0], ResidueGeometry):
        return list(pocket_residues)
    return _resolve_residue_geometries(pocket_residues, residue_reference=residue_reference)


def sample_pocket_local_points(
    pocket_residues: Any,
    residue_reference: Any = None,
    max_points: int = 4096,
) -> np.ndarray:
    """Sample pocket-local points for occupancy/accessibility proxy features.

    Approximation strategy:
    - include pocket residue centroids
    - include a sparse subset of pocket heavy atoms
    - include outward/inward offset points along center-to-residue directions
    """
    pocket_geo = _resolve_pocket_geo(pocket_residues, residue_reference=residue_reference)
    if not pocket_geo:
        return np.empty((0, 3), dtype=np.float64)

    pocket_center = _compute_pocket_center(pocket_geo)
    if pocket_center is None:
        return np.empty((0, 3), dtype=np.float64)

    points: list[np.ndarray] = []
    for geo in pocket_geo:
        centroid = np.asarray(geo.centroid, dtype=np.float64).reshape(3)
        if np.isfinite(centroid).all():
            points.append(centroid)

        atom_coords = np.asarray(geo.atom_coords, dtype=np.float64)
        if atom_coords.ndim == 2 and atom_coords.shape[1] == 3 and atom_coords.shape[0] > 0:
            step = max(1, int(atom_coords.shape[0] // 4))
            for atom_point in atom_coords[::step][:4]:
                if np.isfinite(atom_point).all():
                    points.append(atom_point)

        radial_vec = centroid - pocket_center
        radial_norm = float(np.linalg.norm(radial_vec))
        if radial_norm > 1e-8:
            unit = radial_vec / radial_norm
            points.append(centroid + unit * 1.2)
            points.append(centroid - unit * 0.8)

    if not points:
        return np.empty((0, 3), dtype=np.float64)

    arr = np.vstack(points).astype(np.float64, copy=False)
    finite_mask = np.isfinite(arr).all(axis=1)
    arr = arr[finite_mask]
    if arr.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)

    max_n = max(1, int(max_points))
    if arr.shape[0] > max_n:
        idx = np.linspace(0, arr.shape[0] - 1, num=max_n, dtype=int)
        arr = arr[idx]
    return arr


def compute_point_occupancy_by_nanobody(
    points: np.ndarray,
    nanobody_atoms: Any,
    occupancy_radius: float = 2.8,
) -> tuple[np.ndarray, float]:
    """Mark pocket-local points occupied by nanobody heavy atoms."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
        return np.zeros((0,), dtype=bool), float("nan")

    nb_coords = _coords_from_atoms(nanobody_atoms, heavy_only=True)
    if nb_coords.shape[0] == 0:
        empty_occ = np.zeros((pts.shape[0],), dtype=bool)
        return empty_occ, 0.0

    radius = max(float(occupancy_radius), 0.0)
    point_min_to_nb = _rowwise_min_distance(pts, nb_coords)
    occupied_mask = np.isfinite(point_min_to_nb) & (point_min_to_nb <= radius)
    occupied_fraction = float(np.mean(occupied_mask)) if occupied_mask.size > 0 else float("nan")
    return occupied_mask, occupied_fraction


def estimate_path_block_score(
    nanobody_atoms: Any,
    path_start: np.ndarray,
    path_end: np.ndarray,
    path_radius: float = 3.0,
) -> float:
    """Estimate static path blocking along a line segment.

    This is a static geometric approximation, not a dynamic transport or MD
    simulation. Score is in [0, 1].
    """
    nb_coords = _coords_from_atoms(nanobody_atoms, heavy_only=True)
    if nb_coords.shape[0] == 0:
        return 0.0

    start = np.asarray(path_start, dtype=np.float64).reshape(3)
    end = np.asarray(path_end, dtype=np.float64).reshape(3)
    if not np.isfinite(start).all() or not np.isfinite(end).all():
        return _nan()

    radius = max(float(path_radius), 1e-6)
    tube_dist, t_raw = _point_to_segment_distances(nb_coords, start, end)

    t_clip = np.clip(t_raw, 0.0, 1.0)
    radial_weight = np.clip(1.0 - tube_dist / radius, 0.0, 1.0)
    axial_weight = np.clip(1.0 - 1.6 * np.abs(t_clip - 0.5), 0.0, 1.0)
    influence = radial_weight * axial_weight

    if influence.size == 0 or float(np.nanmax(influence)) <= 0.0:
        return 0.0

    seg_len = float(np.linalg.norm(end - start))
    norm_factor = max(1.0, seg_len / radius)
    score = 1.0 - float(np.exp(-np.sum(influence) / norm_factor))
    return float(np.clip(score, 0.0, 1.0))


def estimate_path_bottleneck_features(
    nanobody_atoms: Any,
    path_start: np.ndarray,
    path_end: np.ndarray,
    path_radius: float = 3.0,
    n_samples: int = 25,
) -> dict[str, float]:
    """Estimate static path bottleneck features from sampled path points.

    Returns:
    - ligand_path_block_fraction: plain fraction of blocked path samples
    - ligand_path_block_fraction_weighted: center-weighted blocked fraction
    - ligand_path_min_clearance: minimum sampled clearance to nanobody atoms
    - ligand_path_bottleneck_score: normalized bottleneck severity in [0, 1]
    - ligand_path_sample_count: number of sampled path points
    """
    start = np.asarray(path_start, dtype=np.float64).reshape(3)
    end = np.asarray(path_end, dtype=np.float64).reshape(3)
    if not np.isfinite(start).all() or not np.isfinite(end).all():
        return {
            "ligand_path_block_fraction": _nan(),
            "ligand_path_block_fraction_weighted": _nan(),
            "ligand_path_min_clearance": _nan(),
            "ligand_path_bottleneck_score": _nan(),
            "ligand_path_sample_count": 0.0,
        }

    radius = max(float(path_radius), 1e-6)
    n = max(5, int(n_samples))
    t = np.linspace(0.0, 1.0, num=n, dtype=np.float64)
    seg = end - start
    sample_points = start[None, :] + t[:, None] * seg[None, :]

    result = {
        "ligand_path_block_fraction": 0.0,
        "ligand_path_block_fraction_weighted": 0.0,
        "ligand_path_min_clearance": _nan(),
        "ligand_path_bottleneck_score": 0.0,
        "ligand_path_sample_count": float(n),
    }

    nb_coords = _coords_from_atoms(nanobody_atoms, heavy_only=True)
    if nb_coords.shape[0] == 0:
        return result

    min_clearance = _rowwise_min_distance(sample_points, nb_coords)
    finite = np.isfinite(min_clearance)
    if not finite.any():
        return result

    blocked = np.zeros((n,), dtype=np.float64)
    blocked[finite] = (min_clearance[finite] <= radius).astype(np.float64)

    plain_fraction = float(np.mean(blocked[finite]))

    # Center-weighted ratio emphasizes bottleneck around the middle path section.
    center_weight = np.clip(1.0 - np.abs(2.0 * t - 1.0), 0.15, 1.0)
    center_weight = np.where(finite, center_weight, 0.0)
    if float(np.sum(center_weight)) > 1e-12:
        weighted_fraction = float(np.sum(blocked * center_weight) / np.sum(center_weight))
    else:
        weighted_fraction = plain_fraction

    min_clear = float(np.nanmin(min_clearance[finite]))
    bottleneck_score = float(np.clip(1.0 - min_clear / (radius * 1.25), 0.0, 1.0))

    result["ligand_path_block_fraction"] = plain_fraction
    result["ligand_path_block_fraction_weighted"] = weighted_fraction
    result["ligand_path_min_clearance"] = min_clear
    result["ligand_path_bottleneck_score"] = bottleneck_score
    return result


def compute_pocket_occupancy_features(
    nanobody_atoms: Any,
    pocket_residues: Any,
    residue_reference: Any = None,
    occupancy_radius: float = 2.8,
    baseline_occupancy: float | None = None,
) -> dict[str, float]:
    """Compute pocket occupancy/accessibility proxy features.

    These are approximate proxy features derived from static geometry around the
    pocket region in the complex structure.
    """
    result = {
        "delta_pocket_access_proxy": _nan(),
        "delta_pocket_occupancy_proxy": _nan(),
        "pocket_block_volume_proxy": _nan(),
        "pocket_local_point_count": 0.0,
        "pocket_local_occupied_count": 0.0,
    }

    pocket_geo = _resolve_pocket_geo(pocket_residues, residue_reference=residue_reference)
    if not pocket_geo:
        return result

    local_points = sample_pocket_local_points(pocket_geo)
    result["pocket_local_point_count"] = float(local_points.shape[0])
    if local_points.shape[0] == 0:
        return result

    occupied_mask, occ_fraction = compute_point_occupancy_by_nanobody(
        points=local_points,
        nanobody_atoms=nanobody_atoms,
        occupancy_radius=occupancy_radius,
    )
    occupied_count = int(np.sum(occupied_mask)) if occupied_mask.size > 0 else 0
    result["pocket_local_occupied_count"] = float(occupied_count)

    if not np.isfinite(occ_fraction):
        return result

    base_occ = 0.0 if baseline_occupancy is None else float(baseline_occupancy)
    if not np.isfinite(base_occ):
        base_occ = 0.0
    base_occ = float(np.clip(base_occ, 0.0, 1.0))

    delta_occ = float(occ_fraction - base_occ)
    delta_access = float(-delta_occ)

    pocket_center = _compute_pocket_center(pocket_geo)
    if pocket_center is None:
        block_volume = _nan()
    else:
        radial = np.linalg.norm(local_points - pocket_center[None, :], axis=1)
        radial = radial[np.isfinite(radial)]
        if radial.size == 0:
            block_volume = _nan()
        else:
            local_radius = max(float(np.nanpercentile(radial, 75)), 1e-3)
            local_volume = (4.0 / 3.0) * pi * (local_radius ** 3)
            block_volume = float(max(0.0, occ_fraction) * local_volume)

    result["delta_pocket_access_proxy"] = delta_access
    result["delta_pocket_occupancy_proxy"] = delta_occ
    result["pocket_block_volume_proxy"] = block_volume
    return result


def compute_pocket_features(
    nanobody_atoms: Any,
    pocket_residues: Any,
    atom_contact_threshold: float = 4.5,
    residue_reference: Any = None,
) -> dict[str, float]:
    """Compute pocket-contact geometry features.

    Features:
    - pocket_hit_count
    - pocket_hit_fraction
    - pocket_atom_contact_count
    - min_distance_to_pocket
    - mean_min_distance_to_pocket_residues
    """
    threshold = max(float(atom_contact_threshold), 0.0)
    nb_coords = _coords_from_atoms(nanobody_atoms, heavy_only=True)
    pocket_geo = _resolve_pocket_geo(pocket_residues, residue_reference=residue_reference)
    residue_count = len(pocket_geo)
    result = {
        "pocket_hit_count": 0.0,
        "pocket_hit_fraction": _nan(),
        "pocket_atom_contact_count": 0.0,
        "min_distance_to_pocket": _nan(),
        "mean_min_distance_to_pocket_residues": _nan(),
    }

    if residue_count == 0:
        return result

    if nb_coords.shape[0] == 0:
        result["pocket_hit_fraction"] = 0.0
        return result

    pocket_atom_coords = _concat_residue_atoms(pocket_geo)
    if pocket_atom_coords.shape[0] == 0:
        result["pocket_hit_fraction"] = 0.0
        return result

    # Atom-level distances: pocket heavy atoms -> nanobody heavy atoms.
    atom_min_to_nb = _rowwise_min_distance(pocket_atom_coords, nb_coords)
    result["pocket_atom_contact_count"] = float(np.sum(atom_min_to_nb <= threshold))
    result["min_distance_to_pocket"] = _global_min_distance(nb_coords, pocket_atom_coords)

    # Residue-level distances: each pocket residue (atom cloud) -> nanobody atoms.
    residue_min_dists = compute_residue_min_distances(
        residue_geometries=pocket_geo,
        reference_atoms=nb_coords,
        distance_level="atom",
    )
    hit_mask = np.isfinite(residue_min_dists) & (residue_min_dists <= threshold)
    hit_count = int(np.sum(hit_mask))

    result["pocket_hit_count"] = float(hit_count)
    result["pocket_hit_fraction"] = float(hit_count / residue_count) if residue_count > 0 else _nan()
    if residue_min_dists.size > 0:
        result["mean_min_distance_to_pocket_residues"] = float(np.nanmean(residue_min_dists))
    return result


def compute_pocket_shape_features(
    pocket_residues: Any,
    residue_reference: Any = None,
) -> dict[str, float]:
    """Describe pocket input size/spread without changing downstream scoring.

    These features are meant as QC/proxy inputs. A high overwide proxy does not
    mean the pocket is wrong; it means the residue definition is broad enough
    that downstream blocking scores may need tighter interpretation.
    """
    result = {
        "pocket_shape_residue_count": 0.0,
        "pocket_shape_centroid_radius_mean": _nan(),
        "pocket_shape_centroid_radius_p90": _nan(),
        "pocket_shape_centroid_radius_max": _nan(),
        "pocket_shape_atom_bbox_volume": _nan(),
        "pocket_shape_overwide_proxy": _nan(),
        "pocket_shape_tightness_proxy": _nan(),
    }

    pocket_geo = _resolve_pocket_geo(pocket_residues, residue_reference=residue_reference)
    result["pocket_shape_residue_count"] = float(len(pocket_geo))
    if not pocket_geo:
        return result

    centroids = np.vstack([geo.centroid for geo in pocket_geo]).astype(np.float64, copy=False)
    centroids = centroids[np.isfinite(centroids).all(axis=1)]
    if centroids.shape[0] == 0:
        return result

    pocket_center = np.mean(centroids, axis=0, dtype=np.float64)
    centroid_radius = np.linalg.norm(centroids - pocket_center[None, :], axis=1)
    centroid_radius = centroid_radius[np.isfinite(centroid_radius)]
    if centroid_radius.size > 0:
        result["pocket_shape_centroid_radius_mean"] = float(np.mean(centroid_radius))
        result["pocket_shape_centroid_radius_p90"] = float(np.percentile(centroid_radius, 90))
        result["pocket_shape_centroid_radius_max"] = float(np.max(centroid_radius))

    atom_coords = _concat_residue_atoms(pocket_geo)
    bbox_volume = _nan()
    if atom_coords.shape[0] > 0:
        finite_atoms = atom_coords[np.isfinite(atom_coords).all(axis=1)]
        if finite_atoms.shape[0] > 0:
            span = np.max(finite_atoms, axis=0) - np.min(finite_atoms, axis=0)
            span = np.clip(span, 0.0, None)
            bbox_volume = float(np.prod(span))
            result["pocket_shape_atom_bbox_volume"] = bbox_volume

    residue_count = float(len(pocket_geo))
    count_penalty = float(np.clip((residue_count - 12.0) / 24.0, 0.0, 1.0))
    radius_p90 = result["pocket_shape_centroid_radius_p90"]
    radius_penalty = float(np.clip((radius_p90 - 9.0) / 12.0, 0.0, 1.0)) if np.isfinite(radius_p90) else 0.0
    volume_penalty = float(np.clip((np.log1p(bbox_volume) - 6.0) / 3.0, 0.0, 1.0)) if np.isfinite(bbox_volume) else 0.0

    overwide_proxy = float(np.clip(0.45 * count_penalty + 0.35 * radius_penalty + 0.20 * volume_penalty, 0.0, 1.0))
    result["pocket_shape_overwide_proxy"] = overwide_proxy
    result["pocket_shape_tightness_proxy"] = float(np.clip(1.0 - overwide_proxy, 0.0, 1.0))
    return result


def compute_catalytic_features(
    nanobody_atoms: Any,
    catalytic_residues: Any = None,
    catalytic_contact_threshold: float = 4.5,
    residue_reference: Any = None,
    catalytic_block_min_hits: int = 1,
) -> dict[str, float]:
    """Compute catalytic-site blocking geometry features.

    Features:
    - catalytic_hit_count
    - catalytic_hit_fraction
    - min_distance_to_catalytic_residues
    - catalytic_block_flag
    """
    threshold = max(float(catalytic_contact_threshold), 0.0)
    min_hits = max(int(catalytic_block_min_hits), 1)
    nb_coords = _coords_from_atoms(nanobody_atoms, heavy_only=True)
    catalytic_geo = _resolve_pocket_geo(catalytic_residues, residue_reference=residue_reference)
    cat_atoms = _concat_residue_atoms(catalytic_geo)

    residue_count = len(catalytic_geo)
    result = {
        "catalytic_hit_count": 0.0,
        "catalytic_hit_fraction": _nan(),
        "min_distance_to_catalytic_residues": _nan(),
        "catalytic_block_flag": 0.0,
    }

    if residue_count == 0:
        return result

    if nb_coords.shape[0] == 0:
        result["catalytic_hit_fraction"] = 0.0
        return result

    result["min_distance_to_catalytic_residues"] = _global_min_distance(nb_coords, cat_atoms)

    residue_min_dists = compute_residue_min_distances(
        residue_geometries=catalytic_geo,
        reference_atoms=nb_coords,
        distance_level="atom",
    )
    hit_mask = np.isfinite(residue_min_dists) & (residue_min_dists <= threshold)
    hit_count = int(np.sum(hit_mask))

    hit_fraction = float(hit_count / residue_count)
    result["catalytic_hit_count"] = float(hit_count)
    result["catalytic_hit_fraction"] = hit_fraction
    result["catalytic_block_flag"] = 1.0 if hit_count >= min_hits else 0.0
    return result


def _coerce_shell_radii(radii: Any) -> tuple[float, ...]:
    """Normalize user/config shell radii and keep stable ordering."""
    if radii is None:
        return DEFAULT_CATALYTIC_ANCHOR_SHELL_RADII
    if isinstance(radii, str):
        raw_items = [item.strip() for item in radii.split(",") if item.strip()]
    else:
        try:
            raw_items = list(radii)
        except TypeError:
            raw_items = [radii]

    clean: list[float] = []
    for item in raw_items:
        try:
            value = float(item)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and value > 0.0:
            clean.append(value)

    if not clean:
        return DEFAULT_CATALYTIC_ANCHOR_SHELL_RADII
    return tuple(sorted(set(clean)))


def compute_catalytic_anchor_pocket_features(
    antigen_residues: Any,
    nanobody_atoms: Any,
    catalytic_residues: Any = None,
    pocket_residues: Any = None,
    residue_reference: Any = None,
    shell_radii: Any = DEFAULT_CATALYTIC_ANCHOR_SHELL_RADII,
    primary_radius: float = 6.0,
    contact_threshold: float = 4.5,
) -> dict[str, Any]:
    """Infer enzyme active-site pocket diagnostics from catalytic anchors.

    The method treats literature/user catalytic residues as high-confidence
    spatial anchors, then creates antigen-residue shells around those anchors.
    It is intentionally diagnostic: it does not replace manual pocket input or
    assign formal ranking weights.
    """
    radii = _coerce_shell_radii(shell_radii)
    primary = float(primary_radius)
    if not np.isfinite(primary) or primary <= 0.0:
        primary = 6.0
    if not any(abs(primary - radius) <= 1e-9 for radius in radii):
        radii = tuple(sorted(set((*radii, primary))))

    threshold = max(float(contact_threshold), 0.0)
    result: dict[str, Any] = {
        "catalytic_anchor_has_input": 0.0,
        "catalytic_anchor_core_residue_count": 0.0,
        "catalytic_anchor_antigen_residue_count": 0.0,
        "catalytic_anchor_primary_radius": primary,
        "catalytic_anchor_primary_shell_residue_count": 0.0,
        "catalytic_anchor_primary_shell_fraction_of_antigen": _nan(),
        "catalytic_anchor_primary_shell_hit_count": 0.0,
        "catalytic_anchor_primary_shell_hit_fraction": _nan(),
        "catalytic_anchor_min_distance_to_primary_shell": _nan(),
        "catalytic_anchor_manual_overlap_count": 0.0,
        "catalytic_anchor_manual_overlap_fraction_of_shell": _nan(),
        "catalytic_anchor_manual_overlap_fraction_of_pocket": _nan(),
        "catalytic_anchor_shell_overwide_proxy": _nan(),
        "catalytic_anchor_shell_overwide_flag": 0.0,
        "catalytic_anchor_primary_shell_keys": "",
    }
    for radius in radii:
        label = _radius_label(radius)
        result[f"catalytic_anchor_shell_{label}_residue_count"] = 0.0
        result[f"catalytic_anchor_shell_{label}_fraction_of_antigen"] = _nan()
        result[f"catalytic_anchor_shell_{label}_hit_count"] = 0.0
        result[f"catalytic_anchor_shell_{label}_hit_fraction"] = _nan()
        result[f"catalytic_anchor_shell_{label}_min_distance_to_nanobody"] = _nan()

    reference = residue_reference if residue_reference is not None else antigen_residues
    all_geo = _resolve_pocket_geo(antigen_residues, residue_reference=reference)
    catalytic_geo = _resolve_pocket_geo(catalytic_residues, residue_reference=reference)
    if not all_geo or not catalytic_geo:
        return result

    catalytic_atoms = _concat_residue_atoms(catalytic_geo)
    if catalytic_atoms.shape[0] == 0:
        return result

    result["catalytic_anchor_has_input"] = 1.0
    result["catalytic_anchor_core_residue_count"] = float(len(catalytic_geo))
    result["catalytic_anchor_antigen_residue_count"] = float(len(all_geo))

    residue_to_anchor_dist = compute_residue_min_distances(
        residue_geometries=all_geo,
        reference_atoms=catalytic_atoms,
        distance_level="atom",
    )
    nb_coords = _coords_from_atoms(nanobody_atoms, heavy_only=True)
    pocket_geo = _resolve_pocket_geo(pocket_residues, residue_reference=reference)
    pocket_keys = {geo.key for geo in pocket_geo}

    shell_by_radius: dict[float, list[ResidueGeometry]] = {}
    for radius in radii:
        mask = np.isfinite(residue_to_anchor_dist) & (residue_to_anchor_dist <= float(radius))
        shell_geo = [geo for geo, keep in zip(all_geo, mask) if bool(keep)]
        shell_by_radius[float(radius)] = shell_geo
        label = _radius_label(float(radius))
        shell_count = len(shell_geo)
        result[f"catalytic_anchor_shell_{label}_residue_count"] = float(shell_count)
        result[f"catalytic_anchor_shell_{label}_fraction_of_antigen"] = float(shell_count / max(len(all_geo), 1))

        shell_atoms = _concat_residue_atoms(shell_geo)
        if shell_count > 0 and nb_coords.shape[0] > 0 and shell_atoms.shape[0] > 0:
            result[f"catalytic_anchor_shell_{label}_min_distance_to_nanobody"] = _global_min_distance(
                nb_coords,
                shell_atoms,
            )
            shell_min_dists = compute_residue_min_distances(
                residue_geometries=shell_geo,
                reference_atoms=nb_coords,
                distance_level="atom",
            )
            hit_mask = np.isfinite(shell_min_dists) & (shell_min_dists <= threshold)
            hit_count = int(np.sum(hit_mask))
            result[f"catalytic_anchor_shell_{label}_hit_count"] = float(hit_count)
            result[f"catalytic_anchor_shell_{label}_hit_fraction"] = float(hit_count / shell_count)
        elif shell_count > 0:
            result[f"catalytic_anchor_shell_{label}_hit_fraction"] = 0.0

    primary_key = min(shell_by_radius, key=lambda radius: abs(radius - primary))
    primary_geo = shell_by_radius.get(primary_key, [])
    primary_label = _radius_label(primary_key)
    primary_keys = {geo.key for geo in primary_geo}
    result["catalytic_anchor_primary_radius"] = float(primary_key)
    result["catalytic_anchor_primary_shell_residue_count"] = result[
        f"catalytic_anchor_shell_{primary_label}_residue_count"
    ]
    result["catalytic_anchor_primary_shell_fraction_of_antigen"] = result[
        f"catalytic_anchor_shell_{primary_label}_fraction_of_antigen"
    ]
    result["catalytic_anchor_primary_shell_hit_count"] = result[
        f"catalytic_anchor_shell_{primary_label}_hit_count"
    ]
    result["catalytic_anchor_primary_shell_hit_fraction"] = result[
        f"catalytic_anchor_shell_{primary_label}_hit_fraction"
    ]
    result["catalytic_anchor_min_distance_to_primary_shell"] = result[
        f"catalytic_anchor_shell_{primary_label}_min_distance_to_nanobody"
    ]
    result["catalytic_anchor_primary_shell_keys"] = ";".join(
        sorted(primary_keys, key=_residue_key_sort_key)
    )

    if primary_keys and pocket_keys:
        overlap_count = len(primary_keys & pocket_keys)
        result["catalytic_anchor_manual_overlap_count"] = float(overlap_count)
        result["catalytic_anchor_manual_overlap_fraction_of_shell"] = float(overlap_count / max(len(primary_keys), 1))
        result["catalytic_anchor_manual_overlap_fraction_of_pocket"] = float(overlap_count / max(len(pocket_keys), 1))

    shell_count = float(len(primary_geo))
    shell_fraction = shell_count / max(float(len(all_geo)), 1.0)
    count_penalty = float(np.clip((shell_count - 24.0) / 48.0, 0.0, 1.0))
    fraction_penalty = float(np.clip((shell_fraction - 0.18) / 0.32, 0.0, 1.0))
    radius_penalty = float(np.clip((primary_key - 6.0) / 6.0, 0.0, 1.0))
    overwide_proxy = float(np.clip(0.45 * count_penalty + 0.40 * fraction_penalty + 0.15 * radius_penalty, 0.0, 1.0))
    result["catalytic_anchor_shell_overwide_proxy"] = overwide_proxy
    result["catalytic_anchor_shell_overwide_flag"] = 1.0 if overwide_proxy >= 0.55 else 0.0
    return result


def _compute_pocket_center(pocket_geo: list[ResidueGeometry]) -> np.ndarray | None:
    """Compute pocket center from pocket residue atom coordinates."""
    pocket_atoms = _concat_residue_atoms(pocket_geo)
    if pocket_atoms.shape[0] == 0:
        return None
    return np.mean(pocket_atoms, axis=0, dtype=np.float64)


def compute_pocket_center_features(
    antigen_atoms: Any,
    nanobody_atoms: Any,
    pocket_residues: Any,
    residue_reference: Any = None,
    interface_contact_threshold: float = 6.0,
) -> dict[str, float]:
    """Compute pocket-center related geometric features.

    Features:
    - distance_to_pocket_center
    - nanobody_centroid_to_pocket_center
    - distance_interface_centroid_to_pocket_center
    """
    interface_thr = max(float(interface_contact_threshold), 0.0)
    nb_coords = _coords_from_atoms(nanobody_atoms, heavy_only=True)
    ag_coords = _coords_from_atoms(antigen_atoms, heavy_only=True)
    pocket_geo = _resolve_pocket_geo(pocket_residues, residue_reference=residue_reference)
    pocket_center = _compute_pocket_center(pocket_geo)

    result = {
        "distance_to_pocket_center": _nan(),
        "nanobody_centroid_to_pocket_center": _nan(),
        "distance_interface_centroid_to_pocket_center": _nan(),
    }

    if pocket_center is None or nb_coords.shape[0] == 0:
        return result

    # Atom-level distance: nanobody heavy atoms -> pocket center.
    d_nb_to_center = np.linalg.norm(nb_coords - pocket_center[None, :], axis=1)
    result["distance_to_pocket_center"] = float(np.min(d_nb_to_center))

    # Centroid-level distance: nanobody centroid -> pocket center.
    nb_centroid = np.mean(nb_coords, axis=0, dtype=np.float64)
    result["nanobody_centroid_to_pocket_center"] = float(np.linalg.norm(nb_centroid - pocket_center))

    if ag_coords.shape[0] == 0:
        return result

    nb_min_to_antigen = _rowwise_min_distance(nb_coords, ag_coords)
    interface_mask = np.isfinite(nb_min_to_antigen) & (nb_min_to_antigen <= interface_thr)
    if not np.any(interface_mask):
        return result

    # Interface-centroid-level distance: interface subset centroid -> pocket center.
    interface_centroid = np.mean(nb_coords[interface_mask], axis=0, dtype=np.float64)
    result["distance_interface_centroid_to_pocket_center"] = float(
        np.linalg.norm(interface_centroid - pocket_center)
    )
    return result


def infer_mouth_residues(
    pocket_residues: Any,
    mouth_residue_fraction: float = 0.30,
    residue_reference: Any = None,
) -> list[ResidueGeometry]:
    """Infer pocket-mouth residues from radial and exposure-like proxies.

    This is still an approximate geometric method (not explicit solvent or
    dynamics), but more informative than simple outermost-radius sorting.
    """
    chosen_geo, _ = _infer_mouth_candidates(
        pocket_residues=pocket_residues,
        mouth_residue_fraction=mouth_residue_fraction,
        residue_reference=residue_reference,
    )
    return chosen_geo


def _infer_mouth_candidates(
    pocket_residues: Any,
    mouth_residue_fraction: float = 0.30,
    residue_reference: Any = None,
) -> tuple[list[ResidueGeometry], np.ndarray]:
    """Infer mouth candidates with combined radial/surface/exposure scoring."""
    ratio = float(mouth_residue_fraction)
    ratio = float(np.clip(ratio, 0.0, 1.0))

    pocket_geo = _resolve_pocket_geo(pocket_residues, residue_reference=residue_reference)
    if not pocket_geo or ratio <= 0.0:
        return [], np.empty((0,), dtype=np.float64)

    pocket_center = _compute_pocket_center(pocket_geo)
    if pocket_center is None:
        return [], np.empty((0,), dtype=np.float64)

    radial_dist = np.asarray(
        [float(np.linalg.norm(geo.centroid - pocket_center)) for geo in pocket_geo],
        dtype=np.float64,
    )
    if radial_dist.size == 0:
        return [], np.empty((0,), dtype=np.float64)

    # Higher radial distance generally indicates closer-to-mouth residues.
    radial_rank = np.argsort(np.argsort(radial_dist)).astype(np.float64)
    if len(radial_rank) > 1:
        radial_rank = radial_rank / float(len(radial_rank) - 1)

    centroids = np.vstack([geo.centroid for geo in pocket_geo]).astype(np.float64, copy=False)
    if centroids.shape[0] > 1:
        diff = centroids[:, None, :] - centroids[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff, optimize=True)
        dmat = np.sqrt(np.maximum(d2, 0.0))
        local_neighbor_count = np.sum((dmat <= 7.5) & (dmat > 1e-8), axis=1).astype(np.float64)
        max_neighbors = max(float(np.max(local_neighbor_count)), 1.0)
        surface_score = 1.0 - (local_neighbor_count / max_neighbors)
    else:
        surface_score = np.ones((centroids.shape[0],), dtype=np.float64)

    antigen_context = (
        _coords_from_atoms(residue_reference, heavy_only=True)
        if residue_reference is not None
        else np.empty((0, 3), dtype=np.float64)
    )
    exposure_score = np.asarray(
        [estimate_local_exposure_proxy(geo.centroid, antigen_context) for geo in pocket_geo],
        dtype=np.float64,
    )

    # Combined approximate mouth score:
    # - radial_rank: farther from pocket center
    # - surface_score: fewer pocket-neighbor residues nearby
    # - exposure_score: lower local antigen crowding
    mouth_score = (
        0.50 * np.clip(radial_rank, 0.0, 1.0)
        + 0.30 * np.clip(surface_score, 0.0, 1.0)
        + 0.20 * np.clip(exposure_score, 0.0, 1.0)
    )

    select_n = max(1, int(ceil(ratio * len(pocket_geo))))
    order = np.argsort(-mouth_score)
    chosen = [pocket_geo[int(idx)] for idx in order[:select_n]]
    return chosen, mouth_score[order[:select_n]]


def _build_mouth_axis_geometry(
    pocket_geo: list[ResidueGeometry],
    mouth_geo: list[ResidueGeometry],
    mouth_weights: np.ndarray,
) -> dict[str, Any]:
    """Build mouth-axis geometry used by tighter occlusion and path proxies."""
    pocket_center = _compute_pocket_center(pocket_geo)
    if pocket_center is None or not mouth_geo:
        return {
            "pocket_center": pocket_center,
            "mouth_center": None,
            "mouth_axis": None,
            "mouth_aperture_radius": _nan(),
        }

    mouth_centroids = np.vstack([geo.centroid for geo in mouth_geo]).astype(np.float64, copy=False)
    mouth_center = _weighted_centroid(mouth_centroids, mouth_weights)
    if mouth_center is None:
        mouth_center = np.mean(mouth_centroids, axis=0, dtype=np.float64)

    mouth_axis = _safe_unit_vector(mouth_center - pocket_center)
    if mouth_axis is None:
        for geo in mouth_geo:
            mouth_axis = _safe_unit_vector(geo.centroid - pocket_center)
            if mouth_axis is not None:
                break

    if mouth_axis is None:
        mouth_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    mouth_atoms = _concat_residue_atoms(mouth_geo)
    geometry_points = mouth_atoms if mouth_atoms.shape[0] > 0 else mouth_centroids
    rel = geometry_points - mouth_center[None, :]
    axial = rel @ mouth_axis
    planar = rel - axial[:, None] * mouth_axis[None, :]
    planar_dist = np.linalg.norm(planar, axis=1)
    finite_planar = planar_dist[np.isfinite(planar_dist)]

    if finite_planar.size > 0:
        aperture_radius = float(np.nanpercentile(finite_planar, 70))
    else:
        centroid_dist = np.linalg.norm(mouth_centroids - mouth_center[None, :], axis=1)
        finite_centroid = centroid_dist[np.isfinite(centroid_dist)]
        aperture_radius = float(np.nanpercentile(finite_centroid, 60)) if finite_centroid.size > 0 else _nan()

    if (not np.isfinite(aperture_radius)) or (aperture_radius < 1.0):
        aperture_radius = 2.6
    aperture_radius = float(np.clip(aperture_radius, 1.5, 8.0))

    return {
        "pocket_center": pocket_center,
        "mouth_center": mouth_center,
        "mouth_axis": mouth_axis,
        "mouth_aperture_radius": aperture_radius,
    }


def _sample_mouth_axis_points(
    mouth_center: np.ndarray,
    mouth_axis: np.ndarray,
    aperture_radius: float,
    n_samples: int = 9,
) -> np.ndarray:
    """Sample points along the inferred mouth axis from pocket side to outside."""
    center = np.asarray(mouth_center, dtype=np.float64).reshape(3)
    axis = _safe_unit_vector(mouth_axis)
    if axis is None or not np.isfinite(center).all():
        return np.empty((0, 3), dtype=np.float64)

    radius = max(float(aperture_radius), 1.0)
    inside_span = max(0.8, 0.55 * radius)
    outside_span = max(1.4, 1.20 * radius)
    offsets = np.linspace(-inside_span, outside_span, num=max(5, int(n_samples)), dtype=np.float64)
    return center[None, :] + offsets[:, None] * axis[None, :]


def _sample_mouth_aperture_points(
    mouth_center: np.ndarray,
    mouth_axis: np.ndarray,
    aperture_radius: float,
) -> np.ndarray:
    """Sample points across the inferred mouth aperture plane and just outside it."""
    center = np.asarray(mouth_center, dtype=np.float64).reshape(3)
    axis = _safe_unit_vector(mouth_axis)
    if axis is None or not np.isfinite(center).all():
        return np.empty((0, 3), dtype=np.float64)

    radius = max(float(aperture_radius), 1.0)
    basis_u, basis_v = _build_orthonormal_basis(axis)
    points: list[np.ndarray] = [center]

    axial_shifts = [0.0, 0.35 * radius]
    for axial_shift in axial_shifts:
        shifted_center = center + axial_shift * axis
        points.append(shifted_center)
        for radial_frac, n_angles in [(0.45, 6), (0.85, 10)]:
            ring_radius = radius * radial_frac
            for angle in np.linspace(0.0, 2.0 * pi, num=n_angles, endpoint=False, dtype=np.float64):
                direction = np.cos(angle) * basis_u + np.sin(angle) * basis_v
                points.append(shifted_center + ring_radius * direction)

    arr = np.vstack(points).astype(np.float64, copy=False)
    return _deduplicate_points(arr, min_distance=max(0.4, 0.10 * radius))


def compute_mouth_occlusion_features(
    nanobody_atoms: Any,
    pocket_residues: Any,
    atom_contact_threshold: float = 4.5,
    mouth_residue_fraction: float = 0.30,
    residue_reference: Any = None,
) -> dict[str, float]:
    """Compute mouth-occlusion features using improved mouth inference.

    Features:
    - mouth_hit_count
    - mouth_hit_fraction
    - mouth_occlusion_score

    Notes:
    - This is an approximate static geometry score.
    - It is not a dynamic opening/closing simulation.
    """
    threshold = max(float(atom_contact_threshold), 0.0)
    nb_coords = _coords_from_atoms(nanobody_atoms, heavy_only=True)
    mouth_geo, mouth_weights = _infer_mouth_candidates(
        pocket_residues=pocket_residues,
        mouth_residue_fraction=mouth_residue_fraction,
        residue_reference=residue_reference,
    )

    result = {
        "mouth_hit_count": 0.0,
        "mouth_hit_fraction": _nan(),
        "mouth_occlusion_score": _nan(),
        "mouth_axis_block_fraction": _nan(),
        "mouth_aperture_block_fraction": _nan(),
        "mouth_min_clearance": _nan(),
    }

    if not mouth_geo:
        return result
    if nb_coords.shape[0] == 0:
        result["mouth_hit_fraction"] = 0.0
        result["mouth_occlusion_score"] = 0.0
        result["mouth_axis_block_fraction"] = 0.0
        result["mouth_aperture_block_fraction"] = 0.0
        return result

    min_dists = compute_residue_min_distances(
        residue_geometries=mouth_geo,
        reference_atoms=nb_coords,
        distance_level="atom",
    )
    hit_mask = np.isfinite(min_dists) & (min_dists <= threshold)
    hit_count = int(np.sum(hit_mask))

    mouth_count = len(mouth_geo)
    result["mouth_hit_count"] = float(hit_count)
    result["mouth_hit_fraction"] = float(hit_count / mouth_count) if mouth_count > 0 else _nan()

    softness = threshold + 1.5
    residue_occlusion_score = 0.0
    if softness <= 1e-8:
        residue_occlusion_score = 0.0
    else:
        dist_arr = np.asarray(min_dists, dtype=np.float64)
        score_arr = np.clip(1.0 - dist_arr / softness, 0.0, 1.0)
        weights = np.asarray(mouth_weights, dtype=np.float64)
        if weights.size != score_arr.size:
            weights = np.ones_like(score_arr)
        weights = np.where(np.isfinite(weights), weights, 0.0)
        if np.sum(weights) <= 1e-12:
            residue_occlusion_score = float(np.nanmean(score_arr))
        else:
            residue_occlusion_score = float(np.nansum(score_arr * weights) / np.sum(weights))
    if not np.isfinite(residue_occlusion_score):
        residue_occlusion_score = 0.0

    pocket_geo = _resolve_pocket_geo(pocket_residues, residue_reference=residue_reference)
    mouth_axis_geo = _build_mouth_axis_geometry(
        pocket_geo=pocket_geo,
        mouth_geo=mouth_geo,
        mouth_weights=mouth_weights,
    )
    mouth_center = mouth_axis_geo.get("mouth_center")
    mouth_axis = mouth_axis_geo.get("mouth_axis")
    aperture_radius = float(mouth_axis_geo.get("mouth_aperture_radius", _nan()))

    axis_fraction = _nan()
    aperture_fraction = _nan()
    mouth_min_clearance = _nan()
    if mouth_center is not None and mouth_axis is not None and np.isfinite(aperture_radius):
        sample_radius = min(max(1.2, 0.55 * aperture_radius), threshold + 0.8)
        axis_points = _sample_mouth_axis_points(mouth_center, mouth_axis, aperture_radius)
        aperture_points = _sample_mouth_aperture_points(mouth_center, mouth_axis, aperture_radius)
        axis_fraction, axis_clearance = _estimate_point_coverage(axis_points, nb_coords, sample_radius)
        aperture_fraction, aperture_clearance = _estimate_point_coverage(aperture_points, nb_coords, sample_radius)
        clearance_candidates = np.asarray([axis_clearance, aperture_clearance], dtype=np.float64)
        finite_clearance = clearance_candidates[np.isfinite(clearance_candidates)]
        if finite_clearance.size > 0:
            mouth_min_clearance = float(np.nanmin(finite_clearance))

    result["mouth_axis_block_fraction"] = axis_fraction
    result["mouth_aperture_block_fraction"] = aperture_fraction
    result["mouth_min_clearance"] = mouth_min_clearance

    axis_fraction_clipped = float(np.clip(axis_fraction, 0.0, 1.0)) if np.isfinite(axis_fraction) else _nan()
    aperture_fraction_clipped = float(np.clip(aperture_fraction, 0.0, 1.0)) if np.isfinite(aperture_fraction) else _nan()
    mouth_geometry_consensus = _nan()
    mouth_geometry_support = _nan()
    if np.isfinite(axis_fraction_clipped) and np.isfinite(aperture_fraction_clipped):
        # Favor consistent blockage across both the axis path and the aperture plane,
        # rather than letting one strong sub-signal dominate the final mouth score.
        denominator = max(axis_fraction_clipped + aperture_fraction_clipped, 1e-6)
        harmonic = 2.0 * axis_fraction_clipped * aperture_fraction_clipped / denominator
        balance = 1.0 - 0.50 * abs(axis_fraction_clipped - aperture_fraction_clipped)
        mouth_geometry_consensus = float(np.clip(harmonic * np.clip(balance, 0.0, 1.0), 0.0, 1.0))
        mouth_geometry_support = float(np.clip(0.50 * (axis_fraction_clipped + aperture_fraction_clipped), 0.0, 1.0))
    elif np.isfinite(axis_fraction_clipped):
        mouth_geometry_consensus = float(np.clip(0.85 * axis_fraction_clipped, 0.0, 1.0))
        mouth_geometry_support = axis_fraction_clipped
    elif np.isfinite(aperture_fraction_clipped):
        mouth_geometry_consensus = float(np.clip(0.85 * aperture_fraction_clipped, 0.0, 1.0))
        mouth_geometry_support = aperture_fraction_clipped

    component_pairs: list[tuple[float, float]] = []
    if np.isfinite(residue_occlusion_score):
        component_pairs.append((residue_occlusion_score, 0.35))
    if np.isfinite(mouth_geometry_consensus):
        component_pairs.append((mouth_geometry_consensus, 0.40))
    if np.isfinite(mouth_geometry_support):
        component_pairs.append((mouth_geometry_support, 0.15))
    if np.isfinite(mouth_min_clearance):
        clearance_score = float(np.clip(1.0 - mouth_min_clearance / max(threshold + 1.0, 1e-6), 0.0, 1.0))
        component_pairs.append((clearance_score, 0.10))

    num = float(np.sum([value * weight for value, weight in component_pairs]))
    den = float(np.sum([weight for _, weight in component_pairs]))
    result["mouth_occlusion_score"] = float(np.clip(num / den, 0.0, 1.0)) if den > 0.0 else residue_occlusion_score
    return result


def _build_candidate_path_endpoints(
    mouth_geo: list[ResidueGeometry],
    mouth_weights: np.ndarray,
    pocket_center: np.ndarray | None,
    mouth_center: np.ndarray | None,
    mouth_axis: np.ndarray | None,
    aperture_radius: float,
    max_candidates: int = 6,
) -> np.ndarray:
    """Build multiple plausible ligand-exit endpoints instead of a single mouth center."""
    if max_candidates <= 0:
        return np.empty((0, 3), dtype=np.float64)

    radius = max(float(aperture_radius), 1.0)
    points: list[np.ndarray] = []
    axis_unit = _safe_unit_vector(mouth_axis) if mouth_axis is not None else None

    if mouth_center is not None and axis_unit is not None:
        exit_shift = max(1.4, 1.15 * radius)
        points.append(np.asarray(mouth_center, dtype=np.float64).reshape(3) + exit_shift * axis_unit)
        points.append(np.asarray(mouth_center, dtype=np.float64).reshape(3) + (exit_shift + 0.8) * axis_unit)

    if mouth_geo:
        centroids = np.vstack([geo.centroid for geo in mouth_geo]).astype(np.float64, copy=False)
        weights = np.asarray(mouth_weights, dtype=np.float64).reshape(-1)
        if weights.size != centroids.shape[0]:
            weights = np.ones((centroids.shape[0],), dtype=np.float64)
        weights = np.where(np.isfinite(weights), weights, 0.0)
        order = np.argsort(-weights)

        for idx in order[: max_candidates]:
            centroid = centroids[int(idx)]
            outward = None
            if pocket_center is not None:
                outward = _safe_unit_vector(centroid - pocket_center)
            if outward is None:
                outward = axis_unit
            if outward is None:
                continue
            shift = max(1.0, 0.85 * radius)
            points.append(centroid + shift * outward)

    if not points:
        return np.empty((0, 3), dtype=np.float64)

    dedup = _deduplicate_points(
        np.vstack(points).astype(np.float64, copy=False),
        min_distance=max(0.7, 0.20 * radius),
    )
    return dedup[: max_candidates]


def _point_to_segment_distances(
    points: np.ndarray,
    seg_start: np.ndarray,
    seg_end: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute distances from points to a 3D line segment.

    Returns:
        distances: shortest distance from each point to segment.
        t_values: projection coefficient before clipping to [0, 1].
    """
    v = seg_end - seg_start
    denom = float(np.dot(v, v))
    if denom < 1e-12:
        diff = points - seg_start[None, :]
        d = np.linalg.norm(diff, axis=1)
        return d, np.zeros(points.shape[0], dtype=np.float64)

    rel = points - seg_start[None, :]
    t_raw = rel @ v / denom
    t_clip = np.clip(t_raw, 0.0, 1.0)
    proj = seg_start[None, :] + t_clip[:, None] * v[None, :]
    d = np.linalg.norm(points - proj, axis=1)
    return d, t_raw


def _ligand_coords_from_input(ligand_template: Any) -> np.ndarray:
    """Extract ligand coordinates from flexible ligand-template input."""
    if ligand_template is None:
        return np.empty((0, 3), dtype=np.float64)

    if hasattr(ligand_template, "coordinates"):
        try:
            arr = np.asarray(getattr(ligand_template, "coordinates"), dtype=np.float64)
            if arr.ndim == 2 and arr.shape[1] == 3:
                mask = np.isfinite(arr).all(axis=1)
                return arr[mask]
        except Exception:
            pass

    return _coords_from_atoms(ligand_template, heavy_only=True)


def compute_substrate_clash_features(
    nanobody_atoms: Any,
    ligand_template: Any = None,
    pocket_residues: Any = None,
    residue_reference: Any = None,
    substrate_clash_threshold: float = 2.8,
    ligand_path_radius: float = 3.0,
) -> dict[str, float]:
    """Compute ligand-template clash and path-blocking features.

    Features:
    - substrate_clash_count
    - substrate_clash_fraction
    - substrate_overlap_score
    - ligand_path_block_score
    - ligand_path_block_fraction
    - ligand_path_block_fraction_weighted
    - ligand_path_min_clearance
    - ligand_path_bottleneck_score

    Notes:
    - Clash and overlap are static atom-geometry features.
    - ligand_path_block_score is a static path-blocking approximation,
      not a dynamic transport simulation.
    """
    clash_thr = max(float(substrate_clash_threshold), 0.0)
    path_radius = max(float(ligand_path_radius), 0.0)
    nb_coords = _coords_from_atoms(nanobody_atoms, heavy_only=True)
    lig_coords = _ligand_coords_from_input(ligand_template)

    result = {
        "substrate_clash_count": 0.0,
        "substrate_clash_fraction": _nan(),
        "substrate_overlap_score": _nan(),
        "ligand_path_block_score": _nan(),
        "ligand_path_block_fraction": _nan(),
        "ligand_path_block_fraction_weighted": _nan(),
        "ligand_path_min_clearance": _nan(),
        "ligand_path_bottleneck_score": _nan(),
        "ligand_path_exit_block_fraction": _nan(),
        "ligand_path_candidate_count": 0.0,
        "ligand_path_sample_count": 0.0,
    }

    if lig_coords.shape[0] == 0:
        return result

    if nb_coords.shape[0] == 0:
        result["substrate_clash_fraction"] = 0.0
        result["substrate_overlap_score"] = 0.0
        result["ligand_path_block_score"] = 0.0
        result["ligand_path_block_fraction"] = 0.0
        result["ligand_path_block_fraction_weighted"] = 0.0
        result["ligand_path_bottleneck_score"] = 0.0
        result["ligand_path_exit_block_fraction"] = 0.0
        return result

    lig_min_to_nb = _rowwise_min_distance(lig_coords, nb_coords)
    clash_mask = np.isfinite(lig_min_to_nb) & (lig_min_to_nb <= clash_thr)
    clash_count = int(np.sum(clash_mask))

    result["substrate_clash_count"] = float(clash_count)
    result["substrate_clash_fraction"] = float(clash_count / lig_coords.shape[0])

    overlap_soft_radius = clash_thr + 1.2
    if overlap_soft_radius <= 1e-8:
        overlap_score = 0.0
    else:
        overlap_score = float(
            np.nanmean(np.clip(1.0 - lig_min_to_nb / overlap_soft_radius, 0.0, 1.0))
        )
    result["substrate_overlap_score"] = overlap_score

    pocket_geo = _resolve_pocket_geo(pocket_residues, residue_reference=residue_reference)
    pocket_center = _compute_pocket_center(pocket_geo)
    ligand_centroid = np.mean(lig_coords, axis=0, dtype=np.float64)

    mouth_geo, mouth_weights = _infer_mouth_candidates(
        pocket_residues=pocket_geo,
        mouth_residue_fraction=0.35,
        residue_reference=residue_reference,
    )
    if mouth_geo:
        mouth_centroids = np.vstack([geo.centroid for geo in mouth_geo]).astype(np.float64, copy=False)
        if mouth_weights.size == mouth_centroids.shape[0] and np.sum(mouth_weights) > 1e-8:
            mouth_center = np.average(mouth_centroids, axis=0, weights=mouth_weights)
        else:
            mouth_center = np.mean(mouth_centroids, axis=0, dtype=np.float64)
    else:
        mouth_center = pocket_center

    mouth_axis_geo = _build_mouth_axis_geometry(
        pocket_geo=pocket_geo,
        mouth_geo=mouth_geo,
        mouth_weights=mouth_weights,
    )
    path_endpoints = _build_candidate_path_endpoints(
        mouth_geo=mouth_geo,
        mouth_weights=mouth_weights,
        pocket_center=mouth_axis_geo.get("pocket_center"),
        mouth_center=mouth_axis_geo.get("mouth_center") if mouth_axis_geo.get("mouth_center") is not None else mouth_center,
        mouth_axis=mouth_axis_geo.get("mouth_axis"),
        aperture_radius=float(mouth_axis_geo.get("mouth_aperture_radius", path_radius)),
        max_candidates=6,
    )
    if path_endpoints.shape[0] == 0 and mouth_center is not None:
        path_endpoints = np.asarray(mouth_center, dtype=np.float64).reshape(1, 3)
    if path_endpoints.shape[0] == 0 and pocket_center is not None:
        path_endpoints = np.asarray(pocket_center, dtype=np.float64).reshape(1, 3)

    if path_endpoints.shape[0] == 0:
        result["ligand_path_block_score"] = 0.0
        result["ligand_path_block_fraction"] = 0.0
        result["ligand_path_block_fraction_weighted"] = 0.0
        result["ligand_path_bottleneck_score"] = 0.0
        result["ligand_path_exit_block_fraction"] = 0.0
    else:
        candidate_scores: list[float] = []
        candidate_plain_fraction: list[float] = []
        candidate_weighted_fraction: list[float] = []
        candidate_bottleneck_score: list[float] = []
        candidate_clearance: list[float] = []
        total_sample_count = 0.0

        for endpoint in path_endpoints:
            influence_score = estimate_path_block_score(
                nanobody_atoms=nb_coords,
                path_start=ligand_centroid,
                path_end=endpoint,
                path_radius=path_radius,
            )

            bottleneck = estimate_path_bottleneck_features(
                nanobody_atoms=nb_coords,
                path_start=ligand_centroid,
                path_end=endpoint,
                path_radius=path_radius,
                n_samples=25,
            )
            total_sample_count += float(bottleneck.get("ligand_path_sample_count", 0.0))

            weighted_fraction = float(bottleneck.get("ligand_path_block_fraction_weighted", _nan()))
            bottleneck_score = float(bottleneck.get("ligand_path_bottleneck_score", _nan()))
            influence_safe = float(influence_score) if np.isfinite(influence_score) else 0.0
            weighted_safe = weighted_fraction if np.isfinite(weighted_fraction) else 0.0
            bottleneck_safe = bottleneck_score if np.isfinite(bottleneck_score) else 0.0

            # Blend atom-influence score with continuity and bottleneck severity per candidate exit.
            combined = 0.45 * influence_safe + 0.30 * weighted_safe + 0.25 * bottleneck_safe
            candidate_scores.append(float(np.clip(combined, 0.0, 1.0)))
            candidate_plain_fraction.append(float(bottleneck.get("ligand_path_block_fraction", _nan())))
            candidate_weighted_fraction.append(weighted_fraction)
            candidate_bottleneck_score.append(bottleneck_score)
            candidate_clearance.append(float(bottleneck.get("ligand_path_min_clearance", _nan())))

        score_arr = np.asarray(candidate_scores, dtype=np.float64)
        plain_arr = np.asarray(candidate_plain_fraction, dtype=np.float64)
        weighted_arr = np.asarray(candidate_weighted_fraction, dtype=np.float64)
        bottleneck_arr = np.asarray(candidate_bottleneck_score, dtype=np.float64)
        clearance_arr = np.asarray(candidate_clearance, dtype=np.float64)

        finite_scores = score_arr[np.isfinite(score_arr)]
        if finite_scores.size > 0:
            mean_score = float(np.nanmean(finite_scores))
            median_score = float(np.nanmedian(finite_scores))
            best_escape_score = float(np.nanmin(finite_scores))
            exit_block_fraction = float(np.mean(finite_scores >= 0.55))
            open_route_fraction = float(np.mean(finite_scores <= 0.35))
            result["ligand_path_exit_block_fraction"] = exit_block_fraction
            result["ligand_path_candidate_count"] = float(finite_scores.size)
            finite_clearance_for_score = clearance_arr[np.isfinite(clearance_arr)]
            best_clearance_score = 0.0
            if finite_clearance_for_score.size > 0:
                best_clearance = float(np.nanmax(finite_clearance_for_score))
                best_clearance_score = float(
                    np.clip(best_clearance / max(path_radius + 1.0, 1e-6), 0.0, 1.0)
                )
            # Keep multi-exit consensus as the main signal, but explicitly reduce
            # blockage score when one exit route remains comparatively open.
            block_core = 0.40 * mean_score + 0.25 * median_score + 0.20 * best_escape_score + 0.15 * exit_block_fraction
            open_route_penalty = 0.20 * open_route_fraction + 0.15 * best_clearance_score
            result["ligand_path_block_score"] = float(np.clip(block_core - open_route_penalty, 0.0, 1.0))
        else:
            result["ligand_path_block_score"] = 0.0
            result["ligand_path_exit_block_fraction"] = 0.0
            result["ligand_path_candidate_count"] = 0.0

        finite_plain = plain_arr[np.isfinite(plain_arr)]
        if finite_plain.size > 0:
            result["ligand_path_block_fraction"] = float(
                np.clip(0.55 * np.nanmean(finite_plain) + 0.45 * np.nanmin(finite_plain), 0.0, 1.0)
            )
        else:
            result["ligand_path_block_fraction"] = 0.0

        finite_weighted = weighted_arr[np.isfinite(weighted_arr)]
        if finite_weighted.size > 0:
            result["ligand_path_block_fraction_weighted"] = float(
                np.clip(0.55 * np.nanmean(finite_weighted) + 0.45 * np.nanmin(finite_weighted), 0.0, 1.0)
            )
        else:
            result["ligand_path_block_fraction_weighted"] = 0.0

        finite_bottleneck = bottleneck_arr[np.isfinite(bottleneck_arr)]
        if finite_bottleneck.size > 0:
            result["ligand_path_bottleneck_score"] = float(
                np.clip(0.55 * np.nanmean(finite_bottleneck) + 0.45 * np.nanmin(finite_bottleneck), 0.0, 1.0)
            )
        else:
            result["ligand_path_bottleneck_score"] = 0.0

        finite_clearance = clearance_arr[np.isfinite(clearance_arr)]
        if finite_clearance.size > 0:
            # Use the best available exit clearance so one open route is not hidden by a blocked side branch.
            result["ligand_path_min_clearance"] = float(np.nanmax(finite_clearance))

        result["ligand_path_sample_count"] = float(total_sample_count)

    return result


def compute_all_geometry_features(
    antigen_atoms: Any,
    antigen_residues: Any,
    nanobody_atoms: Any,
    nanobody_residues: Any,
    pocket_residues: Any,
    catalytic_residues: Any = None,
    ligand_template: Any = None,
    atom_contact_threshold: float = 4.5,
    catalytic_contact_threshold: float = 4.5,
    substrate_clash_threshold: float = 2.8,
    mouth_residue_fraction: float = 0.30,
    interface_contact_threshold: float = 6.0,
    ligand_path_radius: float = 3.0,
    catalytic_block_min_hits: int = 1,
) -> dict[str, Any]:
    """Compute all geometry features required by the screening pipeline.

    The function is robust to missing pocket/catalytic/ligand inputs and
    returns NaN/default values instead of crashing.
    """
    _ = nanobody_residues  # Reserved for future extensions.

    residue_reference = antigen_residues if antigen_residues is not None else antigen_atoms
    warnings: list[str] = []

    pocket_defaults = {
        "pocket_hit_count": 0.0,
        "pocket_hit_fraction": _nan(),
        "pocket_atom_contact_count": 0.0,
        "min_distance_to_pocket": _nan(),
        "mean_min_distance_to_pocket_residues": _nan(),
    }
    pocket_shape_defaults = {
        "pocket_shape_residue_count": 0.0,
        "pocket_shape_centroid_radius_mean": _nan(),
        "pocket_shape_centroid_radius_p90": _nan(),
        "pocket_shape_centroid_radius_max": _nan(),
        "pocket_shape_atom_bbox_volume": _nan(),
        "pocket_shape_overwide_proxy": _nan(),
        "pocket_shape_tightness_proxy": _nan(),
    }
    catalytic_defaults = {
        "catalytic_hit_count": 0.0,
        "catalytic_hit_fraction": _nan(),
        "min_distance_to_catalytic_residues": _nan(),
        "catalytic_block_flag": 0.0,
    }
    catalytic_anchor_defaults = {
        "catalytic_anchor_has_input": 0.0,
        "catalytic_anchor_core_residue_count": 0.0,
        "catalytic_anchor_antigen_residue_count": 0.0,
        "catalytic_anchor_primary_radius": 6.0,
        "catalytic_anchor_primary_shell_residue_count": 0.0,
        "catalytic_anchor_primary_shell_fraction_of_antigen": _nan(),
        "catalytic_anchor_primary_shell_hit_count": 0.0,
        "catalytic_anchor_primary_shell_hit_fraction": _nan(),
        "catalytic_anchor_min_distance_to_primary_shell": _nan(),
        "catalytic_anchor_manual_overlap_count": 0.0,
        "catalytic_anchor_manual_overlap_fraction_of_shell": _nan(),
        "catalytic_anchor_manual_overlap_fraction_of_pocket": _nan(),
        "catalytic_anchor_shell_overwide_proxy": _nan(),
        "catalytic_anchor_shell_overwide_flag": 0.0,
        "catalytic_anchor_primary_shell_keys": "",
    }
    for radius in DEFAULT_CATALYTIC_ANCHOR_SHELL_RADII:
        label = _radius_label(radius)
        catalytic_anchor_defaults[f"catalytic_anchor_shell_{label}_residue_count"] = 0.0
        catalytic_anchor_defaults[f"catalytic_anchor_shell_{label}_fraction_of_antigen"] = _nan()
        catalytic_anchor_defaults[f"catalytic_anchor_shell_{label}_hit_count"] = 0.0
        catalytic_anchor_defaults[f"catalytic_anchor_shell_{label}_hit_fraction"] = _nan()
        catalytic_anchor_defaults[f"catalytic_anchor_shell_{label}_min_distance_to_nanobody"] = _nan()
    center_defaults = {
        "distance_to_pocket_center": _nan(),
        "nanobody_centroid_to_pocket_center": _nan(),
        "distance_interface_centroid_to_pocket_center": _nan(),
    }
    mouth_defaults = {
        "mouth_hit_count": 0.0,
        "mouth_hit_fraction": _nan(),
        "mouth_occlusion_score": _nan(),
        "mouth_axis_block_fraction": _nan(),
        "mouth_aperture_block_fraction": _nan(),
        "mouth_min_clearance": _nan(),
    }
    occupancy_defaults = {
        "delta_pocket_access_proxy": _nan(),
        "delta_pocket_occupancy_proxy": _nan(),
        "pocket_block_volume_proxy": _nan(),
        "pocket_local_point_count": 0.0,
        "pocket_local_occupied_count": 0.0,
    }
    substrate_defaults = {
        "substrate_clash_count": 0.0,
        "substrate_clash_fraction": _nan(),
        "substrate_overlap_score": _nan(),
        "ligand_path_block_score": _nan(),
        "ligand_path_block_fraction": _nan(),
        "ligand_path_block_fraction_weighted": _nan(),
        "ligand_path_min_clearance": _nan(),
        "ligand_path_bottleneck_score": _nan(),
        "ligand_path_exit_block_fraction": _nan(),
        "ligand_path_candidate_count": 0.0,
        "ligand_path_sample_count": 0.0,
    }

    def _safe_compute(name: str, fn: Any, defaults: dict[str, float]) -> dict[str, float]:
        try:
            return dict(fn())
        except Exception as exc:
            warnings.append(f"{name}: {exc}")
            return dict(defaults)

    pocket_features = _safe_compute(
        "pocket",
        lambda: compute_pocket_features(
            nanobody_atoms=nanobody_atoms,
            pocket_residues=pocket_residues,
            atom_contact_threshold=atom_contact_threshold,
            residue_reference=residue_reference,
        ),
        pocket_defaults,
    )
    pocket_shape_features = _safe_compute(
        "pocket_shape",
        lambda: compute_pocket_shape_features(
            pocket_residues=pocket_residues,
            residue_reference=residue_reference,
        ),
        pocket_shape_defaults,
    )
    catalytic_features = _safe_compute(
        "catalytic",
        lambda: compute_catalytic_features(
            nanobody_atoms=nanobody_atoms,
            catalytic_residues=catalytic_residues,
            catalytic_contact_threshold=catalytic_contact_threshold,
            residue_reference=residue_reference,
            catalytic_block_min_hits=catalytic_block_min_hits,
        ),
        catalytic_defaults,
    )
    catalytic_anchor_features = _safe_compute(
        "catalytic_anchor",
        lambda: compute_catalytic_anchor_pocket_features(
            antigen_residues=antigen_residues,
            nanobody_atoms=nanobody_atoms,
            catalytic_residues=catalytic_residues,
            pocket_residues=pocket_residues,
            residue_reference=residue_reference,
            shell_radii=DEFAULT_CATALYTIC_ANCHOR_SHELL_RADII,
            primary_radius=6.0,
            contact_threshold=atom_contact_threshold,
        ),
        catalytic_anchor_defaults,
    )
    center_features = _safe_compute(
        "center",
        lambda: compute_pocket_center_features(
            antigen_atoms=antigen_atoms,
            nanobody_atoms=nanobody_atoms,
            pocket_residues=pocket_residues,
            residue_reference=residue_reference,
            interface_contact_threshold=interface_contact_threshold,
        ),
        center_defaults,
    )
    mouth_features = _safe_compute(
        "mouth",
        lambda: compute_mouth_occlusion_features(
            nanobody_atoms=nanobody_atoms,
            pocket_residues=pocket_residues,
            atom_contact_threshold=atom_contact_threshold,
            mouth_residue_fraction=mouth_residue_fraction,
            residue_reference=residue_reference,
        ),
        mouth_defaults,
    )
    occupancy_features = _safe_compute(
        "occupancy",
        lambda: compute_pocket_occupancy_features(
            nanobody_atoms=nanobody_atoms,
            pocket_residues=pocket_residues,
            residue_reference=residue_reference,
            occupancy_radius=substrate_clash_threshold,
        ),
        occupancy_defaults,
    )
    substrate_features = _safe_compute(
        "substrate",
        lambda: compute_substrate_clash_features(
            nanobody_atoms=nanobody_atoms,
            ligand_template=ligand_template,
            pocket_residues=pocket_residues,
            residue_reference=residue_reference,
            substrate_clash_threshold=substrate_clash_threshold,
            ligand_path_radius=ligand_path_radius,
        ),
        substrate_defaults,
    )

    all_features: dict[str, Any] = {}
    all_features.update(pocket_features)
    all_features.update(pocket_shape_features)
    all_features.update(catalytic_features)
    all_features.update(catalytic_anchor_features)
    all_features.update(center_features)
    all_features.update(mouth_features)
    all_features.update(occupancy_features)
    all_features.update(substrate_features)

    pocket_geo_summary = _resolve_pocket_geo(pocket_residues, residue_reference=residue_reference)
    catalytic_geo_summary = _resolve_pocket_geo(catalytic_residues, residue_reference=residue_reference)
    ligand_coords_summary = _ligand_coords_from_input(ligand_template)
    all_features["geometry_has_pocket_input"] = 1.0 if len(pocket_geo_summary) > 0 else 0.0
    all_features["geometry_has_catalytic_input"] = 1.0 if len(catalytic_geo_summary) > 0 else 0.0
    all_features["geometry_has_ligand_input"] = 1.0 if ligand_coords_summary.shape[0] > 0 else 0.0
    all_features["geometry_debug_warning_count"] = float(len(warnings))
    all_features["geometry_debug_summary"] = (
        f"pocket={len(pocket_geo_summary)};"
        f"catalytic={len(catalytic_geo_summary)};"
        f"ligand_atoms={ligand_coords_summary.shape[0]};"
        f"warnings={len(warnings)}"
    )

    # Keep all required keys stable even when future changes add/remove fields.
    required_feature_keys = [
        "pocket_hit_count",
        "pocket_hit_fraction",
        "pocket_atom_contact_count",
        "min_distance_to_pocket",
        "mean_min_distance_to_pocket_residues",
        "pocket_shape_residue_count",
        "pocket_shape_centroid_radius_mean",
        "pocket_shape_centroid_radius_p90",
        "pocket_shape_centroid_radius_max",
        "pocket_shape_atom_bbox_volume",
        "pocket_shape_overwide_proxy",
        "pocket_shape_tightness_proxy",
        "catalytic_hit_count",
        "catalytic_hit_fraction",
        "min_distance_to_catalytic_residues",
        "catalytic_block_flag",
        "catalytic_anchor_has_input",
        "catalytic_anchor_core_residue_count",
        "catalytic_anchor_antigen_residue_count",
        "catalytic_anchor_primary_radius",
        "catalytic_anchor_primary_shell_residue_count",
        "catalytic_anchor_primary_shell_fraction_of_antigen",
        "catalytic_anchor_primary_shell_hit_count",
        "catalytic_anchor_primary_shell_hit_fraction",
        "catalytic_anchor_min_distance_to_primary_shell",
        "catalytic_anchor_manual_overlap_count",
        "catalytic_anchor_manual_overlap_fraction_of_shell",
        "catalytic_anchor_manual_overlap_fraction_of_pocket",
        "catalytic_anchor_shell_overwide_proxy",
        "catalytic_anchor_shell_overwide_flag",
        "catalytic_anchor_shell_4A_residue_count",
        "catalytic_anchor_shell_4A_fraction_of_antigen",
        "catalytic_anchor_shell_4A_hit_count",
        "catalytic_anchor_shell_4A_hit_fraction",
        "catalytic_anchor_shell_4A_min_distance_to_nanobody",
        "catalytic_anchor_shell_6A_residue_count",
        "catalytic_anchor_shell_6A_fraction_of_antigen",
        "catalytic_anchor_shell_6A_hit_count",
        "catalytic_anchor_shell_6A_hit_fraction",
        "catalytic_anchor_shell_6A_min_distance_to_nanobody",
        "catalytic_anchor_shell_8A_residue_count",
        "catalytic_anchor_shell_8A_fraction_of_antigen",
        "catalytic_anchor_shell_8A_hit_count",
        "catalytic_anchor_shell_8A_hit_fraction",
        "catalytic_anchor_shell_8A_min_distance_to_nanobody",
        "distance_to_pocket_center",
        "nanobody_centroid_to_pocket_center",
        "distance_interface_centroid_to_pocket_center",
        "mouth_hit_count",
        "mouth_hit_fraction",
        "mouth_occlusion_score",
        "mouth_axis_block_fraction",
        "mouth_aperture_block_fraction",
        "mouth_min_clearance",
        "delta_pocket_access_proxy",
        "delta_pocket_occupancy_proxy",
        "pocket_block_volume_proxy",
        "substrate_clash_count",
        "substrate_clash_fraction",
        "substrate_overlap_score",
        "ligand_path_block_score",
        "ligand_path_block_fraction",
        "ligand_path_block_fraction_weighted",
        "ligand_path_min_clearance",
        "ligand_path_bottleneck_score",
        "ligand_path_exit_block_fraction",
        "ligand_path_candidate_count",
        "ligand_path_sample_count",
    ]
    for key in required_feature_keys:
        all_features.setdefault(key, _nan())

    return all_features


__all__ = [
    "ResidueGeometry",
    "GeometryFeatureConfig",
    "compute_residue_min_distances",
    "estimate_local_exposure_proxy",
    "sample_pocket_local_points",
    "compute_point_occupancy_by_nanobody",
    "estimate_path_block_score",
    "estimate_path_bottleneck_features",
    "compute_pocket_features",
    "compute_pocket_shape_features",
    "compute_catalytic_features",
    "compute_catalytic_anchor_pocket_features",
    "compute_pocket_center_features",
    "infer_mouth_residues",
    "compute_mouth_occlusion_features",
    "compute_pocket_occupancy_features",
    "compute_substrate_clash_features",
    "compute_all_geometry_features",
]
