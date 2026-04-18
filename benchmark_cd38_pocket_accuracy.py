from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import numpy as np
import pandas as pd
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure

from pdb_parser import get_residue_uid, load_complex_pdb
from pocket_io import load_residue_set, match_residues_in_structure, normalize_residue_key


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark CD38 pocket residue predictions against reported active-site residues.")
    parser.add_argument("--pdb_path", default=None, help="Path to CD38 structure PDB file")
    parser.add_argument("--rcsb_pdb_id", default=None, help="Optional RCSB PDB ID to download, e.g. 3F6Y")
    parser.add_argument(
        "--truth_file",
        default="benchmarks/cd38/cd38_active_site_truth.txt",
        help="Ground-truth residue definition file",
    )
    parser.add_argument("--predicted_pocket_file", required=True, help="Predicted pocket residue file to evaluate")
    parser.add_argument("--out_dir", default="cd38_pocket_benchmark_outputs", help="Output directory")
    parser.add_argument("--truth_chain", default=None, help="Optional chain override for truth residues")
    parser.add_argument("--predicted_chain", default=None, help="Optional chain override for predicted residues")
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


def _load_residue_keys(path_like: str | Path, chain_override: str | None = None) -> set[str]:
    path = Path(path_like).expanduser().resolve()
    residue_keys = load_residue_set(str(path))
    if not chain_override:
        return residue_keys
    chain_id = str(chain_override).strip()
    remapped: set[str] = set()
    for key in residue_keys:
        parts = str(key).split(":")
        if len(parts) == 2:
            _, resseq = parts
            remapped.add(normalize_residue_key(chain_id, int(resseq), ""))
        elif len(parts) == 3:
            _, resseq, icode = parts
            remapped.add(normalize_residue_key(chain_id, int(resseq), icode))
        else:
            raise ValueError(f"Unexpected canonical residue key: {key!r}")
    return remapped


def _iter_structure_residues(structure_or_entity: Any) -> list[Residue]:
    if isinstance(structure_or_entity, Residue):
        return [structure_or_entity]
    if isinstance(structure_or_entity, Chain):
        return list(structure_or_entity.get_residues())
    if isinstance(structure_or_entity, Model):
        return list(structure_or_entity.get_residues())
    if isinstance(structure_or_entity, Structure):
        models = list(structure_or_entity.get_models())
        if not models:
            return []
        return list(models[0].get_residues())
    if hasattr(structure_or_entity, "get_residues"):
        return list(structure_or_entity.get_residues())
    raise TypeError("Unsupported structure entity for residue extraction.")


def _residue_atom_coords(residue: Residue) -> np.ndarray:
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


def _build_match_table(match_result: Any, label: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if match_result is None:
        return pd.DataFrame(columns=["label", "requested_key", "matched_key", "uid", "chain_id", "resseq", "icode", "resname"])

    for detail in match_result.details:
        residues = list(match_result.key_to_residues.get(detail.requested_key, ()))
        if not residues:
            rows.append(
                {
                    "label": label,
                    "requested_key": str(detail.requested_key),
                    "matched_key": str(detail.matched_key) if detail.matched_key else None,
                    "uid": None,
                    "chain_id": None,
                    "resseq": None,
                    "icode": None,
                    "resname": None,
                }
            )
            continue
        for residue in residues:
            parent = residue.get_parent()
            chain_id = str(getattr(parent, "id", "")).strip() or "_"
            _, resseq, icode = residue.id
            rows.append(
                {
                    "label": label,
                    "requested_key": str(detail.requested_key),
                    "matched_key": str(detail.matched_key) if detail.matched_key else None,
                    "uid": get_residue_uid(residue),
                    "chain_id": chain_id,
                    "resseq": int(resseq),
                    "icode": str(icode).strip(),
                    "resname": str(residue.get_resname()).strip(),
                }
            )
    return pd.DataFrame(rows)


def _build_near_metrics(
    truth_match: Any,
    predicted_match: Any,
    near_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    truth_residues = list(truth_match.matched_residues if truth_match is not None else ())
    predicted_residues = list(predicted_match.matched_residues if predicted_match is not None else ())
    predicted_uid_set = {get_residue_uid(res) for res in predicted_residues}
    truth_uid_set = {get_residue_uid(res) for res in truth_residues}

    pred_coord_map = {get_residue_uid(res): _residue_atom_coords(res) for res in predicted_residues}
    truth_coord_map = {get_residue_uid(res): _residue_atom_coords(res) for res in truth_residues}

    rows: list[dict[str, Any]] = []
    truth_near_hit = 0
    for residue in truth_residues:
        truth_uid = get_residue_uid(residue)
        truth_coords = truth_coord_map[truth_uid]
        best_uid = None
        best_distance = float("nan")
        for pred_uid, pred_coords in pred_coord_map.items():
            distance = _global_min_distance(truth_coords, pred_coords)
            if not np.isfinite(distance):
                continue
            if best_uid is None or distance < best_distance:
                best_uid = pred_uid
                best_distance = float(distance)
        exact_overlap = truth_uid in predicted_uid_set
        near_hit = bool(np.isfinite(best_distance) and best_distance <= float(near_threshold))
        truth_near_hit += int(near_hit)
        rows.append(
            {
                "truth_uid": truth_uid,
                "best_predicted_uid": best_uid,
                "min_distance_to_predicted": best_distance,
                "is_exact_overlap": exact_overlap,
                "is_near_hit": near_hit,
            }
        )

    predicted_near_hit = 0
    predicted_rows: list[dict[str, Any]] = []
    for residue in predicted_residues:
        pred_uid = get_residue_uid(residue)
        pred_coords = pred_coord_map[pred_uid]
        best_truth_uid = None
        best_distance = float("nan")
        for truth_uid, truth_coords in truth_coord_map.items():
            distance = _global_min_distance(pred_coords, truth_coords)
            if not np.isfinite(distance):
                continue
            if not np.isfinite(best_distance) or distance < best_distance:
                best_truth_uid = truth_uid
                best_distance = float(distance)
        exact_overlap = pred_uid in truth_uid_set
        near_truth = bool(np.isfinite(best_distance) and best_distance <= float(near_threshold))
        if near_truth:
            predicted_near_hit += 1
        predicted_rows.append(
            {
                "predicted_uid": pred_uid,
                "best_truth_uid": best_truth_uid,
                "min_distance_to_truth": best_distance,
                "is_exact_overlap": exact_overlap,
                "is_near_truth": near_truth,
            }
        )

    detail_df = pd.DataFrame(rows)
    predicted_detail_df = pd.DataFrame(predicted_rows)
    far_predicted_count = int(len(predicted_residues) - predicted_near_hit)
    extra_predicted_count = int(len(predicted_uid_set - truth_uid_set))
    predicted_to_truth_ratio = float(len(predicted_residues) / len(truth_residues)) if truth_residues else float("nan")
    extra_fraction = float(extra_predicted_count / len(predicted_residues)) if predicted_residues else float("nan")
    far_fraction = float(far_predicted_count / len(predicted_residues)) if predicted_residues else float("nan")
    size_penalty = float(min(max((predicted_to_truth_ratio - 1.0) / 3.0, 0.0), 1.0)) if np.isfinite(predicted_to_truth_ratio) else float("nan")
    if np.isfinite(extra_fraction) and np.isfinite(far_fraction) and np.isfinite(size_penalty):
        overwide_score = float(0.45 * extra_fraction + 0.35 * far_fraction + 0.20 * size_penalty)
    else:
        overwide_score = float("nan")
    metrics = {
        "truth_near_coverage": float(truth_near_hit / len(truth_residues)) if truth_residues else float("nan"),
        "predicted_near_precision": float(predicted_near_hit / len(predicted_residues)) if predicted_residues else float("nan"),
        "mean_truth_min_distance": float(np.nanmean(_to_numeric(detail_df["min_distance_to_predicted"]).to_numpy(dtype=float)))
        if not detail_df.empty
        else float("nan"),
        "median_truth_min_distance": float(np.nanmedian(_to_numeric(detail_df["min_distance_to_predicted"]).to_numpy(dtype=float)))
        if not detail_df.empty
        else float("nan"),
        "near_threshold": float(near_threshold),
        "predicted_to_truth_residue_ratio": predicted_to_truth_ratio,
        "extra_predicted_residue_count": float(extra_predicted_count),
        "extra_predicted_fraction": extra_fraction,
        "far_predicted_residue_count": float(far_predicted_count),
        "far_predicted_fraction": far_fraction,
        "overwide_pocket_score": overwide_score,
    }
    return detail_df, predicted_detail_df, metrics


def main() -> None:
    args = _build_parser().parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.pdb_path:
        pdb_path = Path(args.pdb_path).expanduser().resolve()
    elif args.rcsb_pdb_id:
        pdb_path = _download_rcsb_pdb(args.rcsb_pdb_id, out_dir / "inputs")
    else:
        raise ValueError("Either --pdb_path or --rcsb_pdb_id must be provided.")
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    structure = load_complex_pdb(str(pdb_path))
    structure_residues = _iter_structure_residues(structure)
    chain_ids = sorted({str(res.get_parent().id).strip() or "_" for res in structure_residues})

    truth_keys = _load_residue_keys(args.truth_file, chain_override=args.truth_chain)
    predicted_keys = _load_residue_keys(args.predicted_pocket_file, chain_override=args.predicted_chain)
    truth_match = match_residues_in_structure(structure, truth_keys)
    predicted_match = match_residues_in_structure(structure, predicted_keys)

    truth_table = _build_match_table(truth_match, label="truth")
    predicted_table = _build_match_table(predicted_match, label="predicted")
    truth_uid_set = {str(uid) for uid in truth_table["uid"].dropna().astype(str).tolist()}
    predicted_uid_set = {str(uid) for uid in predicted_table["uid"].dropna().astype(str).tolist()}
    overlap_uid_set = truth_uid_set & predicted_uid_set

    exact_truth_coverage = float(len(overlap_uid_set) / len(truth_uid_set)) if truth_uid_set else float("nan")
    exact_predicted_precision = float(len(overlap_uid_set) / len(predicted_uid_set)) if predicted_uid_set else float("nan")
    exact_jaccard = float(len(overlap_uid_set) / len(truth_uid_set | predicted_uid_set)) if (truth_uid_set or predicted_uid_set) else float("nan")
    if np.isfinite(exact_truth_coverage) and np.isfinite(exact_predicted_precision) and (exact_truth_coverage + exact_predicted_precision) > 0:
        exact_f1 = float(2.0 * exact_truth_coverage * exact_predicted_precision / (exact_truth_coverage + exact_predicted_precision))
    else:
        exact_f1 = float("nan")

    near_detail_df, predicted_near_detail_df, near_metrics = _build_near_metrics(
        truth_match=truth_match,
        predicted_match=predicted_match,
        near_threshold=float(args.near_threshold),
    )

    overlap_table = truth_table.loc[truth_table["uid"].astype(str).isin(overlap_uid_set)].reset_index(drop=True)
    missed_truth = truth_table.loc[~truth_table["uid"].astype(str).isin(overlap_uid_set)].reset_index(drop=True)
    extra_predicted = predicted_table.loc[~predicted_table["uid"].astype(str).isin(overlap_uid_set)].reset_index(drop=True)

    truth_table.to_csv(out_dir / "truth_residue_table.csv", index=False)
    predicted_table.to_csv(out_dir / "predicted_residue_table.csv", index=False)
    overlap_table.to_csv(out_dir / "exact_overlap_table.csv", index=False)
    missed_truth.to_csv(out_dir / "missed_truth_table.csv", index=False)
    extra_predicted.to_csv(out_dir / "extra_predicted_table.csv", index=False)
    near_detail_df.to_csv(out_dir / "near_hit_table.csv", index=False)
    predicted_near_detail_df.to_csv(out_dir / "predicted_to_truth_distance_table.csv", index=False)

    summary = {
        "pdb_path": str(pdb_path),
        "rcsb_pdb_id": str(args.rcsb_pdb_id).strip().upper() if args.rcsb_pdb_id else None,
        "truth_file": str(Path(args.truth_file).expanduser().resolve()),
        "predicted_pocket_file": str(Path(args.predicted_pocket_file).expanduser().resolve()),
        "structure_chain_ids": chain_ids,
        "counts": {
            "truth_defined_keys": int(len(truth_keys)),
            "truth_matched_keys": int(len(getattr(truth_match, "matched_keys", ()))),
            "predicted_defined_keys": int(len(predicted_keys)),
            "predicted_matched_keys": int(len(getattr(predicted_match, "matched_keys", ()))),
            "truth_matched_residues": int(len(truth_uid_set)),
            "predicted_matched_residues": int(len(predicted_uid_set)),
            "exact_overlap_residues": int(len(overlap_uid_set)),
        },
        "metrics": {
            "exact_truth_coverage": exact_truth_coverage,
            "exact_predicted_precision": exact_predicted_precision,
            "exact_jaccard": exact_jaccard,
            "exact_f1": exact_f1,
            **near_metrics,
        },
        "warnings": {
            "truth": list(getattr(truth_match, "warnings", ()) or ()),
            "predicted": list(getattr(predicted_match, "warnings", ()) or ()),
        },
        "overlap_uids": sorted(overlap_uid_set),
    }
    summary_path = out_dir / "cd38_pocket_accuracy_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)

    report_lines = [
        "# CD38 Pocket Accuracy Benchmark",
        "",
        f"- pdb_path: `{pdb_path}`",
        f"- truth_file: `{Path(args.truth_file).expanduser().resolve()}`",
        f"- predicted_pocket_file: `{Path(args.predicted_pocket_file).expanduser().resolve()}`",
        f"- structure_chain_ids: `{', '.join(chain_ids)}`",
        "",
        "## Exact Match Metrics",
        "",
        f"- exact_truth_coverage: `{exact_truth_coverage:.4f}`" if np.isfinite(exact_truth_coverage) else "- exact_truth_coverage: `nan`",
        f"- exact_predicted_precision: `{exact_predicted_precision:.4f}`" if np.isfinite(exact_predicted_precision) else "- exact_predicted_precision: `nan`",
        f"- exact_jaccard: `{exact_jaccard:.4f}`" if np.isfinite(exact_jaccard) else "- exact_jaccard: `nan`",
        f"- exact_f1: `{exact_f1:.4f}`" if np.isfinite(exact_f1) else "- exact_f1: `nan`",
        "",
        "## Near-Hit Metrics",
        "",
        f"- truth_near_coverage @ {float(args.near_threshold):.2f}A: `{near_metrics['truth_near_coverage']:.4f}`" if np.isfinite(near_metrics["truth_near_coverage"]) else f"- truth_near_coverage @ {float(args.near_threshold):.2f}A: `nan`",
        f"- predicted_near_precision @ {float(args.near_threshold):.2f}A: `{near_metrics['predicted_near_precision']:.4f}`" if np.isfinite(near_metrics["predicted_near_precision"]) else f"- predicted_near_precision @ {float(args.near_threshold):.2f}A: `nan`",
        f"- mean_truth_min_distance: `{near_metrics['mean_truth_min_distance']:.4f}`" if np.isfinite(near_metrics["mean_truth_min_distance"]) else "- mean_truth_min_distance: `nan`",
        "",
        "## Boundary Tightness Metrics",
        "",
        f"- predicted_to_truth_residue_ratio: `{near_metrics['predicted_to_truth_residue_ratio']:.4f}`" if np.isfinite(near_metrics["predicted_to_truth_residue_ratio"]) else "- predicted_to_truth_residue_ratio: `nan`",
        f"- extra_predicted_fraction: `{near_metrics['extra_predicted_fraction']:.4f}`" if np.isfinite(near_metrics["extra_predicted_fraction"]) else "- extra_predicted_fraction: `nan`",
        f"- far_predicted_fraction: `{near_metrics['far_predicted_fraction']:.4f}`" if np.isfinite(near_metrics["far_predicted_fraction"]) else "- far_predicted_fraction: `nan`",
        f"- overwide_pocket_score: `{near_metrics['overwide_pocket_score']:.4f}`" if np.isfinite(near_metrics["overwide_pocket_score"]) else "- overwide_pocket_score: `nan`",
        "",
        "## Outputs",
        "",
        "- `cd38_pocket_accuracy_summary.json`",
        "- `truth_residue_table.csv`",
        "- `predicted_residue_table.csv`",
        "- `exact_overlap_table.csv`",
        "- `missed_truth_table.csv`",
        "- `extra_predicted_table.csv`",
        "- `near_hit_table.csv`",
        "- `predicted_to_truth_distance_table.csv`",
    ]
    report_path = out_dir / "cd38_pocket_accuracy_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Saved: {summary_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
