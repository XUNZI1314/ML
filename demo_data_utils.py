from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEMO_SIGNAL_COLUMNS = [
    "pocket_hit_fraction",
    "catalytic_hit_fraction",
    "mouth_occlusion_score",
    "substrate_overlap_score",
    "ligand_path_block_score",
    "ligand_path_exit_block_fraction",
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def make_synthetic_pose_features(
    out_csv: str | Path,
    *,
    n_nanobodies: int = 14,
    n_conformers: int = 3,
    n_poses: int = 8,
    seed: int = 20260408,
) -> dict[str, Any]:
    """Create a deterministic demo/smoke-test pose_features.csv."""

    out_path = Path(out_csv).expanduser().resolve()
    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, Any]] = []

    for nb_idx in range(int(n_nanobodies)):
        nanobody_id = f"NB_{nb_idx + 1:03d}"
        nb_latent = rng.normal(0.0, 0.9)

        for cf_idx in range(int(n_conformers)):
            conformer_id = f"CF_{cf_idx + 1:02d}"
            cf_latent = nb_latent + rng.normal(0.0, 0.35)

            for ps_idx in range(int(n_poses)):
                pose_id = f"P_{ps_idx + 1:03d}"
                pose_latent = cf_latent + rng.normal(0.0, 0.50)
                block_prob = float(_sigmoid(np.array([1.15 * pose_latent]))[0])

                pocket_hit = float(_clip01(np.array([0.10 + 0.78 * block_prob + rng.normal(0.0, 0.08)]))[0])
                catalytic_hit = float(_clip01(np.array([0.08 + 0.70 * block_prob + rng.normal(0.0, 0.09)]))[0])
                mouth_occ = float(_clip01(np.array([0.12 + 0.73 * block_prob + rng.normal(0.0, 0.10)]))[0])
                mouth_axis_block = float(_clip01(np.array([0.10 + 0.76 * block_prob + rng.normal(0.0, 0.09)]))[0])
                mouth_aperture_block = float(_clip01(np.array([0.08 + 0.72 * block_prob + rng.normal(0.0, 0.09)]))[0])
                mouth_min_clearance = float(np.clip(4.4 - 2.3 * block_prob + rng.normal(0.0, 0.40), 0.4, 7.5))
                delta_occ = float(_clip01(np.array([0.10 + 0.68 * block_prob + rng.normal(0.0, 0.10)]))[0])
                substrate_overlap = float(_clip01(np.array([0.08 + 0.65 * block_prob + rng.normal(0.0, 0.11)]))[0])
                ligand_path_block = float(_clip01(np.array([0.08 + 0.67 * block_prob + rng.normal(0.0, 0.11)]))[0])
                ligand_path_fraction = float(_clip01(np.array([0.10 + 0.70 * block_prob + rng.normal(0.0, 0.09)]))[0])
                ligand_path_bottleneck = float(_clip01(np.array([0.08 + 0.72 * block_prob + rng.normal(0.0, 0.09)]))[0])
                ligand_path_exit_block = float(_clip01(np.array([0.08 + 0.74 * block_prob + rng.normal(0.0, 0.08)]))[0])
                ligand_path_clearance = float(np.clip(4.8 - 2.6 * block_prob + rng.normal(0.0, 0.45), 0.4, 8.0))

                min_distance = float(np.clip(7.0 - 4.5 * block_prob + rng.normal(0.0, 0.7), 0.8, 12.0))
                rsite_accuracy = float(_clip01(np.array([0.30 + 0.55 * block_prob + rng.normal(0.0, 0.10)]))[0])
                mmgbsa = float(np.clip(-30.0 - 38.0 * block_prob + rng.normal(0.0, 4.0), -120.0, 15.0))
                interface_dg = float(np.clip(-3.0 - 7.5 * block_prob + rng.normal(0.0, 1.3), -25.0, 8.0))
                hdock_score = float(np.clip(-110.0 - 220.0 * block_prob + rng.normal(0.0, 30.0), -800.0, -10.0))

                label_prob = float(np.clip(0.05 + 0.90 * block_prob, 0.01, 0.99))
                label = int(rng.binomial(1, label_prob))

                rows.append(
                    {
                        "nanobody_id": nanobody_id,
                        "conformer_id": conformer_id,
                        "pose_id": pose_id,
                        "status": "ok",
                        "pocket_hit_fraction": pocket_hit,
                        "catalytic_hit_fraction": catalytic_hit,
                        "mouth_occlusion_score": mouth_occ,
                        "mouth_axis_block_fraction": mouth_axis_block,
                        "mouth_aperture_block_fraction": mouth_aperture_block,
                        "mouth_min_clearance": mouth_min_clearance,
                        "delta_pocket_occupancy_proxy": delta_occ,
                        "substrate_overlap_score": substrate_overlap,
                        "ligand_path_block_score": ligand_path_block,
                        "ligand_path_block_fraction": ligand_path_fraction,
                        "ligand_path_bottleneck_score": ligand_path_bottleneck,
                        "ligand_path_exit_block_fraction": ligand_path_exit_block,
                        "ligand_path_min_clearance": ligand_path_clearance,
                        "min_distance_to_pocket": min_distance,
                        "rsite_accuracy": rsite_accuracy,
                        "mmgbsa": mmgbsa,
                        "interface_dg": interface_dg,
                        "hdock_score": hdock_score,
                        "label": float(label),
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Synthetic feature table is empty.")

    rng_missing = np.random.default_rng(int(seed) + 999)
    for col in ["mmgbsa", "interface_dg", "rsite_accuracy"]:
        miss_mask = rng_missing.random(df.shape[0]) < 0.03
        df.loc[miss_mask, col] = np.nan

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    return {
        "rows": int(df.shape[0]),
        "nanobodies": int(df["nanobody_id"].nunique()),
        "conformers": int(df[["nanobody_id", "conformer_id"]].drop_duplicates().shape[0]),
        "positive_rate": float(df["label"].mean()),
        "output_csv": str(out_path),
        "seed": int(seed),
    }


def build_demo_experiment_override(
    *,
    feature_csv: str | Path,
    out_csv: str | Path,
) -> dict[str, Any]:
    """Create a deterministic demo experiment override with balanced validation labels."""

    feature_path = Path(feature_csv).expanduser().resolve()
    out_path = Path(out_csv).expanduser().resolve()
    df = pd.read_csv(feature_path, low_memory=False)
    if "nanobody_id" not in df.columns:
        raise ValueError("feature_csv must contain nanobody_id")

    signal_cols = [col for col in DEMO_SIGNAL_COLUMNS if col in df.columns]
    if not signal_cols:
        raise ValueError("feature_csv does not contain demo signal columns")

    work = df.copy()
    work["_demo_signal"] = work[signal_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    grouped = (
        work.groupby("nanobody_id", dropna=False)["_demo_signal"]
        .mean()
        .reset_index()
        .sort_values("_demo_signal", ascending=False)
        .reset_index(drop=True)
    )
    if grouped.empty:
        raise ValueError("No nanobody rows available for demo experiment override")

    split_index = max(1, int(np.ceil(len(grouped) / 2.0)))
    labels = [1 if idx < split_index else 0 for idx in range(len(grouped))]
    if len(grouped) == 1:
        labels = [1]

    rows: list[dict[str, Any]] = []
    for idx, row in grouped.iterrows():
        label = int(labels[idx])
        rows.append(
            {
                "nanobody_id": str(row["nanobody_id"]),
                "plan_override": "",
                "experiment_status": "",
                "experiment_result": "positive" if label == 1 else "negative",
                "validation_label": label,
                "experiment_owner": "demo",
                "experiment_cost": 0,
                "experiment_note": "Synthetic demo validation label generated from blocking proxy signal.",
                "manual_plan_reason": "Demo data only; do not treat as wet-lab evidence.",
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return {
        "output_csv": str(out_path),
        "candidate_count": int(len(rows)),
        "positive_count": int(sum(labels)),
        "negative_count": int(len(labels) - sum(labels)),
        "source_feature_csv": str(feature_path),
    }


def write_demo_manifest(
    *,
    out_json: str | Path,
    feature_summary: dict[str, Any],
    override_summary: dict[str, Any],
    pipeline_summary_json: str | Path | None = None,
) -> dict[str, Any]:
    manifest = {
        "manifest_type": "ml_demo_dataset",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "feature_summary": feature_summary,
        "override_summary": override_summary,
        "pipeline_summary_json": str(Path(pipeline_summary_json).expanduser().resolve())
        if pipeline_summary_json is not None
        else None,
        "notes": [
            "This is synthetic demo data for workflow demonstration only.",
            "Synthetic validation labels are generated from proxy signal and are not wet-lab evidence.",
        ],
    }
    out_path = Path(out_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")
    return manifest
