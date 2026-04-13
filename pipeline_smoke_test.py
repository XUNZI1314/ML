"""End-to-end smoke test for rule + ML + calibration + comparison pipeline.

This script creates a deterministic synthetic `pose_features.csv`, then runs:
1) rule_ranker.py
2) train_pose_model.py
3) rank_nanobodies.py
4) compare_rule_ml_rankings.py (baseline rule vs ML)
5) calibrate_rule_ranker.py (optionally guarded by baseline metrics)
6) compare_rule_ml_rankings.py (calibrated rule vs ML)
7) summarize_rule_ml_improvement.py (baseline vs calibrated deltas)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline_runner_common import (
    CommandResult,
    build_execution_report,
    ensure_files_exist,
    load_strategy_seed,
    resolve_calibration_strategy,
    resolve_strategy_seed_path,
    run_command,
    serialize_command_results,
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _make_synthetic_pose_features(
    out_csv: Path,
    n_nanobodies: int,
    n_conformers: int,
    n_poses: int,
    seed: int,
) -> dict[str, Any]:
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

    # Inject a small controlled missingness pattern.
    rng_missing = np.random.default_rng(int(seed) + 999)
    for col in ["mmgbsa", "interface_dg", "rsite_accuracy"]:
        miss_mask = rng_missing.random(df.shape[0]) < 0.03
        df.loc[miss_mask, col] = np.nan

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    return {
        "rows": int(df.shape[0]),
        "nanobodies": int(df["nanobody_id"].nunique()),
        "conformers": int(df[["nanobody_id", "conformer_id"]].drop_duplicates().shape[0]),
        "positive_rate": float(df["label"].mean()),
        "output_csv": str(out_csv),
    }

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic end-to-end smoke test")
    parser.add_argument("--out_dir", default="smoke_test_outputs", help="Output directory")
    parser.add_argument("--seed", type=int, default=20260408)
    parser.add_argument("--n_nanobodies", type=int, default=14)
    parser.add_argument("--n_conformers", type=int, default=3)
    parser.add_argument("--n_poses", type=int, default=8)

    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--n_feature_trials", type=int, default=24)
    parser.add_argument("--top_feature_trials_for_agg", type=int, default=6)
    parser.add_argument("--calibration_rank_consistency_weight", type=float, default=0.40)
    parser.add_argument(
        "--calibration_rank_consistency_metric",
        choices=["score_spearman", "rank_spearman"],
        default="score_spearman",
    )
    parser.add_argument(
        "--calibration_selection_metric",
        choices=["objective", "nanobody_auc", "rank_consistency"],
        default="objective",
    )
    parser.add_argument(
        "--disable_calibration_baseline_guard",
        action="store_true",
        help="Disable baseline-anchored constraints when running calibration",
    )
    parser.add_argument(
        "--calibration_rank_guard_tolerance",
        type=float,
        default=0.005,
        help="Allowed rank-consistency drop from baseline when baseline guard is enabled",
    )
    parser.add_argument(
        "--calibration_auc_guard_tolerance",
        type=float,
        default=0.0,
        help="Allowed nanobody-AUC drop from baseline when baseline guard is enabled",
    )
    parser.add_argument(
        "--strategy_seed_json",
        default=None,
        help="Optional recommended_strategy.json used to seed calibration strategy parameters",
    )
    parser.add_argument(
        "--disable_auto_seed_from_previous_strategy",
        action="store_true",
        help="Disable auto-loading <out_dir>/strategy_optimization/recommended_strategy.json before calibration",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    repo_root = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    input_dir = out_dir / "input"
    rule_out = out_dir / "rule_outputs"
    ml_model_out = out_dir / "model_outputs"
    ml_rank_out = out_dir / "ml_ranking_outputs"
    calibration_out = out_dir / "calibration_outputs"
    compare_rule_vs_ml_out = out_dir / "comparison_rule_vs_ml"
    compare_calib_vs_ml_out = out_dir / "comparison_calibrated_rule_vs_ml"
    improvement_out = out_dir / "improvement_summary"
    strategy_opt_out = out_dir / "strategy_optimization"

    pose_feature_csv = input_dir / "pose_features.csv"
    synth_summary = _make_synthetic_pose_features(
        out_csv=pose_feature_csv,
        n_nanobodies=int(args.n_nanobodies),
        n_conformers=int(args.n_conformers),
        n_poses=int(args.n_poses),
        seed=int(args.seed),
    )

    py = sys.executable
    commands: list[CommandResult] = []

    commands.append(
        run_command(
            name="rule_ranker",
            command=[
                py,
                str(repo_root / "rule_ranker.py"),
                "--feature_csv",
                str(pose_feature_csv),
                "--out_dir",
                str(rule_out),
                "--top_k",
                "3",
            ],
            cwd=repo_root,
        )
    )

    commands.append(
        run_command(
            name="train_pose_model",
            command=[
                py,
                str(repo_root / "train_pose_model.py"),
                "--feature_csv",
                str(pose_feature_csv),
                "--out_dir",
                str(ml_model_out),
                "--epochs",
                str(int(args.train_epochs)),
                "--batch_size",
                "64",
                "--val_ratio",
                "0.25",
                "--early_stopping_patience",
                "5",
                "--seed",
                str(int(args.seed)),
            ],
            cwd=repo_root,
        )
    )

    commands.append(
        run_command(
            name="rank_nanobodies",
            command=[
                py,
                str(repo_root / "rank_nanobodies.py"),
                "--pred_csv",
                str(ml_model_out / "pose_predictions.csv"),
                "--out_dir",
                str(ml_rank_out),
                "--top_k",
                "3",
            ],
            cwd=repo_root,
        )
    )

    commands.append(
        run_command(
            name="compare_rule_vs_ml",
            command=[
                py,
                str(repo_root / "compare_rule_ml_rankings.py"),
                "--rule_csv",
                str(rule_out / "nanobody_rule_ranking.csv"),
                "--ml_csv",
                str(ml_rank_out / "nanobody_ranking.csv"),
                "--pose_feature_csv",
                str(pose_feature_csv),
                "--label_col",
                "label",
                "--out_dir",
                str(compare_rule_vs_ml_out),
            ],
            cwd=repo_root,
        )
    )

    baseline_summary_path = compare_rule_vs_ml_out / "ranking_comparison_summary.json"
    baseline_summary_pre = json.loads(baseline_summary_path.read_text(encoding="utf-8"))

    strategy_seed_path = resolve_strategy_seed_path(
        repo_root=repo_root,
        strategy_seed_json=args.strategy_seed_json,
        auto_seed_path=strategy_opt_out / "recommended_strategy.json",
        disable_auto_seed=bool(args.disable_auto_seed_from_previous_strategy),
    )

    strategy_seed_payload: dict[str, Any] | None = None
    if strategy_seed_path is not None:
        strategy_seed_payload = load_strategy_seed(strategy_seed_path)

    calibration_strategy = resolve_calibration_strategy(
        default_rank_consistency_weight=float(args.calibration_rank_consistency_weight),
        default_selection_metric=str(args.calibration_selection_metric),
        strategy_seed_payload=strategy_seed_payload,
        strategy_seed_path=strategy_seed_path,
        baseline_summary=baseline_summary_pre,
        enable_baseline_guard=not bool(args.disable_calibration_baseline_guard),
        rank_guard_tolerance=float(args.calibration_rank_guard_tolerance),
        auc_guard_tolerance=float(args.calibration_auc_guard_tolerance),
    )
    strategy_seed_applied: dict[str, Any] = dict(calibration_strategy["summary"])

    calibrate_cmd = [
        py,
        str(repo_root / "calibrate_rule_ranker.py"),
        "--feature_csv",
        str(pose_feature_csv),
        "--label_col",
        "label",
        "--out_dir",
        str(calibration_out),
        "--n_feature_trials",
        str(int(args.n_feature_trials)),
        "--top_feature_trials_for_agg",
        str(int(args.top_feature_trials_for_agg)),
        "--ml_ranking_csv",
        str(ml_rank_out / "nanobody_ranking.csv"),
        "--ml_score_col",
        "final_score",
        "--rank_consistency_metric",
        str(args.calibration_rank_consistency_metric),
        "--rank_consistency_weight",
        str(float(calibration_strategy["rank_consistency_weight"])),
        "--selection_metric",
        str(calibration_strategy["selection_metric"]),
        "--seed",
        str(int(args.seed)),
    ]
    if calibration_strategy["min_rank_consistency"] is not None:
        calibrate_cmd.extend(
            ["--min_rank_consistency", f"{float(calibration_strategy['min_rank_consistency']):.12f}"]
        )
    if calibration_strategy["min_nanobody_auc"] is not None:
        calibrate_cmd.extend(
            ["--min_nanobody_auc", f"{float(calibration_strategy['min_nanobody_auc']):.12f}"]
        )

    commands.append(
        run_command(
            name="calibrate_rule_ranker",
            command=calibrate_cmd,
            cwd=repo_root,
        )
    )

    commands.append(
        run_command(
            name="compare_calibrated_rule_vs_ml",
            command=[
                py,
                str(repo_root / "compare_rule_ml_rankings.py"),
                "--rule_csv",
                str(calibration_out / "calibrated_rule_outputs" / "nanobody_rule_ranking.csv"),
                "--ml_csv",
                str(ml_rank_out / "nanobody_ranking.csv"),
                "--pose_feature_csv",
                str(pose_feature_csv),
                "--label_col",
                "label",
                "--out_dir",
                str(compare_calib_vs_ml_out),
            ],
            cwd=repo_root,
        )
    )

    commands.append(
        run_command(
            name="summarize_rule_ml_improvement",
            command=[
                py,
                str(repo_root / "summarize_rule_ml_improvement.py"),
                "--baseline_summary",
                str(compare_rule_vs_ml_out / "ranking_comparison_summary.json"),
                "--calibrated_summary",
                str(compare_calib_vs_ml_out / "ranking_comparison_summary.json"),
                "--calibrated_config",
                str(calibration_out / "calibrated_rule_config.json"),
                "--out_dir",
                str(improvement_out),
            ],
            cwd=repo_root,
        )
    )

    commands.append(
        run_command(
            name="optimize_calibration_strategy",
            command=[
                py,
                str(repo_root / "optimize_calibration_strategy.py"),
                "--aggregation_trials_csv",
                str(calibration_out / "aggregation_calibration_trials.csv"),
                "--baseline_summary_json",
                str(compare_rule_vs_ml_out / "ranking_comparison_summary.json"),
                "--use_baseline_guard",
                "--rank_guard_tolerance",
                str(float(args.calibration_rank_guard_tolerance)),
                "--auc_guard_tolerance",
                str(float(args.calibration_auc_guard_tolerance)),
                "--out_dir",
                str(strategy_opt_out),
            ],
            cwd=repo_root,
        )
    )

    required_files = [
        rule_out / "pose_rule_scores.csv",
        rule_out / "conformer_rule_scores.csv",
        rule_out / "nanobody_rule_ranking.csv",
        ml_model_out / "pose_predictions.csv",
        ml_model_out / "training_summary.json",
        ml_rank_out / "conformer_scores.csv",
        ml_rank_out / "nanobody_ranking.csv",
        calibration_out / "calibrated_rule_config.json",
        calibration_out / "feature_calibration_trials.csv",
        calibration_out / "aggregation_calibration_trials.csv",
        calibration_out / "calibrated_rule_outputs" / "pose_rule_scores.csv",
        calibration_out / "calibrated_rule_outputs" / "conformer_rule_scores.csv",
        calibration_out / "calibrated_rule_outputs" / "nanobody_rule_ranking.csv",
        compare_rule_vs_ml_out / "ranking_comparison_table.csv",
        compare_rule_vs_ml_out / "ranking_comparison_summary.json",
        compare_rule_vs_ml_out / "ranking_comparison_report.md",
        compare_calib_vs_ml_out / "ranking_comparison_table.csv",
        compare_calib_vs_ml_out / "ranking_comparison_summary.json",
        compare_calib_vs_ml_out / "ranking_comparison_report.md",
        improvement_out / "calibration_improvement_metrics.csv",
        improvement_out / "calibration_improvement_summary.json",
        improvement_out / "calibration_improvement_report.md",
        strategy_opt_out / "strategy_sweep_results.csv",
        strategy_opt_out / "recommended_strategy.json",
        strategy_opt_out / "recommended_strategy_report.md",
    ]
    ensure_files_exist(required_files)

    baseline_summary = json.loads((compare_rule_vs_ml_out / "ranking_comparison_summary.json").read_text(encoding="utf-8"))
    calibrated_summary = json.loads((compare_calib_vs_ml_out / "ranking_comparison_summary.json").read_text(encoding="utf-8"))
    improvement_summary = json.loads((improvement_out / "calibration_improvement_summary.json").read_text(encoding="utf-8"))
    strategy_summary = json.loads((strategy_opt_out / "recommended_strategy.json").read_text(encoding="utf-8"))
    report_path = out_dir / "smoke_test_report.md"
    artifacts = {
        "smoke_summary_json": str(out_dir / "smoke_test_summary.json"),
        "rule_ranking_csv": str(rule_out / "nanobody_rule_ranking.csv"),
        "ml_ranking_csv": str(ml_rank_out / "nanobody_ranking.csv"),
        "baseline_comparison_summary_json": str(compare_rule_vs_ml_out / "ranking_comparison_summary.json"),
        "baseline_comparison_report_md": str(compare_rule_vs_ml_out / "ranking_comparison_report.md"),
        "calibrated_rule_config_json": str(calibration_out / "calibrated_rule_config.json"),
        "calibrated_comparison_summary_json": str(compare_calib_vs_ml_out / "ranking_comparison_summary.json"),
        "calibrated_comparison_report_md": str(compare_calib_vs_ml_out / "ranking_comparison_report.md"),
        "improvement_summary_json": str(improvement_out / "calibration_improvement_summary.json"),
        "improvement_report_md": str(improvement_out / "calibration_improvement_report.md"),
        "strategy_recommended_json": str(strategy_opt_out / "recommended_strategy.json"),
        "strategy_report_md": str(strategy_opt_out / "recommended_strategy_report.md"),
        "execution_report_md": str(report_path),
    }

    smoke_summary = {
        "synthetic_data": synth_summary,
        "commands": serialize_command_results(commands),
        "comparison": {
            "baseline_rule_vs_ml": baseline_summary,
            "calibrated_rule_vs_ml": calibrated_summary,
        },
        "calibration_strategy_seed": strategy_seed_applied,
        "improvement": improvement_summary,
        "strategy_optimization": strategy_summary,
        "artifacts": artifacts,
    }

    smoke_summary_path = out_dir / "smoke_test_summary.json"
    smoke_summary_path.write_text(json.dumps(smoke_summary, ensure_ascii=True, indent=2), encoding="utf-8")
    report_path.write_text(
        build_execution_report(
            title="Pipeline Smoke Test Report",
            start_mode="synthetic_feature_csv",
            feature_csv=str(pose_feature_csv),
            synthetic_data=synth_summary,
            strategy_seed_summary=strategy_seed_applied,
            baseline_summary=baseline_summary,
            calibrated_summary=calibrated_summary,
            improvement_summary=improvement_summary,
            strategy_summary=strategy_summary,
            artifacts=artifacts,
            commands=commands,
            notes=[
                "Synthetic pose_features.csv is regenerated deterministically from the provided seed.",
                "Calibration is executed on the synthetic labels and can optionally reuse a previous recommended strategy.",
            ],
        ),
        encoding="utf-8",
    )

    print(f"Smoke test completed. Summary: {smoke_summary_path}")
    print(f"Report: {report_path}")
    print(f"Baseline rule-vs-ML spearman(rank): {baseline_summary.get('rank_spearman', float('nan')):.4f}")
    print(f"Calibrated rule-vs-ML spearman(rank): {calibrated_summary.get('rank_spearman', float('nan')):.4f}")

    # Print the most tracked gain: rule AUC delta.
    rule_auc_delta = float("nan")
    for item in improvement_summary.get("metrics", []) or []:
        if str(item.get("metric_key")) == "rule_auc":
            try:
                rule_auc_delta = float(item.get("delta"))
            except (TypeError, ValueError):
                rule_auc_delta = float("nan")
            break
    print(f"Calibrated minus baseline rule AUC: {rule_auc_delta:+.4f}")


if __name__ == "__main__":
    main()
