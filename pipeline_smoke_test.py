"""End-to-end smoke test for rule + ML + calibration + comparison pipeline.

This script creates a deterministic synthetic `pose_features.csv`, then runs:
1) rule_ranker.py
2) train_pose_model.py
3) rank_nanobodies.py
4) build_consensus_ranking.py (Rule + ML decision-support consensus)
5) build_quality_gate.py (PASS/WARN/FAIL run-quality decision)
6) build_geometry_proxy_audit.py (mouth/path/pocket proxy consistency audit)
7) build_score_explanation_cards.py (product-facing score interpretation)
8) analyze_ranking_parameter_sensitivity.py (ranking robustness check)
9) build_candidate_comparisons.py (pairwise candidate trade-off explanations)
10) build_candidate_report_cards.py (per-nanobody HTML reports)
11) suggest_next_experiments.py (active-learning style review queue)
12) build_validation_evidence_audit.py (top-candidate validation evidence coverage)
13) build_batch_decision_summary.py (run-level conclusion and risk summary)
14) compare_rule_ml_rankings.py (baseline rule vs ML)
15) calibrate_rule_ranker.py (optionally guarded by baseline metrics)
16) compare_rule_ml_rankings.py (calibrated rule vs ML)
17) summarize_rule_ml_improvement.py (baseline vs calibrated deltas)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from demo_data_utils import make_synthetic_pose_features
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
    consensus_out = out_dir / "consensus_outputs"
    quality_gate_out = out_dir / "quality_gate"
    geometry_audit_out = out_dir / "geometry_proxy_audit"
    score_explanation_out = out_dir / "score_explanation_cards"
    sensitivity_out = out_dir / "parameter_sensitivity"
    candidate_report_out = out_dir / "candidate_report_cards"
    candidate_report_zip = out_dir / "candidate_report_cards.zip"
    candidate_compare_out = out_dir / "candidate_comparisons"
    experiment_suggestion_out = out_dir / "experiment_suggestions"
    validation_evidence_out = out_dir / "validation_evidence_audit"
    batch_decision_out = out_dir / "batch_decision_summary"
    calibration_out = out_dir / "calibration_outputs"
    compare_rule_vs_ml_out = out_dir / "comparison_rule_vs_ml"
    compare_calib_vs_ml_out = out_dir / "comparison_calibrated_rule_vs_ml"
    improvement_out = out_dir / "improvement_summary"
    strategy_opt_out = out_dir / "strategy_optimization"

    pose_feature_csv = input_dir / "pose_features.csv"
    synth_summary = make_synthetic_pose_features(
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
            name="build_consensus_ranking",
            command=[
                py,
                str(repo_root / "build_consensus_ranking.py"),
                "--rule_csv",
                str(rule_out / "nanobody_rule_ranking.csv"),
                "--ml_csv",
                str(ml_rank_out / "nanobody_ranking.csv"),
                "--feature_csv",
                str(pose_feature_csv),
                "--out_dir",
                str(consensus_out),
            ],
            cwd=repo_root,
        )
    )

    commands.append(
        run_command(
            name="build_quality_gate",
            command=[
                py,
                str(repo_root / "build_quality_gate.py"),
                "--feature_csv",
                str(pose_feature_csv),
                "--out_dir",
                str(quality_gate_out),
                "--label_col",
                "label",
            ],
            cwd=repo_root,
        )
    )

    commands.append(
        run_command(
            name="build_geometry_proxy_audit",
            command=[
                py,
                str(repo_root / "build_geometry_proxy_audit.py"),
                "--feature_csv",
                str(pose_feature_csv),
                "--out_dir",
                str(geometry_audit_out),
            ],
            cwd=repo_root,
        )
    )

    commands.append(
        run_command(
            name="build_score_explanation_cards",
            command=[
                py,
                str(repo_root / "build_score_explanation_cards.py"),
                "--consensus_csv",
                str(consensus_out / "consensus_ranking.csv"),
                "--feature_csv",
                str(pose_feature_csv),
                "--label_col",
                "label",
                "--out_dir",
                str(score_explanation_out),
            ],
            cwd=repo_root,
        )
    )

    commands.append(
        run_command(
            name="analyze_ranking_parameter_sensitivity",
            command=[
                py,
                str(repo_root / "analyze_ranking_parameter_sensitivity.py"),
                "--consensus_csv",
                str(consensus_out / "consensus_ranking.csv"),
                "--out_dir",
                str(sensitivity_out),
                "--top_n",
                "5",
            ],
            cwd=repo_root,
        )
    )

    commands.append(
        run_command(
            name="build_candidate_comparisons",
            command=[
                py,
                str(repo_root / "build_candidate_comparisons.py"),
                "--consensus_csv",
                str(consensus_out / "consensus_ranking.csv"),
                "--out_dir",
                str(candidate_compare_out),
                "--top_n",
                "12",
                "--pair_mode",
                "adjacent",
            ],
            cwd=repo_root,
        )
    )

    commands.append(
        run_command(
            name="build_candidate_report_cards",
            command=[
                py,
                str(repo_root / "build_candidate_report_cards.py"),
                "--consensus_csv",
                str(consensus_out / "consensus_ranking.csv"),
                "--rule_csv",
                str(rule_out / "nanobody_rule_ranking.csv"),
                "--ml_csv",
                str(ml_rank_out / "nanobody_ranking.csv"),
                "--feature_csv",
                str(pose_feature_csv),
                "--pose_predictions_csv",
                str(ml_model_out / "pose_predictions.csv"),
                "--candidate_pairwise_csv",
                str(candidate_compare_out / "candidate_pairwise_comparisons.csv"),
                "--out_dir",
                str(candidate_report_out),
                "--zip_path",
                str(candidate_report_zip),
            ],
            cwd=repo_root,
        )
    )

    commands.append(
        run_command(
            name="suggest_next_experiments",
            command=[
                py,
                str(repo_root / "suggest_next_experiments.py"),
                "--consensus_csv",
                str(consensus_out / "consensus_ranking.csv"),
                "--out_dir",
                str(experiment_suggestion_out),
            ],
            cwd=repo_root,
        )
    )

    commands.append(
        run_command(
            name="build_validation_evidence_audit",
            command=[
                py,
                str(repo_root / "build_validation_evidence_audit.py"),
                "--consensus_csv",
                str(consensus_out / "consensus_ranking.csv"),
                "--experiment_plan_csv",
                str(experiment_suggestion_out / "experiment_plan.csv"),
                "--experiment_plan_state_ledger_csv",
                str(experiment_suggestion_out / "experiment_plan_state_ledger.csv"),
                "--out_dir",
                str(validation_evidence_out),
                "--top_k",
                "10",
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
        consensus_out / "consensus_ranking.csv",
        consensus_out / "consensus_summary.json",
        consensus_out / "consensus_report.md",
        quality_gate_out / "quality_gate_summary.json",
        quality_gate_out / "quality_gate_checks.csv",
        quality_gate_out / "quality_gate_report.md",
        geometry_audit_out / "geometry_proxy_audit_summary.json",
        geometry_audit_out / "geometry_proxy_audit_report.md",
        geometry_audit_out / "geometry_proxy_feature_summary.csv",
        geometry_audit_out / "geometry_proxy_flagged_poses.csv",
        geometry_audit_out / "geometry_proxy_candidate_audit.csv",
        score_explanation_out / "score_explanation_cards.csv",
        score_explanation_out / "score_explanation_cards_summary.json",
        score_explanation_out / "score_explanation_cards.md",
        score_explanation_out / "score_explanation_cards.html",
        sensitivity_out / "scenario_rankings.csv",
        sensitivity_out / "scenario_summary.csv",
        sensitivity_out / "candidate_rank_sensitivity.csv",
        sensitivity_out / "sensitive_candidates.csv",
        sensitivity_out / "parameter_sensitivity_summary.json",
        sensitivity_out / "parameter_sensitivity_report.md",
        candidate_report_out / "index.html",
        candidate_report_out / "candidate_report_cards.csv",
        candidate_report_out / "candidate_report_cards_summary.json",
        candidate_report_zip,
        candidate_compare_out / "candidate_tradeoff_table.csv",
        candidate_compare_out / "candidate_pairwise_comparisons.csv",
        candidate_compare_out / "candidate_comparison_summary.json",
        candidate_compare_out / "candidate_comparison_report.md",
        experiment_suggestion_out / "next_experiment_suggestions.csv",
        experiment_suggestion_out / "next_experiment_suggestions_summary.json",
        experiment_suggestion_out / "next_experiment_suggestions_report.md",
        experiment_suggestion_out / "experiment_plan.csv",
        experiment_suggestion_out / "experiment_plan_summary.json",
        experiment_suggestion_out / "experiment_plan.md",
        experiment_suggestion_out / "experiment_plan_state_ledger.csv",
        validation_evidence_out / "validation_evidence_summary.json",
        validation_evidence_out / "validation_evidence_report.md",
        validation_evidence_out / "validation_evidence_by_candidate.csv",
        validation_evidence_out / "validation_evidence_topk.csv",
        validation_evidence_out / "validation_evidence_action_items.csv",
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
        "consensus_ranking_csv": str(consensus_out / "consensus_ranking.csv"),
        "consensus_summary_json": str(consensus_out / "consensus_summary.json"),
        "consensus_report_md": str(consensus_out / "consensus_report.md"),
        "quality_gate_summary_json": str(quality_gate_out / "quality_gate_summary.json"),
        "quality_gate_checks_csv": str(quality_gate_out / "quality_gate_checks.csv"),
        "quality_gate_report_md": str(quality_gate_out / "quality_gate_report.md"),
        "geometry_proxy_audit_summary_json": str(geometry_audit_out / "geometry_proxy_audit_summary.json"),
        "geometry_proxy_audit_report_md": str(geometry_audit_out / "geometry_proxy_audit_report.md"),
        "geometry_proxy_feature_summary_csv": str(geometry_audit_out / "geometry_proxy_feature_summary.csv"),
        "geometry_proxy_flagged_poses_csv": str(geometry_audit_out / "geometry_proxy_flagged_poses.csv"),
        "geometry_proxy_candidate_audit_csv": str(geometry_audit_out / "geometry_proxy_candidate_audit.csv"),
        "score_explanation_cards_csv": str(score_explanation_out / "score_explanation_cards.csv"),
        "score_explanation_cards_summary_json": str(score_explanation_out / "score_explanation_cards_summary.json"),
        "score_explanation_cards_md": str(score_explanation_out / "score_explanation_cards.md"),
        "score_explanation_cards_html": str(score_explanation_out / "score_explanation_cards.html"),
        "parameter_sensitivity_candidate_csv": str(sensitivity_out / "candidate_rank_sensitivity.csv"),
        "parameter_sensitivity_sensitive_csv": str(sensitivity_out / "sensitive_candidates.csv"),
        "parameter_sensitivity_summary_json": str(sensitivity_out / "parameter_sensitivity_summary.json"),
        "parameter_sensitivity_report_md": str(sensitivity_out / "parameter_sensitivity_report.md"),
        "candidate_report_index_html": str(candidate_report_out / "index.html"),
        "candidate_report_manifest_csv": str(candidate_report_out / "candidate_report_cards.csv"),
        "candidate_report_summary_json": str(candidate_report_out / "candidate_report_cards_summary.json"),
        "candidate_report_zip": str(candidate_report_zip),
        "candidate_tradeoff_table_csv": str(candidate_compare_out / "candidate_tradeoff_table.csv"),
        "candidate_pairwise_comparisons_csv": str(candidate_compare_out / "candidate_pairwise_comparisons.csv"),
        "candidate_comparison_summary_json": str(candidate_compare_out / "candidate_comparison_summary.json"),
        "candidate_comparison_report_md": str(candidate_compare_out / "candidate_comparison_report.md"),
        "experiment_suggestions_csv": str(experiment_suggestion_out / "next_experiment_suggestions.csv"),
        "experiment_suggestions_summary_json": str(experiment_suggestion_out / "next_experiment_suggestions_summary.json"),
        "experiment_suggestions_report_md": str(experiment_suggestion_out / "next_experiment_suggestions_report.md"),
        "experiment_plan_csv": str(experiment_suggestion_out / "experiment_plan.csv"),
        "experiment_plan_summary_json": str(experiment_suggestion_out / "experiment_plan_summary.json"),
        "experiment_plan_md": str(experiment_suggestion_out / "experiment_plan.md"),
        "experiment_plan_state_ledger_csv": str(experiment_suggestion_out / "experiment_plan_state_ledger.csv"),
        "validation_evidence_summary_json": str(validation_evidence_out / "validation_evidence_summary.json"),
        "validation_evidence_report_md": str(validation_evidence_out / "validation_evidence_report.md"),
        "validation_evidence_by_candidate_csv": str(validation_evidence_out / "validation_evidence_by_candidate.csv"),
        "validation_evidence_topk_csv": str(validation_evidence_out / "validation_evidence_topk.csv"),
        "validation_evidence_action_items_csv": str(validation_evidence_out / "validation_evidence_action_items.csv"),
        "batch_decision_summary_json": str(batch_decision_out / "batch_decision_summary.json"),
        "batch_decision_summary_md": str(batch_decision_out / "batch_decision_summary.md"),
        "batch_decision_summary_cards_csv": str(batch_decision_out / "batch_decision_summary_cards.csv"),
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

    smoke_notes = [
        "Synthetic pose_features.csv is regenerated deterministically from the provided seed.",
        "Calibration is executed on the synthetic labels and can optionally reuse a previous recommended strategy.",
        "Geometry proxy audit checks mouth/path/pocket consistency without changing scores.",
        "Validation evidence audit checks top candidate label coverage without changing scores.",
    ]
    smoke_summary = {
        "out_dir": str(out_dir),
        "feature_csv": str(pose_feature_csv),
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
        "notes": smoke_notes,
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
            notes=smoke_notes,
        ),
        encoding="utf-8",
    )

    commands.append(
        run_command(
            name="build_batch_decision_summary",
            command=[
                py,
                str(repo_root / "build_batch_decision_summary.py"),
                "--summary_json",
                str(smoke_summary_path),
                "--feature_csv",
                str(pose_feature_csv),
                "--out_dir",
                str(batch_decision_out),
                "--top_n",
                "5",
            ],
            cwd=repo_root,
        )
    )
    batch_decision_paths = [
        batch_decision_out / "batch_decision_summary.json",
        batch_decision_out / "batch_decision_summary.md",
        batch_decision_out / "batch_decision_summary_cards.csv",
    ]
    ensure_files_exist(batch_decision_paths)
    batch_decision_payload = json.loads(batch_decision_paths[0].read_text(encoding="utf-8"))
    smoke_notes.append("Generated batch decision summary with run-level recommendation, candidate highlights and risk top-N.")
    smoke_summary["notes"] = smoke_notes
    smoke_summary["commands"] = serialize_command_results(commands)
    smoke_summary["artifacts"] = artifacts
    smoke_summary["batch_decision_summary"] = {
        "quality_gate": batch_decision_payload.get("quality_gate"),
        "batch_decision": batch_decision_payload.get("batch_decision"),
        "validation_evidence": batch_decision_payload.get("validation_evidence"),
        "candidate_highlights": batch_decision_payload.get("candidate_highlights"),
    }
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
            notes=smoke_notes,
        ),
        encoding="utf-8",
    )

    print(f"Smoke test completed. Summary: {smoke_summary_path}")
    print(f"Report: {report_path}")
    print(f"Validation evidence audit: {validation_evidence_out / 'validation_evidence_report.md'}")
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
