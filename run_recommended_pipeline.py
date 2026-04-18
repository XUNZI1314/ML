"""Run the recommended next-step pipeline described in not_perfect.md.

This entrypoint supports two starts:
1) from raw pose input CSV -> build_feature_table -> rule/ML -> optional label-aware steps
2) from existing pose_features.csv -> rule/ML -> optional label-aware steps

Label-aware steps are enabled automatically only when enough valid labels exist.
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
from runtime_dependency_utils import (
    check_runtime_dependencies,
    format_missing_dependency_summary,
    get_pipeline_runtime_dependency_specs,
)


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _count_label_classes(feature_df: pd.DataFrame, label_col: str) -> int:
    if label_col not in feature_df.columns:
        return 0
    values = _to_numeric(feature_df[label_col]).to_numpy(dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0
    binary = (finite >= 0.5).astype(np.int32)
    return int(np.unique(binary).size)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the recommended next-step pipeline")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input_csv", default=None, help="Raw pose input CSV to feed build_feature_table.py")
    src.add_argument("--feature_csv", default=None, help="Existing pose_features.csv to start from directly")

    parser.add_argument("--out_dir", default="recommended_pipeline_outputs", help="Output root directory")
    parser.add_argument("--label_col", default="label", help="Label column used for compare/calibration")
    parser.add_argument(
        "--disable_label_aware_steps",
        action="store_true",
        help="Skip compare/calibration/strategy optimization even if labels are available",
    )

    # build_feature_table.py options when --input_csv is used
    parser.add_argument("--atom_contact_threshold", type=float, default=4.5)
    parser.add_argument("--catalytic_contact_threshold", type=float, default=4.5)
    parser.add_argument("--substrate_clash_threshold", type=float, default=2.8)
    parser.add_argument("--mouth_residue_fraction", type=float, default=0.30)
    parser.add_argument("--default_pocket_file", default=None)
    parser.add_argument("--default_catalytic_file", default=None)
    parser.add_argument("--default_ligand_file", default=None)
    parser.add_argument("--default_antigen_chain", default=None)
    parser.add_argument("--default_nanobody_chain", default=None)
    parser.add_argument("--skip_failed_rows", action="store_true")

    # shared ranking/train options
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument(
        "--pocket_overwide_penalty_weight",
        type=float,
        default=0.0,
        help="Optional penalty weight for broad pocket definitions; default 0 keeps ranking unchanged",
    )
    parser.add_argument(
        "--pocket_overwide_threshold",
        type=float,
        default=0.55,
        help="Threshold for pocket_shape_overwide_proxy before optional penalty starts",
    )
    parser.add_argument("--train_epochs", type=int, default=20)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--train_val_ratio", type=float, default=0.25)
    parser.add_argument("--train_early_stopping_patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    # calibration/strategy options
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
    parser.add_argument("--disable_calibration_baseline_guard", action="store_true")
    parser.add_argument("--calibration_rank_guard_tolerance", type=float, default=0.005)
    parser.add_argument("--calibration_auc_guard_tolerance", type=float, default=0.0)
    parser.add_argument(
        "--strategy_seed_json",
        default=None,
        help="Optional recommended_strategy.json used to seed calibration parameters",
    )
    parser.add_argument(
        "--disable_auto_seed_from_previous_strategy",
        action="store_true",
        help="Disable auto-loading <out_dir>/strategy_optimization/recommended_strategy.json",
    )
    parser.add_argument(
        "--enable_ai_assistant",
        action="store_true",
        help="Generate optional AI-readable run summaries after pipeline artifacts are written",
    )
    parser.add_argument(
        "--ai_provider",
        choices=["none", "openai"],
        default="none",
        help="AI provider used by ai_assistant.py; default none keeps the step local/offline",
    )
    parser.add_argument(
        "--ai_model",
        default=None,
        help="Optional OpenAI model name when --ai_provider openai is used",
    )
    parser.add_argument(
        "--ai_max_rows",
        type=int,
        default=8,
        help="Maximum rows from ranking/comparison CSVs included in AI summary context",
    )
    parser.add_argument(
        "--allow_ai_sensitive_paths",
        action="store_true",
        help="Allow absolute/local paths in AI context; by default paths are redacted",
    )
    parser.add_argument(
        "--suggestion_diversity_mode",
        choices=["auto", "profile", "column", "none"],
        default="auto",
        help="Diversity mode passed to suggest_next_experiments.py",
    )
    parser.add_argument(
        "--suggestion_diversity_group_col",
        default="auto",
        help="Diversity group column for suggest_next_experiments.py",
    )
    parser.add_argument(
        "--suggestion_diversity_penalty",
        type=float,
        default=0.08,
        help="Soft penalty for repeated diversity groups in experiment suggestions",
    )
    parser.add_argument(
        "--suggestion_diversity_max_per_group",
        type=int,
        default=2,
        help="Soft target before repeated diversity groups are further penalized",
    )
    parser.add_argument("--experiment_plan_budget", type=int, default=8, help="Max include_now rows in experiment_plan.csv")
    parser.add_argument("--experiment_plan_validate_budget", type=int, default=4, help="Max validate_now rows in experiment_plan.csv")
    parser.add_argument("--experiment_plan_review_budget", type=int, default=3, help="Max review_first rows in experiment_plan.csv")
    parser.add_argument("--experiment_plan_standby_budget", type=int, default=3, help="Max standby rows in experiment_plan.csv")
    parser.add_argument("--experiment_plan_group_quota", type=int, default=2, help="Max include_now rows per diversity group")
    parser.add_argument(
        "--experiment_plan_override_csv",
        default=None,
        help=(
            "Optional CSV keyed by nanobody_id to lock/exclude/defer candidates and carry "
            "experiment cost/owner/status/note fields into experiment_plan.csv"
        ),
    )
    parser.add_argument(
        "--disable_provenance_card",
        action="store_true",
        help="Skip run_provenance_card.json/md generation",
    )
    parser.add_argument(
        "--provenance_hash_max_mb",
        type=float,
        default=100.0,
        help="Maximum file size to SHA256 hash in provenance manifest",
    )
    return parser


def _append_optional_arg(command: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    text = str(value).strip()
    if not text:
        return
    command.extend([flag, text])


def _ensure_runtime_dependencies(*, start_mode: str, python_executable: str) -> dict[str, Any]:
    dependency_report = check_runtime_dependencies(
        get_pipeline_runtime_dependency_specs(start_mode),
        python_executable=python_executable,
    )
    if dependency_report.get("error_message"):
        raise RuntimeError(f"Runtime dependency precheck failed: {dependency_report['error_message']}")

    missing = (
        dependency_report.get("missing_dependencies")
        if isinstance(dependency_report.get("missing_dependencies"), list)
        else []
    )
    if missing:
        summary_text = format_missing_dependency_summary(missing)
        raise RuntimeError(
            "Runtime dependency precheck failed: missing Python packages -> "
            f"{summary_text}. Install requirements first, then retry."
        )
    return dependency_report


def run_recommended_pipeline(
    *,
    input_csv: str | Path | None = None,
    feature_csv: str | Path | None = None,
    out_dir: str | Path = "recommended_pipeline_outputs",
    label_col: str = "label",
    disable_label_aware_steps: bool = False,
    atom_contact_threshold: float = 4.5,
    catalytic_contact_threshold: float = 4.5,
    substrate_clash_threshold: float = 2.8,
    mouth_residue_fraction: float = 0.30,
    default_pocket_file: str | Path | None = None,
    default_catalytic_file: str | Path | None = None,
    default_ligand_file: str | Path | None = None,
    default_antigen_chain: str | None = None,
    default_nanobody_chain: str | None = None,
    skip_failed_rows: bool = False,
    top_k: int = 3,
    pocket_overwide_penalty_weight: float = 0.0,
    pocket_overwide_threshold: float = 0.55,
    train_epochs: int = 20,
    train_batch_size: int = 64,
    train_val_ratio: float = 0.25,
    train_early_stopping_patience: int = 8,
    seed: int = 42,
    n_feature_trials: int = 24,
    top_feature_trials_for_agg: int = 6,
    calibration_rank_consistency_weight: float = 0.40,
    calibration_rank_consistency_metric: str = "score_spearman",
    calibration_selection_metric: str = "objective",
    disable_calibration_baseline_guard: bool = False,
    calibration_rank_guard_tolerance: float = 0.005,
    calibration_auc_guard_tolerance: float = 0.0,
    strategy_seed_json: str | Path | None = None,
    disable_auto_seed_from_previous_strategy: bool = False,
    enable_ai_assistant: bool = False,
    ai_provider: str = "none",
    ai_model: str | None = None,
    ai_max_rows: int = 8,
    allow_ai_sensitive_paths: bool = False,
    suggestion_diversity_mode: str = "auto",
    suggestion_diversity_group_col: str = "auto",
    suggestion_diversity_penalty: float = 0.08,
    suggestion_diversity_max_per_group: int = 2,
    experiment_plan_budget: int = 8,
    experiment_plan_validate_budget: int = 4,
    experiment_plan_review_budget: int = 3,
    experiment_plan_standby_budget: int = 3,
    experiment_plan_group_quota: int = 2,
    experiment_plan_override_csv: str | Path | None = None,
    disable_provenance_card: bool = False,
    provenance_hash_max_mb: float = 100.0,
    repo_root: str | Path | None = None,
    python_executable: str | None = None,
) -> dict[str, Any]:
    if (input_csv is None) == (feature_csv is None):
        raise ValueError("Exactly one of input_csv or feature_csv must be provided.")

    start_mode = "input_csv" if input_csv is not None else "feature_csv"
    repo_root = Path(__file__).resolve().parent if repo_root is None else Path(repo_root).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_out = out_dir / "pose_features.csv"
    feature_qc_out = out_dir / "feature_qc.json"
    quality_gate_out = out_dir / "quality_gate"
    geometry_audit_out = out_dir / "geometry_proxy_audit"
    rule_out = out_dir / "rule_outputs"
    model_out = out_dir / "model_outputs"
    ml_rank_out = out_dir / "ml_ranking_outputs"
    consensus_out = out_dir / "consensus_outputs"
    sensitivity_out = out_dir / "parameter_sensitivity"
    score_explanation_out = out_dir / "score_explanation_cards"
    candidate_report_out = out_dir / "candidate_report_cards"
    candidate_report_zip = out_dir / "candidate_report_cards.zip"
    candidate_compare_out = out_dir / "candidate_comparisons"
    experiment_suggestion_out = out_dir / "experiment_suggestions"
    validation_evidence_out = out_dir / "validation_evidence_audit"
    batch_decision_out = out_dir / "batch_decision_summary"
    ai_out = out_dir / "ai_outputs"
    provenance_out = out_dir / "provenance"
    compare_out = out_dir / "comparison_rule_vs_ml"
    calibration_out = out_dir / "calibration_outputs"
    compare_calib_out = out_dir / "comparison_calibrated_rule_vs_ml"
    improvement_out = out_dir / "improvement_summary"
    strategy_out = out_dir / "strategy_optimization"

    py = str(python_executable or sys.executable)
    dependency_report = _ensure_runtime_dependencies(start_mode=start_mode, python_executable=py)
    commands: list[CommandResult] = []
    summary_notes: list[str] = [
        "Runtime dependency precheck passed for the current start mode."
    ]

    if input_csv is not None:
        input_csv = Path(input_csv).expanduser().resolve()
        if not input_csv.exists():
            raise FileNotFoundError(f"input_csv not found: {input_csv}")

        build_cmd = [
            py,
            str(repo_root / "build_feature_table.py"),
            "--input_csv",
            str(input_csv),
            "--out_csv",
            str(feature_out),
            "--qc_json",
            str(feature_qc_out),
            "--atom_contact_threshold",
            str(float(atom_contact_threshold)),
            "--catalytic_contact_threshold",
            str(float(catalytic_contact_threshold)),
            "--substrate_clash_threshold",
            str(float(substrate_clash_threshold)),
            "--mouth_residue_fraction",
            str(float(mouth_residue_fraction)),
        ]
        _append_optional_arg(build_cmd, "--default_pocket_file", default_pocket_file)
        _append_optional_arg(build_cmd, "--default_catalytic_file", default_catalytic_file)
        _append_optional_arg(build_cmd, "--default_ligand_file", default_ligand_file)
        _append_optional_arg(build_cmd, "--default_antigen_chain", default_antigen_chain)
        _append_optional_arg(build_cmd, "--default_nanobody_chain", default_nanobody_chain)
        if bool(skip_failed_rows):
            build_cmd.append("--skip_failed_rows")

        commands.append(run_command("build_feature_table", build_cmd, cwd=repo_root))
        feature_csv = feature_out
    else:
        feature_csv = Path(str(feature_csv)).expanduser().resolve()
        if not feature_csv.exists():
            raise FileNotFoundError(f"feature_csv not found: {feature_csv}")
        summary_notes.append("Started from existing pose_features.csv; build_feature_table step skipped.")

    quality_gate_cmd = [
        py,
        str(repo_root / "build_quality_gate.py"),
        "--feature_csv",
        str(feature_csv),
        "--out_dir",
        str(quality_gate_out),
        "--label_col",
        str(label_col),
        "--pocket_overwide_threshold",
        str(float(pocket_overwide_threshold)),
    ]
    if input_csv is not None and feature_qc_out.exists():
        quality_gate_cmd.extend(["--feature_qc_json", str(feature_qc_out)])
    commands.append(run_command("build_quality_gate", quality_gate_cmd, cwd=repo_root))
    summary_notes.append("Generated PASS/WARN/FAIL quality gate from feature QC and label coverage.")

    commands.append(
        run_command(
            "build_geometry_proxy_audit",
            [
                py,
                str(repo_root / "build_geometry_proxy_audit.py"),
                "--feature_csv",
                str(feature_csv),
                "--out_dir",
                str(geometry_audit_out),
                "--overwide_threshold",
                str(float(pocket_overwide_threshold)),
            ],
            cwd=repo_root,
        )
    )
    summary_notes.append("Generated geometry proxy audit for mouth/path/pocket consistency without changing scores.")

    commands.append(
        run_command(
            "rule_ranker",
            [
                py,
                str(repo_root / "rule_ranker.py"),
                "--feature_csv",
                str(feature_csv),
                "--out_dir",
                str(rule_out),
                "--top_k",
                str(int(top_k)),
                "--pocket_overwide_penalty_weight",
                str(float(pocket_overwide_penalty_weight)),
                "--pocket_overwide_threshold",
                str(float(pocket_overwide_threshold)),
            ],
            cwd=repo_root,
        )
    )

    commands.append(
        run_command(
            "train_pose_model",
            [
                py,
                str(repo_root / "train_pose_model.py"),
                "--feature_csv",
                str(feature_csv),
                "--out_dir",
                str(model_out),
                "--epochs",
                str(int(train_epochs)),
                "--batch_size",
                str(int(train_batch_size)),
                "--val_ratio",
                str(float(train_val_ratio)),
                "--early_stopping_patience",
                str(int(train_early_stopping_patience)),
                "--seed",
                str(int(seed)),
            ],
            cwd=repo_root,
        )
    )

    commands.append(
        run_command(
            "rank_nanobodies",
            [
                py,
                str(repo_root / "rank_nanobodies.py"),
                "--pred_csv",
                str(model_out / "pose_predictions.csv"),
                "--out_dir",
                str(ml_rank_out),
                "--top_k",
                str(int(top_k)),
                "--pocket_overwide_penalty_weight",
                str(float(pocket_overwide_penalty_weight)),
                "--pocket_overwide_threshold",
                str(float(pocket_overwide_threshold)),
            ],
            cwd=repo_root,
        )
    )

    commands.append(
        run_command(
            "build_consensus_ranking",
            [
                py,
                str(repo_root / "build_consensus_ranking.py"),
                "--rule_csv",
                str(rule_out / "nanobody_rule_ranking.csv"),
                "--ml_csv",
                str(ml_rank_out / "nanobody_ranking.csv"),
                "--feature_csv",
                str(feature_csv),
                "--out_dir",
                str(consensus_out),
                "--overwide_threshold",
                str(float(pocket_overwide_threshold)),
            ],
            cwd=repo_root,
        )
    )
    summary_notes.append("Generated Rule + ML consensus ranking for decision support.")

    commands.append(
        run_command(
            "build_score_explanation_cards",
            [
                py,
                str(repo_root / "build_score_explanation_cards.py"),
                "--consensus_csv",
                str(consensus_out / "consensus_ranking.csv"),
                "--feature_csv",
                str(feature_csv),
                "--label_col",
                str(label_col),
                "--out_dir",
                str(score_explanation_out),
            ],
            cwd=repo_root,
        )
    )
    summary_notes.append("Generated product-facing score explanation cards without changing scores.")

    commands.append(
        run_command(
            "analyze_ranking_parameter_sensitivity",
            [
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
    summary_notes.append("Generated lightweight ranking parameter sensitivity analysis.")

    commands.append(
        run_command(
            "build_candidate_comparisons",
            [
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
    summary_notes.append("Generated pairwise and group-level candidate comparison explanations.")

    commands.append(
        run_command(
            "build_candidate_report_cards",
            [
                py,
                str(repo_root / "build_candidate_report_cards.py"),
                "--consensus_csv",
                str(consensus_out / "consensus_ranking.csv"),
                "--rule_csv",
                str(rule_out / "nanobody_rule_ranking.csv"),
                "--ml_csv",
                str(ml_rank_out / "nanobody_ranking.csv"),
                "--feature_csv",
                str(feature_csv),
                "--pose_predictions_csv",
                str(model_out / "pose_predictions.csv"),
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
    summary_notes.append("Generated per-nanobody candidate report cards with embedded comparison context.")

    commands.append(
        run_command(
            "suggest_next_experiments",
            [
                py,
                str(repo_root / "suggest_next_experiments.py"),
                "--consensus_csv",
                str(consensus_out / "consensus_ranking.csv"),
                "--out_dir",
                str(experiment_suggestion_out),
                "--diversity_mode",
                str(suggestion_diversity_mode),
                "--diversity_group_col",
                str(suggestion_diversity_group_col),
                "--diversity_penalty",
                str(float(suggestion_diversity_penalty)),
                "--diversity_max_per_group",
                str(int(suggestion_diversity_max_per_group)),
                "--experiment_plan_budget",
                str(int(experiment_plan_budget)),
                "--experiment_plan_validate_budget",
                str(int(experiment_plan_validate_budget)),
                "--experiment_plan_review_budget",
                str(int(experiment_plan_review_budget)),
                "--experiment_plan_standby_budget",
                str(int(experiment_plan_standby_budget)),
                "--experiment_plan_group_quota",
                str(int(experiment_plan_group_quota)),
            ]
            + (
                ["--experiment_plan_override_csv", str(experiment_plan_override_csv)]
                if experiment_plan_override_csv is not None and str(experiment_plan_override_csv).strip()
                else []
            ),
            cwd=repo_root,
        )
    )
    summary_notes.append("Generated active-learning style next experiment suggestions.")

    commands.append(
        run_command(
            "build_validation_evidence_audit",
            [
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
    summary_notes.append("Generated validation evidence audit for top candidates without changing scores.")

    feature_df = pd.read_csv(feature_csv, low_memory=False)
    label_valid_count = int(_to_numeric(feature_df[label_col]).notna().sum()) if label_col in feature_df.columns else 0
    label_class_count = _count_label_classes(feature_df, str(label_col))
    label_compare_possible = label_valid_count > 0 and label_class_count >= 2
    calibration_possible = label_valid_count >= 8 and label_class_count >= 2

    strategy_seed_applied: dict[str, Any] = {
        "enabled": False,
        "source_json": None,
        "applied_rank_consistency_weight": None,
        "applied_selection_metric": None,
        "applied_min_rank_consistency": None,
        "applied_min_nanobody_auc": None,
    }

    baseline_summary: dict[str, Any] | None = None
    calibrated_summary: dict[str, Any] | None = None
    improvement_summary: dict[str, Any] | None = None
    strategy_summary: dict[str, Any] | None = None

    if bool(disable_label_aware_steps):
        summary_notes.append("Label-aware steps disabled by --disable_label_aware_steps.")
    elif not label_compare_possible:
        summary_notes.append(
            "Skipped label-aware steps because valid labels are missing or label classes are degenerate."
        )
    else:
        commands.append(
            run_command(
                "compare_rule_vs_ml",
                [
                    py,
                    str(repo_root / "compare_rule_ml_rankings.py"),
                    "--rule_csv",
                    str(rule_out / "nanobody_rule_ranking.csv"),
                    "--ml_csv",
                    str(ml_rank_out / "nanobody_ranking.csv"),
                    "--pose_feature_csv",
                    str(feature_csv),
                    "--label_col",
                    str(label_col),
                    "--out_dir",
                    str(compare_out),
                ],
                cwd=repo_root,
            )
        )
        baseline_summary = json.loads((compare_out / "ranking_comparison_summary.json").read_text(encoding="utf-8"))

        if not calibration_possible:
            summary_notes.append(
                f"Skipped calibration because valid label rows are insufficient for calibration (need >= 8, got {label_valid_count})."
            )
        else:
            strategy_seed_path = resolve_strategy_seed_path(
                repo_root=repo_root,
                strategy_seed_json=None if strategy_seed_json is None else str(strategy_seed_json),
                auto_seed_path=strategy_out / "recommended_strategy.json",
                disable_auto_seed=bool(disable_auto_seed_from_previous_strategy),
            )
            strategy_seed_payload: dict[str, Any] | None = None
            if strategy_seed_path is not None:
                strategy_seed_payload = load_strategy_seed(strategy_seed_path)
            calibration_strategy = resolve_calibration_strategy(
                default_rank_consistency_weight=float(calibration_rank_consistency_weight),
                default_selection_metric=str(calibration_selection_metric),
                strategy_seed_payload=strategy_seed_payload,
                strategy_seed_path=strategy_seed_path,
                baseline_summary=baseline_summary,
                enable_baseline_guard=not bool(disable_calibration_baseline_guard),
                rank_guard_tolerance=float(calibration_rank_guard_tolerance),
                auc_guard_tolerance=float(calibration_auc_guard_tolerance),
            )
            strategy_seed_applied = dict(calibration_strategy["summary"])

            calibrate_cmd = [
                py,
                str(repo_root / "calibrate_rule_ranker.py"),
                "--feature_csv",
                str(feature_csv),
                "--label_col",
                str(label_col),
                "--out_dir",
                str(calibration_out),
                "--n_feature_trials",
                str(int(n_feature_trials)),
                "--top_feature_trials_for_agg",
                str(int(top_feature_trials_for_agg)),
                "--ml_ranking_csv",
                str(ml_rank_out / "nanobody_ranking.csv"),
                "--ml_score_col",
                "final_score",
                "--rank_consistency_metric",
                str(calibration_rank_consistency_metric),
                "--rank_consistency_weight",
                str(float(calibration_strategy["rank_consistency_weight"])),
                "--selection_metric",
                str(calibration_strategy["selection_metric"]),
                "--pocket_overwide_penalty_weight",
                str(float(pocket_overwide_penalty_weight)),
                "--pocket_overwide_threshold",
                str(float(pocket_overwide_threshold)),
                "--seed",
                str(int(seed)),
            ]
            if calibration_strategy["min_rank_consistency"] is not None:
                calibrate_cmd.extend(
                    ["--min_rank_consistency", f"{float(calibration_strategy['min_rank_consistency']):.12f}"]
                )
            if calibration_strategy["min_nanobody_auc"] is not None:
                calibrate_cmd.extend(
                    ["--min_nanobody_auc", f"{float(calibration_strategy['min_nanobody_auc']):.12f}"]
                )

            commands.append(run_command("calibrate_rule_ranker", calibrate_cmd, cwd=repo_root))

            commands.append(
                run_command(
                    "compare_calibrated_rule_vs_ml",
                    [
                        py,
                        str(repo_root / "compare_rule_ml_rankings.py"),
                        "--rule_csv",
                        str(calibration_out / "calibrated_rule_outputs" / "nanobody_rule_ranking.csv"),
                        "--ml_csv",
                        str(ml_rank_out / "nanobody_ranking.csv"),
                        "--pose_feature_csv",
                        str(feature_csv),
                        "--label_col",
                        str(label_col),
                        "--out_dir",
                        str(compare_calib_out),
                    ],
                    cwd=repo_root,
                )
            )

            commands.append(
                run_command(
                    "summarize_rule_ml_improvement",
                    [
                        py,
                        str(repo_root / "summarize_rule_ml_improvement.py"),
                        "--baseline_summary",
                        str(compare_out / "ranking_comparison_summary.json"),
                        "--calibrated_summary",
                        str(compare_calib_out / "ranking_comparison_summary.json"),
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
                    "optimize_calibration_strategy",
                    [
                        py,
                        str(repo_root / "optimize_calibration_strategy.py"),
                        "--aggregation_trials_csv",
                        str(calibration_out / "aggregation_calibration_trials.csv"),
                        "--baseline_summary_json",
                        str(compare_out / "ranking_comparison_summary.json"),
                        "--use_baseline_guard",
                        "--rank_guard_tolerance",
                        str(float(calibration_rank_guard_tolerance)),
                        "--auc_guard_tolerance",
                        str(float(calibration_auc_guard_tolerance)),
                        "--out_dir",
                        str(strategy_out),
                    ],
                    cwd=repo_root,
                )
            )

            calibrated_summary = json.loads((compare_calib_out / "ranking_comparison_summary.json").read_text(encoding="utf-8"))
            improvement_summary = json.loads((improvement_out / "calibration_improvement_summary.json").read_text(encoding="utf-8"))
            strategy_summary = json.loads((strategy_out / "recommended_strategy.json").read_text(encoding="utf-8"))

    required_files = [
        rule_out / "pose_rule_scores.csv",
        rule_out / "conformer_rule_scores.csv",
        rule_out / "nanobody_rule_ranking.csv",
        model_out / "pose_predictions.csv",
        model_out / "training_summary.json",
        ml_rank_out / "conformer_scores.csv",
        ml_rank_out / "nanobody_ranking.csv",
        consensus_out / "consensus_ranking.csv",
        consensus_out / "consensus_summary.json",
        consensus_out / "consensus_report.md",
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
        candidate_compare_out / "candidate_group_comparison_summary.csv",
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
        quality_gate_out / "quality_gate_summary.json",
        quality_gate_out / "quality_gate_checks.csv",
        quality_gate_out / "quality_gate_report.md",
        geometry_audit_out / "geometry_proxy_audit_summary.json",
        geometry_audit_out / "geometry_proxy_audit_report.md",
        geometry_audit_out / "geometry_proxy_feature_summary.csv",
        geometry_audit_out / "geometry_proxy_flagged_poses.csv",
        geometry_audit_out / "geometry_proxy_candidate_audit.csv",
    ]
    if input_csv is not None:
        required_files.extend([feature_out, feature_qc_out])
    if baseline_summary is not None:
        required_files.extend(
            [
                compare_out / "ranking_comparison_table.csv",
                compare_out / "ranking_comparison_summary.json",
                compare_out / "ranking_comparison_report.md",
            ]
        )
    if calibrated_summary is not None:
        required_files.extend(
            [
                calibration_out / "feature_calibration_trials.csv",
                calibration_out / "aggregation_calibration_trials.csv",
                calibration_out / "calibrated_rule_config.json",
                calibration_out / "calibrated_rule_outputs" / "pose_rule_scores.csv",
                calibration_out / "calibrated_rule_outputs" / "conformer_rule_scores.csv",
                calibration_out / "calibrated_rule_outputs" / "nanobody_rule_ranking.csv",
                compare_calib_out / "ranking_comparison_table.csv",
                compare_calib_out / "ranking_comparison_summary.json",
                compare_calib_out / "ranking_comparison_report.md",
                improvement_out / "calibration_improvement_metrics.csv",
                improvement_out / "calibration_improvement_summary.json",
                improvement_out / "calibration_improvement_report.md",
                strategy_out / "strategy_sweep_results.csv",
                strategy_out / "recommended_strategy.json",
                strategy_out / "recommended_strategy_report.md",
            ]
        )
    ensure_files_exist(required_files)

    report_path = out_dir / "recommended_pipeline_report.md"
    summary = {
        "start_mode": start_mode,
        "input_csv": str(Path(input_csv).expanduser().resolve()) if input_csv is not None else None,
        "feature_csv": str(feature_csv),
        "out_dir": str(out_dir),
        "runtime_dependencies": dependency_report,
        "label_col": str(label_col),
        "label_valid_count": int(label_valid_count),
        "label_class_count": int(label_class_count),
        "label_compare_possible": bool(label_compare_possible),
        "calibration_possible": bool(calibration_possible),
        "strategy_seed": strategy_seed_applied,
        "ranking_config": {
            "top_k": int(top_k),
            "pocket_overwide_penalty_weight": float(pocket_overwide_penalty_weight),
            "pocket_overwide_threshold": float(pocket_overwide_threshold),
        },
        "suggestion_config": {
            "diversity_mode": str(suggestion_diversity_mode),
            "diversity_group_col": str(suggestion_diversity_group_col),
            "diversity_penalty": float(suggestion_diversity_penalty),
            "diversity_max_per_group": int(suggestion_diversity_max_per_group),
        },
        "experiment_plan_config": {
            "experiment_budget": int(experiment_plan_budget),
            "validate_budget": int(experiment_plan_validate_budget),
            "review_budget": int(experiment_plan_review_budget),
            "standby_budget": int(experiment_plan_standby_budget),
            "group_quota": int(experiment_plan_group_quota),
            "override_csv": str(Path(experiment_plan_override_csv).expanduser().resolve())
            if experiment_plan_override_csv is not None and str(experiment_plan_override_csv).strip()
            else None,
        },
        "input_file_defaults": {
            "default_pocket_file": str(default_pocket_file) if default_pocket_file is not None else None,
            "default_catalytic_file": str(default_catalytic_file) if default_catalytic_file is not None else None,
            "default_ligand_file": str(default_ligand_file) if default_ligand_file is not None else None,
        },
        "commands": serialize_command_results(commands),
        "notes": summary_notes,
        "artifacts": {
            "feature_qc_json": str(feature_qc_out) if feature_qc_out.exists() else None,
            "quality_gate_summary_json": str(quality_gate_out / "quality_gate_summary.json"),
            "quality_gate_checks_csv": str(quality_gate_out / "quality_gate_checks.csv"),
            "quality_gate_report_md": str(quality_gate_out / "quality_gate_report.md"),
            "geometry_proxy_audit_summary_json": str(geometry_audit_out / "geometry_proxy_audit_summary.json"),
            "geometry_proxy_audit_report_md": str(geometry_audit_out / "geometry_proxy_audit_report.md"),
            "geometry_proxy_feature_summary_csv": str(geometry_audit_out / "geometry_proxy_feature_summary.csv"),
            "geometry_proxy_flagged_poses_csv": str(geometry_audit_out / "geometry_proxy_flagged_poses.csv"),
            "geometry_proxy_candidate_audit_csv": str(geometry_audit_out / "geometry_proxy_candidate_audit.csv"),
            "rule_ranking_csv": str(rule_out / "nanobody_rule_ranking.csv"),
            "ml_ranking_csv": str(ml_rank_out / "nanobody_ranking.csv"),
            "consensus_ranking_csv": str(consensus_out / "consensus_ranking.csv"),
            "consensus_summary_json": str(consensus_out / "consensus_summary.json"),
            "consensus_report_md": str(consensus_out / "consensus_report.md"),
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
            "candidate_group_comparison_summary_csv": str(candidate_compare_out / "candidate_group_comparison_summary.csv"),
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
            "baseline_comparison_summary_json": str(compare_out / "ranking_comparison_summary.json") if baseline_summary is not None else None,
            "baseline_comparison_report_md": str(compare_out / "ranking_comparison_report.md") if baseline_summary is not None else None,
            "calibrated_rule_config_json": str(calibration_out / "calibrated_rule_config.json") if calibrated_summary is not None else None,
            "calibrated_comparison_summary_json": str(compare_calib_out / "ranking_comparison_summary.json") if calibrated_summary is not None else None,
            "calibrated_comparison_report_md": str(compare_calib_out / "ranking_comparison_report.md") if calibrated_summary is not None else None,
            "improvement_summary_json": str(improvement_out / "calibration_improvement_summary.json") if improvement_summary is not None else None,
            "improvement_report_md": str(improvement_out / "calibration_improvement_report.md") if improvement_summary is not None else None,
            "strategy_recommended_json": str(strategy_out / "recommended_strategy.json") if strategy_summary is not None else None,
            "strategy_report_md": str(strategy_out / "recommended_strategy_report.md") if strategy_summary is not None else None,
            "execution_report_md": str(report_path),
        },
        "comparison": {
            "baseline_rule_vs_ml": baseline_summary,
            "calibrated_rule_vs_ml": calibrated_summary,
        },
        "improvement": improvement_summary,
        "strategy_optimization": strategy_summary,
    }

    summary_path = out_dir / "recommended_pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    report_path.write_text(
        build_execution_report(
            title="Recommended Pipeline Execution Report",
            start_mode=summary["start_mode"],
            input_csv=summary["input_csv"],
            feature_csv=summary["feature_csv"],
            label_summary={
                "label_valid_count": summary["label_valid_count"],
                "label_class_count": summary["label_class_count"],
                "label_compare_possible": summary["label_compare_possible"],
                "calibration_possible": summary["calibration_possible"],
            },
            strategy_seed_summary=strategy_seed_applied,
            baseline_summary=baseline_summary,
            calibrated_summary=calibrated_summary,
            improvement_summary=improvement_summary,
            strategy_summary=strategy_summary,
            artifacts=summary["artifacts"],
            commands=commands,
            notes=summary_notes,
        ),
        encoding="utf-8",
    )

    batch_decision_cmd = [
        py,
        str(repo_root / "build_batch_decision_summary.py"),
        "--summary_json",
        str(summary_path),
        "--out_dir",
        str(batch_decision_out),
        "--top_n",
        "5",
    ]
    commands.append(run_command("build_batch_decision_summary", batch_decision_cmd, cwd=repo_root))
    batch_decision_summary_path = batch_decision_out / "batch_decision_summary.json"
    batch_decision_artifacts = {
        "batch_decision_summary_json": str(batch_decision_summary_path),
        "batch_decision_summary_md": str(batch_decision_out / "batch_decision_summary.md"),
        "batch_decision_summary_cards_csv": str(batch_decision_out / "batch_decision_summary_cards.csv"),
    }
    ensure_files_exist([Path(path) for path in batch_decision_artifacts.values()])
    batch_decision_payload = json.loads(batch_decision_summary_path.read_text(encoding="utf-8"))
    summary_notes.append("Generated batch decision summary with run-level recommendation, candidate highlights and risk top-N.")
    summary["notes"] = summary_notes
    summary["commands"] = serialize_command_results(commands)
    summary["artifacts"].update(batch_decision_artifacts)
    summary["batch_decision_summary"] = {
        "quality_gate": batch_decision_payload.get("quality_gate"),
        "batch_decision": batch_decision_payload.get("batch_decision"),
        "validation_evidence": batch_decision_payload.get("validation_evidence"),
        "candidate_highlights": batch_decision_payload.get("candidate_highlights"),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    report_path.write_text(
        build_execution_report(
            title="Recommended Pipeline Execution Report",
            start_mode=summary["start_mode"],
            input_csv=summary["input_csv"],
            feature_csv=summary["feature_csv"],
            label_summary={
                "label_valid_count": summary["label_valid_count"],
                "label_class_count": summary["label_class_count"],
                "label_compare_possible": summary["label_compare_possible"],
                "calibration_possible": summary["calibration_possible"],
            },
            strategy_seed_summary=strategy_seed_applied,
            baseline_summary=baseline_summary,
            calibrated_summary=calibrated_summary,
            improvement_summary=improvement_summary,
            strategy_summary=strategy_summary,
            artifacts=summary["artifacts"],
            commands=commands,
            notes=summary_notes,
        ),
        encoding="utf-8",
    )

    if bool(enable_ai_assistant):
        ai_cmd = [
            py,
            str(repo_root / "ai_assistant.py"),
            "--summary_json",
            str(summary_path),
            "--out_dir",
            str(ai_out),
            "--provider",
            str(ai_provider),
            "--max_rows",
            str(int(ai_max_rows)),
        ]
        if ai_model:
            ai_cmd.extend(["--model", str(ai_model)])
        if bool(allow_ai_sensitive_paths):
            ai_cmd.append("--allow_sensitive_paths")

        commands.append(run_command("ai_assistant", ai_cmd, cwd=repo_root))
        ai_summary_path = ai_out / "ai_assistant_summary.json"
        ai_artifacts = {
            "ai_run_summary_md": str(ai_out / "ai_run_summary.md"),
            "ai_top_candidates_explanation_md": str(ai_out / "ai_top_candidates_explanation.md"),
            "ai_failure_diagnosis_md": str(ai_out / "ai_failure_diagnosis.md"),
            "ai_assistant_summary_json": str(ai_summary_path),
        }
        ensure_files_exist([Path(path) for path in ai_artifacts.values()])
        summary_notes.append("Generated optional AI assistant summaries from existing pipeline artifacts.")
        summary["notes"] = summary_notes
        summary["commands"] = serialize_command_results(commands)
        summary["artifacts"].update(ai_artifacts)
        summary["ai_assistant"] = json.loads(ai_summary_path.read_text(encoding="utf-8"))
        summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
        report_path.write_text(
            build_execution_report(
                title="Recommended Pipeline Execution Report",
                start_mode=summary["start_mode"],
                input_csv=summary["input_csv"],
                feature_csv=summary["feature_csv"],
                label_summary={
                    "label_valid_count": summary["label_valid_count"],
                    "label_class_count": summary["label_class_count"],
                    "label_compare_possible": summary["label_compare_possible"],
                    "calibration_possible": summary["calibration_possible"],
                },
                strategy_seed_summary=strategy_seed_applied,
                baseline_summary=baseline_summary,
                calibrated_summary=calibrated_summary,
                improvement_summary=improvement_summary,
                strategy_summary=strategy_summary,
                artifacts=summary["artifacts"],
                commands=commands,
                notes=summary_notes,
            ),
            encoding="utf-8",
        )

    if not bool(disable_provenance_card):
        provenance_cmd = [
            py,
            str(repo_root / "build_run_provenance.py"),
            "--summary_json",
            str(summary_path),
            "--out_dir",
            str(provenance_out),
            "--repo_root",
            str(repo_root),
            "--hash_max_mb",
            str(float(provenance_hash_max_mb)),
        ]
        commands.append(run_command("build_run_provenance", provenance_cmd, cwd=repo_root))
        provenance_artifacts = {
            "run_provenance_card_json": str(provenance_out / "run_provenance_card.json"),
            "run_provenance_card_md": str(provenance_out / "run_provenance_card.md"),
            "run_artifact_manifest_csv": str(provenance_out / "run_artifact_manifest.csv"),
            "run_input_file_manifest_csv": str(provenance_out / "run_input_file_manifest.csv"),
            "run_provenance_integrity_json": str(provenance_out / "run_provenance_integrity.json"),
        }
        ensure_files_exist([Path(path) for path in provenance_artifacts.values()])
        provenance_payload = json.loads(
            (provenance_out / "run_provenance_card.json").read_text(encoding="utf-8")
        )
        provenance_integrity_payload = json.loads(
            (provenance_out / "run_provenance_integrity.json").read_text(encoding="utf-8")
        )
        summary_notes.append("Generated run provenance card with input, artifact, source, dependency and git hashes.")
        summary["notes"] = summary_notes
        summary["commands"] = serialize_command_results(commands)
        summary["artifacts"].update(provenance_artifacts)
        summary["provenance"] = {
            "parameter_hash": provenance_payload.get("parameter_hash"),
            "file_manifest_hash": provenance_payload.get("file_manifest_hash"),
            "input_file_manifest": provenance_payload.get("input_file_manifest"),
            "integrity": {
                "signature_payload_hash": provenance_integrity_payload.get("signature_payload_hash"),
                "signature_type": provenance_integrity_payload.get("signature_type"),
                "signature_json": provenance_artifacts["run_provenance_integrity_json"],
            },
            "git": provenance_payload.get("git"),
            "environment": provenance_payload.get("environment"),
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
        report_path.write_text(
            build_execution_report(
                title="Recommended Pipeline Execution Report",
                start_mode=summary["start_mode"],
                input_csv=summary["input_csv"],
                feature_csv=summary["feature_csv"],
                label_summary={
                    "label_valid_count": summary["label_valid_count"],
                    "label_class_count": summary["label_class_count"],
                    "label_compare_possible": summary["label_compare_possible"],
                    "calibration_possible": summary["calibration_possible"],
                },
                strategy_seed_summary=strategy_seed_applied,
                baseline_summary=baseline_summary,
                calibrated_summary=calibrated_summary,
                improvement_summary=improvement_summary,
                strategy_summary=strategy_summary,
                artifacts=summary["artifacts"],
                commands=commands,
                notes=summary_notes,
            ),
            encoding="utf-8",
        )

    print(f"Pipeline completed. Summary: {summary_path}")
    print(f"Report: {report_path}")
    print(f"Rule ranking: {rule_out / 'nanobody_rule_ranking.csv'}")
    print(f"ML ranking: {ml_rank_out / 'nanobody_ranking.csv'}")
    print(f"Consensus ranking: {consensus_out / 'consensus_ranking.csv'}")
    print(f"Parameter sensitivity: {sensitivity_out / 'parameter_sensitivity_report.md'}")
    print(f"Candidate report cards: {candidate_report_out / 'index.html'}")
    print(f"Candidate comparisons: {candidate_compare_out / 'candidate_comparison_report.md'}")
    print(f"Experiment suggestions: {experiment_suggestion_out / 'next_experiment_suggestions.csv'}")
    print(f"Validation evidence audit: {validation_evidence_out / 'validation_evidence_report.md'}")
    print(f"Batch decision summary: {batch_decision_out / 'batch_decision_summary.md'}")
    if bool(enable_ai_assistant):
        print(f"AI summaries: {ai_out / 'ai_run_summary.md'}")
    if not bool(disable_provenance_card):
        print(f"Provenance card: {provenance_out / 'run_provenance_card.md'}")
    if baseline_summary is not None:
        print(f"Baseline rule-vs-ML spearman(rank): {baseline_summary.get('rank_spearman', float('nan')):.4f}")
    if calibrated_summary is not None:
        print(f"Calibrated rule-vs-ML spearman(rank): {calibrated_summary.get('rank_spearman', float('nan')):.4f}")
    return summary


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    run_recommended_pipeline(
        input_csv=args.input_csv,
        feature_csv=args.feature_csv,
        out_dir=args.out_dir,
        label_col=args.label_col,
        disable_label_aware_steps=bool(args.disable_label_aware_steps),
        atom_contact_threshold=float(args.atom_contact_threshold),
        catalytic_contact_threshold=float(args.catalytic_contact_threshold),
        substrate_clash_threshold=float(args.substrate_clash_threshold),
        mouth_residue_fraction=float(args.mouth_residue_fraction),
        default_pocket_file=args.default_pocket_file,
        default_catalytic_file=args.default_catalytic_file,
        default_ligand_file=args.default_ligand_file,
        default_antigen_chain=args.default_antigen_chain,
        default_nanobody_chain=args.default_nanobody_chain,
        skip_failed_rows=bool(args.skip_failed_rows),
        top_k=int(args.top_k),
        pocket_overwide_penalty_weight=float(args.pocket_overwide_penalty_weight),
        pocket_overwide_threshold=float(args.pocket_overwide_threshold),
        train_epochs=int(args.train_epochs),
        train_batch_size=int(args.train_batch_size),
        train_val_ratio=float(args.train_val_ratio),
        train_early_stopping_patience=int(args.train_early_stopping_patience),
        seed=int(args.seed),
        n_feature_trials=int(args.n_feature_trials),
        top_feature_trials_for_agg=int(args.top_feature_trials_for_agg),
        calibration_rank_consistency_weight=float(args.calibration_rank_consistency_weight),
        calibration_rank_consistency_metric=str(args.calibration_rank_consistency_metric),
        calibration_selection_metric=str(args.calibration_selection_metric),
        disable_calibration_baseline_guard=bool(args.disable_calibration_baseline_guard),
        calibration_rank_guard_tolerance=float(args.calibration_rank_guard_tolerance),
        calibration_auc_guard_tolerance=float(args.calibration_auc_guard_tolerance),
        strategy_seed_json=args.strategy_seed_json,
        disable_auto_seed_from_previous_strategy=bool(args.disable_auto_seed_from_previous_strategy),
        enable_ai_assistant=bool(args.enable_ai_assistant),
        ai_provider=str(args.ai_provider),
        ai_model=args.ai_model,
        ai_max_rows=int(args.ai_max_rows),
        allow_ai_sensitive_paths=bool(args.allow_ai_sensitive_paths),
        suggestion_diversity_mode=str(args.suggestion_diversity_mode),
        suggestion_diversity_group_col=str(args.suggestion_diversity_group_col),
        suggestion_diversity_penalty=float(args.suggestion_diversity_penalty),
        suggestion_diversity_max_per_group=int(args.suggestion_diversity_max_per_group),
        experiment_plan_budget=int(args.experiment_plan_budget),
        experiment_plan_validate_budget=int(args.experiment_plan_validate_budget),
        experiment_plan_review_budget=int(args.experiment_plan_review_budget),
        experiment_plan_standby_budget=int(args.experiment_plan_standby_budget),
        experiment_plan_group_quota=int(args.experiment_plan_group_quota),
        experiment_plan_override_csv=args.experiment_plan_override_csv,
        disable_provenance_card=bool(args.disable_provenance_card),
        provenance_hash_max_mb=float(args.provenance_hash_max_mb),
    )


if __name__ == "__main__":
    main()
