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
    return parser


def _append_optional_arg(command: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    text = str(value).strip()
    if not text:
        return
    command.extend([flag, text])


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
    repo_root: str | Path | None = None,
    python_executable: str | None = None,
) -> dict[str, Any]:
    if (input_csv is None) == (feature_csv is None):
        raise ValueError("Exactly one of input_csv or feature_csv must be provided.")

    repo_root = Path(__file__).resolve().parent if repo_root is None else Path(repo_root).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_out = out_dir / "pose_features.csv"
    feature_qc_out = out_dir / "feature_qc.json"
    rule_out = out_dir / "rule_outputs"
    model_out = out_dir / "model_outputs"
    ml_rank_out = out_dir / "ml_ranking_outputs"
    compare_out = out_dir / "comparison_rule_vs_ml"
    calibration_out = out_dir / "calibration_outputs"
    compare_calib_out = out_dir / "comparison_calibrated_rule_vs_ml"
    improvement_out = out_dir / "improvement_summary"
    strategy_out = out_dir / "strategy_optimization"

    py = str(python_executable or sys.executable)
    commands: list[CommandResult] = []
    summary_notes: list[str] = []

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
            ],
            cwd=repo_root,
        )
    )

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
        "start_mode": "input_csv" if input_csv is not None else "feature_csv",
        "input_csv": str(Path(input_csv).expanduser().resolve()) if input_csv is not None else None,
        "feature_csv": str(feature_csv),
        "out_dir": str(out_dir),
        "label_col": str(label_col),
        "label_valid_count": int(label_valid_count),
        "label_class_count": int(label_class_count),
        "label_compare_possible": bool(label_compare_possible),
        "calibration_possible": bool(calibration_possible),
        "strategy_seed": strategy_seed_applied,
        "commands": serialize_command_results(commands),
        "notes": summary_notes,
        "artifacts": {
            "feature_qc_json": str(feature_qc_out) if feature_qc_out.exists() else None,
            "rule_ranking_csv": str(rule_out / "nanobody_rule_ranking.csv"),
            "ml_ranking_csv": str(ml_rank_out / "nanobody_ranking.csv"),
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

    print(f"Pipeline completed. Summary: {summary_path}")
    print(f"Report: {report_path}")
    print(f"Rule ranking: {rule_out / 'nanobody_rule_ranking.csv'}")
    print(f"ML ranking: {ml_rank_out / 'nanobody_ranking.csv'}")
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
    )


if __name__ == "__main__":
    main()
