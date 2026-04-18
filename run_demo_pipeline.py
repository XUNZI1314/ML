from __future__ import annotations

import argparse
from pathlib import Path

from demo_data_utils import (
    build_demo_experiment_override,
    make_synthetic_pose_features,
    write_demo_manifest,
)
from demo_report_utils import write_demo_interpretation, write_demo_overview_html, write_demo_readme
from real_data_starter_utils import write_real_data_starter_kit
from run_recommended_pipeline import run_recommended_pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a deterministic demo dataset and run the recommended ML pipeline."
    )
    parser.add_argument("--demo_dir", default="demo_data", help="Directory for generated demo inputs.")
    parser.add_argument("--out_dir", default="demo_outputs", help="Directory for pipeline outputs.")
    parser.add_argument("--seed", type=int, default=20260418)
    parser.add_argument("--n_nanobodies", type=int, default=12)
    parser.add_argument("--n_conformers", type=int, default=3)
    parser.add_argument("--n_poses", type=int, default=6)
    parser.add_argument("--train_epochs", type=int, default=3)
    parser.add_argument("--n_feature_trials", type=int, default=6)
    parser.add_argument("--top_feature_trials_for_agg", type=int, default=2)
    parser.add_argument(
        "--skip_pipeline",
        action="store_true",
        help="Only generate demo inputs; do not run the pipeline.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    demo_dir = Path(args.demo_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    demo_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_csv = demo_dir / "demo_pose_features.csv"
    override_csv = demo_dir / "demo_experiment_plan_override.csv"
    manifest_json = demo_dir / "demo_manifest.json"
    demo_readme = out_dir / "DEMO_README.md"
    demo_interpretation = out_dir / "DEMO_INTERPRETATION.md"
    demo_overview = out_dir / "DEMO_OVERVIEW.html"
    real_data_starter_dir = out_dir / "REAL_DATA_STARTER"

    feature_summary = make_synthetic_pose_features(
        feature_csv,
        n_nanobodies=int(args.n_nanobodies),
        n_conformers=int(args.n_conformers),
        n_poses=int(args.n_poses),
        seed=int(args.seed),
    )
    override_summary = build_demo_experiment_override(
        feature_csv=feature_csv,
        out_csv=override_csv,
    )

    pipeline_summary: dict[str, Any] | None = None
    if not bool(args.skip_pipeline):
        pipeline_summary = run_recommended_pipeline(
            feature_csv=feature_csv,
            out_dir=out_dir,
            label_col="label",
            train_epochs=int(args.train_epochs),
            train_early_stopping_patience=2,
            n_feature_trials=int(args.n_feature_trials),
            top_feature_trials_for_agg=int(args.top_feature_trials_for_agg),
            experiment_plan_override_csv=override_csv,
            experiment_plan_budget=int(args.n_nanobodies),
            experiment_plan_validate_budget=int(args.n_nanobodies),
            experiment_plan_review_budget=int(args.n_nanobodies),
            experiment_plan_standby_budget=0,
            experiment_plan_group_quota=int(args.n_nanobodies),
            disable_auto_seed_from_previous_strategy=True,
            seed=int(args.seed),
        )

    write_demo_manifest(
        out_json=manifest_json,
        feature_summary=feature_summary,
        override_summary=override_summary,
        pipeline_summary_json=out_dir / "recommended_pipeline_summary.json"
        if pipeline_summary is not None
        else None,
    )
    starter_manifest = write_real_data_starter_kit(real_data_starter_dir)
    write_demo_interpretation(
        out_path=demo_interpretation,
        feature_csv=feature_csv,
        override_csv=override_csv,
        manifest_json=manifest_json,
        real_data_starter_dir=real_data_starter_dir,
        summary=pipeline_summary,
    )
    write_demo_overview_html(
        out_path=demo_overview,
        feature_csv=feature_csv,
        override_csv=override_csv,
        manifest_json=manifest_json,
        readme_md=demo_readme,
        interpretation_md=demo_interpretation,
        real_data_starter_dir=real_data_starter_dir,
        summary=pipeline_summary,
    )
    write_demo_readme(
        out_path=demo_readme,
        feature_csv=feature_csv,
        override_csv=override_csv,
        manifest_json=manifest_json,
        overview_html=demo_overview,
        interpretation_md=demo_interpretation,
        real_data_starter_dir=real_data_starter_dir,
        summary=pipeline_summary,
    )

    print(f"Demo feature CSV: {feature_csv}")
    print(f"Demo override CSV: {override_csv}")
    print(f"Demo manifest: {manifest_json}")
    print(f"Demo guide: {demo_readme}")
    print(f"Demo interpretation: {demo_interpretation}")
    print(f"Demo overview: {demo_overview}")
    print(f"Real data starter kit: {starter_manifest.get('outputs', {}).get('readme_md')}")
    print(f"Mini PDB example: {starter_manifest.get('outputs', {}).get('mini_pdb_example_readme_md')}")
    if pipeline_summary is not None:
        artifacts = pipeline_summary.get("artifacts", {}) if isinstance(pipeline_summary.get("artifacts"), dict) else {}
        print(f"Batch decision summary: {artifacts.get('batch_decision_summary_md')}")
        print(f"Candidate report cards: {artifacts.get('candidate_report_index_html')}")


if __name__ == "__main__":
    main()
