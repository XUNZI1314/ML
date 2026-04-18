from __future__ import annotations

import argparse
import json
import shutil
import stat
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from app_metadata import APP_NAME, APP_RELEASE_CHANNEL, APP_RELEASE_DATE, APP_VERSION


RUNTIME_FILES = [
    "app_metadata.py",
    "assets/app_icon.ico",
    "assets/app_icon.png",
    "ai_assistant.py",
    "analyze_ranking_parameter_sensitivity.py",
    "build_candidate_comparisons.py",
    "build_candidate_report_cards.py",
    "build_consensus_ranking.py",
    "build_experiment_state_ledger.py",
    "build_experiment_validation_report.py",
    "build_feature_table.py",
    "build_result_archive.py",
    "build_run_provenance.py",
    "build_validation_retrain_comparison.py",
    "calibrate_rule_ranker.py",
    "compare_rule_ml_rankings.py",
    "core_utils.py",
    "export_structure_annotations.py",
    "geometry_features.py",
    "local_ml_app.py",
    "ml_desktop_launcher.py",
    "optimize_calibration_strategy.py",
    "pdb_parser.py",
    "pipeline_runner_common.py",
    "pocket_io.py",
    "ranking_common.py",
    "rank_nanobodies.py",
    "rule_ranker.py",
    "run_recommended_pipeline.py",
    "start_local_app.bat",
    "suggest_next_experiments.py",
    "summarize_rule_ml_improvement.py",
    "train_pose_model.py",
    "requirements.txt",
    "README.md",
    "MODEL_QUICKSTART.md",
    "runtime_dependency_utils.py",
]


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", "*.log"),
    )


def _remove_readonly(func: Any, path: str, _: Any) -> None:
    Path(path).chmod(stat.S_IWRITE)
    func(path)


def _safe_rmtree(path: Path, retries: int = 6, wait_sec: float = 1.0) -> None:
    last_error: Exception | None = None
    for _ in range(max(1, int(retries))):
        if not path.exists():
            return
        try:
            shutil.rmtree(path, onexc=_remove_readonly)
            return
        except Exception as exc:
            last_error = exc
            time.sleep(max(0.1, float(wait_sec)))
    if path.exists() and last_error is not None:
        raise last_error


def build_portable_app_dir(
    *,
    repo_root: str | Path,
    app_dir: str | Path,
) -> list[dict[str, Any]]:
    repo_root = Path(repo_root).expanduser().resolve()
    app_dir = Path(app_dir).expanduser().resolve()
    venv_src = repo_root / ".venv"

    if not venv_src.exists():
        raise FileNotFoundError(f".venv not found: {venv_src}")

    if app_dir.exists():
        _safe_rmtree(app_dir)
    app_dir.mkdir(parents=True, exist_ok=True)

    copied_entries: list[dict[str, Any]] = []
    for relative_name in RUNTIME_FILES:
        source = repo_root / relative_name
        if not source.exists():
            raise FileNotFoundError(f"Runtime file not found: {source}")
        destination = app_dir / relative_name
        _copy_file(source, destination)
        copied_entries.append(
            {
                "type": "file",
                "relative_path": f"app/{relative_name}",
                "size_bytes": int(source.stat().st_size),
            }
        )

    _copy_tree(venv_src, app_dir / ".venv")
    venv_size = sum(p.stat().st_size for p in (app_dir / ".venv").rglob("*") if p.is_file())
    copied_entries.append(
        {
            "type": "dir",
            "relative_path": "app/.venv",
            "size_bytes": int(venv_size),
        }
    )

    (app_dir / "local_app_runs").mkdir(parents=True, exist_ok=True)
    return copied_entries


def build_portable_bundle(
    *,
    repo_root: str | Path,
    bundle_dir: str | Path,
    launcher_exe: str | Path | None = None,
) -> Path:
    repo_root = Path(repo_root).expanduser().resolve()
    bundle_dir = Path(bundle_dir).expanduser().resolve()
    app_dir = bundle_dir / "app"

    launcher_src = Path(launcher_exe).expanduser().resolve() if launcher_exe is not None else repo_root / "dist" / "ML_Local_App.exe"
    if not launcher_src.exists():
        raise FileNotFoundError(f"Launcher exe not found: {launcher_src}")

    if bundle_dir.exists():
        _safe_rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    copied_files = build_portable_app_dir(
        repo_root=repo_root,
        app_dir=app_dir,
    )

    _copy_file(launcher_src, bundle_dir / "ML_Local_App.exe")
    copied_files.append(
        {
            "type": "file",
            "relative_path": "ML_Local_App.exe",
            "size_bytes": int(launcher_src.stat().st_size),
        }
    )
    portable_readme = bundle_dir / "PORTABLE_README.txt"
    portable_readme.write_text(
        "\n".join(
            [
                f"{APP_NAME} Portable Bundle",
                f"Version: {APP_VERSION}",
                f"Channel: {APP_RELEASE_CHANNEL}",
                f"Release date: {APP_RELEASE_DATE}",
                "",
                "How to use:",
                "1. Keep this whole folder structure unchanged.",
                "2. Double-click ML_Local_App.exe.",
                "3. The launcher will start the local browser UI automatically.",
                "",
                "Notes:",
                "- This is a portable directory bundle, not a single fully standalone exe.",
                "- The bundled app/.venv is required at runtime.",
                "- app/local_app_runs/ will store run outputs and logs.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (bundle_dir / "APP_VERSION.json").write_text(
        json.dumps(
            {
                "app_name": APP_NAME,
                "app_version": APP_VERSION,
                "release_channel": APP_RELEASE_CHANNEL,
                "release_date": APP_RELEASE_DATE,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest = {
        "bundle_type": "ml_portable_directory_bundle",
        "app_name": APP_NAME,
        "app_version": APP_VERSION,
        "release_channel": APP_RELEASE_CHANNEL,
        "release_date": APP_RELEASE_DATE,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "repo_root": str(repo_root),
        "bundle_dir": str(bundle_dir),
        "launcher_exe": str(launcher_src),
        "entries": copied_files,
    }
    (bundle_dir / "portable_bundle_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return bundle_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a portable directory bundle for the ML local app")
    parser.add_argument("--repo_root", default=".", help="Repository root")
    parser.add_argument("--bundle_dir", default="portable_dist/ML_Portable", help="Portable bundle output directory")
    parser.add_argument("--launcher_exe", default=None, help="Optional path to ML_Local_App.exe")
    args = parser.parse_args(argv)

    bundle_dir = build_portable_bundle(
        repo_root=args.repo_root,
        bundle_dir=args.bundle_dir,
        launcher_exe=args.launcher_exe,
    )
    print(f"Portable bundle created: {bundle_dir}")
    print(f"Launcher: {bundle_dir / 'ML_Local_App.exe'}")
    print(f"App root: {bundle_dir / 'app'}")
    print(f"Virtual env: {bundle_dir / 'app' / '.venv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
