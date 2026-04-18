from __future__ import annotations

import html
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from app_metadata import APP_NAME, APP_RELEASE_CHANNEL, APP_VERSION
from build_candidate_comparisons import build_candidate_comparison_outputs
from build_experiment_state_ledger import build_global_experiment_ledger
from build_experiment_validation_report import build_experiment_validation_feedback
from build_result_archive import build_result_archive
from build_validation_retrain_comparison import build_validation_retrain_comparison
from demo_data_utils import (
    build_demo_experiment_override,
    make_synthetic_pose_features,
    write_demo_manifest,
)
from demo_report_utils import write_demo_interpretation, write_demo_overview_html, write_demo_readme
from input_path_repair import (
    AUTO_REPAIR_CONFIDENCE_THRESHOLD,
    analyze_input_path_repair_dataframe,
    apply_input_path_repair_plan,
)
from real_data_starter_utils import write_real_data_starter_kit
from run_cd38_public_starter import run_cd38_public_starter
from runtime_dependency_utils import (
    check_runtime_dependencies,
    format_missing_dependency_summary,
    get_pipeline_runtime_dependency_specs,
)

REPO_ROOT = Path(__file__).resolve().parent
LOCAL_APP_RUN_ROOT = REPO_ROOT / "local_app_runs"
COMPARE_EXPORT_ROOT = LOCAL_APP_RUN_ROOT / "compare_exports"
PARAM_TEMPLATE_ROOT = LOCAL_APP_RUN_ROOT / "parameter_templates"
GENERATED_INPUT_ROOT = LOCAL_APP_RUN_ROOT / "_generated_inputs"
DEMO_INPUT_ROOT = LOCAL_APP_RUN_ROOT / "_demo_inputs"
CD38_BENCHMARK_ROOT = REPO_ROOT / "benchmarks" / "cd38"
CD38_EXTERNAL_INPUT_ROOT = CD38_BENCHMARK_ROOT / "external_tool_inputs"
CD38_TRANSFER_ROOT = CD38_BENCHMARK_ROOT / "external_tool_transfer"
CD38_TRANSFER_ZIP_PATH = CD38_TRANSFER_ROOT / "cd38_external_tool_inputs_transfer.zip"
CD38_FINALIZE_ROOT = CD38_EXTERNAL_INPUT_ROOT / "finalize"
CD38_IMPORT_ROOT = CD38_EXTERNAL_INPUT_ROOT / "imported_outputs"
CD38_IMPORT_GATE_ROOT = CD38_EXTERNAL_INPUT_ROOT / "import_gate"
CD38_NEXT_RUN_REPORT_PATH = CD38_EXTERNAL_INPUT_ROOT / "cd38_external_tool_next_run.md"
CD38_NEXT_RUN_PLAN_PATH = CD38_EXTERNAL_INPUT_ROOT / "cd38_external_tool_next_run_plan.csv"
CD38_NEXT_RUN_PS1_PATH = CD38_EXTERNAL_INPUT_ROOT / "run_cd38_external_next_benchmark.ps1"
CD38_NEXT_RUN_SH_PATH = CD38_EXTERNAL_INPUT_ROOT / "run_cd38_external_next_benchmark.sh"
CD38_RETURN_SELFTEST_ROOT = REPO_ROOT / "smoke_test_outputs" / "cd38_return_import_selftest"
CD38_WORKFLOW_SELFTEST_ROOT = REPO_ROOT / "smoke_test_outputs" / "cd38_external_workflow_selftest"
CD38_PUBLIC_STARTER_ROOT = CD38_BENCHMARK_ROOT / "public_starter"
CD38_PUBLIC_STARTER_SUMMARY_PATH = CD38_PUBLIC_STARTER_ROOT / "cd38_public_starter_summary.json"
CD38_PUBLIC_STARTER_REPORT_PATH = CD38_PUBLIC_STARTER_ROOT / "cd38_public_starter_report.md"
APP_RUN_METADATA_NAME = "app_run_metadata.json"
APP_STDOUT_NAME = "app_stdout.log"
APP_STDERR_NAME = "app_stderr.log"
LOCAL_APP_RUN_ROOT.mkdir(parents=True, exist_ok=True)
COMPARE_EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
PARAM_TEMPLATE_ROOT.mkdir(parents=True, exist_ok=True)
GENERATED_INPUT_ROOT.mkdir(parents=True, exist_ok=True)
DEMO_INPUT_ROOT.mkdir(parents=True, exist_ok=True)

PDF_PAGE_WIDTH = 1240
PDF_PAGE_HEIGHT = 1754
PDF_PAGE_MARGIN = 72

START_MODE_LABELS = {
    "input_csv": "从 input_csv 开始",
    "feature_csv": "从 pose_features.csv 开始",
}

LOWER_IS_BETTER_COMPARE_METRICS = {
    "best_val_loss",
    "failed_rows",
    "warning_rows",
}

COMPARE_METRIC_LABELS = {
    "calibrated_rank_spearman": "Calibrated Rank Spearman",
    "baseline_rank_spearman": "Baseline Rank Spearman",
    "baseline_rule_auc": "Baseline Rule AUC",
    "calibrated_rule_auc": "Calibrated Rule AUC",
    "best_val_loss": "Best Val Loss",
    "failed_rows": "Failed Rows",
    "warning_rows": "Warning Rows",
}

KNOWN_STAGE_SUGGESTIONS = {
    "build_feature_table": "优先检查 input_csv 中的 pdb_path、pocket_file、catalytic_file、ligand_file 是否存在且可访问。",
    "rule_ranker": "先打开 pose_features.csv，确认关键特征列是否存在且不是全空。",
    "train_pose_model": "优先检查特征表是否有足够有效样本，以及环境中 torch 是否可导入。",
    "rank_nanobodies": "优先检查 model_outputs/pose_predictions.csv 是否已正常生成。",
    "compare_rule_vs_ml": "优先检查 rule 和 ML 排名文件是否同时存在，并确认 label 列可用。",
    "calibrate_rule_ranker": "优先检查 label 列是否存在、是否同时包含正负样本，以及样本量是否足够。",
    "compare_calibrated_rule_vs_ml": "优先检查 calibrated_rule_outputs/nanobody_rule_ranking.csv 是否生成成功。",
    "summarize_rule_ml_improvement": "优先检查 baseline/calibrated comparison summary JSON 是否存在。",
    "optimize_calibration_strategy": "优先检查 aggregation_calibration_trials.csv 是否生成成功。",
}

INPUT_CSV_REQUIRED_COLUMNS = [
    "nanobody_id",
    "conformer_id",
    "pose_id",
    "pdb_path",
]
FEATURE_CSV_REQUIRED_COLUMNS = [
    "nanobody_id",
    "conformer_id",
    "pose_id",
]
INPUT_CSV_OPTIONAL_HINT_COLUMNS = [
    "pocket_file",
    "catalytic_file",
    "ligand_file",
    "label",
]
FEATURE_CSV_OPTIONAL_HINT_COLUMNS = [
    "label",
    "experiment_label",
    "pred_prob",
    "pocket_hit_fraction",
    "catalytic_hit_fraction",
    "mouth_occlusion_score",
    "min_distance_to_pocket",
]
BUNDLE_DETECTION_RULES: list[tuple[str, list[str], str | None, set[str] | None]] = [
    ("input_csv_local_path", ["input_pose_table.csv"], "input", {".csv"}),
    ("feature_csv_local_path", ["pose_features.csv"], "feature", {".csv"}),
    ("default_pocket_local_path", [], "pocket", None),
    ("default_catalytic_local_path", [], "catalytic", None),
    ("default_ligand_local_path", [], "ligand", None),
    ("experiment_plan_override_local_path", ["experiment_plan_override.csv", "experiment_plan_overrides.csv"], "experiment", {".csv"}),
]
BUNDLE_IMPORT_ROOT = LOCAL_APP_RUN_ROOT / "_bundle_imports"
BUNDLE_IMPORT_ROOT.mkdir(parents=True, exist_ok=True)
AUTO_INPUT_PDB_SUFFIXES = {".pdb"}
AUTO_INPUT_EXCLUDE_NAME_TOKENS = {
    "pocket",
    "fpocket",
    "ligand",
    "substrate",
    "catalytic",
    "template",
    "predicted",
}


def _slugify_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    text = text.strip("._")
    return text or datetime.now().strftime("run_%Y%m%d_%H%M%S")


def _default_run_name() -> str:
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


FORM_DEFAULTS: dict[str, Any] = {
    "start_mode": "input_csv",
    "run_name": _default_run_name(),
    "input_csv_local_path": "",
    "feature_csv_local_path": "",
    "default_pocket_local_path": "",
    "default_catalytic_local_path": "",
    "default_ligand_local_path": "",
    "experiment_plan_override_local_path": "",
    "default_antigen_chain": "",
    "default_nanobody_chain": "",
    "label_col": "label",
    "top_k": 3,
    "pocket_overwide_penalty_weight": 0.0,
    "pocket_overwide_threshold": 0.55,
    "train_epochs": 20,
    "train_batch_size": 64,
    "train_val_ratio": 0.25,
    "train_early_stopping_patience": 8,
    "seed": 42,
    "skip_failed_rows": True,
    "disable_label_aware_steps": False,
    "enable_ai_assistant": False,
    "ai_provider": "none",
    "ai_model": "",
    "ai_max_rows": 8,
    "template_name": "default",
    "selected_template_name": "",
    "selected_history_label": "",
}

FORM_FIELD_KEYS = [
    "start_mode",
    "run_name",
    "input_csv_local_path",
    "feature_csv_local_path",
    "default_pocket_local_path",
    "default_catalytic_local_path",
    "default_ligand_local_path",
    "experiment_plan_override_local_path",
    "default_antigen_chain",
    "default_nanobody_chain",
    "label_col",
    "top_k",
    "pocket_overwide_penalty_weight",
    "pocket_overwide_threshold",
    "train_epochs",
    "train_batch_size",
    "train_val_ratio",
    "train_early_stopping_patience",
    "seed",
    "skip_failed_rows",
    "disable_label_aware_steps",
    "enable_ai_assistant",
    "ai_provider",
    "ai_model",
    "ai_max_rows",
]


def _now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


def _build_sample_input_csv_text() -> str:
    df = pd.DataFrame(
        [
            {
                "nanobody_id": "NB001",
                "conformer_id": "conf_01",
                "pose_id": "pose_001",
                "pdb_path": r"data\complex_001.pdb",
                "antigen_chain": "A",
                "nanobody_chain": "H",
                "pocket_file": r"data\pocket_residues.txt",
                "catalytic_file": r"data\catalytic_residues.txt",
                "ligand_file": r"data\ligand_template.pdb",
                "label": 1,
            }
        ]
    )
    return df.to_csv(index=False)


def _build_sample_feature_csv_text() -> str:
    df = pd.DataFrame(
        [
            {
                "nanobody_id": "NB001",
                "conformer_id": "conf_01",
                "pose_id": "pose_001",
                "pocket_hit_fraction": 0.72,
                "catalytic_hit_fraction": 0.35,
                "mouth_occlusion_score": 0.64,
                "min_distance_to_pocket": 3.1,
                "pred_prob": 0.81,
                "label": 1,
            }
        ]
    )
    return df.to_csv(index=False)


def _build_sample_experiment_plan_override_csv_text() -> str:
    df = pd.DataFrame(
        [
            {
                "nanobody_id": "NB001",
                "plan_override": "include",
                "experiment_status": "pending",
                "experiment_result": "",
                "validation_label": "",
                "experiment_owner": "lab_member_A",
                "experiment_cost": 1,
                "experiment_note": "priority candidate",
            },
            {
                "nanobody_id": "NB002",
                "plan_override": "exclude",
                "experiment_status": "blocked",
                "experiment_result": "",
                "validation_label": "",
                "experiment_owner": "",
                "experiment_cost": "",
                "experiment_note": "material unavailable",
            },
        ]
    )
    return df.to_csv(index=False)


def _generate_local_demo_inputs() -> dict[str, Any]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    demo_dir = DEMO_INPUT_ROOT / f"demo_{stamp}"
    demo_dir.mkdir(parents=True, exist_ok=True)

    feature_csv = demo_dir / "demo_pose_features.csv"
    override_csv = demo_dir / "demo_experiment_plan_override.csv"
    manifest_json = demo_dir / "demo_manifest.json"

    feature_summary = make_synthetic_pose_features(
        out_csv=feature_csv,
        n_nanobodies=8,
        n_conformers=2,
        n_poses=4,
        seed=20260418,
    )
    override_summary = build_demo_experiment_override(
        feature_csv=feature_csv,
        out_csv=override_csv,
    )
    manifest = write_demo_manifest(
        out_json=manifest_json,
        feature_summary=feature_summary,
        override_summary=override_summary,
    )

    return {
        "demo_dir": str(demo_dir),
        "feature_csv": str(feature_csv),
        "override_csv": str(override_csv),
        "manifest_json": str(manifest_json),
        "manifest": manifest,
    }


def _apply_demo_inputs_to_session_state(demo_payload: dict[str, Any], *, action_message: str) -> None:
    demo_feature_csv = str(demo_payload.get("feature_csv") or "")
    demo_override_csv = str(demo_payload.get("override_csv") or "")
    demo_dir = str(demo_payload.get("demo_dir") or "")
    st.session_state["start_mode"] = "feature_csv"
    st.session_state["feature_csv_local_path"] = demo_feature_csv
    st.session_state["experiment_plan_override_local_path"] = demo_override_csv
    st.session_state["label_col"] = "label"
    st.session_state["run_name"] = f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    st.session_state["train_epochs"] = min(int(st.session_state.get("train_epochs") or 3), 3)
    st.session_state["train_early_stopping_patience"] = min(
        int(st.session_state.get("train_early_stopping_patience") or 2),
        2,
    )
    st.session_state["disable_label_aware_steps"] = False
    st.session_state["last_scheduler_message"] = action_message.format(demo_dir=demo_dir)


def _load_demo_inputs_into_session_state() -> None:
    try:
        demo_payload = _generate_local_demo_inputs()
        _apply_demo_inputs_to_session_state(
            demo_payload,
            action_message="已生成并载入 demo 输入: {demo_dir}。确认后点击“立即运行”。",
        )
    except Exception as exc:
        st.session_state["last_scheduler_message"] = f"生成 demo 输入失败: {exc}"


def _run_demo_now_from_session_state() -> None:
    if isinstance(st.session_state.get("active_run_info"), dict):
        st.session_state["last_scheduler_message"] = "当前已有任务在运行，不能同时启动 demo。"
        return

    try:
        demo_payload = _generate_local_demo_inputs()
        _apply_demo_inputs_to_session_state(
            demo_payload,
            action_message="已生成 demo 输入并启动运行: {demo_dir}。",
        )
        run_request = _prepare_run_request_from_form(
            input_csv_upload=None,
            feature_csv_upload=None,
            default_pocket_upload=None,
            default_catalytic_upload=None,
            default_ligand_upload=None,
            experiment_plan_override_upload=None,
        )
        st.session_state["queue_auto_run"] = False
        _start_background_run(run_request, queue_source="demo_run")
    except Exception as exc:
        st.session_state["last_scheduler_message"] = f"启动 demo 运行失败: {exc}"


def _detect_demo_source_dir(form_payload: dict[str, Any]) -> Path | None:
    feature_path_text = str(form_payload.get("feature_csv_local_path") or "").strip()
    if not feature_path_text:
        return None
    try:
        feature_path = Path(feature_path_text).expanduser().resolve()
        demo_root = DEMO_INPUT_ROOT.resolve()
        if feature_path.is_relative_to(demo_root):
            return feature_path.parent
    except (OSError, RuntimeError, ValueError):
        return None
    return None


def _prepare_local_demo_run_files(
    *,
    form_payload: dict[str, Any],
    input_dir: Path,
    out_dir: Path,
    resolved_inputs: dict[str, Any],
) -> dict[str, Any] | None:
    demo_source_dir = _detect_demo_source_dir(form_payload)
    if demo_source_dir is None:
        return None

    source_manifest = demo_source_dir / "demo_manifest.json"
    copied_manifest: Path | None = None
    if source_manifest.exists():
        copied_manifest = input_dir / "demo_manifest.json"
        shutil.copy2(source_manifest, copied_manifest)

    demo_readme = out_dir / "DEMO_README.md"
    demo_interpretation = out_dir / "DEMO_INTERPRETATION.md"
    demo_overview = out_dir / "DEMO_OVERVIEW.html"
    real_data_starter_dir = out_dir / "REAL_DATA_STARTER"
    starter_manifest = write_real_data_starter_kit(real_data_starter_dir)
    write_demo_interpretation(
        out_path=demo_interpretation,
        feature_csv=str(resolved_inputs.get("feature_csv") or ""),
        override_csv=str(resolved_inputs.get("experiment_plan_override_csv") or ""),
        manifest_json=str(copied_manifest or source_manifest if source_manifest.exists() else ""),
        real_data_starter_dir=real_data_starter_dir,
        summary=None,
    )
    write_demo_overview_html(
        out_path=demo_overview,
        feature_csv=str(resolved_inputs.get("feature_csv") or ""),
        override_csv=str(resolved_inputs.get("experiment_plan_override_csv") or ""),
        manifest_json=str(copied_manifest or source_manifest if source_manifest.exists() else ""),
        readme_md=demo_readme,
        interpretation_md=demo_interpretation,
        real_data_starter_dir=real_data_starter_dir,
        summary=None,
    )
    write_demo_readme(
        out_path=demo_readme,
        feature_csv=str(resolved_inputs.get("feature_csv") or ""),
        override_csv=str(resolved_inputs.get("experiment_plan_override_csv") or ""),
        overview_html=demo_overview,
        interpretation_md=demo_interpretation,
        manifest_json=str(copied_manifest or source_manifest if source_manifest.exists() else ""),
        real_data_starter_dir=real_data_starter_dir,
        summary=None,
    )
    return {
        "demo_source_dir": str(demo_source_dir),
        "demo_manifest_json": str(copied_manifest or source_manifest if source_manifest.exists() else ""),
        "demo_readme_md": str(demo_readme),
        "demo_interpretation_md": str(demo_interpretation),
        "demo_overview_html": str(demo_overview),
        "real_data_starter_dir": str(real_data_starter_dir),
        "real_data_starter_readme_md": str(starter_manifest.get("outputs", {}).get("readme_md") or ""),
        "real_data_starter_manifest_json": str(starter_manifest.get("outputs", {}).get("manifest_json") or ""),
        "mini_pdb_example_readme_md": str(starter_manifest.get("outputs", {}).get("mini_pdb_example_readme_md") or ""),
        "mini_pdb_example_input_csv": str(starter_manifest.get("outputs", {}).get("mini_pdb_example_input_csv") or ""),
        "synthetic_demo": True,
    }


def _refresh_local_demo_run_files(
    *,
    metadata: dict[str, Any],
    summary: dict[str, Any] | None,
    out_dir: Path,
) -> dict[str, Any] | None:
    demo_mode = metadata.get("demo_mode") if isinstance(metadata.get("demo_mode"), dict) else None
    if not isinstance(demo_mode, dict) or not bool(demo_mode.get("synthetic_demo")):
        return None

    resolved_inputs = metadata.get("resolved_inputs") if isinstance(metadata.get("resolved_inputs"), dict) else {}
    feature_csv = str(resolved_inputs.get("feature_csv") or "")
    override_csv = str(resolved_inputs.get("experiment_plan_override_csv") or "")
    manifest_json = str(demo_mode.get("demo_manifest_json") or "")
    demo_readme = out_dir / "DEMO_README.md"
    demo_interpretation = out_dir / "DEMO_INTERPRETATION.md"
    demo_overview = out_dir / "DEMO_OVERVIEW.html"
    real_data_starter_dir = out_dir / "REAL_DATA_STARTER"
    starter_manifest = write_real_data_starter_kit(real_data_starter_dir)

    write_demo_interpretation(
        out_path=demo_interpretation,
        feature_csv=feature_csv,
        override_csv=override_csv,
        manifest_json=manifest_json,
        real_data_starter_dir=real_data_starter_dir,
        summary=summary,
    )
    write_demo_overview_html(
        out_path=demo_overview,
        feature_csv=feature_csv,
        override_csv=override_csv,
        manifest_json=manifest_json,
        readme_md=demo_readme,
        interpretation_md=demo_interpretation,
        real_data_starter_dir=real_data_starter_dir,
        summary=summary,
    )
    write_demo_readme(
        out_path=demo_readme,
        feature_csv=feature_csv,
        override_csv=override_csv,
        manifest_json=manifest_json,
        overview_html=demo_overview,
        interpretation_md=demo_interpretation,
        real_data_starter_dir=real_data_starter_dir,
        summary=summary,
    )
    return {
        "demo_readme_md": str(demo_readme),
        "demo_interpretation_md": str(demo_interpretation),
        "demo_overview_html": str(demo_overview),
        "real_data_starter_dir": str(real_data_starter_dir),
        "real_data_starter_readme_md": str(starter_manifest.get("outputs", {}).get("readme_md") or ""),
        "real_data_starter_manifest_json": str(starter_manifest.get("outputs", {}).get("manifest_json") or ""),
        "mini_pdb_example_readme_md": str(starter_manifest.get("outputs", {}).get("mini_pdb_example_readme_md") or ""),
        "mini_pdb_example_input_csv": str(starter_manifest.get("outputs", {}).get("mini_pdb_example_input_csv") or ""),
    }


PLAN_OVERRIDE_COLUMNS = [
    "nanobody_id",
    "plan_override",
    "experiment_status",
    "experiment_result",
    "validation_label",
    "experiment_owner",
    "experiment_cost",
    "experiment_note",
    "manual_plan_reason",
]
PLAN_OVERRIDE_ACTION_OPTIONS = ["", "include", "exclude", "standby", "defer"]
PLAN_STATUS_OPTIONS = ["", "planned", "pending", "in_progress", "completed", "blocked", "failed", "cancelled"]
PLAN_RESULT_OPTIONS = ["", "positive", "negative", "inconclusive"]
PLAN_LABEL_OPTIONS = ["", "1", "0"]


def _clean_plan_cell(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null", "na", "n/a"}:
        return ""
    return text


def _normalize_editor_plan_override(value: Any) -> str:
    text = _clean_plan_cell(value).lower().replace("-", "_").replace(" ", "_")
    if text in {"include", "include_now", "force_include", "lock", "locked", "must_include"}:
        return "include"
    if text in {"exclude", "force_exclude", "skip", "remove", "reject", "blocked", "do_not_test"}:
        return "exclude"
    if text in {"standby", "backup", "reserve", "waitlist", "wait_list"}:
        return "standby"
    if text in {"defer", "later", "later_round", "postpone", "hold"}:
        return "defer"
    return ""


def _build_experiment_plan_override_editor_df(plan_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in plan_df.iterrows():
        manual_action = _normalize_editor_plan_override(row.get("manual_plan_action"))
        status = _clean_plan_cell(row.get("experiment_status"))
        if not status and _clean_plan_cell(row.get("plan_decision")) == "include_now":
            status = "planned"
        rows.append(
            {
                "nanobody_id": _clean_plan_cell(row.get("nanobody_id")),
                "plan_override": manual_action,
                "experiment_status": status,
                "experiment_result": _clean_plan_cell(row.get("experiment_result")),
                "validation_label": _clean_plan_cell(row.get("validation_label")),
                "experiment_owner": _clean_plan_cell(row.get("experiment_owner")),
                "experiment_cost": _clean_plan_cell(row.get("experiment_cost")),
                "experiment_note": _clean_plan_cell(row.get("experiment_note")),
                "manual_plan_reason": _clean_plan_cell(row.get("manual_plan_reason")),
            }
        )
    editor_df = pd.DataFrame(rows, columns=PLAN_OVERRIDE_COLUMNS)
    return editor_df.drop_duplicates(subset=["nanobody_id"], keep="first").reset_index(drop=True)


def _sanitize_plan_override_editor_df(editor_df: pd.DataFrame) -> pd.DataFrame:
    if editor_df.empty:
        return pd.DataFrame(columns=PLAN_OVERRIDE_COLUMNS)
    cleaned = editor_df.copy()
    for col in PLAN_OVERRIDE_COLUMNS:
        if col not in cleaned.columns:
            cleaned[col] = ""
    cleaned = cleaned[PLAN_OVERRIDE_COLUMNS].copy()
    for col in PLAN_OVERRIDE_COLUMNS:
        cleaned[col] = cleaned[col].map(_clean_plan_cell)
    cleaned["plan_override"] = cleaned["plan_override"].map(_normalize_editor_plan_override)
    cleaned = cleaned[cleaned["nanobody_id"].astype(str).str.strip().ne("")]
    return cleaned.reset_index(drop=True)


def _load_page_icon() -> Image.Image | str:
    icon_path = REPO_ROOT / "assets" / "app_icon.png"
    if icon_path.exists():
        try:
            return Image.open(icon_path)
        except Exception:
            return "ML"
    return "ML"


def _save_uploaded_file(uploaded_file: Any, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    out_path = dst_dir / str(uploaded_file.name)
    out_path.write_bytes(bytes(uploaded_file.getbuffer()))
    return out_path


def _source_present(uploaded_file: Any, local_path_text: str) -> bool:
    if str(local_path_text or "").strip():
        return True
    return uploaded_file is not None


def _resolve_input_file(
    *,
    uploaded_file: Any,
    local_path_text: str,
    dst_dir: Path,
    required: bool,
    label: str,
) -> Path | None:
    local_path = str(local_path_text or "").strip()
    if local_path:
        path = Path(local_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"{label} local path not found: {path}")
        return path
    if uploaded_file is not None:
        return _save_uploaded_file(uploaded_file, dst_dir)
    if required:
        raise ValueError(f"{label} is required")
    return None


def _build_pipeline_command(
    *,
    start_mode: str,
    input_csv: Path | None,
    feature_csv: Path | None,
    out_dir: Path,
    default_pocket_file: Path | None,
    default_catalytic_file: Path | None,
    default_ligand_file: Path | None,
    experiment_plan_override_file: Path | None,
    default_antigen_chain: str,
    default_nanobody_chain: str,
    label_col: str,
    top_k: int,
    pocket_overwide_penalty_weight: float,
    pocket_overwide_threshold: float,
    train_epochs: int,
    train_batch_size: int,
    train_val_ratio: float,
    train_early_stopping_patience: int,
    seed: int,
    skip_failed_rows: bool,
    disable_label_aware_steps: bool,
    enable_ai_assistant: bool,
    ai_provider: str,
    ai_model: str,
    ai_max_rows: int,
) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "run_recommended_pipeline.py"),
        "--out_dir",
        str(out_dir),
        "--top_k",
        str(int(top_k)),
        "--pocket_overwide_penalty_weight",
        str(float(pocket_overwide_penalty_weight)),
        "--pocket_overwide_threshold",
        str(float(pocket_overwide_threshold)),
        "--train_epochs",
        str(int(train_epochs)),
        "--train_batch_size",
        str(int(train_batch_size)),
        "--train_val_ratio",
        str(float(train_val_ratio)),
        "--train_early_stopping_patience",
        str(int(train_early_stopping_patience)),
        "--seed",
        str(int(seed)),
        "--label_col",
        str(label_col or "label"),
    ]
    if start_mode == "input_csv":
        if input_csv is None:
            raise ValueError("input_csv is required for input_csv mode")
        command.extend(["--input_csv", str(input_csv)])
    else:
        if feature_csv is None:
            raise ValueError("feature_csv is required for feature_csv mode")
        command.extend(["--feature_csv", str(feature_csv)])

    if default_pocket_file is not None:
        command.extend(["--default_pocket_file", str(default_pocket_file)])
    if default_catalytic_file is not None:
        command.extend(["--default_catalytic_file", str(default_catalytic_file)])
    if default_ligand_file is not None:
        command.extend(["--default_ligand_file", str(default_ligand_file)])
    if experiment_plan_override_file is not None:
        command.extend(["--experiment_plan_override_csv", str(experiment_plan_override_file)])

    antigen_chain = str(default_antigen_chain or "").strip()
    if antigen_chain:
        command.extend(["--default_antigen_chain", antigen_chain])

    nanobody_chain = str(default_nanobody_chain or "").strip()
    if nanobody_chain:
        command.extend(["--default_nanobody_chain", nanobody_chain])

    if skip_failed_rows:
        command.append("--skip_failed_rows")
    if disable_label_aware_steps:
        command.append("--disable_label_aware_steps")
    if enable_ai_assistant:
        command.append("--enable_ai_assistant")
        command.extend(["--ai_provider", str(ai_provider or "none")])
        command.extend(["--ai_max_rows", str(int(ai_max_rows))])
        model_text = str(ai_model or "").strip()
        if model_text:
            command.extend(["--ai_model", model_text])

    return command


def _build_pipeline_kwargs(
    *,
    form_payload: dict[str, Any],
    out_dir: Path,
    input_csv: Path | None,
    feature_csv: Path | None,
    default_pocket_file: Path | None,
    default_catalytic_file: Path | None,
    default_ligand_file: Path | None,
    experiment_plan_override_file: Path | None,
) -> dict[str, Any]:
    return {
        "input_csv": None if input_csv is None else str(input_csv),
        "feature_csv": None if feature_csv is None else str(feature_csv),
        "out_dir": str(out_dir),
        "label_col": str(form_payload.get("label_col") or "label"),
        "disable_label_aware_steps": bool(form_payload.get("disable_label_aware_steps", False)),
        "default_pocket_file": None if default_pocket_file is None else str(default_pocket_file),
        "default_catalytic_file": None if default_catalytic_file is None else str(default_catalytic_file),
        "default_ligand_file": None if default_ligand_file is None else str(default_ligand_file),
        "experiment_plan_override_csv": None
        if experiment_plan_override_file is None
        else str(experiment_plan_override_file),
        "default_antigen_chain": str(form_payload.get("default_antigen_chain") or ""),
        "default_nanobody_chain": str(form_payload.get("default_nanobody_chain") or ""),
        "skip_failed_rows": bool(form_payload.get("skip_failed_rows", True)),
        "top_k": int(form_payload.get("top_k", 3)),
        "pocket_overwide_penalty_weight": float(form_payload.get("pocket_overwide_penalty_weight", 0.0)),
        "pocket_overwide_threshold": float(form_payload.get("pocket_overwide_threshold", 0.55)),
        "train_epochs": int(form_payload.get("train_epochs", 20)),
        "train_batch_size": int(form_payload.get("train_batch_size", 64)),
        "train_val_ratio": float(form_payload.get("train_val_ratio", 0.25)),
        "train_early_stopping_patience": int(form_payload.get("train_early_stopping_patience", 8)),
        "seed": int(form_payload.get("seed", 42)),
        "enable_ai_assistant": bool(form_payload.get("enable_ai_assistant", False)),
        "ai_provider": str(form_payload.get("ai_provider") or "none"),
        "ai_model": str(form_payload.get("ai_model") or "").strip() or None,
        "ai_max_rows": int(form_payload.get("ai_max_rows", 8)),
    }


def _run_pipeline_in_process(pipeline_kwargs: dict[str, Any]) -> tuple[int, str, str, dict[str, Any] | None]:
    from run_recommended_pipeline import run_recommended_pipeline

    stdout_buffer = StringIO()
    stderr_buffer = StringIO()
    summary: dict[str, Any] | None = None

    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            summary = run_recommended_pipeline(**pipeline_kwargs)
        return 0, stdout_buffer.getvalue(), stderr_buffer.getvalue(), summary
    except Exception:
        trace_text = traceback.format_exc()
        stderr_text = stderr_buffer.getvalue()
        if stderr_text and not stderr_text.endswith("\n"):
            stderr_text += "\n"
        stderr_text += trace_text
        return 1, stdout_buffer.getvalue(), stderr_text, summary


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, low_memory=False)


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _extract_tail(text: str, max_lines: int = 20) -> str:
    lines = [line for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        return ""
    return "\n".join(lines[-max_lines:])


def _detect_failed_stage(text: str) -> str | None:
    match = re.search(r"Command failed \[([^\]]+)\]", text)
    if match:
        return str(match.group(1)).strip()
    return None


def _build_form_payload() -> dict[str, Any]:
    return {key: st.session_state.get(key) for key in FORM_FIELD_KEYS}


def _apply_form_payload(payload: dict[str, Any] | None) -> None:
    if not isinstance(payload, dict):
        return
    for key in FORM_FIELD_KEYS:
        if key in payload:
            st.session_state[key] = payload[key]


def _list_parameter_templates() -> list[Path]:
    return sorted(PARAM_TEMPLATE_ROOT.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def _save_parameter_template(template_name: str) -> Path:
    safe_name = _slugify_name(template_name)
    out_path = PARAM_TEMPLATE_ROOT / f"{safe_name}.json"
    payload = {
        "template_name": safe_name,
        "saved_at": _now_text(),
        "fields": _build_form_payload(),
    }
    _write_json(out_path, payload)
    return out_path


def _load_parameter_template(template_name: str) -> dict[str, Any] | None:
    safe_name = _slugify_name(template_name)
    path = PARAM_TEMPLATE_ROOT / f"{safe_name}.json"
    return _load_json(path)


def _build_history_requeue_form_payload(run_root: Path) -> dict[str, Any] | None:
    metadata = _read_run_metadata(run_root)
    form_payload = metadata.get("form_payload") if isinstance(metadata.get("form_payload"), dict) else None
    if not isinstance(form_payload, dict):
        return None

    payload = dict(form_payload)
    resolved_inputs = metadata.get("resolved_inputs") if isinstance(metadata.get("resolved_inputs"), dict) else {}
    field_mapping = {
        "input_csv": "input_csv_local_path",
        "feature_csv": "feature_csv_local_path",
        "default_pocket_file": "default_pocket_local_path",
        "default_catalytic_file": "default_catalytic_local_path",
        "default_ligand_file": "default_ligand_local_path",
    }
    for resolved_key, field_key in field_mapping.items():
        current_value = str(payload.get(field_key) or "").strip()
        if current_value:
            continue
        resolved_value = str(resolved_inputs.get(resolved_key) or "").strip()
        if resolved_value:
            payload[field_key] = resolved_value

    payload["run_name"] = f"{run_root.name}_rerun"
    return payload


def _allocate_run_identity(preferred_name: str) -> tuple[str, Path]:
    safe_name = _slugify_name(preferred_name)
    candidate_root = LOCAL_APP_RUN_ROOT / safe_name
    if not candidate_root.exists():
        return safe_name, candidate_root

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    indexed_name = f"{safe_name}_{stamp}"
    candidate_root = LOCAL_APP_RUN_ROOT / indexed_name
    if not candidate_root.exists():
        return indexed_name, candidate_root

    suffix = 2
    while True:
        next_name = f"{indexed_name}_{suffix:02d}"
        next_root = LOCAL_APP_RUN_ROOT / next_name
        if not next_root.exists():
            return next_name, next_root
        suffix += 1


def _prepare_run_request_from_form(
    *,
    input_csv_upload: Any,
    feature_csv_upload: Any,
    default_pocket_upload: Any,
    default_catalytic_upload: Any,
    default_ligand_upload: Any,
    experiment_plan_override_upload: Any,
) -> dict[str, Any]:
    form_payload = _build_form_payload()
    _ensure_runtime_dependencies_ready(str(form_payload.get("start_mode") or "input_csv"))
    safe_run_name, run_root = _allocate_run_identity(str(form_payload.get("run_name", "")))
    input_dir = run_root / "inputs"
    out_dir = run_root / "outputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_csv = _resolve_input_file(
        uploaded_file=input_csv_upload,
        local_path_text=str(form_payload.get("input_csv_local_path") or ""),
        dst_dir=input_dir,
        required=str(form_payload.get("start_mode")) == "input_csv",
        label="input_csv",
    )
    feature_csv = _resolve_input_file(
        uploaded_file=feature_csv_upload,
        local_path_text=str(form_payload.get("feature_csv_local_path") or ""),
        dst_dir=input_dir,
        required=str(form_payload.get("start_mode")) == "feature_csv",
        label="feature_csv",
    )
    default_pocket_file = _resolve_input_file(
        uploaded_file=default_pocket_upload,
        local_path_text=str(form_payload.get("default_pocket_local_path") or ""),
        dst_dir=input_dir,
        required=False,
        label="default_pocket_file",
    )
    default_catalytic_file = _resolve_input_file(
        uploaded_file=default_catalytic_upload,
        local_path_text=str(form_payload.get("default_catalytic_local_path") or ""),
        dst_dir=input_dir,
        required=False,
        label="default_catalytic_file",
    )
    default_ligand_file = _resolve_input_file(
        uploaded_file=default_ligand_upload,
        local_path_text=str(form_payload.get("default_ligand_local_path") or ""),
        dst_dir=input_dir,
        required=False,
        label="default_ligand_file",
    )
    experiment_plan_override_file = _resolve_input_file(
        uploaded_file=experiment_plan_override_upload,
        local_path_text=str(form_payload.get("experiment_plan_override_local_path") or ""),
        dst_dir=input_dir,
        required=False,
        label="experiment_plan_override_csv",
    )

    resolved_inputs = {
        "input_csv": None if input_csv is None else str(input_csv),
        "feature_csv": None if feature_csv is None else str(feature_csv),
        "default_pocket_file": None if default_pocket_file is None else str(default_pocket_file),
        "default_catalytic_file": None if default_catalytic_file is None else str(default_catalytic_file),
        "default_ligand_file": None if default_ligand_file is None else str(default_ligand_file),
        "experiment_plan_override_csv": None
        if experiment_plan_override_file is None
        else str(experiment_plan_override_file),
    }
    demo_metadata = _prepare_local_demo_run_files(
        form_payload=form_payload,
        input_dir=input_dir,
        out_dir=out_dir,
        resolved_inputs=resolved_inputs,
    )

    command = _build_pipeline_command(
        start_mode=str(form_payload.get("start_mode")),
        input_csv=input_csv,
        feature_csv=feature_csv,
        out_dir=out_dir,
        default_pocket_file=default_pocket_file,
        default_catalytic_file=default_catalytic_file,
        default_ligand_file=default_ligand_file,
        experiment_plan_override_file=experiment_plan_override_file,
        default_antigen_chain=str(form_payload.get("default_antigen_chain") or ""),
        default_nanobody_chain=str(form_payload.get("default_nanobody_chain") or ""),
        label_col=str(form_payload.get("label_col") or "label"),
        top_k=int(form_payload.get("top_k", 3)),
        pocket_overwide_penalty_weight=float(form_payload.get("pocket_overwide_penalty_weight", 0.0)),
        pocket_overwide_threshold=float(form_payload.get("pocket_overwide_threshold", 0.55)),
        train_epochs=int(form_payload.get("train_epochs", 20)),
        train_batch_size=int(form_payload.get("train_batch_size", 64)),
        train_val_ratio=float(form_payload.get("train_val_ratio", 0.25)),
        train_early_stopping_patience=int(form_payload.get("train_early_stopping_patience", 8)),
        seed=int(form_payload.get("seed", 42)),
        skip_failed_rows=bool(form_payload.get("skip_failed_rows", True)),
        disable_label_aware_steps=bool(form_payload.get("disable_label_aware_steps", False)),
        enable_ai_assistant=bool(form_payload.get("enable_ai_assistant", False)),
        ai_provider=str(form_payload.get("ai_provider") or "none"),
        ai_model=str(form_payload.get("ai_model") or ""),
        ai_max_rows=int(form_payload.get("ai_max_rows", 8)),
    )
    pipeline_kwargs = _build_pipeline_kwargs(
        form_payload=form_payload,
        out_dir=out_dir,
        input_csv=input_csv,
        feature_csv=feature_csv,
        default_pocket_file=default_pocket_file,
        default_catalytic_file=default_catalytic_file,
        default_ligand_file=default_ligand_file,
        experiment_plan_override_file=experiment_plan_override_file,
    )
    summary_path = out_dir / "recommended_pipeline_summary.json"
    metadata = {
        "run_name": safe_run_name,
        "created_at": _now_text(),
        "status": "prepared",
        "execution_mode": "subprocess_cli_background",
        "start_mode": str(form_payload.get("start_mode")),
        "form_payload": form_payload,
        "resolved_inputs": resolved_inputs,
        "demo_mode": demo_metadata,
        "command": command,
        "pipeline_kwargs": pipeline_kwargs,
        "returncode": None,
        "out_dir": str(out_dir),
        "summary_path": str(summary_path),
        "diagnostics": None,
        "error_text": "",
    }
    _write_json(run_root / APP_RUN_METADATA_NAME, metadata)
    _write_text(run_root / APP_STDOUT_NAME, "")
    _write_text(run_root / APP_STDERR_NAME, "")

    return {
        "run_name": safe_run_name,
        "run_root": str(run_root),
        "input_dir": str(input_dir),
        "out_dir": str(out_dir),
        "summary_path": str(summary_path),
        "command": command,
        "resolved_inputs": resolved_inputs,
        "form_payload": form_payload,
        "pipeline_kwargs": pipeline_kwargs,
    }


def _enqueue_history_run_request(run_root: Path) -> str:
    payload = _build_history_requeue_form_payload(run_root)
    if not isinstance(payload, dict):
        raise ValueError("所选历史记录缺少可复用的表单配置，无法重新入队。")

    _apply_form_payload(payload)
    run_request = _prepare_run_request_from_form(
        input_csv_upload=None,
        feature_csv_upload=None,
        default_pocket_upload=None,
        default_catalytic_upload=None,
        default_ligand_upload=None,
        experiment_plan_override_upload=None,
    )
    queue = st.session_state.get("run_queue")
    if not isinstance(queue, list):
        queue = []
    queue.append(run_request)
    st.session_state["run_queue"] = queue
    message = f"已按历史配置重新入队: {run_request['run_name']}"
    st.session_state["last_scheduler_message"] = message
    return message


def _read_run_metadata(run_root: Path) -> dict[str, Any]:
    metadata = _load_json(run_root / APP_RUN_METADATA_NAME)
    return metadata if isinstance(metadata, dict) else {}


def _write_run_metadata(run_root: Path, metadata: dict[str, Any]) -> None:
    _write_json(run_root / APP_RUN_METADATA_NAME, metadata)


def _build_cancelled_diagnostics(
    *,
    stdout_text: str,
    stderr_text: str,
    out_dir: Path,
    resolved_inputs: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "status": "cancelled",
        "messages": ["运行已被用户主动停止。"],
        "suggestions": [
            "如需继续，先检查当前输出目录里是否已有部分产物，再决定是否重跑。",
            "如果这次输入需要保留，可直接从运行历史载入并再次加入队列。",
        ],
        "failed_stage": None,
        "resolved_inputs": resolved_inputs or {},
        "stdout_excerpt": _extract_tail(stdout_text),
        "stderr_excerpt": _extract_tail(stderr_text),
        "out_dir": str(out_dir),
        "output_dir_exists": out_dir.exists(),
    }


def _finalize_background_run(
    *,
    run_info: dict[str, Any],
    returncode: int,
    forced_status: str | None = None,
    forced_error_text: str | None = None,
) -> dict[str, Any]:
    run_root = Path(str(run_info["run_root"]))
    out_dir = Path(str(run_info["out_dir"]))
    summary_path = Path(str(run_info["summary_path"]))
    resolved_inputs = run_info.get("resolved_inputs") if isinstance(run_info.get("resolved_inputs"), dict) else {}
    stdout_text = _read_text(run_root / APP_STDOUT_NAME)
    stderr_text = _read_text(run_root / APP_STDERR_NAME)
    summary = _load_json(summary_path)
    metadata = _read_run_metadata(run_root)
    demo_outputs = _refresh_local_demo_run_files(
        metadata=metadata,
        summary=summary,
        out_dir=out_dir,
    )
    if demo_outputs:
        demo_mode = metadata.get("demo_mode") if isinstance(metadata.get("demo_mode"), dict) else {}
        demo_mode.update(demo_outputs)
        metadata["demo_mode"] = demo_mode

    if forced_status == "cancelled":
        status = "cancelled"
        diagnostics = _build_cancelled_diagnostics(
            stdout_text=stdout_text,
            stderr_text=stderr_text,
            out_dir=out_dir,
            resolved_inputs=resolved_inputs,
        )
        error_text = forced_error_text or f"Run cancelled by user. Output directory: {out_dir}"
    elif returncode == 0:
        status = "success"
        diagnostics = _build_success_diagnostics(summary, resolved_inputs, out_dir)
        error_text = ""
    else:
        status = "failed"
        diagnostics = _build_failure_diagnostics(
            returncode=int(returncode),
            stdout_text=stdout_text,
            stderr_text=stderr_text,
            summary=summary,
            out_dir=out_dir,
            resolved_inputs=resolved_inputs,
        )
        error_text = forced_error_text or (
            f"Pipeline failed with rc={returncode}. "
            f"Check diagnostics, stderr and output directory: {out_dir}"
        )

    metadata.update(
        {
            "status": status,
            "finished_at": _now_text(),
            "returncode": int(returncode),
            "diagnostics": diagnostics,
            "error_text": error_text,
        }
    )
    _write_run_metadata(run_root, metadata)

    _set_last_run_state(
        run_root=run_root,
        summary=summary,
        metadata=metadata,
        stdout_text=stdout_text,
        stderr_text=stderr_text,
        diagnostics=diagnostics,
        error_text=error_text,
    )
    return {
        "status": status,
        "error_text": error_text,
        "run_root": str(run_root),
    }


def _start_background_run(run_info: dict[str, Any], *, queue_source: str) -> None:
    run_root = Path(str(run_info["run_root"]))
    stdout_path = run_root / APP_STDOUT_NAME
    stderr_path = run_root / APP_STDERR_NAME
    metadata = _read_run_metadata(run_root)
    metadata.update(
        {
            "status": "running",
            "started_at": _now_text(),
            "queue_source": queue_source,
        }
    )

    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
        process = subprocess.Popen(
            run_info["command"],
            cwd=str(REPO_ROOT),
            stdout=stdout_handle,
            stderr=stderr_handle,
        )

    metadata["pid"] = int(process.pid)
    _write_run_metadata(run_root, metadata)
    active_run = dict(run_info)
    active_run["pid"] = int(process.pid)
    active_run["queue_source"] = queue_source
    st.session_state["active_process"] = process
    st.session_state["active_run_info"] = active_run
    st.session_state["last_scheduler_message"] = f"已启动运行: {run_info['run_name']}"


def _maybe_start_next_queued_run() -> bool:
    queue = st.session_state.get("run_queue")
    if not isinstance(queue, list) or not queue:
        st.session_state["queue_auto_run"] = False
        return False
    if st.session_state.get("active_process") is not None:
        return False

    next_run = queue.pop(0)
    st.session_state["run_queue"] = queue
    _start_background_run(next_run, queue_source="batch_queue")
    return True


def _sync_background_run_state() -> None:
    process = st.session_state.get("active_process")
    active_run = st.session_state.get("active_run_info")
    if process is None or not isinstance(active_run, dict):
        return

    returncode = process.poll()
    if returncode is None:
        return

    result = _finalize_background_run(
        run_info=active_run,
        returncode=int(returncode),
    )
    st.session_state["active_process"] = None
    st.session_state["active_run_info"] = None

    if result["status"] == "success":
        st.session_state["last_scheduler_message"] = f"运行完成: {active_run['run_name']}"
    elif result["status"] == "failed":
        st.session_state["last_scheduler_message"] = f"运行失败: {active_run['run_name']}"
    else:
        st.session_state["last_scheduler_message"] = f"运行结束: {active_run['run_name']}"

    if bool(st.session_state.get("queue_auto_run", False)):
        started = _maybe_start_next_queued_run()
        if started:
            st.session_state["last_scheduler_message"] += "；已自动启动队列中的下一项。"


def _stop_active_run() -> str:
    process = st.session_state.get("active_process")
    active_run = st.session_state.get("active_run_info")
    if process is None or not isinstance(active_run, dict):
        return "当前没有正在运行的任务。"

    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)

    returncode = int(process.poll() if process.poll() is not None else -1)
    _finalize_background_run(
        run_info=active_run,
        returncode=returncode,
        forced_status="cancelled",
        forced_error_text="Run cancelled by user.",
    )
    st.session_state["active_process"] = None
    st.session_state["active_run_info"] = None
    st.session_state["queue_auto_run"] = False
    message = f"已停止当前运行: {active_run['run_name']}；剩余队列已保留。"
    st.session_state["last_scheduler_message"] = message
    return message


def _build_run_queue_table(queue: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(queue, start=1):
        rows.append(
            {
                "序号": idx,
                "run_name": str(item.get("run_name") or ""),
                "start_mode": str(item.get("form_payload", {}).get("start_mode") or ""),
                "out_dir": str(item.get("out_dir") or ""),
            }
        )
    return pd.DataFrame(rows)


def _build_run_queue_option_label(item: dict[str, Any], index: int) -> str:
    start_mode = str((item.get("form_payload") or {}).get("start_mode") or "N/A")
    return f"{index + 1}. {str(item.get('run_name') or 'N/A')} | {start_mode}"


def _move_run_queue_item(
    queue: list[dict[str, Any]],
    *,
    selected_index: int,
    delta: int,
) -> tuple[list[dict[str, Any]], int]:
    if not queue:
        return queue, 0
    current_index = max(0, min(int(selected_index), len(queue) - 1))
    next_index = max(0, min(current_index + int(delta), len(queue) - 1))
    if next_index == current_index:
        return list(queue), current_index

    updated_queue = list(queue)
    item = updated_queue.pop(current_index)
    updated_queue.insert(next_index, item)
    return updated_queue, next_index


def _remove_run_queue_item(
    queue: list[dict[str, Any]],
    *,
    selected_index: int,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, int]:
    if not queue:
        return queue, None, 0
    current_index = max(0, min(int(selected_index), len(queue) - 1))
    updated_queue = list(queue)
    removed_item = updated_queue.pop(current_index)
    next_index = max(0, min(current_index, len(updated_queue) - 1)) if updated_queue else 0
    return updated_queue, removed_item, next_index


def _render_scheduler_panel() -> None:
    st.subheader("运行调度")
    scheduler_message = str(st.session_state.get("last_scheduler_message") or "")
    if scheduler_message:
        st.info(scheduler_message)

    refresh_col1, refresh_col2 = st.columns(2)
    refresh_clicked = refresh_col1.button("刷新运行状态", use_container_width=True)
    auto_queue_enabled = bool(st.session_state.get("queue_auto_run", False))
    refresh_col2.metric("队列自动续跑", "开" if auto_queue_enabled else "关")
    if refresh_clicked:
        st.rerun()

    active_run = st.session_state.get("active_run_info")
    if isinstance(active_run, dict):
        active_df = pd.DataFrame(
            [
                {
                    "当前状态": "running",
                    "run_name": str(active_run.get("run_name") or ""),
                    "来源": str(active_run.get("queue_source") or "direct"),
                    "PID": int(active_run.get("pid") or 0),
                    "输出目录": str(active_run.get("out_dir") or ""),
                }
            ]
        )
        st.dataframe(active_df, use_container_width=True, hide_index=True)
    else:
        st.caption("当前没有正在运行的任务。")

    queue = st.session_state.get("run_queue")
    if isinstance(queue, list) and queue:
        st.caption(f"当前队列长度: {len(queue)}")
        st.dataframe(_build_run_queue_table(queue), use_container_width=True, hide_index=True)
        queue_index_options = list(range(len(queue)))
        selected_queue_index = st.selectbox(
            "选择要调整的队列项",
            options=queue_index_options,
            format_func=lambda idx: _build_run_queue_option_label(queue[int(idx)], int(idx)),
            key="scheduler_selected_queue_index",
        )
        queue_action_col1, queue_action_col2, queue_action_col3 = st.columns(3)
        move_up_clicked = queue_action_col1.button(
            "上移选中项",
            use_container_width=True,
            disabled=int(selected_queue_index) <= 0,
            key="scheduler_move_queue_up",
        )
        move_down_clicked = queue_action_col2.button(
            "下移选中项",
            use_container_width=True,
            disabled=int(selected_queue_index) >= len(queue) - 1,
            key="scheduler_move_queue_down",
        )
        remove_clicked = queue_action_col3.button(
            "移除选中项",
            use_container_width=True,
            key="scheduler_remove_queue_item",
        )

        if move_up_clicked:
            updated_queue, next_index = _move_run_queue_item(
                queue,
                selected_index=int(selected_queue_index),
                delta=-1,
            )
            st.session_state["run_queue"] = updated_queue
            st.session_state["scheduler_selected_queue_index"] = next_index
            st.session_state["last_scheduler_message"] = (
                f"已上移队列项: {str(queue[int(selected_queue_index)].get('run_name') or 'N/A')}"
            )
            st.rerun()

        if move_down_clicked:
            updated_queue, next_index = _move_run_queue_item(
                queue,
                selected_index=int(selected_queue_index),
                delta=1,
            )
            st.session_state["run_queue"] = updated_queue
            st.session_state["scheduler_selected_queue_index"] = next_index
            st.session_state["last_scheduler_message"] = (
                f"已下移队列项: {str(queue[int(selected_queue_index)].get('run_name') or 'N/A')}"
            )
            st.rerun()

        if remove_clicked:
            updated_queue, removed_item, next_index = _remove_run_queue_item(
                queue,
                selected_index=int(selected_queue_index),
            )
            st.session_state["run_queue"] = updated_queue
            st.session_state["scheduler_selected_queue_index"] = next_index
            st.session_state["last_scheduler_message"] = (
                f"已移除队列项: {str((removed_item or {}).get('run_name') or 'N/A')}"
            )
            st.rerun()
    else:
        st.caption("当前队列为空。")


def _get_out_dir(summary: dict[str, Any] | None, metadata: dict[str, Any] | None = None) -> Path | None:
    if isinstance(summary, dict) and summary.get("out_dir"):
        return Path(str(summary["out_dir"]))
    if isinstance(metadata, dict) and metadata.get("out_dir"):
        return Path(str(metadata["out_dir"]))
    return None


def _resolve_output_file(
    *parts: str,
    summary: dict[str, Any] | None,
    metadata: dict[str, Any] | None = None,
) -> Path | None:
    out_dir = _get_out_dir(summary, metadata)
    if out_dir is None:
        return None
    return out_dir.joinpath(*parts)


def _format_size_text(path: Path) -> str:
    if not path.exists():
        return "N/A"
    if path.is_dir():
        try:
            child_count = len(list(path.iterdir()))
        except OSError:
            return "dir"
        return f"{child_count} items"

    size = float(path.stat().st_size)
    units = ["B", "KB", "MB", "GB"]
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return "N/A"


def _format_byte_count(byte_count: int) -> str:
    size = float(max(0, int(byte_count)))
    units = ["B", "KB", "MB", "GB"]
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return "N/A"


def _pick_default_view_columns(
    df: pd.DataFrame,
    *,
    preferred_columns: list[str] | None = None,
    max_columns: int = 12,
) -> list[str]:
    if df.empty:
        return []
    preferred = [column for column in (preferred_columns or []) if column in df.columns]
    if preferred:
        return preferred
    return [str(column) for column in df.columns[: min(int(max_columns), len(df.columns))]]


def _apply_dataframe_view(
    df: pd.DataFrame,
    *,
    visible_columns: list[str] | None = None,
    sort_column: str = "",
    descending: bool = True,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    working_df = df.copy()
    sort_key = str(sort_column or "").strip()
    if sort_key and sort_key in working_df.columns:
        working_df = working_df.sort_values(
            by=sort_key,
            ascending=not descending,
            kind="stable",
            na_position="last",
        )

    selected_columns = [str(column) for column in (visible_columns or []) if column in working_df.columns]
    if not selected_columns:
        selected_columns = [str(column) for column in working_df.columns]
    return working_df.loc[:, selected_columns]


def _list_numeric_filterable_columns(df: pd.DataFrame) -> list[str]:
    numeric_columns: list[str] = []
    for column in df.columns:
        series = pd.to_numeric(df[column], errors="coerce")
        if series.notna().any():
            numeric_columns.append(str(column))
    return numeric_columns


def _render_numeric_threshold_filters(
    df: pd.DataFrame,
    *,
    key_prefix: str,
    title: str = "数值阈值筛选",
) -> tuple[pd.DataFrame, list[str]]:
    if df.empty:
        return df.copy(), []

    numeric_columns = _list_numeric_filterable_columns(df)
    if not numeric_columns:
        return df.copy(), []

    summary_lines: list[str] = []
    selected_columns: list[str] = []
    filtered_df = df.copy()
    with st.expander(title, expanded=False):
        selected_columns = st.multiselect(
            "选择要启用阈值筛选的数值列",
            options=numeric_columns,
            default=[],
            key=f"{key_prefix}_threshold_columns",
        )
        if not selected_columns:
            st.caption("当前没有启用数值阈值筛选。")

        if selected_columns:
            mask = pd.Series(True, index=df.index)
            for column in selected_columns:
                series = pd.to_numeric(df[column], errors="coerce")
                finite_values = series.dropna()
                if finite_values.empty:
                    continue
                min_bound = float(finite_values.min())
                max_bound = float(finite_values.max())
                step_value = max(abs(max_bound - min_bound) / 100.0, 0.0001)
                filter_col1, filter_col2 = st.columns(2)
                lower_value = float(
                    filter_col1.number_input(
                        f"{column} 最小值",
                        value=min_bound,
                        step=step_value,
                        key=f"{key_prefix}_{column}_min",
                    )
                )
                upper_value = float(
                    filter_col2.number_input(
                        f"{column} 最大值",
                        value=max_bound,
                        step=step_value,
                        key=f"{key_prefix}_{column}_max",
                    )
                )
                applied_lower = min(lower_value, upper_value)
                applied_upper = max(lower_value, upper_value)
                if lower_value > upper_value:
                    st.caption(f"{column} 的上下界已自动交换为 [{_metric_text(applied_lower)}, {_metric_text(applied_upper)}]。")
                mask = mask & series.notna() & series.ge(applied_lower) & series.le(applied_upper)
                summary_lines.append(
                    f"{column} in [{_metric_text(applied_lower)}, {_metric_text(applied_upper)}]"
                )

            filtered_df = df.loc[mask].copy()
            st.caption(f"数值阈值筛选后保留 {len(filtered_df)} / {len(df)} 条。")

    return filtered_df, summary_lines


def _render_dataframe_view_controls(
    df: pd.DataFrame,
    *,
    key_prefix: str,
    preferred_columns: list[str] | None = None,
    default_sort_column: str = "",
    default_descending: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df.copy(), df.copy()

    default_columns = _pick_default_view_columns(df, preferred_columns=preferred_columns)
    visible_columns = st.multiselect(
        "显示列",
        options=[str(column) for column in df.columns],
        default=default_columns,
        key=f"{key_prefix}_visible_columns",
    )
    sort_options = [""] + [str(column) for column in df.columns]
    initial_sort_column = (
        str(default_sort_column)
        if str(default_sort_column or "") in df.columns
        else (str(default_columns[0]) if default_columns else "")
    )
    sort_column = st.selectbox(
        "排序列",
        options=sort_options,
        index=sort_options.index(initial_sort_column) if initial_sort_column in sort_options else 0,
        key=f"{key_prefix}_sort_column",
    )
    sort_descending = st.checkbox(
        "降序排列",
        value=bool(default_descending),
        key=f"{key_prefix}_sort_descending",
    )
    view_df = _apply_dataframe_view(
        df,
        visible_columns=[str(column) for column in visible_columns],
        sort_column=str(sort_column or ""),
        descending=bool(sort_descending),
    )
    return view_df, _apply_dataframe_view(
        df,
        visible_columns=[str(column) for column in df.columns],
        sort_column=str(sort_column or ""),
        descending=bool(sort_descending),
    )


def _count_label_stats(df: pd.DataFrame, label_col: str = "label") -> dict[str, Any]:
    if label_col not in df.columns:
        return {
            "has_label": False,
            "label_valid_count": 0,
            "label_class_count": 0,
            "label_compare_possible": False,
            "calibration_possible": False,
        }

    values = pd.to_numeric(df[label_col], errors="coerce")
    valid_values = values.dropna()
    if valid_values.empty:
        return {
            "has_label": True,
            "label_valid_count": 0,
            "label_class_count": 0,
            "label_compare_possible": False,
            "calibration_possible": False,
        }

    binary = (valid_values >= 0.5).astype(int)
    label_class_count = int(binary.nunique())
    label_valid_count = int(valid_values.shape[0])
    return {
        "has_label": True,
        "label_valid_count": label_valid_count,
        "label_class_count": label_class_count,
        "label_compare_possible": bool(label_valid_count > 0 and label_class_count >= 2),
        "calibration_possible": bool(label_valid_count >= 8 and label_class_count >= 2),
    }


def _clean_cell_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _unique_non_empty_count(df: pd.DataFrame, column: str) -> int:
    if column not in df.columns:
        return 0
    values = df[column].map(_clean_cell_text)
    values = values.loc[values.ne("")]
    return int(values.nunique(dropna=True))


def _duplicate_id_summary(df: pd.DataFrame) -> dict[str, int]:
    id_cols = [col for col in ["nanobody_id", "conformer_id", "pose_id"] if col in df.columns]
    if len(id_cols) < 3 or df.empty:
        return {
            "duplicate_id_key_count": 0,
            "duplicate_id_row_count": 0,
        }
    work = df.loc[:, id_cols].fillna("").astype(str)
    duplicate_mask = work.duplicated(keep=False)
    duplicate_keys = work.loc[duplicate_mask].drop_duplicates()
    return {
        "duplicate_id_key_count": int(len(duplicate_keys)),
        "duplicate_id_row_count": int(duplicate_mask.sum()),
    }


def _resolve_reference_path(raw_value: str, base_dir: Path | None) -> Path | None:
    text = _clean_cell_text(raw_value)
    if not text:
        return None
    path = Path(text).expanduser()
    if path.is_absolute():
        return path.resolve()
    if base_dir is None:
        return None
    return (base_dir / path).resolve()


def _build_default_source_info(
    *,
    local_path_text: str,
    uploaded_file: Any,
) -> dict[str, Any]:
    local_path = str(local_path_text or "").strip()
    if local_path:
        path = Path(local_path).expanduser().resolve()
        return {
            "available": True,
            "source": "local_path",
            "path_text": str(path),
            "exists": bool(path.exists()),
            "display": str(path),
        }
    if uploaded_file is not None:
        return {
            "available": True,
            "source": "uploaded_file",
            "path_text": "",
            "exists": True,
            "display": str(getattr(uploaded_file, "name", "uploaded file")),
        }
    return {
        "available": False,
        "source": "",
        "path_text": "",
        "exists": False,
        "display": "",
    }


def _summarize_path_column(
    df: pd.DataFrame,
    *,
    column: str,
    base_dir: Path | None,
    default_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    default_info = default_info if isinstance(default_info, dict) else {}
    default_available = bool(default_info.get("available"))
    default_source = str(default_info.get("source") or "")
    default_path_text = str(default_info.get("path_text") or "")
    default_exists = bool(default_info.get("exists"))

    row_value_count = 0
    covered_by_default_count = 0
    missing_value_count = 0
    existing_path_count = 0
    missing_path_count = 0
    unchecked_relative_count = 0
    sample_missing_paths: list[str] = []
    sample_unchecked_relative_paths: list[str] = []

    if column not in df.columns:
        missing_value_count = 0 if default_available else int(len(df))
        covered_by_default_count = int(len(df)) if default_available else 0
        if default_source == "local_path":
            if default_exists:
                existing_path_count = covered_by_default_count
            else:
                missing_path_count = covered_by_default_count
                if default_path_text:
                    sample_missing_paths.append(default_path_text)
        return {
            "column": column,
            "column_present": False,
            "total_rows": int(len(df)),
            "row_value_count": 0,
            "covered_by_default_count": covered_by_default_count,
            "missing_value_count": missing_value_count,
            "existing_path_count": existing_path_count,
            "missing_path_count": missing_path_count,
            "unchecked_relative_path_count": unchecked_relative_count,
            "default_source": default_source,
            "default_display": str(default_info.get("display") or ""),
            "sample_missing_paths": sample_missing_paths,
            "sample_unchecked_relative_paths": sample_unchecked_relative_paths,
        }

    for raw in df[column].tolist():
        value = _clean_cell_text(raw)
        used_default = False
        if not value and default_available:
            value = default_path_text
            used_default = True
            covered_by_default_count += 1

        if not value:
            missing_value_count += 1
            continue

        if not used_default:
            row_value_count += 1

        path = _resolve_reference_path(value, base_dir)
        if path is None:
            unchecked_relative_count += 1
            if len(sample_unchecked_relative_paths) < 5:
                sample_unchecked_relative_paths.append(value)
            continue
        if path.exists():
            existing_path_count += 1
        else:
            missing_path_count += 1
            if len(sample_missing_paths) < 5:
                sample_missing_paths.append(str(path))

    return {
        "column": column,
        "column_present": True,
        "total_rows": int(len(df)),
        "row_value_count": int(row_value_count),
        "covered_by_default_count": int(covered_by_default_count),
        "missing_value_count": int(missing_value_count),
        "existing_path_count": int(existing_path_count),
        "missing_path_count": int(missing_path_count),
        "unchecked_relative_path_count": int(unchecked_relative_count),
        "default_source": default_source,
        "default_display": str(default_info.get("display") or ""),
        "sample_missing_paths": sample_missing_paths,
        "sample_unchecked_relative_paths": sample_unchecked_relative_paths,
    }


def _build_batch_profile(
    df: pd.DataFrame,
    *,
    mode: str,
    base_dir: Path | None = None,
    default_path_info: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    duplicate_summary = _duplicate_id_summary(df)
    profile: dict[str, Any] = {
        "mode": mode,
        "row_count": int(len(df)),
        "unique_nanobody_count": _unique_non_empty_count(df, "nanobody_id"),
        "unique_conformer_count": _unique_non_empty_count(df, "conformer_id"),
        "unique_pose_count": _unique_non_empty_count(df, "pose_id"),
        **duplicate_summary,
    }

    if mode == "input_csv":
        default_path_info = default_path_info or {}
        profile["unique_pdb_path_count"] = _unique_non_empty_count(df, "pdb_path")
        profile["path_checks"] = [
            _summarize_path_column(
                df,
                column="pdb_path",
                base_dir=base_dir,
                default_info={"available": False},
            ),
            _summarize_path_column(
                df,
                column="pocket_file",
                base_dir=base_dir,
                default_info=default_path_info.get("pocket_file"),
            ),
            _summarize_path_column(
                df,
                column="catalytic_file",
                base_dir=base_dir,
                default_info=default_path_info.get("catalytic_file"),
            ),
            _summarize_path_column(
                df,
                column="ligand_file",
                base_dir=base_dir,
                default_info=default_path_info.get("ligand_file"),
            ),
        ]
    else:
        numeric_cols = []
        for col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            if int(series.notna().sum()) > 0:
                numeric_cols.append(str(col))
        profile["numeric_column_count"] = int(len(numeric_cols))
        profile["numeric_columns_preview"] = numeric_cols[:12]
        profile["has_pred_prob"] = bool("pred_prob" in df.columns)
        profile["has_rule_score"] = bool("final_rule_score" in df.columns or "rule_blocking_score" in df.columns)

    return profile


def _build_processing_plan(
    *,
    start_mode: str,
    csv_summary: dict[str, Any],
    form_payload: dict[str, Any],
) -> dict[str, Any]:
    row_count = int(csv_summary.get("row_count") or 0)
    label_col = str(csv_summary.get("label_col") or form_payload.get("label_col") or "label")
    label_compare_possible = bool(csv_summary.get("label_compare_possible"))
    calibration_possible = bool(csv_summary.get("calibration_possible"))
    disable_label_aware = bool(form_payload.get("disable_label_aware_steps", False))

    stages: list[dict[str, str]] = []
    if start_mode == "input_csv":
        stages.append(
            {
                "阶段": "build_feature_table",
                "作用": f"逐行解析 {row_count} 条 pose 输入并生成 pose_features.csv",
            }
        )
    else:
        stages.append(
            {
                "阶段": "build_feature_table",
                "作用": "跳过；直接使用已提供的 pose_features.csv",
            }
        )
    stages.extend(
        [
            {"阶段": "rule_ranker", "作用": "生成规则版 pose/conformer/nanobody 排名"},
            {"阶段": "train_pose_model", "作用": "训练或拟合 ML pose 模型并输出 pose_predictions.csv"},
            {"阶段": "rank_nanobodies", "作用": "聚合 ML pose 预测为 nanobody 排名"},
            {"阶段": "build_consensus_ranking", "作用": "合并 Rule、ML 与 QC 风险，生成共识排名"},
        ]
    )
    if disable_label_aware:
        label_plan = "用户禁用了 label-aware 步骤；compare/calibration/strategy optimize 不执行。"
    elif not label_compare_possible:
        label_plan = f"`{label_col}` 不足或类别不足；compare/calibration/strategy optimize 会自动跳过。"
    elif calibration_possible:
        label_plan = f"`{label_col}` 满足要求；将执行 rule-vs-ML compare、calibration 和 strategy optimize。"
        stages.extend(
            [
                {"阶段": "compare_rule_vs_ml", "作用": "对比 Rule 与 ML 排名一致性"},
                {"阶段": "calibrate_rule_ranker", "作用": f"用 `{label_col}` 自动校准规则权重"},
                {"阶段": "strategy_optimization", "作用": "生成下一轮校准推荐策略"},
            ]
        )
    else:
        label_plan = f"`{label_col}` 可用于 compare，但不足 8 条；calibration 会自动跳过。"
        stages.append({"阶段": "compare_rule_vs_ml", "作用": "对比 Rule 与 ML 排名一致性"})

    return {
        "start_mode": start_mode,
        "row_count": row_count,
        "top_k": int(form_payload.get("top_k", 3) or 3),
        "skip_failed_rows": bool(form_payload.get("skip_failed_rows", True)),
        "label_plan": label_plan,
        "stages": stages,
        "expected_key_outputs": [
            "recommended_pipeline_summary.json",
            "recommended_pipeline_report.md",
            "rule_outputs/nanobody_rule_ranking.csv",
            "model_outputs/pose_predictions.csv",
            "ml_ranking_outputs/nanobody_ranking.csv",
            "consensus_outputs/consensus_ranking.csv",
            "parameter_sensitivity/parameter_sensitivity_report.md",
            "parameter_sensitivity/candidate_rank_sensitivity.csv",
            "candidate_report_cards/index.html",
            "candidate_report_cards.zip",
            "candidate_comparisons/candidate_comparison_report.md",
            "experiment_suggestions/next_experiment_suggestions.csv",
        ],
    }


def _summarize_csv_dataframe(
    df: pd.DataFrame,
    *,
    required_columns: list[str],
    optional_columns: list[str] | None = None,
    mode: str,
    label_col: str = "label",
    base_dir: Path | None = None,
    default_path_info: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    missing_required = [col for col in required_columns if col not in df.columns]
    optional_present = [col for col in (optional_columns or []) if col in df.columns]
    preview_df = df.head(5).copy()
    preview_df.columns = [str(col) for col in preview_df.columns]
    label_stats = _count_label_stats(df, label_col=label_col)
    return {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "columns_preview": [str(col) for col in df.columns[:12].tolist()],
        "label_col": str(label_col or "label"),
        "missing_required_columns": missing_required,
        "optional_present_columns": optional_present,
        "preview_rows": preview_df.to_dict(orient="records"),
        "batch_profile": _build_batch_profile(
            df,
            mode=mode,
            base_dir=base_dir,
            default_path_info=default_path_info,
        ),
        **label_stats,
    }


def _read_local_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def _inspect_csv_source(
    *,
    uploaded_file: Any,
    local_path_text: str,
    required_columns: list[str],
    optional_columns: list[str] | None = None,
    mode: str,
    label_col: str = "label",
    default_path_info: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    local_path = str(local_path_text or "").strip()
    if local_path:
        path = Path(local_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"路径不存在: {path}")
        df = _read_local_csv(path)
        size_text = _format_size_text(path)
        summary = _summarize_csv_dataframe(
            df,
            required_columns=required_columns,
            optional_columns=optional_columns,
            mode=mode,
            label_col=label_col,
            base_dir=path.parent.resolve(),
            default_path_info=default_path_info,
        )
        summary.update(
            {
                "source": "本地路径",
                "display_name": path.name,
                "path": str(path),
                "size_text": size_text,
            }
        )
        if mode == "input_csv" and not summary.get("missing_required_columns"):
            batch_profile = summary.get("batch_profile") if isinstance(summary.get("batch_profile"), dict) else {}
            path_checks = batch_profile.get("path_checks") if isinstance(batch_profile.get("path_checks"), list) else []
            needs_path_repair = any(
                int(item.get("missing_path_count") or 0) > 0
                or int(item.get("unchecked_relative_path_count") or 0) > 0
                or (str(item.get("column") or "") == "pdb_path" and int(item.get("missing_value_count") or 0) > 0)
                for item in path_checks
                if isinstance(item, dict)
            )
            if needs_path_repair:
                repair_search_roots = [path.parent.resolve()]
                parent_root = path.parent.parent.resolve()
                if parent_root != path.parent.resolve() and str(parent_root) != parent_root.anchor:
                    repair_search_roots.append(parent_root)
                path_repair = analyze_input_path_repair_dataframe(
                    df,
                    base_dir=path.parent.resolve(),
                    search_roots=repair_search_roots,
                )
                summary["path_repair"] = path_repair
        return summary

    if uploaded_file is None:
        raise ValueError("当前没有可检查的 CSV 输入。")

    file_bytes = bytes(uploaded_file.getbuffer())
    df = pd.read_csv(BytesIO(file_bytes), low_memory=False)
    summary = _summarize_csv_dataframe(
        df,
        required_columns=required_columns,
        optional_columns=optional_columns,
        mode=mode,
        label_col=label_col,
        base_dir=None,
        default_path_info=default_path_info,
    )
    summary.update(
        {
            "source": "上传文件",
            "display_name": str(uploaded_file.name),
            "path": None,
            "size_text": _format_byte_count(len(file_bytes)),
        }
    )
    return summary


def _build_source_status_row(
    *,
    label: str,
    uploaded_file: Any,
    local_path_text: str,
    required: bool,
) -> dict[str, Any]:
    local_path = str(local_path_text or "").strip()
    if local_path:
        path = Path(local_path).expanduser().resolve()
        return {
            "输入项": label,
            "来源": "本地路径",
            "状态": "就绪" if path.exists() else "路径不存在",
            "详情": str(path),
        }

    if uploaded_file is not None:
        upload_size = getattr(uploaded_file, "size", None)
        size_text = _format_byte_count(int(upload_size)) if upload_size is not None else "上传文件"
        return {
            "输入项": label,
            "来源": "上传文件",
            "状态": "就绪",
            "详情": f"{uploaded_file.name} ({size_text})",
        }

    return {
        "输入项": label,
        "来源": "未提供",
        "状态": "缺失" if required else "可选",
        "详情": "",
    }


def _choose_bundle_candidate(
    root: Path,
    *,
    exact_names: list[str],
    contains_text: str | None = None,
    suffixes: set[str] | None = None,
) -> Path | None:
    exact = {name.lower() for name in exact_names}
    contains = None if contains_text is None else contains_text.lower()
    candidates: list[tuple[int, int, int, str, Path]] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        suffix = path.suffix.lower()
        if suffixes and suffix not in suffixes:
            continue

        priority: int | None = None
        if name in exact:
            priority = 0
        elif contains and contains in name:
            priority = 1

        if priority is None:
            continue

        relative_depth = len(path.relative_to(root).parts)
        candidates.append((priority, relative_depth, len(name), str(path).lower(), path))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return candidates[0][-1]


def _is_probable_pose_pdb(path: Path) -> bool:
    if path.suffix.lower() not in AUTO_INPUT_PDB_SUFFIXES:
        return False
    name = path.name.lower()
    if name.endswith("_atm.pdb"):
        return False
    return not any(token in name for token in AUTO_INPUT_EXCLUDE_NAME_TOKENS)


def _derive_auto_pose_ids(path: Path, index: int) -> dict[str, str]:
    stem = path.stem.strip()
    tokens = [tok for tok in re.split(r"[_\-\s]+", stem) if tok]
    if len(tokens) >= 3:
        nanobody_id = tokens[0]
        conformer_id = tokens[1]
        pose_id = "_".join(tokens[2:])
    elif len(tokens) == 2:
        nanobody_id = tokens[0]
        conformer_id = tokens[1]
        pose_id = stem
    elif len(tokens) == 1:
        nanobody_id = tokens[0]
        conformer_id = "conf_01"
        pose_id = stem
    else:
        nanobody_id = f"NB{index:03d}"
        conformer_id = "conf_01"
        pose_id = f"pose_{index:03d}"
    return {
        "nanobody_id": nanobody_id,
        "conformer_id": conformer_id,
        "pose_id": pose_id,
    }


def _build_auto_input_table(
    *,
    bundle_root: Path,
    detected_paths: dict[str, str],
    output_root: Path,
) -> dict[str, Any] | None:
    if detected_paths.get("input_csv_local_path") or detected_paths.get("feature_csv_local_path"):
        return None

    pdb_candidates = sorted(
        [path for path in bundle_root.rglob("*") if path.is_file() and _is_probable_pose_pdb(path)],
        key=lambda p: str(p.relative_to(bundle_root)).lower(),
    )
    if not pdb_candidates:
        return None

    output_root.mkdir(parents=True, exist_ok=True)
    out_csv = output_root / "auto_input_pose_table.csv"

    default_pocket = detected_paths.get("default_pocket_local_path", "")
    default_catalytic = detected_paths.get("default_catalytic_local_path", "")
    default_ligand = detected_paths.get("default_ligand_local_path", "")

    rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str]] = set()
    for index, pdb_path in enumerate(pdb_candidates, start=1):
        ids = _derive_auto_pose_ids(pdb_path, index)
        key = (ids["nanobody_id"], ids["conformer_id"], ids["pose_id"])
        if key in seen_keys:
            ids["pose_id"] = f"{ids['pose_id']}_{index:03d}"
            key = (ids["nanobody_id"], ids["conformer_id"], ids["pose_id"])
        seen_keys.add(key)

        row = {
            **ids,
            "pdb_path": str(pdb_path.resolve()),
            "pocket_file": str(Path(default_pocket).resolve()) if default_pocket else "",
            "catalytic_file": str(Path(default_catalytic).resolve()) if default_catalytic else "",
            "ligand_file": str(Path(default_ligand).resolve()) if default_ligand else "",
        }
        rows.append(row)

    df = pd.DataFrame(rows, columns=["nanobody_id", "conformer_id", "pose_id", "pdb_path", "pocket_file", "catalytic_file", "ligand_file"])
    df.to_csv(out_csv, index=False)
    return {
        "generated_csv_path": str(out_csv.resolve()),
        "row_count": int(len(df)),
        "pose_pdb_count": int(len(pdb_candidates)),
        "default_pocket_file": default_pocket,
        "default_catalytic_file": default_catalytic,
        "default_ligand_file": default_ligand,
        "preview_rows": df.head(8).to_dict(orient="records"),
    }


def _scan_input_bundle_root(
    *,
    bundle_root: Path,
    source_name: str,
    source_kind: str,
    source_path: str,
    import_root: Path | None = None,
) -> dict[str, Any]:
    bundle_root = bundle_root.expanduser().resolve()
    if not bundle_root.exists():
        raise FileNotFoundError(f"导入目录不存在: {bundle_root}")
    if not bundle_root.is_dir():
        raise NotADirectoryError(f"导入目录不是文件夹: {bundle_root}")

    detected_paths: dict[str, str] = {}
    for key, exact_names, contains_text, suffixes in BUNDLE_DETECTION_RULES:
        detected_path = _choose_bundle_candidate(
            bundle_root,
            exact_names=exact_names,
            contains_text=contains_text,
            suffixes=suffixes,
        )
        if detected_path is not None:
            detected_paths[key] = str(detected_path.resolve())

    generated_root = import_root if import_root is not None else GENERATED_INPUT_ROOT / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_slugify_name(source_name)}"
    generated_input_table = _build_auto_input_table(
        bundle_root=bundle_root,
        detected_paths=detected_paths,
        output_root=generated_root,
    )
    if generated_input_table is not None:
        detected_paths["input_csv_local_path"] = str(generated_input_table["generated_csv_path"])

    bundle_files = sorted([path for path in bundle_root.rglob("*") if path.is_file()])
    preview_items = [
        {"文件": str(path.relative_to(bundle_root))}
        for path in bundle_files[:12]
    ]
    return {
        "source_name": source_name,
        "source_kind": source_kind,
        "source_path": source_path,
        "import_root": None if import_root is None else str(import_root.resolve()),
        "bundle_root": str(bundle_root),
        "file_count": len(bundle_files),
        "detected_paths": detected_paths,
        "detected_items": [
            {"字段": key, "识别结果": value}
            for key, value in detected_paths.items()
        ],
        "bundle_preview_items": preview_items,
        "generated_input_table": generated_input_table,
    }


def _import_input_bundle(*, uploaded_file: Any, local_path_text: str) -> dict[str, Any]:
    local_path = str(local_path_text or "").strip()
    source_name = ""
    zip_path: Path | None = None
    zip_bytes: bytes | None = None

    if local_path:
        zip_path = Path(local_path).expanduser().resolve()
        if not zip_path.exists():
            raise FileNotFoundError(f"数据包路径不存在: {zip_path}")
        source_name = zip_path.name
    elif uploaded_file is not None:
        source_name = str(uploaded_file.name)
        zip_bytes = bytes(uploaded_file.getbuffer())
    else:
        raise ValueError("请先上传 zip 数据包，或填写 zip 本地路径。")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    import_root = BUNDLE_IMPORT_ROOT / f"{stamp}_{_slugify_name(Path(source_name).stem)}"
    extracted_root = import_root / "extracted"
    import_root.mkdir(parents=True, exist_ok=True)
    extracted_root.mkdir(parents=True, exist_ok=True)

    if zip_path is None:
        zip_path = import_root / source_name
        zip_path.write_bytes(zip_bytes or b"")

    try:
        with zipfile.ZipFile(zip_path, mode="r") as zf:
            zf.extractall(extracted_root)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"无效 zip 文件: {zip_path}") from exc

    report = _scan_input_bundle_root(
        bundle_root=extracted_root,
        source_name=source_name,
        source_kind="zip",
        source_path=str(zip_path),
        import_root=import_root,
    )
    report["source_zip"] = str(zip_path)
    report["extracted_root"] = str(extracted_root.resolve())
    return report


def _import_input_directory(local_dir_text: str) -> dict[str, Any]:
    directory_path = Path(str(local_dir_text or "").strip()).expanduser().resolve()
    if not directory_path.exists():
        raise FileNotFoundError(f"数据目录不存在: {directory_path}")
    if not directory_path.is_dir():
        raise NotADirectoryError(f"数据目录不是文件夹: {directory_path}")

    return _scan_input_bundle_root(
        bundle_root=directory_path,
        source_name=directory_path.name,
        source_kind="directory",
        source_path=str(directory_path),
        import_root=None,
    )


def _import_bundle_source(
    *,
    uploaded_file: Any,
    zip_local_path_text: str,
    directory_local_path_text: str,
) -> dict[str, Any]:
    zip_local_path = str(zip_local_path_text or "").strip()
    directory_local_path = str(directory_local_path_text or "").strip()

    if directory_local_path and zip_local_path:
        raise ValueError("zip 本地路径和目录路径请只填写一种。")

    if directory_local_path:
        return _import_input_directory(directory_local_path)
    if zip_local_path:
        return _import_input_bundle(uploaded_file=None, local_path_text=zip_local_path)
    if uploaded_file is not None:
        return _import_input_bundle(uploaded_file=uploaded_file, local_path_text="")

    if uploaded_file is None:
        raise ValueError("请先提供 zip 数据包，或填写本地数据目录路径。")
    raise ValueError("无法识别当前导入来源。")


def _apply_bundle_import(report: dict[str, Any]) -> str:
    detected_paths = report.get("detected_paths") if isinstance(report.get("detected_paths"), dict) else {}
    if not detected_paths:
        return "已导入数据源，但没有自动识别到 input/feature/default 文件，请手动填写路径。"

    for key, value in detected_paths.items():
        if value:
            st.session_state[key] = value

    if detected_paths.get("input_csv_local_path") and not detected_paths.get("feature_csv_local_path"):
        st.session_state["start_mode"] = "input_csv"
    elif detected_paths.get("feature_csv_local_path") and not detected_paths.get("input_csv_local_path"):
        st.session_state["start_mode"] = "feature_csv"

    mapped_labels = {
        "input_csv_local_path": "input_csv",
        "feature_csv_local_path": "feature_csv",
        "default_pocket_local_path": "default_pocket_file",
        "default_catalytic_local_path": "default_catalytic_file",
        "default_ligand_local_path": "default_ligand_file",
    }
    detected_names = [mapped_labels.get(key, key) for key in detected_paths]
    source_kind_text = "目录" if str(report.get("source_kind") or "") == "directory" else "zip 数据包"
    generated_input_table = (
        report.get("generated_input_table")
        if isinstance(report.get("generated_input_table"), dict)
        else None
    )
    generated_text = ""
    if generated_input_table is not None:
        generated_text = f"；已自动生成 input_pose_table.csv（{int(generated_input_table.get('row_count') or 0)} 行）"
    return (
        f"已导入{source_kind_text} {report.get('source_name', '')}，自动识别 {len(detected_paths)} 个输入项："
        + ", ".join(detected_names)
        + generated_text
    )


def _build_runtime_dependency_report(start_mode: str) -> dict[str, Any]:
    return check_runtime_dependencies(
        get_pipeline_runtime_dependency_specs(start_mode),
        python_executable=sys.executable,
    )


def _ensure_runtime_dependencies_ready(start_mode: str) -> None:
    dependency_report = _build_runtime_dependency_report(start_mode)
    error_message = str(dependency_report.get("error_message") or "").strip()
    if error_message:
        raise RuntimeError(f"运行环境依赖预检失败: {error_message}")

    missing_dependencies = (
        dependency_report.get("missing_dependencies")
        if isinstance(dependency_report.get("missing_dependencies"), list)
        else []
    )
    if missing_dependencies:
        summary_text = format_missing_dependency_summary(missing_dependencies)
        raise RuntimeError(
            "当前运行环境缺少关键 Python 依赖: "
            f"{summary_text}。先执行 `pip install -r requirements.txt`，再重新运行。"
        )


def _build_preflight_report(
    *,
    start_mode: str,
    input_csv_upload: Any,
    feature_csv_upload: Any,
    default_pocket_upload: Any,
    default_catalytic_upload: Any,
    default_ligand_upload: Any,
    experiment_plan_override_upload: Any,
    form_payload: dict[str, Any],
) -> dict[str, Any]:
    status = "ready"
    messages: list[str] = []
    label_col = str(form_payload.get("label_col") or "label").strip() or "label"
    dependency_report = _build_runtime_dependency_report(start_mode)
    default_path_info = {
        "pocket_file": _build_default_source_info(
            local_path_text=str(form_payload.get("default_pocket_local_path") or ""),
            uploaded_file=default_pocket_upload,
        ),
        "catalytic_file": _build_default_source_info(
            local_path_text=str(form_payload.get("default_catalytic_local_path") or ""),
            uploaded_file=default_catalytic_upload,
        ),
        "ligand_file": _build_default_source_info(
            local_path_text=str(form_payload.get("default_ligand_local_path") or ""),
            uploaded_file=default_ligand_upload,
        ),
    }

    optional_file_rows = [
        _build_source_status_row(
            label="default_pocket_file",
            uploaded_file=default_pocket_upload,
            local_path_text=str(form_payload.get("default_pocket_local_path") or ""),
            required=False,
        ),
        _build_source_status_row(
            label="default_catalytic_file",
            uploaded_file=default_catalytic_upload,
            local_path_text=str(form_payload.get("default_catalytic_local_path") or ""),
            required=False,
        ),
        _build_source_status_row(
            label="default_ligand_file",
            uploaded_file=default_ligand_upload,
            local_path_text=str(form_payload.get("default_ligand_local_path") or ""),
            required=False,
        ),
        _build_source_status_row(
            label="experiment_plan_override_csv",
            uploaded_file=experiment_plan_override_upload,
            local_path_text=str(form_payload.get("experiment_plan_override_local_path") or ""),
            required=False,
        ),
    ]

    if start_mode == "input_csv":
        csv_summary = _inspect_csv_source(
            uploaded_file=input_csv_upload,
            local_path_text=str(form_payload.get("input_csv_local_path") or ""),
            required_columns=INPUT_CSV_REQUIRED_COLUMNS,
            optional_columns=list(dict.fromkeys(INPUT_CSV_OPTIONAL_HINT_COLUMNS + [label_col])),
            mode="input_csv",
            label_col=label_col,
            default_path_info=default_path_info,
        )
        missing_required = csv_summary["missing_required_columns"]
        if missing_required:
            status = "error"
            messages.append(f"input_csv 缺少必需列: {missing_required}")
        else:
            messages.append("input_csv 必需列完整，可进入 build_feature_table 阶段。")

        missing_geometry_hints: list[str] = []
        for col_name, field_name in [
            ("pocket_file", "default_pocket_local_path"),
            ("catalytic_file", "default_catalytic_local_path"),
            ("ligand_file", "default_ligand_local_path"),
        ]:
            has_csv_col = col_name in csv_summary["optional_present_columns"]
            uploaded_default_file = None
            if field_name == "default_pocket_local_path":
                uploaded_default_file = default_pocket_upload
            elif field_name == "default_catalytic_local_path":
                uploaded_default_file = default_catalytic_upload
            elif field_name == "default_ligand_local_path":
                uploaded_default_file = default_ligand_upload
            has_default_file = _source_present(uploaded_default_file, str(form_payload.get(field_name) or ""))
            if not has_csv_col and not has_default_file:
                missing_geometry_hints.append(col_name)

        if missing_geometry_hints and status != "error":
            status = "warning"
            messages.append(
                "当前未在 input_csv 中发现这些列，也未提供对应默认文件："
                f"{missing_geometry_hints}；相关几何特征可能为空或依赖行内数据补全。"
            )
        elif not missing_geometry_hints:
            messages.append("pocket/catalytic/ligand 输入已从 CSV 列或默认文件中至少覆盖一套来源。")

        batch_profile = csv_summary.get("batch_profile") if isinstance(csv_summary.get("batch_profile"), dict) else {}
        duplicate_rows = int(batch_profile.get("duplicate_id_row_count") or 0)
        if duplicate_rows > 0 and status != "error":
            status = "warning"
            messages.append(f"发现 {duplicate_rows} 行 nanobody/conformer/pose ID 组合重复，建议确认是否为预期重复 pose。")

        path_checks = batch_profile.get("path_checks") if isinstance(batch_profile.get("path_checks"), list) else []
        for path_check in path_checks:
            if not isinstance(path_check, dict):
                continue
            column = str(path_check.get("column") or "")
            missing_values = int(path_check.get("missing_value_count") or 0)
            missing_paths = int(path_check.get("missing_path_count") or 0)
            unchecked_relative = int(path_check.get("unchecked_relative_path_count") or 0)
            if column == "pdb_path" and (missing_values > 0 or missing_paths > 0):
                status = "error"
                messages.append(
                    f"pdb_path 有 {missing_values} 行未填写、{missing_paths} 行路径不存在；这些行无法构建结构特征。"
                )
            elif column != "pdb_path" and missing_paths > 0 and status != "error":
                status = "warning"
                messages.append(f"{column} 有 {missing_paths} 行引用的文件路径不存在，相关几何特征可能为空。")
            if unchecked_relative > 0 and status != "error":
                status = "warning"
                messages.append(
                    f"{column} 有 {unchecked_relative} 行相对路径暂无法在检查阶段确认；"
                    "如果这是上传 CSV，请优先使用 zip/目录导入以保留相对路径上下文。"
                )
        path_repair = csv_summary.get("path_repair") if isinstance(csv_summary.get("path_repair"), dict) else {}
        repair_summary = path_repair.get("summary") if isinstance(path_repair.get("summary"), dict) else {}
        repairable_count = int(repair_summary.get("auto_repair_count") or 0)
        suggested_count = int(repair_summary.get("suggested_replacement_count") or 0)
        if repairable_count > 0:
            messages.append(
                f"已自动定位到 {repairable_count} 条高可信路径修复建议，可在下方下载修复版 input_csv。"
            )
        elif suggested_count > 0 and status != "error":
            status = "warning"
            messages.append(f"发现 {suggested_count} 条路径候选建议，但可信度不足，需要人工确认。")
    else:
        csv_summary = _inspect_csv_source(
            uploaded_file=feature_csv_upload,
            local_path_text=str(form_payload.get("feature_csv_local_path") or ""),
            required_columns=FEATURE_CSV_REQUIRED_COLUMNS,
            optional_columns=list(dict.fromkeys(FEATURE_CSV_OPTIONAL_HINT_COLUMNS + [label_col])),
            mode="feature_csv",
            label_col=label_col,
        )
        missing_required = csv_summary["missing_required_columns"]
        if missing_required:
            status = "error"
            messages.append(f"pose_features.csv 缺少必需列: {missing_required}")
        else:
            messages.append("pose_features.csv 最小 ID 列完整，可直接进入 rule/ML 排名阶段。")

        batch_profile = csv_summary.get("batch_profile") if isinstance(csv_summary.get("batch_profile"), dict) else {}
        duplicate_rows = int(batch_profile.get("duplicate_id_row_count") or 0)
        if duplicate_rows > 0 and status != "error":
            status = "warning"
            messages.append(f"发现 {duplicate_rows} 行 nanobody/conformer/pose ID 组合重复，建议确认是否为预期重复 pose。")

        if not csv_summary.get("has_label"):
            if status != "error":
                status = "warning"
            messages.append(f"当前没有 `{label_col}` 列，compare/calibration/strategy optimize 会被自动跳过。")
        elif not csv_summary.get("label_compare_possible"):
            if status != "error":
                status = "warning"
            messages.append(f"`{label_col}` 有效样本不足或类别退化，label-aware 步骤会被自动跳过。")
        else:
            messages.append(
                f"检测到 {csv_summary['label_valid_count']} 条有效 `{label_col}`，"
                f"类别数 {csv_summary['label_class_count']}。"
            )
            if csv_summary.get("calibration_possible"):
                messages.append("当前 label 数量已满足自动校准的最小要求。")
            else:
                if status != "error":
                    status = "warning"
                messages.append("当前 label 还不足 8 条，compare 可执行，但 calibration 会被自动跳过。")

    dependency_error = str(dependency_report.get("error_message") or "").strip()
    missing_dependencies = (
        dependency_report.get("missing_dependencies")
        if isinstance(dependency_report.get("missing_dependencies"), list)
        else []
    )
    if dependency_error:
        status = "error"
        messages.append(f"运行环境依赖预检失败: {dependency_error}")
    elif missing_dependencies:
        status = "error"
        messages.append(
            "当前运行环境缺少关键 Python 依赖："
            + format_missing_dependency_summary(missing_dependencies)
            + "。"
        )
        messages.append("建议先执行 `pip install -r requirements.txt`，再重新开始运行或入队。")
    else:
        messages.append("当前运行环境已满足本次流程的关键 Python 依赖。")

    return {
        "checked_at": _now_text(),
        "start_mode": start_mode,
        "status": status,
        "messages": messages,
        "main_csv": csv_summary,
        "processing_plan": _build_processing_plan(
            start_mode=start_mode,
            csv_summary=csv_summary,
            form_payload=form_payload,
        ),
        "optional_files": optional_file_rows,
        "runtime_dependencies": dependency_report,
    }


def _execute_preflight_check(
    *,
    start_mode: str,
    input_csv_upload: Any,
    feature_csv_upload: Any,
    default_pocket_upload: Any,
    default_catalytic_upload: Any,
    default_ligand_upload: Any,
    experiment_plan_override_upload: Any,
    form_payload: dict[str, Any],
) -> dict[str, Any]:
    preflight_report = _build_preflight_report(
        start_mode=start_mode,
        input_csv_upload=input_csv_upload,
        feature_csv_upload=feature_csv_upload,
        default_pocket_upload=default_pocket_upload,
        default_catalytic_upload=default_catalytic_upload,
        default_ligand_upload=default_ligand_upload,
        experiment_plan_override_upload=experiment_plan_override_upload,
        form_payload=form_payload,
    )
    st.session_state["last_preflight"] = preflight_report
    return preflight_report


def _render_preflight_report(report: dict[str, Any] | None) -> None:
    st.subheader("运行前检查")
    if not isinstance(report, dict):
        st.info("可在左侧点击“检查当前输入”，先确认必需列、label 状态和默认文件来源。")
        return

    checked_at = str(report.get("checked_at") or "")
    if checked_at:
        st.caption(f"最近一次检查时间: {checked_at}。如果你刚修改了输入，请重新点击“检查当前输入”。")

    status = str(report.get("status") or "warning")
    messages = report.get("messages") if isinstance(report.get("messages"), list) else []
    if status == "ready":
        st.success("运行前检查通过。")
    elif status == "warning":
        st.warning("运行前检查存在提醒项。")
    else:
        st.error("运行前检查发现阻塞问题。")

    for message in messages:
        st.write(f"- {message}")

    main_csv = report.get("main_csv") if isinstance(report.get("main_csv"), dict) else {}
    if main_csv:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("主输入来源", str(main_csv.get("source") or "N/A"))
        c2.metric("主输入大小", str(main_csv.get("size_text") or "N/A"))
        c3.metric("CSV 行数", int(main_csv.get("row_count") or 0))
        c4.metric("CSV 列数", int(main_csv.get("column_count") or 0))

        st.caption(
            "字段预览: "
            + ", ".join([str(col) for col in main_csv.get("columns_preview", [])])
        )

        missing_required = main_csv.get("missing_required_columns") if isinstance(main_csv.get("missing_required_columns"), list) else []
        if missing_required:
            st.error(f"缺少必需列: {missing_required}")

        optional_present = main_csv.get("optional_present_columns") if isinstance(main_csv.get("optional_present_columns"), list) else []
        if optional_present:
            st.caption("已识别可选列: " + ", ".join(optional_present))

        if main_csv.get("has_label"):
            label_col = str(main_csv.get("label_col") or "label")
            c5, c6, c7 = st.columns(3)
            c5.metric(f"有效 {label_col}", int(main_csv.get("label_valid_count") or 0))
            c6.metric(f"{label_col} 类别数", int(main_csv.get("label_class_count") or 0))
            c7.metric(
                "可做 calibration",
                "是" if bool(main_csv.get("calibration_possible")) else "否",
            )

        batch_profile = main_csv.get("batch_profile") if isinstance(main_csv.get("batch_profile"), dict) else {}
        if batch_profile:
            st.subheader("批量数据画像")
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("nanobody 数", int(batch_profile.get("unique_nanobody_count") or 0))
            p2.metric("conformer 数", int(batch_profile.get("unique_conformer_count") or 0))
            p3.metric("pose 数", int(batch_profile.get("unique_pose_count") or 0))
            p4.metric("重复 ID 行", int(batch_profile.get("duplicate_id_row_count") or 0))

            if str(batch_profile.get("mode") or "") == "input_csv":
                path_checks = batch_profile.get("path_checks") if isinstance(batch_profile.get("path_checks"), list) else []
                if path_checks:
                    path_rows: list[dict[str, Any]] = []
                    for item in path_checks:
                        if not isinstance(item, dict):
                            continue
                        path_rows.append(
                            {
                                "字段": item.get("column"),
                                "CSV列存在": item.get("column_present"),
                                "行内值": item.get("row_value_count"),
                                "默认覆盖": item.get("covered_by_default_count"),
                                "未填写": item.get("missing_value_count"),
                                "路径存在": item.get("existing_path_count"),
                                "路径不存在": item.get("missing_path_count"),
                                "相对路径未确认": item.get("unchecked_relative_path_count"),
                                "默认来源": item.get("default_source"),
                            }
                        )
                    st.caption("路径检查会按 build_feature_table 的规则解析本地 CSV 相对路径；上传 CSV 中的相对路径需要 zip/目录导入保留上下文。")
                    st.dataframe(pd.DataFrame(path_rows), use_container_width=True, hide_index=True)

                    sample_rows: list[dict[str, str]] = []
                    for item in path_checks:
                        if not isinstance(item, dict):
                            continue
                        for path_text in item.get("sample_missing_paths", []) or []:
                            sample_rows.append({"字段": str(item.get("column") or ""), "类型": "路径不存在", "示例": str(path_text)})
                        for path_text in item.get("sample_unchecked_relative_paths", []) or []:
                            sample_rows.append({"字段": str(item.get("column") or ""), "类型": "相对路径未确认", "示例": str(path_text)})
                    if sample_rows:
                        with st.expander("查看路径问题示例"):
                            st.dataframe(pd.DataFrame(sample_rows), use_container_width=True, hide_index=True)

                    path_repair = main_csv.get("path_repair") if isinstance(main_csv.get("path_repair"), dict) else {}
                    repair_summary = (
                        path_repair.get("summary")
                        if isinstance(path_repair.get("summary"), dict)
                        else {}
                    )
                    repair_rows = (
                        path_repair.get("plan_rows")
                        if isinstance(path_repair.get("plan_rows"), list)
                        else []
                    )
                    if repair_summary:
                        st.subheader("缺失路径自动定位建议")
                        r1, r2, r3, r4 = st.columns(4)
                        r1.metric("缺失引用", int(repair_summary.get("missing_reference_count") or 0))
                        r2.metric("候选建议", int(repair_summary.get("suggested_replacement_count") or 0))
                        r3.metric("可自动修复", int(repair_summary.get("auto_repair_count") or 0))
                        r4.metric("索引文件数", int(repair_summary.get("indexed_file_count") or 0))
                        search_roots = repair_summary.get("search_roots") if isinstance(repair_summary.get("search_roots"), list) else []
                        if search_roots:
                            st.caption("搜索目录: " + " | ".join([str(root) for root in search_roots]))
                        if bool(repair_summary.get("index_truncated")):
                            st.warning("搜索目录文件数超过索引上限，修复建议可能不完整。")
                        if repair_rows:
                            repair_df = pd.DataFrame(repair_rows)
                            st.dataframe(repair_df.head(100), use_container_width=True, hide_index=True)
                            st.download_button(
                                label="下载路径修复建议 CSV",
                                data=repair_df.to_csv(index=False).encode("utf-8-sig"),
                                file_name="input_path_repair_plan.csv",
                                mime="text/csv",
                                use_container_width=True,
                                key="download_input_path_repair_plan_csv",
                            )

                            csv_path_text = str(main_csv.get("path") or "").strip()
                            auto_repair_count = int(repair_summary.get("auto_repair_count") or 0)
                            if csv_path_text and auto_repair_count > 0:
                                try:
                                    source_csv_path = Path(csv_path_text).expanduser().resolve()
                                    source_df = _read_local_csv(source_csv_path)
                                    repaired_df = apply_input_path_repair_plan(
                                        source_df,
                                        repair_rows,
                                        min_confidence=AUTO_REPAIR_CONFIDENCE_THRESHOLD,
                                    )
                                    repaired_csv_bytes = repaired_df.to_csv(index=False).encode("utf-8-sig")
                                    st.download_button(
                                        label="下载自动修复版 input_csv",
                                        data=repaired_csv_bytes,
                                        file_name=f"{source_csv_path.stem}_path_repaired.csv",
                                        mime="text/csv",
                                        use_container_width=True,
                                        key="download_input_path_repaired_csv",
                                    )
                                    if st.button(
                                        "保存并使用自动修复版 input_csv",
                                        use_container_width=True,
                                        key="save_and_use_input_path_repaired_csv",
                                    ):
                                        repair_dir = LOCAL_APP_RUN_ROOT / "_input_path_repairs"
                                        repair_dir.mkdir(parents=True, exist_ok=True)
                                        repaired_path = (
                                            repair_dir
                                            / f"{_slugify_name(source_csv_path.stem)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_path_repaired.csv"
                                        )
                                        repaired_df.to_csv(repaired_path, index=False)
                                        st.session_state["input_csv_local_path"] = str(repaired_path)
                                        st.session_state["start_mode"] = "input_csv"
                                        st.success(f"已保存修复版 input_csv: {repaired_path}")
                                        st.rerun()
                                except Exception as exc:
                                    st.warning(f"生成修复版 input_csv 失败: {exc}")
            else:
                feature_meta = [
                    ("可转数值列", int(batch_profile.get("numeric_column_count") or 0)),
                    ("包含 pred_prob", "是" if bool(batch_profile.get("has_pred_prob")) else "否"),
                    ("包含规则分数", "是" if bool(batch_profile.get("has_rule_score")) else "否"),
                ]
                st.dataframe(
                    pd.DataFrame([{"项目": name, "值": value} for name, value in feature_meta]),
                    use_container_width=True,
                    hide_index=True,
                )
                numeric_preview = batch_profile.get("numeric_columns_preview") if isinstance(batch_profile.get("numeric_columns_preview"), list) else []
                if numeric_preview:
                    st.caption("数值列预览: " + ", ".join([str(col) for col in numeric_preview]))

        processing_plan = report.get("processing_plan") if isinstance(report.get("processing_plan"), dict) else {}
        if processing_plan:
            st.subheader("本次将如何处理")
            q1, q2, q3 = st.columns(3)
            q1.metric("处理行数", int(processing_plan.get("row_count") or 0))
            q2.metric("top_k", int(processing_plan.get("top_k") or 0))
            q3.metric("skip_failed_rows", "是" if bool(processing_plan.get("skip_failed_rows")) else "否")
            label_plan = str(processing_plan.get("label_plan") or "")
            if label_plan:
                st.caption(label_plan)
            stages = processing_plan.get("stages") if isinstance(processing_plan.get("stages"), list) else []
            if stages:
                st.dataframe(pd.DataFrame(stages), use_container_width=True, hide_index=True)
            expected_outputs = processing_plan.get("expected_key_outputs") if isinstance(processing_plan.get("expected_key_outputs"), list) else []
            if expected_outputs:
                st.caption("关键输出: " + ", ".join([str(item) for item in expected_outputs]))

        preview_rows = main_csv.get("preview_rows") if isinstance(main_csv.get("preview_rows"), list) else []
        if preview_rows:
            st.subheader("CSV 前 5 行预览")
            st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)

    optional_files = report.get("optional_files") if isinstance(report.get("optional_files"), list) else []
    if optional_files:
        st.caption("默认文件来源检查")
        st.dataframe(pd.DataFrame(optional_files), use_container_width=True, hide_index=True)

    runtime_dependencies = (
        report.get("runtime_dependencies") if isinstance(report.get("runtime_dependencies"), dict) else {}
    )
    if runtime_dependencies:
        st.caption("运行环境依赖检查")
        checked_python = str(runtime_dependencies.get("checked_python_executable") or "N/A")
        st.caption(f"检查解释器: {checked_python}")
        dep_col1, dep_col2 = st.columns(2)
        checked_dependencies = (
            runtime_dependencies.get("checked_dependencies")
            if isinstance(runtime_dependencies.get("checked_dependencies"), list)
            else []
        )
        missing_dependencies = (
            runtime_dependencies.get("missing_dependencies")
            if isinstance(runtime_dependencies.get("missing_dependencies"), list)
            else []
        )
        dep_col1.metric("依赖检查项", len(checked_dependencies))
        dep_col2.metric("缺失依赖", len(missing_dependencies))

        dependency_error = str(runtime_dependencies.get("error_message") or "").strip()
        if dependency_error:
            st.error(f"依赖检查失败: {dependency_error}")
        elif missing_dependencies:
            st.error(
                "缺少依赖: "
                + format_missing_dependency_summary(missing_dependencies)
                + "。"
            )
            st.dataframe(pd.DataFrame(missing_dependencies), use_container_width=True, hide_index=True)
        else:
            st.success("关键 Python 依赖检查通过。")

    st.download_button(
        label="下载输入检查报告 JSON",
        data=json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="input_preflight_report.json",
        mime="application/json",
        use_container_width=True,
    )


def _render_input_status_panel(
    *,
    start_mode: str,
    input_csv_upload: Any,
    feature_csv_upload: Any,
    default_pocket_upload: Any,
    default_catalytic_upload: Any,
    default_ligand_upload: Any,
    experiment_plan_override_upload: Any,
    form_payload: dict[str, Any],
) -> None:
    st.subheader("当前输入状态")
    status_rows: list[dict[str, Any]] = []
    if start_mode == "input_csv":
        status_rows.append(
            _build_source_status_row(
                label="主输入 input_csv",
                uploaded_file=input_csv_upload,
                local_path_text=str(form_payload.get("input_csv_local_path") or ""),
                required=True,
            )
        )
    else:
        status_rows.append(
            _build_source_status_row(
                label="主输入 feature_csv",
                uploaded_file=feature_csv_upload,
                local_path_text=str(form_payload.get("feature_csv_local_path") or ""),
                required=True,
            )
        )
    status_rows.extend(
        [
            _build_source_status_row(
                label="default_pocket_file",
                uploaded_file=default_pocket_upload,
                local_path_text=str(form_payload.get("default_pocket_local_path") or ""),
                required=False,
            ),
            _build_source_status_row(
                label="default_catalytic_file",
                uploaded_file=default_catalytic_upload,
                local_path_text=str(form_payload.get("default_catalytic_local_path") or ""),
                required=False,
            ),
            _build_source_status_row(
                label="default_ligand_file",
                uploaded_file=default_ligand_upload,
                local_path_text=str(form_payload.get("default_ligand_local_path") or ""),
                required=False,
            ),
            _build_source_status_row(
                label="experiment_plan_override_csv",
                uploaded_file=experiment_plan_override_upload,
                local_path_text=str(form_payload.get("experiment_plan_override_local_path") or ""),
                required=False,
            ),
        ]
    )
    st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

    with st.expander("输入模板与格式说明"):
        st.write("最常用的两种主输入格式如下。先下载模板填一版，再上传到本地软件即可。")
        sample_col1, sample_col2 = st.columns(2)
        sample_col1.download_button(
            label="下载 input_csv 模板",
            data=_build_sample_input_csv_text().encode("utf-8"),
            file_name="input_pose_table_template.csv",
            mime="text/csv",
            use_container_width=True,
        )
        sample_col2.download_button(
            label="下载 pose_features 模板",
            data=_build_sample_feature_csv_text().encode("utf-8"),
            file_name="pose_features_template.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            label="下载实验计划覆盖 CSV 模板",
            data=_build_sample_experiment_plan_override_csv_text().encode("utf-8"),
            file_name="experiment_plan_override_template.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption("input_csv 必需列: " + ", ".join(INPUT_CSV_REQUIRED_COLUMNS))
        st.caption("pose_features.csv 最小必需列: " + ", ".join(FEATURE_CSV_REQUIRED_COLUMNS))
        st.caption(
            "实验计划覆盖 CSV 可选列: nanobody_id, plan_override, experiment_status, "
            "experiment_result, validation_label, experiment_owner, experiment_cost, experiment_note。"
        )
        st.caption(
            "如果使用 zip 或目录导入，建议文件名尽量使用 "
            "`input_pose_table.csv`、`pose_features.csv`、`pocket*`、`catalytic*`、`ligand*`，"
            "这样页面可以自动识别并回填路径。"
        )

    last_bundle_import = st.session_state.get("last_bundle_import")
    last_bundle_import_message = str(st.session_state.get("last_bundle_import_message") or "")
    if last_bundle_import_message:
        st.info(last_bundle_import_message)
    if isinstance(last_bundle_import, dict):
        source_kind = "目录" if str(last_bundle_import.get("source_kind") or "") == "directory" else "zip"
        st.caption(
            f"最近导入数据源: {last_bundle_import.get('source_name', 'N/A')} ({source_kind}) | "
            f"共扫描 {int(last_bundle_import.get('file_count') or 0)} 个文件"
        )
        source_path = str(last_bundle_import.get("source_path") or "")
        if source_path:
            st.code(source_path, language="text")
        detected_items = last_bundle_import.get("detected_items")
        if isinstance(detected_items, list) and detected_items:
            st.dataframe(pd.DataFrame(detected_items), use_container_width=True, hide_index=True)
        generated_input_table = (
            last_bundle_import.get("generated_input_table")
            if isinstance(last_bundle_import.get("generated_input_table"), dict)
            else None
        )
        if generated_input_table is not None:
            st.subheader("自动生成的 input_pose_table.csv")
            gen_col1, gen_col2 = st.columns(2)
            gen_col1.metric("生成行数", int(generated_input_table.get("row_count") or 0))
            gen_col2.metric("识别 PDB 数", int(generated_input_table.get("pose_pdb_count") or 0))
            generated_csv_path = Path(str(generated_input_table.get("generated_csv_path") or ""))
            st.caption("自动生成表只做保守推断；运行前仍建议点击“检查当前输入”确认 ID 和路径是否符合预期。")
            st.code(str(generated_csv_path), language="text")
            preview_rows = generated_input_table.get("preview_rows")
            if isinstance(preview_rows, list) and preview_rows:
                st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)
            if generated_csv_path.exists():
                st.download_button(
                    label="下载自动生成的 input_pose_table.csv",
                    data=generated_csv_path.read_bytes(),
                    file_name=generated_csv_path.name,
                    mime="text/csv",
                    use_container_width=True,
                    key="download_generated_input_pose_table_csv",
                )
        preview_items = last_bundle_import.get("bundle_preview_items")
        if isinstance(preview_items, list) and preview_items:
            with st.expander("查看导入源文件预览"):
                st.dataframe(pd.DataFrame(preview_items), use_container_width=True, hide_index=True)


def _try_relative_to(path: Path, base: Path) -> Path | None:
    try:
        return path.resolve().relative_to(base.resolve())
    except ValueError:
        return None


def _open_local_path(path: Path) -> tuple[bool, str]:
    target = path.expanduser().resolve()
    if not target.exists():
        return False, f"路径不存在: {target}"

    try:
        if os.name == "nt":
            os.startfile(str(target))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target)])
        return True, f"已打开: {target}"
    except Exception as exc:
        return False, f"打开失败: {exc}"


def _collect_download_artifacts(
    summary: dict[str, Any] | None,
    metadata: dict[str, Any] | None = None,
) -> list[tuple[str, Path]]:
    if not isinstance(summary, dict):
        summary = {}
    artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}

    ordered_keys = [
        "recommended_pipeline_summary_json",
        "demo_overview_html",
        "demo_readme_md",
        "demo_interpretation_md",
        "real_data_starter_readme_md",
        "real_data_starter_manifest_json",
        "mini_pdb_example_readme_md",
        "mini_pdb_example_input_csv",
        "execution_report_md",
        "quality_gate_summary_json",
        "quality_gate_checks_csv",
        "quality_gate_report_md",
        "geometry_proxy_audit_summary_json",
        "geometry_proxy_audit_report_md",
        "geometry_proxy_feature_summary_csv",
        "geometry_proxy_flagged_poses_csv",
        "geometry_proxy_candidate_audit_csv",
        "batch_decision_summary_json",
        "batch_decision_summary_md",
        "batch_decision_summary_cards_csv",
        "rule_ranking_csv",
        "ml_ranking_csv",
        "consensus_ranking_csv",
        "consensus_summary_json",
        "consensus_report_md",
        "score_explanation_cards_csv",
        "score_explanation_cards_summary_json",
        "score_explanation_cards_md",
        "score_explanation_cards_html",
        "parameter_sensitivity_candidate_csv",
        "parameter_sensitivity_sensitive_csv",
        "parameter_sensitivity_summary_json",
        "parameter_sensitivity_report_md",
        "candidate_report_index_html",
        "candidate_report_manifest_csv",
        "candidate_report_summary_json",
        "candidate_report_zip",
        "candidate_tradeoff_table_csv",
        "candidate_pairwise_comparisons_csv",
        "candidate_group_comparison_summary_csv",
        "candidate_comparison_summary_json",
        "candidate_comparison_report_md",
        "ai_run_summary_md",
        "ai_top_candidates_explanation_md",
        "ai_failure_diagnosis_md",
        "ai_assistant_summary_json",
        "run_provenance_card_json",
        "run_provenance_card_md",
        "run_artifact_manifest_csv",
        "run_input_file_manifest_csv",
        "run_provenance_integrity_json",
        "experiment_suggestions_csv",
        "experiment_suggestions_summary_json",
        "experiment_suggestions_report_md",
        "experiment_plan_csv",
        "experiment_plan_summary_json",
        "experiment_plan_md",
        "experiment_plan_state_ledger_csv",
        "validation_evidence_summary_json",
        "validation_evidence_report_md",
        "validation_evidence_by_candidate_csv",
        "validation_evidence_topk_csv",
        "validation_evidence_action_items_csv",
        "pose_predictions_csv",
        "training_summary_json",
        "training_log_csv",
        "feature_qc_json",
        "baseline_comparison_summary_json",
        "baseline_comparison_report_md",
        "calibrated_rule_config_json",
        "calibrated_comparison_summary_json",
        "calibrated_comparison_report_md",
        "improvement_summary_json",
        "improvement_report_md",
        "strategy_recommended_json",
        "strategy_report_md",
    ]

    derived_paths = {
        "recommended_pipeline_summary_json": _resolve_output_file(
            "recommended_pipeline_summary.json",
            summary=summary,
            metadata=metadata,
        ),
        "demo_overview_html": _resolve_output_file(
            "DEMO_OVERVIEW.html",
            summary=summary,
            metadata=metadata,
        ),
        "demo_readme_md": _resolve_output_file(
            "DEMO_README.md",
            summary=summary,
            metadata=metadata,
        ),
        "demo_interpretation_md": _resolve_output_file(
            "DEMO_INTERPRETATION.md",
            summary=summary,
            metadata=metadata,
        ),
        "real_data_starter_readme_md": _resolve_output_file(
            "REAL_DATA_STARTER",
            "README_REAL_DATA_STARTER.md",
            summary=summary,
            metadata=metadata,
        ),
        "real_data_starter_manifest_json": _resolve_output_file(
            "REAL_DATA_STARTER",
            "real_data_starter_manifest.json",
            summary=summary,
            metadata=metadata,
        ),
        "mini_pdb_example_readme_md": _resolve_output_file(
            "REAL_DATA_STARTER",
            "MINI_PDB_EXAMPLE",
            "README_MINI_PDB_EXAMPLE.md",
            summary=summary,
            metadata=metadata,
        ),
        "mini_pdb_example_input_csv": _resolve_output_file(
            "REAL_DATA_STARTER",
            "MINI_PDB_EXAMPLE",
            "input_pose_table.csv",
            summary=summary,
            metadata=metadata,
        ),
        "pose_predictions_csv": _resolve_output_file(
            "model_outputs",
            "pose_predictions.csv",
            summary=summary,
            metadata=metadata,
        ),
        "training_summary_json": _resolve_output_file(
            "model_outputs",
            "training_summary.json",
            summary=summary,
            metadata=metadata,
        ),
        "training_log_csv": _resolve_output_file(
            "model_outputs",
            "train_log.csv",
            summary=summary,
            metadata=metadata,
        ),
    }

    out: list[tuple[str, Path]] = []
    for key in ordered_keys:
        value = artifacts.get(key) or derived_paths.get(key)
        if not value:
            continue
        path = value if isinstance(value, Path) else Path(str(value))
        if path.exists():
            out.append((key, path))
    return out


def _collect_export_bundle_members(
    run_root: Path,
    summary: dict[str, Any] | None,
    metadata: dict[str, Any] | None = None,
) -> tuple[list[tuple[Path, Path]], list[str]]:
    members: list[tuple[Path, Path]] = []
    notes: list[str] = []
    seen_sources: set[str] = set()

    def add_file(source: Path | None, arcname: Path) -> None:
        if source is None:
            return
        source = source.expanduser().resolve()
        if not source.exists() or not source.is_file():
            return
        source_key = str(source)
        if source_key in seen_sources:
            return
        members.append((source, arcname))
        seen_sources.add(source_key)

    for filename in [APP_RUN_METADATA_NAME, APP_STDOUT_NAME, APP_STDERR_NAME]:
        add_file(run_root / filename, Path(run_root.name) / filename)

    inputs_dir = run_root / "inputs"
    if inputs_dir.exists():
        for file_path in sorted(inputs_dir.rglob("*")):
            if file_path.is_file():
                add_file(
                    file_path,
                    Path(run_root.name) / "inputs" / file_path.relative_to(inputs_dir),
                )

    feature_csv = summary.get("feature_csv") if isinstance(summary, dict) else None
    if feature_csv:
        feature_path = Path(str(feature_csv))
        if feature_path.exists():
            relative = _try_relative_to(feature_path, run_root)
            if relative is not None:
                add_file(
                    feature_path,
                    Path(run_root.name) / relative,
                )
            else:
                add_file(
                    feature_path,
                    Path(run_root.name) / "referenced_inputs" / feature_path.name,
                )

    resolved_inputs = metadata.get("resolved_inputs") if isinstance(metadata, dict) and isinstance(metadata.get("resolved_inputs"), dict) else {}
    for key, value in resolved_inputs.items():
        if not value:
            continue
        path = Path(str(value))
        if not path.exists() or not path.is_file():
            notes.append(f"missing_resolved_input::{key}::{path}")
            continue
        resolved = path.expanduser().resolve()
        relative = _try_relative_to(resolved, run_root)
        if relative is not None:
            add_file(
                resolved,
                Path(run_root.name) / relative,
            )
        else:
            add_file(
                resolved,
                Path(run_root.name) / "referenced_inputs" / f"{key}_{resolved.name}",
            )

    for artifact_name, artifact_path in _collect_download_artifacts(summary, metadata):
        add_file(
            artifact_path,
            Path(run_root.name) / "artifacts" / artifact_path.name,
        )

    exports_dir = run_root / "exports"
    if exports_dir.exists():
        for pattern in ["*.html", "*.pdf"]:
            for file_path in sorted(exports_dir.glob(pattern)):
                add_file(
                    file_path,
                    Path(run_root.name) / "exports" / file_path.name,
                )

    return members, notes


def _create_export_bundle(
    run_root: Path,
    summary: dict[str, Any] | None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    exports_dir = run_root / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    bundle_name = f"{run_root.name}_summary_bundle.zip"
    bundle_path = exports_dir / bundle_name

    members, notes = _collect_export_bundle_members(run_root, summary, metadata)
    manifest = {
        "bundle_type": "local_ml_app_summary_bundle",
        "run_name": run_root.name,
        "exported_at": _now_text(),
        "member_count": len(members),
        "notes": notes,
        "files": [
            {
                "source": str(source),
                "arcname": str(arcname).replace("\\", "/"),
                "size_bytes": int(source.stat().st_size),
            }
            for source, arcname in members
        ],
    }

    with zipfile.ZipFile(bundle_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for source, arcname in members:
            zf.write(source, arcname=str(arcname).replace("\\", "/"))
        zf.writestr(
            f"{run_root.name}/bundle_manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2),
        )

    return bundle_path


def _metric_text(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        if value is None:
            return "N/A"
        text = str(value).strip()
        return text or "N/A"


def _ratio_text(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def _load_feature_qc_payload(
    summary: dict[str, Any] | None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    feature_qc_path = None
    if isinstance(summary, dict):
        artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
        if artifacts.get("feature_qc_json"):
            feature_qc_path = Path(str(artifacts.get("feature_qc_json")))

    if feature_qc_path is None:
        feature_qc_path = _resolve_output_file(
            "feature_qc.json",
            summary=summary,
            metadata=metadata,
        )

    if feature_qc_path is None or not feature_qc_path.exists():
        return None
    return _load_json(feature_qc_path)


def _load_feature_df_for_qc(
    summary: dict[str, Any] | None,
    metadata: dict[str, Any] | None = None,
) -> pd.DataFrame | None:
    feature_csv = summary.get("feature_csv") if isinstance(summary, dict) else None
    if feature_csv:
        feature_path = Path(str(feature_csv))
        if feature_path.exists():
            return _load_csv(feature_path)

    fallback_path = _resolve_output_file(
        "pose_features.csv",
        summary=summary,
        metadata=metadata,
    )
    if fallback_path is not None and fallback_path.exists():
        return _load_csv(fallback_path)
    return None


def _df_to_html_table(df: pd.DataFrame | None, *, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return '<p class="muted">暂无可展示内容。</p>'
    view = df.head(int(max_rows)).copy()
    return view.to_html(index=False, escape=True, border=0, classes="data-table")


def _build_metric_cards_html(cards: list[tuple[str, Any]]) -> str:
    items: list[str] = []
    for label, value in cards:
        items.append(
            "<div class='metric-card'>"
            f"<div class='metric-label'>{html.escape(str(label))}</div>"
            f"<div class='metric-value'>{html.escape(_metric_text(value))}</div>"
            "</div>"
        )
    return "".join(items)


def _build_html_export_style() -> str:
    return """
  <style>
    :root {
      --bg: #f4f1ea;
      --card: #fffdf8;
      --line: #d8cfbf;
      --ink: #1f2421;
      --muted: #6b716d;
      --accent: #0d6a57;
      --accent-soft: #dcefe8;
      --warn: #9a6700;
      --bad: #9a2f2f;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      background: linear-gradient(180deg, #efe9da 0%, var(--bg) 52%, #ffffff 100%);
      color: var(--ink);
      line-height: 1.55;
    }
    .page {
      max-width: 1220px;
      margin: 0 auto;
      padding: 28px 20px 56px;
    }
    .hero {
      background: linear-gradient(135deg, #153d35 0%, #21574a 50%, #e3d3aa 100%);
      color: #fffaf0;
      padding: 24px 26px;
      border-radius: 20px;
      box-shadow: 0 18px 60px rgba(22, 45, 38, 0.18);
    }
    .hero h1 {
      margin: 0 0 8px;
      font-size: 30px;
      letter-spacing: 0.2px;
    }
    .hero p {
      margin: 0;
      color: rgba(255, 250, 240, 0.88);
    }
    .section {
      margin-top: 18px;
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 20px 20px 18px;
      box-shadow: 0 10px 28px rgba(28, 38, 33, 0.05);
    }
    .section h2 {
      margin: 0 0 14px;
      font-size: 18px;
    }
    .section h3 {
      margin: 14px 0 8px;
      font-size: 15px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
    }
    .metric-card {
      background: #fff;
      border: 1px solid #e7ded0;
      border-radius: 14px;
      padding: 14px 14px 12px;
    }
    .metric-label {
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 6px;
    }
    .metric-value {
      font-size: 24px;
      font-weight: 700;
    }
    .meta {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 8px 16px;
    }
    .meta-item {
      padding: 10px 12px;
      background: #faf7f0;
      border-radius: 12px;
      border: 1px solid #ece3d5;
    }
    .meta-label {
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
    }
    .meta-value {
      word-break: break-word;
      white-space: pre-wrap;
    }
    .data-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      overflow: hidden;
    }
    .data-table th, .data-table td {
      border: 1px solid #e6ddcf;
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
    }
    .data-table th {
      background: #f4efe5;
    }
    pre {
      background: #15211e;
      color: #f6f2e8;
      padding: 14px;
      border-radius: 14px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: Consolas, "Courier New", monospace;
      font-size: 12px;
    }
    ul {
      margin: 10px 0 0 18px;
      padding: 0;
    }
    .muted {
      color: var(--muted);
    }
    .footer {
      margin-top: 18px;
      color: var(--muted);
      font-size: 12px;
      text-align: center;
    }
    @media print {
      body { background: #fff; }
      .page { max-width: none; padding: 0; }
      .hero, .section { box-shadow: none; }
    }
  </style>
"""


def _load_pdf_font(*, size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates: list[Path] = []
    if os.name == "nt":
        if bold:
            candidates.extend(
                [
                    Path(r"C:\Windows\Fonts\msyhbd.ttc"),
                    Path(r"C:\Windows\Fonts\simhei.ttf"),
                    Path(r"C:\Windows\Fonts\arialbd.ttf"),
                ]
            )
        candidates.extend(
            [
                Path(r"C:\Windows\Fonts\msyh.ttc"),
                Path(r"C:\Windows\Fonts\simhei.ttf"),
                Path(r"C:\Windows\Fonts\arial.ttf"),
            ]
        )
    else:
        candidates.extend(
            [
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
                Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
            ]
        )

    seen_paths: set[str] = set()
    for path in candidates:
        path_key = str(path)
        if path_key in seen_paths:
            continue
        seen_paths.add(path_key)
        if not path.exists():
            continue
        try:
            return ImageFont.truetype(str(path), size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _load_pdf_mono_font(*, size: int) -> ImageFont.ImageFont:
    candidates: list[Path] = []
    if os.name == "nt":
        candidates.extend(
            [
                Path(r"C:\Windows\Fonts\consola.ttf"),
                Path(r"C:\Windows\Fonts\cour.ttf"),
                Path(r"C:\Windows\Fonts\lucon.ttf"),
                Path(r"C:\Windows\Fonts\simhei.ttf"),
            ]
        )
    else:
        candidates.extend(
            [
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
                Path("/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf"),
            ]
        )

    for path in candidates:
        if not path.exists():
            continue
        try:
            return ImageFont.truetype(str(path), size=size)
        except Exception:
            continue
    return _load_pdf_font(size=size)


def _normalize_pdf_section(section: dict[str, Any] | tuple[str, list[str]] | Any) -> dict[str, Any]:
    if isinstance(section, dict):
        normalized = dict(section)
        normalized["title"] = str(section.get("title") or "Untitled")
        normalized["kind"] = str(section.get("kind") or "text")
        if isinstance(section.get("lines"), list):
            normalized["lines"] = [str(item) for item in section.get("lines", [])]
        if isinstance(section.get("items"), list):
            normalized["items"] = [
                (str(item[0]), item[1]) if isinstance(item, tuple) and len(item) == 2 else (str(item), "")
                for item in section.get("items", [])
            ]
        normalized["columns"] = int(section.get("columns") or 2)
        normalized["accent"] = str(section.get("accent") or "")
        return normalized

    if isinstance(section, tuple) and len(section) == 2:
        title, lines = section
        normalized_lines = [str(item) for item in lines] if isinstance(lines, list) else [str(lines)]
        return {
            "title": str(title),
            "kind": "text",
            "lines": normalized_lines,
            "columns": 2,
            "accent": "",
        }

    return {
        "title": "Untitled",
        "kind": "text",
        "lines": [str(section)],
        "columns": 2,
        "accent": "",
    }


def _wrap_pdf_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    if normalized == "":
        return [""]

    wrapped: list[str] = []
    for paragraph in normalized.split("\n"):
        if paragraph == "":
            wrapped.append("")
            continue

        current = ""
        for char in paragraph:
            trial = current + char
            try:
                width = draw.textlength(trial, font=font)
            except Exception:
                width = len(trial) * max(1, int(getattr(font, "size", 16)))
            if current and width > max_width:
                wrapped.append(current)
                current = char
            else:
                current = trial
        if current:
            wrapped.append(current)
    return wrapped or [""]


def _df_to_pdf_lines(
    df: pd.DataFrame | None,
    *,
    max_rows: int = 12,
    max_cols: int = 8,
) -> list[str]:
    if df is None or df.empty:
        return ["暂无可展示内容。"]

    view = df.head(int(max_rows)).copy()
    omitted_columns: list[str] = []
    if len(view.columns) > int(max_cols):
        omitted_columns = [str(column) for column in view.columns[int(max_cols):]]
        view = view.loc[:, list(view.columns[: int(max_cols)])]

    try:
        view = view.fillna("")
    except Exception:
        pass
    text_lines = view.to_string(index=False).splitlines()
    if len(df) > int(max_rows):
        text_lines.append(f"... 省略 {len(df) - int(max_rows)} 行")
    if omitted_columns:
        text_lines.append("... 省略列: " + ", ".join(omitted_columns[:10]))
    return text_lines


def _build_pdf_document(
    pdf_path: Path,
    *,
    title: str,
    subtitle_lines: list[str],
    sections: list[dict[str, Any] | tuple[str, list[str]]],
) -> Path:
    page_width = PDF_PAGE_WIDTH
    page_height = PDF_PAGE_HEIGHT
    margin = PDF_PAGE_MARGIN

    title_font = _load_pdf_font(size=38, bold=True)
    subtitle_font = _load_pdf_font(size=19)
    section_font = _load_pdf_font(size=24, bold=True)
    body_font = _load_pdf_font(size=18)
    value_font = _load_pdf_font(size=24, bold=True)
    label_font = _load_pdf_font(size=14)
    mono_font = _load_pdf_mono_font(size=16)
    small_font = _load_pdf_font(size=16)
    header_font = _load_pdf_font(size=15, bold=True)

    accent_palette = [
        ("#0d6a57", "#e3f0eb"),
        ("#8f5a1f", "#f4e6d2"),
        ("#275a8a", "#deebf7"),
        ("#7c355d", "#f2deea"),
        ("#5b6d1f", "#ebf1d8"),
    ]
    normalized_sections = [_normalize_pdf_section(section) for section in sections]

    images: list[Image.Image] = []
    current_page_number = 1
    image = Image.new("RGB", (page_width, page_height), color="#f7f3ec")
    draw = ImageDraw.Draw(image)
    cursor_y = margin

    def _init_page_canvas(target_draw: ImageDraw.ImageDraw, *, page_no: int) -> None:
        target_draw.rectangle([(0, 0), (page_width, page_height)], fill="#f7f3ec")
        target_draw.rectangle([(0, 0), (page_width, 18)], fill="#124d41")
        target_draw.rectangle([(0, 18), (page_width, 52)], fill="#efe3cb")
        target_draw.text((margin, 24), APP_NAME, font=header_font, fill="#124d41")
        header_right = f"v{APP_VERSION} | PDF Export"
        try:
            header_width = target_draw.textlength(header_right, font=header_font)
        except Exception:
            header_width = len(header_right) * 8
        target_draw.text((page_width - margin - int(header_width), 24), header_right, font=header_font, fill="#124d41")
        page_label = f"Page {page_no}"
        try:
            label_width = target_draw.textlength(page_label, font=small_font)
        except Exception:
            label_width = len(page_label) * 8
        target_draw.text((page_width - margin - int(label_width), page_height - margin + 10), page_label, font=small_font, fill="#6b716d")
        target_draw.line([(margin, page_height - margin + 2), (page_width - margin, page_height - margin + 2)], fill="#d8cfbf", width=1)

    _init_page_canvas(draw, page_no=current_page_number)

    def start_new_page() -> None:
        nonlocal image, draw, cursor_y, current_page_number
        images.append(image)
        current_page_number += 1
        image = Image.new("RGB", (page_width, page_height), color="#f7f3ec")
        draw = ImageDraw.Draw(image)
        _init_page_canvas(draw, page_no=current_page_number)
        cursor_y = margin

    def ensure_space(height: int) -> None:
        nonlocal cursor_y
        if cursor_y + int(height) > page_height - margin:
            start_new_page()

    def flatten_wrapped_lines(lines: list[str], *, font: ImageFont.ImageFont, max_width: int) -> list[str]:
        flattened: list[str] = []
        for raw_line in lines or [""]:
            flattened.extend(_wrap_pdf_text(draw, raw_line, font=font, max_width=max_width))
        return flattened or [""]

    def estimate_section_height(section: dict[str, Any]) -> int:
        inner_width = page_width - margin * 2 - 40
        title_height = 70
        kind = str(section.get("kind") or "text")
        if kind == "metrics":
            items = section.get("items") if isinstance(section.get("items"), list) else []
            columns = max(1, min(3, int(section.get("columns") or 2)))
            rows = max(1, (len(items) + columns - 1) // columns)
            return title_height + rows * 108 + max(0, rows - 1) * 14 + 26
        if kind == "table":
            table_lines = flatten_wrapped_lines(section.get("lines") or ["暂无内容。"], font=mono_font, max_width=inner_width - 28)
            return title_height + len(table_lines) * 22 + 36
        if kind == "bullets":
            bullet_count = 0
            for line in section.get("lines") or ["暂无内容。"]:
                bullet_count += len(_wrap_pdf_text(draw, line, font=body_font, max_width=inner_width - 30))
            return title_height + max(1, bullet_count) * 24 + 24
        text_lines = flatten_wrapped_lines(section.get("lines") or ["暂无内容。"], font=body_font, max_width=inner_width)
        return title_height + len(text_lines) * 24 + 24

    def render_metric_section(section: dict[str, Any], *, left: int, top: int, accent: str, accent_soft: str) -> int:
        content_width = page_width - margin * 2 - 40
        columns = max(1, min(3, int(section.get("columns") or 2)))
        gap = 14
        card_width = int((content_width - gap * (columns - 1)) / columns)
        card_height = 108
        items = section.get("items") if isinstance(section.get("items"), list) else []
        items = items or [("N/A", "暂无内容")]
        x0 = left + 20
        y0 = top + 64
        for index, item in enumerate(items):
            row = index // columns
            col = index % columns
            card_left = x0 + col * (card_width + gap)
            card_top = y0 + row * (card_height + gap)
            card_right = card_left + card_width
            card_bottom = card_top + card_height
            fill = "#ffffff" if index % 2 == 0 else accent_soft
            draw.rounded_rectangle(
                [(card_left, card_top), (card_right, card_bottom)],
                radius=18,
                fill=fill,
                outline="#ddd2c1",
                width=1,
            )
            label, value = item
            label_lines = _wrap_pdf_text(draw, str(label), font=label_font, max_width=card_width - 24)
            value_lines = _wrap_pdf_text(draw, _metric_text(value), font=value_font, max_width=card_width - 24)
            label_y = card_top + 14
            for line in label_lines[:2]:
                draw.text((card_left + 12, label_y), line, font=label_font, fill="#6b716d")
                label_y += 18
            value_y = card_top + 48
            for line in value_lines[:2]:
                draw.text((card_left + 12, value_y), line, font=value_font, fill=accent)
                value_y += 28
        rows = max(1, (len(items) + columns - 1) // columns)
        return top + 64 + rows * card_height + max(0, rows - 1) * gap + 18

    def render_bullet_section(section: dict[str, Any], *, left: int, top: int, accent: str) -> int:
        y = top + 62
        max_width = page_width - margin * 2 - 70
        for raw_line in section.get("lines") or ["暂无内容。"]:
            wrapped = _wrap_pdf_text(draw, raw_line, font=body_font, max_width=max_width)
            bullet_center_y = y + 11
            draw.ellipse([(left + 22, bullet_center_y - 4), (left + 30, bullet_center_y + 4)], fill=accent)
            for line_index, line in enumerate(wrapped):
                text_x = left + 40
                draw.text((text_x, y), line, font=body_font, fill="#1f2421")
                y += 24
                if line_index == 0:
                    bullet_center_y = y
            y += 4
        return y + 8

    def render_table_section(section: dict[str, Any], *, left: int, top: int, accent_soft: str) -> int:
        y = top + 62
        box_left = left + 18
        box_top = y
        box_right = page_width - margin - 18
        content_width = box_right - box_left - 20
        table_lines = flatten_wrapped_lines(section.get("lines") or ["暂无内容。"], font=mono_font, max_width=content_width)
        table_height = len(table_lines) * 22 + 22
        draw.rounded_rectangle(
            [(box_left, box_top), (box_right, box_top + table_height)],
            radius=16,
            fill=accent_soft,
            outline="#d6cdbf",
            width=1,
        )
        current_y = box_top + 12
        for index, line in enumerate(table_lines):
            draw.text((box_left + 10, current_y), line, font=mono_font, fill="#2a2f2c")
            current_y += 22
            if index < len(table_lines) - 1:
                draw.line([(box_left + 8, current_y - 4), (box_right - 8, current_y - 4)], fill="#e7dece", width=1)
        return box_top + table_height + 14

    def render_text_section(section: dict[str, Any], *, left: int, top: int) -> int:
        y = top + 62
        max_width = page_width - margin * 2 - 40
        text_lines = flatten_wrapped_lines(section.get("lines") or ["暂无内容。"], font=body_font, max_width=max_width)
        for line in text_lines:
            draw.text((left + 20, y), line, font=body_font, fill="#1f2421")
            y += 24
        return y + 8

    hero_bottom = margin + 170
    draw.rounded_rectangle(
        [(margin, margin), (page_width - margin, hero_bottom)],
        radius=30,
        fill="#1f4f43",
    )
    draw.rounded_rectangle(
        [(page_width - margin - 210, margin + 18), (page_width - margin - 24, margin + 60)],
        radius=18,
        fill="#2c6658",
    )
    draw.text((page_width - margin - 192, margin + 28), f"v{APP_VERSION}", font=subtitle_font, fill="#f7efe1")
    cursor_y = margin + 22
    draw.text((margin + 24, cursor_y), title, font=title_font, fill="#fffaf0")
    cursor_y += 54
    chip_x = margin + 24
    chip_y = cursor_y
    for subtitle in subtitle_lines:
        chip_text = str(subtitle)
        try:
            chip_width = int(draw.textlength(chip_text, font=subtitle_font)) + 28
        except Exception:
            chip_width = len(chip_text) * 12 + 28
        if chip_x + chip_width > page_width - margin - 24:
            chip_x = margin + 24
            chip_y += 38
        draw.rounded_rectangle(
            [(chip_x, chip_y), (chip_x + chip_width, chip_y + 28)],
            radius=14,
            fill="#2c6658",
        )
        draw.text((chip_x + 14, chip_y + 5), chip_text, font=subtitle_font, fill="#f7efe1")
        chip_x += chip_width + 10
    cursor_y = max(hero_bottom + 12, chip_y + 52)

    for section_index, section in enumerate(normalized_sections):
        section_height = estimate_section_height(section)
        ensure_space(section_height + 12)
        accent, accent_soft = accent_palette[section_index % len(accent_palette)]
        if section.get("accent"):
            accent = str(section.get("accent"))
        card_left = margin
        card_top = cursor_y
        card_right = page_width - margin
        card_bottom = card_top + section_height
        draw.rounded_rectangle(
            [(card_left, card_top), (card_right, card_bottom)],
            radius=24,
            fill="#fffdf8",
            outline="#d8cfbf",
            width=1,
        )
        draw.rounded_rectangle(
            [(card_left, card_top), (card_right, card_top + 52)],
            radius=24,
            fill=accent_soft,
        )
        draw.rectangle([(card_left, card_top + 26), (card_right, card_top + 52)], fill=accent_soft)
        draw.rounded_rectangle(
            [(card_left + 18, card_top + 14), (card_left + 26, card_top + 40)],
            radius=4,
            fill=accent,
        )
        draw.text((card_left + 38, card_top + 12), str(section.get("title") or "Untitled"), font=section_font, fill=accent)

        kind = str(section.get("kind") or "text")
        if kind == "metrics":
            content_bottom = render_metric_section(section, left=card_left, top=card_top, accent=accent, accent_soft=accent_soft)
        elif kind == "bullets":
            content_bottom = render_bullet_section(section, left=card_left, top=card_top, accent=accent)
        elif kind == "table":
            content_bottom = render_table_section(section, left=card_left, top=card_top, accent_soft=accent_soft)
        else:
            content_bottom = render_text_section(section, left=card_left, top=card_top)

        cursor_y = max(card_bottom, content_bottom) + 14

    images.append(image)

    total_pages = len(images)
    footer_text = f"{APP_NAME} | v{APP_VERSION} | Generated at {_now_text()}"
    for page_index, page_image in enumerate(images, start=1):
        page_draw = ImageDraw.Draw(page_image)
        page_draw.text((margin, page_height - margin + 10), footer_text, font=small_font, fill="#6b716d")
        page_label = f"Page {page_index}/{total_pages}"
        try:
            label_width = page_draw.textlength(page_label, font=small_font)
        except Exception:
            label_width = len(page_label) * 8
        page_draw.text((page_width - margin - int(label_width), page_height - margin + 10), page_label, font=small_font, fill="#6b716d")

    rgb_images = [img.convert("RGB") for img in images]
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    if len(rgb_images) == 1:
        rgb_images[0].save(pdf_path, "PDF", resolution=150.0)
    else:
        rgb_images[0].save(
            pdf_path,
            "PDF",
            save_all=True,
            append_images=rgb_images[1:],
            resolution=150.0,
        )
    return pdf_path


def _create_html_summary_export(
    run_root: Path,
    summary: dict[str, Any] | None,
    metadata: dict[str, Any] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> Path:
    exports_dir = run_root / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    html_path = exports_dir / f"{run_root.name}_presentation_summary.html"

    summary = summary if isinstance(summary, dict) else {}
    metadata = metadata if isinstance(metadata, dict) else {}
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}

    artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
    notes = summary.get("notes") if isinstance(summary.get("notes"), list) else []
    out_dir = _get_out_dir(summary, metadata)
    out_dir_text = "" if out_dir is None else str(out_dir)

    rule_ranking_path = Path(str(artifacts.get("rule_ranking_csv"))) if artifacts.get("rule_ranking_csv") else None
    ml_ranking_path = Path(str(artifacts.get("ml_ranking_csv"))) if artifacts.get("ml_ranking_csv") else None
    consensus_ranking_path = Path(str(artifacts.get("consensus_ranking_csv"))) if artifacts.get("consensus_ranking_csv") else None
    report_path = Path(str(artifacts.get("execution_report_md"))) if artifacts.get("execution_report_md") else None
    training_summary_path = _resolve_output_file(
        "model_outputs",
        "training_summary.json",
        summary=summary,
        metadata=metadata,
    )

    rule_df = _load_csv(rule_ranking_path) if rule_ranking_path is not None and rule_ranking_path.exists() else None
    ml_df = _load_csv(ml_ranking_path) if ml_ranking_path is not None and ml_ranking_path.exists() else None
    consensus_df = _load_csv(consensus_ranking_path) if consensus_ranking_path is not None and consensus_ranking_path.exists() else None
    training_payload = _load_json(training_summary_path) if training_summary_path is not None and training_summary_path.exists() else None
    report_text = _read_text(report_path) if report_path is not None and report_path.exists() else ""
    inventory_df = _build_output_inventory(summary, metadata)
    feature_qc_payload = _load_feature_qc_payload(summary, metadata)
    feature_df = _load_feature_df_for_qc(summary, metadata)

    comparison = summary.get("comparison") if isinstance(summary.get("comparison"), dict) else {}
    baseline = comparison.get("baseline_rule_vs_ml") if isinstance(comparison.get("baseline_rule_vs_ml"), dict) else {}
    calibrated = comparison.get("calibrated_rule_vs_ml") if isinstance(comparison.get("calibrated_rule_vs_ml"), dict) else {}

    header_cards = _build_metric_cards_html(
        [
            ("Start Mode", summary.get("start_mode")),
            ("Valid Labels", summary.get("label_valid_count")),
            ("Label Classes", summary.get("label_class_count")),
            ("Calibration", "Yes" if bool(summary.get("calibration_possible", False)) else "No"),
            ("Status", metadata.get("status") or "N/A"),
            ("Execution Mode", metadata.get("execution_mode") or "N/A"),
        ]
    )
    comparison_cards = _build_metric_cards_html(
        [
            ("Baseline Rank Spearman", baseline.get("rank_spearman")),
            ("Baseline Rule AUC", baseline.get("rule_auc")),
            ("Calibrated Rank Spearman", calibrated.get("rank_spearman")),
            ("Calibrated Rule AUC", calibrated.get("rule_auc")),
        ]
    )

    training_cards = ""
    if isinstance(training_payload, dict):
        training_cards = _build_metric_cards_html(
            [
                ("Train Rows", training_payload.get("train_rows")),
                ("Val Rows", training_payload.get("val_rows")),
                ("Best Epoch", training_payload.get("best_epoch")),
                ("Best Val Loss", training_payload.get("best_val_loss")),
                ("Feature Count", training_payload.get("feature_count")),
                ("Pseudo Positive Rate", training_payload.get("pseudo_positive_rate")),
            ]
        )

    notes_html = "".join(f"<li>{html.escape(str(note))}</li>" for note in notes)
    diag_messages = diagnostics.get("messages") if isinstance(diagnostics.get("messages"), list) else []
    diag_suggestions = diagnostics.get("suggestions") if isinstance(diagnostics.get("suggestions"), list) else []
    diag_messages_html = "".join(f"<li>{html.escape(str(item))}</li>" for item in diag_messages)
    diag_suggestions_html = "".join(f"<li>{html.escape(str(item))}</li>" for item in diag_suggestions)

    processing_summary = feature_qc_payload.get("processing_summary") if isinstance(feature_qc_payload, dict) and isinstance(feature_qc_payload.get("processing_summary"), dict) else {}
    feature_qc = feature_qc_payload.get("feature_qc") if isinstance(feature_qc_payload, dict) and isinstance(feature_qc_payload.get("feature_qc"), dict) else {}
    feature_qc_cards = _build_metric_cards_html(
        [
            ("Total Rows", processing_summary.get("total_rows")),
            ("OK Rows", processing_summary.get("ok_rows")),
            ("Failed Rows", processing_summary.get("failed_rows")),
            ("Rows With Warning", processing_summary.get("rows_with_warning_message")),
        ]
    )
    status_counts = feature_qc.get("status_counts") if isinstance(feature_qc.get("status_counts"), dict) else {}
    status_counts_html = "".join(
        f"<li>{html.escape(str(key))}: {html.escape(str(value))}</li>"
        for key, value in status_counts.items()
    )
    all_empty_columns = feature_qc.get("all_empty_columns") if isinstance(feature_qc.get("all_empty_columns"), list) else []
    near_constant_columns = feature_qc.get("near_constant_columns") if isinstance(feature_qc.get("near_constant_columns"), list) else []
    pocket_shape_qc = feature_qc.get("pocket_shape_qc") if isinstance(feature_qc.get("pocket_shape_qc"), dict) else {}
    pocket_shape_cards = _build_metric_cards_html(
        [
            ("Pocket Overwide 阈值", pocket_shape_qc.get("overwide_threshold")),
            ("高 Overwide 行数", pocket_shape_qc.get("high_overwide_row_count")),
            ("高 Overwide 占比", pocket_shape_qc.get("high_overwide_row_fraction")),
            ("Overwide P95", pocket_shape_qc.get("overwide_proxy_p95")),
            ("Overwide Max", pocket_shape_qc.get("overwide_proxy_max")),
        ]
    ) if pocket_shape_qc else ""
    pocket_shape_top_rows = pocket_shape_qc.get("top_overwide_rows") if isinstance(pocket_shape_qc.get("top_overwide_rows"), list) else []
    pocket_shape_top_df = pd.DataFrame(pocket_shape_top_rows) if pocket_shape_top_rows else None

    failed_rows_df = None
    warning_rows_df = None
    if isinstance(feature_df, pd.DataFrame) and not feature_df.empty:
        base_cols = [col for col in ["nanobody_id", "conformer_id", "pose_id", "pdb_path", "split_mode", "status", "error_message", "warning_message"] if col in feature_df.columns]
        if "status" in feature_df.columns:
            failed_rows_df = feature_df[feature_df["status"].astype(str).str.lower().eq("failed")]
            if base_cols:
                failed_rows_df = failed_rows_df.loc[:, base_cols]
        if "warning_message" in feature_df.columns:
            warning_rows_df = feature_df[feature_df["warning_message"].fillna("").astype(str).str.strip().ne("")]
            if base_cols:
                warning_rows_df = warning_rows_df.loc[:, base_cols]

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(APP_NAME)} | {html.escape(run_root.name)} | v{html.escape(APP_VERSION)}</title>
{_build_html_export_style()}
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>{html.escape(APP_NAME)} 展示摘要</h1>
      <p>运行名: {html.escape(run_root.name)} | 版本: v{html.escape(APP_VERSION)} | 生成时间: {html.escape(_now_text())}</p>
    </section>

    <section class="section">
      <h2>核心摘要</h2>
      <div class="grid">{header_cards}</div>
    </section>

    <section class="section">
      <h2>运行信息</h2>
      <div class="meta">
        <div class="meta-item"><div class="meta-label">输出目录</div><div class="meta-value">{html.escape(out_dir_text or "N/A")}</div></div>
        <div class="meta-item"><div class="meta-label">最近命令</div><div class="meta-value">{html.escape(" ".join(metadata.get("command", [])) if isinstance(metadata.get("command"), list) else "")}</div></div>
        <div class="meta-item"><div class="meta-label">开始时间</div><div class="meta-value">{html.escape(str(metadata.get("started_at") or metadata.get("created_at") or "N/A"))}</div></div>
        <div class="meta-item"><div class="meta-label">结束时间</div><div class="meta-value">{html.escape(str(metadata.get("finished_at") or "N/A"))}</div></div>
      </div>
    </section>

    <section class="section">
      <h2>规则 vs ML 对照</h2>
      <div class="grid">{comparison_cards}</div>
    </section>

    <section class="section">
      <h2>训练摘要</h2>
      {"<div class='grid'>" + training_cards + "</div>" if training_cards else "<p class='muted'>当前没有 training_summary.json。</p>"}
    </section>

    <section class="section">
      <h2>执行备注</h2>
      {"<ul>" + notes_html + "</ul>" if notes_html else "<p class='muted'>当前没有执行备注。</p>"}
    </section>

    <section class="section">
      <h2>Rule + ML 共识排名预览</h2>
      {_df_to_html_table(consensus_df, max_rows=10)}
    </section>

    <section class="section">
      <h2>ML 排名预览</h2>
      {_df_to_html_table(ml_df, max_rows=10)}
    </section>

    <section class="section">
      <h2>Rule 排名预览</h2>
      {_df_to_html_table(rule_df, max_rows=10)}
    </section>

    <section class="section">
      <h2>运行产物清单</h2>
      {_df_to_html_table(inventory_df, max_rows=50)}
    </section>

    <section class="section">
      <h2>Feature QC / Warning 摘要</h2>
      {"<div class='grid'>" + feature_qc_cards + "</div>" if feature_qc_cards else "<p class='muted'>当前没有 feature_qc.json。</p>"}
      {"<h3>Status Counts</h3><ul>" + status_counts_html + "</ul>" if status_counts_html else ""}
      {"<p><strong>全空列:</strong> " + html.escape(", ".join([str(col) for col in all_empty_columns[:20]])) + "</p>" if all_empty_columns else ""}
      {"<p><strong>近常量列:</strong> " + html.escape(", ".join([str(col) for col in near_constant_columns[:20]])) + "</p>" if near_constant_columns else ""}
      {"<h3>Pocket Shape QC</h3><div class='grid'>" + pocket_shape_cards + "</div>" if pocket_shape_cards else ""}
      {_df_to_html_table(pocket_shape_top_df, max_rows=10) if pocket_shape_top_df is not None else ""}
    </section>

    <section class="section">
      <h2>失败行预览</h2>
      {_df_to_html_table(failed_rows_df, max_rows=20)}
    </section>

    <section class="section">
      <h2>Warning 行预览</h2>
      {_df_to_html_table(warning_rows_df, max_rows=20)}
    </section>

    <section class="section">
      <h2>诊断摘要</h2>
      {"<ul>" + diag_messages_html + "</ul>" if diag_messages_html else "<p class='muted'>当前没有诊断消息。</p>"}
      {"<h3>建议</h3><ul>" + diag_suggestions_html + "</ul>" if diag_suggestions_html else ""}
    </section>

    <section class="section">
      <h2>执行报告 Markdown</h2>
      <pre>{html.escape(report_text or "当前没有 execution report。")}</pre>
    </section>

    <div class="footer">
      该 HTML 为本地软件自动生成，可直接下载、发送或用浏览器打印为 PDF。
    </div>
  </div>
</body>
</html>
"""
    _write_text(html_path, html_text)
    return html_path


def _create_pdf_summary_export(
    run_root: Path,
    summary: dict[str, Any] | None,
    metadata: dict[str, Any] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> Path:
    exports_dir = run_root / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = exports_dir / f"{run_root.name}_presentation_summary.pdf"

    summary = summary if isinstance(summary, dict) else {}
    metadata = metadata if isinstance(metadata, dict) else {}
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}

    artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
    comparison = summary.get("comparison") if isinstance(summary.get("comparison"), dict) else {}
    baseline = comparison.get("baseline_rule_vs_ml") if isinstance(comparison.get("baseline_rule_vs_ml"), dict) else {}
    calibrated = comparison.get("calibrated_rule_vs_ml") if isinstance(comparison.get("calibrated_rule_vs_ml"), dict) else {}
    notes = summary.get("notes") if isinstance(summary.get("notes"), list) else []
    out_dir = _get_out_dir(summary, metadata)

    rule_ranking_path = Path(str(artifacts.get("rule_ranking_csv"))) if artifacts.get("rule_ranking_csv") else None
    ml_ranking_path = Path(str(artifacts.get("ml_ranking_csv"))) if artifacts.get("ml_ranking_csv") else None
    consensus_ranking_path = Path(str(artifacts.get("consensus_ranking_csv"))) if artifacts.get("consensus_ranking_csv") else None
    report_path = Path(str(artifacts.get("execution_report_md"))) if artifacts.get("execution_report_md") else None
    rule_df = _load_csv(rule_ranking_path) if rule_ranking_path is not None and rule_ranking_path.exists() else None
    ml_df = _load_csv(ml_ranking_path) if ml_ranking_path is not None and ml_ranking_path.exists() else None
    consensus_df = _load_csv(consensus_ranking_path) if consensus_ranking_path is not None and consensus_ranking_path.exists() else None
    report_text = _read_text(report_path) if report_path is not None and report_path.exists() else ""

    training_payload = _load_training_summary_payload(summary, metadata)
    training_summary = training_payload.get("summary") if isinstance(training_payload, dict) and isinstance(training_payload.get("summary"), dict) else {}
    pseudo_block = training_payload.get("pseudo") if isinstance(training_payload, dict) and isinstance(training_payload.get("pseudo"), dict) else {}
    pseudo_distribution = pseudo_block.get("distribution") if isinstance(pseudo_block.get("distribution"), dict) else {}
    qc_payload = _load_feature_qc_payload(summary, metadata)
    processing_summary = qc_payload.get("processing_summary") if isinstance(qc_payload, dict) and isinstance(qc_payload.get("processing_summary"), dict) else {}

    subtitle_lines = [
        f"运行名: {run_root.name}",
        f"版本: v{APP_VERSION}",
        f"生成时间: {_now_text()}",
    ]
    sections: list[dict[str, Any] | tuple[str, list[str]]] = [
        {
            "title": "核心摘要",
            "kind": "metrics",
            "columns": 2,
            "items": [
                ("Start Mode", summary.get("start_mode", "N/A")),
                ("Status", metadata.get("status", "N/A")),
                ("Execution Mode", metadata.get("execution_mode", "N/A")),
                ("Valid Labels", summary.get("label_valid_count", 0)),
                ("Label Classes", summary.get("label_class_count", 0)),
                ("Calibration", "Yes" if bool(summary.get("calibration_possible", False)) else "No"),
            ],
        },
        {
            "title": "规则 vs ML 对照",
            "kind": "metrics",
            "columns": 2,
            "items": [
                ("Baseline Rank Spearman", baseline.get("rank_spearman")),
                ("Baseline Rule AUC", baseline.get("rule_auc")),
                ("Calibrated Rank Spearman", calibrated.get("rank_spearman")),
                ("Calibrated Rule AUC", calibrated.get("rule_auc")),
            ],
        },
        {
            "title": "训练摘要",
            "kind": "metrics",
            "columns": 2,
            "items": [
                ("Training Mode", training_payload.get("mode", "N/A") if isinstance(training_payload, dict) else "N/A"),
                ("Train Rows", training_payload.get("n_rows_train", training_summary.get("train_size", "N/A")) if isinstance(training_payload, dict) else "N/A"),
                ("Val Rows", training_payload.get("n_rows_val", training_summary.get("val_size", "N/A")) if isinstance(training_payload, dict) else "N/A"),
                ("Feature Count", training_payload.get("n_features", "N/A") if isinstance(training_payload, dict) else "N/A"),
                ("Best Epoch", training_summary.get("best_epoch", "N/A")),
                ("Best Val Loss", training_summary.get("best_val_loss")),
                ("Pseudo Positive Rate", pseudo_block.get("pseudo_positive_rate", pseudo_distribution.get("pseudo_positive_rate"))),
                ("Output Dir", str(out_dir or "N/A")),
            ],
        },
        {
            "title": "Feature QC / Warning",
            "kind": "metrics",
            "columns": 2,
            "items": [
                ("Total Rows", processing_summary.get("total_rows", "N/A")),
                ("OK Rows", processing_summary.get("ok_rows", "N/A")),
                ("Failed Rows", processing_summary.get("failed_rows", "N/A")),
                ("Rows With Warning", processing_summary.get("rows_with_warning_message", "N/A")),
            ],
        },
    ]

    if notes:
        sections.append({"title": "执行备注", "kind": "bullets", "lines": [str(note) for note in notes]})

    diag_messages = diagnostics.get("messages") if isinstance(diagnostics.get("messages"), list) else []
    diag_suggestions = diagnostics.get("suggestions") if isinstance(diagnostics.get("suggestions"), list) else []
    if diag_messages or diag_suggestions:
        sections.append(
            {
                "title": "诊断摘要",
                "kind": "bullets",
                "lines": [*(f"消息: {item}" for item in diag_messages), *(f"建议: {item}" for item in diag_suggestions)],
            }
        )

    sections.append({"title": "Rule + ML 共识排名预览", "kind": "table", "lines": _df_to_pdf_lines(consensus_df, max_rows=10, max_cols=6)})
    sections.append({"title": "ML 排名预览", "kind": "table", "lines": _df_to_pdf_lines(ml_df, max_rows=10, max_cols=6)})
    sections.append({"title": "Rule 排名预览", "kind": "table", "lines": _df_to_pdf_lines(rule_df, max_rows=10, max_cols=6)})

    report_lines = report_text.splitlines()[:60] if report_text else ["当前没有 execution report。"]
    if report_text and len(report_text.splitlines()) > 60:
        report_lines.append(f"... 省略 {len(report_text.splitlines()) - 60} 行")
    sections.append({"title": "执行报告 Markdown 摘要", "kind": "table", "lines": report_lines})

    return _build_pdf_document(
        pdf_path,
        title=f"{APP_NAME} 自动化 PDF 摘要",
        subtitle_lines=subtitle_lines,
        sections=sections,
    )


def _build_output_inventory(
    summary: dict[str, Any] | None,
    metadata: dict[str, Any] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seen_paths: set[str] = set()

    out_dir = _get_out_dir(summary, metadata)
    if out_dir is not None:
        rows.append(
            {
                "artifact": "output_dir",
                "exists": out_dir.exists(),
                "kind": "dir",
                "size_or_items": _format_size_text(out_dir) if out_dir.exists() else "N/A",
                "path": str(out_dir),
            }
        )
        seen_paths.add(str(out_dir))

    feature_csv = summary.get("feature_csv") if isinstance(summary, dict) else None
    if feature_csv:
        feature_path = Path(str(feature_csv))
        rows.append(
            {
                "artifact": "feature_csv",
                "exists": feature_path.exists(),
                "kind": "file",
                "size_or_items": _format_size_text(feature_path) if feature_path.exists() else "N/A",
                "path": str(feature_path),
            }
        )
        seen_paths.add(str(feature_path))

    for name, path in _collect_download_artifacts(summary, metadata):
        path_key = str(path)
        if path_key in seen_paths:
            continue
        rows.append(
            {
                "artifact": name,
                "exists": path.exists(),
                "kind": "dir" if path.is_dir() else "file",
                "size_or_items": _format_size_text(path) if path.exists() else "N/A",
                "path": path_key,
            }
        )
        seen_paths.add(path_key)

    return pd.DataFrame(rows)


def _render_download_buttons(
    summary: dict[str, Any] | None,
    metadata: dict[str, Any] | None = None,
) -> None:
    artifacts = _collect_download_artifacts(summary, metadata)
    if not artifacts:
        st.info("当前没有可下载的关键产物。")
        return

    for key, path in artifacts:
        st.download_button(
            label=f"下载 {key}",
            data=path.read_bytes(),
            file_name=path.name,
            mime="application/octet-stream",
            use_container_width=True,
        )


def _render_summary_metrics(summary: dict[str, Any] | None) -> None:
    if not isinstance(summary, dict):
        st.info("还没有可展示的执行摘要。")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Start Mode", str(summary.get("start_mode", "N/A")))
    c2.metric("Valid Labels", int(summary.get("label_valid_count", 0)))
    c3.metric("Label Classes", int(summary.get("label_class_count", 0)))
    c4.metric("Calibration", "Yes" if bool(summary.get("calibration_possible", False)) else "No")

    comparison = summary.get("comparison") if isinstance(summary.get("comparison"), dict) else {}
    baseline = comparison.get("baseline_rule_vs_ml") if isinstance(comparison.get("baseline_rule_vs_ml"), dict) else {}
    calibrated = comparison.get("calibrated_rule_vs_ml") if isinstance(comparison.get("calibrated_rule_vs_ml"), dict) else {}

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Baseline Rank Spearman", _fmt_metric(baseline.get("rank_spearman")))
    c6.metric("Baseline Rule AUC", _fmt_metric(baseline.get("rule_auc")))
    c7.metric("Calibrated Rank Spearman", _fmt_metric(calibrated.get("rank_spearman")))
    c8.metric("Calibrated Rule AUC", _fmt_metric(calibrated.get("rule_auc")))


def _fmt_metric(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "N/A"


def _render_training_summary_panel(
    summary: dict[str, Any] | None,
    metadata: dict[str, Any] | None = None,
) -> None:
    training_summary_path = _resolve_output_file(
        "model_outputs",
        "training_summary.json",
        summary=summary,
        metadata=metadata,
    )
    if training_summary_path is None or not training_summary_path.exists():
        st.info("当前没有 training_summary.json。")
        return

    payload = _load_json(training_summary_path)
    if not isinstance(payload, dict):
        st.info("training_summary.json 内容不可读。")
        return

    summary_block = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    pseudo_block = payload.get("pseudo") if isinstance(payload.get("pseudo"), dict) else {}
    threshold_block = pseudo_block.get("threshold") if isinstance(pseudo_block.get("threshold"), dict) else {}
    used_columns = pseudo_block.get("used_columns") if isinstance(pseudo_block.get("used_columns"), list) else []

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training Mode", str(payload.get("mode", "N/A")))
    c2.metric("Train Rows", int(payload.get("n_rows_train", summary_block.get("train_size", 0))))
    c3.metric("Val Rows", int(payload.get("n_rows_val", summary_block.get("val_size", 0))))
    c4.metric("Features", int(payload.get("n_features", 0)))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Best Epoch", int(summary_block.get("best_epoch", -1)))
    c6.metric("Best Val Loss", _fmt_metric(summary_block.get("best_val_loss")))
    c7.metric("Epochs Ran", int(summary_block.get("epochs_ran", 0)))
    c8.metric(
        "Pseudo Positive Rate",
        _fmt_metric(
            pseudo_block.get(
                "pseudo_positive_rate",
                (pseudo_block.get("distribution") or {}).get("pseudo_positive_rate"),
            )
        ),
    )

    detail_parts = []
    if threshold_block:
        detail_parts.append(
            f"Pseudo threshold: {threshold_block.get('mode', 'N/A')} / {threshold_block.get('value', 'N/A')}"
        )
    if used_columns:
        detail_parts.append(f"Pseudo columns used: {len(used_columns)}")
    if detail_parts:
        st.caption(" | ".join(detail_parts))

    train_log_path = _resolve_output_file(
        "model_outputs",
        "train_log.csv",
        summary=summary,
        metadata=metadata,
    )
    if train_log_path is not None and train_log_path.exists():
        train_log_df = _load_csv(train_log_path)
        if train_log_df is not None and "epoch" in train_log_df.columns:
            chart_columns = [
                col
                for col in [
                    "train_loss",
                    "val_loss",
                    "train_auc",
                    "val_auc",
                    "train_accuracy",
                    "val_accuracy",
                ]
                if col in train_log_df.columns
            ]
            if chart_columns:
                st.subheader("训练曲线")
                st.line_chart(train_log_df[["epoch"] + chart_columns].set_index("epoch"))

    with st.expander("查看 training_summary.json"):
        st.json(payload)


def _render_feature_qc_panel(
    summary: dict[str, Any] | None,
    metadata: dict[str, Any] | None = None,
) -> None:
    qc_payload = _load_feature_qc_payload(summary, metadata)
    feature_df = _load_feature_df_for_qc(summary, metadata)

    if not isinstance(qc_payload, dict) and feature_df is None:
        st.info("当前没有 feature_qc.json 或 pose_features.csv，无法展示 QC / warning 面板。")
        return

    processing_summary = qc_payload.get("processing_summary") if isinstance(qc_payload, dict) and isinstance(qc_payload.get("processing_summary"), dict) else {}
    feature_qc = qc_payload.get("feature_qc") if isinstance(qc_payload, dict) and isinstance(qc_payload.get("feature_qc"), dict) else {}

    st.subheader("处理摘要")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows", int(processing_summary.get("total_rows", 0)))
    c2.metric("OK Rows", int(processing_summary.get("ok_rows", 0)))
    c3.metric("Failed Rows", int(processing_summary.get("failed_rows", 0)))
    c4.metric("Rows With Warning", int(processing_summary.get("rows_with_warning_message", 0)))

    artifacts = summary.get("artifacts") if isinstance(summary, dict) and isinstance(summary.get("artifacts"), dict) else {}
    quality_gate_summary_path = (
        Path(str(artifacts.get("quality_gate_summary_json")))
        if artifacts.get("quality_gate_summary_json")
        else _resolve_output_file("quality_gate", "quality_gate_summary.json", summary=summary, metadata=metadata)
    )
    quality_gate_checks_path = (
        Path(str(artifacts.get("quality_gate_checks_csv")))
        if artifacts.get("quality_gate_checks_csv")
        else _resolve_output_file("quality_gate", "quality_gate_checks.csv", summary=summary, metadata=metadata)
    )
    quality_gate_report_path = (
        Path(str(artifacts.get("quality_gate_report_md")))
        if artifacts.get("quality_gate_report_md")
        else _resolve_output_file("quality_gate", "quality_gate_report.md", summary=summary, metadata=metadata)
    )
    quality_gate_summary = _load_json(quality_gate_summary_path) if quality_gate_summary_path is not None and quality_gate_summary_path.exists() else {}
    if quality_gate_summary:
        st.subheader("统一质量判定")
        gate_col1, gate_col2, gate_col3, gate_col4 = st.columns(4)
        gate_status = str(quality_gate_summary.get("overall_status") or "WARN").upper()
        gate_col1.metric("Quality Gate", gate_status)
        gate_col2.metric("PASS", int(quality_gate_summary.get("pass_count") or 0))
        gate_col3.metric("WARN", int(quality_gate_summary.get("warn_count") or 0))
        gate_col4.metric("FAIL", int(quality_gate_summary.get("fail_count") or 0))
        gate_message = str(quality_gate_summary.get("decision") or "")
        if gate_status == "PASS":
            st.success(gate_message)
        elif gate_status == "FAIL":
            st.error(gate_message)
        else:
            st.warning(gate_message)
        if quality_gate_checks_path is not None and quality_gate_checks_path.exists():
            quality_checks_df = _load_csv(quality_gate_checks_path)
            if quality_checks_df is not None and not quality_checks_df.empty:
                st.dataframe(quality_checks_df, use_container_width=True, hide_index=True)
        qg_col1, qg_col2 = st.columns(2)
        if quality_gate_checks_path is not None and quality_gate_checks_path.exists():
            qg_col1.download_button(
                label="下载质量判定检查 CSV",
                data=quality_gate_checks_path.read_bytes(),
                file_name=quality_gate_checks_path.name,
                mime="text/csv",
                use_container_width=True,
                key="download_quality_gate_checks_csv",
            )
        if quality_gate_report_path is not None and quality_gate_report_path.exists():
            qg_col2.download_button(
                label="下载质量判定报告 Markdown",
                data=quality_gate_report_path.read_bytes(),
                file_name=quality_gate_report_path.name,
                mime="text/markdown",
                use_container_width=True,
                key="download_quality_gate_report_md",
            )
            with st.expander("查看质量判定 Markdown"):
                st.markdown(_read_text(quality_gate_report_path))

    geometry_audit_summary_path = (
        Path(str(artifacts.get("geometry_proxy_audit_summary_json")))
        if artifacts.get("geometry_proxy_audit_summary_json")
        else _resolve_output_file("geometry_proxy_audit", "geometry_proxy_audit_summary.json", summary=summary, metadata=metadata)
    )
    geometry_audit_report_path = (
        Path(str(artifacts.get("geometry_proxy_audit_report_md")))
        if artifacts.get("geometry_proxy_audit_report_md")
        else _resolve_output_file("geometry_proxy_audit", "geometry_proxy_audit_report.md", summary=summary, metadata=metadata)
    )
    geometry_flagged_path = (
        Path(str(artifacts.get("geometry_proxy_flagged_poses_csv")))
        if artifacts.get("geometry_proxy_flagged_poses_csv")
        else _resolve_output_file("geometry_proxy_audit", "geometry_proxy_flagged_poses.csv", summary=summary, metadata=metadata)
    )
    geometry_candidate_path = (
        Path(str(artifacts.get("geometry_proxy_candidate_audit_csv")))
        if artifacts.get("geometry_proxy_candidate_audit_csv")
        else _resolve_output_file("geometry_proxy_audit", "geometry_proxy_candidate_audit.csv", summary=summary, metadata=metadata)
    )
    geometry_audit_summary = (
        _load_json(geometry_audit_summary_path)
        if geometry_audit_summary_path is not None and geometry_audit_summary_path.exists()
        else {}
    )
    if geometry_audit_summary:
        st.subheader("几何 proxy 一致性审计")
        st.caption("检查 mouth/path/pocket/contact 等静态 proxy 是否互相矛盾；只做 QC 和解释，不改变 Rule/ML/Consensus 分数。")
        ga_col1, ga_col2, ga_col3, ga_col4 = st.columns(4)
        audit_status = str(geometry_audit_summary.get("audit_status") or "UNKNOWN").upper()
        ga_col1.metric("Proxy Audit", audit_status)
        ga_col2.metric("Flagged Poses", int(geometry_audit_summary.get("flagged_pose_count") or 0))
        ga_col3.metric("Flagged Fraction", _ratio_text(geometry_audit_summary.get("flagged_pose_fraction")))
        ga_col4.metric("Candidates", int(geometry_audit_summary.get("candidate_count") or 0))
        if audit_status == "PASS":
            st.success("几何 proxy 未发现批次级阻塞风险。")
        elif audit_status == "FAIL":
            st.error("几何 proxy 审计失败，建议先检查输入和特征表。")
        else:
            st.warning("几何 proxy 存在较多不一致信号，建议先复核 flagged rows。")

        flag_counts = geometry_audit_summary.get("flag_counts") if isinstance(geometry_audit_summary.get("flag_counts"), dict) else {}
        if flag_counts:
            st.dataframe(
                pd.DataFrame(
                    [{"flag": key, "count": value} for key, value in flag_counts.items()]
                ).sort_values(by="count", ascending=False),
                use_container_width=True,
                hide_index=True,
            )
        next_actions = geometry_audit_summary.get("next_actions") if isinstance(geometry_audit_summary.get("next_actions"), list) else []
        for action in next_actions[:4]:
            st.write(f"- {action}")

        ga_download_col1, ga_download_col2, ga_download_col3 = st.columns(3)
        if geometry_flagged_path is not None and geometry_flagged_path.exists():
            ga_download_col1.download_button(
                label="下载 flagged poses CSV",
                data=geometry_flagged_path.read_bytes(),
                file_name=geometry_flagged_path.name,
                mime="text/csv",
                use_container_width=True,
                key="download_geometry_proxy_flagged_poses_csv",
            )
        if geometry_candidate_path is not None and geometry_candidate_path.exists():
            ga_download_col2.download_button(
                label="下载 candidate audit CSV",
                data=geometry_candidate_path.read_bytes(),
                file_name=geometry_candidate_path.name,
                mime="text/csv",
                use_container_width=True,
                key="download_geometry_proxy_candidate_audit_csv",
            )
        if geometry_audit_report_path is not None and geometry_audit_report_path.exists():
            ga_download_col3.download_button(
                label="下载 proxy audit Markdown",
                data=geometry_audit_report_path.read_bytes(),
                file_name=geometry_audit_report_path.name,
                mime="text/markdown",
                use_container_width=True,
                key="download_geometry_proxy_audit_report_md",
            )
            with st.expander("查看几何 proxy 审计 Markdown"):
                st.markdown(_read_text(geometry_audit_report_path))

    split_mode_counts = processing_summary.get("split_mode_counts") if isinstance(processing_summary.get("split_mode_counts"), dict) else {}
    status_counts = feature_qc.get("status_counts") if isinstance(feature_qc.get("status_counts"), dict) else {}
    if split_mode_counts:
        st.caption("split_mode 分布: " + ", ".join([f"{k}={v}" for k, v in split_mode_counts.items()]))
    if status_counts:
        st.caption("status 分布: " + ", ".join([f"{k}={v}" for k, v in status_counts.items()]))

    all_empty_columns = feature_qc.get("all_empty_columns") if isinstance(feature_qc.get("all_empty_columns"), list) else []
    near_constant_columns = feature_qc.get("near_constant_columns") if isinstance(feature_qc.get("near_constant_columns"), list) else []
    meta_col1, meta_col2 = st.columns(2)
    if all_empty_columns:
        meta_col1.warning("全空列: " + ", ".join([str(col) for col in all_empty_columns[:12]]))
    else:
        meta_col1.success("当前没有全空列。")
    if near_constant_columns:
        meta_col2.warning("近常量列: " + ", ".join([str(col) for col in near_constant_columns[:12]]))
    else:
        meta_col2.success("当前没有近常量列。")

    pocket_shape_qc = feature_qc.get("pocket_shape_qc") if isinstance(feature_qc.get("pocket_shape_qc"), dict) else {}
    if pocket_shape_qc:
        st.subheader("Pocket Shape QC")
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Overwide 阈值", _metric_text(pocket_shape_qc.get("overwide_threshold")))
        q2.metric("高 Overwide 行数", int(pocket_shape_qc.get("high_overwide_row_count", 0)))
        q3.metric("高 Overwide 占比", _ratio_text(pocket_shape_qc.get("high_overwide_row_fraction")))
        q4.metric("Overwide Max", _metric_text(pocket_shape_qc.get("overwide_proxy_max")))
        top_overwide_rows = pocket_shape_qc.get("top_overwide_rows") if isinstance(pocket_shape_qc.get("top_overwide_rows"), list) else []
        if top_overwide_rows:
            st.caption("以下为 pocket_shape_overwide_proxy 最高的行，适合优先复核 pocket residue 边界。")
            st.dataframe(pd.DataFrame(top_overwide_rows), use_container_width=True)

    if isinstance(feature_df, pd.DataFrame) and not feature_df.empty:
        base_cols = [col for col in ["nanobody_id", "conformer_id", "pose_id", "pdb_path", "split_mode", "status", "error_message", "warning_message"] if col in feature_df.columns]
        failed_rows_df = None
        warning_rows_df = None
        if "status" in feature_df.columns:
            failed_rows_df = feature_df[feature_df["status"].astype(str).str.lower().eq("failed")]
            if base_cols:
                failed_rows_df = failed_rows_df.loc[:, base_cols]
        if "warning_message" in feature_df.columns:
            warning_rows_df = feature_df[feature_df["warning_message"].fillna("").astype(str).str.strip().ne("")]
            if base_cols:
                warning_rows_df = warning_rows_df.loc[:, base_cols]

        st.subheader("失败行")
        if failed_rows_df is None or failed_rows_df.empty:
            st.success("当前没有 failed 行。")
        else:
            failed_preview_rows = st.number_input(
                "失败行预览前 N 行",
                min_value=5,
                max_value=200,
                value=20,
                step=5,
                key="failed_rows_preview_rows",
            )
            st.caption(f"共 {len(failed_rows_df)} 行失败，当前展示前 {min(int(failed_preview_rows), len(failed_rows_df))} 行。")
            st.dataframe(failed_rows_df.head(int(failed_preview_rows)), use_container_width=True)

        st.subheader("Warning 行")
        if warning_rows_df is None or warning_rows_df.empty:
            st.success("当前没有 warning 行。")
        else:
            warning_preview_rows = st.number_input(
                "Warning 行预览前 N 行",
                min_value=5,
                max_value=200,
                value=20,
                step=5,
                key="warning_rows_preview_rows",
            )
            st.caption(f"共 {len(warning_rows_df)} 行带 warning，当前展示前 {min(int(warning_preview_rows), len(warning_rows_df))} 行。")
            st.dataframe(warning_rows_df.head(int(warning_preview_rows)), use_container_width=True)

    if isinstance(qc_payload, dict):
        with st.expander("查看完整 feature_qc.json"):
            st.json(qc_payload)


def _build_history_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not LOCAL_APP_RUN_ROOT.exists():
        return records

    for run_root in LOCAL_APP_RUN_ROOT.iterdir():
        if not run_root.is_dir() or run_root.name == PARAM_TEMPLATE_ROOT.name:
            continue

        metadata_path = run_root / APP_RUN_METADATA_NAME
        summary_path = run_root / "outputs" / "recommended_pipeline_summary.json"
        metadata = _load_json(metadata_path) or {}
        summary = _load_json(summary_path) or {}
        if not metadata and not summary:
            continue

        status = str(metadata.get("status") or ("success" if summary else "unknown"))
        started_at = str(
            metadata.get("started_at")
            or datetime.fromtimestamp(run_root.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        )
        start_mode = str(metadata.get("start_mode") or summary.get("start_mode") or "N/A")
        stage = str((metadata.get("diagnostics") or {}).get("failed_stage") or "N/A")
        label_valid_count = int(summary.get("label_valid_count", 0)) if isinstance(summary, dict) else 0
        label = f"{run_root.name} | {status} | {started_at} | {start_mode}"

        records.append(
            {
                "label": label,
                "run_root": run_root,
                "run_name": run_root.name,
                "status": status,
                "started_at": started_at,
                "start_mode": start_mode,
                "failed_stage": stage,
                "label_valid_count": label_valid_count,
            }
        )

    records.sort(key=lambda item: str(item.get("started_at", "")), reverse=True)
    return records


def _build_history_table(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in records:
        rows.append(
            {
                "run_name": item.get("run_name"),
                "status": item.get("status"),
                "started_at": item.get("started_at"),
                "start_mode": item.get("start_mode"),
                "label_valid_count": item.get("label_valid_count"),
                "failed_stage": item.get("failed_stage"),
            }
        )
    return pd.DataFrame(rows)


def _build_and_save_global_experiment_ledger() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Path]]:
    latest_df, history_df, summary = build_global_experiment_ledger(LOCAL_APP_RUN_ROOT)
    latest_path = LOCAL_APP_RUN_ROOT / "experiment_state_ledger_global.csv"
    history_path = LOCAL_APP_RUN_ROOT / "experiment_state_ledger_global_history.csv"
    summary_path = LOCAL_APP_RUN_ROOT / "experiment_state_ledger_global_summary.json"
    latest_df.to_csv(latest_path, index=False)
    history_df.to_csv(history_path, index=False)
    _write_json(summary_path, summary)
    return latest_df, history_df, summary, {
        "latest_csv": latest_path,
        "history_csv": history_path,
        "summary_json": summary_path,
    }


def _sorted_non_empty_column_values(df: pd.DataFrame, column: str) -> list[str]:
    if df.empty or column not in df.columns:
        return []
    values = df[column].map(_clean_cell_text)
    values = values.loc[values.ne("")]
    return sorted(values.unique().tolist())


def _filter_by_column_values(df: pd.DataFrame, column: str, selected_values: list[str]) -> pd.DataFrame:
    if df.empty or column not in df.columns or not selected_values:
        return df
    selected = {str(value) for value in selected_values}
    mask = df[column].map(_clean_cell_text).isin(selected)
    return df.loc[mask].copy()


def _build_value_count_table(df: pd.DataFrame, column: str, label: str) -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return pd.DataFrame(columns=[label, "count"])
    values = df[column].map(_clean_cell_text)
    values = values.where(values.ne(""), "empty")
    counts = values.value_counts(dropna=False).rename_axis(label).reset_index(name="count")
    return counts


def _experiment_label_ready_count(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    ready_mask = pd.Series(False, index=df.index)
    if "validation_label" in df.columns:
        ready_mask = ready_mask | df["validation_label"].map(_clean_cell_text).ne("")
    if "experiment_result" in df.columns:
        result_values = df["experiment_result"].map(lambda value: _clean_cell_text(value).lower())
        ready_mask = ready_mask | result_values.isin({"positive", "negative", "1", "0", "true", "false"})
    return int(ready_mask.sum())


def _render_global_experiment_ledger_panel() -> None:
    st.subheader("全局实验状态 ledger")
    st.caption(
        "从所有 `local_app_runs` 历史运行中汇总 `experiment_plan_state_ledger.csv` "
        "和已编辑的 `experiment_plan_override_edited.csv`，每个 nanobody 默认保留最新状态。"
    )
    build_clicked = st.button(
        "扫描历史并生成全局 ledger",
        use_container_width=True,
        key="build_global_experiment_ledger",
    )
    if not build_clicked and not (LOCAL_APP_RUN_ROOT / "experiment_state_ledger_global.csv").exists():
        st.info("点击上方按钮生成全局实验状态表；生成后可下载或设置为下一轮 override。")
        return

    try:
        if build_clicked:
            latest_df, history_df, summary, paths = _build_and_save_global_experiment_ledger()
        else:
            latest_path = LOCAL_APP_RUN_ROOT / "experiment_state_ledger_global.csv"
            history_path = LOCAL_APP_RUN_ROOT / "experiment_state_ledger_global_history.csv"
            summary_path = LOCAL_APP_RUN_ROOT / "experiment_state_ledger_global_summary.json"
            loaded_latest_df = _load_csv(latest_path)
            loaded_history_df = _load_csv(history_path)
            latest_df = loaded_latest_df if loaded_latest_df is not None else pd.DataFrame()
            history_df = loaded_history_df if loaded_history_df is not None else pd.DataFrame()
            summary = _load_json(summary_path) or {}
            paths = {
                "latest_csv": latest_path,
                "history_csv": history_path,
                "summary_json": summary_path,
            }
    except Exception as exc:
        st.error(f"生成全局实验状态 ledger 失败: {exc}")
        return

    if latest_df.empty:
        st.info("没有在历史运行中找到实验计划状态 ledger 或编辑后的 override 文件。")
        return

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Nanobody 数", int(summary.get("unique_nanobody_count") or len(latest_df)))
    metric_col2.metric("最新状态行", int(summary.get("latest_row_count") or len(latest_df)))
    metric_col3.metric("历史状态记录", int(summary.get("history_row_count") or len(history_df)))

    status_counts = summary.get("status_counts") if isinstance(summary.get("status_counts"), dict) else {}
    if status_counts:
        st.caption("状态分布: " + ", ".join([f"{key}={value}" for key, value in status_counts.items()]))

    with st.expander("ledger 快速筛选与状态分布", expanded=True):
        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
        selected_statuses = filter_col1.multiselect(
            "experiment_status",
            options=_sorted_non_empty_column_values(latest_df, "experiment_status"),
            default=[],
            key="global_ledger_filter_status",
        )
        selected_results = filter_col2.multiselect(
            "experiment_result",
            options=_sorted_non_empty_column_values(latest_df, "experiment_result"),
            default=[],
            key="global_ledger_filter_result",
        )
        selected_overrides = filter_col3.multiselect(
            "plan_override",
            options=_sorted_non_empty_column_values(latest_df, "plan_override"),
            default=[],
            key="global_ledger_filter_override",
        )
        keyword = str(
            filter_col4.text_input(
                "关键词",
                key="global_ledger_filter_keyword",
                placeholder="nanobody / owner / note",
            )
            or ""
        ).strip().lower()

        filtered_latest_df = latest_df.copy()
        filtered_latest_df = _filter_by_column_values(filtered_latest_df, "experiment_status", selected_statuses)
        filtered_latest_df = _filter_by_column_values(filtered_latest_df, "experiment_result", selected_results)
        filtered_latest_df = _filter_by_column_values(filtered_latest_df, "plan_override", selected_overrides)
        if keyword:
            searchable_columns = [
                column
                for column in ["nanobody_id", "experiment_owner", "experiment_note", "latest_source_run_name"]
                if column in filtered_latest_df.columns
            ]
            if searchable_columns:
                keyword_mask = pd.Series(False, index=filtered_latest_df.index)
                for column in searchable_columns:
                    keyword_mask = keyword_mask | filtered_latest_df[column].map(_clean_cell_text).str.lower().str.contains(
                        keyword,
                        regex=False,
                        na=False,
                    )
                filtered_latest_df = filtered_latest_df.loc[keyword_mask].copy()

        quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)
        quick_col1.metric("筛选后行数", int(len(filtered_latest_df)))
        quick_col2.metric(
            "completed",
            int(
                filtered_latest_df.get("experiment_status", pd.Series(dtype=object))
                .map(lambda value: _clean_cell_text(value).lower())
                .eq("completed")
                .sum()
            ),
        )
        quick_col3.metric("可回灌标签", _experiment_label_ready_count(filtered_latest_df))
        quick_col4.metric(
            "blocked/cancelled",
            int(
                filtered_latest_df.get("experiment_status", pd.Series(dtype=object))
                .map(lambda value: _clean_cell_text(value).lower())
                .isin({"blocked", "cancelled", "canceled"})
                .sum()
            ),
        )

        chart_col1, chart_col2 = st.columns(2)
        status_chart_df = _build_value_count_table(filtered_latest_df, "experiment_status", "status")
        if not status_chart_df.empty:
            chart_col1.caption("实验状态分布")
            chart_col1.bar_chart(status_chart_df.set_index("status"))
        result_chart_df = _build_value_count_table(filtered_latest_df, "experiment_result", "result")
        if not result_chart_df.empty:
            chart_col2.caption("实验结果分布")
            chart_col2.bar_chart(result_chart_df.set_index("result"))
        st.caption(f"当前筛选保留 {len(filtered_latest_df)} / {len(latest_df)} 条；下载按钮会导出当前表格筛选后的可见排序。")

    preferred_columns = [
        "nanobody_id",
        "plan_override",
        "experiment_status",
        "experiment_result",
        "validation_label",
        "experiment_owner",
        "experiment_cost",
        "experiment_note",
        "last_plan_decision",
        "last_plan_rank",
        "latest_source_run_name",
        "latest_source_kind",
        "latest_source_mtime",
        "source_record_count",
    ]
    view_df, export_df = _render_dataframe_view_controls(
        filtered_latest_df,
        key_prefix="global_experiment_ledger_view",
        preferred_columns=preferred_columns,
        default_sort_column="latest_source_mtime",
        default_descending=True,
    )
    st.dataframe(view_df.head(200), use_container_width=True)

    action_col1, action_col2, action_col3 = st.columns(3)
    action_col1.download_button(
        label="下载全局最新 ledger CSV",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=paths["latest_csv"].name,
        mime="text/csv",
        use_container_width=True,
        key="download_global_experiment_ledger_csv",
    )
    action_col2.download_button(
        label="下载完整历史状态 CSV",
        data=history_df.to_csv(index=False).encode("utf-8"),
        file_name=paths["history_csv"].name,
        mime="text/csv",
        use_container_width=True,
        key="download_global_experiment_ledger_history_csv",
    )
    use_global_clicked = action_col3.button(
        "设为下一轮 override",
        use_container_width=True,
        key="use_global_experiment_ledger_as_override",
    )
    if use_global_clicked:
        st.session_state["_pending_experiment_plan_override_local_path"] = str(paths["latest_csv"])
        st.session_state["_pending_experiment_plan_override_message"] = (
            "已把全局实验状态 ledger 设置为下一轮 override 输入。"
        )
        st.success("已设置为下一轮 override，页面将刷新并回填左侧路径。")
        st.rerun()

    with st.expander("生成真实验证回灌报告"):
        st.caption(
            "只把明确的 `validation_label` 或 `experiment_result=positive/negative` 转成训练标签；"
            "`completed` 但没有结果的候选只会进入待补结果清单。"
        )
        feedback_out_dir = LOCAL_APP_RUN_ROOT / "experiment_validation_feedback"
        st.text_input(
            "可选 pose_features.csv 路径",
            key="validation_feedback_feature_csv_path",
            help="填写后会额外生成 pose_features_with_experiment_labels.csv，可直接作为下一轮 feature_csv 输入。",
        )
        st.text_input(
            "可选 consensus_ranking.csv 路径",
            key="validation_feedback_consensus_csv_path",
            help="填写后报告会合并共识排名信息，便于判断验证结果与当前排序是否一致。",
        )
        st.text_input(
            "回灌标签列名",
            key="validation_feedback_label_col",
            help="写入带标签特征表的列名；建议保留 experiment_label，避免覆盖原始 label。",
        )
        feedback_clicked = st.button(
            "生成验证回灌报告",
            use_container_width=True,
            key="build_experiment_validation_feedback",
        )
        feedback_summary_path = feedback_out_dir / "experiment_validation_summary.json"
        if feedback_clicked:
            try:
                feedback_summary = build_experiment_validation_feedback(
                    ledger_csv=paths["latest_csv"],
                    out_dir=feedback_out_dir,
                    feature_csv=str(st.session_state.get("validation_feedback_feature_csv_path") or "").strip()
                    or None,
                    consensus_csv=str(st.session_state.get("validation_feedback_consensus_csv_path") or "").strip()
                    or None,
                    label_col=str(st.session_state.get("validation_feedback_label_col") or "experiment_label"),
                )
                st.success(f"已生成验证回灌报告: {feedback_out_dir}")
            except Exception as exc:
                st.error(f"生成验证回灌报告失败: {exc}")
                feedback_summary = {}
        else:
            feedback_summary = _load_json(feedback_summary_path) or {}

        if isinstance(feedback_summary, dict) and feedback_summary:
            fb_col1, fb_col2, fb_col3 = st.columns(3)
            fb_col1.metric("可回灌标签", int(feedback_summary.get("label_ready_count") or 0))
            fb_col2.metric("阳性", int(feedback_summary.get("positive_label_count") or 0))
            fb_col3.metric("阴性", int(feedback_summary.get("negative_label_count") or 0))
            outputs = feedback_summary.get("outputs") if isinstance(feedback_summary.get("outputs"), dict) else {}
            feedback_downloads = [
                ("下载验证状态报告 CSV", outputs.get("experiment_validation_status_report_csv"), "text/csv"),
                ("下载可回灌标签 CSV", outputs.get("experiment_validation_labels_csv"), "text/csv"),
                ("下载验证回灌 Markdown", outputs.get("experiment_validation_report_md"), "text/markdown"),
                (
                    "下载带实验标签的特征表 CSV",
                    outputs.get("pose_features_with_experiment_labels_csv"),
                    "text/csv",
                ),
            ]
            for idx, (label, path_text, mime) in enumerate(feedback_downloads):
                path = Path(str(path_text)) if path_text else None
                if path is not None and path.exists():
                    st.download_button(
                        label=label,
                        data=path.read_bytes(),
                        file_name=path.name,
                        mime=mime,
                        use_container_width=True,
                        key=f"download_validation_feedback_{idx}",
                    )

            feature_out = outputs.get("pose_features_with_experiment_labels_csv")
            feature_out_path = Path(str(feature_out)) if feature_out else None
            if feature_out_path is not None and feature_out_path.exists():
                st.caption(
                    "可将该特征表设为下一轮输入，并自动把左侧 `label_col` 切换为 "
                    f"`{str(st.session_state.get('validation_feedback_label_col') or 'experiment_label')}`。"
                )
                if st.button(
                    "使用带实验标签特征表作为下一轮输入",
                    use_container_width=True,
                    key="use_validation_labeled_feature_csv",
                ):
                    label_col = str(st.session_state.get("validation_feedback_label_col") or "experiment_label")
                    st.session_state["_pending_feature_csv_local_path"] = str(feature_out_path)
                    st.session_state["_pending_feature_label_col"] = label_col
                    st.session_state["_pending_feature_csv_message"] = (
                        f"已把验证回灌特征表设置为下一轮 feature_csv，并使用 label_col={label_col}。"
                    )
                    st.success("已设置为下一轮输入，页面将刷新并回填左侧路径。")
                    st.rerun()


def _render_result_archive_panel() -> None:
    st.subheader("结果自动归档与长期趋势")
    st.caption(
        "扫描 `local_app_runs`，生成运行级索引、关键产物 manifest、跨批次 lineage，以及验证回灌再训练对照的长期趋势表。"
        "该步骤只建立索引，不复制大文件、不改变历史运行结果。"
    )
    archive_dir = LOCAL_APP_RUN_ROOT / "result_archive"
    build_clicked = st.button(
        "生成/刷新结果归档索引",
        use_container_width=True,
        key="build_result_archive_index",
    )
    if build_clicked:
        try:
            archive_summary = build_result_archive(local_app_runs=LOCAL_APP_RUN_ROOT, out_dir=archive_dir)
            st.success(f"已生成结果归档索引: {archive_dir}")
        except Exception as exc:
            st.error(f"生成结果归档索引失败: {exc}")
            archive_summary = {}
    else:
        archive_summary = _load_json(archive_dir / "result_archive_summary.json") or {}

    if not archive_summary:
        st.info("点击上方按钮生成结果归档索引。")
        return

    ar_col1, ar_col2, ar_col3 = st.columns(3)
    ar_col1.metric("归档运行数", int(archive_summary.get("run_count") or 0))
    ar_col2.metric("已存在产物数", int(archive_summary.get("existing_artifact_count") or 0))
    ar_col3.metric("再训练对照数", int(archive_summary.get("validation_retrain_comparison_count") or 0))
    lineage_col1, lineage_col2, lineage_col3 = st.columns(3)
    lineage_col1.metric("lineage 运行数", int(archive_summary.get("lineage_row_count") or 0))
    lineage_col2.metric("共享输入 manifest 组", int(archive_summary.get("shared_input_manifest_group_count") or 0))
    lineage_col3.metric("共享参数组", int(archive_summary.get("shared_parameter_hash_group_count") or 0))
    graph_col1, graph_col2 = st.columns(2)
    graph_col1.metric("lineage 图边数", int(archive_summary.get("lineage_graph_edge_count") or 0))
    graph_col2.metric("共享 lineage 图组", int(archive_summary.get("lineage_graph_shared_group_count") or 0))

    outputs = archive_summary.get("outputs") if isinstance(archive_summary.get("outputs"), dict) else {}
    runs_csv = Path(str(outputs.get("runs_csv"))) if outputs.get("runs_csv") else archive_dir / "result_archive_runs.csv"
    artifacts_csv = (
        Path(str(outputs.get("artifact_manifest_csv")))
        if outputs.get("artifact_manifest_csv")
        else archive_dir / "result_archive_artifact_manifest.csv"
    )
    trends_csv = (
        Path(str(outputs.get("validation_retrain_trends_csv")))
        if outputs.get("validation_retrain_trends_csv")
        else archive_dir / "result_archive_validation_retrain_trends.csv"
    )
    lineage_csv = (
        Path(str(outputs.get("lineage_csv")))
        if outputs.get("lineage_csv")
        else archive_dir / "result_archive_lineage.csv"
    )
    lineage_graph_json = (
        Path(str(outputs.get("lineage_graph_json")))
        if outputs.get("lineage_graph_json")
        else archive_dir / "result_archive_lineage_graph.json"
    )
    lineage_graph_html = (
        Path(str(outputs.get("lineage_graph_html")))
        if outputs.get("lineage_graph_html")
        else archive_dir / "result_archive_lineage_graph.html"
    )
    lineage_graph_md = (
        Path(str(outputs.get("lineage_graph_md")))
        if outputs.get("lineage_graph_md")
        else archive_dir / "result_archive_lineage_graph.md"
    )
    report_md = Path(str(outputs.get("report_md"))) if outputs.get("report_md") else archive_dir / "result_archive_report.md"

    if runs_csv.exists():
        run_archive_df = _load_csv(runs_csv)
        if run_archive_df is not None and not run_archive_df.empty:
            st.caption("最近归档运行")
            st.dataframe(run_archive_df.head(100), use_container_width=True, hide_index=True)
        st.download_button(
            label="下载运行归档索引 CSV",
            data=runs_csv.read_bytes(),
            file_name=runs_csv.name,
            mime="text/csv",
            use_container_width=True,
            key="download_result_archive_runs_csv",
        )
    if trends_csv.exists():
        trend_df = _load_csv(trends_csv)
        if trend_df is not None and not trend_df.empty:
            st.caption("验证回灌再训练长期趋势")
            chart_columns = [
                column
                for column in ["label_valid_delta", "calibrated_rank_spearman_delta", "best_val_loss_delta"]
                if column in trend_df.columns and pd.to_numeric(trend_df[column], errors="coerce").notna().any()
            ]
            if chart_columns and "comparison_name" in trend_df.columns:
                chart_df = trend_df.loc[:, ["comparison_name"] + chart_columns].copy()
                for column in chart_columns:
                    chart_df[column] = pd.to_numeric(chart_df[column], errors="coerce")
                st.bar_chart(chart_df.set_index("comparison_name"), height=260)
            st.dataframe(trend_df.head(100), use_container_width=True, hide_index=True)
        st.download_button(
            label="下载再训练长期趋势 CSV",
            data=trends_csv.read_bytes(),
            file_name=trends_csv.name,
            mime="text/csv",
            use_container_width=True,
            key="download_result_archive_validation_trends_csv",
        )
    if lineage_csv.exists():
        lineage_df = _load_csv(lineage_csv)
        if lineage_df is not None and not lineage_df.empty:
            st.caption("跨批次 lineage")
            st.dataframe(lineage_df.head(100), use_container_width=True, hide_index=True)
        st.download_button(
            label="下载跨批次 lineage CSV",
            data=lineage_csv.read_bytes(),
            file_name=lineage_csv.name,
            mime="text/csv",
            use_container_width=True,
            key="download_result_archive_lineage_csv",
        )
    if lineage_graph_html.exists():
        st.download_button(
            label="下载 lineage 图形化 HTML",
            data=lineage_graph_html.read_bytes(),
            file_name=lineage_graph_html.name,
            mime="text/html",
            use_container_width=True,
            key="download_result_archive_lineage_graph_html",
        )
        with st.expander("预览 lineage 图形化结果"):
            st.components.v1.html(_read_text(lineage_graph_html), height=620, scrolling=True)
    if lineage_graph_json.exists():
        st.download_button(
            label="下载 lineage graph JSON",
            data=lineage_graph_json.read_bytes(),
            file_name=lineage_graph_json.name,
            mime="application/json",
            use_container_width=True,
            key="download_result_archive_lineage_graph_json",
        )
    if lineage_graph_md.exists():
        st.download_button(
            label="下载 lineage graph Markdown",
            data=lineage_graph_md.read_bytes(),
            file_name=lineage_graph_md.name,
            mime="text/markdown",
            use_container_width=True,
            key="download_result_archive_lineage_graph_md",
        )
    if artifacts_csv.exists():
        st.download_button(
            label="下载产物 manifest CSV",
            data=artifacts_csv.read_bytes(),
            file_name=artifacts_csv.name,
            mime="text/csv",
            use_container_width=True,
            key="download_result_archive_artifact_manifest_csv",
        )
    if report_md.exists():
        st.download_button(
            label="下载归档 Markdown 报告",
            data=report_md.read_bytes(),
            file_name=report_md.name,
            mime="text/markdown",
            use_container_width=True,
            key="download_result_archive_report_md",
        )
        with st.expander("预览归档 Markdown 报告"):
            st.markdown(_read_text(report_md))

    if st.button("打开结果归档目录", use_container_width=True, key="open_result_archive_dir"):
        ok, message = _open_local_path(archive_dir)
        if ok:
            st.success(message)
        else:
            st.error(message)


def _compute_directory_size(path: Path) -> int:
    total_size = 0
    if not path.exists():
        return total_size
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total_size += int(child.stat().st_size)
            except OSError:
                continue
    return total_size


def _filter_history_records(
    records: list[dict[str, Any]],
    *,
    keyword: str = "",
    statuses: list[str] | None = None,
    start_modes: list[str] | None = None,
) -> list[dict[str, Any]]:
    keyword_text = str(keyword or "").strip().lower()
    selected_statuses = {
        str(item).strip()
        for item in (statuses or [])
        if str(item).strip()
    }
    selected_start_modes = {
        str(item).strip()
        for item in (start_modes or [])
        if str(item).strip()
    }

    filtered_records: list[dict[str, Any]] = []
    for item in records:
        item_status = str(item.get("status") or "").strip()
        item_start_mode = str(item.get("start_mode") or "").strip()
        if selected_statuses and item_status not in selected_statuses:
            continue
        if selected_start_modes and item_start_mode not in selected_start_modes:
            continue

        if keyword_text:
            haystack = " | ".join(
                [
                    str(item.get("run_name") or ""),
                    item_status,
                    str(item.get("started_at") or ""),
                    item_start_mode,
                    str(item.get("failed_stage") or ""),
                    str(item.get("label_valid_count") or ""),
                ]
            ).lower()
            if keyword_text not in haystack:
                continue

        filtered_records.append(item)
    return filtered_records


def _render_history_filter_controls(
    records: list[dict[str, Any]],
    *,
    key_prefix: str,
    title: str = "历史筛选",
) -> list[dict[str, Any]]:
    status_options = sorted(
        {
            str(item.get("status") or "").strip()
            for item in records
            if str(item.get("status") or "").strip()
        }
    )
    start_mode_options = sorted(
        {
            str(item.get("start_mode") or "").strip()
            for item in records
            if str(item.get("start_mode") or "").strip()
        }
    )

    st.caption(title)
    filter_col1, filter_col2, filter_col3 = st.columns([2, 1, 1])
    keyword = str(
        filter_col1.text_input(
            "关键词",
            key=f"{key_prefix}_history_keyword",
            placeholder="run_name / status / failed_stage",
        )
        or ""
    ).strip()
    selected_statuses = filter_col2.multiselect(
        "状态",
        options=status_options,
        default=status_options,
        key=f"{key_prefix}_history_statuses",
    )
    selected_start_modes = filter_col3.multiselect(
        "启动方式",
        options=start_mode_options,
        default=start_mode_options,
        key=f"{key_prefix}_history_start_modes",
    )
    filtered_records = _filter_history_records(
        records,
        keyword=keyword,
        statuses=[str(item) for item in selected_statuses],
        start_modes=[str(item) for item in selected_start_modes],
    )
    st.caption(f"当前筛选后共有 {len(filtered_records)} 条历史记录。")
    return filtered_records


def _build_cleanup_targets(history_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []

    for item in history_records:
        run_root = Path(str(item.get("run_root") or "")).resolve()
        if not run_root.exists():
            continue
        targets.append(
            {
                "label": f"运行记录 | {str(item.get('run_name') or 'N/A')} | {str(item.get('status') or 'N/A')}",
                "target_type": "run_root",
                "name": str(item.get("run_name") or run_root.name),
                "created_at": str(item.get("started_at") or ""),
                "path": str(run_root),
                "size_bytes": _compute_directory_size(run_root),
            }
        )

    if COMPARE_EXPORT_ROOT.exists():
        compare_export_dirs = sorted(
            [path for path in COMPARE_EXPORT_ROOT.iterdir() if path.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for path in compare_export_dirs:
            targets.append(
                {
                    "label": f"对比导出 | {path.name}",
                    "target_type": "compare_export",
                    "name": path.name,
                    "created_at": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                    "path": str(path.resolve()),
                    "size_bytes": _compute_directory_size(path),
                }
            )

    if BUNDLE_IMPORT_ROOT.exists():
        bundle_import_dirs = sorted(
            [path for path in BUNDLE_IMPORT_ROOT.iterdir() if path.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for path in bundle_import_dirs:
            targets.append(
                {
                    "label": f"导入缓存 | {path.name}",
                    "target_type": "bundle_import",
                    "name": path.name,
                    "created_at": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                    "path": str(path.resolve()),
                    "size_bytes": _compute_directory_size(path),
                }
            )

    return targets


def _build_cleanup_target_table(targets: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in targets:
        rows.append(
            {
                "类别": str(item.get("target_type") or ""),
                "名称": str(item.get("name") or ""),
                "创建时间": str(item.get("created_at") or ""),
                "大小": _format_byte_count(int(item.get("size_bytes") or 0)),
                "路径": str(item.get("path") or ""),
            }
        )
    return pd.DataFrame(rows)


def _is_path_within_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _delete_cleanup_target(target: dict[str, Any]) -> str:
    target_path = Path(str(target.get("path") or "")).expanduser().resolve()
    if not target_path.exists() or not target_path.is_dir():
        raise FileNotFoundError(f"待清理目录不存在: {target_path}")

    protected_roots = {
        LOCAL_APP_RUN_ROOT.resolve(),
        PARAM_TEMPLATE_ROOT.resolve(),
        COMPARE_EXPORT_ROOT.resolve(),
        BUNDLE_IMPORT_ROOT.resolve(),
    }
    if target_path in protected_roots:
        raise ValueError("不能直接删除应用的根级目录。")
    if not _is_path_within_root(target_path, LOCAL_APP_RUN_ROOT):
        raise ValueError("只允许清理由应用生成的 local_app_runs 子目录。")

    shutil.rmtree(target_path)
    return f"已删除目录: {target_path}"


def _load_training_summary_payload(
    summary: dict[str, Any] | None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    training_summary_path = _resolve_output_file(
        "model_outputs",
        "training_summary.json",
        summary=summary,
        metadata=metadata,
    )
    if training_summary_path is None or not training_summary_path.exists():
        return None
    payload = _load_json(training_summary_path)
    return payload if isinstance(payload, dict) else None


def _coerce_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def _build_run_compare_row(
    record: dict[str, Any],
) -> dict[str, Any]:
    run_root = Path(str(record.get("run_root")))
    metadata = _load_json(run_root / APP_RUN_METADATA_NAME) or {}
    summary = _load_json(run_root / "outputs" / "recommended_pipeline_summary.json") or {}
    diagnostics = metadata.get("diagnostics") if isinstance(metadata.get("diagnostics"), dict) else {}
    comparison = summary.get("comparison") if isinstance(summary.get("comparison"), dict) else {}
    baseline = comparison.get("baseline_rule_vs_ml") if isinstance(comparison.get("baseline_rule_vs_ml"), dict) else {}
    calibrated = comparison.get("calibrated_rule_vs_ml") if isinstance(comparison.get("calibrated_rule_vs_ml"), dict) else {}

    training_payload = _load_training_summary_payload(summary, metadata)
    training_summary = training_payload.get("summary") if isinstance(training_payload, dict) and isinstance(training_payload.get("summary"), dict) else {}
    pseudo_block = training_payload.get("pseudo") if isinstance(training_payload, dict) and isinstance(training_payload.get("pseudo"), dict) else {}
    pseudo_distribution = pseudo_block.get("distribution") if isinstance(pseudo_block.get("distribution"), dict) else {}

    qc_payload = _load_feature_qc_payload(summary, metadata)
    processing_summary = qc_payload.get("processing_summary") if isinstance(qc_payload, dict) and isinstance(qc_payload.get("processing_summary"), dict) else {}

    started_at = str(
        metadata.get("started_at")
        or record.get("started_at")
        or datetime.fromtimestamp(run_root.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    )
    status = str(metadata.get("status") or record.get("status") or ("success" if summary else "unknown"))
    start_mode = str(metadata.get("start_mode") or summary.get("start_mode") or record.get("start_mode") or "N/A")

    return {
        "run_name": run_root.name,
        "status": status,
        "started_at": started_at,
        "start_mode": start_mode,
        "execution_mode": str(metadata.get("execution_mode") or "N/A"),
        "label_valid_count": int(summary.get("label_valid_count") or record.get("label_valid_count") or 0),
        "label_class_count": int(summary.get("label_class_count") or 0),
        "calibration_possible": bool(summary.get("calibration_possible", False)),
        "baseline_rank_spearman": baseline.get("rank_spearman"),
        "baseline_rule_auc": baseline.get("rule_auc"),
        "calibrated_rank_spearman": calibrated.get("rank_spearman"),
        "calibrated_rule_auc": calibrated.get("rule_auc"),
        "training_mode": str(training_payload.get("mode") or "N/A") if isinstance(training_payload, dict) else "N/A",
        "train_rows": (
            training_payload.get("n_rows_train", training_summary.get("train_size"))
            if isinstance(training_payload, dict)
            else None
        ),
        "val_rows": (
            training_payload.get("n_rows_val", training_summary.get("val_size"))
            if isinstance(training_payload, dict)
            else None
        ),
        "feature_count": training_payload.get("n_features") if isinstance(training_payload, dict) else None,
        "best_epoch": training_summary.get("best_epoch"),
        "best_val_loss": training_summary.get("best_val_loss"),
        "pseudo_positive_rate": pseudo_block.get(
            "pseudo_positive_rate",
            pseudo_distribution.get("pseudo_positive_rate"),
        ),
        "total_rows": processing_summary.get("total_rows"),
        "ok_rows": processing_summary.get("ok_rows"),
        "failed_rows": processing_summary.get("failed_rows"),
        "warning_rows": processing_summary.get("rows_with_warning_message"),
        "failed_stage": str(diagnostics.get("failed_stage") or record.get("failed_stage") or "N/A"),
        "diagnostic_category": str(diagnostics.get("category") or "N/A"),
    }


def _build_history_compare_table(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = [_build_run_compare_row(record) for record in records]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _pick_compare_leader(
    compare_df: pd.DataFrame,
    *,
    metric_key: str,
    higher_is_better: bool,
) -> tuple[str | None, float | None]:
    if compare_df.empty or metric_key not in compare_df.columns:
        return None, None

    best_run_name: str | None = None
    best_value: float | None = None
    for row in compare_df.to_dict(orient="records"):
        value = _coerce_float(row.get(metric_key))
        if value is None:
            continue
        if best_value is None:
            best_value = value
            best_run_name = str(row.get("run_name") or "")
            continue
        if higher_is_better and value > best_value:
            best_value = value
            best_run_name = str(row.get("run_name") or "")
        if not higher_is_better and value < best_value:
            best_value = value
            best_run_name = str(row.get("run_name") or "")
    return best_run_name, best_value


def _is_higher_better_compare_metric(metric_key: str) -> bool:
    return metric_key not in LOWER_IS_BETTER_COMPARE_METRICS


def _get_compare_metric_label(metric_key: str) -> str:
    return COMPARE_METRIC_LABELS.get(metric_key, metric_key)


def _sort_compare_df_for_trend(compare_df: pd.DataFrame) -> pd.DataFrame:
    ordered = compare_df.copy()
    ordered["_started_at_dt"] = pd.to_datetime(ordered.get("started_at"), errors="coerce")
    ordered["_original_order"] = list(range(len(ordered)))
    ordered = ordered.sort_values(
        by=["_started_at_dt", "_original_order"],
        ascending=[True, True],
        na_position="last",
    ).reset_index(drop=True)
    return ordered


def _build_compare_delta_table(
    compare_df: pd.DataFrame,
    *,
    baseline_run_name: str,
    metric_key: str,
) -> pd.DataFrame:
    if compare_df.empty:
        return pd.DataFrame()
    baseline_rows = compare_df[compare_df["run_name"].astype(str) == str(baseline_run_name)]
    if baseline_rows.empty:
        return pd.DataFrame()

    baseline_row = baseline_rows.iloc[0].to_dict()
    primary_higher_is_better = _is_higher_better_compare_metric(metric_key)
    tracked_metrics = [
        metric_key,
        "baseline_rank_spearman",
        "calibrated_rank_spearman",
        "baseline_rule_auc",
        "calibrated_rule_auc",
        "best_val_loss",
        "failed_rows",
        "warning_rows",
    ]

    rows: list[dict[str, Any]] = []
    for row in compare_df.to_dict(orient="records"):
        item = {
            "run_name": row.get("run_name"),
            "status": row.get("status"),
            "started_at": row.get("started_at"),
            "baseline_run": baseline_run_name,
            "primary_metric_key": metric_key,
            "primary_metric_value": row.get(metric_key),
            metric_key: row.get(metric_key),
        }

        current_primary = _coerce_float(row.get(metric_key))
        baseline_primary = _coerce_float(baseline_row.get(metric_key))
        if current_primary is not None and baseline_primary is not None:
            raw_delta = current_primary - baseline_primary
            item["primary_metric_delta"] = raw_delta
            item["primary_metric_improvement"] = raw_delta if primary_higher_is_better else -raw_delta
        else:
            item["primary_metric_delta"] = None
            item["primary_metric_improvement"] = None

        for tracked_metric in tracked_metrics:
            if tracked_metric == metric_key or tracked_metric not in compare_df.columns:
                continue
            current_value = _coerce_float(row.get(tracked_metric))
            baseline_value = _coerce_float(baseline_row.get(tracked_metric))
            item[tracked_metric] = row.get(tracked_metric)
            item[f"{tracked_metric}_delta"] = (
                None if current_value is None or baseline_value is None else current_value - baseline_value
            )

        rows.append(item)

    return pd.DataFrame(rows)


def _format_compare_driver_text(metric_key: str, delta_value: float) -> tuple[str, float]:
    label = _get_compare_metric_label(metric_key)
    higher_is_better = _is_higher_better_compare_metric(metric_key)
    quality_impact = float(delta_value) if higher_is_better else -float(delta_value)
    magnitude = abs(float(delta_value))

    if higher_is_better:
        if delta_value > 0:
            text = f"{label} 提升 {_metric_text(magnitude)}"
        else:
            text = f"{label} 下降 {_metric_text(magnitude)}"
    else:
        if delta_value < 0:
            text = f"{label} 降低 {_metric_text(magnitude)}"
        else:
            text = f"{label} 增加 {_metric_text(magnitude)}"
    return text, quality_impact


def _build_compare_attribution_payloads(
    delta_df: pd.DataFrame,
    *,
    baseline_run_name: str,
    metric_key: str,
    metric_label: str,
) -> dict[str, dict[str, Any]]:
    if delta_df.empty:
        return {}

    tracked_driver_keys = [
        key
        for key in [
            "baseline_rank_spearman",
            "calibrated_rank_spearman",
            "baseline_rule_auc",
            "calibrated_rule_auc",
            "best_val_loss",
            "failed_rows",
            "warning_rows",
        ]
        if key != metric_key
    ]

    payloads: dict[str, dict[str, Any]] = {}
    for row in delta_df.to_dict(orient="records"):
        run_name = str(row.get("run_name") or "")
        if not run_name:
            continue

        primary_improvement = _coerce_float(row.get("primary_metric_improvement"))
        positive_drivers: list[tuple[str, float]] = []
        negative_drivers: list[tuple[str, float]] = []
        context_lines: list[str] = []

        for driver_key in tracked_driver_keys:
            delta_value = _coerce_float(row.get(f"{driver_key}_delta"))
            if delta_value is None or abs(delta_value) < 1e-12:
                continue
            text, quality_impact = _format_compare_driver_text(driver_key, delta_value)
            if quality_impact > 0:
                positive_drivers.append((text, abs(quality_impact)))
            elif quality_impact < 0:
                negative_drivers.append((text, abs(quality_impact)))

        positive_drivers.sort(key=lambda item: item[1], reverse=True)
        negative_drivers.sort(key=lambda item: item[1], reverse=True)

        row_status = str(row.get("status") or "N/A")
        failed_stage = str(row.get("failed_stage") or "N/A")
        diagnostic_category = str(row.get("diagnostic_category") or "N/A")
        if row_status.lower() != "success":
            context_lines.append(f"当前运行状态为 {row_status}。")
        if failed_stage not in {"", "N/A", "None"}:
            context_lines.append(f"失败阶段: {failed_stage}。")
        if diagnostic_category not in {"", "N/A", "success"}:
            context_lines.append(f"诊断分类: {diagnostic_category}。")

        if run_name == str(baseline_run_name):
            summary = f"{run_name} 是当前基准运行，本行只作为比较参考。"
            detail_lines = [
                f"基准运行: {run_name}",
                f"{metric_label} 基准值: {_metric_text(row.get(metric_key))}",
                "该运行不做相对自身的归因拆解。",
            ]
            payloads[run_name] = {
                "run_name": run_name,
                "attribution_tag": "基准参考",
                "primary_metric_improvement": 0.0,
                "summary": summary,
                "detail_lines": detail_lines,
                "positive_driver_texts": [],
                "negative_driver_texts": [],
                "top_positive_drivers": "",
                "top_negative_drivers": "",
                "net_signal": "baseline",
            }
            continue

        if primary_improvement is None:
            metric_effect_text = f"{metric_label} 当前缺少可比较值。"
        elif primary_improvement > 0:
            metric_effect_text = f"{metric_label} 相对基准改善了 {_metric_text(primary_improvement)}。"
        elif primary_improvement < 0:
            metric_effect_text = f"{metric_label} 相对基准退化了 {_metric_text(abs(primary_improvement))}。"
        else:
            metric_effect_text = f"{metric_label} 与基准基本持平。"

        tag = "接近基准"
        if primary_improvement is not None:
            if primary_improvement > 0 and not negative_drivers:
                tag = "整体向好"
            elif primary_improvement > 0:
                tag = "向好但有代价"
            elif primary_improvement < 0 and positive_drivers:
                tag = "回退但有亮点"
            elif primary_improvement < 0:
                tag = "整体回退"

        positive_texts = [item[0] for item in positive_drivers[:3]]
        negative_texts = [item[0] for item in negative_drivers[:3]]

        summary_parts = [metric_effect_text]
        if positive_texts:
            summary_parts.append("主要正向驱动: " + "；".join(positive_texts) + "。")
        if negative_texts:
            summary_parts.append("主要负向拖累: " + "；".join(negative_texts) + "。")
        if context_lines:
            summary_parts.extend(context_lines)
        summary = " ".join(summary_parts)

        detail_lines = [f"运行: {run_name}", f"基准运行: {baseline_run_name}", metric_effect_text]
        if positive_texts:
            detail_lines.append("正向驱动: " + "；".join(positive_texts) + "。")
        if negative_texts:
            detail_lines.append("负向拖累: " + "；".join(negative_texts) + "。")
        detail_lines.extend(context_lines)

        payloads[run_name] = {
            "run_name": run_name,
            "attribution_tag": tag,
            "primary_metric_improvement": primary_improvement,
            "summary": summary,
            "detail_lines": detail_lines,
            "positive_driver_texts": positive_texts,
            "negative_driver_texts": negative_texts,
            "top_positive_drivers": "；".join(positive_texts),
            "top_negative_drivers": "；".join(negative_texts),
            "net_signal": "positive" if primary_improvement and primary_improvement > 0 else ("negative" if primary_improvement and primary_improvement < 0 else "mixed"),
        }

    return payloads


def _build_compare_attribution_table(
    attribution_payloads: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_name, payload in attribution_payloads.items():
        rows.append(
            {
                "run_name": run_name,
                "attribution_tag": payload.get("attribution_tag"),
                "primary_metric_improvement": payload.get("primary_metric_improvement"),
                "positive_drivers": payload.get("top_positive_drivers"),
                "negative_drivers": payload.get("top_negative_drivers"),
                "attribution_summary": payload.get("summary"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.rename(
        columns={
            "run_name": "run_name",
            "attribution_tag": "归因标签",
            "primary_metric_improvement": "主指标净改善",
            "positive_drivers": "主要正向驱动",
            "negative_drivers": "主要负向拖累",
            "attribution_summary": "归因摘要",
        }
    )


def _build_compare_insight_lines(
    compare_df: pd.DataFrame,
    *,
    baseline_run_name: str,
    metric_key: str,
    metric_label: str,
) -> list[str]:
    if compare_df.empty:
        return []

    delta_df = _build_compare_delta_table(
        compare_df,
        baseline_run_name=baseline_run_name,
        metric_key=metric_key,
    )
    if delta_df.empty:
        return []

    insight_lines: list[str] = []
    baseline_row = compare_df[compare_df["run_name"].astype(str) == str(baseline_run_name)].iloc[0].to_dict()
    baseline_metric = _coerce_float(baseline_row.get(metric_key))
    if baseline_metric is not None:
        insight_lines.append(
            f"当前基准运行是 {baseline_run_name}，其 {metric_label} = {_metric_text(baseline_metric)}。"
        )
    else:
        insight_lines.append(f"当前基准运行是 {baseline_run_name}，但该运行没有可用的 {metric_label}。")

    comparable_delta_df = delta_df[
        delta_df["run_name"].astype(str) != str(baseline_run_name)
    ].copy()
    comparable_delta_df["primary_metric_improvement_num"] = pd.to_numeric(
        comparable_delta_df.get("primary_metric_improvement"),
        errors="coerce",
    )
    attribution_payloads = _build_compare_attribution_payloads(
        delta_df,
        baseline_run_name=baseline_run_name,
        metric_key=metric_key,
        metric_label=metric_label,
    )

    if not comparable_delta_df.empty and comparable_delta_df["primary_metric_improvement_num"].notna().any():
        best_idx = comparable_delta_df["primary_metric_improvement_num"].idxmax()
        worst_idx = comparable_delta_df["primary_metric_improvement_num"].idxmin()
        best_row = comparable_delta_df.loc[best_idx]
        worst_row = comparable_delta_df.loc[worst_idx]
        best_delta = _coerce_float(best_row.get("primary_metric_improvement_num"))
        worst_delta = _coerce_float(worst_row.get("primary_metric_improvement_num"))
        if best_delta is not None and best_delta > 0:
            insight_lines.append(
                f"相对基准，提升最大的运行是 {best_row.get('run_name')}，"
                f"{metric_label} 改善了 {_metric_text(best_delta)}。"
            )
            best_payload = attribution_payloads.get(str(best_row.get("run_name") or ""))
            if isinstance(best_payload, dict) and best_payload.get("top_positive_drivers"):
                insight_lines.append(
                    f"该运行的主要正向驱动: {best_payload['top_positive_drivers']}。"
                )
        if worst_delta is not None and worst_delta < 0:
            insight_lines.append(
                f"相对基准，回退最明显的运行是 {worst_row.get('run_name')}，"
                f"{metric_label} 退化了 {_metric_text(abs(worst_delta))}。"
            )
            worst_payload = attribution_payloads.get(str(worst_row.get("run_name") or ""))
            if isinstance(worst_payload, dict) and worst_payload.get("top_negative_drivers"):
                insight_lines.append(
                    f"该运行的主要负向拖累: {worst_payload['top_negative_drivers']}。"
                )

    baseline_failed = _coerce_float(baseline_row.get("failed_rows")) or 0.0
    baseline_warning = _coerce_float(baseline_row.get("warning_rows")) or 0.0
    failed_regressions = []
    warning_regressions = []
    for row in compare_df.to_dict(orient="records"):
        run_name = str(row.get("run_name") or "")
        if run_name == str(baseline_run_name):
            continue
        failed_value = _coerce_float(row.get("failed_rows")) or 0.0
        warning_value = _coerce_float(row.get("warning_rows")) or 0.0
        if failed_value > baseline_failed:
            failed_regressions.append(f"{run_name}(+{int(failed_value - baseline_failed)})")
        if warning_value > baseline_warning:
            warning_regressions.append(f"{run_name}(+{int(warning_value - baseline_warning)})")

    if failed_regressions:
        insight_lines.append(
            "相对基准，failed_rows 增加的运行有: " + ", ".join(failed_regressions[:6]) + "。"
        )
    if warning_regressions:
        insight_lines.append(
            "相对基准，warning_rows 增加的运行有: " + ", ".join(warning_regressions[:6]) + "。"
        )

    clean_mask = (
        compare_df["status"].astype(str).str.lower().eq("success")
        & pd.to_numeric(compare_df.get("failed_rows"), errors="coerce").fillna(0).eq(0)
        & pd.to_numeric(compare_df.get("warning_rows"), errors="coerce").fillna(0).eq(0)
    )
    clean_runs = compare_df.loc[clean_mask, "run_name"].astype(str).tolist()
    if clean_runs:
        insight_lines.append("当前选中运行里的 clean run: " + ", ".join(clean_runs[:6]) + "。")

    return insight_lines


def _build_compare_batch_trend_table(
    compare_df: pd.DataFrame,
    *,
    metric_key: str,
) -> pd.DataFrame:
    if compare_df.empty:
        return pd.DataFrame()

    working_df = compare_df.copy()
    working_df["_original_order"] = list(range(len(working_df)))
    working_df["_started_at_dt"] = pd.to_datetime(working_df.get("started_at"), errors="coerce")
    working_df["_batch_dt"] = working_df["_started_at_dt"].dt.normalize()
    working_df["_batch_label"] = working_df["_batch_dt"].dt.strftime("%Y-%m-%d")
    working_df["_batch_label"] = working_df["_batch_label"].fillna("unknown_batch")
    working_df["_status_success"] = working_df["status"].astype(str).str.lower().eq("success")
    working_df["_failed_rows_num"] = pd.to_numeric(working_df.get("failed_rows"), errors="coerce").fillna(0.0)
    working_df["_warning_rows_num"] = pd.to_numeric(working_df.get("warning_rows"), errors="coerce").fillna(0.0)
    working_df["_clean_run"] = (
        working_df["_status_success"]
        & working_df["_failed_rows_num"].eq(0)
        & working_df["_warning_rows_num"].eq(0)
    )
    working_df["_primary_metric_num"] = pd.to_numeric(working_df.get(metric_key), errors="coerce")
    working_df["_best_val_loss_num"] = pd.to_numeric(working_df.get("best_val_loss"), errors="coerce")
    working_df = working_df.sort_values(
        by=["_batch_dt", "_started_at_dt", "_original_order"],
        ascending=[True, True, True],
        na_position="last",
    )
    higher_is_better = _is_higher_better_compare_metric(metric_key)

    rows: list[dict[str, Any]] = []
    for batch_label, batch_group in working_df.groupby("_batch_label", sort=False, dropna=False):
        primary_series = batch_group["_primary_metric_num"].dropna()
        best_val_loss_series = batch_group["_best_val_loss_num"].dropna()
        batch_start = batch_group["_started_at_dt"].min()
        batch_end = batch_group["_started_at_dt"].max()
        run_names = batch_group["run_name"].astype(str).tolist()
        run_count = len(batch_group)
        success_runs = int(batch_group["_status_success"].sum())
        clean_runs = int(batch_group["_clean_run"].sum())

        rows.append(
            {
                "batch_label": str(batch_label or "unknown_batch"),
                "batch_start": (
                    batch_start.strftime("%Y-%m-%d %H:%M:%S")
                    if pd.notna(batch_start)
                    else "N/A"
                ),
                "batch_end": (
                    batch_end.strftime("%Y-%m-%d %H:%M:%S")
                    if pd.notna(batch_end)
                    else "N/A"
                ),
                "run_count": run_count,
                "success_runs": success_runs,
                "clean_runs": clean_runs,
                "success_rate": (success_runs / run_count) if run_count else None,
                "clean_rate": (clean_runs / run_count) if run_count else None,
                "primary_metric_mean": (
                    float(primary_series.mean()) if not primary_series.empty else None
                ),
                "primary_metric_best": (
                    float(primary_series.max() if higher_is_better else primary_series.min())
                    if not primary_series.empty
                    else None
                ),
                "best_val_loss_mean": (
                    float(best_val_loss_series.mean()) if not best_val_loss_series.empty else None
                ),
                "failed_rows_total": float(batch_group["_failed_rows_num"].sum()),
                "warning_rows_total": float(batch_group["_warning_rows_num"].sum()),
                "sample_runs": ", ".join(run_names[:3]) if run_names else "N/A",
            }
        )

    return pd.DataFrame(rows)


def _build_compare_batch_insight_lines(
    batch_df: pd.DataFrame,
    *,
    metric_label: str,
    higher_is_better: bool,
) -> list[str]:
    if batch_df.empty:
        return []

    insight_lines = [
        f"已按 started_at 日期把当前选中的运行聚合成 {len(batch_df)} 个批次。",
    ]
    if len(batch_df) == 1:
        insight_lines.append("当前选中的运行只覆盖 1 个批次，这一块主要用于批次级汇总。")

    metric_series = pd.to_numeric(batch_df.get("primary_metric_mean"), errors="coerce")
    if metric_series.notna().any():
        best_index = metric_series.idxmax() if higher_is_better else metric_series.idxmin()
        best_row = batch_df.loc[best_index]
        insight_lines.append(
            f"{metric_label} 批次均值最优的是 {best_row.get('batch_label')}，"
            f"均值 = {_metric_text(best_row.get('primary_metric_mean'))}。"
        )

    clean_rate_series = pd.to_numeric(batch_df.get("clean_rate"), errors="coerce")
    if clean_rate_series.notna().any():
        clean_index = clean_rate_series.idxmax()
        clean_row = batch_df.loc[clean_index]
        insight_lines.append(
            f"clean run 比例最高的批次是 {clean_row.get('batch_label')}，"
            f"clean rate = {_ratio_text(clean_row.get('clean_rate'))}。"
        )

    failed_series = pd.to_numeric(batch_df.get("failed_rows_total"), errors="coerce").fillna(0)
    warning_series = pd.to_numeric(batch_df.get("warning_rows_total"), errors="coerce").fillna(0)
    noise_series = failed_series + warning_series
    if not noise_series.empty and noise_series.max() > 0:
        noisy_index = noise_series.idxmax()
        noisy_row = batch_df.loc[noisy_index]
        insight_lines.append(
            f"失败/告警压力最大的批次是 {noisy_row.get('batch_label')}，"
            f"Failed Rows 合计 {_metric_text(noisy_row.get('failed_rows_total'))}，"
            f"Warning Rows 合计 {_metric_text(noisy_row.get('warning_rows_total'))}。"
        )

    return insight_lines


def _format_compare_batch_table_for_display(
    batch_df: pd.DataFrame,
    *,
    metric_label: str,
) -> pd.DataFrame:
    if batch_df.empty:
        return batch_df

    return batch_df.rename(
        columns={
            "batch_label": "批次",
            "batch_start": "批次起点",
            "batch_end": "批次终点",
            "run_count": "运行数",
            "success_runs": "成功运行数",
            "clean_runs": "clean run 数",
            "success_rate": "成功率",
            "clean_rate": "clean run 比例",
            "primary_metric_mean": f"{metric_label} 均值",
            "primary_metric_best": f"{metric_label} 最优值",
            "best_val_loss_mean": "Best Val Loss 均值",
            "failed_rows_total": "Failed Rows 合计",
            "warning_rows_total": "Warning Rows 合计",
            "sample_runs": "示例运行",
        }
    )


def _build_compare_export_scope_lines(
    *,
    compare_view_df: pd.DataFrame | None = None,
    compare_export_df: pd.DataFrame | None = None,
    batch_view_df: pd.DataFrame | None = None,
    batch_export_df: pd.DataFrame | None = None,
    delta_view_df: pd.DataFrame | None = None,
    delta_export_df: pd.DataFrame | None = None,
    attribution_view_df: pd.DataFrame | None = None,
    attribution_export_df: pd.DataFrame | None = None,
) -> list[str]:
    table_specs = [
        ("完整对比表", compare_view_df, compare_export_df),
        ("跨批次趋势聚合", batch_view_df, batch_export_df),
        ("相对基准的差异表", delta_view_df, delta_export_df),
        ("差异归因总表", attribution_view_df, attribution_export_df),
    ]
    lines: list[str] = []
    for label, view_df, export_df in table_specs:
        if not isinstance(view_df, pd.DataFrame) or view_df.empty:
            continue
        filtered_row_count = len(export_df) if isinstance(export_df, pd.DataFrame) else len(view_df)
        lines.append(
            f"{label} 已同步当前页面视图：展示 {len(view_df)} 行、{len(view_df.columns)} 列；"
            f"当前筛选后共有 {filtered_row_count} 行。"
        )
    return lines


def _create_history_compare_html_export(
    selected_records: list[dict[str, Any]],
    compare_df: pd.DataFrame,
    *,
    metric_key: str,
    metric_label: str,
    higher_is_better: bool,
    leader_name: str | None,
    leader_value: float | None,
    baseline_run_name: str,
    insight_lines: list[str],
    delta_df: pd.DataFrame | None = None,
    trend_df: pd.DataFrame | None = None,
    batch_df: pd.DataFrame | None = None,
    batch_insight_lines: list[str] | None = None,
    attribution_df: pd.DataFrame | None = None,
    attribution_payloads: dict[str, dict[str, Any]] | None = None,
    compare_view_df: pd.DataFrame | None = None,
    compare_export_df: pd.DataFrame | None = None,
    batch_view_df: pd.DataFrame | None = None,
    batch_export_df: pd.DataFrame | None = None,
    delta_view_df: pd.DataFrame | None = None,
    delta_export_df: pd.DataFrame | None = None,
    attribution_view_df: pd.DataFrame | None = None,
    attribution_export_df: pd.DataFrame | None = None,
) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = COMPARE_EXPORT_ROOT / stamp
    export_dir.mkdir(parents=True, exist_ok=True)
    html_path = export_dir / "history_compare_summary.html"

    selected_rows_html = "".join(
        "<li>"
        f"{html.escape(str(item.get('run_name') or 'N/A'))} | "
        f"{html.escape(str(item.get('status') or 'N/A'))} | "
        f"{html.escape(str(item.get('started_at') or 'N/A'))}"
        "</li>"
        for item in selected_records
    )

    failed_rows_numeric = pd.to_numeric(compare_df.get("failed_rows"), errors="coerce").fillna(0)
    warning_rows_numeric = pd.to_numeric(compare_df.get("warning_rows"), errors="coerce").fillna(0)
    success_mask = compare_df["status"].astype(str).str.lower().eq("success")
    clean_mask = success_mask & failed_rows_numeric.eq(0) & warning_rows_numeric.eq(0)
    compare_cards = _build_metric_cards_html(
        [
            ("Compared Runs", len(compare_df)),
            ("Success Runs", int(success_mask.sum())),
            ("Clean Runs", int(clean_mask.sum())),
            ("Primary Metric", metric_label),
            ("Baseline Run", baseline_run_name),
            ("Leader Run", leader_name or "N/A"),
            ("Leader Value", leader_value),
        ]
    )

    highlight_lines = [
        f"当前主指标: {metric_label}（{'越高越好' if higher_is_better else '越低越好'}）",
        f"领先运行: {leader_name or 'N/A'}",
        f"当前 clean run 数量: {int(clean_mask.sum())} / {len(compare_df)}",
    ]
    if "failed_rows" in compare_df.columns:
        highlight_lines.append(
            f"所有选中运行的 failed_rows 合计: {int(failed_rows_numeric.sum())}"
        )
    if "warning_rows" in compare_df.columns:
        highlight_lines.append(
            f"所有选中运行的 warning_rows 合计: {int(warning_rows_numeric.sum())}"
        )
    highlights_html = "".join(f"<li>{html.escape(line)}</li>" for line in highlight_lines)
    insight_lines_html = "".join(f"<li>{html.escape(line)}</li>" for line in insight_lines)
    batch_insight_lines_html = "".join(
        f"<li>{html.escape(line)}</li>" for line in (batch_insight_lines or [])
    )
    compare_scope_lines = _build_compare_export_scope_lines(
        compare_view_df=compare_view_df,
        compare_export_df=compare_export_df,
        batch_view_df=batch_view_df,
        batch_export_df=batch_export_df,
        delta_view_df=delta_view_df,
        delta_export_df=delta_export_df,
        attribution_view_df=attribution_view_df,
        attribution_export_df=attribution_export_df,
    )
    compare_scope_html = "".join(f"<li>{html.escape(line)}</li>" for line in compare_scope_lines)
    compare_view_df = compare_view_df if isinstance(compare_view_df, pd.DataFrame) else compare_df
    compare_export_df = compare_export_df if isinstance(compare_export_df, pd.DataFrame) else compare_df
    batch_view_df = batch_view_df if isinstance(batch_view_df, pd.DataFrame) else batch_df
    batch_export_df = batch_export_df if isinstance(batch_export_df, pd.DataFrame) else batch_df
    delta_view_df = delta_view_df if isinstance(delta_view_df, pd.DataFrame) else delta_df
    delta_export_df = delta_export_df if isinstance(delta_export_df, pd.DataFrame) else delta_df
    attribution_view_df = attribution_view_df if isinstance(attribution_view_df, pd.DataFrame) else attribution_df
    attribution_export_df = attribution_export_df if isinstance(attribution_export_df, pd.DataFrame) else attribution_df
    delta_table_html = _df_to_html_table(delta_view_df, max_rows=max(20, len(delta_view_df) if isinstance(delta_view_df, pd.DataFrame) else 20))
    trend_table_html = _df_to_html_table(trend_df, max_rows=max(20, len(trend_df) if isinstance(trend_df, pd.DataFrame) else 20))
    batch_table_html = _df_to_html_table(batch_view_df, max_rows=max(20, len(batch_view_df) if isinstance(batch_view_df, pd.DataFrame) else 20))
    attribution_table_html = _df_to_html_table(attribution_view_df, max_rows=max(20, len(attribution_view_df) if isinstance(attribution_view_df, pd.DataFrame) else 20))
    attribution_payloads = attribution_payloads if isinstance(attribution_payloads, dict) else {}
    attribution_detail_html_parts: list[str] = []
    for run_name, payload in attribution_payloads.items():
        detail_lines = payload.get("detail_lines") if isinstance(payload.get("detail_lines"), list) else []
        if not detail_lines:
            continue
        detail_items = "".join(f"<li>{html.escape(str(line))}</li>" for line in detail_lines)
        attribution_detail_html_parts.append(
            "<div class='meta-item'>"
            f"<div class='meta-label'>{html.escape(str(run_name))}</div>"
            f"<ul>{detail_items}</ul>"
            "</div>"
        )
    attribution_detail_html = "".join(attribution_detail_html_parts)

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(APP_NAME)} | History Compare | v{html.escape(APP_VERSION)}</title>
{_build_html_export_style()}
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>{html.escape(APP_NAME)} 多运行对比摘要</h1>
      <p>版本: v{html.escape(APP_VERSION)} | 生成时间: {html.escape(_now_text())}</p>
    </section>

    <section class="section">
      <h2>对比摘要</h2>
      <div class="grid">{compare_cards}</div>
    </section>

    <section class="section">
      <h2>选中的运行</h2>
      {"<ul>" + selected_rows_html + "</ul>" if selected_rows_html else "<p class='muted'>当前没有选中运行。</p>"}
    </section>

    <section class="section">
      <h2>重点观察</h2>
      <ul>{highlights_html}</ul>
    </section>

    <section class="section">
      <h2>当前页面视图同步说明</h2>
      {"<ul>" + compare_scope_html + "</ul>" if compare_scope_html else "<p class='muted'>当前没有可同步的表格视图。</p>"}
    </section>

    <section class="section">
      <h2>基准差异解释</h2>
      {"<ul>" + insight_lines_html + "</ul>" if insight_lines_html else "<p class='muted'>当前没有可用的差异解释。</p>"}
    </section>

    <section class="section">
      <h2>趋势快照</h2>
      {trend_table_html}
    </section>

    <section class="section">
      <h2>跨批次趋势聚合（当前页面视图）</h2>
      {batch_table_html}
    </section>

    <section class="section">
      <h2>批次级观察</h2>
      {"<ul>" + batch_insight_lines_html + "</ul>" if batch_insight_lines_html else "<p class='muted'>当前没有可用的批次级观察。</p>"}
    </section>

    <section class="section">
      <h2>相对基准的差异表（当前页面视图）</h2>
      {delta_table_html}
    </section>

    <section class="section">
      <h2>更细的差异归因总表（当前页面视图）</h2>
      {attribution_table_html}
    </section>

    <section class="section">
      <h2>运行级归因明细</h2>
      {"<div class='meta'>" + attribution_detail_html + "</div>" if attribution_detail_html else "<p class='muted'>当前没有可展示的归因明细。</p>"}
    </section>

    <section class="section">
      <h2>完整对比表（当前页面视图）</h2>
      {_df_to_html_table(compare_view_df, max_rows=max(20, len(compare_view_df)))}
    </section>

    <div class="footer">
      该 HTML 为多运行对比导出页，可直接下载、发送或用浏览器打印为 PDF。
    </div>
  </div>
</body>
    </html>
"""
    _write_text(html_path, html_text)
    (export_dir / "history_compare_table_view.csv").write_text(compare_view_df.to_csv(index=False), encoding="utf-8")
    (export_dir / "history_compare_table.csv").write_text(compare_export_df.to_csv(index=False), encoding="utf-8")
    if isinstance(batch_view_df, pd.DataFrame) and not batch_view_df.empty:
        (export_dir / "history_compare_batch_table_view.csv").write_text(batch_view_df.to_csv(index=False), encoding="utf-8")
    if isinstance(batch_export_df, pd.DataFrame) and not batch_export_df.empty:
        (export_dir / "history_compare_batch_table.csv").write_text(batch_export_df.to_csv(index=False), encoding="utf-8")
    if isinstance(delta_view_df, pd.DataFrame) and not delta_view_df.empty:
        (export_dir / "history_compare_delta_view.csv").write_text(delta_view_df.to_csv(index=False), encoding="utf-8")
    if isinstance(delta_export_df, pd.DataFrame) and not delta_export_df.empty:
        (export_dir / "history_compare_delta_table.csv").write_text(delta_export_df.to_csv(index=False), encoding="utf-8")
    if isinstance(attribution_view_df, pd.DataFrame) and not attribution_view_df.empty:
        (export_dir / "history_compare_attribution_view.csv").write_text(attribution_view_df.to_csv(index=False), encoding="utf-8")
    if isinstance(attribution_export_df, pd.DataFrame) and not attribution_export_df.empty:
        (export_dir / "history_compare_attribution_table.csv").write_text(attribution_export_df.to_csv(index=False), encoding="utf-8")
    return html_path


def _create_history_compare_pdf_export(
    selected_records: list[dict[str, Any]],
    compare_df: pd.DataFrame,
    *,
    metric_key: str,
    metric_label: str,
    higher_is_better: bool,
    leader_name: str | None,
    leader_value: float | None,
    baseline_run_name: str,
    insight_lines: list[str],
    delta_df: pd.DataFrame | None = None,
    trend_df: pd.DataFrame | None = None,
    batch_df: pd.DataFrame | None = None,
    batch_insight_lines: list[str] | None = None,
    attribution_df: pd.DataFrame | None = None,
    attribution_payloads: dict[str, dict[str, Any]] | None = None,
    compare_view_df: pd.DataFrame | None = None,
    compare_export_df: pd.DataFrame | None = None,
    batch_view_df: pd.DataFrame | None = None,
    batch_export_df: pd.DataFrame | None = None,
    delta_view_df: pd.DataFrame | None = None,
    delta_export_df: pd.DataFrame | None = None,
    attribution_view_df: pd.DataFrame | None = None,
    attribution_export_df: pd.DataFrame | None = None,
) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = COMPARE_EXPORT_ROOT / stamp
    export_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = export_dir / "history_compare_summary.pdf"

    selected_lines = [
        f"{item.get('run_name', 'N/A')} | {item.get('status', 'N/A')} | {item.get('started_at', 'N/A')}"
        for item in selected_records
    ]
    subtitle_lines = [
        f"基准运行: {baseline_run_name}",
        f"主指标: {metric_label} ({'越高越好' if higher_is_better else '越低越好'})",
        f"生成时间: {_now_text()}",
    ]
    clean_mask = (
        compare_df["status"].astype(str).str.lower().eq("success")
        & pd.to_numeric(compare_df.get("failed_rows"), errors="coerce").fillna(0).eq(0)
        & pd.to_numeric(compare_df.get("warning_rows"), errors="coerce").fillna(0).eq(0)
    )
    compare_scope_lines = _build_compare_export_scope_lines(
        compare_view_df=compare_view_df,
        compare_export_df=compare_export_df,
        batch_view_df=batch_view_df,
        batch_export_df=batch_export_df,
        delta_view_df=delta_view_df,
        delta_export_df=delta_export_df,
        attribution_view_df=attribution_view_df,
        attribution_export_df=attribution_export_df,
    )
    compare_view_df = compare_view_df if isinstance(compare_view_df, pd.DataFrame) else compare_df
    compare_export_df = compare_export_df if isinstance(compare_export_df, pd.DataFrame) else compare_df
    batch_view_df = batch_view_df if isinstance(batch_view_df, pd.DataFrame) else batch_df
    batch_export_df = batch_export_df if isinstance(batch_export_df, pd.DataFrame) else batch_df
    delta_view_df = delta_view_df if isinstance(delta_view_df, pd.DataFrame) else delta_df
    delta_export_df = delta_export_df if isinstance(delta_export_df, pd.DataFrame) else delta_df
    attribution_view_df = attribution_view_df if isinstance(attribution_view_df, pd.DataFrame) else attribution_df
    attribution_export_df = attribution_export_df if isinstance(attribution_export_df, pd.DataFrame) else attribution_df
    attribution_payloads = attribution_payloads if isinstance(attribution_payloads, dict) else {}
    attribution_detail_lines: list[str] = []
    for run_name, payload in attribution_payloads.items():
        detail_lines = payload.get("detail_lines") if isinstance(payload.get("detail_lines"), list) else []
        if not detail_lines:
            continue
        attribution_detail_lines.append(f"[{run_name}]")
        attribution_detail_lines.extend([str(line) for line in detail_lines])
        attribution_detail_lines.append("")
    sections: list[dict[str, Any] | tuple[str, list[str]]] = [
        {
            "title": "对比摘要",
            "kind": "metrics",
            "columns": 2,
            "items": [
                ("Compared Runs", len(compare_df)),
                ("Clean Runs", int(clean_mask.sum())),
                ("Baseline Run", baseline_run_name),
                ("Leader Run", leader_name or "N/A"),
                ("Leader Value", leader_value),
                ("Primary Metric", metric_label),
            ],
        },
        {"title": "选中的运行", "kind": "bullets", "lines": selected_lines or ["当前没有选中运行。"]},
        {
            "title": "当前页面视图同步说明",
            "kind": "bullets",
            "lines": compare_scope_lines or ["当前没有可同步的表格视图。"],
        },
        {"title": "Run-to-Run 差异解释", "kind": "bullets", "lines": insight_lines or ["当前没有可用的差异解释。"]},
        {
            "title": "趋势快照",
            "kind": "table",
            "lines": _df_to_pdf_lines(trend_df, max_rows=max(20, len(trend_df) if isinstance(trend_df, pd.DataFrame) else 20), max_cols=6),
        },
        {
            "title": "跨批次趋势聚合（当前页面视图）",
            "kind": "table",
            "lines": _df_to_pdf_lines(batch_view_df, max_rows=max(20, len(batch_view_df) if isinstance(batch_view_df, pd.DataFrame) else 20), max_cols=8),
        },
        {
            "title": "批次级观察",
            "kind": "bullets",
            "lines": batch_insight_lines or ["当前没有可用的批次级观察。"],
        },
        {
            "title": "相对基准的差异表（当前页面视图）",
            "kind": "table",
            "lines": _df_to_pdf_lines(delta_view_df, max_rows=max(20, len(delta_view_df) if isinstance(delta_view_df, pd.DataFrame) else 20), max_cols=10),
        },
        {
            "title": "更细的差异归因总表（当前页面视图）",
            "kind": "table",
            "lines": _df_to_pdf_lines(attribution_view_df, max_rows=max(20, len(attribution_view_df) if isinstance(attribution_view_df, pd.DataFrame) else 20), max_cols=6),
        },
        {
            "title": "运行级归因明细",
            "kind": "bullets",
            "lines": attribution_detail_lines or ["当前没有可展示的归因明细。"],
        },
        {
            "title": "完整对比表（当前页面视图）",
            "kind": "table",
            "lines": _df_to_pdf_lines(compare_view_df, max_rows=max(20, len(compare_view_df)), max_cols=10),
        },
    ]

    (export_dir / "history_compare_table_view.csv").write_text(compare_view_df.to_csv(index=False), encoding="utf-8")
    (export_dir / "history_compare_table.csv").write_text(compare_export_df.to_csv(index=False), encoding="utf-8")
    if isinstance(batch_view_df, pd.DataFrame) and not batch_view_df.empty:
        (export_dir / "history_compare_batch_table_view.csv").write_text(batch_view_df.to_csv(index=False), encoding="utf-8")
    if isinstance(batch_export_df, pd.DataFrame) and not batch_export_df.empty:
        (export_dir / "history_compare_batch_table.csv").write_text(batch_export_df.to_csv(index=False), encoding="utf-8")
    if isinstance(delta_view_df, pd.DataFrame) and not delta_view_df.empty:
        (export_dir / "history_compare_delta_view.csv").write_text(delta_view_df.to_csv(index=False), encoding="utf-8")
    if isinstance(delta_export_df, pd.DataFrame) and not delta_export_df.empty:
        (export_dir / "history_compare_delta_table.csv").write_text(delta_export_df.to_csv(index=False), encoding="utf-8")
    if isinstance(attribution_view_df, pd.DataFrame) and not attribution_view_df.empty:
        (export_dir / "history_compare_attribution_view.csv").write_text(attribution_view_df.to_csv(index=False), encoding="utf-8")
    if isinstance(attribution_export_df, pd.DataFrame) and not attribution_export_df.empty:
        (export_dir / "history_compare_attribution_table.csv").write_text(attribution_export_df.to_csv(index=False), encoding="utf-8")
    return _build_pdf_document(
        pdf_path,
        title=f"{APP_NAME} 多运行对比自动化 PDF",
        subtitle_lines=subtitle_lines,
        sections=sections,
    )


def _render_history_compare_panel(history_records: list[dict[str, Any]]) -> None:
    if not history_records:
        st.info("当前还没有历史运行记录，暂时无法做多运行对比。")
        return

    filtered_history_records = _render_history_filter_controls(
        history_records,
        key_prefix="compare",
        title="先筛选要进入多运行对比候选集的历史记录",
    )
    if not filtered_history_records:
        st.info("当前筛选条件下没有可对比的历史运行。")
        return

    record_map = {str(item["label"]): item for item in filtered_history_records}
    compare_options = [str(item["label"]) for item in filtered_history_records]
    default_compare = compare_options[: min(3, len(compare_options))]
    selected_labels = st.multiselect(
        "选择要对比的运行",
        options=compare_options,
        default=default_compare,
        key="compare_history_labels",
    )
    if not selected_labels:
        st.info("先选择至少一条历史运行记录。")
        return

    selected_records = [record_map[label] for label in selected_labels if label in record_map]
    compare_df = _build_history_compare_table(selected_records)
    if compare_df.empty:
        st.info("当前选中的运行没有可对比的摘要信息。")
        return

    metric_key = st.selectbox(
        "主要对比指标",
        options=[
            "calibrated_rank_spearman",
            "baseline_rank_spearman",
            "best_val_loss",
            "failed_rows",
            "warning_rows",
        ],
        format_func=lambda key: _get_compare_metric_label(key),
        key="history_compare_metric_key",
    )
    higher_is_better = _is_higher_better_compare_metric(metric_key)
    leader_name, leader_value = _pick_compare_leader(
        compare_df,
        metric_key=metric_key,
        higher_is_better=higher_is_better,
    )
    ordered_compare_df = _sort_compare_df_for_trend(compare_df)
    baseline_options = ordered_compare_df["run_name"].astype(str).tolist()
    baseline_run_name = st.selectbox(
        "基准运行",
        options=baseline_options,
        index=0,
        key="history_compare_baseline_run",
    )
    delta_df = _build_compare_delta_table(
        compare_df,
        baseline_run_name=baseline_run_name,
        metric_key=metric_key,
    )
    insight_lines = _build_compare_insight_lines(
        compare_df,
        baseline_run_name=baseline_run_name,
        metric_key=metric_key,
        metric_label=_get_compare_metric_label(metric_key),
    )
    attribution_payloads = _build_compare_attribution_payloads(
        delta_df,
        baseline_run_name=baseline_run_name,
        metric_key=metric_key,
        metric_label=_get_compare_metric_label(metric_key),
    )
    attribution_df = _build_compare_attribution_table(attribution_payloads)
    batch_df = _build_compare_batch_trend_table(compare_df, metric_key=metric_key)
    batch_display_df = _format_compare_batch_table_for_display(
        batch_df,
        metric_label=_get_compare_metric_label(metric_key),
    )
    batch_insight_lines = _build_compare_batch_insight_lines(
        batch_df,
        metric_label=_get_compare_metric_label(metric_key),
        higher_is_better=higher_is_better,
    )

    failed_rows_numeric = pd.to_numeric(compare_df.get("failed_rows"), errors="coerce").fillna(0)
    warning_rows_numeric = pd.to_numeric(compare_df.get("warning_rows"), errors="coerce").fillna(0)
    clean_run_mask = compare_df["status"].astype(str).str.lower().eq("success") & failed_rows_numeric.eq(0) & warning_rows_numeric.eq(0)
    success_count = int(compare_df["status"].astype(str).str.lower().eq("success").sum())
    clean_count = int(clean_run_mask.sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Compared Runs", len(compare_df))
    c2.metric("Success Runs", success_count)
    c3.metric("Clean Runs", clean_count)
    c4.metric("Leader Value", _fmt_metric(leader_value))
    st.caption(
        f"当前按 {_get_compare_metric_label(metric_key)} "
        f"{'越高越好' if higher_is_better else '越低越好'} 比较；领先运行: {leader_name or 'N/A'}；"
        f"当前基准运行: {baseline_run_name}"
    )

    chart_columns = []
    for column in [
        metric_key,
        "baseline_rank_spearman",
        "calibrated_rank_spearman",
        "best_val_loss",
        "failed_rows",
        "warning_rows",
    ]:
        if column not in compare_df.columns:
            continue
        series = pd.to_numeric(compare_df[column], errors="coerce")
        if series.notna().any():
            chart_columns.append(column)
    if chart_columns:
        chart_df = compare_df.loc[:, ["run_name"] + chart_columns].copy()
        for column in chart_columns:
            chart_df[column] = pd.to_numeric(chart_df[column], errors="coerce")
        st.bar_chart(chart_df.set_index("run_name"), height=280)

    trend_columns = []
    for column in [metric_key, "best_val_loss", "failed_rows", "warning_rows"]:
        if column not in ordered_compare_df.columns:
            continue
        series = pd.to_numeric(ordered_compare_df[column], errors="coerce")
        if series.notna().any():
            trend_columns.append(column)
    trend_df = pd.DataFrame()
    batch_view_df = pd.DataFrame()
    batch_export_df = pd.DataFrame()
    attribution_view_df = pd.DataFrame()
    attribution_export_df = pd.DataFrame()
    delta_view_df = pd.DataFrame()
    delta_export_df = pd.DataFrame()
    compare_view_df = pd.DataFrame()
    compare_export_df = pd.DataFrame()
    if trend_columns:
        trend_df = ordered_compare_df.loc[:, ["started_at", "run_name"] + trend_columns].copy()
        trend_df["timeline_label"] = (
            trend_df["started_at"].astype(str).fillna("N/A")
            + " | "
            + trend_df["run_name"].astype(str).fillna("N/A")
        )
        chart_ready_df = trend_df.loc[:, ["timeline_label"] + trend_columns].copy()
        for column in trend_columns:
            chart_ready_df[column] = pd.to_numeric(chart_ready_df[column], errors="coerce")
        st.subheader("多运行趋势")
        st.line_chart(chart_ready_df.set_index("timeline_label"), height=260)
        st.caption("趋势图按 started_at 从早到晚排序，方便看不同运行之间的连续变化。")

    if not batch_df.empty:
        st.subheader("跨批次趋势聚合")
        if batch_insight_lines:
            for line in batch_insight_lines:
                st.write(f"- {line}")
        batch_chart_col1, batch_chart_col2 = st.columns(2)
        batch_quality_columns = []
        for column in ["primary_metric_mean", "success_rate", "clean_rate"]:
            series = pd.to_numeric(batch_df.get(column), errors="coerce")
            if series.notna().any():
                batch_quality_columns.append(column)
        if batch_quality_columns:
            batch_quality_df = batch_df.loc[:, ["batch_label"] + batch_quality_columns].copy()
            for column in batch_quality_columns:
                batch_quality_df[column] = pd.to_numeric(batch_quality_df[column], errors="coerce")
            batch_chart_col1.line_chart(batch_quality_df.set_index("batch_label"), height=240)
        else:
            batch_chart_col1.info("当前没有可展示的批次级质量趋势。")

        batch_count_columns = []
        for column in ["run_count", "failed_rows_total", "warning_rows_total"]:
            series = pd.to_numeric(batch_df.get(column), errors="coerce")
            if series.notna().any():
                batch_count_columns.append(column)
        if batch_count_columns:
            batch_count_df = batch_df.loc[:, ["batch_label"] + batch_count_columns].copy()
            for column in batch_count_columns:
                batch_count_df[column] = pd.to_numeric(batch_count_df[column], errors="coerce")
            batch_chart_col2.bar_chart(batch_count_df.set_index("batch_label"), height=240)
        else:
            batch_chart_col2.info("当前没有可展示的批次级计数汇总。")

        st.caption("批次聚合当前按 started_at 的日期维度汇总，便于快速回顾不同实验批次的整体表现。")
        batch_display_df, batch_threshold_summaries = _render_numeric_threshold_filters(
            batch_display_df,
            key_prefix="compare_batch_threshold",
            title="批次聚合数值阈值筛选",
        )
        batch_view_df, batch_export_df = _render_dataframe_view_controls(
            batch_display_df,
            key_prefix="compare_batch_table",
            preferred_columns=[
                "批次",
                "运行数",
                "成功运行数",
                "clean run 数",
                f"{_get_compare_metric_label(metric_key)} 均值",
                f"{_get_compare_metric_label(metric_key)} 最优值",
                "Failed Rows 合计",
                "Warning Rows 合计",
            ],
            default_sort_column=f"{_get_compare_metric_label(metric_key)} 均值",
            default_descending=higher_is_better,
        )
        if batch_threshold_summaries:
            st.caption("已启用阈值: " + "；".join(batch_threshold_summaries))
        st.caption(
            f"当前批次聚合共有 {len(batch_export_df)} 条，当前展示前 {len(batch_view_df)} 条；"
            f"可见列 {len(batch_view_df.columns)} 个。"
        )
        st.dataframe(batch_view_df, use_container_width=True, hide_index=True)
        batch_export_col1, batch_export_col2 = st.columns(2)
        batch_export_col1.download_button(
            label="下载批次聚合可见视图 CSV",
            data=batch_view_df.to_csv(index=False).encode("utf-8"),
            file_name="history_compare_batch_table_view.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_compare_batch_view_csv",
        )
        batch_export_col2.download_button(
            label="下载批次聚合全表 CSV",
            data=batch_export_df.to_csv(index=False).encode("utf-8"),
            file_name="history_compare_batch_table.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_compare_batch_csv",
        )

    st.subheader("Run-to-Run 差异解释")
    if insight_lines:
        for line in insight_lines:
            st.write(f"- {line}")
    else:
        st.info("当前选中的运行还不足以生成稳定的差异解释。")

    if attribution_payloads:
        st.subheader("更细的差异归因")
        attribution_run_candidates = [
            run_name
            for run_name in attribution_payloads.keys()
            if str(run_name) != str(baseline_run_name)
        ] or list(attribution_payloads.keys())
        selected_attribution_run = st.selectbox(
            "查看归因明细的运行",
            options=attribution_run_candidates,
            key="history_compare_attribution_run",
        )
        selected_attribution = attribution_payloads.get(str(selected_attribution_run))
        if isinstance(selected_attribution, dict):
            st.caption(str(selected_attribution.get("summary") or ""))
            detail_col1, detail_col2 = st.columns(2)
            positive_driver_texts = selected_attribution.get("positive_driver_texts") if isinstance(selected_attribution.get("positive_driver_texts"), list) else []
            negative_driver_texts = selected_attribution.get("negative_driver_texts") if isinstance(selected_attribution.get("negative_driver_texts"), list) else []
            if positive_driver_texts:
                detail_col1.success("正向驱动: " + "；".join([str(item) for item in positive_driver_texts]))
            else:
                detail_col1.info("当前没有明显的正向驱动。")
            if negative_driver_texts:
                detail_col2.warning("负向拖累: " + "；".join([str(item) for item in negative_driver_texts]))
            else:
                detail_col2.success("当前没有明显的负向拖累。")
            detail_lines = selected_attribution.get("detail_lines") if isinstance(selected_attribution.get("detail_lines"), list) else []
            for line in detail_lines:
                st.write(f"- {line}")

    if not attribution_df.empty:
        st.subheader("归因总表")
        attribution_df, attribution_threshold_summaries = _render_numeric_threshold_filters(
            attribution_df,
            key_prefix="compare_attribution_threshold",
            title="归因总表数值阈值筛选",
        )
        attribution_view_df, attribution_export_df = _render_dataframe_view_controls(
            attribution_df,
            key_prefix="compare_attribution_table",
            preferred_columns=[
                "run_name",
                "归因标签",
                "主指标净改善",
                "主要正向驱动",
                "主要负向拖累",
            ],
            default_sort_column="主指标净改善",
            default_descending=True,
        )
        if attribution_threshold_summaries:
            st.caption("已启用阈值: " + "；".join(attribution_threshold_summaries))
        st.caption(
            f"当前归因总表共有 {len(attribution_export_df)} 条，当前展示前 {len(attribution_view_df)} 条；"
            f"可见列 {len(attribution_view_df.columns)} 个。"
        )
        st.dataframe(attribution_view_df, use_container_width=True, hide_index=True)
        attribution_export_col1, attribution_export_col2 = st.columns(2)
        attribution_export_col1.download_button(
            label="下载归因总表可见视图 CSV",
            data=attribution_view_df.to_csv(index=False).encode("utf-8"),
            file_name="history_compare_attribution_view.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_compare_attribution_view_csv",
        )
        attribution_export_col2.download_button(
            label="下载归因总表全表 CSV",
            data=attribution_export_df.to_csv(index=False).encode("utf-8"),
            file_name="history_compare_attribution_table.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_compare_attribution_csv",
        )

    if not delta_df.empty:
        st.subheader("相对基准的差异表")
        preferred_delta_columns = [
            "run_name",
            "status",
            "started_at",
            "baseline_run",
            "primary_metric_key",
            "primary_metric_value",
            "primary_metric_delta",
            "primary_metric_improvement",
            "baseline_rank_spearman",
            "baseline_rank_spearman_delta",
            "calibrated_rank_spearman",
            "calibrated_rank_spearman_delta",
            "baseline_rule_auc",
            "baseline_rule_auc_delta",
            "calibrated_rule_auc",
            "calibrated_rule_auc_delta",
            "best_val_loss",
            "best_val_loss_delta",
            "failed_rows",
            "failed_rows_delta",
            "warning_rows",
            "warning_rows_delta",
        ]
        available_delta_columns = [column for column in preferred_delta_columns if column in delta_df.columns]
        delta_display_df = delta_df.loc[:, available_delta_columns].copy()
        delta_display_df, delta_threshold_summaries = _render_numeric_threshold_filters(
            delta_display_df,
            key_prefix="compare_delta_threshold",
            title="差异表数值阈值筛选",
        )
        delta_view_df, delta_export_df = _render_dataframe_view_controls(
            delta_display_df,
            key_prefix="compare_delta_table",
            preferred_columns=[
                "run_name",
                "status",
                "started_at",
                "primary_metric_value",
                "primary_metric_delta",
                "primary_metric_improvement",
                "best_val_loss_delta",
                "failed_rows_delta",
                "warning_rows_delta",
            ],
            default_sort_column="primary_metric_improvement",
            default_descending=True,
        )
        if delta_threshold_summaries:
            st.caption("已启用阈值: " + "；".join(delta_threshold_summaries))
        st.caption(
            f"当前差异表共有 {len(delta_export_df)} 条，当前展示前 {len(delta_view_df)} 条；"
            f"可见列 {len(delta_view_df.columns)} 个。"
        )
        st.dataframe(delta_view_df, use_container_width=True, hide_index=True)
        delta_export_col1, delta_export_col2 = st.columns(2)
        delta_export_col1.download_button(
            label="下载差异表可见视图 CSV",
            data=delta_view_df.to_csv(index=False).encode("utf-8"),
            file_name="history_compare_delta_view.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_compare_delta_view_csv",
        )
        delta_export_col2.download_button(
            label="下载差异表全表 CSV",
            data=delta_export_df.to_csv(index=False).encode("utf-8"),
            file_name="history_compare_delta_table.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_compare_delta_csv",
        )

    with st.expander("验证回灌再训练前后对照报告", expanded=False):
        st.caption(
            "用于比较“回灌前基准运行”和“使用 experiment_label 后重新运行”的变化；"
            "只读取两个运行的 summary / consensus ranking，不改动已有结果。"
        )
        run_name_to_root = {
            Path(str(record.get("run_root"))).name: Path(str(record.get("run_root")))
            for record in selected_records
            if str(record.get("run_root") or "").strip()
        }
        after_options = [name for name in baseline_options if str(name) != str(baseline_run_name)]
        if not after_options:
            st.info("至少需要选择两个历史运行，才能生成验证回灌再训练对照报告。")
        else:
            after_run_name = st.selectbox(
                "回灌后运行",
                options=after_options,
                index=max(0, len(after_options) - 1),
                key="validation_retrain_compare_after_run",
            )
            default_validation_labels = LOCAL_APP_RUN_ROOT / "experiment_validation_feedback" / "experiment_validation_labels.csv"
            if (
                not str(st.session_state.get("validation_retrain_compare_labels_csv_path") or "").strip()
                and default_validation_labels.exists()
            ):
                st.session_state["validation_retrain_compare_labels_csv_path"] = str(default_validation_labels)
            st.text_input(
                "可选 validation labels CSV",
                key="validation_retrain_compare_labels_csv_path",
                help="通常是 local_app_runs/experiment_validation_feedback/experiment_validation_labels.csv；填写后会标记已验证候选在 top-k 中的变化。",
            )
            st.number_input(
                "top_k",
                min_value=1,
                max_value=200,
                step=1,
                key="validation_retrain_compare_top_k",
            )
            before_root = run_name_to_root.get(str(baseline_run_name))
            after_root = run_name_to_root.get(str(after_run_name))
            safe_pair_name = f"{_slugify_name(str(baseline_run_name))}__vs__{_slugify_name(str(after_run_name))}"
            retrain_compare_out = LOCAL_APP_RUN_ROOT / "validation_retrain_comparisons" / safe_pair_name
            generate_retrain_compare_clicked = st.button(
                "生成验证回灌再训练对照报告",
                use_container_width=True,
                key="generate_validation_retrain_comparison",
            )
            if generate_retrain_compare_clicked:
                try:
                    if before_root is None or after_root is None:
                        raise ValueError("无法定位所选运行目录。")
                    labels_path_text = str(st.session_state.get("validation_retrain_compare_labels_csv_path") or "").strip()
                    comparison_summary = build_validation_retrain_comparison(
                        before_summary=before_root / "outputs" / "recommended_pipeline_summary.json",
                        after_summary=after_root / "outputs" / "recommended_pipeline_summary.json",
                        out_dir=retrain_compare_out,
                        validation_labels_csv=labels_path_text or None,
                        top_k=int(st.session_state.get("validation_retrain_compare_top_k") or 10),
                    )
                    st.session_state["last_validation_retrain_comparison"] = comparison_summary
                    st.success(f"已生成验证回灌再训练对照报告: {retrain_compare_out}")
                except Exception as exc:
                    st.error(f"生成验证回灌再训练对照报告失败: {exc}")

            comparison_summary_path = retrain_compare_out / "validation_retrain_comparison_summary.json"
            comparison_summary = _load_json(comparison_summary_path) or {}
            if not comparison_summary:
                last_comparison_summary = st.session_state.get("last_validation_retrain_comparison", {})
                last_outputs = (
                    last_comparison_summary.get("outputs")
                    if isinstance(last_comparison_summary, dict) and isinstance(last_comparison_summary.get("outputs"), dict)
                    else {}
                )
                last_summary_path = str(last_outputs.get("summary_json") or "")
                if last_summary_path and Path(last_summary_path).parent == retrain_compare_out:
                    comparison_summary = last_comparison_summary
            if isinstance(comparison_summary, dict) and comparison_summary:
                overlap = comparison_summary.get("top_k_overlap") if isinstance(comparison_summary.get("top_k_overlap"), dict) else {}
                rt_col1, rt_col2, rt_col3 = st.columns(3)
                rt_col1.metric("Top-k 重叠", int(overlap.get("overlap_count") or 0))
                overlap_fraction = _coerce_float(overlap.get("fraction"))
                rt_col2.metric("Top-k overlap fraction", _fmt_metric(overlap_fraction))
                validated_top_k_count = comparison_summary.get("validated_top_k_count")
                rt_col3.metric(
                    "回灌后 Top-k 已验证",
                    "N/A" if validated_top_k_count is None else int(validated_top_k_count),
                )
                outputs = comparison_summary.get("outputs") if isinstance(comparison_summary.get("outputs"), dict) else {}
                metric_path = Path(str(outputs.get("metric_comparison_csv"))) if outputs.get("metric_comparison_csv") else None
                rank_delta_path = (
                    Path(str(outputs.get("candidate_rank_delta_csv")))
                    if outputs.get("candidate_rank_delta_csv")
                    else None
                )
                report_path = Path(str(outputs.get("report_md"))) if outputs.get("report_md") else None
                if metric_path is not None and metric_path.exists():
                    metric_df = _load_csv(metric_path)
                    if metric_df is not None:
                        st.caption("关键指标对照")
                        st.dataframe(metric_df, use_container_width=True, hide_index=True)
                    st.download_button(
                        label="下载指标对照 CSV",
                        data=metric_path.read_bytes(),
                        file_name=metric_path.name,
                        mime="text/csv",
                        use_container_width=True,
                        key="download_validation_retrain_metrics_csv",
                    )
                if rank_delta_path is not None and rank_delta_path.exists():
                    rank_delta_df = _load_csv(rank_delta_path)
                    if rank_delta_df is not None:
                        st.caption("候选排名变化")
                        st.dataframe(rank_delta_df.head(100), use_container_width=True, hide_index=True)
                    st.download_button(
                        label="下载候选排名变化 CSV",
                        data=rank_delta_path.read_bytes(),
                        file_name=rank_delta_path.name,
                        mime="text/csv",
                        use_container_width=True,
                        key="download_validation_retrain_rank_delta_csv",
                    )
                if report_path is not None and report_path.exists():
                    st.download_button(
                        label="下载 Markdown 对照报告",
                        data=report_path.read_bytes(),
                        file_name=report_path.name,
                        mime="text/markdown",
                        use_container_width=True,
                        key="download_validation_retrain_report_md",
                    )
                    with st.expander("预览 Markdown 对照报告"):
                        st.markdown(_read_text(report_path))

    compare_export_container = st.container()

    st.subheader("完整对比表")
    compare_filtered_df, compare_threshold_summaries = _render_numeric_threshold_filters(
        compare_df,
        key_prefix="compare_main_threshold",
        title="完整对比表数值阈值筛选",
    )
    compare_view_df, compare_export_df = _render_dataframe_view_controls(
        compare_filtered_df,
        key_prefix="compare_main_table",
        preferred_columns=[
            "run_name",
            "status",
            "started_at",
            metric_key,
            "baseline_rank_spearman",
            "calibrated_rank_spearman",
            "best_val_loss",
            "failed_rows",
            "warning_rows",
            "feature_count",
        ],
        default_sort_column=metric_key if metric_key in compare_filtered_df.columns else "started_at",
        default_descending=higher_is_better if metric_key in compare_filtered_df.columns else False,
    )
    if compare_threshold_summaries:
        st.caption("已启用阈值: " + "；".join(compare_threshold_summaries))
    st.caption(
        f"当前对比全表共有 {len(compare_export_df)} 条，当前展示前 {len(compare_view_df)} 条；"
        f"可见列 {len(compare_view_df.columns)} 个。"
    )
    st.dataframe(compare_view_df, use_container_width=True, hide_index=True)
    compare_export_col1, compare_export_col2 = st.columns(2)
    compare_export_col1.download_button(
        label="下载对比表可见视图 CSV",
        data=compare_view_df.to_csv(index=False).encode("utf-8"),
        file_name="history_compare_table_view.csv",
        mime="text/csv",
        use_container_width=True,
        key="download_compare_main_view_csv",
    )
    compare_export_col2.download_button(
        label="下载对比表全表 CSV",
        data=compare_export_df.to_csv(index=False).encode("utf-8"),
        file_name="history_compare_table.csv",
        mime="text/csv",
        use_container_width=True,
        key="download_compare_main_csv",
    )

    with compare_export_container:
        st.subheader("运行对比导出")
        st.caption(
            "这里生成的 HTML / PDF 会同步当前页面四张表的显示列、排序和数值阈值筛选结果；"
            "导出目录中也会附带当前可见视图 CSV 与全筛选结果 CSV。"
        )
        last_compare_export_html = st.session_state.get("last_compare_export_html")
        last_compare_export_html_message = str(st.session_state.get("last_compare_export_html_message") or "")
        last_compare_export_pdf = st.session_state.get("last_compare_export_pdf")
        last_compare_export_pdf_message = str(st.session_state.get("last_compare_export_pdf_message") or "")

        compare_bundle_col1, compare_bundle_col2, compare_bundle_col3 = st.columns(3)
        create_compare_html_clicked = compare_bundle_col1.button(
            "生成当前对比 HTML",
            use_container_width=True,
            key="create_history_compare_html_export",
        )
        create_compare_pdf_clicked = compare_bundle_col2.button(
            "生成当前对比 PDF",
            use_container_width=True,
            key="create_history_compare_pdf_export",
        )
        open_compare_export_dir_clicked = compare_bundle_col3.button(
            "打开对比导出目录",
            use_container_width=True,
            key="open_history_compare_export_dir",
        )

        if create_compare_html_clicked:
            with st.spinner("正在生成当前运行对比 HTML..."):
                compare_html_path = _create_history_compare_html_export(
                    selected_records,
                    compare_df,
                    metric_key=metric_key,
                    metric_label=_get_compare_metric_label(metric_key),
                    higher_is_better=higher_is_better,
                    leader_name=leader_name,
                    leader_value=leader_value,
                    baseline_run_name=baseline_run_name,
                    insight_lines=insight_lines,
                    delta_df=delta_export_df,
                    trend_df=trend_df,
                    batch_df=batch_export_df,
                    batch_insight_lines=batch_insight_lines,
                    attribution_df=attribution_export_df,
                    attribution_payloads=attribution_payloads,
                    compare_view_df=compare_view_df,
                    compare_export_df=compare_export_df,
                    batch_view_df=batch_view_df,
                    batch_export_df=batch_export_df,
                    delta_view_df=delta_view_df,
                    delta_export_df=delta_export_df,
                    attribution_view_df=attribution_view_df,
                    attribution_export_df=attribution_export_df,
                )
            st.session_state["last_compare_export_html"] = str(compare_html_path)
            st.session_state["last_compare_export_html_message"] = f"已生成运行对比 HTML: {compare_html_path}"
            last_compare_export_html = str(compare_html_path)
            last_compare_export_html_message = str(st.session_state["last_compare_export_html_message"])

        if create_compare_pdf_clicked:
            with st.spinner("正在生成当前运行对比 PDF..."):
                compare_pdf_path = _create_history_compare_pdf_export(
                    selected_records,
                    compare_df,
                    metric_key=metric_key,
                    metric_label=_get_compare_metric_label(metric_key),
                    higher_is_better=higher_is_better,
                    leader_name=leader_name,
                    leader_value=leader_value,
                    baseline_run_name=baseline_run_name,
                    insight_lines=insight_lines,
                    delta_df=delta_export_df,
                    trend_df=trend_df,
                    batch_df=batch_export_df,
                    batch_insight_lines=batch_insight_lines,
                    attribution_df=attribution_export_df,
                    attribution_payloads=attribution_payloads,
                    compare_view_df=compare_view_df,
                    compare_export_df=compare_export_df,
                    batch_view_df=batch_view_df,
                    batch_export_df=batch_export_df,
                    delta_view_df=delta_view_df,
                    delta_export_df=delta_export_df,
                    attribution_view_df=attribution_view_df,
                    attribution_export_df=attribution_export_df,
                )
            st.session_state["last_compare_export_pdf"] = str(compare_pdf_path)
            st.session_state["last_compare_export_pdf_message"] = f"已生成运行对比 PDF: {compare_pdf_path}"
            last_compare_export_pdf = str(compare_pdf_path)
            last_compare_export_pdf_message = str(st.session_state["last_compare_export_pdf_message"])

        if open_compare_export_dir_clicked:
            COMPARE_EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
            ok, message = _open_local_path(COMPARE_EXPORT_ROOT)
            if ok:
                st.success(message)
            else:
                st.error(message)

        if last_compare_export_html_message:
            st.success(last_compare_export_html_message)
        if last_compare_export_html and Path(str(last_compare_export_html)).exists():
            compare_html_path = Path(str(last_compare_export_html))
            st.caption(f"当前运行对比 HTML: {compare_html_path}")
            compare_html_col1, compare_html_col2 = st.columns(2)
            compare_html_col1.download_button(
                label="下载当前运行对比 HTML",
                data=compare_html_path.read_bytes(),
                file_name=compare_html_path.name,
                mime="text/html",
                use_container_width=True,
                key="download_history_compare_html_export",
            )
            if compare_html_col2.button("打开当前运行对比 HTML", use_container_width=True, key="open_history_compare_html_export"):
                ok, message = _open_local_path(compare_html_path)
                if ok:
                    st.success(message)
                else:
                    st.error(message)

        if last_compare_export_pdf_message:
            st.success(last_compare_export_pdf_message)
        if last_compare_export_pdf and Path(str(last_compare_export_pdf)).exists():
            compare_pdf_path = Path(str(last_compare_export_pdf))
            st.caption(f"当前运行对比 PDF: {compare_pdf_path}")
            compare_pdf_col1, compare_pdf_col2 = st.columns(2)
            compare_pdf_col1.download_button(
                label="下载当前运行对比 PDF",
                data=compare_pdf_path.read_bytes(),
                file_name=compare_pdf_path.name,
                mime="application/pdf",
                use_container_width=True,
                key="download_history_compare_pdf_export",
            )
            if compare_pdf_col2.button("打开当前运行对比 PDF", use_container_width=True, key="open_history_compare_pdf_export"):
                ok, message = _open_local_path(compare_pdf_path)
                if ok:
                    st.success(message)
                else:
                    st.error(message)


def _build_failure_diagnostics(
    *,
    returncode: int,
    stdout_text: str,
    stderr_text: str,
    summary: dict[str, Any] | None,
    out_dir: Path,
    resolved_inputs: dict[str, str | None],
) -> dict[str, Any]:
    combined = "\n".join([stdout_text or "", stderr_text or ""]).strip()
    messages: list[str] = []
    suggestions: list[str] = []

    failed_stage = _detect_failed_stage(combined)
    category = "runtime_failure"

    missing_inputs = []
    for key, value in resolved_inputs.items():
        if not value:
            continue
        path = Path(str(value))
        if not path.exists():
            missing_inputs.append(f"{key}: {path}")
    if missing_inputs:
        category = "missing_input_file"
        messages.append("发现运行时依赖输入文件缺失。")
        messages.extend(missing_inputs)
        suggestions.append("检查本机路径是否仍然有效，或重新上传所需输入文件。")

    if "ModuleNotFoundError" in combined:
        category = "missing_python_dependency"
        messages.append("Python 依赖缺失，运行环境未满足当前流程要求。")
        suggestions.append("先执行 `pip install -r requirements.txt`，再重新运行。")

    if "Runtime dependency precheck failed" in combined:
        category = "missing_python_dependency"
        messages.append("运行前依赖预检未通过，当前环境缺少关键 Python 依赖。")
        suggestions.append("先执行 `pip install -r requirements.txt`，如果仍缺少 torch，请检查当前 Python 环境是否与已验证环境一致。")

    if "FileNotFoundError" in combined and not missing_inputs:
        category = "file_not_found"
        messages.append("CLI 在运行过程中遇到了文件不存在错误。")
        suggestions.append("优先检查输入 CSV 中引用的本地路径是否真实存在。")

    if "Expected output files not found" in combined:
        category = "incomplete_output_artifacts"
        messages.append("某一步命令返回后，没有生成预期输出文件。")
        suggestions.append("优先检查失败阶段的 stdout/stderr 以及输出目录内容。")

    if failed_stage:
        messages.append(f"检测到失败阶段: {failed_stage}")
        stage_tip = KNOWN_STAGE_SUGGESTIONS.get(failed_stage)
        if stage_tip:
            suggestions.append(stage_tip)

    if summary is None:
        messages.append("当前未生成 recommended_pipeline_summary.json。")
        suggestions.append("先检查 stderr 中最早出现的异常栈和路径错误。")

    if not messages:
        messages.append("未识别出明确错误分类，建议先查看 stderr 末尾日志。")

    if not suggestions:
        suggestions.append("先检查日志末尾、输入路径和输出目录。")

    return {
        "status": "failed",
        "returncode": int(returncode),
        "category": category,
        "failed_stage": failed_stage,
        "messages": messages,
        "suggestions": suggestions,
        "summary_exists": bool(summary is not None),
        "out_dir": str(out_dir),
        "out_dir_exists": bool(out_dir.exists()),
        "resolved_inputs": resolved_inputs,
        "stdout_excerpt": _extract_tail(stdout_text),
        "stderr_excerpt": _extract_tail(stderr_text),
    }


def _build_success_diagnostics(summary: dict[str, Any] | None, resolved_inputs: dict[str, str | None], out_dir: Path) -> dict[str, Any]:
    return {
        "status": "success",
        "returncode": 0,
        "category": "success",
        "failed_stage": None,
        "messages": ["最近一次运行已成功完成。"],
        "suggestions": ["可直接查看 ranking、report 和关键产物下载区。"],
        "summary_exists": bool(summary is not None),
        "out_dir": str(out_dir),
        "out_dir_exists": bool(out_dir.exists()),
        "resolved_inputs": resolved_inputs,
        "stdout_excerpt": "",
        "stderr_excerpt": "",
    }


def _set_last_run_state(
    *,
    run_root: Path,
    summary: dict[str, Any] | None,
    metadata: dict[str, Any] | None,
    stdout_text: str,
    stderr_text: str,
    diagnostics: dict[str, Any] | None,
    error_text: str,
) -> None:
    st.session_state["last_run_dir"] = str(run_root)
    st.session_state["last_summary"] = summary
    st.session_state["last_metadata"] = metadata
    st.session_state["last_command"] = None if not metadata else metadata.get("command")
    st.session_state["last_stdout"] = stdout_text
    st.session_state["last_stderr"] = stderr_text
    st.session_state["last_diagnostics"] = diagnostics
    st.session_state["last_error"] = error_text
    st.session_state["last_export_bundle"] = None
    st.session_state["last_export_message"] = ""
    st.session_state["last_export_html"] = None
    st.session_state["last_export_html_message"] = ""
    st.session_state["last_export_pdf"] = None
    st.session_state["last_export_pdf_message"] = ""


def _load_history_run(run_root: Path) -> None:
    metadata = _load_json(run_root / APP_RUN_METADATA_NAME) or {}
    summary = _load_json(run_root / "outputs" / "recommended_pipeline_summary.json")
    stdout_text = _read_text(run_root / APP_STDOUT_NAME)
    stderr_text = _read_text(run_root / APP_STDERR_NAME)
    diagnostics = metadata.get("diagnostics") if isinstance(metadata.get("diagnostics"), dict) else None
    error_text = str(metadata.get("error_text") or "")
    form_payload = metadata.get("form_payload") if isinstance(metadata.get("form_payload"), dict) else None

    _apply_form_payload(form_payload)
    _set_last_run_state(
        run_root=run_root,
        summary=summary,
        metadata=metadata,
        stdout_text=stdout_text,
        stderr_text=stderr_text,
        diagnostics=diagnostics,
        error_text=error_text,
    )


def _render_diagnostics_panel(
    diagnostics: dict[str, Any] | None,
    error_text: str,
    metadata: dict[str, Any] | None,
) -> None:
    if error_text:
        st.error(error_text)

    if not isinstance(diagnostics, dict):
        st.info("当前还没有可展示的运行诊断。")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Status", str(diagnostics.get("status", "N/A")))
    c2.metric("Category", str(diagnostics.get("category", "N/A")))
    c3.metric("Failed Stage", str(diagnostics.get("failed_stage") or "N/A"))

    messages = diagnostics.get("messages") if isinstance(diagnostics.get("messages"), list) else []
    suggestions = diagnostics.get("suggestions") if isinstance(diagnostics.get("suggestions"), list) else []

    if messages:
        st.subheader("诊断结论")
        for item in messages:
            st.write(f"- {item}")

    if suggestions:
        st.subheader("建议下一步")
        for item in suggestions:
            st.write(f"- {item}")

    resolved_inputs = diagnostics.get("resolved_inputs") if isinstance(diagnostics.get("resolved_inputs"), dict) else {}
    input_rows = [{"name": key, "path": value} for key, value in resolved_inputs.items() if value]
    if input_rows:
        st.subheader("输入定位")
        st.dataframe(pd.DataFrame(input_rows), use_container_width=True, hide_index=True)

    if isinstance(metadata, dict) and metadata.get("out_dir"):
        st.subheader("输出目录")
        st.code(str(metadata["out_dir"]))

    stderr_excerpt = str(diagnostics.get("stderr_excerpt") or "")
    stdout_excerpt = str(diagnostics.get("stdout_excerpt") or "")
    if stderr_excerpt:
        st.subheader("stderr 摘要")
        st.code(stderr_excerpt)
    if stdout_excerpt:
        st.subheader("stdout 摘要")
        st.code(stdout_excerpt)


def _load_cd38_public_starter_summary() -> dict[str, Any] | None:
    current = st.session_state.get("last_cd38_public_starter")
    if isinstance(current, dict):
        return current
    return _load_json(CD38_PUBLIC_STARTER_SUMMARY_PATH)


def _run_cd38_public_starter_from_ui() -> None:
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()
    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            summary = run_cd38_public_starter(
                continue_on_error=True,
                python_executable=sys.executable,
            )
    except Exception as exc:
        st.session_state["last_cd38_public_starter"] = None
        st.session_state["last_cd38_public_starter_message"] = f"CD38 starter 刷新失败: {exc}"
        st.session_state["last_cd38_public_starter_stdout"] = stdout_buffer.getvalue()
        st.session_state["last_cd38_public_starter_stderr"] = stderr_buffer.getvalue() + traceback.format_exc()
        return

    st.session_state["last_cd38_public_starter"] = summary
    st.session_state["last_cd38_public_starter_stdout"] = stdout_buffer.getvalue()
    st.session_state["last_cd38_public_starter_stderr"] = stderr_buffer.getvalue()
    failed_count = int(summary.get("failed_command_count", 0)) if isinstance(summary, dict) else 0
    completed_count = int(summary.get("completed_command_count", 0)) if isinstance(summary, dict) else 0
    status_text = "成功" if failed_count == 0 else f"完成但有 {failed_count} 个子步骤失败"
    st.session_state["last_cd38_public_starter_message"] = (
        f"CD38 public starter 刷新{status_text}；成功子步骤 {completed_count} 个。"
    )


def _run_local_cli_command(command: list[str]) -> tuple[int, str, str]:
    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return int(completed.returncode), str(completed.stdout or ""), str(completed.stderr or "")


def _run_cd38_transfer_package_from_ui() -> None:
    command = [
        sys.executable,
        "package_cd38_external_tool_inputs.py",
        "--package_dir",
        str(CD38_EXTERNAL_INPUT_ROOT),
        "--out_dir",
        str(CD38_TRANSFER_ROOT),
    ]
    returncode, stdout_text, stderr_text = _run_local_cli_command(command)
    st.session_state["last_cd38_transfer_stdout"] = stdout_text
    st.session_state["last_cd38_transfer_stderr"] = stderr_text
    if returncode == 0:
        st.session_state["last_cd38_transfer_message"] = f"CD38 外部工具 transfer zip 已生成: {CD38_TRANSFER_ZIP_PATH}"
    else:
        st.session_state["last_cd38_transfer_message"] = f"CD38 transfer zip 生成失败，returncode={returncode}。"


def _normalize_local_path_text(value: str) -> Path | None:
    text = str(value or "").strip().strip('"').strip("'")
    if not text:
        return None
    path = Path(text).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _run_cd38_finalize_from_ui(*, import_dry_run: bool, run_discovered: bool, run_sensitivity: bool) -> None:
    source_path = _normalize_local_path_text(str(st.session_state.get("cd38_returned_source_path") or ""))
    if source_path is not None and not source_path.exists():
        st.session_state["last_cd38_finalize_message"] = f"返回包路径不存在: {source_path}"
        st.session_state["last_cd38_finalize_stdout"] = ""
        st.session_state["last_cd38_finalize_stderr"] = ""
        return

    if source_path is not None and not import_dry_run:
        gate_ok, gate_message = _run_cd38_preimport_gate(source_path)
        st.session_state["last_cd38_return_gate_message"] = gate_message
        if not gate_ok:
            st.session_state["last_cd38_finalize_message"] = "已阻止导入返回包；请先修复 gate 报告中的问题。"
            st.session_state["last_cd38_finalize_stdout"] = ""
            st.session_state["last_cd38_finalize_stderr"] = ""
            return

    command = [
        sys.executable,
        "finalize_cd38_external_benchmark.py",
        "--package_dir",
        str(CD38_EXTERNAL_INPUT_ROOT),
        "--continue_on_error",
    ]
    if source_path is not None:
        command.extend(["--import_source", str(source_path)])
    if import_dry_run:
        command.append("--import_dry_run")
    if run_discovered:
        command.append("--run_discovered")
    if run_sensitivity:
        command.append("--run_sensitivity")

    returncode, stdout_text, stderr_text = _run_local_cli_command(command)
    st.session_state["last_cd38_finalize_stdout"] = stdout_text
    st.session_state["last_cd38_finalize_stderr"] = stderr_text
    if source_path is not None:
        gate_returncode, gate_stdout, gate_stderr = _run_local_cli_command(
            [
                sys.executable,
                "build_cd38_return_package_gate.py",
                "--out_dir",
                str(CD38_IMPORT_GATE_ROOT),
            ]
        )
        st.session_state["last_cd38_return_gate_stdout"] = gate_stdout
        st.session_state["last_cd38_return_gate_stderr"] = gate_stderr
        if gate_returncode != 0:
            st.session_state["last_cd38_return_gate_message"] = (
                f"CD38 返回包安全门控生成失败，returncode={gate_returncode}。"
            )
        else:
            st.session_state["last_cd38_return_gate_message"] = "CD38 返回包安全门控已生成。"
    if returncode == 0:
        mode = "dry-run 检查" if import_dry_run else "finalize"
        st.session_state["last_cd38_finalize_message"] = f"CD38 外部输出 {mode} 已完成。"
    else:
        st.session_state["last_cd38_finalize_message"] = f"CD38 外部输出 finalize 失败，returncode={returncode}。"


def _run_cd38_return_package_gate_from_ui() -> None:
    command = [
        sys.executable,
        "build_cd38_return_package_gate.py",
        "--out_dir",
        str(CD38_IMPORT_GATE_ROOT),
    ]
    returncode, stdout_text, stderr_text = _run_local_cli_command(command)
    st.session_state["last_cd38_return_gate_stdout"] = stdout_text
    st.session_state["last_cd38_return_gate_stderr"] = stderr_text
    if returncode == 0:
        st.session_state["last_cd38_return_gate_message"] = "CD38 返回包安全门控已生成。"
    else:
        st.session_state["last_cd38_return_gate_message"] = f"CD38 返回包安全门控生成失败，returncode={returncode}。"


def _run_cd38_preimport_gate(source_path: Path) -> tuple[bool, str]:
    preimport_out = CD38_IMPORT_GATE_ROOT / "preimport_dry_run"
    dry_run_command = [
        sys.executable,
        "import_cd38_external_tool_outputs.py",
        "--source",
        str(source_path),
        "--package_dir",
        str(CD38_EXTERNAL_INPUT_ROOT),
        "--out_dir",
        str(preimport_out),
        "--dry_run",
    ]
    dry_returncode, dry_stdout, dry_stderr = _run_local_cli_command(dry_run_command)
    st.session_state["last_cd38_preimport_gate_stdout"] = dry_stdout
    st.session_state["last_cd38_preimport_gate_stderr"] = dry_stderr
    if dry_returncode != 0:
        return False, f"导入前 dry-run 失败，returncode={dry_returncode}。"

    gate_command = [
        sys.executable,
        "build_cd38_return_package_gate.py",
        "--import_summary_json",
        str(preimport_out / "cd38_external_tool_output_import_summary.json"),
        "--scan_csv",
        str(preimport_out / "cd38_external_tool_output_import_scan.csv"),
        "--out_dir",
        str(CD38_IMPORT_GATE_ROOT),
    ]
    gate_returncode, gate_stdout, gate_stderr = _run_local_cli_command(gate_command)
    st.session_state["last_cd38_return_gate_stdout"] = gate_stdout
    st.session_state["last_cd38_return_gate_stderr"] = gate_stderr
    if gate_returncode != 0:
        return False, f"导入前安全门控生成失败，returncode={gate_returncode}。"

    gate_summary = _load_json(CD38_IMPORT_GATE_ROOT / "cd38_return_package_gate_summary.json")
    gate_status = str(gate_summary.get("gate_status") if isinstance(gate_summary, dict) else "")
    decision_message = str(gate_summary.get("decision_message") if isinstance(gate_summary, dict) else "")
    if not gate_status.startswith("PASS"):
        return False, f"导入前安全门控拦截: {gate_status}。{decision_message}"
    return True, f"导入前安全门控通过: {gate_status}。"


def _run_cd38_return_import_selftest_from_ui() -> None:
    command = [
        sys.executable,
        "selftest_cd38_return_import_workflow.py",
        "--package_dir",
        str(CD38_EXTERNAL_INPUT_ROOT),
        "--work_dir",
        str(CD38_RETURN_SELFTEST_ROOT),
    ]
    returncode, stdout_text, stderr_text = _run_local_cli_command(command)
    st.session_state["last_cd38_return_selftest_stdout"] = stdout_text
    st.session_state["last_cd38_return_selftest_stderr"] = stderr_text
    if returncode == 0:
        st.session_state["last_cd38_return_selftest_message"] = "CD38 返回包导入自测通过。"
    else:
        st.session_state["last_cd38_return_selftest_message"] = f"CD38 返回包导入自测失败，returncode={returncode}。"


def _run_cd38_external_workflow_selftest_from_ui() -> None:
    command = [
        sys.executable,
        "selftest_cd38_external_workflow.py",
        "--work_dir",
        str(CD38_WORKFLOW_SELFTEST_ROOT),
    ]
    returncode, stdout_text, stderr_text = _run_local_cli_command(command)
    st.session_state["last_cd38_workflow_selftest_stdout"] = stdout_text
    st.session_state["last_cd38_workflow_selftest_stderr"] = stderr_text
    if returncode == 0:
        st.session_state["last_cd38_workflow_selftest_message"] = "CD38 外部工具链路一键自检通过。"
    else:
        st.session_state["last_cd38_workflow_selftest_message"] = f"CD38 外部工具链路一键自检失败，returncode={returncode}。"


def _download_file_button(label: str, path: Path, *, mime: str, key: str) -> None:
    if path.exists() and path.is_file():
        st.download_button(
            label=label,
            data=path.read_bytes(),
            file_name=path.name,
            mime=mime,
            use_container_width=True,
            key=key,
        )
    else:
        st.button(label, use_container_width=True, disabled=True, key=f"{key}_disabled")


def _open_path_button(label: str, path: Path, *, key: str) -> None:
    if st.button(label, use_container_width=True, disabled=not path.exists(), key=key):
        ok, message = _open_local_path(path)
        if ok:
            st.success(message)
        else:
            st.error(message)


def _render_cd38_transfer_and_return_panel() -> None:
    st.subheader("外部工具转移包和返回包检查")
    st.caption(
        "如果本机没有 P2Rank/fpocket，先生成 transfer zip，拿到 WSL/Linux 或另一台机器运行。"
        "跑完后把返回 zip/目录路径填回来，先 dry-run 检查，再导入并 finalize；"
        "正式导入前会自动先跑安全门控，非 `PASS_*` 状态会被拦截。"
    )

    transfer_cols = st.columns(4)
    if transfer_cols[0].button("生成 transfer zip", use_container_width=True, key="build_cd38_transfer_zip"):
        with st.spinner("正在生成 CD38 外部工具 transfer zip..."):
            _run_cd38_transfer_package_from_ui()
    with transfer_cols[1]:
        _download_file_button(
            "下载 transfer zip",
            CD38_TRANSFER_ZIP_PATH,
            mime="application/zip",
            key="download_cd38_transfer_zip",
        )
    with transfer_cols[2]:
        _open_path_button("打开 transfer 目录", CD38_TRANSFER_ROOT, key="open_cd38_transfer_root")
    with transfer_cols[3]:
        _open_path_button("打开输入包目录", CD38_EXTERNAL_INPUT_ROOT, key="open_cd38_external_input_root")

    transfer_message = str(st.session_state.get("last_cd38_transfer_message") or "")
    if transfer_message:
        if "失败" in transfer_message:
            st.error(transfer_message)
        else:
            st.success(transfer_message)

    transfer_summary = _load_json(CD38_TRANSFER_ROOT / "cd38_external_tool_inputs_transfer_summary.json")
    if isinstance(transfer_summary, dict):
        transfer_metric1, transfer_metric2, transfer_metric3 = st.columns(3)
        transfer_metric1.metric("Transfer files", str(transfer_summary.get("file_count", "N/A")))
        transfer_metric2.metric("Transfer size", _format_size_text(CD38_TRANSFER_ZIP_PATH))
        transfer_metric3.metric("Includes outputs", str(transfer_summary.get("include_existing_outputs", False)))
        runbook_refresh = (
            transfer_summary.get("runbook_refresh")
            if isinstance(transfer_summary.get("runbook_refresh"), dict)
            else {}
        )
        if runbook_refresh:
            selected_count = runbook_refresh.get("selected_action_count", "N/A")
            st.caption(f"Next-run runbook: {runbook_refresh.get('status', 'unknown')}；selected actions={selected_count}")

    runbook_cols = st.columns(4)
    with runbook_cols[0]:
        _download_file_button(
            "下载 next-run 说明",
            CD38_NEXT_RUN_REPORT_PATH,
            mime="text/markdown",
            key="download_cd38_next_run_md",
        )
    with runbook_cols[1]:
        _download_file_button(
            "下载 next-run CSV",
            CD38_NEXT_RUN_PLAN_PATH,
            mime="text/csv",
            key="download_cd38_next_run_csv",
        )
    with runbook_cols[2]:
        _download_file_button(
            "下载 PowerShell 脚本",
            CD38_NEXT_RUN_PS1_PATH,
            mime="text/plain",
            key="download_cd38_next_run_ps1",
        )
    with runbook_cols[3]:
        _download_file_button(
            "下载 Bash 脚本",
            CD38_NEXT_RUN_SH_PATH,
            mime="text/plain",
            key="download_cd38_next_run_sh",
        )

    returned_source_text = st.text_input(
        "返回 zip 或返回目录路径",
        key="cd38_returned_source_path",
        placeholder=r"例如 D:\path\returned_external_tool_inputs.zip",
        help="这里填外部机器跑完 P2Rank/fpocket 后带回来的 zip 或目录。不要填原始 transfer zip，dry-run 会识别这种情况。",
    )
    returned_source = _normalize_local_path_text(returned_source_text)
    returned_source_exists = bool(returned_source is not None and returned_source.exists())
    if returned_source is not None and not returned_source_exists:
        st.warning(f"当前返回路径不存在: {returned_source}")

    finalize_options = st.columns(3)
    run_discovered = finalize_options[0].checkbox(
        "导入后尝试加入 benchmark panel",
        value=True,
        key="cd38_finalize_run_discovered",
    )
    run_sensitivity = finalize_options[1].checkbox(
        "同时刷新参数敏感性",
        value=False,
        key="cd38_finalize_run_sensitivity",
    )
    finalize_options[2].caption("建议先 dry-run；确认候选输出数量后再导入。")

    finalize_cols = st.columns(3)
    if finalize_cols[0].button(
        "Dry-run 检查返回包",
        use_container_width=True,
        disabled=not returned_source_exists,
        key="cd38_finalize_dry_run",
    ):
        with st.spinner("正在 dry-run 检查 CD38 返回包..."):
            _run_cd38_finalize_from_ui(
                import_dry_run=True,
                run_discovered=bool(run_discovered),
                run_sensitivity=False,
            )
    if finalize_cols[1].button(
        "导入返回包并 finalize",
        use_container_width=True,
        disabled=not returned_source_exists,
        key="cd38_finalize_import",
    ):
        with st.spinner("正在导入 CD38 返回包并刷新 readiness..."):
            _run_cd38_finalize_from_ui(
                import_dry_run=False,
                run_discovered=bool(run_discovered),
                run_sensitivity=bool(run_sensitivity),
            )
    if finalize_cols[2].button(
        "仅刷新外部输出状态",
        use_container_width=True,
        key="cd38_finalize_status_only",
    ):
        with st.spinner("正在刷新 CD38 外部输出 preflight/readiness..."):
            _run_cd38_finalize_from_ui(
                import_dry_run=False,
                run_discovered=False,
                run_sensitivity=False,
            )

    finalize_message = str(st.session_state.get("last_cd38_finalize_message") or "")
    if finalize_message:
        if "失败" in finalize_message or "不存在" in finalize_message:
            st.error(finalize_message)
        else:
            st.success(finalize_message)

    finalize_summary_path = CD38_FINALIZE_ROOT / "cd38_external_benchmark_finalize_summary.json"
    finalize_report_path = CD38_FINALIZE_ROOT / "cd38_external_benchmark_finalize_report.md"
    finalize_summary = _load_json(finalize_summary_path)
    if isinstance(finalize_summary, dict):
        import_summary = (
            finalize_summary.get("import_summary")
            if isinstance(finalize_summary.get("import_summary"), dict)
            else {}
        )
        action_plan_summary = (
            finalize_summary.get("action_plan_summary")
            if isinstance(finalize_summary.get("action_plan_summary"), dict)
            else {}
        )
        status_cols = st.columns(4)
        status_cols[0].metric("P2Rank ready", str(finalize_summary.get("p2rank_outputs_ready", "N/A")))
        status_cols[1].metric("fpocket ready", str(finalize_summary.get("fpocket_outputs_ready", "N/A")))
        status_cols[2].metric("Runnable rows", str(finalize_summary.get("external_manifest_row_count", "N/A")))
        status_cols[3].metric("Import candidates", str(import_summary.get("candidate_file_count", "N/A")))
        diagnosis = str(import_summary.get("source_diagnosis") or "")
        if diagnosis:
            st.caption(f"返回包诊断: {diagnosis}")
        if action_plan_summary.get("overall_status"):
            st.caption(f"Action plan 状态: {action_plan_summary.get('overall_status')}")
        next_actions = finalize_summary.get("next_actions") if isinstance(finalize_summary.get("next_actions"), list) else []
        if next_actions:
            with st.expander("Finalize 下一步建议", expanded=False):
                for action in next_actions:
                    st.write(f"- {action}")

    report_cols = st.columns(4)
    with report_cols[0]:
        _download_file_button(
            "下载 finalize 报告",
            finalize_report_path,
            mime="text/markdown",
            key="download_cd38_finalize_report",
        )
    with report_cols[1]:
        _download_file_button(
            "下载返回 coverage CSV",
            CD38_IMPORT_ROOT / "cd38_external_tool_output_import_coverage.csv",
            mime="text/csv",
            key="download_cd38_import_coverage_csv",
        )
    with report_cols[2]:
        _download_file_button(
            "下载返回 repair plan",
            CD38_IMPORT_ROOT / "cd38_external_tool_output_import_repair_plan.csv",
            mime="text/csv",
            key="download_cd38_import_repair_plan_csv",
        )
    with report_cols[3]:
        _open_path_button("打开 finalize 目录", CD38_FINALIZE_ROOT, key="open_cd38_finalize_root")

    gate_summary_path = CD38_IMPORT_GATE_ROOT / "cd38_return_package_gate_summary.json"
    gate_report_path = CD38_IMPORT_GATE_ROOT / "cd38_return_package_gate_report.md"
    gate_decision_csv = CD38_IMPORT_GATE_ROOT / "cd38_return_package_gate_decision.csv"
    st.subheader("返回包安全门控")
    gate_cols = st.columns(4)
    if gate_cols[0].button("生成安全门控", use_container_width=True, key="run_cd38_return_gate"):
        with st.spinner("正在生成 CD38 返回包安全门控..."):
            _run_cd38_return_package_gate_from_ui()
    with gate_cols[1]:
        _download_file_button(
            "下载 gate 报告",
            gate_report_path,
            mime="text/markdown",
            key="download_cd38_return_gate_report",
        )
    with gate_cols[2]:
        _download_file_button(
            "下载 gate 决策 CSV",
            gate_decision_csv,
            mime="text/csv",
            key="download_cd38_return_gate_decision_csv",
        )
    with gate_cols[3]:
        _open_path_button("打开 gate 目录", CD38_IMPORT_GATE_ROOT, key="open_cd38_return_gate_root")

    gate_message = str(st.session_state.get("last_cd38_return_gate_message") or "")
    if gate_message:
        if "失败" in gate_message:
            st.error(gate_message)
        else:
            st.success(gate_message)

    gate_summary = _load_json(gate_summary_path)
    if isinstance(gate_summary, dict):
        gate_status = str(gate_summary.get("gate_status") or "N/A")
        coverage = gate_summary.get("coverage") if isinstance(gate_summary.get("coverage"), dict) else {}
        gate_status_cols = st.columns(4)
        gate_status_cols[0].metric("Gate", gate_status)
        gate_status_cols[1].metric("Candidates", str(gate_summary.get("candidate_file_count", "N/A")))
        gate_status_cols[2].metric(
            "Coverage",
            f"{coverage.get('ready_expected_output_count', 'N/A')}/{coverage.get('expected_output_count', 'N/A')}",
        )
        gate_status_cols[3].metric("Synthetic", str(gate_summary.get("synthetic_fixture_detected", "N/A")))
        decision_message = str(gate_summary.get("decision_message") or "")
        if gate_status.startswith("PASS"):
            st.success(decision_message)
        elif gate_status.startswith("WARN"):
            st.warning(decision_message)
        else:
            st.error(decision_message)
        actions = gate_summary.get("recommended_actions") if isinstance(gate_summary.get("recommended_actions"), list) else []
        if actions:
            with st.expander("Gate 建议动作", expanded=False):
                for action in actions:
                    st.write(f"- {action}")

    transfer_stdout = str(st.session_state.get("last_cd38_transfer_stdout") or "")
    transfer_stderr = str(st.session_state.get("last_cd38_transfer_stderr") or "")
    preimport_stdout = str(st.session_state.get("last_cd38_preimport_gate_stdout") or "")
    preimport_stderr = str(st.session_state.get("last_cd38_preimport_gate_stderr") or "")
    finalize_stdout = str(st.session_state.get("last_cd38_finalize_stdout") or "")
    finalize_stderr = str(st.session_state.get("last_cd38_finalize_stderr") or "")
    gate_stdout = str(st.session_state.get("last_cd38_return_gate_stdout") or "")
    gate_stderr = str(st.session_state.get("last_cd38_return_gate_stderr") or "")
    if (
        transfer_stdout
        or transfer_stderr
        or preimport_stdout
        or preimport_stderr
        or finalize_stdout
        or finalize_stderr
        or gate_stdout
        or gate_stderr
    ):
        with st.expander("查看 transfer/finalize 最近日志", expanded=False):
            if transfer_stdout:
                st.caption("transfer stdout")
                st.code(transfer_stdout)
            if transfer_stderr:
                st.caption("transfer stderr")
                st.code(transfer_stderr)
            if preimport_stdout:
                st.caption("pre-import dry-run stdout")
                st.code(preimport_stdout)
            if preimport_stderr:
                st.caption("pre-import dry-run stderr")
                st.code(preimport_stderr)
            if finalize_stdout:
                st.caption("finalize stdout")
                st.code(finalize_stdout)
            if finalize_stderr:
                st.caption("finalize stderr")
                st.code(finalize_stderr)
            if gate_stdout:
                st.caption("gate stdout")
                st.code(gate_stdout)
            if gate_stderr:
                st.caption("gate stderr")
                st.code(gate_stderr)

    with st.expander("返回包导入流程自测（不使用真实结果）", expanded=False):
        st.caption(
            "这个自测会生成一个合成返回包，只验证 importer 能否识别 `p2rank_outputs/` 和 "
            "`fpocket_runs/*/*_out/` 路径，并确认 expected coverage 能达到 6/6。"
            "它不是 CD38 pocket 准确性 benchmark，不能作为真实工具结果使用。"
        )
        selftest_cols = st.columns(3)
        if selftest_cols[0].button("运行返回包导入自测", use_container_width=True, key="run_cd38_return_selftest"):
            with st.spinner("正在运行 CD38 返回包导入自测..."):
                _run_cd38_return_import_selftest_from_ui()
        with selftest_cols[1]:
            _download_file_button(
                "下载自测报告",
                CD38_RETURN_SELFTEST_ROOT / "cd38_return_import_selftest_report.md",
                mime="text/markdown",
                key="download_cd38_return_selftest_report",
            )
        with selftest_cols[2]:
            _open_path_button("打开自测目录", CD38_RETURN_SELFTEST_ROOT, key="open_cd38_return_selftest_root")

        selftest_message = str(st.session_state.get("last_cd38_return_selftest_message") or "")
        if selftest_message:
            if "失败" in selftest_message:
                st.error(selftest_message)
            else:
                st.success(selftest_message)

        selftest_summary = _load_json(CD38_RETURN_SELFTEST_ROOT / "cd38_return_import_selftest_summary.json")
        if isinstance(selftest_summary, dict):
            importer = (
                selftest_summary.get("importer_summary")
                if isinstance(selftest_summary.get("importer_summary"), dict)
                else {}
            )
            coverage = importer.get("coverage") if isinstance(importer.get("coverage"), dict) else {}
            stest_col1, stest_col2, stest_col3, stest_col4 = st.columns(4)
            stest_col1.metric("Self-test", str(selftest_summary.get("status", "N/A")))
            stest_col2.metric("Candidates", str(importer.get("candidate_file_count", "N/A")))
            stest_col3.metric(
                "Coverage",
                f"{coverage.get('ready_expected_output_count', 'N/A')}/{coverage.get('expected_output_count', 'N/A')}",
            )
            stest_col4.metric("Synthetic files", str(selftest_summary.get("synthetic_file_count", "N/A")))

        selftest_stdout = str(st.session_state.get("last_cd38_return_selftest_stdout") or "")
        selftest_stderr = str(st.session_state.get("last_cd38_return_selftest_stderr") or "")
        if selftest_stdout or selftest_stderr:
            with st.expander("查看返回包导入自测日志", expanded=False):
                if selftest_stdout:
                    st.code(selftest_stdout)
                if selftest_stderr:
                    st.code(selftest_stderr)

    with st.expander("CD38 外部工具链路一键自检", expanded=False):
        st.caption(
            "这个自检会连续验证 transfer zip 生成、原始 transfer zip 的 strict gate 拦截、"
            "synthetic returned fixture 的 importer/gate 行为，以及 public starter 刷新。"
            "它不运行 P2Rank/fpocket，也不会产生真实 benchmark 证据。"
        )
        workflow_cols = st.columns(3)
        if workflow_cols[0].button("运行外部链路一键自检", use_container_width=True, key="run_cd38_workflow_selftest"):
            with st.spinner("正在运行 CD38 外部工具链路一键自检..."):
                _run_cd38_external_workflow_selftest_from_ui()
        with workflow_cols[1]:
            _download_file_button(
                "下载链路自检报告",
                CD38_WORKFLOW_SELFTEST_ROOT / "cd38_external_workflow_selftest_report.md",
                mime="text/markdown",
                key="download_cd38_workflow_selftest_report",
            )
        with workflow_cols[2]:
            _open_path_button("打开链路自检目录", CD38_WORKFLOW_SELFTEST_ROOT, key="open_cd38_workflow_selftest_root")

        workflow_message = str(st.session_state.get("last_cd38_workflow_selftest_message") or "")
        if workflow_message:
            if "失败" in workflow_message:
                st.error(workflow_message)
            else:
                st.success(workflow_message)

        workflow_summary = _load_json(CD38_WORKFLOW_SELFTEST_ROOT / "cd38_external_workflow_selftest_summary.json")
        if isinstance(workflow_summary, dict):
            workflow_status_cols = st.columns(4)
            workflow_status_cols[0].metric("Workflow", str(workflow_summary.get("overall_status", "N/A")))
            workflow_status_cols[1].metric(
                "Passed steps",
                f"{workflow_summary.get('passed_step_count', 'N/A')}/{workflow_summary.get('step_count', 'N/A')}",
            )
            workflow_status_cols[2].metric("Original gate", str(workflow_summary.get("original_transfer_gate_status", "N/A")))
            workflow_status_cols[3].metric("Fixture gate", str(workflow_summary.get("synthetic_fixture_gate_status", "N/A")))

        workflow_stdout = str(st.session_state.get("last_cd38_workflow_selftest_stdout") or "")
        workflow_stderr = str(st.session_state.get("last_cd38_workflow_selftest_stderr") or "")
        if workflow_stdout or workflow_stderr:
            with st.expander("查看外部链路一键自检日志", expanded=False):
                if workflow_stdout:
                    st.code(workflow_stdout)
                if workflow_stderr:
                    st.code(workflow_stderr)


def _render_cd38_public_starter_panel() -> None:
    st.subheader("CD38 公开结构 benchmark starter")
    st.caption(
        "这个面板只刷新公开 CD38 benchmark 的本地产物和 action plan，不运行外部 P2Rank/fpocket，"
        "也不改变当前 ML 排名分数。它的作用是把“还缺哪些真实外部 pocket 输出”直接暴露出来。"
    )

    run_col, _, _ = st.columns(3)
    if run_col.button("刷新 CD38 public starter", use_container_width=True, key="refresh_cd38_public_starter"):
        with st.spinner("正在刷新 CD38 public starter..."):
            _run_cd38_public_starter_from_ui()

    open_col1, open_col2, open_col3 = st.columns(3)
    with open_col1:
        _open_path_button(
            "打开 CD38 benchmark 文件夹",
            CD38_BENCHMARK_ROOT,
            key="open_cd38_benchmark_root",
        )
    with open_col2:
        _open_path_button(
            "打开 starter 报告",
            CD38_PUBLIC_STARTER_REPORT_PATH,
            key="open_cd38_public_starter_report",
        )
    with open_col3:
        _open_path_button(
            "打开外部工具输入包",
            CD38_BENCHMARK_ROOT / "external_tool_inputs",
            key="open_cd38_external_inputs",
        )

    message = str(st.session_state.get("last_cd38_public_starter_message") or "")
    if message:
        if "失败" in message:
            st.error(message)
        else:
            st.success(message)

    summary = _load_cd38_public_starter_summary()
    if not isinstance(summary, dict):
        st.info("当前还没有 CD38 public starter summary。点击上方按钮生成。")
        return

    panel = summary.get("panel") if isinstance(summary.get("panel"), dict) else {}
    readiness = summary.get("readiness") if isinstance(summary.get("readiness"), dict) else {}
    action_plan = summary.get("action_plan") if isinstance(summary.get("action_plan"), dict) else {}
    proxy_calibration = summary.get("proxy_calibration") if isinstance(summary.get("proxy_calibration"), dict) else {}
    proxy_recommendation = (
        proxy_calibration.get("recommendation")
        if isinstance(proxy_calibration.get("recommendation"), dict)
        else {}
    )

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Panel rows", int(panel.get("row_count", 0) or 0))
    metric_col2.metric("Missing / pending", str(readiness.get("missing_or_pending_count", "N/A")))
    metric_col3.metric("Action plan", str(action_plan.get("overall_status") or "N/A"))
    metric_col4.metric("Missing benchmark gaps", str(action_plan.get("missing_benchmark_gap_action_count") or "N/A"))

    method_counts = panel.get("method_counts") if isinstance(panel.get("method_counts"), dict) else {}
    if method_counts:
        st.caption("当前 panel 方法构成: " + ", ".join(f"{key}={value}" for key, value in method_counts.items()))

    if action_plan.get("overall_status") == "blocked_external_outputs_missing":
        st.warning(
            "当前 blocker 不是本地代码不能跑，而是缺真实外部 P2Rank/fpocket 输出。"
            "优先按 action plan 补 `3ROP/4OGW fpocket`，再补 `3F6Y P2Rank/fpocket`。"
        )

    if proxy_recommendation:
        st.info(
            "Proxy 校准结论: "
            f"{proxy_recommendation.get('recommended_policy', 'N/A')}；"
            f"默认 overwide penalty weight={proxy_recommendation.get('recommended_default_penalty_weight', 'N/A')}。"
        )

    action_plan_md = CD38_BENCHMARK_ROOT / "action_plan" / "cd38_external_benchmark_action_plan.md"
    action_plan_csv = CD38_BENCHMARK_ROOT / "action_plan" / "cd38_external_benchmark_action_plan.csv"
    preflight_report = (
        CD38_BENCHMARK_ROOT
        / "external_tool_inputs"
        / "preflight"
        / "cd38_external_tool_preflight_report.md"
    )
    expected_returns_csv = CD38_BENCHMARK_ROOT / "external_tool_inputs" / "cd38_external_tool_expected_returns.csv"
    return_checklist_md = CD38_BENCHMARK_ROOT / "external_tool_inputs" / "cd38_external_tool_return_checklist.md"
    readiness_report = CD38_BENCHMARK_ROOT / "readiness" / "cd38_benchmark_readiness_report.md"

    st.subheader("关键产物")
    file_col1, file_col2, file_col3, file_col4 = st.columns(4)
    with file_col1:
        _download_file_button(
            "下载 starter 报告",
            CD38_PUBLIC_STARTER_REPORT_PATH,
            mime="text/markdown",
            key="download_cd38_public_starter_report",
        )
    with file_col2:
        _download_file_button(
            "下载 action plan CSV",
            action_plan_csv,
            mime="text/csv",
            key="download_cd38_action_plan_csv",
        )
    with file_col3:
        _download_file_button(
            "下载 expected returns CSV",
            expected_returns_csv,
            mime="text/csv",
            key="download_cd38_expected_returns_csv",
        )
    with file_col4:
        _download_file_button(
            "下载返回检查清单",
            return_checklist_md,
            mime="text/markdown",
            key="download_cd38_return_checklist_md",
        )

    open_cols = st.columns(3)
    with open_cols[0]:
        _open_path_button("打开 action plan", action_plan_md, key="open_cd38_action_plan_md")
    with open_cols[1]:
        _open_path_button("打开 preflight 报告", preflight_report, key="open_cd38_preflight_report")
    with open_cols[2]:
        _open_path_button("打开 readiness 报告", readiness_report, key="open_cd38_readiness_report")

    st.divider()
    _render_cd38_transfer_and_return_panel()

    action_df = _load_csv(action_plan_csv)
    if action_df is not None and not action_df.empty:
        st.subheader("Action plan 预览")
        preview_columns = [
            col
            for col in ["priority", "status", "pdb_id", "method", "action_type", "recommended_action", "expected_return_path"]
            if col in action_df.columns
        ]
        st.dataframe(action_df[preview_columns].head(12), use_container_width=True, hide_index=True)

    if CD38_PUBLIC_STARTER_REPORT_PATH.exists():
        with st.expander("查看 starter 报告正文", expanded=False):
            st.markdown(_read_text(CD38_PUBLIC_STARTER_REPORT_PATH))

    starter_stdout = str(st.session_state.get("last_cd38_public_starter_stdout") or "")
    starter_stderr = str(st.session_state.get("last_cd38_public_starter_stderr") or "")
    if starter_stdout or starter_stderr:
        with st.expander("查看最近一次 CD38 starter 刷新日志", expanded=False):
            if starter_stdout:
                st.code(starter_stdout)
            if starter_stderr:
                st.code(starter_stderr)


def _initialize_state() -> None:
    for key, value in FORM_DEFAULTS.items():
        st.session_state.setdefault(key, value)

    st.session_state.setdefault("bundle_zip_local_path", "")
    st.session_state.setdefault("bundle_dir_local_path", "")
    st.session_state.setdefault("last_run_dir", None)
    st.session_state.setdefault("last_summary", None)
    st.session_state.setdefault("last_metadata", None)
    st.session_state.setdefault("last_command", None)
    st.session_state.setdefault("last_stdout", "")
    st.session_state.setdefault("last_stderr", "")
    st.session_state.setdefault("last_diagnostics", None)
    st.session_state.setdefault("last_error", "")
    st.session_state.setdefault("last_export_bundle", None)
    st.session_state.setdefault("last_export_message", "")
    st.session_state.setdefault("last_export_html", None)
    st.session_state.setdefault("last_export_html_message", "")
    st.session_state.setdefault("last_export_pdf", None)
    st.session_state.setdefault("last_export_pdf_message", "")
    st.session_state.setdefault("last_compare_export_html", None)
    st.session_state.setdefault("last_compare_export_html_message", "")
    st.session_state.setdefault("last_compare_export_pdf", None)
    st.session_state.setdefault("last_compare_export_pdf_message", "")
    st.session_state.setdefault("last_bundle_import", None)
    st.session_state.setdefault("last_bundle_import_message", "")
    st.session_state.setdefault("last_preflight", None)
    st.session_state.setdefault("last_cd38_public_starter", None)
    st.session_state.setdefault("last_cd38_public_starter_message", "")
    st.session_state.setdefault("last_cd38_public_starter_stdout", "")
    st.session_state.setdefault("last_cd38_public_starter_stderr", "")
    st.session_state.setdefault("last_cd38_transfer_message", "")
    st.session_state.setdefault("last_cd38_transfer_stdout", "")
    st.session_state.setdefault("last_cd38_transfer_stderr", "")
    st.session_state.setdefault("cd38_returned_source_path", "")
    st.session_state.setdefault("last_cd38_finalize_message", "")
    st.session_state.setdefault("last_cd38_finalize_stdout", "")
    st.session_state.setdefault("last_cd38_finalize_stderr", "")
    st.session_state.setdefault("last_cd38_preimport_gate_stdout", "")
    st.session_state.setdefault("last_cd38_preimport_gate_stderr", "")
    st.session_state.setdefault("last_cd38_return_gate_message", "")
    st.session_state.setdefault("last_cd38_return_gate_stdout", "")
    st.session_state.setdefault("last_cd38_return_gate_stderr", "")
    st.session_state.setdefault("last_cd38_return_selftest_message", "")
    st.session_state.setdefault("last_cd38_return_selftest_stdout", "")
    st.session_state.setdefault("last_cd38_return_selftest_stderr", "")
    st.session_state.setdefault("last_cd38_workflow_selftest_message", "")
    st.session_state.setdefault("last_cd38_workflow_selftest_stdout", "")
    st.session_state.setdefault("last_cd38_workflow_selftest_stderr", "")
    st.session_state.setdefault("run_queue", [])
    st.session_state.setdefault("active_process", None)
    st.session_state.setdefault("active_run_info", None)
    st.session_state.setdefault("queue_auto_run", False)
    st.session_state.setdefault("last_scheduler_message", "")
    st.session_state.setdefault("last_plan_override_message", "")
    st.session_state.setdefault("validation_feedback_feature_csv_path", "")
    st.session_state.setdefault("validation_feedback_consensus_csv_path", "")
    st.session_state.setdefault("validation_feedback_label_col", "experiment_label")
    st.session_state.setdefault("validation_retrain_compare_labels_csv_path", "")
    st.session_state.setdefault("validation_retrain_compare_top_k", 10)


def _apply_pending_plan_override_path() -> None:
    pending_path = str(st.session_state.pop("_pending_experiment_plan_override_local_path", "") or "").strip()
    pending_message = str(st.session_state.pop("_pending_experiment_plan_override_message", "") or "").strip()
    if pending_path:
        st.session_state["experiment_plan_override_local_path"] = pending_path
    if pending_message:
        st.session_state["last_plan_override_message"] = pending_message

    pending_feature_path = str(st.session_state.pop("_pending_feature_csv_local_path", "") or "").strip()
    pending_feature_label_col = str(st.session_state.pop("_pending_feature_label_col", "") or "").strip()
    pending_feature_message = str(st.session_state.pop("_pending_feature_csv_message", "") or "").strip()
    if pending_feature_path:
        st.session_state["start_mode"] = "feature_csv"
        st.session_state["feature_csv_local_path"] = pending_feature_path
        st.session_state["label_col"] = pending_feature_label_col or "experiment_label"
        st.session_state["disable_label_aware_steps"] = False
    if pending_feature_message:
        st.session_state["last_scheduler_message"] = pending_feature_message


def _run_pipeline_from_form(
    *,
    input_csv_upload: Any,
    feature_csv_upload: Any,
    default_pocket_upload: Any,
    default_catalytic_upload: Any,
    default_ligand_upload: Any,
    experiment_plan_override_upload: Any,
) -> None:
    form_payload = _build_form_payload()
    safe_run_name = _slugify_name(str(form_payload.get("run_name", "")))
    run_root = LOCAL_APP_RUN_ROOT / safe_run_name
    input_dir = run_root / "inputs"
    out_dir = run_root / "outputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_csv = _resolve_input_file(
        uploaded_file=input_csv_upload,
        local_path_text=str(form_payload.get("input_csv_local_path") or ""),
        dst_dir=input_dir,
        required=str(form_payload.get("start_mode")) == "input_csv",
        label="input_csv",
    )
    feature_csv = _resolve_input_file(
        uploaded_file=feature_csv_upload,
        local_path_text=str(form_payload.get("feature_csv_local_path") or ""),
        dst_dir=input_dir,
        required=str(form_payload.get("start_mode")) == "feature_csv",
        label="feature_csv",
    )
    default_pocket_file = _resolve_input_file(
        uploaded_file=default_pocket_upload,
        local_path_text=str(form_payload.get("default_pocket_local_path") or ""),
        dst_dir=input_dir,
        required=False,
        label="default_pocket_file",
    )
    default_catalytic_file = _resolve_input_file(
        uploaded_file=default_catalytic_upload,
        local_path_text=str(form_payload.get("default_catalytic_local_path") or ""),
        dst_dir=input_dir,
        required=False,
        label="default_catalytic_file",
    )
    default_ligand_file = _resolve_input_file(
        uploaded_file=default_ligand_upload,
        local_path_text=str(form_payload.get("default_ligand_local_path") or ""),
        dst_dir=input_dir,
        required=False,
        label="default_ligand_file",
    )
    experiment_plan_override_file = _resolve_input_file(
        uploaded_file=experiment_plan_override_upload,
        local_path_text=str(form_payload.get("experiment_plan_override_local_path") or ""),
        dst_dir=input_dir,
        required=False,
        label="experiment_plan_override_csv",
    )

    resolved_inputs = {
        "input_csv": None if input_csv is None else str(input_csv),
        "feature_csv": None if feature_csv is None else str(feature_csv),
        "default_pocket_file": None if default_pocket_file is None else str(default_pocket_file),
        "default_catalytic_file": None if default_catalytic_file is None else str(default_catalytic_file),
        "default_ligand_file": None if default_ligand_file is None else str(default_ligand_file),
        "experiment_plan_override_csv": None
        if experiment_plan_override_file is None
        else str(experiment_plan_override_file),
    }

    command = _build_pipeline_command(
        start_mode=str(form_payload.get("start_mode")),
        input_csv=input_csv,
        feature_csv=feature_csv,
        out_dir=out_dir,
        default_pocket_file=default_pocket_file,
        default_catalytic_file=default_catalytic_file,
        default_ligand_file=default_ligand_file,
        experiment_plan_override_file=experiment_plan_override_file,
        default_antigen_chain=str(form_payload.get("default_antigen_chain") or ""),
        default_nanobody_chain=str(form_payload.get("default_nanobody_chain") or ""),
        label_col=str(form_payload.get("label_col") or "label"),
        top_k=int(form_payload.get("top_k", 3)),
        pocket_overwide_penalty_weight=float(form_payload.get("pocket_overwide_penalty_weight", 0.0)),
        pocket_overwide_threshold=float(form_payload.get("pocket_overwide_threshold", 0.55)),
        train_epochs=int(form_payload.get("train_epochs", 20)),
        train_batch_size=int(form_payload.get("train_batch_size", 64)),
        train_val_ratio=float(form_payload.get("train_val_ratio", 0.25)),
        train_early_stopping_patience=int(form_payload.get("train_early_stopping_patience", 8)),
        seed=int(form_payload.get("seed", 42)),
        skip_failed_rows=bool(form_payload.get("skip_failed_rows", True)),
        disable_label_aware_steps=bool(form_payload.get("disable_label_aware_steps", False)),
        enable_ai_assistant=bool(form_payload.get("enable_ai_assistant", False)),
        ai_provider=str(form_payload.get("ai_provider") or "none"),
        ai_model=str(form_payload.get("ai_model") or ""),
        ai_max_rows=int(form_payload.get("ai_max_rows", 8)),
    )
    pipeline_kwargs = _build_pipeline_kwargs(
        form_payload=form_payload,
        out_dir=out_dir,
        input_csv=input_csv,
        feature_csv=feature_csv,
        default_pocket_file=default_pocket_file,
        default_catalytic_file=default_catalytic_file,
        default_ligand_file=default_ligand_file,
        experiment_plan_override_file=experiment_plan_override_file,
    )

    started_at = _now_text()
    returncode, stdout_text, stderr_text, summary = _run_pipeline_in_process(pipeline_kwargs)
    _write_text(run_root / APP_STDOUT_NAME, stdout_text)
    _write_text(run_root / APP_STDERR_NAME, stderr_text)

    summary_path = out_dir / "recommended_pipeline_summary.json"
    if summary is None:
        summary = _load_json(summary_path)

    if returncode == 0:
        diagnostics = _build_success_diagnostics(summary, resolved_inputs, out_dir)
        status = "success"
        error_text = ""
    else:
        diagnostics = _build_failure_diagnostics(
            returncode=int(returncode),
            stdout_text=stdout_text,
            stderr_text=stderr_text,
            summary=summary,
            out_dir=out_dir,
            resolved_inputs=resolved_inputs,
        )
        status = "failed"
        error_text = (
            f"Pipeline failed with rc={returncode}. "
            f"Check diagnostics, stderr and output directory: {out_dir}"
        )

    metadata = {
        "run_name": safe_run_name,
        "started_at": started_at,
        "status": status,
        "execution_mode": "in_process_function",
        "start_mode": str(form_payload.get("start_mode")),
        "form_payload": form_payload,
        "resolved_inputs": resolved_inputs,
        "command": command,
        "pipeline_kwargs": pipeline_kwargs,
        "returncode": int(returncode),
        "out_dir": str(out_dir),
        "summary_path": str(summary_path),
        "diagnostics": diagnostics,
        "error_text": error_text,
    }
    _write_json(run_root / APP_RUN_METADATA_NAME, metadata)

    _set_last_run_state(
        run_root=run_root,
        summary=summary,
        metadata=metadata,
        stdout_text=stdout_text,
        stderr_text=stderr_text,
        diagnostics=diagnostics,
        error_text=error_text,
    )

    if returncode != 0:
        raise RuntimeError(error_text)


def main() -> None:
    st.set_page_config(
        page_title=f"{APP_NAME} v{APP_VERSION}",
        page_icon=_load_page_icon(),
        layout="wide",
    )
    _initialize_state()
    _apply_pending_plan_override_path()
    _sync_background_run_state()

    st.title(f"{APP_NAME} 本地交互运行器")
    st.caption(
        f"版本 {APP_VERSION} | 通道 {APP_RELEASE_CHANNEL} | "
        "基于当前仓库现有 CLI 流程构建的本地交互壳，当前重点支持上传/路径输入、参数模板、运行历史、失败诊断以及基础任务队列。"
    )

    history_records = _build_history_records()
    history_label_to_path = {str(item["label"]): Path(item["run_root"]) for item in history_records}
    template_files = _list_parameter_templates()
    template_names = [path.stem for path in template_files]

    with st.sidebar:
        st.header("运行配置")
        st.radio(
            "启动方式",
            options=["input_csv", "feature_csv"],
            format_func=lambda x: START_MODE_LABELS.get(x, x),
            key="start_mode",
        )
        st.text_input("运行名称", key="run_name")
        st.caption("每次运行都会写入 local_app_runs/<运行名称>/")

        st.subheader("参数模板")
        st.text_input("模板名称", key="template_name")
        st.selectbox(
            "已保存模板",
            options=[""] + template_names,
            format_func=lambda x: "请选择模板" if not x else x,
            key="selected_template_name",
        )
        template_col1, template_col2 = st.columns(2)
        save_template_clicked = template_col1.button("保存模板", use_container_width=True)
        load_template_clicked = template_col2.button("载入模板", use_container_width=True)

        st.subheader("运行历史")
        history_sidebar_filter_text = str(
            st.text_input(
                "筛选历史记录",
                key="history_sidebar_filter_text",
                placeholder="run_name / status / start_mode",
            )
            or ""
        ).strip()
        sidebar_history_records = _filter_history_records(
            history_records,
            keyword=history_sidebar_filter_text,
        )
        history_labels = [""] + [str(item["label"]) for item in sidebar_history_records]
        st.selectbox(
            "最近运行记录",
            options=history_labels,
            format_func=lambda x: "请选择历史记录" if not x else x,
            key="selected_history_label",
        )
        st.caption(f"当前匹配 {len(sidebar_history_records)} 条历史记录。")
        history_action_col1, history_action_col2 = st.columns(2)
        load_history_clicked = history_action_col1.button("载入历史结果", use_container_width=True)
        requeue_history_clicked = history_action_col2.button("复制并入队", use_container_width=True)

        st.subheader("数据包导入")
        bundle_zip_upload = st.file_uploader("上传 zip 数据包", type=["zip"], key="bundle_zip_upload")
        st.text_input("或填写 zip 本地路径", key="bundle_zip_local_path")
        st.text_input("或填写数据目录路径", key="bundle_dir_local_path")
        bundle_import_clicked = st.button("导入并自动识别输入", use_container_width=True)
        st.caption("支持两种方式：导入 zip，或直接扫描本地数据目录。两者二选一。")

        st.subheader("没有数据时")
        demo_active_running = isinstance(st.session_state.get("active_run_info"), dict)
        demo_col1, demo_col2 = st.columns(2)
        demo_col1.button(
            "生成并载入 demo 输入",
            use_container_width=True,
            disabled=demo_active_running,
            on_click=_load_demo_inputs_into_session_state,
            help="生成 synthetic pose_features 和 synthetic validation override，只用于演示流程，不代表真实实验结论。",
        )
        demo_col2.button(
            "生成并立即运行 demo",
            use_container_width=True,
            disabled=demo_active_running,
            on_click=_run_demo_now_from_session_state,
            help="自动生成 demo 输入并启动后台运行；结果会进入 local_app_runs。",
        )
        st.caption("demo 数据只用于检查安装、流程和导出效果，不代表真实实验结论。")

        st.subheader("主输入")
        input_csv_upload = None
        feature_csv_upload = None

        if str(st.session_state.get("start_mode")) == "input_csv":
            input_csv_upload = st.file_uploader("上传 input_pose_table.csv", type=["csv"], key="input_csv_upload")
            st.text_input("或填写 input_csv 本地路径", key="input_csv_local_path")
            st.caption("如果 CSV 里的 pdb_path / pocket_file 等已经是本机可访问路径，应用会直接沿用。")
        else:
            feature_csv_upload = st.file_uploader("上传 pose_features.csv", type=["csv"], key="feature_csv_upload")
            st.text_input("或填写 feature_csv 本地路径", key="feature_csv_local_path")

        st.subheader("可选默认文件")
        default_pocket_upload = st.file_uploader("上传 default pocket file", key="default_pocket_upload")
        st.text_input("或填写 default_pocket_file 本地路径", key="default_pocket_local_path")
        default_catalytic_upload = st.file_uploader("上传 default catalytic file", key="default_catalytic_upload")
        st.text_input("或填写 default_catalytic_file 本地路径", key="default_catalytic_local_path")
        default_ligand_upload = st.file_uploader("上传 default ligand file", key="default_ligand_upload")
        st.text_input("或填写 default_ligand_file 本地路径", key="default_ligand_local_path")

        st.subheader("实验计划覆盖（可选）")
        experiment_plan_override_upload = st.file_uploader(
            "上传 experiment_plan_override.csv",
            type=["csv"],
            key="experiment_plan_override_upload",
        )
        st.text_input("或填写 experiment_plan_override_csv 本地路径", key="experiment_plan_override_local_path")
        st.caption("用于手工 include/exclude/standby/defer 候选，并把 owner/cost/status/note 写入实验计划单。")
        plan_override_message = str(st.session_state.get("last_plan_override_message") or "").strip()
        if plan_override_message:
            st.success(plan_override_message)

        st.subheader("常用参数")
        st.text_input("default_antigen_chain", key="default_antigen_chain")
        st.text_input("default_nanobody_chain", key="default_nanobody_chain")
        st.text_input(
            "label_col",
            key="label_col",
            help="用于训练、compare 和 calibration 的标签列。默认 label；验证回灌特征表通常使用 experiment_label。",
        )
        st.number_input("top_k", min_value=1, max_value=20, step=1, key="top_k")
        st.number_input(
            "pocket_overwide_penalty_weight",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            format="%.2f",
            key="pocket_overwide_penalty_weight",
            help="默认 0 不改变排名；大于 0 时会轻微惩罚 pocket_shape_overwide_proxy 偏高的口袋。",
        )
        st.number_input(
            "pocket_overwide_threshold",
            min_value=0.0,
            max_value=0.99,
            step=0.05,
            format="%.2f",
            key="pocket_overwide_threshold",
            help="超过该阈值后才开始计算可选 overwide penalty。",
        )
        st.number_input("train_epochs", min_value=1, max_value=2000, step=1, key="train_epochs")
        st.number_input("train_batch_size", min_value=1, max_value=4096, step=1, key="train_batch_size")
        st.number_input("train_val_ratio", min_value=0.05, max_value=0.95, step=0.05, key="train_val_ratio")
        st.number_input(
            "train_early_stopping_patience",
            min_value=1,
            max_value=200,
            step=1,
            key="train_early_stopping_patience",
        )
        st.number_input("seed", min_value=0, max_value=999999, step=1, key="seed")
        st.checkbox("skip_failed_rows", key="skip_failed_rows")
        st.checkbox("disable_label_aware_steps", key="disable_label_aware_steps")

        st.subheader("AI 解释（可选）")
        st.checkbox(
            "生成 AI/离线解释报告",
            key="enable_ai_assistant",
            help="默认只做本地离线摘要；选择 OpenAI 时也只发送压缩后的 summary 和表格前几行，不上传原始 PDB。",
        )
        st.selectbox(
            "AI provider",
            options=["none", "openai"],
            key="ai_provider",
            help="none 表示完全离线；openai 需要设置 OPENAI_API_KEY，失败时会回退为离线摘要。",
        )
        st.text_input(
            "AI model",
            key="ai_model",
            placeholder="默认 gpt-5.4-nano",
            help="仅 provider=openai 时使用；留空则使用 ai_assistant.py 默认模型。",
        )
        st.number_input("AI 摘要表格行数", min_value=1, max_value=50, step=1, key="ai_max_rows")

        current_start_mode = str(st.session_state.get("start_mode") or "input_csv")
        active_running = isinstance(st.session_state.get("active_run_info"), dict)
        run_ready = (
            _source_present(input_csv_upload, str(st.session_state.get("input_csv_local_path") or ""))
            if current_start_mode == "input_csv"
            else _source_present(feature_csv_upload, str(st.session_state.get("feature_csv_local_path") or ""))
        )
        action_row1 = st.columns(2)
        check_inputs_clicked = action_row1[0].button(
            "检查当前输入",
            use_container_width=True,
            disabled=not run_ready,
        )
        run_clicked = action_row1[1].button(
            "立即运行",
            use_container_width=True,
            type="primary",
            disabled=(not run_ready) or active_running,
        )
        action_row2 = st.columns(2)
        enqueue_clicked = action_row2[0].button(
            "加入队列",
            use_container_width=True,
            disabled=not run_ready,
        )
        start_queue_clicked = action_row2[1].button(
            "启动队列",
            use_container_width=True,
            disabled=active_running or not bool(st.session_state.get("run_queue")),
        )
        action_row3 = st.columns(2)
        stop_active_clicked = action_row3[0].button(
            "停止当前运行",
            use_container_width=True,
            disabled=not active_running,
        )
        clear_queue_clicked = action_row3[1].button(
            "清空队列",
            use_container_width=True,
            disabled=not bool(st.session_state.get("run_queue")),
        )
        if not run_ready:
            st.caption("先提供当前启动方式所需的主输入 CSV，再执行检查、加入队列或运行。")
        elif active_running:
            st.caption("当前已有任务在运行。你仍然可以把新的配置加入队列，或先停止当前任务。")

    if save_template_clicked:
        template_name = str(st.session_state.get("template_name") or "").strip()
        if not template_name:
            st.warning("模板名称不能为空。")
        else:
            out_path = _save_parameter_template(template_name)
            st.success(f"已保存模板: {out_path.stem}")
            st.rerun()

    if load_template_clicked:
        selected_template_name = str(st.session_state.get("selected_template_name") or "").strip()
        if not selected_template_name:
            st.warning("请先选择一个模板。")
        else:
            payload = _load_parameter_template(selected_template_name)
            if not payload or not isinstance(payload.get("fields"), dict):
                st.error("模板文件无效，无法载入。")
            else:
                _apply_form_payload(payload["fields"])
                st.success(f"已载入模板: {selected_template_name}")
                st.rerun()

    if bundle_import_clicked:
        try:
            report = _import_bundle_source(
                uploaded_file=bundle_zip_upload,
                zip_local_path_text=str(st.session_state.get("bundle_zip_local_path") or ""),
                directory_local_path_text=str(st.session_state.get("bundle_dir_local_path") or ""),
            )
            st.session_state["last_bundle_import"] = report
            st.session_state["last_bundle_import_message"] = _apply_bundle_import(report)
            st.session_state["last_preflight"] = None
            st.success(st.session_state["last_bundle_import_message"])
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    if enqueue_clicked:
        try:
            preflight_report = _execute_preflight_check(
                start_mode=str(st.session_state.get("start_mode") or "input_csv"),
                input_csv_upload=input_csv_upload,
                feature_csv_upload=feature_csv_upload,
                default_pocket_upload=default_pocket_upload,
                default_catalytic_upload=default_catalytic_upload,
                default_ligand_upload=default_ligand_upload,
                experiment_plan_override_upload=experiment_plan_override_upload,
                form_payload=_build_form_payload(),
            )
            if str(preflight_report.get("status") or "") == "error":
                raise ValueError("运行前检查发现阻塞问题，请先修正后再加入队列。")
            run_request = _prepare_run_request_from_form(
                input_csv_upload=input_csv_upload,
                feature_csv_upload=feature_csv_upload,
                default_pocket_upload=default_pocket_upload,
                default_catalytic_upload=default_catalytic_upload,
                default_ligand_upload=default_ligand_upload,
                experiment_plan_override_upload=experiment_plan_override_upload,
            )
            queue = st.session_state.get("run_queue")
            if not isinstance(queue, list):
                queue = []
            queue.append(run_request)
            st.session_state["run_queue"] = queue
            st.session_state["last_scheduler_message"] = f"已加入队列: {run_request['run_name']}"
            st.success(st.session_state["last_scheduler_message"])
        except Exception as exc:
            st.error(str(exc))

    if start_queue_clicked:
        st.session_state["queue_auto_run"] = True
        started = _maybe_start_next_queued_run()
        if started:
            st.success(str(st.session_state.get("last_scheduler_message") or "已启动队列中的首个任务。"))
        else:
            st.info("当前没有可启动的排队任务。")

    if stop_active_clicked:
        message = _stop_active_run()
        st.warning(message)

    if clear_queue_clicked:
        st.session_state["run_queue"] = []
        st.session_state["queue_auto_run"] = False
        st.session_state["last_scheduler_message"] = "已清空当前队列。"
        st.info(st.session_state["last_scheduler_message"])

    if load_history_clicked:
        selected_history_label = str(st.session_state.get("selected_history_label") or "").strip()
        run_root = history_label_to_path.get(selected_history_label)
        if run_root is None:
            st.warning("请先选择一条历史记录。")
        else:
            _load_history_run(run_root)
            st.success(f"已载入历史结果: {run_root.name}")
            st.rerun()

    if requeue_history_clicked:
        selected_history_label = str(st.session_state.get("selected_history_label") or "").strip()
        run_root = history_label_to_path.get(selected_history_label)
        if run_root is None:
            st.warning("请先选择一条历史记录。")
        else:
            try:
                message = _enqueue_history_run_request(run_root)
                st.success(message)
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    if check_inputs_clicked:
        try:
            preflight_report = _execute_preflight_check(
                start_mode=str(st.session_state.get("start_mode") or "input_csv"),
                input_csv_upload=input_csv_upload,
                feature_csv_upload=feature_csv_upload,
                default_pocket_upload=default_pocket_upload,
                default_catalytic_upload=default_catalytic_upload,
                default_ligand_upload=default_ligand_upload,
                experiment_plan_override_upload=experiment_plan_override_upload,
                form_payload=_build_form_payload(),
            )
            if str(preflight_report.get("status") or "") == "ready":
                st.success("输入检查完成，当前配置可以开始运行。")
            elif str(preflight_report.get("status") or "") == "warning":
                st.warning("输入检查完成，存在提醒项，请查看下方“运行前检查”。")
            else:
                st.error("输入检查完成，发现阻塞问题，请先修正。")
        except Exception as exc:
            st.session_state["last_preflight"] = {
                "checked_at": _now_text(),
                "status": "error",
                "messages": [str(exc)],
            }
            st.error(str(exc))

    if run_clicked:
        try:
            preflight_report = _execute_preflight_check(
                start_mode=str(st.session_state.get("start_mode") or "input_csv"),
                input_csv_upload=input_csv_upload,
                feature_csv_upload=feature_csv_upload,
                default_pocket_upload=default_pocket_upload,
                default_catalytic_upload=default_catalytic_upload,
                default_ligand_upload=default_ligand_upload,
                experiment_plan_override_upload=experiment_plan_override_upload,
                form_payload=_build_form_payload(),
            )
            if str(preflight_report.get("status") or "") == "error":
                raise ValueError("运行前检查发现阻塞问题，请先修正后再启动运行。")
            with st.spinner("正在启动后台运行任务，请等待..."):
                run_request = _prepare_run_request_from_form(
                    input_csv_upload=input_csv_upload,
                    feature_csv_upload=feature_csv_upload,
                    default_pocket_upload=default_pocket_upload,
                    default_catalytic_upload=default_catalytic_upload,
                    default_ligand_upload=default_ligand_upload,
                    experiment_plan_override_upload=experiment_plan_override_upload,
                )
                st.session_state["queue_auto_run"] = False
                _start_background_run(run_request, queue_source="direct_run")
            st.session_state["last_preflight"] = None
            st.success("任务已启动。可在下方“运行调度”查看状态，运行结束后结果面板会自动接管最近一次完成任务。")
        except Exception as exc:
            st.error(str(exc))

    _render_input_status_panel(
        start_mode=str(st.session_state.get("start_mode") or "input_csv"),
        input_csv_upload=input_csv_upload,
        feature_csv_upload=feature_csv_upload,
        default_pocket_upload=default_pocket_upload,
        default_catalytic_upload=default_catalytic_upload,
        default_ligand_upload=default_ligand_upload,
        experiment_plan_override_upload=experiment_plan_override_upload,
        form_payload=_build_form_payload(),
    )
    _render_preflight_report(st.session_state.get("last_preflight"))
    _render_scheduler_panel()

    summary = st.session_state.get("last_summary")
    metadata = st.session_state.get("last_metadata")
    diagnostics = st.session_state.get("last_diagnostics")
    last_run_dir = st.session_state.get("last_run_dir")
    last_command = st.session_state.get("last_command")
    last_error = str(st.session_state.get("last_error") or "")
    last_export_bundle = st.session_state.get("last_export_bundle")
    last_export_message = str(st.session_state.get("last_export_message") or "")
    last_export_html = st.session_state.get("last_export_html")
    last_export_html_message = str(st.session_state.get("last_export_html_message") or "")
    last_export_pdf = st.session_state.get("last_export_pdf")
    last_export_pdf_message = str(st.session_state.get("last_export_pdf_message") or "")
    output_dir = _get_out_dir(summary if isinstance(summary, dict) else None, metadata if isinstance(metadata, dict) else None)

    if last_run_dir:
        st.info(f"最近一次运行目录: {last_run_dir}")
    if isinstance(metadata, dict) and metadata.get("execution_mode"):
        st.caption(f"当前执行模式: {metadata['execution_mode']}；下方命令仅用于复现。")
    if last_command:
        st.code(" ".join(last_command), language="bash")

    action_col1, action_col2 = st.columns(2)
    if action_col1.button("打开最近运行目录", use_container_width=True, disabled=not bool(last_run_dir)):
        ok, message = _open_local_path(Path(str(last_run_dir)))
        if ok:
            st.success(message)
        else:
            st.error(message)
    if action_col2.button("打开输出目录", use_container_width=True, disabled=output_dir is None):
        if output_dir is None:
            st.warning("当前没有可用输出目录。")
        else:
            ok, message = _open_local_path(output_dir)
            if ok:
                st.success(message)
            else:
                st.error(message)

    tab_summary, tab_qc, tab_compare, tab_history, tab_ranking, tab_pose, tab_report, tab_ai, tab_logs, tab_diag = st.tabs(
        ["摘要", "QC/Warning", "运行对比", "历史", "排名结果", "Pose 结果", "执行报告", "AI 解释", "日志", "诊断"]
    )

    with tab_summary:
        _render_summary_metrics(summary)
        notes = summary.get("notes") if isinstance(summary, dict) and isinstance(summary.get("notes"), list) else []
        if notes:
            st.subheader("执行备注")
            for note in notes:
                st.write(f"- {note}")

        artifacts = summary.get("artifacts") if isinstance(summary, dict) and isinstance(summary.get("artifacts"), dict) else {}
        demo_overview_html = (
            Path(str(artifacts.get("demo_overview_html")))
            if artifacts.get("demo_overview_html")
            else _resolve_output_file(
                "DEMO_OVERVIEW.html",
                summary=summary if isinstance(summary, dict) else None,
                metadata=metadata if isinstance(metadata, dict) else None,
            )
        )
        demo_readme_md = (
            Path(str(artifacts.get("demo_readme_md")))
            if artifacts.get("demo_readme_md")
            else _resolve_output_file(
                "DEMO_README.md",
                summary=summary if isinstance(summary, dict) else None,
                metadata=metadata if isinstance(metadata, dict) else None,
            )
        )
        demo_interpretation_md = (
            Path(str(artifacts.get("demo_interpretation_md")))
            if artifacts.get("demo_interpretation_md")
            else _resolve_output_file(
                "DEMO_INTERPRETATION.md",
                summary=summary if isinstance(summary, dict) else None,
                metadata=metadata if isinstance(metadata, dict) else None,
            )
        )
        real_data_starter_readme = (
            Path(str(artifacts.get("real_data_starter_readme_md")))
            if artifacts.get("real_data_starter_readme_md")
            else _resolve_output_file(
                "REAL_DATA_STARTER",
                "README_REAL_DATA_STARTER.md",
                summary=summary if isinstance(summary, dict) else None,
                metadata=metadata if isinstance(metadata, dict) else None,
            )
        )
        mini_pdb_example_readme = (
            Path(str(artifacts.get("mini_pdb_example_readme_md")))
            if artifacts.get("mini_pdb_example_readme_md")
            else _resolve_output_file(
                "REAL_DATA_STARTER",
                "MINI_PDB_EXAMPLE",
                "README_MINI_PDB_EXAMPLE.md",
                summary=summary if isinstance(summary, dict) else None,
                metadata=metadata if isinstance(metadata, dict) else None,
            )
        )
        mini_pdb_example_input_csv = (
            Path(str(artifacts.get("mini_pdb_example_input_csv")))
            if artifacts.get("mini_pdb_example_input_csv")
            else _resolve_output_file(
                "REAL_DATA_STARTER",
                "MINI_PDB_EXAMPLE",
                "input_pose_table.csv",
                summary=summary if isinstance(summary, dict) else None,
                metadata=metadata if isinstance(metadata, dict) else None,
            )
        )
        real_data_starter_dir = (
            real_data_starter_readme.parent
            if real_data_starter_readme is not None and real_data_starter_readme.exists()
            else None
        )
        mini_pdb_example_dir = (
            mini_pdb_example_readme.parent
            if mini_pdb_example_readme is not None and mini_pdb_example_readme.exists()
            else None
        )
        has_demo_outputs = any(
            path is not None and path.exists()
            for path in [
                demo_overview_html,
                demo_readme_md,
                demo_interpretation_md,
                real_data_starter_readme,
                mini_pdb_example_readme,
            ]
        )
        if has_demo_outputs:
            st.subheader("Demo 快速导览")
            st.caption(
                "当前运行包含 synthetic demo 说明文件。优先打开 HTML 导览页；"
                "demo 结果只用于流程演示，不代表真实湿实验验证。"
            )
            demo_col1, demo_col2, demo_col3, demo_col4 = st.columns(4)
            if demo_overview_html is not None and demo_overview_html.exists():
                if demo_col1.button("打开 Demo HTML 导览", use_container_width=True, key="open_demo_overview_html"):
                    ok, message = _open_local_path(demo_overview_html)
                    if ok:
                        st.success(message)
                    else:
                        st.error(message)
                demo_col2.download_button(
                    label="下载 Demo HTML",
                    data=demo_overview_html.read_bytes(),
                    file_name=demo_overview_html.name,
                    mime="text/html",
                    use_container_width=True,
                    key="download_demo_overview_html",
                )
            else:
                demo_col1.button(
                    "打开 Demo HTML 导览",
                    use_container_width=True,
                    disabled=True,
                    key="open_demo_overview_html_disabled",
                )
                demo_col2.button(
                    "下载 Demo HTML",
                    use_container_width=True,
                    disabled=True,
                    key="download_demo_overview_html_disabled",
                )

            if demo_interpretation_md is not None and demo_interpretation_md.exists():
                demo_col3.download_button(
                    label="下载 Demo 解读",
                    data=demo_interpretation_md.read_bytes(),
                    file_name=demo_interpretation_md.name,
                    mime="text/markdown",
                    use_container_width=True,
                    key="download_demo_interpretation_md",
                )
                with st.expander("查看 Demo 结果解读"):
                    st.markdown(_read_text(demo_interpretation_md))
            else:
                demo_col3.button(
                    "下载 Demo 解读",
                    use_container_width=True,
                    disabled=True,
                    key="download_demo_interpretation_md_disabled",
                )

            if demo_readme_md is not None and demo_readme_md.exists():
                demo_col4.download_button(
                    label="下载 Demo README",
                    data=demo_readme_md.read_bytes(),
                    file_name=demo_readme_md.name,
                    mime="text/markdown",
                    use_container_width=True,
                    key="download_demo_readme_md",
                )
            else:
                demo_col4.button(
                    "下载 Demo README",
                    use_container_width=True,
                    disabled=True,
                    key="download_demo_readme_md_disabled",
                )

            if real_data_starter_readme is not None and real_data_starter_readme.exists():
                st.caption("想把 demo 换成自己的数据：先使用 `REAL_DATA_STARTER` 里的模板和检查清单。")
                starter_col1, starter_col2, starter_col3 = st.columns(3)
                starter_col1.download_button(
                    label="下载真实数据 starter README",
                    data=real_data_starter_readme.read_bytes(),
                    file_name=real_data_starter_readme.name,
                    mime="text/markdown",
                    use_container_width=True,
                    key="download_real_data_starter_readme_md",
                )
                if starter_col2.button(
                    "打开真实数据 starter 文件夹",
                    use_container_width=True,
                    key="open_real_data_starter_dir",
                ):
                    ok, message = _open_local_path(real_data_starter_dir or real_data_starter_readme.parent)
                    if ok:
                        st.success(message)
                    else:
                        st.error(message)
                if mini_pdb_example_readme is not None and mini_pdb_example_readme.exists():
                    if starter_col3.button(
                        "打开 mini PDB 示例",
                        use_container_width=True,
                        key="open_mini_pdb_example_dir",
                    ):
                        ok, message = _open_local_path(mini_pdb_example_dir or mini_pdb_example_readme.parent)
                        if ok:
                            st.success(message)
                        else:
                            st.error(message)
                    with st.expander("查看 mini PDB 示例说明"):
                        st.markdown(_read_text(mini_pdb_example_readme))
                        if mini_pdb_example_input_csv is not None and mini_pdb_example_input_csv.exists():
                            st.download_button(
                                label="下载 mini input_pose_table.csv",
                                data=mini_pdb_example_input_csv.read_bytes(),
                                file_name=mini_pdb_example_input_csv.name,
                                mime="text/csv",
                                use_container_width=True,
                                key="download_mini_pdb_example_input_csv",
                            )
                else:
                    starter_col3.button(
                        "打开 mini PDB 示例",
                        use_container_width=True,
                        disabled=True,
                        key="open_mini_pdb_example_dir_disabled",
                    )

        batch_summary_md = (
            Path(str(artifacts.get("batch_decision_summary_md")))
            if artifacts.get("batch_decision_summary_md")
            else _resolve_output_file(
                "batch_decision_summary",
                "batch_decision_summary.md",
                summary=summary if isinstance(summary, dict) else None,
                metadata=metadata if isinstance(metadata, dict) else None,
            )
        )
        batch_summary_json = (
            Path(str(artifacts.get("batch_decision_summary_json")))
            if artifacts.get("batch_decision_summary_json")
            else _resolve_output_file(
                "batch_decision_summary",
                "batch_decision_summary.json",
                summary=summary if isinstance(summary, dict) else None,
                metadata=metadata if isinstance(metadata, dict) else None,
            )
        )
        batch_cards_csv = (
            Path(str(artifacts.get("batch_decision_summary_cards_csv")))
            if artifacts.get("batch_decision_summary_cards_csv")
            else _resolve_output_file(
                "batch_decision_summary",
                "batch_decision_summary_cards.csv",
                summary=summary if isinstance(summary, dict) else None,
                metadata=metadata if isinstance(metadata, dict) else None,
            )
        )
        if (
            (batch_summary_md is not None and batch_summary_md.exists())
            or (batch_summary_json is not None and batch_summary_json.exists())
            or (batch_cards_csv is not None and batch_cards_csv.exists())
        ):
            st.subheader("本批次结论摘要")
            st.caption(
                "把 Quality Gate、共识排名、分数解释卡片、实验计划和验证证据汇总成一页："
                "能不能解读、优先看谁、先修什么风险。"
            )
            batch_payload = _load_json(batch_summary_json) if batch_summary_json is not None and batch_summary_json.exists() else None
            if isinstance(batch_payload, dict):
                batch_decision = batch_payload.get("batch_decision") if isinstance(batch_payload.get("batch_decision"), dict) else {}
                quality_gate = batch_payload.get("quality_gate") if isinstance(batch_payload.get("quality_gate"), dict) else {}
                validation_evidence = (
                    batch_payload.get("validation_evidence")
                    if isinstance(batch_payload.get("validation_evidence"), dict)
                    else {}
                )
                highlights = batch_payload.get("candidate_highlights") if isinstance(batch_payload.get("candidate_highlights"), dict) else {}
                best_candidate = highlights.get("best_candidate") if isinstance(highlights.get("best_candidate"), dict) else {}
                next_candidate = (
                    highlights.get("next_experiment_candidate")
                    if isinstance(highlights.get("next_experiment_candidate"), dict)
                    else {}
                )
                batch_col1, batch_col2, batch_col3, batch_col4 = st.columns(4)
                batch_col1.metric("Quality Gate", str(quality_gate.get("overall_status") or "UNKNOWN"))
                batch_col2.metric("最高排名候选", str(best_candidate.get("nanobody_id") or "N/A"))
                batch_col3.metric("下一轮优先候选", str(next_candidate.get("nanobody_id") or "N/A"))
                validation_status = str(validation_evidence.get("audit_status") or batch_decision.get("validation_evidence_status") or "UNKNOWN")
                validation_coverage = _ratio_text(validation_evidence.get("top_k_validation_coverage"))
                batch_col4.metric("验证证据", validation_status, validation_coverage)
                if batch_decision.get("summary"):
                    st.info(str(batch_decision.get("summary")))
                if batch_decision.get("recommended_next_action"):
                    st.write(str(batch_decision.get("recommended_next_action")))
                if validation_evidence.get("summary"):
                    st.caption(f"验证证据：{validation_evidence.get('summary')}")
            download_cols = st.columns(3)
            if batch_summary_md is not None and batch_summary_md.exists():
                download_cols[0].download_button(
                    label="下载批次摘要 Markdown",
                    data=batch_summary_md.read_bytes(),
                    file_name=batch_summary_md.name,
                    mime="text/markdown",
                    use_container_width=True,
                    key="download_batch_decision_summary_md",
                )
                with st.expander("查看本批次结论摘要"):
                    st.markdown(_read_text(batch_summary_md))
            if batch_summary_json is not None and batch_summary_json.exists():
                download_cols[1].download_button(
                    label="下载批次摘要 JSON",
                    data=batch_summary_json.read_bytes(),
                    file_name=batch_summary_json.name,
                    mime="application/json",
                    use_container_width=True,
                    key="download_batch_decision_summary_json",
                )
            if batch_cards_csv is not None and batch_cards_csv.exists():
                download_cols[2].download_button(
                    label="下载候选结论卡 CSV",
                    data=batch_cards_csv.read_bytes(),
                    file_name=batch_cards_csv.name,
                    mime="text/csv",
                    use_container_width=True,
                    key="download_batch_decision_summary_cards_csv",
                )
                batch_cards_df = _load_csv(batch_cards_csv)
                if batch_cards_df is not None and not batch_cards_df.empty:
                    st.dataframe(batch_cards_df, use_container_width=True, hide_index=True)

        st.subheader("训练摘要")
        _render_training_summary_panel(
            summary if isinstance(summary, dict) else None,
            metadata if isinstance(metadata, dict) else None,
        )

        qc_payload = _load_feature_qc_payload(
            summary if isinstance(summary, dict) else None,
            metadata if isinstance(metadata, dict) else None,
        )
        processing_summary = qc_payload.get("processing_summary") if isinstance(qc_payload, dict) and isinstance(qc_payload.get("processing_summary"), dict) else {}
        if processing_summary:
            failed_rows = int(processing_summary.get("failed_rows", 0))
            warning_rows = int(processing_summary.get("rows_with_warning_message", 0))
            if failed_rows > 0 or warning_rows > 0:
                st.warning(
                    f"当前 feature 构建摘要: failed_rows={failed_rows}, "
                    f"rows_with_warning_message={warning_rows}。"
                    " 详细内容可到“QC/Warning”页查看。"
                )
            else:
                st.success("当前 feature 构建未发现 failed 行或 warning 行。")

        st.subheader("运行产物清单")
        inventory_df = _build_output_inventory(
            summary if isinstance(summary, dict) else None,
            metadata if isinstance(metadata, dict) else None,
        )
        if inventory_df.empty:
            st.info("当前还没有可展示的运行产物。")
        else:
            st.dataframe(inventory_df, use_container_width=True, hide_index=True)

        st.subheader("结果汇总打包")
        bundle_col1, bundle_col2, bundle_col3, bundle_col4 = st.columns(4)
        create_bundle_clicked = bundle_col1.button(
            "生成当前运行汇总包",
            use_container_width=True,
            disabled=not bool(last_run_dir),
        )
        create_html_clicked = bundle_col2.button(
            "生成展示摘要 HTML",
            use_container_width=True,
            disabled=not bool(last_run_dir),
        )
        create_pdf_clicked = bundle_col3.button(
            "生成展示摘要 PDF",
            use_container_width=True,
            disabled=not bool(last_run_dir),
        )
        open_bundle_dir_clicked = bundle_col4.button(
            "打开汇总包目录",
            use_container_width=True,
            disabled=not bool(last_run_dir),
        )

        if create_bundle_clicked:
            if not last_run_dir:
                st.warning("当前还没有可打包的运行记录。")
            else:
                bundle_run_root = Path(str(last_run_dir))
                bundle_summary = summary if isinstance(summary, dict) else None
                bundle_metadata = metadata if isinstance(metadata, dict) else None
                with st.spinner("正在生成当前运行的汇总包..."):
                    bundle_path = _create_export_bundle(bundle_run_root, bundle_summary, bundle_metadata)
                st.session_state["last_export_bundle"] = str(bundle_path)
                st.session_state["last_export_message"] = f"已生成汇总包: {bundle_path}"
                last_export_bundle = str(bundle_path)
                last_export_message = str(st.session_state["last_export_message"])

        if create_html_clicked:
            if not last_run_dir:
                st.warning("当前还没有可导出的运行记录。")
            else:
                html_run_root = Path(str(last_run_dir))
                html_summary = summary if isinstance(summary, dict) else None
                html_metadata = metadata if isinstance(metadata, dict) else None
                html_diagnostics = diagnostics if isinstance(diagnostics, dict) else None
                with st.spinner("正在生成展示摘要 HTML..."):
                    html_path = _create_html_summary_export(
                        html_run_root,
                        html_summary,
                        html_metadata,
                        html_diagnostics,
                    )
                st.session_state["last_export_html"] = str(html_path)
                st.session_state["last_export_html_message"] = f"已生成展示摘要 HTML: {html_path}"
                last_export_html = str(html_path)
                last_export_html_message = str(st.session_state["last_export_html_message"])

        if create_pdf_clicked:
            if not last_run_dir:
                st.warning("当前还没有可导出的运行记录。")
            else:
                pdf_run_root = Path(str(last_run_dir))
                pdf_summary = summary if isinstance(summary, dict) else None
                pdf_metadata = metadata if isinstance(metadata, dict) else None
                pdf_diagnostics = diagnostics if isinstance(diagnostics, dict) else None
                with st.spinner("正在生成展示摘要 PDF..."):
                    pdf_path = _create_pdf_summary_export(
                        pdf_run_root,
                        pdf_summary,
                        pdf_metadata,
                        pdf_diagnostics,
                    )
                st.session_state["last_export_pdf"] = str(pdf_path)
                st.session_state["last_export_pdf_message"] = f"已生成展示摘要 PDF: {pdf_path}"
                last_export_pdf = str(pdf_path)
                last_export_pdf_message = str(st.session_state["last_export_pdf_message"])

        if open_bundle_dir_clicked:
            if not last_run_dir:
                st.warning("当前还没有可用运行目录。")
            else:
                bundle_dir = Path(str(last_run_dir)) / "exports"
                bundle_dir.mkdir(parents=True, exist_ok=True)
                ok, message = _open_local_path(bundle_dir)
                if ok:
                    st.success(message)
                else:
                    st.error(message)

        if last_export_message:
            st.success(last_export_message)
        if last_export_bundle and Path(str(last_export_bundle)).exists():
            bundle_path = Path(str(last_export_bundle))
            st.caption(f"当前汇总包: {bundle_path}")
            st.download_button(
                label="下载当前运行汇总包 zip",
                data=bundle_path.read_bytes(),
                file_name=bundle_path.name,
                mime="application/zip",
                use_container_width=True,
            )

        if last_export_html_message:
            st.success(last_export_html_message)
        if last_export_html and Path(str(last_export_html)).exists():
            html_path = Path(str(last_export_html))
            st.caption(f"当前展示摘要 HTML: {html_path}")
            html_download_col1, html_download_col2 = st.columns(2)
            html_download_col1.download_button(
                label="下载展示摘要 HTML",
                data=html_path.read_bytes(),
                file_name=html_path.name,
                mime="text/html",
                use_container_width=True,
            )
            if html_download_col2.button("打开展示摘要 HTML", use_container_width=True):
                ok, message = _open_local_path(html_path)
                if ok:
                    st.success(message)
                else:
                    st.error(message)
        if last_export_pdf_message:
            st.success(last_export_pdf_message)
        if last_export_pdf and Path(str(last_export_pdf)).exists():
            pdf_path = Path(str(last_export_pdf))
            st.caption(f"当前展示摘要 PDF: {pdf_path}")
            pdf_download_col1, pdf_download_col2 = st.columns(2)
            pdf_download_col1.download_button(
                label="下载展示摘要 PDF",
                data=pdf_path.read_bytes(),
                file_name=pdf_path.name,
                mime="application/pdf",
                use_container_width=True,
            )
            if pdf_download_col2.button("打开展示摘要 PDF", use_container_width=True):
                ok, message = _open_local_path(pdf_path)
                if ok:
                    st.success(message)
                else:
                    st.error(message)

        st.subheader("关键产物下载")
        _render_download_buttons(
            summary if isinstance(summary, dict) else None,
            metadata if isinstance(metadata, dict) else None,
        )

        if isinstance(summary, dict):
            with st.expander("查看完整执行摘要 JSON"):
                st.json(summary)

    with tab_qc:
        _render_feature_qc_panel(
            summary if isinstance(summary, dict) else None,
            metadata if isinstance(metadata, dict) else None,
        )

    with tab_compare:
        _render_history_compare_panel(history_records)

    with tab_history:
        if not history_records:
            st.info("当前还没有历史运行记录。")
        else:
            st.subheader("最近运行记录")
            filtered_history_records = _render_history_filter_controls(
                history_records,
                key_prefix="history_tab",
                title="按关键词 / 状态 / 启动方式筛选历史记录",
            )
            if not filtered_history_records:
                st.info("当前筛选条件下没有历史记录。")
            else:
                filtered_history_df = _build_history_table(filtered_history_records)
                history_metric_col1, history_metric_col2, history_metric_col3 = st.columns(3)
                history_metric_col1.metric("Visible Runs", len(filtered_history_df))
                history_metric_col2.metric(
                    "Success Runs",
                    int(filtered_history_df["status"].astype(str).str.lower().eq("success").sum()),
                )
                history_metric_col3.metric(
                    "Failed / Cancelled",
                    int(
                        filtered_history_df["status"].astype(str).str.lower().isin(["failed", "cancelled"]).sum()
                    ),
                )
                st.dataframe(filtered_history_df, use_container_width=True, hide_index=True)
                st.download_button(
                    label="下载当前筛选后的历史记录 CSV",
                    data=filtered_history_df.to_csv(index=False).encode("utf-8"),
                    file_name="history_runs_filtered.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_filtered_history_csv",
                )
                with st.expander("跨批次实验状态汇总"):
                    _render_global_experiment_ledger_panel()
                with st.expander("结果自动归档与长期趋势"):
                    _render_result_archive_panel()
            st.caption("可在左侧“运行历史”区域加载某次历史结果，并恢复对应表单参数；需要多次运行对比时，直接切到“运行对比”页。")

            with st.expander("运行产物清理工具"):
                cleanup_targets = _build_cleanup_targets(history_records)
                if not cleanup_targets:
                    st.info("当前没有可清理的应用产物目录。")
                else:
                    cleanup_df = _build_cleanup_target_table(cleanup_targets)
                    cleanup_metric_col1, cleanup_metric_col2 = st.columns(2)
                    cleanup_metric_col1.metric("可清理目录数", len(cleanup_targets))
                    cleanup_metric_col2.metric(
                        "可回收空间",
                        _format_byte_count(int(sum(int(item.get("size_bytes") or 0) for item in cleanup_targets))),
                    )
                    st.dataframe(cleanup_df, use_container_width=True, hide_index=True)

                    cleanup_labels = [str(item.get("label") or "") for item in cleanup_targets]
                    selected_cleanup_label = st.selectbox(
                        "选择要清理的目录",
                        options=cleanup_labels,
                        key="selected_cleanup_target_label",
                    )
                    selected_cleanup_target = next(
                        (item for item in cleanup_targets if str(item.get("label") or "") == selected_cleanup_label),
                        None,
                    )
                    confirm_text = str(
                        st.text_input(
                            "输入 DELETE 确认删除",
                            key="cleanup_confirm_text",
                            placeholder="DELETE",
                        )
                        or ""
                    ).strip()
                    active_run_info = st.session_state.get("active_run_info")
                    active_run_root = (
                        Path(str(active_run_info.get("run_root"))).resolve()
                        if isinstance(active_run_info, dict) and active_run_info.get("run_root")
                        else None
                    )
                    selected_cleanup_path = (
                        Path(str(selected_cleanup_target.get("path") or "")).resolve()
                        if isinstance(selected_cleanup_target, dict) and selected_cleanup_target.get("path")
                        else None
                    )
                    cleanup_target_is_active = bool(
                        active_run_root is not None
                        and selected_cleanup_path is not None
                        and selected_cleanup_path == active_run_root
                    )
                    if cleanup_target_is_active:
                        st.warning("当前选中的目录对应正在运行的任务，不能删除。")

                    cleanup_action_col1, cleanup_action_col2 = st.columns(2)
                    open_cleanup_target_clicked = cleanup_action_col1.button(
                        "打开所选目录",
                        use_container_width=True,
                        disabled=selected_cleanup_path is None or not selected_cleanup_path.exists(),
                        key="open_cleanup_target_clicked",
                    )
                    delete_cleanup_target_clicked = cleanup_action_col2.button(
                        "删除所选目录",
                        use_container_width=True,
                        disabled=(
                            selected_cleanup_target is None
                            or confirm_text != "DELETE"
                            or cleanup_target_is_active
                        ),
                        key="delete_cleanup_target_clicked",
                    )

                    if open_cleanup_target_clicked and selected_cleanup_path is not None:
                        ok, message = _open_local_path(selected_cleanup_path)
                        if ok:
                            st.success(message)
                        else:
                            st.error(message)

                    if delete_cleanup_target_clicked and isinstance(selected_cleanup_target, dict):
                        try:
                            message = _delete_cleanup_target(selected_cleanup_target)
                            st.success(message)
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))

    with tab_ranking:
        if not isinstance(summary, dict):
            st.info("先运行一次 pipeline，再查看排名结果。")
        else:
            artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
            ml_ranking_csv = Path(str(artifacts.get("ml_ranking_csv"))) if artifacts.get("ml_ranking_csv") else None
            rule_ranking_csv = Path(str(artifacts.get("rule_ranking_csv"))) if artifacts.get("rule_ranking_csv") else None
            consensus_ranking_csv = Path(str(artifacts.get("consensus_ranking_csv"))) if artifacts.get("consensus_ranking_csv") else None
            score_explanation_cards_csv = (
                Path(str(artifacts.get("score_explanation_cards_csv")))
                if artifacts.get("score_explanation_cards_csv")
                else None
            )
            score_explanation_cards_html = (
                Path(str(artifacts.get("score_explanation_cards_html")))
                if artifacts.get("score_explanation_cards_html")
                else None
            )
            score_explanation_cards_md = (
                Path(str(artifacts.get("score_explanation_cards_md")))
                if artifacts.get("score_explanation_cards_md")
                else None
            )
            parameter_sensitivity_candidate_csv = (
                Path(str(artifacts.get("parameter_sensitivity_candidate_csv")))
                if artifacts.get("parameter_sensitivity_candidate_csv")
                else None
            )
            parameter_sensitivity_report_md = (
                Path(str(artifacts.get("parameter_sensitivity_report_md")))
                if artifacts.get("parameter_sensitivity_report_md")
                else None
            )
            candidate_report_index_html = Path(str(artifacts.get("candidate_report_index_html"))) if artifacts.get("candidate_report_index_html") else None
            candidate_report_zip = Path(str(artifacts.get("candidate_report_zip"))) if artifacts.get("candidate_report_zip") else None
            candidate_pairwise_comparisons_csv = (
                Path(str(artifacts.get("candidate_pairwise_comparisons_csv")))
                if artifacts.get("candidate_pairwise_comparisons_csv")
                else None
            )
            candidate_group_comparison_summary_csv = (
                Path(str(artifacts.get("candidate_group_comparison_summary_csv")))
                if artifacts.get("candidate_group_comparison_summary_csv")
                else None
            )
            candidate_comparison_report_md = (
                Path(str(artifacts.get("candidate_comparison_report_md")))
                if artifacts.get("candidate_comparison_report_md")
                else None
            )
            experiment_suggestions_csv = Path(str(artifacts.get("experiment_suggestions_csv"))) if artifacts.get("experiment_suggestions_csv") else None
            experiment_plan_csv = Path(str(artifacts.get("experiment_plan_csv"))) if artifacts.get("experiment_plan_csv") else None
            experiment_plan_md = Path(str(artifacts.get("experiment_plan_md"))) if artifacts.get("experiment_plan_md") else None
            experiment_plan_state_ledger_csv = (
                Path(str(artifacts.get("experiment_plan_state_ledger_csv")))
                if artifacts.get("experiment_plan_state_ledger_csv")
                else None
            )
            validation_evidence_summary_json = (
                Path(str(artifacts.get("validation_evidence_summary_json")))
                if artifacts.get("validation_evidence_summary_json")
                else None
            )
            validation_evidence_report_md = (
                Path(str(artifacts.get("validation_evidence_report_md")))
                if artifacts.get("validation_evidence_report_md")
                else None
            )
            validation_evidence_topk_csv = (
                Path(str(artifacts.get("validation_evidence_topk_csv")))
                if artifacts.get("validation_evidence_topk_csv")
                else None
            )
            validation_evidence_action_items_csv = (
                Path(str(artifacts.get("validation_evidence_action_items_csv")))
                if artifacts.get("validation_evidence_action_items_csv")
                else None
            )
            ranking_preview_rows = st.number_input(
                "每个排名表预览前 N 行",
                min_value=5,
                max_value=500,
                value=20,
                step=5,
                key="ranking_preview_rows",
            )
            ranking_filter_text = str(st.text_input("按 nanobody_id 过滤", key="ranking_filter_text") or "").strip()

            if (
                (score_explanation_cards_csv is not None and score_explanation_cards_csv.exists())
                or (score_explanation_cards_html is not None and score_explanation_cards_html.exists())
                or (score_explanation_cards_md is not None and score_explanation_cards_md.exists())
            ):
                st.subheader("分数解释卡片")
                st.caption(
                    "把 consensus_score、confidence、Rule/ML 一致性、QC 风险和 label 覆盖翻译成可读结论；"
                    "只解释已有结果，不改动排名分数。"
                )
                if score_explanation_cards_csv is not None and score_explanation_cards_csv.exists():
                    score_card_df = _load_csv(score_explanation_cards_csv)
                    if score_card_df is not None:
                        if ranking_filter_text and "nanobody_id" in score_card_df.columns:
                            score_card_df = score_card_df[
                                score_card_df["nanobody_id"].astype(str).str.contains(
                                    ranking_filter_text,
                                    case=False,
                                    regex=False,
                                    na=False,
                                )
                            ]
                        score_card_view_df, score_card_export_df = _render_dataframe_view_controls(
                            score_card_df,
                            key_prefix="score_explanation_cards_view",
                            preferred_columns=[
                                "nanobody_id",
                                "consensus_rank",
                                "score_band",
                                "confidence_level",
                                "decision_tier",
                                "score_meaning",
                                "main_positive_factors",
                                "main_risk_factors",
                                "recommended_action",
                                "label_status",
                            ],
                            default_sort_column="consensus_rank",
                            default_descending=False,
                        )
                        st.dataframe(score_card_view_df.head(int(ranking_preview_rows)), use_container_width=True)
                        score_card_col1, score_card_col2 = st.columns(2)
                        score_card_col1.download_button(
                            label="下载分数解释卡片 CSV",
                            data=score_card_export_df.to_csv(index=False).encode("utf-8"),
                            file_name="score_explanation_cards_filtered.csv",
                            mime="text/csv",
                            use_container_width=True,
                            key="download_score_explanation_cards_csv",
                        )
                        if score_explanation_cards_html is not None and score_explanation_cards_html.exists():
                            score_card_col2.download_button(
                                label="下载分数解释卡片 HTML",
                                data=score_explanation_cards_html.read_bytes(),
                                file_name=score_explanation_cards_html.name,
                                mime="text/html",
                                use_container_width=True,
                                key="download_score_explanation_cards_html",
                            )
                if score_explanation_cards_html is not None and score_explanation_cards_html.exists():
                    with st.expander("预览分数解释卡片 HTML"):
                        st.components.v1.html(_read_text(score_explanation_cards_html), height=620, scrolling=True)
                if score_explanation_cards_md is not None and score_explanation_cards_md.exists():
                    with st.expander("查看分数解释卡片 Markdown"):
                        st.markdown(_read_text(score_explanation_cards_md))

            if (
                (candidate_report_index_html is not None and candidate_report_index_html.exists())
                or (candidate_report_zip is not None and candidate_report_zip.exists())
            ):
                st.subheader("候选报告卡")
                st.caption("报告卡从共识排名、Rule/ML 排名、pose/QC 和候选对比信息汇总生成，可直接打开 HTML 索引或下载 zip 发送。")
                report_card_col1, report_card_col2 = st.columns(2)
                if candidate_report_index_html is not None and candidate_report_index_html.exists():
                    if report_card_col1.button("打开候选报告卡索引", use_container_width=True, key="open_candidate_report_index"):
                        ok, message = _open_local_path(candidate_report_index_html)
                        if ok:
                            st.success(message)
                        else:
                            st.error(message)
                    report_card_col1.download_button(
                        label="下载报告卡索引 HTML",
                        data=candidate_report_index_html.read_bytes(),
                        file_name=candidate_report_index_html.name,
                        mime="text/html",
                        use_container_width=True,
                        key="download_candidate_report_index_html",
                    )
                if candidate_report_zip is not None and candidate_report_zip.exists():
                    report_card_col2.download_button(
                        label="下载全部候选报告卡 zip",
                        data=candidate_report_zip.read_bytes(),
                        file_name=candidate_report_zip.name,
                        mime="application/zip",
                        use_container_width=True,
                        key="download_candidate_report_zip",
                    )

            if candidate_pairwise_comparisons_csv is not None and candidate_pairwise_comparisons_csv.exists():
                st.subheader("候选对比解释")
                st.caption("该表解释为什么高排名候选排在相邻候选之前，并列出对方候选的反向优势；不重新训练、不改变排名。")
                comparison_df = _load_csv(candidate_pairwise_comparisons_csv)
                if comparison_df is not None:
                    if ranking_filter_text:
                        mask = pd.Series(False, index=comparison_df.index)
                        for col in ["winner_nanobody_id", "runner_up_nanobody_id"]:
                            if col in comparison_df.columns:
                                mask = mask | comparison_df[col].astype(str).str.contains(
                                    ranking_filter_text,
                                    case=False,
                                    regex=False,
                                    na=False,
                                )
                        comparison_df = comparison_df[mask]
                    comparison_view_df, comparison_export_df = _render_dataframe_view_controls(
                        comparison_df,
                        key_prefix="candidate_pairwise_comparison_view",
                        preferred_columns=[
                            "winner_nanobody_id",
                            "runner_up_nanobody_id",
                            "consensus_score_gap",
                            "is_close_decision",
                            "winner_confidence_level",
                            "winner_key_advantages",
                            "runner_up_counterpoints",
                            "comparison_explanation",
                        ],
                        default_sort_column="consensus_score_gap",
                        default_descending=False,
                    )
                    st.dataframe(comparison_view_df.head(int(ranking_preview_rows)), use_container_width=True)
                    candidate_compare_col1, candidate_compare_col2 = st.columns(2)
                    candidate_compare_col1.download_button(
                        label="下载候选对比可见视图 CSV",
                        data=comparison_view_df.to_csv(index=False).encode("utf-8"),
                        file_name="candidate_pairwise_comparisons_filtered.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="download_candidate_pairwise_comparison_view_csv",
                    )
                    candidate_compare_col2.download_button(
                        label="下载候选对比全筛选结果 CSV",
                        data=comparison_export_df.to_csv(index=False).encode("utf-8"),
                        file_name="candidate_pairwise_comparisons_filtered_full.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="download_candidate_pairwise_comparison_full_csv",
                    )
                if candidate_comparison_report_md is not None and candidate_comparison_report_md.exists():
                    with st.expander("查看候选对比解释报告"):
                        st.markdown(_read_text(candidate_comparison_report_md))

            if candidate_group_comparison_summary_csv is not None and candidate_group_comparison_summary_csv.exists():
                with st.expander("候选分组对比小结"):
                    group_comparison_df = _load_csv(candidate_group_comparison_summary_csv)
                    if group_comparison_df is None or group_comparison_df.empty:
                        st.info("当前没有可用的候选分组对比小结。")
                    else:
                        preferred_group_columns = [
                            "group_column",
                            "group_value",
                            "candidate_count",
                            "best_nanobody_id",
                            "mean_consensus_score",
                            "low_confidence_count",
                            "close_decision_count",
                            "representative_strengths",
                            "representative_risks",
                            "group_summary_text",
                        ]
                        group_view_df, group_export_df = _render_dataframe_view_controls(
                            group_comparison_df,
                            key_prefix="candidate_group_comparison_view",
                            preferred_columns=preferred_group_columns,
                            default_sort_column="best_consensus_rank",
                            default_descending=False,
                        )
                        st.dataframe(group_view_df, use_container_width=True)
                        st.download_button(
                            label="下载候选分组对比小结 CSV",
                            data=group_export_df.to_csv(index=False).encode("utf-8"),
                            file_name="candidate_group_comparison_summary.csv",
                            mime="text/csv",
                            use_container_width=True,
                            key="download_candidate_group_comparison_summary_csv",
                        )

            if consensus_ranking_csv is not None and consensus_ranking_csv.exists():
                with st.expander("自定义候选对比（手工选择 2 到 5 个 nanobody）"):
                    custom_consensus_df = _load_csv(consensus_ranking_csv)
                    if custom_consensus_df is None or custom_consensus_df.empty or "nanobody_id" not in custom_consensus_df.columns:
                        st.info("当前 consensus_ranking.csv 不包含可用的 nanobody_id，无法生成自定义对比。")
                    else:
                        custom_work_df = custom_consensus_df.copy()
                        if "consensus_rank" in custom_work_df.columns:
                            custom_work_df["_rank_sort"] = pd.to_numeric(
                                custom_work_df["consensus_rank"],
                                errors="coerce",
                            )
                            custom_work_df = custom_work_df.sort_values(
                                "_rank_sort",
                                ascending=True,
                                na_position="last",
                            ).drop(columns=["_rank_sort"])
                        elif "consensus_score" in custom_work_df.columns:
                            custom_work_df["_score_sort"] = pd.to_numeric(
                                custom_work_df["consensus_score"],
                                errors="coerce",
                            )
                            custom_work_df = custom_work_df.sort_values(
                                "_score_sort",
                                ascending=False,
                                na_position="last",
                            ).drop(columns=["_score_sort"])
                        candidate_ids = custom_work_df["nanobody_id"].astype(str).drop_duplicates().tolist()
                        label_map: dict[str, str] = {}
                        for _, row in custom_work_df.drop_duplicates(subset=["nanobody_id"]).iterrows():
                            nanobody_id = str(row.get("nanobody_id") or "")
                            rank_text = str(row.get("consensus_rank") or "").strip()
                            score_text = _fmt_metric(pd.to_numeric(pd.Series([row.get("consensus_score")]), errors="coerce").iloc[0])
                            label_map[nanobody_id] = f"#{rank_text or '?'} | {nanobody_id} | score={score_text}"
                        selected_custom_ids = st.multiselect(
                            "选择候选",
                            options=candidate_ids,
                            default=candidate_ids[: min(2, len(candidate_ids))],
                            format_func=lambda item: label_map.get(str(item), str(item)),
                            key="custom_candidate_comparison_ids",
                        )
                        custom_pair_mode = st.selectbox(
                            "自定义对比方式",
                            options=["all", "adjacent", "top_vs_rest"],
                            index=0,
                            format_func=lambda item: {
                                "all": "两两全部对比",
                                "adjacent": "按排名相邻对比",
                                "top_vs_rest": "最高排名 vs 其余",
                            }.get(str(item), str(item)),
                            key="custom_candidate_comparison_pair_mode",
                        )
                        custom_param_col1, custom_param_col2 = st.columns(2)
                        custom_close_delta = custom_param_col1.number_input(
                            "close_score_delta",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.03,
                            step=0.01,
                            format="%.3f",
                            key="custom_candidate_close_score_delta",
                        )
                        custom_min_delta = custom_param_col2.number_input(
                            "min_metric_delta",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.02,
                            step=0.01,
                            format="%.3f",
                            key="custom_candidate_min_metric_delta",
                        )
                        if len(selected_custom_ids) < 2:
                            st.info("至少选择 2 个候选。")
                        elif len(selected_custom_ids) > 5:
                            st.warning("为保证解释可读性，自定义对比一次最多选择 5 个候选。")
                        else:
                            try:
                                custom_result = build_candidate_comparison_outputs(
                                    custom_work_df,
                                    top_n=len(selected_custom_ids),
                                    pair_mode=str(custom_pair_mode),
                                    selected_nanobody_ids=[str(item) for item in selected_custom_ids],
                                    close_score_delta=float(custom_close_delta),
                                    min_metric_delta=float(custom_min_delta),
                                    consensus_csv=consensus_ranking_csv,
                                )
                                custom_pairwise_df = custom_result["pairwise"]
                                custom_tradeoff_df = custom_result["tradeoffs"]
                                custom_group_summary_df = custom_result.get("group_summary", pd.DataFrame())
                                custom_summary = custom_result["summary"]
                                metric_col1, metric_col2, metric_col3 = st.columns(3)
                                metric_col1.metric("候选数", int(custom_summary.get("top_candidate_count") or 0))
                                metric_col2.metric("对比数", int(custom_summary.get("pairwise_comparison_count") or 0))
                                metric_col3.metric("Close decisions", int(custom_summary.get("close_decision_count") or 0))
                                custom_pairwise_view_df, custom_pairwise_export_df = _render_dataframe_view_controls(
                                    custom_pairwise_df,
                                    key_prefix="custom_candidate_pairwise_view",
                                    preferred_columns=[
                                        "winner_nanobody_id",
                                        "runner_up_nanobody_id",
                                        "consensus_score_gap",
                                        "is_close_decision",
                                        "winner_key_advantages",
                                        "runner_up_counterpoints",
                                        "comparison_explanation",
                                    ],
                                    default_sort_column="consensus_score_gap",
                                    default_descending=False,
                                )
                                st.dataframe(custom_pairwise_view_df, use_container_width=True)
                                with st.expander("查看自定义候选 trade-off 表"):
                                    st.dataframe(custom_tradeoff_df, use_container_width=True)
                                if isinstance(custom_group_summary_df, pd.DataFrame) and not custom_group_summary_df.empty:
                                    with st.expander("查看自定义候选分组小结"):
                                        st.dataframe(custom_group_summary_df, use_container_width=True)
                                with st.expander("查看自定义对比 Markdown 报告"):
                                    st.markdown(str(custom_result["report"]))
                                custom_download_col1, custom_download_col2, custom_download_col3 = st.columns(3)
                                custom_download_col1.download_button(
                                    label="下载自定义 pairwise CSV",
                                    data=custom_pairwise_export_df.to_csv(index=False).encode("utf-8"),
                                    file_name="custom_candidate_pairwise_comparisons.csv",
                                    mime="text/csv",
                                    use_container_width=True,
                                    key="download_custom_candidate_pairwise_csv",
                                )
                                custom_download_col2.download_button(
                                    label="下载自定义 trade-off CSV",
                                    data=custom_tradeoff_df.to_csv(index=False).encode("utf-8"),
                                    file_name="custom_candidate_tradeoff_table.csv",
                                    mime="text/csv",
                                    use_container_width=True,
                                    key="download_custom_candidate_tradeoff_csv",
                                )
                                custom_download_col3.download_button(
                                    label="下载自定义 Markdown",
                                    data=str(custom_result["report"]).encode("utf-8"),
                                    file_name="custom_candidate_comparison_report.md",
                                    mime="text/markdown",
                                    use_container_width=True,
                                    key="download_custom_candidate_comparison_md",
                                )
                                if st.button(
                                    "保存自定义对比到本次输出目录",
                                    use_container_width=True,
                                    key="save_custom_candidate_comparison_outputs",
                                ):
                                    custom_root = (
                                        candidate_pairwise_comparisons_csv.parent
                                        if candidate_pairwise_comparisons_csv is not None
                                        else consensus_ranking_csv.parent.parent / "candidate_comparisons"
                                    )
                                    selected_slug = _slugify_name("_".join([str(item) for item in selected_custom_ids]))
                                    custom_dir = custom_root / "custom_comparisons" / selected_slug
                                    custom_dir.mkdir(parents=True, exist_ok=True)
                                    custom_tradeoff_df.to_csv(custom_dir / "custom_candidate_tradeoff_table.csv", index=False)
                                    custom_pairwise_df.to_csv(custom_dir / "custom_candidate_pairwise_comparisons.csv", index=False)
                                    if isinstance(custom_group_summary_df, pd.DataFrame):
                                        custom_group_summary_df.to_csv(
                                            custom_dir / "custom_candidate_group_comparison_summary.csv",
                                            index=False,
                                        )
                                    _write_json(custom_dir / "custom_candidate_comparison_summary.json", custom_summary)
                                    _write_text(custom_dir / "custom_candidate_comparison_report.md", str(custom_result["report"]))
                                    st.success(f"已保存自定义候选对比: {custom_dir}")
                            except Exception as exc:
                                st.error(f"生成自定义候选对比失败: {exc}")

            if experiment_plan_csv is not None and experiment_plan_csv.exists():
                st.subheader("本轮实验计划单")
                st.caption("计划单基于下一轮实验建议生成，加入总预算、分层 quota 和 diversity group quota；不改变原始排名或分数。")
                plan_df = _load_csv(experiment_plan_csv)
                if plan_df is not None:
                    if ranking_filter_text and "nanobody_id" in plan_df.columns:
                        plan_df = plan_df[
                            plan_df["nanobody_id"].astype(str).str.contains(
                                ranking_filter_text,
                                case=False,
                                regex=False,
                                na=False,
                            )
                        ]
                    plan_view_df, plan_export_df = _render_dataframe_view_controls(
                        plan_df,
                        key_prefix="experiment_plan_view",
                        preferred_columns=[
                            "plan_rank",
                            "nanobody_id",
                            "plan_decision",
                            "plan_phase",
                            "suggestion_tier",
                            "diversity_group",
                            "manual_plan_action",
                            "manual_plan_override_applied",
                            "experiment_owner",
                            "experiment_status",
                            "experiment_result",
                            "validation_label",
                            "experiment_cost",
                            "experiment_priority_score",
                            "diversity_adjusted_priority_score",
                            "plan_reason",
                            "experiment_note",
                            "recommended_next_action",
                        ],
                        default_sort_column="plan_rank",
                        default_descending=False,
                    )
                    st.dataframe(plan_view_df.head(int(ranking_preview_rows)), use_container_width=True)
                    plan_col1, plan_col2 = st.columns(2)
                    plan_col1.download_button(
                        label="下载当前实验计划视图 CSV",
                        data=plan_view_df.to_csv(index=False).encode("utf-8"),
                        file_name="experiment_plan_filtered.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="download_filtered_experiment_plan_csv",
                    )
                    plan_col2.download_button(
                        label="下载完整实验计划 CSV",
                        data=plan_export_df.to_csv(index=False).encode("utf-8"),
                        file_name="experiment_plan_filtered_full.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="download_filtered_full_experiment_plan_csv",
                    )
                    with st.expander("编辑实验状态并生成下一轮 override CSV"):
                        st.caption(
                            "这里不会改动本次模型分数；只把人工决策、完成状态、负责人、成本和备注整理成 "
                            "`experiment_plan_override.csv`，供下一轮运行自动继承。"
                        )
                        editor_base_df = _build_experiment_plan_override_editor_df(plan_export_df)
                        edited_override_df = st.data_editor(
                            editor_base_df,
                            hide_index=True,
                            use_container_width=True,
                            num_rows="fixed",
                            key=f"experiment_plan_override_editor_{str(experiment_plan_csv)}",
                            column_config={
                                "nanobody_id": st.column_config.TextColumn("nanobody_id", disabled=True),
                                "plan_override": st.column_config.SelectboxColumn(
                                    "plan_override",
                                    options=PLAN_OVERRIDE_ACTION_OPTIONS,
                                    help="include/exclude/standby/defer 会覆盖自动计划；留空则只继承状态字段。",
                                ),
                                "experiment_status": st.column_config.SelectboxColumn(
                                    "experiment_status",
                                    options=PLAN_STATUS_OPTIONS,
                                    help="completed/blocked/in_progress 会在下一轮被识别为状态回灌。",
                                ),
                                "experiment_result": st.column_config.SelectboxColumn(
                                    "experiment_result",
                                    options=PLAN_RESULT_OPTIONS,
                                    help="只有 positive/negative 或显式 validation_label 才会进入训练标签回灌。",
                                ),
                                "validation_label": st.column_config.SelectboxColumn(
                                    "validation_label",
                                    options=PLAN_LABEL_OPTIONS,
                                    help="可选显式标签：1=阳性/有效阻断，0=阴性/无效阻断。",
                                ),
                                "experiment_owner": st.column_config.TextColumn("experiment_owner"),
                                "experiment_cost": st.column_config.TextColumn("experiment_cost"),
                                "experiment_note": st.column_config.TextColumn("experiment_note"),
                                "manual_plan_reason": st.column_config.TextColumn("manual_plan_reason"),
                            },
                        )
                        edited_override_df = _sanitize_plan_override_editor_df(edited_override_df)
                        edited_override_bytes = edited_override_df.to_csv(index=False).encode("utf-8")
                        editor_col1, editor_col2, editor_col3 = st.columns(3)
                        editor_col1.download_button(
                            label="下载编辑后的 override CSV",
                            data=edited_override_bytes,
                            file_name="experiment_plan_override_edited.csv",
                            mime="text/csv",
                            use_container_width=True,
                            key="download_edited_experiment_plan_override_csv",
                        )
                        save_override_clicked = editor_col2.button(
                            "保存到本次运行目录",
                            use_container_width=True,
                            key="save_edited_experiment_plan_override_csv",
                        )
                        save_and_reuse_clicked = editor_col3.button(
                            "保存并用于下一轮",
                            use_container_width=True,
                            key="save_and_reuse_experiment_plan_override_csv",
                        )
                        if save_override_clicked or save_and_reuse_clicked:
                            override_out = experiment_plan_csv.parent / "experiment_plan_override_edited.csv"
                            edited_override_df.to_csv(override_out, index=False)
                            message = f"已保存实验计划覆盖文件: {override_out}"
                            if save_and_reuse_clicked:
                                st.session_state["_pending_experiment_plan_override_local_path"] = str(override_out)
                                st.session_state["_pending_experiment_plan_override_message"] = (
                                    "已把编辑后的实验计划覆盖文件设置为下一轮运行输入。"
                                )
                                st.success("已保存，下一次运行会自动使用该 override CSV。")
                                st.rerun()
                            else:
                                st.success(message)
                        if experiment_plan_state_ledger_csv is not None and experiment_plan_state_ledger_csv.exists():
                            st.download_button(
                                label="下载当前状态 ledger CSV",
                                data=experiment_plan_state_ledger_csv.read_bytes(),
                                file_name=experiment_plan_state_ledger_csv.name,
                                mime="text/csv",
                                use_container_width=True,
                                key="download_experiment_plan_state_ledger_csv",
                            )
                if experiment_plan_md is not None and experiment_plan_md.exists():
                    with st.expander("查看实验计划 Markdown"):
                        st.markdown(_read_text(experiment_plan_md))

            if (
                (validation_evidence_summary_json is not None and validation_evidence_summary_json.exists())
                or (validation_evidence_topk_csv is not None and validation_evidence_topk_csv.exists())
                or (validation_evidence_action_items_csv is not None and validation_evidence_action_items_csv.exists())
            ):
                st.subheader("真实验证证据审计")
                st.caption(
                    "检查当前 top 候选是否已有明确 validation_label 或 experiment_result；"
                    "只评估证据覆盖，不改动模型分数或排名。"
                )
                evidence_payload = (
                    _load_json(validation_evidence_summary_json)
                    if validation_evidence_summary_json is not None and validation_evidence_summary_json.exists()
                    else {}
                )
                if evidence_payload:
                    ev_col1, ev_col2, ev_col3, ev_col4 = st.columns(4)
                    evidence_status = str(evidence_payload.get("audit_status") or "UNKNOWN").upper()
                    ev_col1.metric("Evidence Audit", evidence_status)
                    ev_col2.metric("Label-ready", int(evidence_payload.get("label_ready_count") or 0))
                    ev_col3.metric("Top-k Coverage", _ratio_text(evidence_payload.get("top_k_validation_coverage")))
                    ev_col4.metric("Calibration Ready", "Yes" if bool(evidence_payload.get("calibration_ready")) else "No")
                    if evidence_status == "PASS":
                        st.success("当前验证证据达到最小可用标准。")
                    elif evidence_status == "FAIL":
                        st.error("验证证据审计失败，建议先确认 consensus_ranking.csv 是否存在且可读。")
                    else:
                        st.warning("当前排序还缺少足够真实验证证据，建议先补 top 候选实验结果。")
                    next_actions = evidence_payload.get("next_actions") if isinstance(evidence_payload.get("next_actions"), list) else []
                    for action in next_actions[:4]:
                        st.write(f"- {action}")

                if validation_evidence_topk_csv is not None and validation_evidence_topk_csv.exists():
                    topk_evidence_df = _load_csv(validation_evidence_topk_csv)
                    if topk_evidence_df is not None:
                        if ranking_filter_text and "nanobody_id" in topk_evidence_df.columns:
                            topk_evidence_df = topk_evidence_df[
                                topk_evidence_df["nanobody_id"].astype(str).str.contains(
                                    ranking_filter_text,
                                    case=False,
                                    regex=False,
                                    na=False,
                                )
                            ]
                        topk_view_df, topk_export_df = _render_dataframe_view_controls(
                            topk_evidence_df,
                            key_prefix="validation_evidence_topk_view",
                            preferred_columns=[
                                "nanobody_id",
                                "consensus_rank",
                                "consensus_score",
                                "confidence_level",
                                "validation_label_ready",
                                "validation_label",
                                "validation_gap",
                                "experiment_status",
                                "experiment_result",
                                "recommended_action",
                            ],
                            default_sort_column="consensus_rank",
                            default_descending=False,
                        )
                        st.dataframe(topk_view_df.head(int(ranking_preview_rows)), use_container_width=True)
                        st.download_button(
                            label="下载 top-k 验证证据视图 CSV",
                            data=topk_export_df.to_csv(index=False).encode("utf-8"),
                            file_name="validation_evidence_topk_filtered.csv",
                            mime="text/csv",
                            use_container_width=True,
                            key="download_validation_evidence_topk_filtered_csv",
                        )

                if validation_evidence_action_items_csv is not None and validation_evidence_action_items_csv.exists():
                    with st.expander("查看验证证据行动清单"):
                        action_items_df = _load_csv(validation_evidence_action_items_csv)
                        if action_items_df is None or action_items_df.empty:
                            st.info("当前没有可展示的验证证据行动项。")
                        else:
                            st.dataframe(action_items_df, use_container_width=True, hide_index=True)
                            st.download_button(
                                label="下载验证证据行动清单 CSV",
                                data=action_items_df.to_csv(index=False).encode("utf-8"),
                                file_name=validation_evidence_action_items_csv.name,
                                mime="text/csv",
                                use_container_width=True,
                                key="download_validation_evidence_action_items_csv",
                            )

                evidence_download_col1, evidence_download_col2 = st.columns(2)
                if validation_evidence_summary_json is not None and validation_evidence_summary_json.exists():
                    evidence_download_col1.download_button(
                        label="下载验证证据摘要 JSON",
                        data=validation_evidence_summary_json.read_bytes(),
                        file_name=validation_evidence_summary_json.name,
                        mime="application/json",
                        use_container_width=True,
                        key="download_validation_evidence_summary_json",
                    )
                if validation_evidence_report_md is not None and validation_evidence_report_md.exists():
                    evidence_download_col2.download_button(
                        label="下载验证证据报告 Markdown",
                        data=validation_evidence_report_md.read_bytes(),
                        file_name=validation_evidence_report_md.name,
                        mime="text/markdown",
                        use_container_width=True,
                        key="download_validation_evidence_report_md",
                    )
                    with st.expander("查看验证证据 Markdown"):
                        st.markdown(_read_text(validation_evidence_report_md))

            if experiment_suggestions_csv is not None and experiment_suggestions_csv.exists():
                st.subheader("下一轮实验建议")
                suggestion_df = _load_csv(experiment_suggestions_csv)
                if suggestion_df is not None:
                    if ranking_filter_text and "nanobody_id" in suggestion_df.columns:
                        suggestion_df = suggestion_df[
                            suggestion_df["nanobody_id"].astype(str).str.contains(
                                ranking_filter_text,
                                case=False,
                                regex=False,
                                na=False,
                            )
                        ]
                    suggestion_df, suggestion_threshold_summaries = _render_numeric_threshold_filters(
                        suggestion_df,
                        key_prefix="experiment_suggestions",
                        title="实验建议数值阈值筛选",
                    )
                    suggestion_view_df, suggestion_export_df = _render_dataframe_view_controls(
                        suggestion_df,
                        key_prefix="experiment_suggestions_view",
                        preferred_columns=[
                            "nanobody_id",
                            "suggestion_rank",
                            "suggestion_tier",
                            "experiment_priority_score",
                            "diversity_adjusted_priority_score",
                            "diversity_group",
                            "diversity_seen_before",
                            "diversity_adjustment",
                            "diversity_note",
                            "recommended_next_action",
                            "suggestion_reason",
                            "consensus_score",
                            "confidence_score",
                            "qc_risk_score",
                            "abs_rank_delta",
                            "review_reason_flags",
                            "low_confidence_reasons",
                        ],
                        default_sort_column="suggestion_rank",
                        default_descending=False,
                    )
                    if suggestion_threshold_summaries:
                        st.caption("已启用阈值: " + "；".join(suggestion_threshold_summaries))
                    st.caption(
                        f"当前筛选后共 {len(suggestion_export_df)} 条，当前展示前 {min(int(ranking_preview_rows), len(suggestion_view_df))} 条；"
                        f"可见列 {len(suggestion_view_df.columns)} 个。"
                    )
                    st.dataframe(suggestion_view_df.head(int(ranking_preview_rows)), use_container_width=True)
                    suggestion_download_col1, suggestion_download_col2 = st.columns(2)
                    suggestion_download_col1.download_button(
                        label="下载当前实验建议可见视图 CSV",
                        data=suggestion_view_df.to_csv(index=False).encode("utf-8"),
                        file_name="experiment_suggestions_filtered.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="download_filtered_experiment_suggestions_csv",
                    )
                    suggestion_download_col2.download_button(
                        label="下载当前实验建议全筛选结果 CSV",
                        data=suggestion_export_df.to_csv(index=False).encode("utf-8"),
                        file_name="experiment_suggestions_filtered_full.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="download_filtered_full_experiment_suggestions_csv",
                    )

            if parameter_sensitivity_candidate_csv is not None and parameter_sensitivity_candidate_csv.exists():
                st.subheader("参数敏感性分析")
                st.caption("该表不重新训练模型，只检查共识排名在不同 Rule/ML 权重、rank-agreement 权重和 QC 惩罚强度下是否稳定。")
                sensitivity_df = _load_csv(parameter_sensitivity_candidate_csv)
                if sensitivity_df is not None:
                    if ranking_filter_text and "nanobody_id" in sensitivity_df.columns:
                        sensitivity_df = sensitivity_df[
                            sensitivity_df["nanobody_id"].astype(str).str.contains(
                                ranking_filter_text,
                                case=False,
                                regex=False,
                                na=False,
                            )
                        ]
                    sensitivity_view_df, sensitivity_export_df = _render_dataframe_view_controls(
                        sensitivity_df,
                        key_prefix="parameter_sensitivity_view",
                        preferred_columns=[
                            "nanobody_id",
                            "baseline_rank",
                            "best_rank",
                            "worst_rank",
                            "rank_span",
                            "top_n_frequency",
                            "top_n_unstable",
                            "is_sensitive",
                            "sensitivity_reason",
                            "decision_tier",
                            "confidence_level",
                        ],
                        default_sort_column="rank_span",
                        default_descending=True,
                    )
                    st.dataframe(sensitivity_view_df.head(int(ranking_preview_rows)), use_container_width=True)
                    sens_col1, sens_col2 = st.columns(2)
                    sens_col1.download_button(
                        label="下载参数敏感性可见视图 CSV",
                        data=sensitivity_view_df.to_csv(index=False).encode("utf-8"),
                        file_name="parameter_sensitivity_filtered.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="download_parameter_sensitivity_view_csv",
                    )
                    sens_col2.download_button(
                        label="下载参数敏感性全表 CSV",
                        data=sensitivity_export_df.to_csv(index=False).encode("utf-8"),
                        file_name="parameter_sensitivity_filtered_full.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="download_parameter_sensitivity_full_csv",
                    )
                if parameter_sensitivity_report_md is not None and parameter_sensitivity_report_md.exists():
                    with st.expander("查看参数敏感性报告"):
                        st.markdown(_read_text(parameter_sensitivity_report_md))

            if consensus_ranking_csv is not None and consensus_ranking_csv.exists():
                st.subheader("Rule + ML 共识排名")
                consensus_df = _load_csv(consensus_ranking_csv)
                if consensus_df is not None:
                    if ranking_filter_text and "nanobody_id" in consensus_df.columns:
                        consensus_df = consensus_df[
                            consensus_df["nanobody_id"].astype(str).str.contains(
                                ranking_filter_text,
                                case=False,
                                regex=False,
                                na=False,
                            )
                        ]
                    consensus_df, consensus_threshold_summaries = _render_numeric_threshold_filters(
                        consensus_df,
                        key_prefix="consensus_ranking",
                        title="共识排名数值阈值筛选",
                    )
                    consensus_view_df, consensus_export_df = _render_dataframe_view_controls(
                        consensus_df,
                        key_prefix="consensus_ranking_view",
                        preferred_columns=[
                            "nanobody_id",
                            "consensus_rank",
                            "consensus_score",
                            "confidence_level",
                            "confidence_score",
                            "decision_tier",
                            "ml_score",
                            "rule_score",
                            "rank_agreement_score",
                            "qc_risk_score",
                            "review_reason_flags",
                            "low_confidence_reasons",
                            "risk_flags",
                            "consensus_explanation",
                        ],
                        default_sort_column="consensus_score",
                        default_descending=True,
                    )
                    if consensus_threshold_summaries:
                        st.caption("已启用阈值: " + "；".join(consensus_threshold_summaries))
                    st.caption(
                        f"当前筛选后共 {len(consensus_export_df)} 条，当前展示前 {min(int(ranking_preview_rows), len(consensus_view_df))} 条；"
                        f"可见列 {len(consensus_view_df.columns)} 个。"
                    )
                    st.dataframe(consensus_view_df.head(int(ranking_preview_rows)), use_container_width=True)
                    consensus_download_col1, consensus_download_col2 = st.columns(2)
                    consensus_download_col1.download_button(
                        label="下载当前共识可见视图 CSV",
                        data=consensus_view_df.to_csv(index=False).encode("utf-8"),
                        file_name="consensus_ranking_filtered.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="download_filtered_consensus_ranking_csv",
                    )
                    consensus_download_col2.download_button(
                        label="下载当前共识全筛选结果 CSV",
                        data=consensus_export_df.to_csv(index=False).encode("utf-8"),
                        file_name="consensus_ranking_filtered_full.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="download_filtered_full_consensus_ranking_csv",
                    )

            if ml_ranking_csv is not None and ml_ranking_csv.exists():
                st.subheader("ML 排名")
                ml_df = _load_csv(ml_ranking_csv)
                if ml_df is not None:
                    if ranking_filter_text and "nanobody_id" in ml_df.columns:
                        ml_df = ml_df[
                            ml_df["nanobody_id"].astype(str).str.contains(
                                ranking_filter_text,
                                case=False,
                                regex=False,
                                na=False,
                            )
                        ]
                    ml_df, ml_threshold_summaries = _render_numeric_threshold_filters(
                        ml_df,
                        key_prefix="ml_ranking",
                        title="ML 排名数值阈值筛选",
                    )
                    ml_view_df, ml_export_df = _render_dataframe_view_controls(
                        ml_df,
                        key_prefix="ml_ranking_view",
                        preferred_columns=[
                            "nanobody_id",
                            "final_score",
                            "best_conformer_score",
                            "mean_conformer_score",
                            "pocket_consistency_score",
                            "std_conformer_score",
                            "top_conformer_id",
                            "explanation",
                        ],
                        default_sort_column="final_score",
                        default_descending=True,
                    )
                    if ml_threshold_summaries:
                        st.caption("已启用阈值: " + "；".join(ml_threshold_summaries))
                    st.caption(
                        f"当前筛选后共 {len(ml_export_df)} 条，当前展示前 {min(int(ranking_preview_rows), len(ml_view_df))} 条；"
                        f"可见列 {len(ml_view_df.columns)} 个。"
                    )
                    st.dataframe(ml_view_df.head(int(ranking_preview_rows)), use_container_width=True)
                    ml_download_col1, ml_download_col2 = st.columns(2)
                    ml_download_col1.download_button(
                        label="下载当前 ML 可见视图 CSV",
                        data=ml_view_df.to_csv(index=False).encode("utf-8"),
                        file_name="ml_ranking_filtered.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="download_filtered_ml_ranking_csv",
                    )
                    ml_download_col2.download_button(
                        label="下载当前 ML 全筛选结果 CSV",
                        data=ml_export_df.to_csv(index=False).encode("utf-8"),
                        file_name="ml_ranking_filtered_full.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="download_filtered_full_ml_ranking_csv",
                    )

            if rule_ranking_csv is not None and rule_ranking_csv.exists():
                st.subheader("Rule 排名")
                rule_df = _load_csv(rule_ranking_csv)
                if rule_df is not None:
                    if ranking_filter_text and "nanobody_id" in rule_df.columns:
                        rule_df = rule_df[
                            rule_df["nanobody_id"].astype(str).str.contains(
                                ranking_filter_text,
                                case=False,
                                regex=False,
                                na=False,
                            )
                        ]
                    rule_df, rule_threshold_summaries = _render_numeric_threshold_filters(
                        rule_df,
                        key_prefix="rule_ranking",
                        title="Rule 排名数值阈值筛选",
                    )
                    rule_view_df, rule_export_df = _render_dataframe_view_controls(
                        rule_df,
                        key_prefix="rule_ranking_view",
                        preferred_columns=[
                            "nanobody_id",
                            "final_rule_score",
                            "best_conformer_rule_score",
                            "mean_conformer_rule_score",
                            "pocket_consistency_score",
                            "std_conformer_rule_score",
                            "top_conformer_id",
                            "explanation",
                        ],
                        default_sort_column="final_rule_score",
                        default_descending=True,
                    )
                    if rule_threshold_summaries:
                        st.caption("已启用阈值: " + "；".join(rule_threshold_summaries))
                    st.caption(
                        f"当前筛选后共 {len(rule_export_df)} 条，当前展示前 {min(int(ranking_preview_rows), len(rule_view_df))} 条；"
                        f"可见列 {len(rule_view_df.columns)} 个。"
                    )
                    st.dataframe(rule_view_df.head(int(ranking_preview_rows)), use_container_width=True)
                    rule_download_col1, rule_download_col2 = st.columns(2)
                    rule_download_col1.download_button(
                        label="下载当前 Rule 可见视图 CSV",
                        data=rule_view_df.to_csv(index=False).encode("utf-8"),
                        file_name="rule_ranking_filtered.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="download_filtered_rule_ranking_csv",
                    )
                    rule_download_col2.download_button(
                        label="下载当前 Rule 全筛选结果 CSV",
                        data=rule_export_df.to_csv(index=False).encode("utf-8"),
                        file_name="rule_ranking_filtered_full.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="download_filtered_full_rule_ranking_csv",
                    )

    with tab_pose:
        if not isinstance(summary, dict):
            st.info("先运行一次 pipeline，再查看 pose 结果。")
        else:
            artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
            out_dir_text = summary.get("out_dir")
            pose_pred_csv = None
            if out_dir_text:
                pose_pred_csv = Path(str(out_dir_text)) / "model_outputs" / "pose_predictions.csv"
            pose_preview_rows = st.number_input(
                "Pose 结果预览前 N 行",
                min_value=10,
                max_value=1000,
                value=100,
                step=10,
                key="pose_preview_rows",
            )
            pose_filter_text = str(st.text_input("按 nanobody_id / conformer_id / pose_id 过滤", key="pose_filter_text") or "").strip()
            if pose_pred_csv is not None and pose_pred_csv.exists():
                pose_df = _load_csv(pose_pred_csv)
                if pose_df is not None:
                    if pose_filter_text:
                        mask = pd.Series(False, index=pose_df.index)
                        for column in ["nanobody_id", "conformer_id", "pose_id"]:
                            if column in pose_df.columns:
                                mask = mask | pose_df[column].astype(str).str.contains(
                                    pose_filter_text,
                                    case=False,
                                    regex=False,
                                    na=False,
                                )
                        pose_df = pose_df[mask]
                    pose_df, pose_threshold_summaries = _render_numeric_threshold_filters(
                        pose_df,
                        key_prefix="pose_results",
                        title="Pose 结果数值阈值筛选",
                    )
                    pose_view_df, pose_export_df = _render_dataframe_view_controls(
                        pose_df,
                        key_prefix="pose_view",
                        preferred_columns=[
                            "nanobody_id",
                            "conformer_id",
                            "pose_id",
                            "pred_prob",
                            "pred_logit",
                            "pseudo_label",
                            "pocket_hit_fraction",
                            "catalytic_hit_fraction",
                            "mouth_occlusion_score",
                            "top_contributing_features",
                        ],
                        default_sort_column="pred_prob",
                        default_descending=True,
                    )
                    if pose_threshold_summaries:
                        st.caption("已启用阈值: " + "；".join(pose_threshold_summaries))
                    st.caption(
                        f"当前筛选后共 {len(pose_export_df)} 条，当前展示前 {min(int(pose_preview_rows), len(pose_view_df))} 条；"
                        f"可见列 {len(pose_view_df.columns)} 个。"
                    )
                    st.dataframe(pose_view_df.head(int(pose_preview_rows)), use_container_width=True)
                    pose_download_col1, pose_download_col2 = st.columns(2)
                    pose_download_col1.download_button(
                        label="下载当前 Pose 可见视图 CSV",
                        data=pose_view_df.to_csv(index=False).encode("utf-8"),
                        file_name="pose_predictions_filtered.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="download_filtered_pose_csv",
                    )
                    pose_download_col2.download_button(
                        label="下载当前 Pose 全筛选结果 CSV",
                        data=pose_export_df.to_csv(index=False).encode("utf-8"),
                        file_name="pose_predictions_filtered_full.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="download_filtered_full_pose_csv",
                    )
            else:
                st.info("当前没有 pose_predictions.csv。")

    with tab_report:
        if not isinstance(summary, dict):
            st.info("先运行一次 pipeline，再查看报告。")
        else:
            artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
            report_path = Path(str(artifacts.get("execution_report_md"))) if artifacts.get("execution_report_md") else None
            if report_path is not None and report_path.exists():
                st.markdown(_read_text(report_path))
            else:
                st.info("当前没有 execution report。")

    with tab_ai:
        if not isinstance(summary, dict):
            st.info("先运行一次 pipeline，再查看 AI 解释。")
        else:
            artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
            ai_summary_path = Path(str(artifacts.get("ai_run_summary_md"))) if artifacts.get("ai_run_summary_md") else None
            ai_candidate_path = (
                Path(str(artifacts.get("ai_top_candidates_explanation_md")))
                if artifacts.get("ai_top_candidates_explanation_md")
                else None
            )
            ai_diag_path = (
                Path(str(artifacts.get("ai_failure_diagnosis_md")))
                if artifacts.get("ai_failure_diagnosis_md")
                else None
            )
            ai_meta_path = (
                Path(str(artifacts.get("ai_assistant_summary_json")))
                if artifacts.get("ai_assistant_summary_json")
                else None
            )
            if ai_summary_path is None or not ai_summary_path.exists():
                st.info("当前运行未启用 AI 解释。可在左侧“AI 解释（可选）”打开后重新运行。")
            else:
                ai_meta = _load_json(ai_meta_path) if ai_meta_path is not None else None
                if isinstance(ai_meta, dict):
                    st.caption(
                        f"Provider: {ai_meta.get('provider_requested', 'N/A')}；"
                        f"Model: {ai_meta.get('model', 'N/A')}；"
                        f"生成时间: {ai_meta.get('created_at', 'N/A')}"
                    )
                    errors = ai_meta.get("errors") if isinstance(ai_meta.get("errors"), dict) else {}
                    if errors:
                        st.warning("在线 AI provider 不可用时已自动回退为本地离线摘要。")
                st.subheader("运行摘要")
                st.markdown(_read_text(ai_summary_path))
                if ai_candidate_path is not None and ai_candidate_path.exists():
                    with st.expander("Top candidates 解释", expanded=False):
                        st.markdown(_read_text(ai_candidate_path))
                if ai_diag_path is not None and ai_diag_path.exists():
                    with st.expander("失败诊断 / 产物检查", expanded=False):
                        st.markdown(_read_text(ai_diag_path))

    with tab_logs:
        stdout_text = str(st.session_state.get("last_stdout") or "")
        stderr_text = str(st.session_state.get("last_stderr") or "")
        if stdout_text:
            st.subheader("STDOUT")
            st.code(stdout_text)
        if stderr_text:
            st.subheader("STDERR")
            st.code(stderr_text)
        if not stdout_text and not stderr_text:
            st.info("当前没有日志可显示。")

    with tab_diag:
        _render_diagnostics_panel(diagnostics, last_error, metadata)
        st.divider()
        _render_cd38_public_starter_panel()


if __name__ == "__main__":
    main()
