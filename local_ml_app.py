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

REPO_ROOT = Path(__file__).resolve().parent
LOCAL_APP_RUN_ROOT = REPO_ROOT / "local_app_runs"
COMPARE_EXPORT_ROOT = LOCAL_APP_RUN_ROOT / "compare_exports"
PARAM_TEMPLATE_ROOT = LOCAL_APP_RUN_ROOT / "parameter_templates"
APP_RUN_METADATA_NAME = "app_run_metadata.json"
APP_STDOUT_NAME = "app_stdout.log"
APP_STDERR_NAME = "app_stderr.log"
LOCAL_APP_RUN_ROOT.mkdir(parents=True, exist_ok=True)
COMPARE_EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
PARAM_TEMPLATE_ROOT.mkdir(parents=True, exist_ok=True)

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
]
BUNDLE_IMPORT_ROOT = LOCAL_APP_RUN_ROOT / "_bundle_imports"
BUNDLE_IMPORT_ROOT.mkdir(parents=True, exist_ok=True)


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
    "default_antigen_chain": "",
    "default_nanobody_chain": "",
    "top_k": 3,
    "train_epochs": 20,
    "train_batch_size": 64,
    "train_val_ratio": 0.25,
    "train_early_stopping_patience": 8,
    "seed": 42,
    "skip_failed_rows": True,
    "disable_label_aware_steps": False,
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
    "default_antigen_chain",
    "default_nanobody_chain",
    "top_k",
    "train_epochs",
    "train_batch_size",
    "train_val_ratio",
    "train_early_stopping_patience",
    "seed",
    "skip_failed_rows",
    "disable_label_aware_steps",
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
    default_antigen_chain: str,
    default_nanobody_chain: str,
    top_k: int,
    train_epochs: int,
    train_batch_size: int,
    train_val_ratio: float,
    train_early_stopping_patience: int,
    seed: int,
    skip_failed_rows: bool,
    disable_label_aware_steps: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "run_recommended_pipeline.py"),
        "--out_dir",
        str(out_dir),
        "--top_k",
        str(int(top_k)),
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
) -> dict[str, Any]:
    return {
        "input_csv": None if input_csv is None else str(input_csv),
        "feature_csv": None if feature_csv is None else str(feature_csv),
        "out_dir": str(out_dir),
        "label_col": "label",
        "disable_label_aware_steps": bool(form_payload.get("disable_label_aware_steps", False)),
        "default_pocket_file": None if default_pocket_file is None else str(default_pocket_file),
        "default_catalytic_file": None if default_catalytic_file is None else str(default_catalytic_file),
        "default_ligand_file": None if default_ligand_file is None else str(default_ligand_file),
        "default_antigen_chain": str(form_payload.get("default_antigen_chain") or ""),
        "default_nanobody_chain": str(form_payload.get("default_nanobody_chain") or ""),
        "skip_failed_rows": bool(form_payload.get("skip_failed_rows", True)),
        "top_k": int(form_payload.get("top_k", 3)),
        "train_epochs": int(form_payload.get("train_epochs", 20)),
        "train_batch_size": int(form_payload.get("train_batch_size", 64)),
        "train_val_ratio": float(form_payload.get("train_val_ratio", 0.25)),
        "train_early_stopping_patience": int(form_payload.get("train_early_stopping_patience", 8)),
        "seed": int(form_payload.get("seed", 42)),
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
) -> dict[str, Any]:
    form_payload = _build_form_payload()
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

    resolved_inputs = {
        "input_csv": None if input_csv is None else str(input_csv),
        "feature_csv": None if feature_csv is None else str(feature_csv),
        "default_pocket_file": None if default_pocket_file is None else str(default_pocket_file),
        "default_catalytic_file": None if default_catalytic_file is None else str(default_catalytic_file),
        "default_ligand_file": None if default_ligand_file is None else str(default_ligand_file),
    }

    command = _build_pipeline_command(
        start_mode=str(form_payload.get("start_mode")),
        input_csv=input_csv,
        feature_csv=feature_csv,
        out_dir=out_dir,
        default_pocket_file=default_pocket_file,
        default_catalytic_file=default_catalytic_file,
        default_ligand_file=default_ligand_file,
        default_antigen_chain=str(form_payload.get("default_antigen_chain") or ""),
        default_nanobody_chain=str(form_payload.get("default_nanobody_chain") or ""),
        top_k=int(form_payload.get("top_k", 3)),
        train_epochs=int(form_payload.get("train_epochs", 20)),
        train_batch_size=int(form_payload.get("train_batch_size", 64)),
        train_val_ratio=float(form_payload.get("train_val_ratio", 0.25)),
        train_early_stopping_patience=int(form_payload.get("train_early_stopping_patience", 8)),
        seed=int(form_payload.get("seed", 42)),
        skip_failed_rows=bool(form_payload.get("skip_failed_rows", True)),
        disable_label_aware_steps=bool(form_payload.get("disable_label_aware_steps", False)),
    )
    pipeline_kwargs = _build_pipeline_kwargs(
        form_payload=form_payload,
        out_dir=out_dir,
        input_csv=input_csv,
        feature_csv=feature_csv,
        default_pocket_file=default_pocket_file,
        default_catalytic_file=default_catalytic_file,
        default_ligand_file=default_ligand_file,
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

    metadata = _read_run_metadata(run_root)
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


def _summarize_csv_dataframe(
    df: pd.DataFrame,
    *,
    required_columns: list[str],
    optional_columns: list[str] | None = None,
) -> dict[str, Any]:
    missing_required = [col for col in required_columns if col not in df.columns]
    optional_present = [col for col in (optional_columns or []) if col in df.columns]
    preview_df = df.head(5).copy()
    preview_df.columns = [str(col) for col in preview_df.columns]
    label_stats = _count_label_stats(df)
    return {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "columns_preview": [str(col) for col in df.columns[:12].tolist()],
        "missing_required_columns": missing_required,
        "optional_present_columns": optional_present,
        "preview_rows": preview_df.to_dict(orient="records"),
        **label_stats,
    }


def _inspect_csv_source(
    *,
    uploaded_file: Any,
    local_path_text: str,
    required_columns: list[str],
    optional_columns: list[str] | None = None,
) -> dict[str, Any]:
    local_path = str(local_path_text or "").strip()
    if local_path:
        path = Path(local_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"路径不存在: {path}")
        df = pd.read_csv(path, low_memory=False)
        size_text = _format_size_text(path)
        summary = _summarize_csv_dataframe(
            df,
            required_columns=required_columns,
            optional_columns=optional_columns,
        )
        summary.update(
            {
                "source": "本地路径",
                "display_name": path.name,
                "path": str(path),
                "size_text": size_text,
            }
        )
        return summary

    if uploaded_file is None:
        raise ValueError("当前没有可检查的 CSV 输入。")

    file_bytes = bytes(uploaded_file.getbuffer())
    df = pd.read_csv(BytesIO(file_bytes), low_memory=False)
    summary = _summarize_csv_dataframe(
        df,
        required_columns=required_columns,
        optional_columns=optional_columns,
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
    return (
        f"已导入{source_kind_text} {report.get('source_name', '')}，自动识别 {len(detected_paths)} 个输入项："
        + ", ".join(detected_names)
    )


def _build_preflight_report(
    *,
    start_mode: str,
    input_csv_upload: Any,
    feature_csv_upload: Any,
    default_pocket_upload: Any,
    default_catalytic_upload: Any,
    default_ligand_upload: Any,
    form_payload: dict[str, Any],
) -> dict[str, Any]:
    status = "ready"
    messages: list[str] = []

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
    ]

    if start_mode == "input_csv":
        csv_summary = _inspect_csv_source(
            uploaded_file=input_csv_upload,
            local_path_text=str(form_payload.get("input_csv_local_path") or ""),
            required_columns=INPUT_CSV_REQUIRED_COLUMNS,
            optional_columns=INPUT_CSV_OPTIONAL_HINT_COLUMNS,
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
    else:
        csv_summary = _inspect_csv_source(
            uploaded_file=feature_csv_upload,
            local_path_text=str(form_payload.get("feature_csv_local_path") or ""),
            required_columns=FEATURE_CSV_REQUIRED_COLUMNS,
            optional_columns=FEATURE_CSV_OPTIONAL_HINT_COLUMNS,
        )
        missing_required = csv_summary["missing_required_columns"]
        if missing_required:
            status = "error"
            messages.append(f"pose_features.csv 缺少必需列: {missing_required}")
        else:
            messages.append("pose_features.csv 最小 ID 列完整，可直接进入 rule/ML 排名阶段。")

        if not csv_summary.get("has_label"):
            if status != "error":
                status = "warning"
            messages.append("当前没有 label 列，compare/calibration/strategy optimize 会被自动跳过。")
        elif not csv_summary.get("label_compare_possible"):
            if status != "error":
                status = "warning"
            messages.append("label 有效样本不足或类别退化，label-aware 步骤会被自动跳过。")
        else:
            messages.append(
                f"检测到 {csv_summary['label_valid_count']} 条有效 label，"
                f"类别数 {csv_summary['label_class_count']}。"
            )
            if csv_summary.get("calibration_possible"):
                messages.append("当前 label 数量已满足自动校准的最小要求。")
            else:
                if status != "error":
                    status = "warning"
                messages.append("当前 label 还不足 8 条，compare 可执行，但 calibration 会被自动跳过。")

    return {
        "checked_at": _now_text(),
        "start_mode": start_mode,
        "status": status,
        "messages": messages,
        "main_csv": csv_summary,
        "optional_files": optional_file_rows,
    }


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
            c5, c6, c7 = st.columns(3)
            c5.metric("有效 label", int(main_csv.get("label_valid_count") or 0))
            c6.metric("label 类别数", int(main_csv.get("label_class_count") or 0))
            c7.metric(
                "可做 calibration",
                "是" if bool(main_csv.get("calibration_possible")) else "否",
            )

        preview_rows = main_csv.get("preview_rows") if isinstance(main_csv.get("preview_rows"), list) else []
        if preview_rows:
            st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)

    optional_files = report.get("optional_files") if isinstance(report.get("optional_files"), list) else []
    if optional_files:
        st.caption("默认文件来源检查")
        st.dataframe(pd.DataFrame(optional_files), use_container_width=True, hide_index=True)


def _render_input_status_panel(
    *,
    start_mode: str,
    input_csv_upload: Any,
    feature_csv_upload: Any,
    default_pocket_upload: Any,
    default_catalytic_upload: Any,
    default_ligand_upload: Any,
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
        st.caption("input_csv 必需列: " + ", ".join(INPUT_CSV_REQUIRED_COLUMNS))
        st.caption("pose_features.csv 最小必需列: " + ", ".join(FEATURE_CSV_REQUIRED_COLUMNS))
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
        "execution_report_md",
        "rule_ranking_csv",
        "ml_ranking_csv",
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

    ml_ranking_path = Path(str(artifacts.get("ml_ranking_csv"))) if artifacts.get("ml_ranking_csv") else None
    rule_ranking_path = Path(str(artifacts.get("rule_ranking_csv"))) if artifacts.get("rule_ranking_csv") else None
    report_path = Path(str(artifacts.get("execution_report_md"))) if artifacts.get("execution_report_md") else None
    training_summary_path = _resolve_output_file(
        "model_outputs",
        "training_summary.json",
        summary=summary,
        metadata=metadata,
    )

    ml_df = _load_csv(ml_ranking_path) if ml_ranking_path is not None and ml_ranking_path.exists() else None
    rule_df = _load_csv(rule_ranking_path) if rule_ranking_path is not None and rule_ranking_path.exists() else None
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

    ml_ranking_path = Path(str(artifacts.get("ml_ranking_csv"))) if artifacts.get("ml_ranking_csv") else None
    rule_ranking_path = Path(str(artifacts.get("rule_ranking_csv"))) if artifacts.get("rule_ranking_csv") else None
    report_path = Path(str(artifacts.get("execution_report_md"))) if artifacts.get("execution_report_md") else None
    ml_df = _load_csv(ml_ranking_path) if ml_ranking_path is not None and ml_ranking_path.exists() else None
    rule_df = _load_csv(rule_ranking_path) if rule_ranking_path is not None and rule_ranking_path.exists() else None
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
    delta_table_html = _df_to_html_table(delta_df, max_rows=max(20, len(delta_df) if isinstance(delta_df, pd.DataFrame) else 20))
    trend_table_html = _df_to_html_table(trend_df, max_rows=max(20, len(trend_df) if isinstance(trend_df, pd.DataFrame) else 20))
    batch_table_html = _df_to_html_table(batch_df, max_rows=max(20, len(batch_df) if isinstance(batch_df, pd.DataFrame) else 20))
    attribution_table_html = _df_to_html_table(attribution_df, max_rows=max(20, len(attribution_df) if isinstance(attribution_df, pd.DataFrame) else 20))
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
      <h2>基准差异解释</h2>
      {"<ul>" + insight_lines_html + "</ul>" if insight_lines_html else "<p class='muted'>当前没有可用的差异解释。</p>"}
    </section>

    <section class="section">
      <h2>趋势快照</h2>
      {trend_table_html}
    </section>

    <section class="section">
      <h2>跨批次趋势聚合</h2>
      {batch_table_html}
    </section>

    <section class="section">
      <h2>批次级观察</h2>
      {"<ul>" + batch_insight_lines_html + "</ul>" if batch_insight_lines_html else "<p class='muted'>当前没有可用的批次级观察。</p>"}
    </section>

    <section class="section">
      <h2>相对基准的差异表</h2>
      {delta_table_html}
    </section>

    <section class="section">
      <h2>更细的差异归因总表</h2>
      {attribution_table_html}
    </section>

    <section class="section">
      <h2>运行级归因明细</h2>
      {"<div class='meta'>" + attribution_detail_html + "</div>" if attribution_detail_html else "<p class='muted'>当前没有可展示的归因明细。</p>"}
    </section>

    <section class="section">
      <h2>完整对比表</h2>
      {_df_to_html_table(compare_df, max_rows=max(20, len(compare_df)))}
    </section>

    <div class="footer">
      该 HTML 为多运行对比导出页，可直接下载、发送或用浏览器打印为 PDF。
    </div>
  </div>
</body>
</html>
"""
    _write_text(html_path, html_text)
    csv_path = export_dir / "history_compare_table.csv"
    csv_path.write_text(compare_df.to_csv(index=False), encoding="utf-8")
    if isinstance(batch_df, pd.DataFrame) and not batch_df.empty:
        batch_csv_path = export_dir / "history_compare_batch_table.csv"
        batch_csv_path.write_text(batch_df.to_csv(index=False), encoding="utf-8")
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
        {"title": "Run-to-Run 差异解释", "kind": "bullets", "lines": insight_lines or ["当前没有可用的差异解释。"]},
        {
            "title": "趋势快照",
            "kind": "table",
            "lines": _df_to_pdf_lines(trend_df, max_rows=max(20, len(trend_df) if isinstance(trend_df, pd.DataFrame) else 20), max_cols=6),
        },
        {
            "title": "跨批次趋势聚合",
            "kind": "table",
            "lines": _df_to_pdf_lines(batch_df, max_rows=max(20, len(batch_df) if isinstance(batch_df, pd.DataFrame) else 20), max_cols=8),
        },
        {
            "title": "批次级观察",
            "kind": "bullets",
            "lines": batch_insight_lines or ["当前没有可用的批次级观察。"],
        },
        {
            "title": "相对基准的差异表",
            "kind": "table",
            "lines": _df_to_pdf_lines(delta_df, max_rows=max(20, len(delta_df) if isinstance(delta_df, pd.DataFrame) else 20), max_cols=10),
        },
        {
            "title": "更细的差异归因总表",
            "kind": "table",
            "lines": _df_to_pdf_lines(attribution_df, max_rows=max(20, len(attribution_df) if isinstance(attribution_df, pd.DataFrame) else 20), max_cols=6),
        },
        {
            "title": "运行级归因明细",
            "kind": "bullets",
            "lines": attribution_detail_lines or ["当前没有可展示的归因明细。"],
        },
        {
            "title": "完整对比表",
            "kind": "table",
            "lines": _df_to_pdf_lines(compare_df, max_rows=max(20, len(compare_df)), max_cols=10),
        },
    ]

    csv_path = export_dir / "history_compare_table.csv"
    csv_path.write_text(compare_df.to_csv(index=False), encoding="utf-8")
    if isinstance(batch_df, pd.DataFrame) and not batch_df.empty:
        batch_csv_path = export_dir / "history_compare_batch_table.csv"
        batch_csv_path.write_text(batch_df.to_csv(index=False), encoding="utf-8")
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

    export_col1, export_col2, export_col3 = st.columns(3)
    create_compare_html_clicked = export_col1.button(
        "生成当前对比 HTML",
        use_container_width=True,
        key="create_compare_html_clicked",
    )
    create_compare_pdf_clicked = export_col2.button(
        "生成当前对比 PDF",
        use_container_width=True,
        key="create_compare_pdf_clicked",
    )
    if create_compare_html_clicked:
        with st.spinner("正在生成多运行对比 HTML..."):
            html_path = _create_history_compare_html_export(
                selected_records,
                compare_df,
                metric_key=metric_key,
                metric_label=_get_compare_metric_label(metric_key),
                higher_is_better=higher_is_better,
                leader_name=leader_name,
                leader_value=leader_value,
                baseline_run_name=baseline_run_name,
                insight_lines=insight_lines,
                delta_df=delta_df,
                trend_df=trend_df,
                batch_df=batch_display_df,
                batch_insight_lines=batch_insight_lines,
                attribution_df=attribution_df,
                attribution_payloads=attribution_payloads,
            )
        st.session_state["last_compare_export_html"] = str(html_path)
        st.session_state["last_compare_export_html_message"] = f"已生成多运行对比 HTML: {html_path}"
    if create_compare_pdf_clicked:
        with st.spinner("正在生成多运行对比 PDF..."):
            pdf_path = _create_history_compare_pdf_export(
                selected_records,
                compare_df,
                metric_key=metric_key,
                metric_label=_get_compare_metric_label(metric_key),
                higher_is_better=higher_is_better,
                leader_name=leader_name,
                leader_value=leader_value,
                baseline_run_name=baseline_run_name,
                insight_lines=insight_lines,
                delta_df=delta_df,
                trend_df=trend_df,
                batch_df=batch_display_df,
                batch_insight_lines=batch_insight_lines,
                attribution_df=attribution_df,
                attribution_payloads=attribution_payloads,
            )
        st.session_state["last_compare_export_pdf"] = str(pdf_path)
        st.session_state["last_compare_export_pdf_message"] = f"已生成多运行对比 PDF: {pdf_path}"

    last_compare_export_html = st.session_state.get("last_compare_export_html")
    last_compare_export_html_message = str(st.session_state.get("last_compare_export_html_message") or "")
    last_compare_export_pdf = st.session_state.get("last_compare_export_pdf")
    last_compare_export_pdf_message = str(st.session_state.get("last_compare_export_pdf_message") or "")
    if last_compare_export_html_message:
        st.success(last_compare_export_html_message)
    if last_compare_export_html and Path(str(last_compare_export_html)).exists():
        compare_html_path = Path(str(last_compare_export_html))
        st.caption(f"当前多运行对比 HTML: {compare_html_path}")
        export_col1.download_button(
            label="下载当前对比 HTML",
            data=compare_html_path.read_bytes(),
            file_name=compare_html_path.name,
            mime="text/html",
            use_container_width=True,
        )
        if export_col3.button("打开当前对比 HTML", use_container_width=True, key="open_compare_html_clicked"):
            ok, message = _open_local_path(compare_html_path)
            if ok:
                st.success(message)
            else:
                st.error(message)
    if last_compare_export_pdf_message:
        st.success(last_compare_export_pdf_message)
    if last_compare_export_pdf and Path(str(last_compare_export_pdf)).exists():
        compare_pdf_path = Path(str(last_compare_export_pdf))
        st.caption(f"当前多运行对比 PDF: {compare_pdf_path}")
        pdf_download_col1, pdf_download_col2 = st.columns(2)
        pdf_download_col1.download_button(
            label="下载当前对比 PDF",
            data=compare_pdf_path.read_bytes(),
            file_name=compare_pdf_path.name,
            mime="application/pdf",
            use_container_width=True,
        )
        if pdf_download_col2.button("打开当前对比 PDF", use_container_width=True, key="open_compare_pdf_clicked"):
            ok, message = _open_local_path(compare_pdf_path)
            if ok:
                st.success(message)
            else:
                st.error(message)

    st.subheader("完整对比表")
    compare_view_df, compare_export_df = _render_dataframe_view_controls(
        compare_df,
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
        default_sort_column=metric_key if metric_key in compare_df.columns else "started_at",
        default_descending=higher_is_better if metric_key in compare_df.columns else False,
    )
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
    st.session_state.setdefault("run_queue", [])
    st.session_state.setdefault("active_process", None)
    st.session_state.setdefault("active_run_info", None)
    st.session_state.setdefault("queue_auto_run", False)
    st.session_state.setdefault("last_scheduler_message", "")


def _run_pipeline_from_form(
    *,
    input_csv_upload: Any,
    feature_csv_upload: Any,
    default_pocket_upload: Any,
    default_catalytic_upload: Any,
    default_ligand_upload: Any,
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

    resolved_inputs = {
        "input_csv": None if input_csv is None else str(input_csv),
        "feature_csv": None if feature_csv is None else str(feature_csv),
        "default_pocket_file": None if default_pocket_file is None else str(default_pocket_file),
        "default_catalytic_file": None if default_catalytic_file is None else str(default_catalytic_file),
        "default_ligand_file": None if default_ligand_file is None else str(default_ligand_file),
    }

    command = _build_pipeline_command(
        start_mode=str(form_payload.get("start_mode")),
        input_csv=input_csv,
        feature_csv=feature_csv,
        out_dir=out_dir,
        default_pocket_file=default_pocket_file,
        default_catalytic_file=default_catalytic_file,
        default_ligand_file=default_ligand_file,
        default_antigen_chain=str(form_payload.get("default_antigen_chain") or ""),
        default_nanobody_chain=str(form_payload.get("default_nanobody_chain") or ""),
        top_k=int(form_payload.get("top_k", 3)),
        train_epochs=int(form_payload.get("train_epochs", 20)),
        train_batch_size=int(form_payload.get("train_batch_size", 64)),
        train_val_ratio=float(form_payload.get("train_val_ratio", 0.25)),
        train_early_stopping_patience=int(form_payload.get("train_early_stopping_patience", 8)),
        seed=int(form_payload.get("seed", 42)),
        skip_failed_rows=bool(form_payload.get("skip_failed_rows", True)),
        disable_label_aware_steps=bool(form_payload.get("disable_label_aware_steps", False)),
    )
    pipeline_kwargs = _build_pipeline_kwargs(
        form_payload=form_payload,
        out_dir=out_dir,
        input_csv=input_csv,
        feature_csv=feature_csv,
        default_pocket_file=default_pocket_file,
        default_catalytic_file=default_catalytic_file,
        default_ligand_file=default_ligand_file,
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

        st.subheader("常用参数")
        st.text_input("default_antigen_chain", key="default_antigen_chain")
        st.text_input("default_nanobody_chain", key="default_nanobody_chain")
        st.number_input("top_k", min_value=1, max_value=20, step=1, key="top_k")
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
            run_request = _prepare_run_request_from_form(
                input_csv_upload=input_csv_upload,
                feature_csv_upload=feature_csv_upload,
                default_pocket_upload=default_pocket_upload,
                default_catalytic_upload=default_catalytic_upload,
                default_ligand_upload=default_ligand_upload,
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
            preflight_report = _build_preflight_report(
                start_mode=str(st.session_state.get("start_mode") or "input_csv"),
                input_csv_upload=input_csv_upload,
                feature_csv_upload=feature_csv_upload,
                default_pocket_upload=default_pocket_upload,
                default_catalytic_upload=default_catalytic_upload,
                default_ligand_upload=default_ligand_upload,
                form_payload=_build_form_payload(),
            )
            st.session_state["last_preflight"] = preflight_report
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
            with st.spinner("正在启动后台运行任务，请等待..."):
                run_request = _prepare_run_request_from_form(
                    input_csv_upload=input_csv_upload,
                    feature_csv_upload=feature_csv_upload,
                    default_pocket_upload=default_pocket_upload,
                    default_catalytic_upload=default_catalytic_upload,
                    default_ligand_upload=default_ligand_upload,
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

    tab_summary, tab_qc, tab_compare, tab_history, tab_ranking, tab_pose, tab_report, tab_logs, tab_diag = st.tabs(
        ["摘要", "QC/Warning", "运行对比", "历史", "排名结果", "Pose 结果", "执行报告", "日志", "诊断"]
    )

    with tab_summary:
        _render_summary_metrics(summary)
        notes = summary.get("notes") if isinstance(summary, dict) and isinstance(summary.get("notes"), list) else []
        if notes:
            st.subheader("执行备注")
            for note in notes:
                st.write(f"- {note}")

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
            ranking_preview_rows = st.number_input(
                "每个排名表预览前 N 行",
                min_value=5,
                max_value=500,
                value=20,
                step=5,
                key="ranking_preview_rows",
            )
            ranking_filter_text = str(st.text_input("按 nanobody_id 过滤", key="ranking_filter_text") or "").strip()

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


if __name__ == "__main__":
    main()
