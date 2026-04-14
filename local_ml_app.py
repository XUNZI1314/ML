from __future__ import annotations

import json
import os
import re
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
from PIL import Image

from app_metadata import APP_NAME, APP_RELEASE_CHANNEL, APP_VERSION

REPO_ROOT = Path(__file__).resolve().parent
LOCAL_APP_RUN_ROOT = REPO_ROOT / "local_app_runs"
PARAM_TEMPLATE_ROOT = LOCAL_APP_RUN_ROOT / "parameter_templates"
APP_RUN_METADATA_NAME = "app_run_metadata.json"
APP_STDOUT_NAME = "app_stdout.log"
APP_STDERR_NAME = "app_stderr.log"
LOCAL_APP_RUN_ROOT.mkdir(parents=True, exist_ok=True)
PARAM_TEMPLATE_ROOT.mkdir(parents=True, exist_ok=True)

START_MODE_LABELS = {
    "input_csv": "从 input_csv 开始",
    "feature_csv": "从 pose_features.csv 开始",
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
        history_labels = [""] + [str(item["label"]) for item in history_records]
        st.selectbox(
            "最近运行记录",
            options=history_labels,
            format_func=lambda x: "请选择历史记录" if not x else x,
            key="selected_history_label",
        )
        load_history_clicked = st.button("载入所选历史结果", use_container_width=True)

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

    tab_summary, tab_history, tab_ranking, tab_pose, tab_report, tab_logs, tab_diag = st.tabs(
        ["摘要", "历史", "排名结果", "Pose 结果", "执行报告", "日志", "诊断"]
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
        bundle_col1, bundle_col2 = st.columns(2)
        create_bundle_clicked = bundle_col1.button(
            "生成当前运行汇总包",
            use_container_width=True,
            disabled=not bool(last_run_dir),
        )
        open_bundle_dir_clicked = bundle_col2.button(
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

        st.subheader("关键产物下载")
        _render_download_buttons(
            summary if isinstance(summary, dict) else None,
            metadata if isinstance(metadata, dict) else None,
        )

        if isinstance(summary, dict):
            with st.expander("查看完整执行摘要 JSON"):
                st.json(summary)

    with tab_history:
        if not history_records:
            st.info("当前还没有历史运行记录。")
        else:
            st.subheader("最近运行记录")
            st.dataframe(_build_history_table(history_records), use_container_width=True, hide_index=True)
            st.caption("可在左侧“运行历史”区域加载某次历史结果，并恢复对应表单参数。")

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
                    st.caption(f"共 {len(ml_df)} 条，当前展示前 {min(int(ranking_preview_rows), len(ml_df))} 条。")
                    st.dataframe(ml_df.head(int(ranking_preview_rows)), use_container_width=True)

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
                    st.caption(f"共 {len(rule_df)} 条，当前展示前 {min(int(ranking_preview_rows), len(rule_df))} 条。")
                    st.dataframe(rule_df.head(int(ranking_preview_rows)), use_container_width=True)

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
            pose_sort_desc = st.checkbox("按 pred_prob 降序显示", value=True, key="pose_sort_desc")
            if pose_pred_csv is not None and pose_pred_csv.exists():
                pose_df = _load_csv(pose_pred_csv)
                if pose_df is not None:
                    if pose_sort_desc and "pred_prob" in pose_df.columns:
                        pose_df = pose_df.sort_values("pred_prob", ascending=False, kind="stable")
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
                    st.caption(f"共 {len(pose_df)} 条，当前展示前 {min(int(pose_preview_rows), len(pose_df))} 条。")
                    st.dataframe(pose_df.head(int(pose_preview_rows)), use_container_width=True)
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
