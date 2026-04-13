from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parent
LOCAL_APP_RUN_ROOT = REPO_ROOT / "local_app_runs"
LOCAL_APP_RUN_ROOT.mkdir(parents=True, exist_ok=True)


def _slugify_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    text = text.strip("._")
    return text or datetime.now().strftime("run_%Y%m%d_%H%M%S")


def _default_run_name() -> str:
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def _save_uploaded_file(uploaded_file: Any, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    out_path = dst_dir / str(uploaded_file.name)
    out_path.write_bytes(bytes(uploaded_file.getbuffer()))
    return out_path


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


def _collect_download_artifacts(summary: dict[str, Any] | None) -> list[tuple[str, Path]]:
    if not isinstance(summary, dict):
        return []
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, dict):
        return []

    ordered_keys = [
        "execution_report_md",
        "rule_ranking_csv",
        "ml_ranking_csv",
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

    out: list[tuple[str, Path]] = []
    for key in ordered_keys:
        value = artifacts.get(key)
        if not value:
            continue
        path = Path(str(value))
        if path.exists():
            out.append((key, path))
    return out


def _render_download_buttons(summary: dict[str, Any] | None) -> None:
    artifacts = _collect_download_artifacts(summary)
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


def _initialize_state() -> None:
    st.session_state.setdefault("last_run_dir", None)
    st.session_state.setdefault("last_summary", None)
    st.session_state.setdefault("last_command", None)
    st.session_state.setdefault("last_stdout", "")
    st.session_state.setdefault("last_stderr", "")
    st.session_state.setdefault("last_error", "")


def _run_pipeline_from_form(
    *,
    run_name: str,
    start_mode: str,
    input_csv_upload: Any,
    feature_csv_upload: Any,
    input_csv_local_path: str,
    feature_csv_local_path: str,
    default_pocket_upload: Any,
    default_catalytic_upload: Any,
    default_ligand_upload: Any,
    default_pocket_local_path: str,
    default_catalytic_local_path: str,
    default_ligand_local_path: str,
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
) -> None:
    safe_run_name = _slugify_name(run_name)
    run_root = LOCAL_APP_RUN_ROOT / safe_run_name
    input_dir = run_root / "inputs"
    out_dir = run_root / "outputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_csv = _resolve_input_file(
        uploaded_file=input_csv_upload,
        local_path_text=input_csv_local_path,
        dst_dir=input_dir,
        required=start_mode == "input_csv",
        label="input_csv",
    )
    feature_csv = _resolve_input_file(
        uploaded_file=feature_csv_upload,
        local_path_text=feature_csv_local_path,
        dst_dir=input_dir,
        required=start_mode == "feature_csv",
        label="feature_csv",
    )
    default_pocket_file = _resolve_input_file(
        uploaded_file=default_pocket_upload,
        local_path_text=default_pocket_local_path,
        dst_dir=input_dir,
        required=False,
        label="default_pocket_file",
    )
    default_catalytic_file = _resolve_input_file(
        uploaded_file=default_catalytic_upload,
        local_path_text=default_catalytic_local_path,
        dst_dir=input_dir,
        required=False,
        label="default_catalytic_file",
    )
    default_ligand_file = _resolve_input_file(
        uploaded_file=default_ligand_upload,
        local_path_text=default_ligand_local_path,
        dst_dir=input_dir,
        required=False,
        label="default_ligand_file",
    )

    command = _build_pipeline_command(
        start_mode=start_mode,
        input_csv=input_csv,
        feature_csv=feature_csv,
        out_dir=out_dir,
        default_pocket_file=default_pocket_file,
        default_catalytic_file=default_catalytic_file,
        default_ligand_file=default_ligand_file,
        default_antigen_chain=default_antigen_chain,
        default_nanobody_chain=default_nanobody_chain,
        top_k=top_k,
        train_epochs=train_epochs,
        train_batch_size=train_batch_size,
        train_val_ratio=train_val_ratio,
        train_early_stopping_patience=train_early_stopping_patience,
        seed=seed,
        skip_failed_rows=skip_failed_rows,
        disable_label_aware_steps=disable_label_aware_steps,
    )

    proc = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )

    summary_path = out_dir / "recommended_pipeline_summary.json"
    summary = _load_json(summary_path)

    st.session_state["last_run_dir"] = str(run_root)
    st.session_state["last_command"] = command
    st.session_state["last_stdout"] = proc.stdout or ""
    st.session_state["last_stderr"] = proc.stderr or ""
    st.session_state["last_summary"] = summary
    st.session_state["last_error"] = ""

    if proc.returncode != 0:
        st.session_state["last_error"] = (
            f"Pipeline failed with rc={proc.returncode}. "
            f"Check stderr and output directory: {out_dir}"
        )
        raise RuntimeError(st.session_state["last_error"])


def main() -> None:
    st.set_page_config(page_title="ML Local Runner", layout="wide")
    _initialize_state()

    st.title("ML 本地交互运行器")
    st.caption("基于当前仓库现有 CLI 流程构建的最小本地交互壳，优先解决文件上传、参数填写、结果预览和导出。")

    with st.sidebar:
        st.header("运行配置")
        start_mode_label = st.radio(
            "启动方式",
            options=["从 input_csv 开始", "从 pose_features.csv 开始"],
            index=0,
        )
        start_mode = "input_csv" if start_mode_label == "从 input_csv 开始" else "feature_csv"

        run_name = st.text_input("运行名称", value=_default_run_name())
        st.caption("每次运行都会写入 local_app_runs/<运行名称>/")

        st.subheader("主输入")
        input_csv_upload = None
        feature_csv_upload = None
        input_csv_local_path = ""
        feature_csv_local_path = ""

        if start_mode == "input_csv":
            input_csv_upload = st.file_uploader("上传 input_pose_table.csv", type=["csv"], key="input_csv_upload")
            input_csv_local_path = st.text_input("或填写 input_csv 本地路径", value="")
            st.caption("如果 CSV 里的 pdb_path / pocket_file 等已经是本机可访问路径，应用会直接沿用。")
        else:
            feature_csv_upload = st.file_uploader("上传 pose_features.csv", type=["csv"], key="feature_csv_upload")
            feature_csv_local_path = st.text_input("或填写 feature_csv 本地路径", value="")

        st.subheader("可选默认文件")
        default_pocket_upload = st.file_uploader("上传 default pocket file", key="default_pocket_upload")
        default_pocket_local_path = st.text_input("或填写 default_pocket_file 本地路径", value="")
        default_catalytic_upload = st.file_uploader("上传 default catalytic file", key="default_catalytic_upload")
        default_catalytic_local_path = st.text_input("或填写 default_catalytic_file 本地路径", value="")
        default_ligand_upload = st.file_uploader("上传 default ligand file", key="default_ligand_upload")
        default_ligand_local_path = st.text_input("或填写 default_ligand_file 本地路径", value="")

        st.subheader("常用参数")
        default_antigen_chain = st.text_input("default_antigen_chain", value="")
        default_nanobody_chain = st.text_input("default_nanobody_chain", value="")
        top_k = st.number_input("top_k", min_value=1, max_value=20, value=3, step=1)
        train_epochs = st.number_input("train_epochs", min_value=1, max_value=2000, value=20, step=1)
        train_batch_size = st.number_input("train_batch_size", min_value=1, max_value=4096, value=64, step=1)
        train_val_ratio = st.number_input("train_val_ratio", min_value=0.05, max_value=0.95, value=0.25, step=0.05)
        train_early_stopping_patience = st.number_input(
            "train_early_stopping_patience",
            min_value=1,
            max_value=200,
            value=8,
            step=1,
        )
        seed = st.number_input("seed", min_value=0, max_value=999999, value=42, step=1)
        skip_failed_rows = st.checkbox("skip_failed_rows", value=True)
        disable_label_aware_steps = st.checkbox("disable_label_aware_steps", value=False)

        run_clicked = st.button("运行推荐流程", use_container_width=True, type="primary")

    if run_clicked:
        try:
            with st.spinner("正在执行 pipeline，这一步会调用现有 Python CLI，请等待..."):
                _run_pipeline_from_form(
                    run_name=run_name,
                    start_mode=start_mode,
                    input_csv_upload=input_csv_upload,
                    feature_csv_upload=feature_csv_upload,
                    input_csv_local_path=input_csv_local_path,
                    feature_csv_local_path=feature_csv_local_path,
                    default_pocket_upload=default_pocket_upload,
                    default_catalytic_upload=default_catalytic_upload,
                    default_ligand_upload=default_ligand_upload,
                    default_pocket_local_path=default_pocket_local_path,
                    default_catalytic_local_path=default_catalytic_local_path,
                    default_ligand_local_path=default_ligand_local_path,
                    default_antigen_chain=default_antigen_chain,
                    default_nanobody_chain=default_nanobody_chain,
                    top_k=int(top_k),
                    train_epochs=int(train_epochs),
                    train_batch_size=int(train_batch_size),
                    train_val_ratio=float(train_val_ratio),
                    train_early_stopping_patience=int(train_early_stopping_patience),
                    seed=int(seed),
                    skip_failed_rows=bool(skip_failed_rows),
                    disable_label_aware_steps=bool(disable_label_aware_steps),
                )
            st.success("Pipeline 执行完成。")
        except Exception as exc:
            st.error(str(exc))

    summary = st.session_state.get("last_summary")
    last_run_dir = st.session_state.get("last_run_dir")
    last_command = st.session_state.get("last_command")

    if last_run_dir:
        st.info(f"最近一次运行目录: {last_run_dir}")
    if last_command:
        st.code(" ".join(last_command), language="bash")

    tab_summary, tab_ranking, tab_pose, tab_report, tab_logs = st.tabs(
        ["摘要", "排名结果", "Pose 结果", "执行报告", "日志"]
    )

    with tab_summary:
        _render_summary_metrics(summary)
        notes = summary.get("notes") if isinstance(summary, dict) and isinstance(summary.get("notes"), list) else []
        if notes:
            st.subheader("执行备注")
            for note in notes:
                st.write(f"- {note}")

        st.subheader("关键产物下载")
        _render_download_buttons(summary)

        if isinstance(summary, dict):
            st.subheader("完整执行摘要 JSON")
            st.json(summary)

    with tab_ranking:
        if not isinstance(summary, dict):
            st.info("先运行一次 pipeline，再查看排名结果。")
        else:
            artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
            ml_ranking_csv = Path(str(artifacts.get("ml_ranking_csv"))) if artifacts.get("ml_ranking_csv") else None
            rule_ranking_csv = Path(str(artifacts.get("rule_ranking_csv"))) if artifacts.get("rule_ranking_csv") else None

            if ml_ranking_csv is not None and ml_ranking_csv.exists():
                st.subheader("ML 排名")
                ml_df = _load_csv(ml_ranking_csv)
                if ml_df is not None:
                    st.dataframe(ml_df.head(50), use_container_width=True)

            if rule_ranking_csv is not None and rule_ranking_csv.exists():
                st.subheader("Rule 排名")
                rule_df = _load_csv(rule_ranking_csv)
                if rule_df is not None:
                    st.dataframe(rule_df.head(50), use_container_width=True)

    with tab_pose:
        if not isinstance(summary, dict):
            st.info("先运行一次 pipeline，再查看 pose 结果。")
        else:
            artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
            out_dir_text = summary.get("out_dir")
            pose_pred_csv = None
            if out_dir_text:
                pose_pred_csv = Path(str(out_dir_text)) / "model_outputs" / "pose_predictions.csv"
            if pose_pred_csv is not None and pose_pred_csv.exists():
                pose_df = _load_csv(pose_pred_csv)
                if pose_df is not None:
                    st.dataframe(pose_df.head(100), use_container_width=True)
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


if __name__ == "__main__":
    main()
