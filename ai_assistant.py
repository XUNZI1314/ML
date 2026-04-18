"""Optional AI-readable report layer for completed ML pipeline runs.

The default provider is local/offline. When ``--provider openai`` is selected,
only compact run summaries and top rows from generated CSV/JSON artifacts are
sent to the API; raw PDB files and full input CSV files are never uploaded.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_OPENAI_MODEL = "gpt-5.4-nano"
DEFAULT_MAX_ROWS = 8
MAX_PROMPT_CHARS = 52000

TOP_CANDIDATE_COLUMNS = [
    "nanobody_id",
    "consensus_rank",
    "rank",
    "consensus_score",
    "final_score",
    "confidence_level",
    "confidence_score",
    "low_confidence_reasons",
    "review_reason_flags",
    "rule_rank",
    "ml_rank",
    "rule_score",
    "ml_score",
    "best_conformer_id",
    "best_pose_id",
    "pocket_hit_fraction",
    "catalytic_hit_fraction",
    "mouth_occlusion_score",
    "ligand_path_block_score",
    "min_distance_to_pocket",
]

COMPARISON_COLUMNS = [
    "winner_nanobody_id",
    "loser_nanobody_id",
    "left_nanobody_id",
    "right_nanobody_id",
    "rank_gap",
    "score_delta",
    "winner_advantages",
    "runner_up_counterpoints",
    "decision_summary",
    "is_close_decision",
]

SUGGESTION_COLUMNS = [
    "nanobody_id",
    "consensus_rank",
    "confidence_level",
    "priority",
    "recommended_action",
    "suggested_action",
    "reason",
    "low_confidence_reasons",
    "review_reason_flags",
]

SENSITIVE_COLUMNS = [
    "nanobody_id",
    "consensus_rank",
    "rank_std",
    "rank_range",
    "top_n_frequency",
    "sensitivity_warning",
    "reason",
]


def _now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _read_csv_rows(path: Path | None, max_rows: int) -> list[dict[str, Any]]:
    if path is None or not path.exists() or not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if isinstance(row, dict):
                rows.append({str(k): "" if v is None else str(v) for k, v in row.items() if k is not None})
            if len(rows) >= max(1, int(max_rows)):
                break
    return rows


def _safe_float(value: Any) -> float | None:
    try:
        out = float(str(value))
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def _format_value(value: Any) -> str:
    if value is None:
        return "N/A"
    text = str(value).strip()
    if not text:
        return "N/A"
    numeric = _safe_float(text)
    if numeric is not None and re.fullmatch(r"[-+]?\d+(\.\d+)?([eE][-+]?\d+)?", text):
        return f"{numeric:.4f}".rstrip("0").rstrip(".")
    return text


def _looks_like_path(text: str) -> bool:
    if not text:
        return False
    if re.match(r"^[A-Za-z]:[\\/]", text):
        return True
    if text.startswith(("/", "\\\\")):
        return True
    return "/" in text or "\\" in text


def _redact_value(value: Any, *, allow_sensitive_paths: bool) -> Any:
    if isinstance(value, dict):
        return {str(k): _redact_value(v, allow_sensitive_paths=allow_sensitive_paths) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_value(v, allow_sensitive_paths=allow_sensitive_paths) for v in value]
    if allow_sensitive_paths or not isinstance(value, str):
        return value
    text = str(value)
    if not _looks_like_path(text):
        return text
    name = Path(text).name or "path"
    return f"<redacted_path:{name}>"


def _resolve_artifact(summary: dict[str, Any], summary_path: Path, key: str) -> Path | None:
    artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
    raw_path = artifacts.get(key)
    if not raw_path:
        return None
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = (summary_path.parent / path).resolve()
    return path if path.exists() else None


def _project_rows(rows: list[dict[str, Any]], preferred_columns: list[str]) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    for row in rows:
        columns = [col for col in preferred_columns if col in row]
        if not columns:
            columns = list(row.keys())[: min(8, len(row))]
        projected.append({col: row.get(col, "") for col in columns})
    return projected


def _markdown_table(rows: list[dict[str, Any]], preferred_columns: list[str], max_rows: int) -> str:
    if not rows:
        return "暂无可展示数据。"
    rows = _project_rows(rows[: max(1, int(max_rows))], preferred_columns)
    columns: list[str] = []
    for row in rows:
        for col in row:
            if col not in columns:
                columns.append(col)
    if not columns:
        return "暂无可展示数据。"

    def cell(value: Any) -> str:
        text = _format_value(value)
        return text.replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def _compact_run_context(
    *,
    summary_path: Path,
    max_rows: int,
    allow_sensitive_paths: bool,
) -> dict[str, Any]:
    summary = _read_json(summary_path)
    max_rows = max(1, int(max_rows))

    artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
    artifact_status = {}
    for key, value in artifacts.items():
        if not value:
            artifact_status[key] = None
            continue
        path = Path(str(value)).expanduser()
        artifact_status[key] = {
            "path": _redact_value(str(path), allow_sensitive_paths=allow_sensitive_paths),
            "exists": path.exists(),
            "size_bytes": int(path.stat().st_size) if path.exists() and path.is_file() else None,
        }

    consensus_rows = _read_csv_rows(
        _resolve_artifact(summary, summary_path, "consensus_ranking_csv"),
        max_rows,
    )
    comparison_rows = _read_csv_rows(
        _resolve_artifact(summary, summary_path, "candidate_pairwise_comparisons_csv"),
        max_rows,
    )
    suggestion_rows = _read_csv_rows(
        _resolve_artifact(summary, summary_path, "experiment_suggestions_csv"),
        max_rows,
    )
    sensitive_rows = _read_csv_rows(
        _resolve_artifact(summary, summary_path, "parameter_sensitivity_sensitive_csv"),
        max_rows,
    )

    commands = summary.get("commands") if isinstance(summary.get("commands"), list) else []
    failed_commands = [
        item
        for item in commands
        if isinstance(item, dict) and int(item.get("returncode", 0) or 0) != 0
    ]

    context = {
        "source_summary_json": _redact_value(str(summary_path), allow_sensitive_paths=allow_sensitive_paths),
        "generated_at": _now_text(),
        "start_mode": summary.get("start_mode"),
        "label_summary": {
            "label_col": summary.get("label_col"),
            "label_valid_count": summary.get("label_valid_count"),
            "label_class_count": summary.get("label_class_count"),
            "label_compare_possible": summary.get("label_compare_possible"),
            "calibration_possible": summary.get("calibration_possible"),
        },
        "ranking_config": summary.get("ranking_config"),
        "notes": summary.get("notes") if isinstance(summary.get("notes"), list) else [],
        "comparison": summary.get("comparison"),
        "improvement": summary.get("improvement"),
        "strategy_optimization": summary.get("strategy_optimization"),
        "top_candidates": _project_rows(consensus_rows, TOP_CANDIDATE_COLUMNS),
        "candidate_comparisons": _project_rows(comparison_rows, COMPARISON_COLUMNS),
        "next_experiment_suggestions": _project_rows(suggestion_rows, SUGGESTION_COLUMNS),
        "sensitive_candidates": _project_rows(sensitive_rows, SENSITIVE_COLUMNS),
        "failed_commands": _redact_value(failed_commands, allow_sensitive_paths=allow_sensitive_paths),
        "command_count": len(commands),
        "artifact_status": artifact_status,
        "privacy": {
            "raw_pdb_uploaded": False,
            "full_input_csv_uploaded": False,
            "absolute_paths_redacted": not bool(allow_sensitive_paths),
        },
    }
    return _redact_value(context, allow_sensitive_paths=allow_sensitive_paths)


def _first_present(row: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return "N/A"


def _build_local_run_summary(context: dict[str, Any], max_rows: int) -> str:
    label_summary = context.get("label_summary") if isinstance(context.get("label_summary"), dict) else {}
    ranking_config = context.get("ranking_config") if isinstance(context.get("ranking_config"), dict) else {}
    notes = context.get("notes") if isinstance(context.get("notes"), list) else []
    top_candidates = context.get("top_candidates") if isinstance(context.get("top_candidates"), list) else []
    suggestions = context.get("next_experiment_suggestions") if isinstance(context.get("next_experiment_suggestions"), list) else []
    sensitive = context.get("sensitive_candidates") if isinstance(context.get("sensitive_candidates"), list) else []

    lines = [
        "# AI Run Summary",
        "",
        "生成方式：本地离线摘要。该文件只解释已有 pipeline 输出，不改变模型分数或排序。",
        "",
        "## 运行概览",
        "",
        f"- Start mode: `{context.get('start_mode', 'N/A')}`",
        f"- Label valid rows: `{label_summary.get('label_valid_count', 'N/A')}`",
        f"- Label classes: `{label_summary.get('label_class_count', 'N/A')}`",
        f"- Calibration possible: `{label_summary.get('calibration_possible', 'N/A')}`",
        f"- Top K: `{ranking_config.get('top_k', 'N/A')}`",
        f"- Pocket overwide penalty: `{ranking_config.get('pocket_overwide_penalty_weight', 'N/A')}`",
        "",
    ]
    if notes:
        lines.extend(["## 关键备注", ""])
        for note in notes[:8]:
            lines.append(f"- {note}")
        lines.append("")

    lines.extend(
        [
            "## Top Candidates",
            "",
            _markdown_table(top_candidates, TOP_CANDIDATE_COLUMNS, max_rows),
            "",
            "## 参数敏感或需要复核的候选",
            "",
            _markdown_table(sensitive, SENSITIVE_COLUMNS, max_rows),
            "",
            "## 下一轮实验建议",
            "",
            _markdown_table(suggestions, SUGGESTION_COLUMNS, max_rows),
            "",
            "## 使用边界",
            "",
            "- 该摘要不会替代实验验证，也不会重新计算结构特征。",
            "- 若启用在线 OpenAI provider，默认也只发送压缩后的 summary 和表格前几行，不发送原始 PDB 或完整 CSV。",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _build_local_candidate_explanation(context: dict[str, Any], max_rows: int) -> str:
    top_candidates = context.get("top_candidates") if isinstance(context.get("top_candidates"), list) else []
    comparisons = context.get("candidate_comparisons") if isinstance(context.get("candidate_comparisons"), list) else []

    lines = [
        "# Top Candidate Explanation",
        "",
        "该解释基于 `consensus_ranking.csv` 和 candidate comparison 产物生成，不改变任何排序。",
        "",
        "## 候选概览",
        "",
        _markdown_table(top_candidates, TOP_CANDIDATE_COLUMNS, max_rows),
        "",
        "## 相邻候选对比",
        "",
        _markdown_table(comparisons, COMPARISON_COLUMNS, max_rows),
        "",
        "## 解读规则",
        "",
    ]
    if top_candidates:
        first = top_candidates[0] if isinstance(top_candidates[0], dict) else {}
        candidate_id = _first_present(first, ["nanobody_id", "candidate_id", "id"])
        confidence = _first_present(first, ["confidence_level", "confidence_score"])
        low_reasons = _first_present(first, ["low_confidence_reasons", "review_reason_flags"])
        lines.append(f"- 当前首位候选是 `{candidate_id}`，可信度字段为 `{confidence}`。")
        lines.append(f"- 若出现低可信原因，应优先查看 `{low_reasons}` 对应的复核项。")
    else:
        lines.append("- 当前没有读取到 consensus ranking 行，需先确认 pipeline 是否完成并生成排名 CSV。")
    lines.extend(
        [
            "- 分差很小或敏感性较高时，不建议只看第一名，应结合候选对比和下一轮实验建议一起决策。",
            "- 若 Rule 与 ML 排名分歧明显，应优先补实验标签或检查几何 proxy 输入质量。",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _build_local_failure_diagnosis(context: dict[str, Any], max_rows: int) -> str:
    failed_commands = context.get("failed_commands") if isinstance(context.get("failed_commands"), list) else []
    artifact_status = context.get("artifact_status") if isinstance(context.get("artifact_status"), dict) else {}
    missing_artifacts = [
        {"artifact": key, "exists": item.get("exists") if isinstance(item, dict) else item}
        for key, item in artifact_status.items()
        if item is not None and isinstance(item, dict) and not bool(item.get("exists"))
    ]

    lines = [
        "# Failure Diagnosis",
        "",
        "该诊断只读取 pipeline summary 中记录的命令、stderr 摘要和产物状态。",
        "",
    ]
    if failed_commands:
        lines.extend(
            [
                "## 失败命令",
                "",
                _markdown_table(
                    [
                        {
                            "name": item.get("name"),
                            "returncode": item.get("returncode"),
                            "stderr_tail": item.get("stderr_tail"),
                        }
                        for item in failed_commands
                        if isinstance(item, dict)
                    ],
                    ["name", "returncode", "stderr_tail"],
                    max_rows,
                ),
                "",
                "## 建议处理",
                "",
                "- 先从第一条失败命令开始处理，不要同时修改多个阶段。",
                "- 若 stderr 指向缺少依赖，先修复环境；若指向输入列或文件路径，先修复 CSV 和路径。",
                "- 修复后重新运行同一个输出目录，便于对比产物差异。",
            ]
        )
    else:
        lines.extend(
            [
                "## 未记录失败命令",
                "",
                "- 当前 summary 中没有失败命令记录。如果本地页面显示失败，但没有 summary，请查看该运行目录下的 `app_run_stderr.txt`。",
                "- 如果 pipeline 成功但缺少某些产物，通常是对应可选步骤未启用，或 label 条件不足导致 label-aware 分析被跳过。",
            ]
        )
    if missing_artifacts:
        lines.extend(
            [
                "",
                "## 缺失产物",
                "",
                _markdown_table(missing_artifacts, ["artifact", "exists"], max_rows),
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _extract_response_text(payload: dict[str, Any]) -> str:
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    parts: list[str] = []
    output = payload.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
    return "\n\n".join(parts).strip()


def _call_openai_responses_api(
    *,
    prompt: str,
    model: str,
    timeout_sec: float,
) -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    payload = {
        "model": str(model),
        "input": prompt,
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(timeout_sec)) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API HTTP {exc.code}: {body[:1000]}") from exc
    text = _extract_response_text(response_payload)
    if not text:
        raise RuntimeError("OpenAI API response did not contain output text")
    return text.strip() + "\n"


def _build_prompt(context: dict[str, Any], task: str) -> str:
    context_text = json.dumps(context, ensure_ascii=False, indent=2)
    if len(context_text) > MAX_PROMPT_CHARS:
        context_text = context_text[:MAX_PROMPT_CHARS] + "\n...<truncated>"
    return (
        "你是一名严谨的生物信息学 ML 软件解释助手。"
        "请只解释给定 JSON 上下文中的 pipeline 输出，不要声称你重新训练、重新评分或完成实验验证。"
        "不要要求用户上传原始 PDB；如果信息不足，要明确说明缺口。"
        "请用中文 Markdown 输出，结构清晰、适合本地软件展示。\n\n"
        f"任务：{task}\n\n"
        f"上下文 JSON：\n```json\n{context_text}\n```"
    )


def _generate_text(
    *,
    provider: str,
    model: str,
    context: dict[str, Any],
    task: str,
    fallback_text: str,
    timeout_sec: float,
) -> tuple[str, str, str | None]:
    provider = str(provider or "none").strip().lower()
    if provider != "openai":
        return fallback_text, "local", None
    try:
        prompt = _build_prompt(context, task)
        return _call_openai_responses_api(prompt=prompt, model=model, timeout_sec=timeout_sec), "openai", None
    except Exception as exc:
        note = f"\n\n> OpenAI provider 不可用，已回退到本地离线摘要。原因：{exc}\n"
        return fallback_text.rstrip() + note, "local_fallback", str(exc)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate optional AI-readable summaries from pipeline outputs")
    parser.add_argument("--summary_json", required=True, help="Path to recommended_pipeline_summary.json")
    parser.add_argument("--out_dir", default=None, help="Output directory for AI summaries")
    parser.add_argument(
        "--provider",
        choices=["none", "openai"],
        default=os.environ.get("ML_AI_PROVIDER", "none"),
        help="AI provider. Default is local/offline none.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("ML_AI_MODEL", DEFAULT_OPENAI_MODEL),
        help="OpenAI model name when --provider openai is used.",
    )
    parser.add_argument("--max_rows", type=int, default=DEFAULT_MAX_ROWS)
    parser.add_argument("--timeout_sec", type=float, default=60.0)
    parser.add_argument("--allow_sensitive_paths", action="store_true")
    parser.add_argument(
        "--action",
        choices=["all", "summarize_run", "explain_top_candidates", "diagnose_failure"],
        default="all",
    )
    return parser


def run_ai_assistant(
    *,
    summary_json: str | Path,
    out_dir: str | Path | None = None,
    provider: str = "none",
    model: str = DEFAULT_OPENAI_MODEL,
    max_rows: int = DEFAULT_MAX_ROWS,
    timeout_sec: float = 60.0,
    allow_sensitive_paths: bool = False,
    action: str = "all",
) -> dict[str, Any]:
    summary_path = Path(summary_json).expanduser().resolve()
    if not summary_path.exists():
        raise FileNotFoundError(f"summary_json not found: {summary_path}")
    out_path = Path(out_dir).expanduser().resolve() if out_dir is not None else summary_path.parent / "ai_outputs"
    out_path.mkdir(parents=True, exist_ok=True)

    context = _compact_run_context(
        summary_path=summary_path,
        max_rows=max_rows,
        allow_sensitive_paths=allow_sensitive_paths,
    )
    selected_actions = (
        ["summarize_run", "explain_top_candidates", "diagnose_failure"]
        if action == "all"
        else [str(action)]
    )

    fallback_by_action = {
        "summarize_run": _build_local_run_summary(context, max_rows),
        "explain_top_candidates": _build_local_candidate_explanation(context, max_rows),
        "diagnose_failure": _build_local_failure_diagnosis(context, max_rows),
    }
    task_by_action = {
        "summarize_run": "生成一份运行摘要，突出 top candidates、可信度提醒和下一步实验建议。",
        "explain_top_candidates": "解释前排候选为什么值得优先复核，并指出分差小、低可信或 Rule/ML 分歧风险。",
        "diagnose_failure": "根据命令记录、stderr tail 和产物状态，生成失败诊断和下一步排查建议。",
    }
    file_by_action = {
        "summarize_run": out_path / "ai_run_summary.md",
        "explain_top_candidates": out_path / "ai_top_candidates_explanation.md",
        "diagnose_failure": out_path / "ai_failure_diagnosis.md",
    }

    provider_results: dict[str, str] = {}
    errors: dict[str, str] = {}
    for item in selected_actions:
        text, used_provider, error = _generate_text(
            provider=provider,
            model=model,
            context=context,
            task=task_by_action[item],
            fallback_text=fallback_by_action[item],
            timeout_sec=timeout_sec,
        )
        _write_text(file_by_action[item], text)
        provider_results[item] = used_provider
        if error:
            errors[item] = error

    artifact_keys = {
        "summarize_run": "ai_run_summary_md",
        "explain_top_candidates": "ai_top_candidates_explanation_md",
        "diagnose_failure": "ai_failure_diagnosis_md",
    }
    artifacts_out = {
        artifact_keys[item]: str(file_by_action[item])
        for item in selected_actions
        if file_by_action[item].exists()
    }

    summary_out = {
        "created_at": _now_text(),
        "source_summary_json": str(summary_path),
        "out_dir": str(out_path),
        "provider_requested": str(provider),
        "provider_used_by_action": provider_results,
        "model": str(model),
        "max_rows": int(max_rows),
        "privacy": context.get("privacy"),
        "errors": errors,
        "artifacts": artifacts_out,
    }
    summary_out_path = out_path / "ai_assistant_summary.json"
    summary_out["artifacts"]["ai_assistant_summary_json"] = str(summary_out_path)
    summary_out_path.write_text(json.dumps(summary_out, ensure_ascii=True, indent=2), encoding="utf-8")
    return summary_out


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = run_ai_assistant(
        summary_json=args.summary_json,
        out_dir=args.out_dir,
        provider=args.provider,
        model=args.model,
        max_rows=int(args.max_rows),
        timeout_sec=float(args.timeout_sec),
        allow_sensitive_paths=bool(args.allow_sensitive_paths),
        action=str(args.action),
    )
    print(f"AI assistant outputs: {summary['out_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
