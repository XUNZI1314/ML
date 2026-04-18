from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any


def _load_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.exists():
        return None
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _artifact_path(summary: dict[str, Any] | None, key: str) -> Path | None:
    if not isinstance(summary, dict):
        return None
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, dict):
        return None
    value = artifacts.get(key)
    if not value:
        return None
    path = Path(str(value))
    return path if path.exists() else None


def _read_csv_rows(path: str | Path | None, *, limit: int = 5) -> list[dict[str, str]]:
    if path is None:
        return []
    candidate = Path(path)
    if not candidate.exists():
        return []
    rows: list[dict[str, str]] = []
    try:
        with candidate.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append({str(k): "" if v is None else str(v) for k, v in row.items()})
                if len(rows) >= int(limit):
                    break
    except OSError:
        return []
    return rows


def _clean_text(value: Any, default: str = "N/A") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _format_float(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def _format_percent(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def _escape_md_cell(value: Any) -> str:
    return _clean_text(value).replace("|", "\\|").replace("\n", " ")


def _escape_html(value: Any) -> str:
    return html.escape(_clean_text(value), quote=True)


def _relative_or_absolute(path: str | Path | None, *, base_dir: Path | None = None) -> str:
    if path is None:
        return "N/A"
    candidate = Path(path)
    if base_dir is not None:
        try:
            return str(candidate.resolve().relative_to(base_dir.resolve()))
        except (OSError, RuntimeError, ValueError):
            pass
    return str(candidate)


def _load_artifact_json(summary: dict[str, Any] | None, key: str) -> dict[str, Any] | None:
    return _load_json(_artifact_path(summary, key))


def _candidate_highlight(
    batch_summary: dict[str, Any] | None,
    key: str,
) -> dict[str, Any] | None:
    if not isinstance(batch_summary, dict):
        return None
    highlights = batch_summary.get("candidate_highlights")
    if not isinstance(highlights, dict):
        return None
    payload = highlights.get(key)
    return payload if isinstance(payload, dict) else None


def _path_href(path: str | Path | None, *, base_dir: Path) -> str:
    if path is None:
        return "#"
    relative = _relative_or_absolute(path, base_dir=base_dir)
    return Path(relative).as_posix()


def write_demo_readme(
    *,
    out_path: str | Path,
    feature_csv: str | Path,
    override_csv: str | Path,
    summary: dict[str, Any] | None,
    manifest_json: str | Path | None = None,
    overview_html: str | Path | None = None,
    interpretation_md: str | Path | None = None,
    real_data_starter_dir: str | Path | None = None,
) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    artifacts = summary.get("artifacts") if isinstance(summary, dict) and isinstance(summary.get("artifacts"), dict) else {}
    base_dir = out.parent

    lines = [
        "# ML Demo Run",
        "",
        "This is a deterministic synthetic demo. It is meant for workflow demonstration, not biological validation.",
        "",
        "## Demo Inputs",
        "",
        f"- Feature CSV: `{_relative_or_absolute(feature_csv, base_dir=base_dir)}`",
        f"- Experiment override CSV: `{_relative_or_absolute(override_csv, base_dir=base_dir)}`",
    ]
    if manifest_json:
        lines.append(f"- Demo manifest: `{_relative_or_absolute(manifest_json, base_dir=base_dir)}`")

    lines.extend(["", "## Read First", ""])
    if overview_html:
        lines.append(
            f"- `{_relative_or_absolute(overview_html, base_dir=base_dir)}`: browser-friendly demo overview"
        )
    if interpretation_md:
        lines.append(
            f"- `{_relative_or_absolute(interpretation_md, base_dir=base_dir)}`: demo result interpretation guide"
        )
    if real_data_starter_dir:
        starter_readme = Path(real_data_starter_dir) / "README_REAL_DATA_STARTER.md"
        mini_readme = Path(real_data_starter_dir) / "MINI_PDB_EXAMPLE" / "README_MINI_PDB_EXAMPLE.md"
        lines.append(
            f"- `{_relative_or_absolute(starter_readme, base_dir=base_dir)}`: templates for replacing demo data with real project data"
        )
        lines.append(
            f"- `{_relative_or_absolute(mini_readme, base_dir=base_dir)}`: runnable toy PDB example for checking the raw input_csv path"
        )

    read_order = [
        ("recommended_pipeline_summary_json", "Full machine-readable run summary"),
        ("execution_report_md", "Pipeline execution report"),
        ("batch_decision_summary_md", "One-page run decision summary"),
        ("validation_evidence_report_md", "Synthetic validation evidence audit"),
        ("consensus_ranking_csv", "Final consensus ranking"),
        ("candidate_report_index_html", "Candidate report card index"),
        ("score_explanation_cards_html", "Score explanation cards"),
    ]
    for key, label in read_order:
        value = artifacts.get(key)
        if value:
            lines.append(f"- `{_relative_or_absolute(value, base_dir=base_dir)}`: {label}")
    if not summary:
        lines.append("- Pipeline has not finished yet. Refresh this file after the demo run completes.")

    lines.extend(
        [
            "",
            "## Important Boundary",
            "",
            "- Demo validation labels are generated from synthetic proxy signals.",
            "- Do not cite demo labels as wet-lab evidence.",
            "- Use this demo to check installation, UI flow, export files and explanation reports.",
            "",
        ]
    )
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_demo_overview_html(
    *,
    out_path: str | Path,
    feature_csv: str | Path,
    override_csv: str | Path,
    summary: dict[str, Any] | None,
    manifest_json: str | Path | None = None,
    readme_md: str | Path | None = None,
    interpretation_md: str | Path | None = None,
    real_data_starter_dir: str | Path | None = None,
) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    base_dir = out.parent

    batch_summary = _load_artifact_json(summary, "batch_decision_summary_json")
    validation_summary = _load_artifact_json(summary, "validation_evidence_summary_json")
    quality_summary = _load_artifact_json(summary, "quality_gate_summary_json")
    consensus_rows = _read_csv_rows(_artifact_path(summary, "consensus_ranking_csv"), limit=5)

    batch_decision = batch_summary.get("batch_decision", {}) if isinstance(batch_summary, dict) else {}
    quality_gate = batch_summary.get("quality_gate", {}) if isinstance(batch_summary, dict) else {}
    validation_block = batch_summary.get("validation_evidence", {}) if isinstance(batch_summary, dict) else {}
    if not isinstance(batch_decision, dict):
        batch_decision = {}
    if not isinstance(quality_gate, dict):
        quality_gate = {}
    if not isinstance(validation_block, dict):
        validation_block = {}
    if not quality_gate and isinstance(quality_summary, dict):
        quality_gate = quality_summary
    if not validation_block and isinstance(validation_summary, dict):
        validation_block = validation_summary

    best_candidate = _candidate_highlight(batch_summary, "best_candidate")
    next_candidate = _candidate_highlight(batch_summary, "next_experiment_candidate")
    risk_candidate = _candidate_highlight(batch_summary, "highest_risk_candidate")

    artifacts = summary.get("artifacts") if isinstance(summary, dict) and isinstance(summary.get("artifacts"), dict) else {}
    artifact_links = [
        ("Batch decision", artifacts.get("batch_decision_summary_md")),
        ("Candidate report cards", artifacts.get("candidate_report_index_html")),
        ("Score explanation cards", artifacts.get("score_explanation_cards_html")),
        ("Consensus ranking CSV", artifacts.get("consensus_ranking_csv")),
        ("Validation evidence audit", artifacts.get("validation_evidence_report_md")),
        ("Pipeline report", artifacts.get("execution_report_md")),
        ("Demo interpretation", interpretation_md),
        ("Demo README", readme_md),
        (
            "Real data starter kit",
            (Path(real_data_starter_dir) / "README_REAL_DATA_STARTER.md") if real_data_starter_dir else None,
        ),
        (
            "Runnable mini PDB example",
            (Path(real_data_starter_dir) / "MINI_PDB_EXAMPLE" / "README_MINI_PDB_EXAMPLE.md")
            if real_data_starter_dir
            else None,
        ),
    ]

    candidate_rows_html = []
    for row in consensus_rows:
        candidate_rows_html.append(
            "<tr>"
            f"<td>{_escape_html(row.get('consensus_rank') or row.get('rank'))}</td>"
            f"<td>{_escape_html(row.get('nanobody_id'))}</td>"
            f"<td>{_escape_html(_format_float(row.get('consensus_score')))}</td>"
            f"<td>{_escape_html(row.get('confidence_level'))}</td>"
            f"<td>{_escape_html(row.get('decision_tier'))}</td>"
            "</tr>"
        )
    if not candidate_rows_html:
        candidate_rows_html.append(
            "<tr><td colspan='5'>Pipeline output is not ready yet. Refresh after the demo run completes.</td></tr>"
        )

    links_html = []
    for label, path in artifact_links:
        if not path:
            continue
        href = _path_href(path, base_dir=base_dir)
        links_html.append(f"<a class='link-card' href='{_escape_html(href)}'>{_escape_html(label)}</a>")

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ML Demo Overview</title>
  <style>
    :root {{
      --ink: #172018;
      --muted: #526057;
      --paper: #fff9ec;
      --panel: rgba(255, 255, 255, 0.82);
      --line: rgba(23, 32, 24, 0.14);
      --accent: #c85f2e;
      --accent-2: #2e6f5d;
      --gold: #d8a032;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at 10% 5%, rgba(216, 160, 50, 0.24), transparent 34%),
        radial-gradient(circle at 85% 12%, rgba(46, 111, 93, 0.22), transparent 30%),
        linear-gradient(135deg, #fff4d7 0%, #f7efe1 45%, #e8f1e9 100%);
      font-family: Georgia, "Times New Roman", "Noto Serif SC", serif;
      line-height: 1.55;
    }}
    main {{ width: min(1160px, calc(100% - 36px)); margin: 0 auto; padding: 34px 0 52px; }}
    .hero {{
      border: 1px solid var(--line);
      border-radius: 28px;
      padding: 34px;
      background: rgba(255, 249, 236, 0.72);
      box-shadow: 0 24px 70px rgba(35, 45, 34, 0.13);
    }}
    .eyebrow {{ letter-spacing: 0.16em; text-transform: uppercase; color: var(--accent-2); font-size: 0.78rem; }}
    h1 {{ margin: 8px 0 12px; font-size: clamp(2.1rem, 5vw, 4.8rem); line-height: 0.96; max-width: 880px; }}
    .lead {{ max-width: 820px; font-size: 1.08rem; color: var(--muted); }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin-top: 22px; }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 18px;
      min-height: 116px;
    }}
    .label {{ color: var(--muted); font-size: 0.82rem; }}
    .value {{ font-size: 1.45rem; font-weight: 700; margin-top: 6px; }}
    .section {{ margin-top: 22px; padding: 24px; border-radius: 24px; background: rgba(255,255,255,0.70); border: 1px solid var(--line); }}
    h2 {{ margin: 0 0 14px; font-size: 1.35rem; }}
    .candidate-strip {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }}
    .mini {{ border-left: 5px solid var(--accent); }}
    .mini:nth-child(2) {{ border-left-color: var(--accent-2); }}
    .mini:nth-child(3) {{ border-left-color: var(--gold); }}
    table {{ width: 100%; border-collapse: collapse; overflow: hidden; border-radius: 16px; background: white; }}
    th, td {{ padding: 12px 14px; border-bottom: 1px solid var(--line); text-align: left; }}
    th {{ color: var(--muted); font-size: 0.82rem; }}
    .links {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; }}
    .link-card {{
      display: block;
      padding: 14px 16px;
      text-decoration: none;
      color: var(--ink);
      border: 1px solid var(--line);
      border-radius: 16px;
      background: white;
    }}
    .boundary {{ color: #6c3f22; background: rgba(255, 236, 199, 0.72); }}
    code {{ background: rgba(23,32,24,0.08); padding: 2px 6px; border-radius: 7px; }}
    @media (max-width: 820px) {{
      .grid, .candidate-strip {{ grid-template-columns: 1fr; }}
      .hero {{ padding: 24px; }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <div class="eyebrow">Synthetic workflow demo</div>
      <h1>ML pocket-blocking demo overview</h1>
      <p class="lead">这个页面用于快速展示 demo 运行结果。它证明流程能跑通、报告能生成、证据链能被解释；它不证明真实候选已经被实验验证。</p>
      <div class="grid">
        <div class="card"><div class="label">Quality Gate</div><div class="value">{_escape_html(quality_gate.get('overall_status') or quality_gate.get('status'))}</div></div>
        <div class="card"><div class="label">Batch Decision</div><div class="value">{_escape_html(batch_decision.get('decision_tier'))}</div></div>
        <div class="card"><div class="label">Validation Evidence</div><div class="value">{_escape_html(validation_block.get('audit_status') or validation_block.get('validation_evidence_status'))}</div></div>
        <div class="card"><div class="label">Top-k Coverage</div><div class="value">{_escape_html(_format_percent(validation_block.get('top_k_validation_coverage')))}</div></div>
      </div>
    </section>

    <section class="section">
      <h2>Candidate Highlights</h2>
      <div class="candidate-strip">
        <div class="card mini"><div class="label">Best ranked</div><div class="value">{_escape_html((best_candidate or {}).get('nanobody_id'))}</div><p>Rank {_escape_html((best_candidate or {}).get('consensus_rank'))}, score {_escape_html(_format_float((best_candidate or {}).get('consensus_score')))}</p></div>
        <div class="card mini"><div class="label">Next experiment</div><div class="value">{_escape_html((next_candidate or {}).get('nanobody_id'))}</div><p>{_escape_html((next_candidate or {}).get('recommended_action'))}</p></div>
        <div class="card mini"><div class="label">Needs review</div><div class="value">{_escape_html((risk_candidate or {}).get('nanobody_id'))}</div><p>{_escape_html((risk_candidate or {}).get('risk_summary'))}</p></div>
      </div>
    </section>

    <section class="section">
      <h2>Top Consensus Ranking</h2>
      <table>
        <thead><tr><th>Rank</th><th>Nanobody</th><th>Score</th><th>Confidence</th><th>Decision</th></tr></thead>
        <tbody>{''.join(candidate_rows_html)}</tbody>
      </table>
    </section>

    <section class="section">
      <h2>Open Next</h2>
      <div class="links">{''.join(links_html) if links_html else '<p>Pipeline artifacts are not ready yet.</p>'}</div>
    </section>

    <section class="section boundary">
      <h2>Boundary</h2>
      <p>Demo labels are synthetic proxy labels. Do not present them as wet-lab validation. For real analysis, replace <code>{_escape_html(feature_csv)}</code> and <code>{_escape_html(override_csv)}</code> with your own data and explicit experiment results.</p>
      <p>Manifest: <code>{_escape_html(manifest_json)}</code></p>
    </section>
  </main>
</body>
</html>
"""
    out.write_text(html_text, encoding="utf-8")
    return out


def write_demo_interpretation(
    *,
    out_path: str | Path,
    feature_csv: str | Path,
    override_csv: str | Path,
    summary: dict[str, Any] | None,
    manifest_json: str | Path | None = None,
    real_data_starter_dir: str | Path | None = None,
) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    batch_summary = _load_artifact_json(summary, "batch_decision_summary_json")
    validation_summary = _load_artifact_json(summary, "validation_evidence_summary_json")
    quality_summary = _load_artifact_json(summary, "quality_gate_summary_json")
    consensus_rows = _read_csv_rows(_artifact_path(summary, "consensus_ranking_csv"), limit=5)

    batch_decision = batch_summary.get("batch_decision", {}) if isinstance(batch_summary, dict) else {}
    quality_gate = batch_summary.get("quality_gate", {}) if isinstance(batch_summary, dict) else {}
    if not isinstance(batch_decision, dict):
        batch_decision = {}
    if not isinstance(quality_gate, dict):
        quality_gate = {}
    if not quality_gate and isinstance(quality_summary, dict):
        quality_gate = quality_summary
    validation_block = batch_summary.get("validation_evidence", {}) if isinstance(batch_summary, dict) else {}
    if not isinstance(validation_block, dict):
        validation_block = {}
    if not validation_block and isinstance(validation_summary, dict):
        validation_block = validation_summary

    best_candidate = _candidate_highlight(batch_summary, "best_candidate")
    next_candidate = _candidate_highlight(batch_summary, "next_experiment_candidate")
    risk_candidate = _candidate_highlight(batch_summary, "highest_risk_candidate")

    lines = [
        "# Demo Result Interpretation",
        "",
        "这份解读只针对 synthetic demo。它用于说明软件流程和结果阅读方式，不用于证明真实生物学有效性。",
        "",
        "## 1. 快速结论",
        "",
        f"- Quality Gate: `{_clean_text(quality_gate.get('overall_status') or quality_gate.get('status'))}`",
        f"- Batch decision: `{_clean_text(batch_decision.get('decision_tier'))}`",
        f"- Validation evidence: `{_clean_text(validation_block.get('audit_status') or validation_block.get('validation_evidence_status'))}`",
        f"- Top-k validation coverage: `{_format_percent(validation_block.get('top_k_validation_coverage'))}`",
        f"- Label-ready candidates: `{_clean_text(validation_block.get('label_ready_count'))}`",
    ]
    if best_candidate:
        lines.append(
            "- Best ranked candidate: "
            f"`{_clean_text(best_candidate.get('nanobody_id'))}` "
            f"(rank `{_clean_text(best_candidate.get('consensus_rank'))}`, "
            f"score `{_format_float(best_candidate.get('consensus_score'))}`)"
        )
    if next_candidate:
        lines.append(
            "- Suggested next experiment candidate: "
            f"`{_clean_text(next_candidate.get('nanobody_id'))}` "
            f"(rank `{_clean_text(next_candidate.get('consensus_rank'))}`)"
        )

    lines.extend(
        [
            "",
            "## 2. 这个 demo 证明了什么",
            "",
            "- 软件可以从一个 `pose_features.csv` 启动完整 Rule + ML + QC + report 流程。",
            "- synthetic validation override 可以打通验证证据审计、批次结论摘要和实验计划单。",
            "- 输出结果能形成可读报告、候选卡片、CSV/JSON 产物和可下载汇总包。",
            "- 如果 Quality Gate 为 PASS，只说明 demo 输入和流程内部一致，不代表真实候选已经被实验验证。",
            "",
            "## 3. 这个 demo 不证明什么",
            "",
            "- 不证明任意真实纳米抗体一定有效。",
            "- 不证明 synthetic label 等同于湿实验结果。",
            "- 不证明当前几何 proxy 已经完成跨蛋白泛化验证。",
            "- 不替代真实 PDB、真实 docking pose、真实 pocket 定义和真实实验回灌。",
            "",
            "## 4. Top consensus candidates",
            "",
        ]
    )

    if consensus_rows:
        lines.extend(
            [
                "| Rank | Nanobody | Consensus score | Confidence | Decision |",
                "|---:|---|---:|---|---|",
            ]
        )
        for row in consensus_rows:
            lines.append(
                "| "
                f"{_escape_md_cell(row.get('consensus_rank') or row.get('rank'))} | "
                f"{_escape_md_cell(row.get('nanobody_id'))} | "
                f"{_escape_md_cell(_format_float(row.get('consensus_score')))} | "
                f"{_escape_md_cell(row.get('confidence_level'))} | "
                f"{_escape_md_cell(row.get('decision_tier'))} |"
            )
    else:
        lines.append("当前还没有可读取的 `consensus_ranking.csv`。如果 demo 仍在运行，等待完成后刷新。")

    lines.extend(["", "## 5. 候选解读提示", ""])
    if best_candidate:
        lines.append(f"- 最高综合排名候选：`{_clean_text(best_candidate.get('nanobody_id'))}`。")
        lines.append(f"- 推荐动作：{_clean_text(best_candidate.get('recommended_action'))}")
        risk_summary = _clean_text(best_candidate.get("risk_summary"), default="")
        if risk_summary:
            lines.append(f"- 主要风险：{risk_summary}")
    if next_candidate:
        lines.append(f"- 下一轮实验优先候选：`{_clean_text(next_candidate.get('nanobody_id'))}`。")
        lines.append(f"- 计划建议：{_clean_text(next_candidate.get('recommended_action'))}")
    if risk_candidate:
        lines.append(f"- 最需要复核候选：`{_clean_text(risk_candidate.get('nanobody_id'))}`。")

    lines.extend(
        [
            "",
            "## 6. 建议真实数据下一步",
            "",
            "1. 用真实 `input_pose_table.csv` 或真实 `pose_features.csv` 替换 demo 输入。",
            "2. 明确 pocket / catalytic / ligand 文件来源，并检查路径是否可访问。",
            "3. 跑完后先看 `quality_gate/quality_gate_report.md` 和 `batch_decision_summary/batch_decision_summary.md`。",
            "4. 如果 top 候选没有真实 `experiment_result` 或 `validation_label`，不要把模型排序讲成实验确认。",
            "5. 实验回灌后再用 `experiment_label` 重新训练，并查看再训练前后对照报告。",
            f"6. 如果不确定从哪里开始，先复制 `{real_data_starter_dir}` 里的模板；需要先验证真实 PDB 输入链路时，运行其中的 `MINI_PDB_EXAMPLE`。"
            if real_data_starter_dir
            else "6. 如果不确定从哪里开始，先下载本地软件里的 input_csv / pose_features 模板。",
            "",
            "## 7. Demo 输入位置",
            "",
            f"- Feature CSV: `{feature_csv}`",
            f"- Experiment override CSV: `{override_csv}`",
        ]
    )
    if manifest_json:
        lines.append(f"- Demo manifest: `{manifest_json}`")
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


__all__ = ["write_demo_interpretation", "write_demo_overview_html", "write_demo_readme"]
