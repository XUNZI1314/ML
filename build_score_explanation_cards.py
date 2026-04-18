"""Build product-facing score explanation cards from consensus ranking outputs.

This is a presentation layer only. It reads existing consensus/ranking evidence
and writes human-readable cards; it does not change Rule, ML or consensus scores.
"""

from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _read_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    try:
        return pd.read_csv(csv_path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return default
    return number


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _split_tokens(value: Any) -> list[str]:
    text = _clean_text(value)
    if not text or text.lower() == "none":
        return []
    tokens: list[str] = []
    for part in text.replace("|", ";").split(";"):
        item = part.strip()
        if item:
            tokens.append(item)
    return list(dict.fromkeys(tokens))


def _trim_terminal_punctuation(value: str) -> str:
    return str(value or "").rstrip("。.!！ ")


def _score_band(score: float) -> str:
    if score >= 0.80:
        return "very_high"
    if score >= 0.70:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


def _score_meaning(score: float, tier: str, confidence: str) -> str:
    band = _score_band(score)
    if band == "very_high":
        return f"综合分很高，且当前决策层级为 {tier}，适合作为优先复核对象。"
    if band == "high":
        return f"综合分较高，当前可信度为 {confidence}，建议结合风险项决定是否进入下一轮。"
    if band == "medium":
        return "综合分处于中间区间，更适合与相邻候选横向比较后再决定。"
    return "综合分偏低，除非有外部实验或结构证据支持，否则不建议优先投入。"


def _label_context(feature_csv: str | Path | None, label_col: str) -> dict[str, Any]:
    if feature_csv is None or not str(feature_csv).strip():
        return {
            "label_status": "unknown",
            "label_valid_count": 0,
            "label_class_count": 0,
            "label_context": "未提供 feature_csv，无法判断真实 label 覆盖；当前解释仅基于已有排序和 QC 字段。",
        }
    path = Path(feature_csv).expanduser().resolve()
    if not path.exists():
        return {
            "label_status": "feature_csv_missing",
            "label_valid_count": 0,
            "label_class_count": 0,
            "label_context": f"feature_csv 不存在: {path}；当前解释不使用 label 覆盖信息。",
        }
    df = _read_csv(path)
    if label_col not in df.columns:
        return {
            "label_status": "no_label_column",
            "label_valid_count": 0,
            "label_class_count": 0,
            "label_context": f"未发现 `{label_col}` 列；当前排序主要是 Rule/ML 弱监督或伪标签结果。",
        }
    labels = pd.to_numeric(df[label_col], errors="coerce").dropna()
    label_valid_count = int(len(labels))
    label_class_count = int(labels.nunique()) if label_valid_count > 0 else 0
    if label_valid_count >= 8 and label_class_count >= 2:
        status = "calibration_usable"
        context = f"`{label_col}` 有 {label_valid_count} 条有效值、{label_class_count} 个类别，可支持对照和校准。"
    elif label_valid_count > 0 and label_class_count >= 2:
        status = "compare_only"
        context = f"`{label_col}` 有 {label_valid_count} 条有效值，可做对照，但样本数不足以稳定校准。"
    elif label_valid_count > 0:
        status = "degenerate_label"
        context = f"`{label_col}` 有值但类别不足，不能作为可靠监督信号。"
    else:
        status = "no_valid_label"
        context = f"`{label_col}` 没有有效数值；当前排序不应解读为真实监督模型结论。"
    return {
        "label_status": status,
        "label_valid_count": label_valid_count,
        "label_class_count": label_class_count,
        "label_context": context,
    }


def _positive_factors(row: pd.Series) -> list[str]:
    factors: list[str] = []
    consensus_score = _safe_float(row.get("consensus_score"), 0.0)
    confidence_score = _safe_float(row.get("confidence_score"), 0.0)
    rank_agreement = _safe_float(row.get("rank_agreement_score"), 0.0)
    score_alignment = _safe_float(row.get("score_alignment_score"), 0.0)
    ml_score = _safe_float(row.get("ml_score"), float("nan"))
    rule_score = _safe_float(row.get("rule_score"), float("nan"))
    pocket_consistency = _safe_float(row.get("ml_pocket_consistency_score"), float("nan"))
    if not np.isfinite(pocket_consistency):
        pocket_consistency = _safe_float(row.get("rule_pocket_consistency_score"), float("nan"))

    if consensus_score >= 0.75:
        factors.append(f"综合分较高 ({consensus_score:.3f})")
    if confidence_score >= 0.65:
        factors.append(f"可信度分较高 ({confidence_score:.3f})")
    if rank_agreement >= 0.70:
        factors.append(f"Rule/ML 排名一致性较好 ({rank_agreement:.3f})")
    if score_alignment >= 0.70:
        factors.append(f"Rule/ML 分数方向一致 ({score_alignment:.3f})")
    if np.isfinite(ml_score) and ml_score >= 0.70:
        factors.append(f"ML 分数支持 ({ml_score:.3f})")
    if np.isfinite(rule_score) and rule_score >= 0.70:
        factors.append(f"规则分数支持 ({rule_score:.3f})")
    if np.isfinite(pocket_consistency) and pocket_consistency >= 0.65:
        factors.append(f"构象间 pocket 相关信号较稳定 ({pocket_consistency:.3f})")
    if not factors:
        factors.append("没有特别突出的正向信号，建议主要参考横向对比和实验优先级。")
    return factors[:5]


def _risk_factors(row: pd.Series, label_status: str) -> list[str]:
    risks: list[str] = []
    for token in _split_tokens(row.get("risk_flags")) + _split_tokens(row.get("review_reason_flags")):
        if token and token != "none":
            risks.append(token)
    low_confidence = _clean_text(row.get("low_confidence_reasons"))
    if low_confidence and low_confidence != "未发现明显低可信原因":
        risks.append(low_confidence)
    qc_risk = _safe_float(row.get("qc_risk_score"), 0.0)
    if qc_risk >= 0.40:
        risks.append(f"综合 QC 风险偏高 ({qc_risk:.3f})")
    overwide = _safe_float(row.get("pocket_overwide_proxy_for_consensus"), float("nan"))
    if np.isfinite(overwide) and overwide >= 0.55:
        risks.append(f"pocket 可能偏宽 ({overwide:.3f})")
    if label_status in {"unknown", "feature_csv_missing", "no_label_column", "no_valid_label", "degenerate_label"}:
        risks.append("缺少可用真实 label，不能把分数解读为实验成功概率")
    if not risks:
        risks.append("未发现明显高风险项，但仍建议人工检查结构和输入质量。")
    return list(dict.fromkeys(risks))[:6]


def _recommended_action(row: pd.Series, risks: list[str]) -> str:
    tier = _clean_text(row.get("decision_tier")).lower()
    confidence = _clean_text(row.get("confidence_level")).lower()
    score = _safe_float(row.get("consensus_score"), 0.0)
    risk_text = ";".join(risks).lower()
    if tier == "priority" and confidence == "high" and score >= 0.75 and "缺少可用真实 label" not in risk_text:
        return "优先进入下一轮实验验证；同时保留结构/QC 复核。"
    if tier == "priority":
        return "可作为优先候选，但建议先复核风险项再投入实验。"
    if tier == "review":
        return "建议进入人工复核或补充结构证据后再决定。"
    if score >= 0.60:
        return "可作为备选候选；优先比较相邻候选和实验资源。"
    return "暂不建议优先投入；除非有外部证据支持。"


def _rule_ml_note(row: pd.Series) -> str:
    ml_score = _safe_float(row.get("ml_score"), float("nan"))
    rule_score = _safe_float(row.get("rule_score"), float("nan"))
    rank_agreement = _safe_float(row.get("rank_agreement_score"), float("nan"))
    if not np.isfinite(ml_score) or not np.isfinite(rule_score):
        return "只有一条排序路线可用，解释时应降低置信度。"
    gap = abs(ml_score - rule_score)
    if gap <= 0.10 and np.isfinite(rank_agreement) and rank_agreement >= 0.70:
        return "ML 和 Rule 给出的方向基本一致，说明当前候选不是单一路线偶然高分。"
    if gap > 0.25:
        return "ML 和 Rule 分数差距较大，需要人工查看几何信号、QC warning 和输入质量。"
    return "ML 和 Rule 存在一定差异，建议结合候选对比表和报告卡复核。"


def build_score_explanation_cards(
    consensus_df: pd.DataFrame,
    *,
    label_context: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if consensus_df.empty:
        raise ValueError("consensus_df is empty.")
    if "nanobody_id" not in consensus_df.columns:
        raise ValueError("consensus_df is missing nanobody_id column.")

    label_context = dict(label_context or {})
    label_status = str(label_context.get("label_status") or "unknown")
    rows: list[dict[str, Any]] = []
    for _, row in consensus_df.iterrows():
        score = _safe_float(row.get("consensus_score"), 0.0)
        confidence = _clean_text(row.get("confidence_level")) or "unknown"
        tier = _clean_text(row.get("decision_tier")) or "standard"
        positives = _positive_factors(row)
        risks = _risk_factors(row, label_status)
        positive_summary = _trim_terminal_punctuation("；".join(positives[:3]))
        risk_summary = _trim_terminal_punctuation("；".join(risks[:3]))
        action = _recommended_action(row, risks)
        rows.append(
            {
                "nanobody_id": _clean_text(row.get("nanobody_id")),
                "consensus_rank": _safe_int(row.get("consensus_rank"), 0),
                "decision_tier": tier,
                "confidence_level": confidence,
                "score_band": _score_band(score),
                "consensus_score": score,
                "confidence_score": _safe_float(row.get("confidence_score"), 0.0),
                "ml_score": _safe_float(row.get("ml_score"), float("nan")),
                "rule_score": _safe_float(row.get("rule_score"), float("nan")),
                "qc_risk_score": _safe_float(row.get("qc_risk_score"), 0.0),
                "score_meaning": _score_meaning(score, tier, confidence),
                "main_positive_factors": "；".join(positives),
                "main_risk_factors": "；".join(risks),
                "rule_ml_interpretation": _rule_ml_note(row),
                "label_status": label_status,
                "label_valid_count": int(label_context.get("label_valid_count") or 0),
                "label_class_count": int(label_context.get("label_class_count") or 0),
                "label_context": str(label_context.get("label_context") or ""),
                "recommended_action": action,
                "review_checklist": "检查结构输入；核对 pocket 定义；查看 QC/warning；查看候选横向对比；必要时补真实验证标签",
                "plain_language_summary": (
                    f"{_clean_text(row.get('nanobody_id'))}: {_score_meaning(score, tier, confidence)} "
                    f"主要依据：{positive_summary}。主要风险：{risk_summary}。"
                ),
            }
        )
    out = pd.DataFrame(rows)
    if "consensus_rank" in out.columns:
        out = out.sort_values("consensus_rank", ascending=True, na_position="last")
    else:
        out = out.sort_values("consensus_score", ascending=False, na_position="last")
    return out.reset_index(drop=True)


def _markdown_table(df: pd.DataFrame, *, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    columns = [
        "consensus_rank",
        "nanobody_id",
        "score_band",
        "confidence_level",
        "decision_tier",
        "consensus_score",
        "recommended_action",
    ]
    work = df.loc[:, [col for col in columns if col in df.columns]].head(max_rows)
    header = "| " + " | ".join(work.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(work.columns)) + " |"
    body = ["| " + " | ".join(str(row.get(col, "")) for col in work.columns) + " |" for _, row in work.iterrows()]
    return "\n".join([header, sep] + body)


def _build_report(cards_df: pd.DataFrame, summary: dict[str, Any]) -> str:
    lines = [
        "# Score Explanation Cards",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Candidates: `{summary['candidate_count']}`",
        f"- Label status: `{summary['label_status']}`",
        f"- High / very-high score candidates: `{summary['high_score_candidate_count']}`",
        f"- Review candidates: `{summary['review_candidate_count']}`",
        "",
        "## How To Read",
        "",
        "- `consensus_score` is a decision-support score that combines ML, Rule agreement and QC risk.",
        "- `confidence_level` describes how much the available evidence agrees, not whether biology is proven.",
        "- `main_risk_factors` must be checked before treating a high score as an experimental priority.",
        "",
        "## Top Cards",
        "",
        _markdown_table(cards_df, max_rows=20),
        "",
        "## Top Plain-Language Summaries",
        "",
    ]
    for _, row in cards_df.head(10).iterrows():
        lines.append(f"- {row.get('plain_language_summary', '')}")
    lines.append("")
    return "\n".join(lines)


def _build_html(cards_df: pd.DataFrame, summary: dict[str, Any]) -> str:
    cards: list[str] = []
    for _, row in cards_df.head(80).iterrows():
        tier = html.escape(str(row.get("decision_tier") or "standard"))
        band = html.escape(str(row.get("score_band") or "medium"))
        cards.append(
            f"""
    <article class="card {tier}">
      <div class="card-head">
        <span class="rank">#{html.escape(str(row.get('consensus_rank') or '?'))}</span>
        <h2>{html.escape(str(row.get('nanobody_id') or ''))}</h2>
        <span class="band">{band}</span>
      </div>
      <p class="summary">{html.escape(str(row.get('plain_language_summary') or ''))}</p>
      <div class="metrics">
        <div><strong>{_safe_float(row.get('consensus_score'), 0.0):.3f}</strong><span>Consensus</span></div>
        <div><strong>{html.escape(str(row.get('confidence_level') or ''))}</strong><span>Confidence</span></div>
        <div><strong>{_safe_float(row.get('qc_risk_score'), 0.0):.3f}</strong><span>QC risk</span></div>
      </div>
      <section>
        <h3>Why It Scores This Way</h3>
        <p>{html.escape(str(row.get('main_positive_factors') or ''))}</p>
      </section>
      <section>
        <h3>Risks To Check</h3>
        <p>{html.escape(str(row.get('main_risk_factors') or ''))}</p>
      </section>
      <section>
        <h3>Recommended Action</h3>
        <p>{html.escape(str(row.get('recommended_action') or ''))}</p>
      </section>
    </article>
"""
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Score Explanation Cards</title>
  <style>
    :root {{
      --bg: #f6f1e7;
      --ink: #18202f;
      --muted: #687385;
      --card: #fffdf8;
      --line: #e0d6c5;
      --accent: #0f766e;
      --warn: #b45309;
    }}
    body {{
      margin: 0;
      padding: 34px;
      background: radial-gradient(circle at top left, #dff8ef 0, transparent 32%), var(--bg);
      color: var(--ink);
      font-family: "Segoe UI", "Noto Sans", sans-serif;
    }}
    header, main {{
      max-width: 1160px;
      margin: 0 auto;
    }}
    h1 {{
      margin: 0 0 8px 0;
      font-size: 34px;
      letter-spacing: -0.04em;
    }}
    .summary-grid, .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 12px;
      margin: 18px 0;
    }}
    .summary-grid div, .metrics div {{
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 12px 14px;
      background: rgba(255,255,255,0.75);
    }}
    strong {{
      display: block;
      font-size: 24px;
    }}
    span, p {{
      color: var(--muted);
    }}
    .card {{
      margin: 18px 0;
      padding: 20px;
      border: 1px solid var(--line);
      border-left: 8px solid var(--accent);
      border-radius: 22px;
      background: var(--card);
      box-shadow: 0 20px 60px rgba(31, 24, 15, 0.08);
    }}
    .card.review {{
      border-left-color: var(--warn);
    }}
    .card-head {{
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }}
    .card h2 {{
      margin: 0;
      font-size: 22px;
    }}
    .rank, .band {{
      padding: 5px 10px;
      border-radius: 999px;
      background: #e8f2ed;
      color: var(--ink);
      font-weight: 700;
    }}
    .summary {{
      font-size: 15px;
      color: var(--ink);
    }}
    h3 {{
      margin-bottom: 4px;
      font-size: 14px;
      color: var(--ink);
    }}
  </style>
</head>
<body>
  <header>
    <h1>Score Explanation Cards</h1>
    <p>Human-readable interpretation of consensus scores. These cards do not change ranking outputs.</p>
    <div class="summary-grid">
      <div><strong>{int(summary.get('candidate_count') or 0)}</strong><span>Candidates</span></div>
      <div><strong>{int(summary.get('high_score_candidate_count') or 0)}</strong><span>High score</span></div>
      <div><strong>{int(summary.get('review_candidate_count') or 0)}</strong><span>Review tier</span></div>
      <div><strong>{html.escape(str(summary.get('label_status') or 'unknown'))}</strong><span>Label status</span></div>
    </div>
  </header>
  <main>
    {''.join(cards)}
  </main>
</body>
</html>
"""


def build_score_explanation_outputs(
    *,
    consensus_csv: str | Path,
    out_dir: str | Path = "score_explanation_cards",
    feature_csv: str | Path | None = None,
    label_col: str = "label",
) -> dict[str, Any]:
    consensus_path = Path(consensus_csv).expanduser().resolve()
    consensus_df = _read_csv(consensus_path)
    context = _label_context(feature_csv, label_col)
    cards_df = build_score_explanation_cards(consensus_df, label_context=context)

    output_dir = Path(out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cards_csv = output_dir / "score_explanation_cards.csv"
    summary_json = output_dir / "score_explanation_cards_summary.json"
    report_md = output_dir / "score_explanation_cards.md"
    report_html = output_dir / "score_explanation_cards.html"

    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "consensus_csv": str(consensus_path),
        "feature_csv": None if feature_csv is None else str(Path(feature_csv).expanduser().resolve()),
        "label_col": str(label_col),
        "candidate_count": int(len(cards_df)),
        "high_score_candidate_count": int(cards_df["score_band"].isin(["high", "very_high"]).sum()),
        "review_candidate_count": int(cards_df["decision_tier"].astype(str).str.lower().eq("review").sum()),
        **context,
        "outputs": {
            "score_explanation_cards_csv": str(cards_csv),
            "score_explanation_cards_summary_json": str(summary_json),
            "score_explanation_cards_md": str(report_md),
            "score_explanation_cards_html": str(report_html),
        },
    }
    cards_df.to_csv(cards_csv, index=False)
    summary_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    report_md.write_text(_build_report(cards_df, summary), encoding="utf-8")
    report_html.write_text(_build_html(cards_df, summary), encoding="utf-8")
    return {
        "summary": summary,
        "cards": cards_df,
        "outputs": summary["outputs"],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build product-facing score explanation cards.")
    parser.add_argument("--consensus_csv", required=True, help="Path to consensus_ranking.csv")
    parser.add_argument("--out_dir", default="score_explanation_cards")
    parser.add_argument("--feature_csv", default=None, help="Optional feature CSV for label coverage context")
    parser.add_argument("--label_col", default="label")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = build_score_explanation_outputs(
        consensus_csv=args.consensus_csv,
        out_dir=args.out_dir,
        feature_csv=args.feature_csv,
        label_col=str(args.label_col),
    )
    outputs = result["outputs"]
    for key, value in outputs.items():
        print(f"Saved {key}: {value}")
    summary = result["summary"]
    print(
        "Score explanation cards: "
        f"candidates={summary.get('candidate_count', 0)}, "
        f"high_score={summary.get('high_score_candidate_count', 0)}, "
        f"review={summary.get('review_candidate_count', 0)}"
    )


if __name__ == "__main__":
    main()
