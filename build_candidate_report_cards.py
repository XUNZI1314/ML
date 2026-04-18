"""Build per-nanobody candidate report cards from ranking outputs.

The cards are a presentation layer. They do not change Rule, ML or consensus
scores; they only collect the existing evidence for quick review.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build nanobody candidate report cards")
    parser.add_argument("--consensus_csv", required=True, help="Path to consensus_ranking.csv")
    parser.add_argument("--out_dir", default="candidate_report_cards", help="Output directory for HTML cards")
    parser.add_argument("--rule_csv", default=None, help="Optional nanobody_rule_ranking.csv")
    parser.add_argument("--ml_csv", default=None, help="Optional nanobody_ranking.csv")
    parser.add_argument("--feature_csv", default=None, help="Optional pose_features.csv")
    parser.add_argument("--pose_predictions_csv", default=None, help="Optional pose_predictions.csv")
    parser.add_argument("--candidate_pairwise_csv", default=None, help="Optional candidate_pairwise_comparisons.csv")
    parser.add_argument("--top_n", type=int, default=0, help="Only build first N candidates; 0 means all")
    parser.add_argument("--max_pose_rows", type=int, default=8, help="Top pose rows shown in each card")
    parser.add_argument("--zip_path", default=None, help="Optional output zip path")
    return parser


def _read_csv_optional(path_text: str | Path | None) -> pd.DataFrame | None:
    if not path_text:
        return None
    path = Path(path_text).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path, low_memory=False)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _fmt_value(value: Any) -> str:
    if _is_missing(value):
        return "N/A"
    if isinstance(value, (bool, np.bool_)):
        return "Yes" if bool(value) else "No"
    number = _safe_float(value)
    if np.isfinite(number):
        if abs(number - round(number)) < 1e-9 and abs(number) < 1e9:
            return str(int(round(number)))
        return f"{number:.4f}"
    text = str(value).strip()
    return text if text else "N/A"


def _escape(value: Any) -> str:
    return html.escape(_fmt_value(value))


def _safe_filename(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = text.strip("._-")
    if not text:
        text = fallback
    return text[:96]


def _sort_consensus(consensus_df: pd.DataFrame) -> pd.DataFrame:
    work = consensus_df.copy()
    if "consensus_rank" in work.columns:
        work["_sort_rank"] = pd.to_numeric(work["consensus_rank"], errors="coerce")
        return work.sort_values(by=["_sort_rank"], ascending=True, na_position="last").drop(columns=["_sort_rank"])
    if "consensus_score" in work.columns:
        work["_sort_score"] = pd.to_numeric(work["consensus_score"], errors="coerce")
        return work.sort_values(by=["_sort_score"], ascending=False, na_position="last").drop(columns=["_sort_score"])
    return work


def _lookup_record(df: pd.DataFrame | None, nanobody_id: str) -> dict[str, Any]:
    if df is None or df.empty or "nanobody_id" not in df.columns:
        return {}
    mask = df["nanobody_id"].astype(str).eq(str(nanobody_id))
    if not mask.any():
        return {}
    return df.loc[mask].iloc[0].to_dict()


def _pick_columns(df: pd.DataFrame, preferred_columns: list[str], max_cols: int = 12) -> pd.DataFrame:
    columns = [col for col in preferred_columns if col in df.columns]
    if not columns:
        columns = [str(col) for col in df.columns[:max_cols]]
    return df.loc[:, columns[:max_cols]].copy()


def _row_table_html(title: str, record: dict[str, Any], fields: list[str]) -> str:
    rows: list[str] = []
    for field in fields:
        if field not in record:
            continue
        rows.append(f"<tr><th>{html.escape(field)}</th><td>{_escape(record.get(field))}</td></tr>")
    if not rows:
        rows.append("<tr><td class='muted' colspan='2'>No data available.</td></tr>")
    return f"<section class='panel'><h2>{html.escape(title)}</h2><table class='kv'>{''.join(rows)}</table></section>"


def _df_table_html(title: str, df: pd.DataFrame | None, *, preferred_columns: list[str], max_rows: int = 8) -> str:
    if df is None or df.empty:
        return f"<section class='panel'><h2>{html.escape(title)}</h2><p class='muted'>No data available.</p></section>"

    view = _pick_columns(df.head(int(max_rows)), preferred_columns=preferred_columns)
    header = "".join(f"<th>{html.escape(str(col))}</th>" for col in view.columns)
    body_rows: list[str] = []
    for _, row in view.iterrows():
        body_rows.append("".join(f"<td>{_escape(row.get(col))}</td>" for col in view.columns))
    body = "".join(f"<tr>{row_html}</tr>" for row_html in body_rows)
    return (
        f"<section class='panel'><h2>{html.escape(title)}</h2>"
        "<div class='table-scroll'><table>"
        f"<thead><tr>{header}</tr></thead><tbody>{body}</tbody>"
        "</table></div></section>"
    )


def _feature_summary_record(feature_df: pd.DataFrame | None, nanobody_id: str) -> dict[str, Any]:
    if feature_df is None or feature_df.empty or "nanobody_id" not in feature_df.columns:
        return {}
    group = feature_df[feature_df["nanobody_id"].astype(str).eq(str(nanobody_id))]
    if group.empty:
        return {}

    record: dict[str, Any] = {"pose_row_count": int(len(group))}
    if "status" in group.columns:
        status = group["status"].fillna("").astype(str).str.lower()
        record["failed_pose_count"] = int(status.eq("failed").sum())
        record["ok_pose_count"] = int(status.eq("ok").sum())
    if "warning_message" in group.columns:
        warning = group["warning_message"].fillna("").astype(str).str.strip().ne("")
        record["warning_pose_count"] = int(warning.sum())

    metric_columns = [
        "pred_prob",
        "pocket_hit_fraction",
        "catalytic_hit_fraction",
        "mouth_occlusion_score",
        "mouth_aperture_block_fraction",
        "substrate_overlap_score",
        "ligand_path_block_score",
        "ligand_path_exit_block_fraction",
        "min_distance_to_pocket",
        "pocket_shape_overwide_proxy",
    ]
    for col in metric_columns:
        if col not in group.columns:
            continue
        values = pd.to_numeric(group[col], errors="coerce").dropna()
        if values.empty:
            continue
        record[f"{col}_mean"] = float(values.mean())
        record[f"{col}_max"] = float(values.max())
        record[f"{col}_min"] = float(values.min())
    return record


def _candidate_pose_rows(
    *,
    nanobody_id: str,
    pose_pred_df: pd.DataFrame | None,
    feature_df: pd.DataFrame | None,
    max_rows: int,
) -> tuple[pd.DataFrame | None, str]:
    source_df = None
    source_name = ""
    if pose_pred_df is not None and not pose_pred_df.empty and "nanobody_id" in pose_pred_df.columns:
        source_df = pose_pred_df[pose_pred_df["nanobody_id"].astype(str).eq(str(nanobody_id))].copy()
        source_name = "pose_predictions"
    if (source_df is None or source_df.empty) and feature_df is not None and not feature_df.empty and "nanobody_id" in feature_df.columns:
        source_df = feature_df[feature_df["nanobody_id"].astype(str).eq(str(nanobody_id))].copy()
        source_name = "pose_features"
    if source_df is None or source_df.empty:
        return None, ""

    sort_candidates = [
        "pred_prob",
        "pseudo_score",
        "pocket_hit_fraction",
        "catalytic_hit_fraction",
        "mouth_occlusion_score",
        "ligand_path_block_score",
    ]
    sort_col = next((col for col in sort_candidates if col in source_df.columns), "")
    if sort_col:
        source_df["_sort_value"] = pd.to_numeric(source_df[sort_col], errors="coerce")
        source_df = source_df.sort_values(by="_sort_value", ascending=False, na_position="last").drop(columns=["_sort_value"])
    return source_df.head(int(max_rows)), source_name


def _candidate_pairwise_rows(
    pairwise_df: pd.DataFrame | None,
    nanobody_id: str,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    if pairwise_df is None or pairwise_df.empty:
        return None, {
            "candidate_pairwise_comparison_count": 0,
            "candidate_pairwise_close_decision_count": 0,
            "candidate_pairwise_winner_count": 0,
            "candidate_pairwise_runner_up_count": 0,
        }
    if "winner_nanobody_id" not in pairwise_df.columns or "runner_up_nanobody_id" not in pairwise_df.columns:
        return None, {
            "candidate_pairwise_comparison_count": 0,
            "candidate_pairwise_close_decision_count": 0,
            "candidate_pairwise_winner_count": 0,
            "candidate_pairwise_runner_up_count": 0,
        }

    winner_mask = pairwise_df["winner_nanobody_id"].astype(str).eq(str(nanobody_id))
    runner_mask = pairwise_df["runner_up_nanobody_id"].astype(str).eq(str(nanobody_id))
    rows = pairwise_df[winner_mask | runner_mask].copy()
    if rows.empty:
        return None, {
            "candidate_pairwise_comparison_count": 0,
            "candidate_pairwise_close_decision_count": 0,
            "candidate_pairwise_winner_count": 0,
            "candidate_pairwise_runner_up_count": 0,
        }

    roles: list[str] = []
    counterparts: list[str] = []
    for _, row in rows.iterrows():
        if str(row.get("winner_nanobody_id")) == str(nanobody_id):
            roles.append("winner")
            counterparts.append(str(row.get("runner_up_nanobody_id")))
        else:
            roles.append("runner_up")
            counterparts.append(str(row.get("winner_nanobody_id")))
    rows.insert(0, "candidate_role", roles)
    rows.insert(1, "compared_with", counterparts)

    close_count = 0
    if "is_close_decision" in rows.columns:
        close_count = int(rows["is_close_decision"].astype(str).str.lower().isin(["true", "1", "yes"]).sum())
    stats = {
        "candidate_pairwise_comparison_count": int(len(rows)),
        "candidate_pairwise_close_decision_count": int(close_count),
        "candidate_pairwise_winner_count": int(sum(role == "winner" for role in roles)),
        "candidate_pairwise_runner_up_count": int(sum(role == "runner_up" for role in roles)),
    }
    return rows, stats


def _style_html() -> str:
    return """
<style>
  :root {
    --ink: #1f2522;
    --muted: #65706a;
    --bg: #f2efe7;
    --card: #fffdf7;
    --line: #ddd2bf;
    --accent: #0e6b57;
    --accent2: #a8662a;
    --bad: #9a332f;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
    background: radial-gradient(circle at top left, #dfeee8 0, transparent 34%), linear-gradient(180deg, #efe7d8, var(--bg));
    color: var(--ink);
    line-height: 1.55;
  }
  .page { max-width: 1180px; margin: 0 auto; padding: 28px 20px 60px; }
  .hero {
    border-radius: 24px;
    padding: 26px;
    color: #fffaf0;
    background: linear-gradient(135deg, #163d35 0%, #0e6b57 54%, #bd8a47 100%);
    box-shadow: 0 18px 48px rgba(24, 48, 40, 0.18);
  }
  .hero h1 { margin: 0 0 8px; font-size: 30px; }
  .hero p { margin: 0; color: rgba(255, 250, 240, 0.88); }
  .nav { margin-top: 12px; }
  .nav a { color: #fff7d9; font-weight: 700; text-decoration: none; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin-top: 16px; }
  .metric, .panel {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 18px;
    box-shadow: 0 10px 28px rgba(28, 38, 33, 0.06);
  }
  .metric { padding: 16px; }
  .metric-label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; }
  .metric-value { margin-top: 6px; font-size: 24px; font-weight: 800; }
  .panel { margin-top: 16px; padding: 18px; }
  .panel h2 { margin: 0 0 12px; font-size: 18px; }
  .callout {
    margin-top: 16px;
    padding: 14px 16px;
    border-radius: 16px;
    border: 1px solid #d6c195;
    background: #fff6df;
  }
  .callout.review, .callout.low { border-color: #e2b9a8; background: #fff0eb; }
  .callout.priority, .callout.high { border-color: #a7d2bf; background: #eaf7f1; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { border: 1px solid #e6dccb; padding: 8px 10px; text-align: left; vertical-align: top; }
  th { background: #f5eee2; }
  .kv th { width: 280px; color: #3e4742; }
  .table-scroll { overflow-x: auto; }
  .muted { color: var(--muted); }
  .footer { text-align: center; color: var(--muted); font-size: 12px; margin-top: 20px; }
  @media print {
    body { background: #fff; }
    .page { max-width: none; padding: 0; }
    .hero, .metric, .panel { box-shadow: none; }
  }
</style>
"""


def _metric_card(label: str, value: Any) -> str:
    return (
        "<div class='metric'>"
        f"<div class='metric-label'>{html.escape(label)}</div>"
        f"<div class='metric-value'>{_escape(value)}</div>"
        "</div>"
    )


def _build_card_html(
    *,
    nanobody_id: str,
    consensus_record: dict[str, Any],
    rule_record: dict[str, Any],
    ml_record: dict[str, Any],
    feature_record: dict[str, Any],
    pose_rows: pd.DataFrame | None,
    pose_source_name: str,
    pairwise_rows: pd.DataFrame | None,
    pairwise_stats: dict[str, Any],
    generated_at: str,
) -> str:
    tier = str(consensus_record.get("decision_tier") or "standard").lower()
    level = str(consensus_record.get("confidence_level") or "medium").lower()
    callout_class = "priority" if tier == "priority" or level == "high" else ("review" if tier == "review" or level == "low" else "")

    metric_html = "".join(
        [
            _metric_card("Consensus Rank", consensus_record.get("consensus_rank")),
            _metric_card("Consensus Score", consensus_record.get("consensus_score")),
            _metric_card("Confidence", consensus_record.get("confidence_level")),
            _metric_card("Decision Tier", consensus_record.get("decision_tier")),
            _metric_card("ML Score", consensus_record.get("ml_score")),
            _metric_card("Rule Score", consensus_record.get("rule_score")),
            _metric_card("Pairwise Reviews", pairwise_stats.get("candidate_pairwise_comparison_count")),
            _metric_card("Close Pairs", pairwise_stats.get("candidate_pairwise_close_decision_count")),
        ]
    )

    consensus_fields = [
        "nanobody_id",
        "consensus_rank",
        "decision_tier",
        "confidence_level",
        "consensus_score",
        "confidence_score",
        "ml_score",
        "rule_score",
        "ml_rank",
        "rule_rank",
        "abs_rank_delta",
        "rank_agreement_score",
        "score_alignment_score",
        "qc_risk_score",
        "review_reason_flags",
        "low_confidence_reasons",
        "close_score_competition_warning",
        "nearest_competitor_score_gap",
        "conformer_instability_score",
        "risk_flags",
        "consensus_explanation",
    ]
    ml_fields = [
        "rank",
        "final_score",
        "best_conformer_score",
        "mean_conformer_score",
        "std_conformer_score",
        "pocket_consistency_score",
        "best_pose_prob",
        "explanation",
    ]
    rule_fields = [
        "rank",
        "final_rule_score",
        "best_conformer_rule_score",
        "mean_conformer_rule_score",
        "std_conformer_rule_score",
        "pocket_consistency_score",
        "best_pose_rule_score",
        "explanation",
    ]
    feature_fields = [
        "pose_row_count",
        "ok_pose_count",
        "failed_pose_count",
        "warning_pose_count",
        "pocket_hit_fraction_mean",
        "catalytic_hit_fraction_mean",
        "mouth_occlusion_score_mean",
        "ligand_path_block_score_mean",
        "pocket_shape_overwide_proxy_mean",
        "pocket_shape_overwide_proxy_max",
        "min_distance_to_pocket_min",
    ]
    pose_columns = [
        "nanobody_id",
        "conformer_id",
        "pose_id",
        "pred_prob",
        "label",
        "pseudo_score",
        "pocket_hit_fraction",
        "catalytic_hit_fraction",
        "mouth_occlusion_score",
        "ligand_path_block_score",
        "pocket_shape_overwide_proxy",
    ]
    pairwise_columns = [
        "candidate_role",
        "compared_with",
        "winner_nanobody_id",
        "runner_up_nanobody_id",
        "consensus_score_gap",
        "is_close_decision",
        "winner_key_advantages",
        "runner_up_counterpoints",
        "comparison_explanation",
    ]
    explanation = consensus_record.get("consensus_explanation") or "No consensus explanation available."
    risk_flags = consensus_record.get("risk_flags") or "none"
    low_confidence_reasons = consensus_record.get("low_confidence_reasons") or "未发现明显低可信原因"

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Candidate Report | {html.escape(str(nanobody_id))}</title>
  {_style_html()}
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>Candidate Report: {html.escape(str(nanobody_id))}</h1>
      <p>Rule + ML + QC evidence summary | Generated at {html.escape(generated_at)}</p>
      <div class="nav"><a href="../index.html">Back to candidate index</a></div>
    </section>
    <div class="grid">{metric_html}</div>
    <div class="callout {html.escape(callout_class)}">
      <strong>Decision note:</strong> {html.escape(str(explanation))}<br />
      <strong>Risk flags:</strong> {html.escape(str(risk_flags))}<br />
      <strong>Review reasons:</strong> {html.escape(str(low_confidence_reasons))}
    </div>
    {_row_table_html("Consensus Evidence", consensus_record, consensus_fields)}
    {_row_table_html("ML Ranking Summary", ml_record, ml_fields)}
    {_row_table_html("Rule Ranking Summary", rule_record, rule_fields)}
    {_row_table_html("Feature / QC Summary", feature_record, feature_fields)}
    {_df_table_html("Candidate Pairwise Comparison Context", pairwise_rows, preferred_columns=pairwise_columns, max_rows=8)}
    {_df_table_html(f"Top Pose Rows ({pose_source_name or 'N/A'})", pose_rows, preferred_columns=pose_columns, max_rows=12)}
    <section class="panel">
      <h2>Interpretation Guardrail</h2>
      <p class="muted">This card is a decision-support report generated from existing pipeline outputs. It is not an experimental validation result and should be reviewed together with structure quality and biological context.</p>
    </section>
    <div class="footer">ML Local App candidate report card</div>
  </div>
</body>
</html>
"""


def _build_index_html(cards: list[dict[str, Any]], generated_at: str) -> str:
    rows: list[str] = []
    for card in cards:
        rows.append(
            "<tr>"
            f"<td>{_escape(card.get('consensus_rank'))}</td>"
            f"<td><a href='{html.escape(str(card.get('card_relpath')))}'>{html.escape(str(card.get('nanobody_id')))}</a></td>"
            f"<td>{_escape(card.get('decision_tier'))}</td>"
            f"<td>{_escape(card.get('confidence_level'))}</td>"
            f"<td>{_escape(card.get('consensus_score'))}</td>"
            f"<td>{_escape(card.get('ml_score'))}</td>"
            f"<td>{_escape(card.get('rule_score'))}</td>"
            f"<td>{_escape(card.get('risk_flags'))}</td>"
            "</tr>"
        )
    table_rows = "".join(rows) if rows else "<tr><td colspan='8' class='muted'>No candidates.</td></tr>"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Candidate Report Cards</title>
  {_style_html()}
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>Candidate Report Cards</h1>
      <p>Per-nanobody HTML reports generated from consensus ranking | Generated at {html.escape(generated_at)}</p>
    </section>
    <section class="panel">
      <h2>Candidate Index</h2>
      <div class="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Rank</th><th>Nanobody</th><th>Tier</th><th>Confidence</th><th>Consensus</th><th>ML</th><th>Rule</th><th>Risk</th>
            </tr>
          </thead>
          <tbody>{table_rows}</tbody>
        </table>
      </div>
    </section>
    <div class="footer">Open any nanobody ID to view its report card. Use browser print to save a card as PDF.</div>
  </div>
</body>
</html>
"""


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_sanitize(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if np.isfinite(out) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if _is_missing(value):
        return None
    return value


def build_candidate_report_cards(
    *,
    consensus_df: pd.DataFrame,
    out_dir: Path,
    rule_df: pd.DataFrame | None = None,
    ml_df: pd.DataFrame | None = None,
    feature_df: pd.DataFrame | None = None,
    pose_pred_df: pd.DataFrame | None = None,
    candidate_pairwise_df: pd.DataFrame | None = None,
    candidate_pairwise_csv_path: str | Path | None = None,
    top_n: int = 0,
    max_pose_rows: int = 8,
    zip_path: Path | None = None,
) -> dict[str, Any]:
    if consensus_df.empty:
        raise ValueError("consensus_csv is empty.")
    if "nanobody_id" not in consensus_df.columns:
        raise ValueError("consensus_csv is missing nanobody_id column.")

    out_dir = out_dir.expanduser().resolve()
    cards_dir = out_dir / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().isoformat(timespec="seconds")

    consensus_sorted = _sort_consensus(consensus_df).reset_index(drop=True)
    if int(top_n) > 0:
        consensus_sorted = consensus_sorted.head(int(top_n)).copy()

    cards: list[dict[str, Any]] = []
    for index, (_, row) in enumerate(consensus_sorted.iterrows(), start=1):
        nanobody_id = str(row.get("nanobody_id"))
        consensus_record = row.to_dict()
        rule_record = _lookup_record(rule_df, nanobody_id)
        ml_record = _lookup_record(ml_df, nanobody_id)
        feature_record = _feature_summary_record(feature_df, nanobody_id)
        pose_rows, pose_source_name = _candidate_pose_rows(
            nanobody_id=nanobody_id,
            pose_pred_df=pose_pred_df,
            feature_df=feature_df,
            max_rows=int(max_pose_rows),
        )
        pairwise_rows, pairwise_stats = _candidate_pairwise_rows(candidate_pairwise_df, nanobody_id)

        rank_text = _safe_filename(consensus_record.get("consensus_rank"), fallback=f"{index:03d}")
        file_name = f"rank_{rank_text}_{_safe_filename(nanobody_id, fallback=f'candidate_{index:03d}')}.html"
        card_path = cards_dir / file_name
        card_html = _build_card_html(
            nanobody_id=nanobody_id,
            consensus_record=consensus_record,
            rule_record=rule_record,
            ml_record=ml_record,
            feature_record=feature_record,
            pose_rows=pose_rows,
            pose_source_name=pose_source_name,
            pairwise_rows=pairwise_rows,
            pairwise_stats=pairwise_stats,
            generated_at=generated_at,
        )
        card_path.write_text(card_html, encoding="utf-8")

        cards.append(
            {
                "nanobody_id": nanobody_id,
                "consensus_rank": consensus_record.get("consensus_rank"),
                "decision_tier": consensus_record.get("decision_tier"),
                "confidence_level": consensus_record.get("confidence_level"),
                "consensus_score": consensus_record.get("consensus_score"),
                "ml_score": consensus_record.get("ml_score"),
                "rule_score": consensus_record.get("rule_score"),
                "risk_flags": consensus_record.get("risk_flags"),
                **pairwise_stats,
                "card_relpath": str(Path("cards") / file_name).replace("\\", "/"),
                "card_path": str(card_path),
            }
        )

    index_html_path = out_dir / "index.html"
    manifest_csv_path = out_dir / "candidate_report_cards.csv"
    summary_json_path = out_dir / "candidate_report_cards_summary.json"
    index_html_path.write_text(_build_index_html(cards, generated_at), encoding="utf-8")
    pd.DataFrame(cards).to_csv(manifest_csv_path, index=False)

    if zip_path is None:
        zip_path = out_dir.with_suffix(".zip")
    zip_path = zip_path.expanduser().resolve()
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "generated_at": generated_at,
        "candidate_count": int(len(cards)),
        "out_dir": str(out_dir),
        "index_html": str(index_html_path),
        "manifest_csv": str(manifest_csv_path),
        "summary_json": str(summary_json_path),
        "zip_path": str(zip_path),
        "candidate_pairwise_csv": None if candidate_pairwise_csv_path is None else str(Path(candidate_pairwise_csv_path).expanduser().resolve()),
        "pairwise_embedded": bool(candidate_pairwise_df is not None and not candidate_pairwise_df.empty),
        "cards": cards,
    }
    summary_json_path.write_text(json.dumps(_json_sanitize(summary), ensure_ascii=True, indent=2), encoding="utf-8")

    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(out_dir.rglob("*")):
            if file_path.is_file():
                zf.write(file_path, arcname=str(file_path.relative_to(out_dir)).replace("\\", "/"))

    return summary


def main() -> None:
    args = _build_parser().parse_args()
    consensus_path = Path(args.consensus_csv).expanduser().resolve()
    if not consensus_path.exists():
        raise FileNotFoundError(f"consensus_csv not found: {consensus_path}")

    consensus_df = pd.read_csv(consensus_path, low_memory=False)
    summary = build_candidate_report_cards(
        consensus_df=consensus_df,
        out_dir=Path(args.out_dir),
        rule_df=_read_csv_optional(args.rule_csv),
        ml_df=_read_csv_optional(args.ml_csv),
        feature_df=_read_csv_optional(args.feature_csv),
        pose_pred_df=_read_csv_optional(args.pose_predictions_csv),
        candidate_pairwise_df=_read_csv_optional(args.candidate_pairwise_csv),
        candidate_pairwise_csv_path=args.candidate_pairwise_csv,
        top_n=int(args.top_n),
        max_pose_rows=int(args.max_pose_rows),
        zip_path=Path(args.zip_path) if args.zip_path else None,
    )
    print(f"Saved: {summary['index_html']}")
    print(f"Saved: {summary['manifest_csv']}")
    print(f"Saved: {summary['summary_json']}")
    print(f"Saved: {summary['zip_path']}")


if __name__ == "__main__":
    main()
