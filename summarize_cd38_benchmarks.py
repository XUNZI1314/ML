from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate structured CD38 benchmark result directories into one summary table.")
    parser.add_argument("--results_root", default="benchmarks/cd38/results", help="Root directory containing benchmark result folders")
    parser.add_argument("--out_csv", default=None, help="Optional output CSV path")
    parser.add_argument("--out_md", default=None, help="Optional output Markdown path")
    return parser


def _stringify_list(values: Any) -> str:
    if values is None:
        return ""
    if isinstance(values, (list, tuple)):
        return ",".join(str(x) for x in values if str(x).strip())
    return str(values)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _infer_pdb_id(manifest: dict[str, Any], summary: dict[str, Any]) -> str:
    explicit = str(manifest.get("rcsb_pdb_id") or summary.get("rcsb_pdb_id") or "").strip().upper()
    if explicit:
        return explicit
    for raw_path in (summary.get("pdb_path"), manifest.get("pdb_path")):
        if not raw_path:
            continue
        stem = Path(str(raw_path)).stem.strip().upper()
        if len(stem) == 4 and stem.isalnum():
            return stem
    return ""


def _collect_one_result_dir(result_dir: Path) -> dict[str, Any] | None:
    summary_path = result_dir / "cd38_pocket_accuracy_summary.json"
    manifest_path = result_dir / "run_manifest.json"
    if not summary_path.exists():
        return None

    summary = _load_json(summary_path)
    manifest = _load_json(manifest_path) if manifest_path.exists() else {}
    metrics = dict(summary.get("metrics") or {})
    counts = dict(summary.get("counts") or {})

    row = {
        "result_dir": str(result_dir),
        "result_name": result_dir.name,
        "method": str(manifest.get("method") or "").strip() or "unknown",
        "pdb_id": _infer_pdb_id(manifest, summary),
        "pdb_path": str(summary.get("pdb_path") or ""),
        "protein_chain": str(manifest.get("protein_chain") or manifest.get("chain_filter") or "").strip(),
        "ligand_chain": str(manifest.get("ligand_chain") or "").strip(),
        "ligand_resnames": _stringify_list(manifest.get("ligand_resnames")),
        "ligand_resseqs": _stringify_list(manifest.get("ligand_resseqs")),
        "p2rank_rank": manifest.get("rank"),
        "p2rank_name": str(manifest.get("name") or "").strip(),
        "predicted_residue_count": int(counts.get("predicted_matched_residues") or 0),
        "truth_residue_count": int(counts.get("truth_matched_residues") or 0),
        "exact_overlap_residue_count": int(counts.get("exact_overlap_residues") or 0),
        "exact_truth_coverage": metrics.get("exact_truth_coverage"),
        "exact_predicted_precision": metrics.get("exact_predicted_precision"),
        "exact_jaccard": metrics.get("exact_jaccard"),
        "exact_f1": metrics.get("exact_f1"),
        "truth_near_coverage": metrics.get("truth_near_coverage"),
        "predicted_near_precision": metrics.get("predicted_near_precision"),
        "mean_truth_min_distance": metrics.get("mean_truth_min_distance"),
        "predicted_to_truth_residue_ratio": metrics.get("predicted_to_truth_residue_ratio"),
        "extra_predicted_fraction": metrics.get("extra_predicted_fraction"),
        "far_predicted_fraction": metrics.get("far_predicted_fraction"),
        "overwide_pocket_score": metrics.get("overwide_pocket_score"),
        "near_threshold": metrics.get("near_threshold"),
    }
    return row


def _build_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "# CD38 Benchmark Panel\n\nNo benchmark result directories were found.\n"

    display_cols = [
        "result_name",
        "method",
        "pdb_id",
        "protein_chain",
        "ligand_resnames",
        "p2rank_rank",
        "predicted_residue_count",
        "exact_truth_coverage",
        "exact_predicted_precision",
        "exact_f1",
        "truth_near_coverage",
        "predicted_near_precision",
        "predicted_to_truth_residue_ratio",
        "overwide_pocket_score",
    ]
    work = df.loc[:, display_cols].copy()
    for col in [
        "exact_truth_coverage",
        "exact_predicted_precision",
        "exact_f1",
        "truth_near_coverage",
        "predicted_near_precision",
        "predicted_to_truth_residue_ratio",
        "overwide_pocket_score",
    ]:
        work[col] = pd.to_numeric(work[col], errors="coerce").map(lambda x: f"{float(x):.4f}" if pd.notna(x) else "")
    rank_series = pd.to_numeric(work["p2rank_rank"], errors="coerce")
    work["p2rank_rank"] = rank_series.map(lambda x: str(int(x)) if pd.notna(x) else "")
    work["predicted_residue_count"] = pd.to_numeric(work["predicted_residue_count"], errors="coerce").fillna(0).astype(int)

    lines = ["# CD38 Benchmark Panel", ""]
    headers = work.columns.tolist()
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in work.iterrows():
        values = [str(row[col]) if pd.notna(row[col]) else "" for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _build_parser().parse_args()
    results_root = Path(args.results_root).expanduser().resolve()
    if not results_root.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")

    rows: list[dict[str, Any]] = []
    for child in sorted(results_root.iterdir()):
        if not child.is_dir():
            continue
        row = _collect_one_result_dir(child)
        if row is not None:
            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            by=["pdb_id", "method", "protein_chain", "result_name"],
            ascending=[True, True, True, True],
            na_position="last",
        ).reset_index(drop=True)

    out_csv = Path(args.out_csv).expanduser().resolve() if args.out_csv else results_root / "cd38_benchmark_panel.csv"
    out_md = Path(args.out_md).expanduser().resolve() if args.out_md else results_root / "cd38_benchmark_panel.md"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_csv, index=False)
    out_md.write_text(_build_markdown_table(df), encoding="utf-8")

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_md}")
    print(f"Benchmarks aggregated: {len(df)}")


if __name__ == "__main__":
    main()
