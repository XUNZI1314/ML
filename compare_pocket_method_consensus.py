"""Compare pocket residue sets from multiple methods.

The script expects each method to be represented as a normalized residue-list
file, for example files produced by ligand-contact, P2Rank or fpocket extractor
wrappers. It does not run external pocket tools.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pocket_io import load_residue_set


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare pocket residue definitions from multiple methods")
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        help="Method residue file as NAME=PATH. Can be repeated, e.g. --method ligand=ligand.txt --method p2rank=p2rank.txt",
    )
    parser.add_argument("--manual_file", default=None, help="Optional manual residue file")
    parser.add_argument("--ligand_contact_file", default=None, help="Optional ligand-contact residue file")
    parser.add_argument("--p2rank_file", default=None, help="Optional P2Rank residue file")
    parser.add_argument("--fpocket_file", default=None, help="Optional fpocket residue file")
    parser.add_argument("--truth_file", default=None, help="Optional ground-truth residue file")
    parser.add_argument("--out_dir", default="pocket_method_consensus", help="Output directory")
    parser.add_argument("--min_method_count", type=int, default=2, help="Minimum method support for consensus residues")
    parser.add_argument("--min_support_fraction", type=float, default=0.0, help="Optional consensus support fraction")
    return parser


def _sort_residue_key(key: str) -> tuple[str, int, str]:
    parts = [part.strip() for part in str(key).split(":")]
    chain = parts[0] if len(parts) >= 1 else ""
    try:
        resseq = int(parts[1]) if len(parts) >= 2 else 0
    except ValueError:
        resseq = 0
    icode = parts[2] if len(parts) >= 3 else ""
    return chain, resseq, icode


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_sanitize(v) for v in value]
    if isinstance(value, tuple):
        return [_json_sanitize(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if np.isfinite(out) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _parse_method_specs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    specs: list[tuple[str, Path]] = []

    named_files = [
        ("manual", args.manual_file),
        ("ligand_contact", args.ligand_contact_file),
        ("p2rank", args.p2rank_file),
        ("fpocket", args.fpocket_file),
    ]
    for name, path_text in named_files:
        if path_text:
            specs.append((name, Path(path_text).expanduser().resolve()))

    for item in args.method or []:
        text = str(item).strip()
        if "=" not in text:
            raise ValueError(f"Invalid --method {text!r}. Expected NAME=PATH.")
        name, path_text = text.split("=", 1)
        name = name.strip()
        path_text = path_text.strip()
        if not name or not path_text:
            raise ValueError(f"Invalid --method {text!r}. Expected non-empty NAME=PATH.")
        specs.append((name, Path(path_text).expanduser().resolve()))

    if len(specs) < 2:
        raise ValueError("At least two pocket methods are required for consensus analysis.")

    seen: set[str] = set()
    out: list[tuple[str, Path]] = []
    for name, path in specs:
        clean_name = str(name).strip()
        if clean_name in seen:
            raise ValueError(f"Duplicate method name: {clean_name}")
        seen.add(clean_name)
        out.append((clean_name, path))
    return out


def _load_methods(method_specs: list[tuple[str, Path]]) -> dict[str, dict[str, Any]]:
    methods: dict[str, dict[str, Any]] = {}
    for name, path in method_specs:
        if not path.exists():
            raise FileNotFoundError(f"Method residue file not found for {name}: {path}")
        residues = set(load_residue_set(str(path)))
        methods[name] = {
            "source_path": str(path),
            "residues": residues,
        }
    return methods


def _pairwise_overlap_rows(methods: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    names = sorted(methods)
    for idx, name_a in enumerate(names):
        set_a = set(methods[name_a]["residues"])
        for name_b in names[idx + 1 :]:
            set_b = set(methods[name_b]["residues"])
            inter = set_a & set_b
            union = set_a | set_b
            rows.append(
                {
                    "method_a": name_a,
                    "method_b": name_b,
                    "count_a": int(len(set_a)),
                    "count_b": int(len(set_b)),
                    "overlap_count": int(len(inter)),
                    "union_count": int(len(union)),
                    "jaccard": float(len(inter) / len(union)) if union else float("nan"),
                    "overlap_fraction_a": float(len(inter) / len(set_a)) if set_a else float("nan"),
                    "overlap_fraction_b": float(len(inter) / len(set_b)) if set_b else float("nan"),
                }
            )
    return rows


def build_pocket_method_consensus(
    methods: dict[str, dict[str, Any]],
    *,
    truth_residues: set[str] | None = None,
    min_method_count: int = 2,
    min_support_fraction: float = 0.0,
) -> dict[str, Any]:
    if len(methods) < 2:
        raise ValueError("At least two methods are required.")

    method_names = sorted(methods)
    method_count = len(method_names)
    requested_min_count = max(1, int(min_method_count))
    fraction_min_count = int(np.ceil(float(np.clip(min_support_fraction, 0.0, 1.0)) * method_count))
    effective_min_count = min(method_count, max(requested_min_count, fraction_min_count, 1))

    union_residues = sorted(set().union(*(set(methods[name]["residues"]) for name in method_names)), key=_sort_residue_key)
    membership_rows: list[dict[str, Any]] = []
    consensus_residues: list[str] = []
    method_specific_rows: list[dict[str, Any]] = []

    for residue_key in union_residues:
        present_methods = [name for name in method_names if residue_key in methods[name]["residues"]]
        support_count = len(present_methods)
        support_fraction = support_count / method_count if method_count > 0 else float("nan")
        is_consensus = support_count >= effective_min_count
        if is_consensus:
            consensus_residues.append(residue_key)
        row: dict[str, Any] = {
            "residue_key": residue_key,
            "support_count": int(support_count),
            "support_fraction": float(support_fraction),
            "methods": ";".join(present_methods),
            "is_consensus": bool(is_consensus),
            "is_method_specific": bool(support_count == 1),
        }
        if truth_residues is not None:
            row["in_truth"] = bool(residue_key in truth_residues)
        for name in method_names:
            row[f"in_{name}"] = bool(residue_key in methods[name]["residues"])
        membership_rows.append(row)

        if support_count == 1:
            method_specific_rows.append(
                {
                    "method": present_methods[0],
                    "residue_key": residue_key,
                    "in_truth": bool(residue_key in truth_residues) if truth_residues is not None else None,
                }
            )

    consensus_set = set(consensus_residues)
    union_set = set(union_residues)
    method_summaries: list[dict[str, Any]] = []
    counts = [len(methods[name]["residues"]) for name in method_names]
    median_count = float(np.median(counts)) if counts else 0.0

    for name in method_names:
        residues = set(methods[name]["residues"])
        consensus_overlap = residues & consensus_set
        method_specific = residues - set().union(*(set(methods[other]["residues"]) for other in method_names if other != name))
        row = {
            "method": name,
            "source_path": methods[name]["source_path"],
            "residue_count": int(len(residues)),
            "consensus_overlap_count": int(len(consensus_overlap)),
            "consensus_overlap_fraction": float(len(consensus_overlap) / len(residues)) if residues else float("nan"),
            "method_specific_count": int(len(method_specific)),
            "count_to_median_ratio": float(len(residues) / median_count) if median_count > 0 else float("nan"),
        }
        if truth_residues is not None:
            truth_overlap = residues & truth_residues
            row.update(
                {
                    "truth_overlap_count": int(len(truth_overlap)),
                    "truth_coverage": float(len(truth_overlap) / len(truth_residues)) if truth_residues else float("nan"),
                    "truth_precision": float(len(truth_overlap) / len(residues)) if residues else float("nan"),
                }
            )
        method_summaries.append(row)

    truth_summary: dict[str, Any] = {
        "truth_available": bool(truth_residues is not None),
    }
    if truth_residues is not None:
        truth_set = set(truth_residues)
        truth_overlap = consensus_set & truth_set
        missing_truth = sorted(truth_set - consensus_set, key=_sort_residue_key)
        extra_consensus = sorted(consensus_set - truth_set, key=_sort_residue_key)
        coverage = float(len(truth_overlap) / len(truth_set)) if truth_set else float("nan")
        precision = float(len(truth_overlap) / len(consensus_set)) if consensus_set else float("nan")
        truth_summary.update(
            {
                "truth_count": int(len(truth_set)),
                "consensus_truth_overlap_count": int(len(truth_overlap)),
                "consensus_truth_coverage": coverage,
                "consensus_truth_precision": precision,
                "missing_truth_count": int(len(missing_truth)),
                "extra_consensus_count": int(len(extra_consensus)),
                "missing_truth_risk": float(1.0 - coverage) if np.isfinite(coverage) else None,
                "overwide_risk": float(1.0 - precision) if np.isfinite(precision) else None,
                "missing_truth_residues": missing_truth,
                "extra_consensus_residues": extra_consensus,
            }
        )
    else:
        expansion = float((len(union_set) - len(consensus_set)) / len(union_set)) if union_set else 0.0
        max_count = max(counts) if counts else 0
        size_dispersion = float(max(0.0, (max_count / median_count) - 1.0)) if median_count > 0 else 0.0
        truth_summary.update(
            {
                "truth_count": None,
                "missing_truth_risk": "unknown_no_truth_file",
                "overwide_risk": float(np.clip(0.70 * expansion + 0.30 * min(size_dispersion, 1.0), 0.0, 1.0)),
                "overwide_risk_interpretation": "truth-free proxy from union-vs-consensus expansion and method size dispersion",
            }
        )

    summary = {
        "method_count": int(method_count),
        "methods": method_names,
        "effective_min_method_count": int(effective_min_count),
        "union_residue_count": int(len(union_residues)),
        "consensus_residue_count": int(len(consensus_residues)),
        "method_specific_residue_count": int(len(method_specific_rows)),
        "all_methods_residue_count": int(sum(1 for row in membership_rows if int(row["support_count"]) == method_count)),
        "truth": truth_summary,
        "method_summaries": method_summaries,
    }

    return {
        "summary": summary,
        "consensus_residues": consensus_residues,
        "union_residues": union_residues,
        "membership_rows": membership_rows,
        "method_specific_rows": method_specific_rows,
        "pairwise_rows": _pairwise_overlap_rows(methods),
    }


def _write_residue_file(path: Path, residues: list[str]) -> None:
    path.write_text("\n".join(residues) + ("\n" if residues else ""), encoding="utf-8")


def _build_report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    truth = summary["truth"]
    lines = [
        "# Pocket Method Consensus Analysis",
        "",
        f"- Methods: `{summary['method_count']}` ({', '.join(summary['methods'])})",
        f"- Union residues: `{summary['union_residue_count']}`",
        f"- Consensus residues: `{summary['consensus_residue_count']}`",
        f"- Method-specific residues: `{summary['method_specific_residue_count']}`",
        f"- Effective minimum method count: `{summary['effective_min_method_count']}`",
        "",
        "## Risk Notes",
        "",
    ]
    if truth.get("truth_available"):
        lines.extend(
            [
                f"- Truth residues: `{truth.get('truth_count')}`",
                f"- Consensus truth coverage: `{truth.get('consensus_truth_coverage'):.4f}`",
                f"- Consensus truth precision: `{truth.get('consensus_truth_precision'):.4f}`",
                f"- Missing truth risk: `{truth.get('missing_truth_risk'):.4f}`",
                f"- Overwide risk: `{truth.get('overwide_risk'):.4f}`",
            ]
        )
        missing = truth.get("missing_truth_residues") or []
        extra = truth.get("extra_consensus_residues") or []
        if missing:
            lines.append(f"- Missing truth residues: `{', '.join(missing)}`")
        if extra:
            lines.append(f"- Extra consensus residues: `{', '.join(extra)}`")
    else:
        lines.extend(
            [
                "- Truth file not provided, so missing-truth risk is unknown.",
                f"- Truth-free overwide proxy: `{truth.get('overwide_risk'):.4f}`",
                f"- Interpretation: {truth.get('overwide_risk_interpretation')}",
            ]
        )

    lines.extend(
        [
            "",
            "## Method Summary",
            "",
            "| method | residues | consensus_overlap | method_specific | count_to_median |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["method_summaries"]:
        lines.append(
            "| "
            f"{row['method']} | "
            f"{row['residue_count']} | "
            f"{row['consensus_overlap_count']} | "
            f"{row['method_specific_count']} | "
            f"{row.get('count_to_median_ratio', float('nan')):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Pairwise Overlap",
            "",
            "| method_a | method_b | overlap | jaccard |",
            "|---|---|---:|---:|",
        ]
    )
    for row in result["pairwise_rows"]:
        lines.append(
            "| "
            f"{row['method_a']} | "
            f"{row['method_b']} | "
            f"{row['overlap_count']} | "
            f"{row['jaccard']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `consensus_pocket_residues.txt`: residues supported by the configured minimum number of methods.",
            "- `union_pocket_residues.txt`: all residues found by at least one method.",
            "- `residue_method_membership.csv`: per-residue support across methods.",
            "- `method_specific_residues.csv`: residues found by exactly one method.",
            "- `method_overlap_matrix.csv`: pairwise method overlap metrics.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = _build_parser().parse_args()
    method_specs = _parse_method_specs(args)
    methods = _load_methods(method_specs)

    truth_residues: set[str] | None = None
    truth_path: Path | None = None
    if args.truth_file:
        truth_path = Path(args.truth_file).expanduser().resolve()
        if not truth_path.exists():
            raise FileNotFoundError(f"truth_file not found: {truth_path}")
        truth_residues = set(load_residue_set(str(truth_path)))

    result = build_pocket_method_consensus(
        methods,
        truth_residues=truth_residues,
        min_method_count=int(args.min_method_count),
        min_support_fraction=float(args.min_support_fraction),
    )
    result["summary"]["truth"]["truth_file"] = None if truth_path is None else str(truth_path)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    consensus_txt = out_dir / "consensus_pocket_residues.txt"
    union_txt = out_dir / "union_pocket_residues.txt"
    membership_csv = out_dir / "residue_method_membership.csv"
    method_specific_csv = out_dir / "method_specific_residues.csv"
    overlap_csv = out_dir / "method_overlap_matrix.csv"
    summary_json = out_dir / "pocket_method_consensus_summary.json"
    report_md = out_dir / "pocket_method_consensus_report.md"

    _write_residue_file(consensus_txt, result["consensus_residues"])
    _write_residue_file(union_txt, result["union_residues"])
    pd.DataFrame(result["membership_rows"]).to_csv(membership_csv, index=False)
    pd.DataFrame(result["method_specific_rows"]).to_csv(method_specific_csv, index=False)
    pd.DataFrame(result["pairwise_rows"]).to_csv(overlap_csv, index=False)
    summary_json.write_text(json.dumps(_json_sanitize(result["summary"]), ensure_ascii=True, indent=2), encoding="utf-8")
    report_md.write_text(_build_report(result), encoding="utf-8")

    for path in [
        consensus_txt,
        union_txt,
        membership_csv,
        method_specific_csv,
        overlap_csv,
        summary_json,
        report_md,
    ]:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
