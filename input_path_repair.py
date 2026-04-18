from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


PATH_COLUMNS = ("pdb_path", "pocket_file", "catalytic_file", "ligand_file")
AUTO_REPAIR_CONFIDENCE_THRESHOLD = 0.70
DEFAULT_MAX_INDEXED_FILES = 50000
SKIP_DIR_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "node_modules",
    "result_archive",
}


def _clean_cell_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


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


def _normalize_search_roots(search_roots: Iterable[str | Path] | None, base_dir: Path | None) -> list[Path]:
    roots: list[Path] = []
    for root in list(search_roots or []) + ([base_dir] if base_dir is not None else []):
        if root is None:
            continue
        path = Path(root).expanduser().resolve()
        if path.exists() and path.is_dir() and path not in roots:
            roots.append(path)
    return roots


def _iter_indexable_files(search_roots: list[Path], *, max_files: int) -> tuple[list[Path], bool]:
    files: list[Path] = []
    truncated = False
    for root in search_roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [name for name in dirnames if name not in SKIP_DIR_NAMES and not name.startswith(".")]
            for filename in filenames:
                files.append(Path(dirpath) / filename)
                if len(files) >= max_files:
                    return files, True
    return files, truncated


def _build_file_index(search_roots: list[Path], *, max_files: int = DEFAULT_MAX_INDEXED_FILES) -> dict[str, Any]:
    files, truncated = _iter_indexable_files(search_roots, max_files=max_files)
    by_name: dict[str, list[Path]] = {}
    by_stem: dict[str, list[Path]] = {}
    for path in files:
        by_name.setdefault(path.name.lower(), []).append(path.resolve())
        by_stem.setdefault(path.stem.lower(), []).append(path.resolve())
    return {
        "search_roots": [str(root) for root in search_roots],
        "indexed_file_count": int(len(files)),
        "index_truncated": bool(truncated),
        "by_name": by_name,
        "by_stem": by_stem,
    }


def _common_tail_parts(left: Path, right: Path) -> int:
    left_parts = [part.lower() for part in left.parts]
    right_parts = [part.lower() for part in right.parts]
    count = 0
    for left_part, right_part in zip(reversed(left_parts), reversed(right_parts)):
        if left_part != right_part:
            break
        count += 1
    return count


def _best_candidate(original_value: str, file_index: dict[str, Any]) -> dict[str, Any]:
    original_path = Path(original_value)
    name_key = original_path.name.lower()
    stem_key = original_path.stem.lower()
    by_name = file_index.get("by_name") if isinstance(file_index.get("by_name"), dict) else {}
    by_stem = file_index.get("by_stem") if isinstance(file_index.get("by_stem"), dict) else {}

    exact_candidates = list(by_name.get(name_key, []))
    stem_candidates = list(by_stem.get(stem_key, [])) if stem_key else []
    candidates = exact_candidates or stem_candidates
    if not candidates:
        return {
            "suggested_path": "",
            "suggestion_confidence": 0.0,
            "suggestion_reason": "no_filename_match",
            "candidate_count": 0,
        }

    scored: list[tuple[int, str, Path]] = []
    for candidate in candidates:
        tail_score = _common_tail_parts(original_path, candidate)
        scored.append((tail_score, str(candidate).lower(), candidate))
    scored.sort(key=lambda item: (-item[0], item[1]))
    best_tail, _, best_path = scored[0]
    candidate_count = len(candidates)
    has_unique_best_tail = candidate_count == 1 or best_tail > scored[1][0]

    if exact_candidates and candidate_count == 1:
        confidence = 0.96
        reason = "unique_exact_filename"
    elif exact_candidates and best_tail >= 2 and has_unique_best_tail:
        confidence = 0.90
        reason = "exact_filename_with_suffix_context"
    elif exact_candidates and has_unique_best_tail:
        confidence = 0.78
        reason = "exact_filename_best_context"
    elif exact_candidates:
        confidence = 0.58
        reason = "multiple_exact_filename_candidates"
    elif candidate_count == 1:
        confidence = 0.72
        reason = "unique_stem_match"
    elif has_unique_best_tail:
        confidence = 0.62
        reason = "stem_match_best_context"
    else:
        confidence = 0.45
        reason = "multiple_stem_match_candidates"

    return {
        "suggested_path": str(best_path),
        "suggestion_confidence": round(float(confidence), 3),
        "suggestion_reason": reason,
        "candidate_count": int(candidate_count),
    }


def analyze_input_path_repair_dataframe(
    df: pd.DataFrame,
    *,
    base_dir: str | Path | None,
    search_roots: Iterable[str | Path] | None = None,
    path_columns: Iterable[str] = PATH_COLUMNS,
    max_indexed_files: int = DEFAULT_MAX_INDEXED_FILES,
    auto_repair_confidence_threshold: float = AUTO_REPAIR_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    base_path = Path(base_dir).expanduser().resolve() if base_dir is not None else None
    roots = _normalize_search_roots(search_roots, base_path)
    file_index = _build_file_index(roots, max_files=max_indexed_files) if roots else {
        "search_roots": [],
        "indexed_file_count": 0,
        "index_truncated": False,
        "by_name": {},
        "by_stem": {},
    }

    rows: list[dict[str, Any]] = []
    for row_index, row in df.iterrows():
        for column in path_columns:
            if column not in df.columns:
                continue
            original_value = _clean_cell_text(row.get(column))
            if not original_value:
                if column == "pdb_path":
                    rows.append(
                        {
                            "row_index": int(row_index),
                            "column": column,
                            "original_value": "",
                            "resolved_path": "",
                            "issue": "missing_required_value",
                            "suggested_path": "",
                            "suggestion_confidence": 0.0,
                            "suggestion_reason": "empty_value",
                            "candidate_count": 0,
                            "auto_repair": False,
                            "recommended_action": "fill_required_path_manually",
                        }
                    )
                continue

            resolved = _resolve_reference_path(original_value, base_path)
            if resolved is not None and resolved.exists():
                continue

            suggestion = _best_candidate(original_value, file_index)
            confidence = float(suggestion.get("suggestion_confidence") or 0.0)
            auto_repair = bool(confidence >= auto_repair_confidence_threshold and suggestion.get("suggested_path"))
            rows.append(
                {
                    "row_index": int(row_index),
                    "column": column,
                    "original_value": original_value,
                    "resolved_path": "" if resolved is None else str(resolved),
                    "issue": "unchecked_relative_path" if resolved is None else "missing_path",
                    "suggested_path": str(suggestion.get("suggested_path") or ""),
                    "suggestion_confidence": confidence,
                    "suggestion_reason": str(suggestion.get("suggestion_reason") or ""),
                    "candidate_count": int(suggestion.get("candidate_count") or 0),
                    "auto_repair": auto_repair,
                    "recommended_action": "replace_with_suggested_path" if auto_repair else "manual_review",
                }
            )

    plan_df = pd.DataFrame(rows)
    if plan_df.empty:
        plan_df = pd.DataFrame(
            columns=[
                "row_index",
                "column",
                "original_value",
                "resolved_path",
                "issue",
                "suggested_path",
                "suggestion_confidence",
                "suggestion_reason",
                "candidate_count",
                "auto_repair",
                "recommended_action",
            ]
        )

    missing_ref_count = int(len(plan_df))
    suggested_count = int(plan_df["suggested_path"].fillna("").astype(str).str.len().gt(0).sum()) if not plan_df.empty else 0
    auto_repair_count = int(plan_df["auto_repair"].fillna(False).sum()) if not plan_df.empty else 0
    required_missing_count = (
        int(((plan_df["column"] == "pdb_path") & (plan_df["issue"] == "missing_required_value")).sum())
        if not plan_df.empty
        else 0
    )
    return {
        "summary": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "row_count": int(len(df)),
            "checked_columns": [str(column) for column in path_columns if column in df.columns],
            "search_roots": file_index["search_roots"],
            "indexed_file_count": int(file_index["indexed_file_count"]),
            "index_truncated": bool(file_index["index_truncated"]),
            "missing_reference_count": missing_ref_count,
            "suggested_replacement_count": suggested_count,
            "auto_repair_count": auto_repair_count,
            "required_missing_value_count": required_missing_count,
            "auto_repair_confidence_threshold": float(auto_repair_confidence_threshold),
        },
        "plan_rows": plan_df.to_dict(orient="records"),
    }


def apply_input_path_repair_plan(
    df: pd.DataFrame,
    plan_rows: Iterable[dict[str, Any]],
    *,
    min_confidence: float = AUTO_REPAIR_CONFIDENCE_THRESHOLD,
) -> pd.DataFrame:
    repaired = df.copy()
    for row in plan_rows:
        try:
            row_index = int(row.get("row_index"))
        except (TypeError, ValueError):
            continue
        column = str(row.get("column") or "")
        suggested_path = str(row.get("suggested_path") or "").strip()
        confidence = float(row.get("suggestion_confidence") or 0.0)
        if not suggested_path or not column or column not in repaired.columns:
            continue
        if confidence < min_confidence:
            continue
        if row_index not in repaired.index:
            continue
        repaired.at[row_index, column] = suggested_path
    return repaired


def _build_markdown_report(summary: dict[str, Any], plan_df: pd.DataFrame) -> str:
    lines = [
        "# Input Path Repair Report",
        "",
        f"- Generated at: `{summary.get('generated_at', '')}`",
        f"- Rows: `{summary.get('row_count', 0)}`",
        f"- Indexed files: `{summary.get('indexed_file_count', 0)}`",
        f"- Missing references: `{summary.get('missing_reference_count', 0)}`",
        f"- Suggested replacements: `{summary.get('suggested_replacement_count', 0)}`",
        f"- Auto-repairable rows: `{summary.get('auto_repair_count', 0)}`",
        "",
        "## Search Roots",
        "",
    ]
    roots = summary.get("search_roots") if isinstance(summary.get("search_roots"), list) else []
    if roots:
        lines.extend([f"- `{root}`" for root in roots])
    else:
        lines.append("_No search roots were available._")
    lines.extend(["", "## Repair Plan", ""])
    if plan_df.empty:
        lines.append("_No missing path references were detected._")
    else:
        preview = plan_df.head(30)
        columns = ["row_index", "column", "original_value", "suggested_path", "suggestion_confidence", "recommended_action"]
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
        for _, row in preview.iterrows():
            lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    lines.append("")
    return "\n".join(lines)


def build_input_path_repair_outputs(
    *,
    input_csv: str | Path,
    out_dir: str | Path | None = None,
    search_roots: Iterable[str | Path] | None = None,
    write_repaired_csv: bool = False,
    max_indexed_files: int = DEFAULT_MAX_INDEXED_FILES,
) -> dict[str, Any]:
    input_path = Path(input_csv).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"input_csv not found: {input_path}")
    df = _read_csv(input_path)
    result = analyze_input_path_repair_dataframe(
        df,
        base_dir=input_path.parent,
        search_roots=search_roots,
        max_indexed_files=max_indexed_files,
    )
    output_dir = Path(out_dir).expanduser().resolve() if out_dir is not None else input_path.parent / "input_path_repair"
    output_dir.mkdir(parents=True, exist_ok=True)

    plan_df = pd.DataFrame(result["plan_rows"])
    plan_csv = output_dir / "input_path_repair_plan.csv"
    summary_json = output_dir / "input_path_repair_summary.json"
    report_md = output_dir / "input_path_repair_report.md"
    plan_df.to_csv(plan_csv, index=False)
    summary = dict(result["summary"])
    outputs = {
        "plan_csv": str(plan_csv),
        "summary_json": str(summary_json),
        "report_md": str(report_md),
    }
    if write_repaired_csv:
        repaired_df = apply_input_path_repair_plan(df, result["plan_rows"])
        repaired_csv = output_dir / f"{input_path.stem}_repaired.csv"
        repaired_df.to_csv(repaired_csv, index=False)
        outputs["repaired_csv"] = str(repaired_csv)
    summary["input_csv"] = str(input_path)
    summary["outputs"] = outputs
    summary_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    report_md.write_text(_build_markdown_report(summary, plan_df), encoding="utf-8")
    return {
        "summary": summary,
        "plan_rows": result["plan_rows"],
        "outputs": outputs,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Find missing input_csv path references and suggest safe replacements.")
    parser.add_argument("--input_csv", required=True, help="Path to input_pose_table.csv")
    parser.add_argument("--out_dir", default=None, help="Output directory for repair plan files")
    parser.add_argument(
        "--search_root",
        action="append",
        default=None,
        help="Directory to recursively search. Can be provided multiple times. Defaults to input_csv parent.",
    )
    parser.add_argument("--write_repaired_csv", action="store_true", help="Also write an auto-repaired CSV.")
    parser.add_argument("--max_indexed_files", type=int, default=DEFAULT_MAX_INDEXED_FILES)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = build_input_path_repair_outputs(
        input_csv=args.input_csv,
        out_dir=args.out_dir,
        search_roots=args.search_root,
        write_repaired_csv=bool(args.write_repaired_csv),
        max_indexed_files=int(args.max_indexed_files),
    )
    outputs = result["outputs"]
    for key, value in outputs.items():
        print(f"Saved {key}: {value}")
    summary = result["summary"]
    print(
        "Input path repair: "
        f"missing={summary.get('missing_reference_count', 0)}, "
        f"suggested={summary.get('suggested_replacement_count', 0)}, "
        f"auto_repair={summary.get('auto_repair_count', 0)}"
    )


if __name__ == "__main__":
    main()
