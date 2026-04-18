from __future__ import annotations

import argparse
import json
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from build_pocket_evidence import build_pocket_evidence
from pocket_io import load_residue_set
from result_tree_io import (
    DEFAULT_ANTIGEN_CHAIN,
    DEFAULT_NANOBODY_CHAIN,
    build_input_table_from_result_tree,
    find_result_tree_root,
)


DEFAULT_POCKET_NAMES = (
    "rsite/rsite.txt",
    "rsite.txt",
    "cd38_pocket_from_rsite_chain_B.txt",
    "pocket.txt",
    "pocket_residues.txt",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build project-level pocket evidence from a canonical result/ tree, "
            "then write an input pose table using the curated candidate pocket."
        )
    )
    parser.add_argument("--project_root", default=".", help="Project parent directory containing result/ or rsite/result/.")
    parser.add_argument("--result_root", default=None, help="Optional explicit result tree root.")
    parser.add_argument("--out_dir", default=None, help="Output directory. Default: <project_root>/pocket_evidence_outputs.")
    parser.add_argument(
        "--input_csv_out",
        default=None,
        help="Generated input pose table. Default: <project_root>/input_pose_table_with_pocket_evidence.csv.",
    )
    parser.add_argument("--manual_pocket_file", default=None, help="Manual/rsite pocket residue file.")
    parser.add_argument("--literature_file", default=None, help="Literature-curated residue file.")
    parser.add_argument("--catalytic_file", default=None, help="Catalytic/function residue file.")
    parser.add_argument("--literature_source_table", default=None, help="Optional CSV/TSV source audit table for literature_file.")
    parser.add_argument("--catalytic_source_table", default=None, help="Optional CSV/TSV source audit table for catalytic_file.")
    parser.add_argument("--ai_pocket_file", default=None, help="AI prior residue file.")
    parser.add_argument(
        "--ai_source_table",
        "--ai_prior_source_table",
        dest="ai_source_table",
        default=None,
        help="Optional CSV/TSV offline audit table for AI prior residues.",
    )
    parser.add_argument("--p2rank_file", default=None, help="P2Rank predictions.csv, directory, or residue list.")
    parser.add_argument("--fpocket_file", default=None, help="fpocket output, pocket PDB, or residue list.")
    parser.add_argument("--ligand_file", default=None, help="Ligand/template PDB in representative PDB coordinate frame.")
    parser.add_argument("--antigen_chain", default=DEFAULT_ANTIGEN_CHAIN)
    parser.add_argument("--nanobody_chain", default=DEFAULT_NANOBODY_CHAIN)
    parser.add_argument(
        "--target_prefix",
        default="",
        help="Target folder prefix filter. Empty default includes any target such as CD38_1 or other proteins.",
    )
    parser.add_argument("--path_mode", choices=["absolute", "relative"], default="absolute")
    parser.add_argument("--allow_single_pdb_fallback", action="store_true")
    parser.add_argument(
        "--representative_strategy",
        choices=["min_mmpbsa", "first"],
        default="min_mmpbsa",
        help="How to select one representative PDB from the result tree.",
    )
    parser.add_argument("--anchor_shell_radii", default="4,6,8")
    parser.add_argument("--ligand_contact_threshold", type=float, default=4.5)
    parser.add_argument("--curated_min_support", type=float, default=1.20)
    parser.add_argument("--external_overwide_max_residue_count", type=int, default=35)
    parser.add_argument("--external_overwide_max_fraction", type=float, default=0.18)
    parser.add_argument("--disable_external_precision_guard", action="store_true")
    parser.add_argument("--p2rank_top_n", type=int, default=1)
    parser.add_argument("--p2rank_rank", type=int, default=None)
    parser.add_argument("--p2rank_name", default=None)
    parser.add_argument("--p2rank_min_probability", type=float, default=None)
    return parser


def _sort_residue_key(key: str) -> tuple[str, int, str]:
    parts = [part.strip() for part in str(key).split(":")]
    chain = parts[0] if parts else ""
    try:
        resseq = int(parts[1]) if len(parts) > 1 else 0
    except ValueError:
        resseq = 0
    icode = parts[2] if len(parts) > 2 else ""
    return chain, resseq, icode


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_sanitize(v) for v in value]
    if isinstance(value, Path):
        return str(value)
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


def _resolve_optional_path(path_text: str | None, *, base_dir: Path) -> Path | None:
    text = str(path_text or "").strip().strip('"').strip("'")
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute() and not path.exists():
        path = base_dir / path
    return path.resolve()


def _find_default_pocket_file(project_root: Path, result_root: Path) -> Path | None:
    candidates: list[Path] = []
    for base in [project_root, result_root, result_root.parent, Path.cwd()]:
        for rel_name in DEFAULT_POCKET_NAMES:
            candidates.append(base / rel_name)

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            path = candidate.expanduser().resolve()
        except OSError:
            continue
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            return path
    return None


def _write_chain_filtered_residue_file(
    source: Path | None,
    *,
    out_dir: Path,
    antigen_chain: str,
    label: str,
) -> tuple[Path | None, dict[str, Any]]:
    if source is None:
        return None, {
            "label": label,
            "mode": "not_found",
            "source_file": None,
            "filtered_file": None,
            "input_residue_count": 0,
            "kept_residue_count": 0,
        }

    keys = set(load_residue_set(str(source)))
    chain = str(antigen_chain or "").strip()
    filtered = sorted(
        [key for key in keys if not chain or str(key).split(":", 1)[0] == chain],
        key=_sort_residue_key,
    )
    if not filtered:
        return None, {
            "label": label,
            "mode": "empty_after_antigen_chain_filter",
            "source_file": str(source),
            "filtered_file": None,
            "input_residue_count": int(len(keys)),
            "kept_residue_count": 0,
            "antigen_chain": chain,
        }

    auto_dir = out_dir / "auto_inputs"
    auto_dir.mkdir(parents=True, exist_ok=True)
    filtered_path = auto_dir / f"{label}_antigen_{chain or 'all'}.txt"
    filtered_path.write_text("\n".join(filtered) + "\n", encoding="utf-8")
    return filtered_path.resolve(), {
        "label": label,
        "mode": "filtered_to_antigen_chain",
        "source_file": str(source),
        "filtered_file": str(filtered_path.resolve()),
        "input_residue_count": int(len(keys)),
        "kept_residue_count": int(len(filtered)),
        "antigen_chain": chain,
    }


def _select_representative_pose(input_df: pd.DataFrame, *, strategy: str) -> tuple[pd.Series, dict[str, Any]]:
    if input_df.empty:
        raise ValueError("No valid pose rows were discovered from the result tree.")

    work = input_df.copy()
    work["_row_order"] = range(len(work))
    selected_idx: int
    selection_reason: str

    energy = pd.to_numeric(work.get("MMPBSA_energy"), errors="coerce")
    if strategy == "min_mmpbsa" and energy.notna().any():
        selected_idx = int(energy.idxmin())
        selection_reason = "minimum_MMPBSA_energy"
    else:
        sort_cols = [col for col in ["nanobody_id", "target_variant_index", "conformer_id", "pose_index", "pose_id"] if col in work.columns]
        selected_idx = int(work.sort_values(by=sort_cols + ["_row_order"], kind="mergesort").index[0]) if sort_cols else int(work.index[0])
        selection_reason = "first_pose_fallback"

    row = input_df.loc[selected_idx]
    summary = {
        "selection_reason": selection_reason,
        "representative_row_index": int(selected_idx),
        "representative_pdb_path": str(row.get("pdb_path", "")),
        "nanobody_id": str(row.get("nanobody_id", "")),
        "conformer_id": str(row.get("conformer_id", "")),
        "pose_id": str(row.get("pose_id", "")),
        "MMPBSA_energy": None
        if pd.isna(row.get("MMPBSA_energy", np.nan))
        else float(row.get("MMPBSA_energy")),
    }
    return row, summary


def _format_generated_path(path: Path | None, *, base_dir: Path, path_mode: str) -> str:
    if path is None:
        return ""
    resolved = path.expanduser().resolve()
    if str(path_mode).lower() == "relative":
        try:
            return str(resolved.relative_to(base_dir.resolve()))
        except ValueError:
            return str(resolved)
    return str(resolved)


def _count_residue_file(path: Path) -> int:
    if not path.exists() or not path.is_file():
        return 0
    return len([line for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()])


def _write_project_report(summary: dict[str, Any], report_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Project Pocket Evidence Report")
    lines.append("")
    lines.append("This report selects one representative PDB from a standard result tree, builds project-level pocket evidence, and writes an input table using the curated candidate pocket.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Project root: `{summary['project_root']}`")
    lines.append(f"- Result root: `{summary['result_root']}`")
    lines.append(f"- Representative PDB: `{summary['representative_selection']['representative_pdb_path']}`")
    lines.append(f"- Selection reason: `{summary['representative_selection']['selection_reason']}`")
    lines.append(f"- Input rows written: {summary['generated_input_table']['row_count']}")
    lines.append(f"- Candidate pocket residues: {summary['candidate_pocket']['residue_count']}")
    source_audit = (
        summary.get("pocket_evidence_summary", {}).get("source_audit", {})
        if isinstance(summary.get("pocket_evidence_summary"), dict)
        else {}
    )
    if source_audit and source_audit.get("enabled"):
        lines.append(f"- Source audit rows: {source_audit.get('audit_row_count', 0)}")
        lines.append(f"- Missing traceable source rows: {source_audit.get('missing_traceable_source_count', 0)}")
    precision_guard = (
        summary.get("pocket_evidence_summary", {}).get("external_precision_guard", {})
        if isinstance(summary.get("pocket_evidence_summary"), dict)
        else {}
    )
    if precision_guard and precision_guard.get("enabled"):
        lines.append(f"- External precision guard threshold: {precision_guard.get('threshold_count', 0)}")
        lines.append(f"- External precision guarded residues: {precision_guard.get('guarded_residue_count', 0)}")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    for key, value in summary.get("output_files", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    warnings = summary.get("warnings") or []
    if warnings:
        lines.append("")
        lines.append("## Warnings")
        lines.append("")
        for warning in warnings:
            lines.append(f"- {warning}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_project_pocket_evidence(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(args.project_root).expanduser().resolve()
    if not project_root.exists() or not project_root.is_dir():
        raise NotADirectoryError(f"project_root is not a directory: {project_root}")

    target_prefix = str(args.target_prefix or "")
    if args.result_root:
        result_root = Path(args.result_root).expanduser().resolve()
    else:
        discovered, discovery_mode = find_result_tree_root(project_root, target_prefix=target_prefix)
        if discovered is None:
            raise FileNotFoundError(f"Could not find result/ tree under {project_root}")
        result_root = discovered

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else project_root / "pocket_evidence_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    input_csv_out = (
        Path(args.input_csv_out).expanduser().resolve()
        if args.input_csv_out
        else project_root / "input_pose_table_with_pocket_evidence.csv"
    )

    input_df, result_summary = build_input_table_from_result_tree(
        result_root,
        default_antigen_chain=str(args.antigen_chain or DEFAULT_ANTIGEN_CHAIN),
        default_nanobody_chain=str(args.nanobody_chain or DEFAULT_NANOBODY_CHAIN),
        path_mode="absolute",
        allow_single_pdb_fallback=bool(args.allow_single_pdb_fallback),
        target_prefix=target_prefix,
    )
    representative_row, representative_summary = _select_representative_pose(
        input_df,
        strategy=str(args.representative_strategy),
    )
    representative_pdb = Path(str(representative_row["pdb_path"])).expanduser().resolve()
    if not representative_pdb.exists():
        raise FileNotFoundError(f"Representative PDB not found: {representative_pdb}")

    manual_source = _resolve_optional_path(args.manual_pocket_file, base_dir=project_root)
    if manual_source is None:
        manual_source = _find_default_pocket_file(project_root, result_root)
    literature_source = _resolve_optional_path(args.literature_file, base_dir=project_root)
    catalytic_source = _resolve_optional_path(args.catalytic_file, base_dir=project_root)
    literature_source_table = _resolve_optional_path(args.literature_source_table, base_dir=project_root)
    catalytic_source_table = _resolve_optional_path(args.catalytic_source_table, base_dir=project_root)
    ai_source = _resolve_optional_path(args.ai_pocket_file, base_dir=project_root)
    ai_source_table = _resolve_optional_path(args.ai_source_table, base_dir=project_root)

    manual_file, manual_info = _write_chain_filtered_residue_file(
        manual_source,
        out_dir=out_dir,
        antigen_chain=str(args.antigen_chain or DEFAULT_ANTIGEN_CHAIN),
        label="manual_pocket",
    )
    literature_file, literature_info = _write_chain_filtered_residue_file(
        literature_source,
        out_dir=out_dir,
        antigen_chain=str(args.antigen_chain or DEFAULT_ANTIGEN_CHAIN),
        label="literature_residue",
    )
    catalytic_file, catalytic_info = _write_chain_filtered_residue_file(
        catalytic_source,
        out_dir=out_dir,
        antigen_chain=str(args.antigen_chain or DEFAULT_ANTIGEN_CHAIN),
        label="catalytic_residue",
    )
    ai_file, ai_info = _write_chain_filtered_residue_file(
        ai_source,
        out_dir=out_dir,
        antigen_chain=str(args.antigen_chain or DEFAULT_ANTIGEN_CHAIN),
        label="ai_prior",
    )

    evidence_source_count = sum(
        bool(path)
        for path in [
            manual_file,
            literature_file,
            catalytic_file,
            ai_file,
            args.p2rank_file,
            args.fpocket_file,
            args.ligand_file,
        ]
    )
    if evidence_source_count == 0:
        raise ValueError("No pocket evidence source was provided or auto-detected.")

    evidence_args = Namespace(
        pdb_path=str(representative_pdb),
        out_dir=str(out_dir),
        antigen_chain=str(args.antigen_chain or DEFAULT_ANTIGEN_CHAIN),
        include_non_antigen_residues=False,
        manual_pocket_file=None if manual_file is None else str(manual_file),
        literature_file=None if literature_file is None else str(literature_file),
        catalytic_file=None if catalytic_file is None else str(catalytic_file),
        literature_source_table=None if literature_source_table is None else str(literature_source_table),
        catalytic_source_table=None if catalytic_source_table is None else str(catalytic_source_table),
        ai_pocket_file=None if ai_file is None else str(ai_file),
        ai_source_table=None if ai_source_table is None else str(ai_source_table),
        p2rank_file=args.p2rank_file,
        p2rank_top_n=int(args.p2rank_top_n),
        p2rank_rank=args.p2rank_rank,
        p2rank_name=args.p2rank_name,
        p2rank_min_probability=args.p2rank_min_probability,
        fpocket_file=args.fpocket_file,
        ligand_file=args.ligand_file,
        ligand_contact_threshold=float(args.ligand_contact_threshold),
        anchor_shell_radii=str(args.anchor_shell_radii),
        curated_min_support=float(args.curated_min_support),
        external_overwide_max_residue_count=int(args.external_overwide_max_residue_count),
        external_overwide_max_fraction=float(args.external_overwide_max_fraction),
        disable_external_precision_guard=bool(args.disable_external_precision_guard),
    )
    evidence_summary = build_pocket_evidence(evidence_args)

    candidate_pocket = out_dir / "candidate_curated_pocket.txt"
    candidate_count = _count_residue_file(candidate_pocket)
    generated_df = input_df.copy()
    input_base = input_csv_out.parent.resolve()
    generated_pocket_value = _format_generated_path(
        candidate_pocket if candidate_count > 0 else None,
        base_dir=input_base,
        path_mode=str(args.path_mode),
    )
    generated_catalytic_value = _format_generated_path(
        catalytic_file,
        base_dir=input_base,
        path_mode=str(args.path_mode),
    )
    generated_df["antigen_chain"] = str(args.antigen_chain or DEFAULT_ANTIGEN_CHAIN)
    generated_df["nanobody_chain"] = str(args.nanobody_chain or DEFAULT_NANOBODY_CHAIN)
    generated_df["pocket_file"] = generated_pocket_value
    if generated_catalytic_value:
        generated_df["catalytic_file"] = generated_catalytic_value

    input_csv_out.parent.mkdir(parents=True, exist_ok=True)
    generated_df.to_csv(input_csv_out, index=False)

    warnings: list[str] = []
    if candidate_count == 0:
        warnings.append("candidate_curated_pocket.txt is empty; generated input table pocket_file column was left blank.")
    warnings.extend(str(item) for item in evidence_summary.get("warnings", []) if item)

    output_files = {
        "input_csv_with_pocket_evidence": str(input_csv_out),
        "candidate_curated_pocket_txt": str(candidate_pocket),
        "pocket_evidence_csv": str(out_dir / "pocket_evidence.csv"),
        "pocket_residue_support_csv": str(out_dir / "pocket_residue_support.csv"),
        "evidence_source_audit_csv": str(out_dir / "evidence_source_audit.csv"),
        "evidence_source_template_csv": str(out_dir / "evidence_source_template.csv"),
        "ai_prior_audit_csv": str(out_dir / "ai_prior_audit.csv"),
        "ai_prior_template_csv": str(out_dir / "ai_prior_template.csv"),
        "project_summary_json": str(out_dir / "project_pocket_evidence_summary.json"),
        "project_report_md": str(out_dir / "project_pocket_evidence_report.md"),
    }
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(project_root),
        "result_root": str(result_root),
        "target_prefix": target_prefix,
        "antigen_chain": str(args.antigen_chain or DEFAULT_ANTIGEN_CHAIN),
        "nanobody_chain": str(args.nanobody_chain or DEFAULT_NANOBODY_CHAIN),
        "result_tree_summary": result_summary,
        "representative_selection": representative_summary,
        "source_files": {
            "manual": manual_info,
            "literature": literature_info,
            "catalytic": catalytic_info,
            "literature_source_table": None if literature_source_table is None else str(literature_source_table),
            "catalytic_source_table": None if catalytic_source_table is None else str(catalytic_source_table),
            "ai_prior": ai_info,
            "ai_source_table": None if ai_source_table is None else str(ai_source_table),
            "p2rank_file": args.p2rank_file,
            "fpocket_file": args.fpocket_file,
            "ligand_file": args.ligand_file,
        },
        "pocket_evidence_summary": evidence_summary,
        "candidate_pocket": {
            "path": str(candidate_pocket),
            "residue_count": int(candidate_count),
            "written_to_input_table": bool(candidate_count > 0),
        },
        "generated_input_table": {
            "path": str(input_csv_out),
            "row_count": int(len(generated_df)),
            "pocket_file_value": generated_pocket_value,
            "catalytic_file_value": generated_catalytic_value,
        },
        "warnings": warnings,
        "output_files": output_files,
    }

    summary_path = out_dir / "project_pocket_evidence_summary.json"
    report_path = out_dir / "project_pocket_evidence_report.md"
    summary_path.write_text(json.dumps(_json_sanitize(summary), ensure_ascii=True, indent=2), encoding="utf-8")
    _write_project_report(summary, report_path)
    return summary


def main() -> None:
    args = _build_parser().parse_args()
    summary = build_project_pocket_evidence(args)
    print(f"Saved input CSV: {summary['output_files']['input_csv_with_pocket_evidence']}")
    print(f"Saved project summary: {summary['output_files']['project_summary_json']}")
    print(f"Saved project report: {summary['output_files']['project_report_md']}")
    print(f"Representative PDB: {summary['representative_selection']['representative_pdb_path']}")
    print(f"Candidate pocket residues: {summary['candidate_pocket']['residue_count']}")
    for warning in summary.get("warnings", []):
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
