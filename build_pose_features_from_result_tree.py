"""One-command feature extraction from a canonical result/ tree.

Run this from the parent directory that contains result/ or rsite/result/:

    python build_pose_features_from_result_tree.py

The default output is ./pose_features.csv, matching the normal feature-table
format while adding optional sidecar-derived numeric columns when available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build_feature_table import build_feature_table, read_input_table, save_feature_table, summarize_processing_results
from pocket_io import load_residue_set
from result_tree_io import (
    DEFAULT_ANTIGEN_CHAIN,
    DEFAULT_NANOBODY_CHAIN,
    build_input_table_from_result_tree,
    write_result_tree_report,
)


def _sort_residue_key(key: str) -> tuple[str, int, str]:
    parts = [part.strip() for part in str(key).split(":")]
    chain = parts[0] if parts else ""
    try:
        resseq = int(parts[1]) if len(parts) > 1 else 0
    except ValueError:
        resseq = 0
    icode = parts[2] if len(parts) > 2 else ""
    return chain, resseq, icode


def _find_auto_rsite_file(search_roots: list[Path]) -> Path | None:
    candidates: list[Path] = []
    for root in search_roots:
        if not root:
            continue
        resolved = root.expanduser().resolve()
        candidates.extend(
            [
                resolved / "rsite" / "rsite.txt",
                resolved / "rsite.txt",
                resolved / "cd38_pocket_from_rsite_chain_B.txt",
                resolved / "pocket.txt",
                resolved / "pocket_residues.txt",
            ]
        )
        if resolved.name.lower() == "result":
            candidates.extend(
                [
                    resolved.parent / "rsite" / "rsite.txt",
                    resolved.parent / "rsite.txt",
                ]
            )
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            path = candidate.resolve()
        except OSError:
            continue
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            return path
    return None


def _prepare_auto_pocket_file(
    *,
    out_dir: Path,
    result_root: Path,
    antigen_chain: str,
    explicit_pocket_file: str | None,
    disable_auto_rsite_pocket: bool,
) -> tuple[str | None, dict[str, Any]]:
    if explicit_pocket_file:
        return explicit_pocket_file, {
            "mode": "explicit",
            "source_file": str(Path(explicit_pocket_file).expanduser().resolve()),
            "filtered_file": None,
            "residue_count": None,
        }
    if disable_auto_rsite_pocket:
        return None, {
            "mode": "disabled",
            "source_file": None,
            "filtered_file": None,
            "residue_count": 0,
        }

    source = _find_auto_rsite_file([out_dir, result_root, Path.cwd()])
    if source is None:
        return None, {
            "mode": "not_found",
            "source_file": None,
            "filtered_file": None,
            "residue_count": 0,
        }

    keys = load_residue_set(source)
    chain = str(antigen_chain or "").strip()
    filtered = sorted(
        [key for key in keys if not chain or str(key).split(":", 1)[0] == chain],
        key=_sort_residue_key,
    )
    if not filtered:
        return str(source.resolve()), {
            "mode": "source_unfiltered_empty_after_chain_filter",
            "source_file": str(source.resolve()),
            "filtered_file": None,
            "residue_count": 0,
        }

    auto_dir = out_dir / ".ml_auto"
    auto_dir.mkdir(parents=True, exist_ok=True)
    filtered_path = auto_dir / f"auto_pocket_antigen_{chain or 'all'}.txt"
    filtered_path.write_text("\n".join(filtered) + "\n", encoding="utf-8")
    return str(filtered_path.resolve()), {
        "mode": "auto_rsite_filtered_to_antigen_chain",
        "source_file": str(source.resolve()),
        "filtered_file": str(filtered_path.resolve()),
        "residue_count": int(len(filtered)),
        "antigen_chain": chain,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build pose_features.csv directly from result/ tree")
    parser.add_argument(
        "--result_root",
        default=".",
        help="Project/result root. Default current directory; supports ./result and ./rsite/result.",
    )
    parser.add_argument(
        "--out_csv",
        default="pose_features.csv",
        help="Output feature CSV. Default ./pose_features.csv.",
    )
    parser.add_argument(
        "--input_csv",
        default=None,
        help="Intermediate input pose table. Default is input_pose_table.csv next to out_csv.",
    )
    parser.add_argument("--qc_json", default=None, help="Feature QC JSON. Default feature_qc.json next to out_csv.")
    parser.add_argument("--summary_json", default=None, help="Result-tree summary JSON.")
    parser.add_argument("--report_md", default=None, help="Result-tree input report Markdown.")
    parser.add_argument("--default_pocket_file", default=None)
    parser.add_argument("--default_catalytic_file", default=None)
    parser.add_argument("--default_ligand_file", default=None)
    parser.add_argument("--default_antigen_chain", default=DEFAULT_ANTIGEN_CHAIN)
    parser.add_argument("--default_nanobody_chain", default=DEFAULT_NANOBODY_CHAIN)
    parser.add_argument("--disable_auto_rsite_pocket", action="store_true")
    parser.add_argument(
        "--target_prefix",
        default="",
        help="Target folder prefix filter. Empty default includes any target such as CD38_1 or other proteins.",
    )
    parser.add_argument("--path_mode", choices=["absolute", "relative"], default="absolute")
    parser.add_argument("--allow_single_pdb_fallback", action="store_true")
    parser.add_argument("--skip_failed_rows", action="store_true")
    parser.add_argument("--atom_contact_threshold", type=float, default=4.5)
    parser.add_argument("--catalytic_contact_threshold", type=float, default=4.5)
    parser.add_argument("--substrate_clash_threshold", type=float, default=2.8)
    parser.add_argument("--mouth_residue_fraction", type=float, default=0.30)
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    result_root = Path(args.result_root).expanduser().resolve()
    out_csv = Path(args.out_csv).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    input_csv = Path(args.input_csv).expanduser().resolve() if args.input_csv else out_csv.with_name("input_pose_table.csv")
    qc_json = Path(args.qc_json).expanduser().resolve() if args.qc_json else out_csv.with_name("feature_qc.json")
    summary_json = Path(args.summary_json).expanduser().resolve() if args.summary_json else input_csv.with_suffix(".summary.json")
    report_md = Path(args.report_md).expanduser().resolve() if args.report_md else input_csv.with_suffix(".report.md")

    default_pocket, pocket_info = _prepare_auto_pocket_file(
        out_dir=out_csv.parent,
        result_root=result_root,
        antigen_chain=str(args.default_antigen_chain or DEFAULT_ANTIGEN_CHAIN),
        explicit_pocket_file=args.default_pocket_file,
        disable_auto_rsite_pocket=bool(args.disable_auto_rsite_pocket),
    )
    default_catalytic = args.default_catalytic_file or default_pocket

    input_df, input_summary = build_input_table_from_result_tree(
        result_root,
        default_pocket_file=default_pocket,
        default_catalytic_file=default_catalytic,
        default_ligand_file=args.default_ligand_file,
        default_antigen_chain=args.default_antigen_chain,
        default_nanobody_chain=args.default_nanobody_chain,
        path_mode=args.path_mode,
        out_csv_path=input_csv,
        allow_single_pdb_fallback=bool(args.allow_single_pdb_fallback),
        target_prefix=str(args.target_prefix or ""),
    )
    input_csv.parent.mkdir(parents=True, exist_ok=True)
    input_df.to_csv(input_csv, index=False)
    input_summary["auto_pocket"] = pocket_info
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(input_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_result_tree_report(summary=input_summary, out_report=report_md, out_csv=input_csv)

    read_df = read_input_table(str(input_csv))
    feature_df = build_feature_table(
        df=read_df,
        base_dir=input_csv.parent.resolve(),
        atom_contact_threshold=float(args.atom_contact_threshold),
        catalytic_contact_threshold=float(args.catalytic_contact_threshold),
        substrate_clash_threshold=float(args.substrate_clash_threshold),
        mouth_residue_fraction=float(args.mouth_residue_fraction),
        default_pocket_file=default_pocket,
        default_catalytic_file=default_catalytic,
        default_ligand_file=args.default_ligand_file,
        default_antigen_chain=args.default_antigen_chain,
        default_nanobody_chain=args.default_nanobody_chain,
        skip_failed_rows=bool(args.skip_failed_rows),
    )
    save_feature_table(feature_df, str(out_csv), qc_json_path=str(qc_json))

    processing = summarize_processing_results(feature_df)
    print(f"Saved feature CSV: {out_csv}")
    print(f"Saved input CSV: {input_csv}")
    print(f"Rows: {len(feature_df)}; ok={processing.get('ok_rows', 0)}; failed={processing.get('failed_rows', 0)}")
    print(f"Auto pocket: {pocket_info.get('mode')} ({pocket_info.get('residue_count')} residues)")
    print(f"Saved QC JSON: {qc_json}")
    print(f"Saved input report: {report_md}")


if __name__ == "__main__":
    main()
