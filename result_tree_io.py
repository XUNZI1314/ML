"""Utilities for the canonical result/ VHH-CD38 pose folder layout.

Canonical layout:

result/
  vhh1/
    CD38_1/
      1/
        1.pdb
      2/
        2.pdb

The scanner converts this tree into the input_pose_table.csv format consumed by
build_feature_table.py and run_recommended_pipeline.py.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


RESULT_TREE_COLUMNS = [
    "nanobody_id",
    "conformer_id",
    "pose_id",
    "pdb_path",
    "antigen_chain",
    "nanobody_chain",
    "pocket_file",
    "catalytic_file",
    "ligand_file",
    "MMPBSA_energy",
    "mmpbsa_result_file",
    "mmpbsa_parse_status",
    "mmpbsa_parse_warning",
    "target_id",
    "target_variant_id",
    "target_variant_index",
    "pose_index",
    "complex_id",
    "pose_dir",
    "source_relative_dir",
    "sidecar_file_count",
    "sidecar_files",
    "layout_status",
    "layout_warning",
]

PDB_SUFFIXES = {".pdb"}
TARGET_VARIANT_RE = re.compile(r"^(?P<target>[A-Za-z0-9]+)_(?P<index>\d+)$")
FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][-+]?\d+)?")
MMPBSA_RESULT_NAME = "FINAL_RESULTS_MMPBSA.dat"
DEFAULT_ANTIGEN_CHAIN = "B"
DEFAULT_NANOBODY_CHAIN = "A"


def _is_probable_pose_pdb(path: Path) -> bool:
    if path.suffix.lower() not in PDB_SUFFIXES:
        return False
    name = path.name.lower()
    return not (name.endswith("_atm.pdb") or "pocket" in name or "ligand" in name)


def _safe_int(text: str) -> int | None:
    try:
        return int(str(text).strip())
    except (TypeError, ValueError):
        return None


def parse_target_variant(name: str) -> tuple[str, str, int | None]:
    """Parse target variant folder names such as CD38_1."""
    text = str(name).strip()
    match = TARGET_VARIANT_RE.match(text)
    if match:
        return match.group("target"), text, int(match.group("index"))
    if "_" in text:
        return text.split("_", 1)[0], text, None
    return text, text, None


def _format_path(path_text: str | Path | None, *, base_dir: Path, path_mode: str) -> str:
    if path_text is None:
        return ""
    text = str(path_text).strip()
    if not text:
        return ""
    path = Path(text).expanduser()
    if path_mode == "relative":
        resolved = path.resolve()
        try:
            return str(resolved.relative_to(base_dir.resolve()))
        except ValueError:
            return str(resolved)
    return str(path.resolve())


def _select_pose_pdb(
    pose_dir: Path,
    *,
    allow_single_pdb_fallback: bool = False,
) -> tuple[Path | None, str, str]:
    """Select the canonical pose PDB from one pose directory."""
    expected = pose_dir / f"{pose_dir.name}.pdb"
    if expected.is_file():
        return expected, "ok", ""

    candidates = sorted(
        [path for path in pose_dir.iterdir() if path.is_file() and _is_probable_pose_pdb(path)],
        key=lambda p: p.name.lower(),
    )
    if allow_single_pdb_fallback and len(candidates) == 1:
        return candidates[0], "warning", f"used single PDB fallback instead of expected {expected.name}"
    if not candidates:
        return None, "missing_pdb", f"expected {expected.name}"
    return None, "ambiguous_pdb", f"expected {expected.name}; found {len(candidates)} pdb files"


def parse_mmpbsa_energy(dat_path: str | Path) -> tuple[float | None, str, str]:
    """Extract the final DELTA TOTAL energy from FINAL_RESULTS_MMPBSA.dat.

    The parser prefers the last DELTA TOTAL row because Amber-style reports can
    contain both GB and PB sections. If no DELTA TOTAL row exists, it falls back
    to the last TOTAL-like numeric row.
    """
    path = Path(dat_path).expanduser()
    if not path.exists():
        return None, "missing", "FINAL_RESULTS_MMPBSA.dat not found"
    if not path.is_file():
        return None, "not_file", "MMPBSA path is not a file"

    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError as exc:
        return None, "read_error", str(exc)

    delta_values: list[float] = []
    fallback_values: list[float] = []
    for line in lines:
        upper = line.upper()
        numbers = [float(item) for item in FLOAT_RE.findall(line)]
        if not numbers:
            continue
        if "DELTA" in upper and "TOTAL" in upper:
            delta_values.append(numbers[0])
        elif "TOTAL" in upper:
            fallback_values.append(numbers[0])

    if delta_values:
        return float(delta_values[-1]), "ok", "parsed last DELTA TOTAL value"
    if fallback_values:
        return float(fallback_values[-1]), "warning", "parsed fallback TOTAL value"
    return None, "no_energy", "no DELTA TOTAL or TOTAL numeric row found"


def is_probable_result_tree(result_root: str | Path, *, target_prefix: str = "CD38") -> bool:
    """Return True when a directory looks like result/vhh/CD38_x/pose/pose.pdb."""
    root = Path(result_root).expanduser()
    if not root.exists() or not root.is_dir():
        return False

    target_filter = str(target_prefix or "").strip().upper()
    checked_pose_dirs = 0
    for vhh_dir in root.iterdir():
        if not vhh_dir.is_dir():
            continue
        for variant_dir in vhh_dir.iterdir():
            if not variant_dir.is_dir():
                continue
            if target_filter and parse_target_variant(variant_dir.name)[0].upper() != target_filter:
                continue
            for pose_dir in variant_dir.iterdir():
                if not pose_dir.is_dir():
                    continue
                checked_pose_dirs += 1
                if (pose_dir / f"{pose_dir.name}.pdb").is_file():
                    return True
                if checked_pose_dirs >= 50:
                    return False
    return False


def find_result_tree_root(root: str | Path, *, target_prefix: str = "CD38") -> tuple[Path | None, str]:
    """Find the canonical result tree root from a selected project or result directory.

    The search is intentionally shallow so selecting a large project folder does
    not trigger an expensive recursive walk. Supported cases:
    - root is already result/
    - root/result is the canonical tree
    - root has one wrapper child that is result/ or contains result/
    """
    base = Path(root).expanduser()
    if not base.exists() or not base.is_dir():
        return None, "not_found"

    base = base.resolve()
    direct_result = base / "result"
    if is_probable_result_tree(direct_result, target_prefix=target_prefix):
        return direct_result.resolve(), "child_result"

    if is_probable_result_tree(base, target_prefix=target_prefix):
        return base, "direct"

    try:
        child_dirs = sorted([path for path in base.iterdir() if path.is_dir()], key=lambda p: p.name.lower())
    except OSError:
        return None, "not_found"

    for child in child_dirs:
        if child.name.lower() == "result" and is_probable_result_tree(child, target_prefix=target_prefix):
            return child.resolve(), "child_result"

    for child in child_dirs:
        nested_result = child / "result"
        if is_probable_result_tree(nested_result, target_prefix=target_prefix):
            return nested_result.resolve(), "wrapper_child_result"
        if is_probable_result_tree(child, target_prefix=target_prefix):
            return child.resolve(), "wrapper_direct"

    return None, "not_found"


def build_input_table_from_result_tree(
    result_root: str | Path,
    *,
    default_pocket_file: str | Path | None = None,
    default_catalytic_file: str | Path | None = None,
    default_ligand_file: str | Path | None = None,
    default_antigen_chain: str | None = DEFAULT_ANTIGEN_CHAIN,
    default_nanobody_chain: str | None = DEFAULT_NANOBODY_CHAIN,
    path_mode: str = "absolute",
    out_csv_path: str | Path | None = None,
    allow_single_pdb_fallback: bool = False,
    target_prefix: str = "CD38",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Scan the canonical result tree and build an input pose table."""
    requested_root = Path(result_root).expanduser().resolve()
    if not requested_root.exists():
        raise FileNotFoundError(f"result_root not found: {requested_root}")
    if not requested_root.is_dir():
        raise NotADirectoryError(f"result_root is not a directory: {requested_root}")

    mode = str(path_mode).strip().lower()
    if mode not in {"absolute", "relative"}:
        raise ValueError("path_mode must be 'absolute' or 'relative'")

    target_filter = str(target_prefix or "").strip().upper()
    discovered_root, discovery_mode = find_result_tree_root(requested_root, target_prefix=target_filter)
    root = discovered_root if discovered_root is not None else requested_root

    output_base = Path(out_csv_path).expanduser().resolve().parent if out_csv_path else Path.cwd().resolve()
    default_pocket = _format_path(default_pocket_file, base_dir=output_base, path_mode=mode)
    default_catalytic = _format_path(default_catalytic_file, base_dir=output_base, path_mode=mode)
    default_ligand = _format_path(default_ligand_file, base_dir=output_base, path_mode=mode)

    rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    vhh_dirs = sorted([path for path in root.iterdir() if path.is_dir()], key=lambda p: p.name.lower())

    for vhh_dir in vhh_dirs:
        nanobody_id = vhh_dir.name
        variant_dirs = sorted([path for path in vhh_dir.iterdir() if path.is_dir()], key=lambda p: p.name.lower())
        for variant_dir in variant_dirs:
            target_id, target_variant_id, target_variant_index = parse_target_variant(variant_dir.name)
            if target_filter and target_id.upper() != target_filter:
                continue

            pose_dirs = sorted(
                [path for path in variant_dir.iterdir() if path.is_dir()],
                key=lambda p: (_safe_int(p.name) is None, _safe_int(p.name) or 0, p.name.lower()),
            )
            for pose_dir in pose_dirs:
                pose_index = _safe_int(pose_dir.name)
                pdb_path, status, warning = _select_pose_pdb(
                    pose_dir,
                    allow_single_pdb_fallback=allow_single_pdb_fallback,
                )
                rel_pose_dir = pose_dir.relative_to(root)
                if pdb_path is None:
                    skipped_rows.append(
                        {
                            "nanobody_id": nanobody_id,
                            "target_variant_id": target_variant_id,
                            "pose_id": pose_dir.name,
                            "source_relative_dir": str(rel_pose_dir),
                            "layout_status": status,
                            "layout_warning": warning,
                        }
                    )
                    continue

                sidecar_files = sorted(
                    [
                        str(path.relative_to(pose_dir))
                        for path in pose_dir.iterdir()
                        if path.is_file() and path.resolve() != pdb_path.resolve()
                    ]
                )
                mmpbsa_file = pose_dir / MMPBSA_RESULT_NAME
                mmpbsa_energy, mmpbsa_status, mmpbsa_warning = parse_mmpbsa_energy(mmpbsa_file)
                complex_id = f"{nanobody_id}__{target_variant_id}__pose_{pose_dir.name}"
                rows.append(
                    {
                        "nanobody_id": nanobody_id,
                        "conformer_id": target_variant_id,
                        "pose_id": pose_dir.name,
                        "pdb_path": _format_path(pdb_path, base_dir=output_base, path_mode=mode),
                        "antigen_chain": str(default_antigen_chain or "").strip(),
                        "nanobody_chain": str(default_nanobody_chain or "").strip(),
                        "pocket_file": default_pocket,
                        "catalytic_file": default_catalytic,
                        "ligand_file": default_ligand,
                        "MMPBSA_energy": mmpbsa_energy if mmpbsa_energy is not None else "",
                        "mmpbsa_result_file": _format_path(mmpbsa_file, base_dir=output_base, path_mode=mode) if mmpbsa_file.exists() else "",
                        "mmpbsa_parse_status": mmpbsa_status,
                        "mmpbsa_parse_warning": "" if mmpbsa_status == "ok" else mmpbsa_warning,
                        "target_id": target_id,
                        "target_variant_id": target_variant_id,
                        "target_variant_index": target_variant_index if target_variant_index is not None else "",
                        "pose_index": pose_index if pose_index is not None else "",
                        "complex_id": complex_id,
                        "pose_dir": _format_path(pose_dir, base_dir=output_base, path_mode=mode),
                        "source_relative_dir": str(rel_pose_dir),
                        "sidecar_file_count": len(sidecar_files),
                        "sidecar_files": ";".join(sidecar_files),
                        "layout_status": status,
                        "layout_warning": warning,
                    }
                )

    df = pd.DataFrame(rows, columns=RESULT_TREE_COLUMNS)
    variant_counts = df.groupby("nanobody_id")["target_variant_id"].nunique().to_dict() if not df.empty else {}
    pose_counts = df.groupby(["nanobody_id", "target_variant_id"]).size().reset_index(name="pose_count") if not df.empty else pd.DataFrame()
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "requested_root": str(requested_root),
        "result_root": str(root),
        "result_root_discovery_mode": discovery_mode,
        "row_count": int(len(df)),
        "nanobody_count": int(df["nanobody_id"].nunique()) if not df.empty else 0,
        "target_variant_count": int(df["target_variant_id"].nunique()) if not df.empty else 0,
        "pose_dir_skipped_count": int(len(skipped_rows)),
        "warning_row_count": int((df["layout_status"].astype(str) == "warning").sum()) if not df.empty else 0,
        "mmpbsa_energy_count": int(pd.to_numeric(df["MMPBSA_energy"], errors="coerce").notna().sum()) if not df.empty else 0,
        "path_mode": mode,
        "target_prefix": target_filter,
        "allow_single_pdb_fallback": bool(allow_single_pdb_fallback),
        "variant_count_by_nanobody": {str(k): int(v) for k, v in variant_counts.items()},
        "pose_count_by_variant_preview": pose_counts.head(20).to_dict(orient="records") if not pose_counts.empty else [],
        "skipped_preview": skipped_rows[:20],
    }
    return df, summary


def write_result_tree_report(
    *,
    summary: dict[str, Any],
    out_report: str | Path,
    out_csv: str | Path,
) -> None:
    """Write a small Markdown report for the generated input table."""
    report_path = Path(out_report).expanduser()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Result Tree Input Report",
        "",
        "Canonical layout:",
        "",
        "```text",
        "result/<nanobody_id>/<target_variant_id>/<pose_id>/<pose_id>.pdb",
        "result/vhh1/CD38_1/1/1.pdb",
        "```",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| rows | {summary.get('row_count', 0)} |",
        f"| nanobodies | {summary.get('nanobody_count', 0)} |",
        f"| target variants | {summary.get('target_variant_count', 0)} |",
        f"| skipped pose dirs | {summary.get('pose_dir_skipped_count', 0)} |",
        f"| warning rows | {summary.get('warning_row_count', 0)} |",
        f"| parsed MMPBSA energies | {summary.get('mmpbsa_energy_count', 0)} |",
        "",
        f"Requested root: `{summary.get('requested_root', '')}`",
        "",
        f"Resolved result root: `{summary.get('result_root', '')}`",
        "",
        f"Discovery mode: `{summary.get('result_root_discovery_mode', '')}`",
        "",
        f"Output CSV: `{Path(out_csv).expanduser().resolve()}`",
        "",
    ]
    skipped = summary.get("skipped_preview")
    if isinstance(skipped, list) and skipped:
        lines.extend(["## Skipped Preview", "", "| nanobody_id | target_variant_id | pose_id | status | warning |", "|---|---|---|---|---|"])
        for row in skipped[:20]:
            lines.append(
                "| {nanobody_id} | {target_variant_id} | {pose_id} | {layout_status} | {layout_warning} |".format(
                    nanobody_id=row.get("nanobody_id", ""),
                    target_variant_id=row.get("target_variant_id", ""),
                    pose_id=row.get("pose_id", ""),
                    layout_status=row.get("layout_status", ""),
                    layout_warning=row.get("layout_warning", ""),
                )
            )
        lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build input_pose_table.csv from canonical result/ tree")
    parser.add_argument("--result_root", required=True, help="Path to result/ root")
    parser.add_argument("--out_csv", default="input_pose_table.csv", help="Output input pose table CSV")
    parser.add_argument("--default_pocket_file", default=None)
    parser.add_argument("--default_catalytic_file", default=None)
    parser.add_argument("--default_ligand_file", default=None)
    parser.add_argument("--default_antigen_chain", default=DEFAULT_ANTIGEN_CHAIN)
    parser.add_argument("--default_nanobody_chain", default=DEFAULT_NANOBODY_CHAIN)
    parser.add_argument("--target_prefix", default="CD38", help="Target folder prefix to include, default CD38")
    parser.add_argument("--path_mode", choices=["absolute", "relative"], default="absolute")
    parser.add_argument(
        "--allow_single_pdb_fallback",
        action="store_true",
        help="Use the only PDB in a pose folder when <pose_id>.pdb is missing",
    )
    parser.add_argument("--summary_json", default=None, help="Optional summary JSON path")
    parser.add_argument("--report_md", default=None, help="Optional Markdown report path")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    out_csv = Path(args.out_csv).expanduser().resolve()
    df, summary = build_input_table_from_result_tree(
        args.result_root,
        default_pocket_file=args.default_pocket_file,
        default_catalytic_file=args.default_catalytic_file,
        default_ligand_file=args.default_ligand_file,
        default_antigen_chain=args.default_antigen_chain,
        default_nanobody_chain=args.default_nanobody_chain,
        path_mode=args.path_mode,
        out_csv_path=out_csv,
        allow_single_pdb_fallback=bool(args.allow_single_pdb_fallback),
        target_prefix=args.target_prefix,
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    summary_json = Path(args.summary_json).expanduser().resolve() if args.summary_json else out_csv.with_suffix(".summary.json")
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")

    report_md = Path(args.report_md).expanduser().resolve() if args.report_md else out_csv.with_suffix(".report.md")
    write_result_tree_report(summary=summary, out_report=report_md, out_csv=out_csv)

    print(f"Saved input CSV: {out_csv}")
    print(f"Rows: {summary['row_count']}; nanobodies: {summary['nanobody_count']}; skipped pose dirs: {summary['pose_dir_skipped_count']}")
    print(f"Saved summary: {summary_json}")
    print(f"Saved report: {report_md}")


if __name__ == "__main__":
    main()
