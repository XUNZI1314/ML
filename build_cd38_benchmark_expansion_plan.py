from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


PLAN_COLUMNS = [
    "pdb_id",
    "method",
    "status",
    "priority",
    "protein_chain",
    "ligand_chain",
    "ligand_resnames",
    "ligand_resseqs",
    "manifest_row_count",
    "result_count",
    "manifest_result_names",
    "result_names",
    "p2rank_predictions_csv",
    "fpocket_root",
    "recommended_action",
    "recommended_command",
    "notes",
]


TARGET_COLUMNS = [
    "enabled",
    "priority",
    "pdb_id",
    "protein_chain",
    "ligand_chain",
    "ligand_resnames",
    "ligand_resseqs",
    "desired_methods",
    "p2rank_predictions_csv",
    "fpocket_root",
    "notes",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a CD38 benchmark expansion plan from target structures, manifest rows, and existing results."
    )
    parser.add_argument(
        "--structure_targets_csv",
        default="benchmarks/cd38/cd38_structure_targets.csv",
        help="Target structures and desired methods to audit.",
    )
    parser.add_argument(
        "--manifest_csv",
        default="benchmarks/cd38/cd38_benchmark_manifest.csv",
        help="Current CD38 benchmark manifest.",
    )
    parser.add_argument("--results_root", default="benchmarks/cd38/results", help="CD38 benchmark results root.")
    parser.add_argument(
        "--panel_csv",
        default=None,
        help="Optional summarized benchmark panel CSV. Defaults to <results_root>/cd38_benchmark_panel.csv.",
    )
    parser.add_argument(
        "--out_dir",
        default="benchmarks/cd38/expansion_plan",
        help="Output directory for expansion plan tables and report.",
    )
    parser.add_argument(
        "--default_methods",
        default="ligand_contact,p2rank,fpocket",
        help="Methods to use when a target row leaves desired_methods blank.",
    )
    parser.add_argument("--near_threshold", type=float, default=4.5, help="Recommended near-hit threshold.")
    return parser


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _truthy(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    text = str(value).strip().lower()
    if not text or text == "default":
        return default
    return text in {"1", "true", "yes", "y", "on"}


def _cell(row: dict[str, Any], key: str) -> str:
    return str(row.get(key) or "").strip()


def _split_csv(value: Any) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def _normalize_method(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "ligand": "ligand_contact",
        "contact": "ligand_contact",
        "ligandcontact": "ligand_contact",
        "p2_rank": "p2rank",
        "p2rank_predictions": "p2rank",
    }
    return aliases.get(text, text)


def _normalize_pdb_id(value: Any) -> str:
    return str(value or "").strip().upper()


def _resolve_optional_path(value: Any, repo_root: Path) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def _path_for_command(value: Any) -> str:
    return str(value).replace("/", "\\")


def _safe_name(value: Any) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value or "").strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "unknown"


def _format_command(parts: list[str]) -> str:
    formatted: list[str] = []
    for item in parts:
        text = str(item)
        formatted.append(f'"{text}"' if re.search(r"\s", text) else text)
    return " ".join(formatted)


def _desired_methods(row: dict[str, str], default_methods: list[str]) -> list[str]:
    raw = _split_csv(row.get("desired_methods"))
    methods = raw or default_methods
    normalized = []
    for method in methods:
        method_name = _normalize_method(method)
        if method_name and method_name not in normalized:
            normalized.append(method_name)
    return normalized


def _match_manifest_rows(manifest_rows: list[dict[str, str]], pdb_id: str, method: str) -> list[dict[str, str]]:
    matches = []
    for row in manifest_rows:
        if not _truthy(row.get("enabled"), default=True):
            continue
        if _normalize_pdb_id(row.get("rcsb_pdb_id")) != pdb_id:
            continue
        if _normalize_method(row.get("method")) != method:
            continue
        matches.append(row)
    return matches


def _match_result_rows(panel_rows: list[dict[str, str]], pdb_id: str, method: str) -> list[dict[str, str]]:
    matches = []
    for row in panel_rows:
        if _normalize_pdb_id(row.get("pdb_id")) != pdb_id:
            continue
        if _normalize_method(row.get("method")) != method:
            continue
        matches.append(row)
    return matches


def _result_names(rows: list[dict[str, str]]) -> str:
    names = [_cell(row, "result_name") for row in rows]
    return ",".join(name for name in names if name)


def _ligand_contact_command(row: dict[str, str], pdb_id: str, results_root: Path, near_threshold: float) -> str:
    protein_chain = _cell(row, "protein_chain") or "A"
    ligand_chain = _cell(row, "ligand_chain")
    ligand_resnames = _cell(row, "ligand_resnames")
    ligand_resseqs = _cell(row, "ligand_resseqs")
    ligand_label = ligand_resnames or ligand_resseqs or "ligand"
    result_name = f"{pdb_id}_ligand_contact_chain{_safe_name(protein_chain)}_{_safe_name(ligand_label)}"
    command = [
        "python",
        "run_cd38_ligand_contact_benchmark.py",
        "--rcsb_pdb_id",
        pdb_id,
        "--protein_chain",
        protein_chain,
        "--out_dir",
        _path_for_command(results_root / result_name),
        "--near_threshold",
        str(float(near_threshold)),
    ]
    if ligand_chain:
        command.extend(["--ligand_chain", ligand_chain])
    if ligand_resnames:
        command.extend(["--ligand_resnames", ligand_resnames])
    if ligand_resseqs:
        command.extend(["--ligand_resseqs", ligand_resseqs])
    return _format_command(command)


def _p2rank_command(row: dict[str, str], pdb_id: str, repo_root: Path, results_root: Path, near_threshold: float) -> str:
    predictions_path = _resolve_optional_path(row.get("p2rank_predictions_csv"), repo_root)
    protein_chain = _cell(row, "protein_chain") or "A"
    result_name = f"{pdb_id}_p2rank_top1_chain{_safe_name(protein_chain)}"
    command = [
        "python",
        "run_cd38_p2rank_benchmark.py",
        "--predictions_csv",
        _path_for_command(predictions_path or _cell(row, "p2rank_predictions_csv")),
        "--rcsb_pdb_id",
        pdb_id,
        "--chain_filter",
        protein_chain,
        "--top_n",
        "1",
        "--out_dir",
        _path_for_command(results_root / result_name),
        "--near_threshold",
        str(float(near_threshold)),
    ]
    return _format_command(command)


def _p2rank_prepare_command(row: dict[str, str], pdb_id: str, results_root: Path, near_threshold: float) -> str:
    protein_chain = _cell(row, "protein_chain") or "A"
    command = [
        "python",
        "prepare_cd38_p2rank_panel.py",
        "--p2rank_root",
        f"your_p2rank_outputs\\{pdb_id}",
        "--rcsb_pdb_id",
        pdb_id,
        "--chain_filter",
        protein_chain,
        "--results_root",
        _path_for_command(results_root),
        "--near_threshold",
        str(float(near_threshold)),
        "--manifest_out",
        f"benchmarks\\cd38\\p2rank_{pdb_id}_manifest.csv",
    ]
    return _format_command(command)


def _fpocket_command(row: dict[str, str], pdb_id: str, results_root: Path, near_threshold: float) -> str:
    fpocket_root = _cell(row, "fpocket_root")
    protein_chain = _cell(row, "protein_chain") or "A"
    command = [
        "python",
        "prepare_cd38_fpocket_panel.py",
        "--fpocket_root",
        fpocket_root or f"your_fpocket_outputs\\{pdb_id}_out",
        "--rcsb_pdb_id",
        pdb_id,
        "--chain_filter",
        protein_chain,
        "--results_root",
        _path_for_command(results_root),
        "--near_threshold",
        str(float(near_threshold)),
        "--manifest_out",
        f"benchmarks\\cd38\\fpocket_{pdb_id}_manifest.csv",
    ]
    return _format_command(command)


def _status_for_method(
    *,
    target: dict[str, str],
    pdb_id: str,
    method: str,
    manifest_matches: list[dict[str, str]],
    result_matches: list[dict[str, str]],
    repo_root: Path,
    results_root: Path,
    near_threshold: float,
) -> tuple[str, str, str]:
    if result_matches:
        return "complete", "Result already exists; keep it in the summary panel.", ""

    if manifest_matches:
        names = _result_names(manifest_matches)
        command = _format_command(
            [
                "python",
                "run_cd38_benchmark_manifest.py",
                "--only",
                names,
                "--results_root",
                _path_for_command(results_root),
            ]
        )
        return "manifest_ready", "Manifest row exists but result is missing; run this manifest row.", command

    if method == "ligand_contact":
        ligand_chain = _cell(target, "ligand_chain")
        ligand_resnames = _cell(target, "ligand_resnames")
        ligand_resseqs = _cell(target, "ligand_resseqs")
        if not ligand_chain or (not ligand_resnames and not ligand_resseqs):
            return (
                "needs_ligand_metadata",
                "Fill ligand_chain plus ligand_resnames or ligand_resseqs before ligand-contact benchmark.",
                "",
            )
        return (
            "needs_manifest_row",
            "Ligand metadata is present; add a manifest row or run this direct ligand-contact command.",
            _ligand_contact_command(target, pdb_id, results_root, near_threshold),
        )

    if method == "p2rank":
        predictions_path = _resolve_optional_path(target.get("p2rank_predictions_csv"), repo_root)
        if predictions_path is None or not predictions_path.exists():
            return (
                "needs_p2rank_output",
                "Run P2Rank externally, then generate a discovered manifest from its predictions.csv output.",
                _p2rank_prepare_command(target, pdb_id, results_root, near_threshold),
            )
        return (
            "needs_manifest_row",
            "P2Rank predictions exist; add a manifest row or run this direct P2Rank command.",
            _p2rank_command(target, pdb_id, repo_root, results_root, near_threshold),
        )

    if method == "fpocket":
        fpocket_root = _resolve_optional_path(target.get("fpocket_root"), repo_root)
        if fpocket_root is None or not fpocket_root.exists():
            return (
                "needs_fpocket_output",
                "Run fpocket externally and fill fpocket_root, then generate a discovered manifest.",
                _fpocket_command(target, pdb_id, results_root, near_threshold),
            )
        return (
            "needs_fpocket_discovery",
            "fpocket_root exists; generate a discovered manifest and review the readiness report.",
            _fpocket_command(target, pdb_id, results_root, near_threshold),
        )

    return "unsupported_method", f"Unsupported method: {method}", ""


def _build_plan_rows(
    *,
    targets: list[dict[str, str]],
    manifest_rows: list[dict[str, str]],
    panel_rows: list[dict[str, str]],
    default_methods: list[str],
    repo_root: Path,
    results_root: Path,
    near_threshold: float,
) -> list[dict[str, Any]]:
    plan_rows: list[dict[str, Any]] = []
    for target in targets:
        enabled = _truthy(target.get("enabled"), default=True)
        pdb_id = _normalize_pdb_id(target.get("pdb_id"))
        if not pdb_id:
            continue
        methods = _desired_methods(target, default_methods)
        for method in methods:
            manifest_matches = _match_manifest_rows(manifest_rows, pdb_id, method)
            result_matches = _match_result_rows(panel_rows, pdb_id, method)
            if not enabled:
                status, action, command = "target_disabled", "Target is disabled in structure target CSV.", ""
            else:
                status, action, command = _status_for_method(
                    target=target,
                    pdb_id=pdb_id,
                    method=method,
                    manifest_matches=manifest_matches,
                    result_matches=result_matches,
                    repo_root=repo_root,
                    results_root=results_root,
                    near_threshold=near_threshold,
                )
            plan_rows.append(
                {
                    "pdb_id": pdb_id,
                    "method": method,
                    "status": status,
                    "priority": _cell(target, "priority"),
                    "protein_chain": _cell(target, "protein_chain") or "A",
                    "ligand_chain": _cell(target, "ligand_chain"),
                    "ligand_resnames": _cell(target, "ligand_resnames"),
                    "ligand_resseqs": _cell(target, "ligand_resseqs"),
                    "manifest_row_count": len(manifest_matches),
                    "result_count": len(result_matches),
                    "manifest_result_names": _result_names(manifest_matches),
                    "result_names": _result_names(result_matches),
                    "p2rank_predictions_csv": _cell(target, "p2rank_predictions_csv"),
                    "fpocket_root": _cell(target, "fpocket_root"),
                    "recommended_action": action,
                    "recommended_command": command,
                    "notes": _cell(target, "notes"),
                }
            )
    return plan_rows


def _build_summary(plan_rows: list[dict[str, Any]], args: argparse.Namespace, panel_csv: Path) -> dict[str, Any]:
    status_counts = Counter(str(row.get("status") or "") for row in plan_rows)
    method_counts = Counter(str(row.get("method") or "") for row in plan_rows)
    complete_structures = sorted(
        {
            str(row.get("pdb_id"))
            for row in plan_rows
            if row.get("status") == "complete"
        }
    )
    missing_rows = [row for row in plan_rows if row.get("status") not in {"complete", "target_disabled"}]
    return {
        "structure_targets_csv": str(Path(args.structure_targets_csv).expanduser().resolve()),
        "manifest_csv": str(Path(args.manifest_csv).expanduser().resolve()),
        "results_root": str(Path(args.results_root).expanduser().resolve()),
        "panel_csv": str(panel_csv),
        "out_dir": str(Path(args.out_dir).expanduser().resolve()),
        "target_structure_count": len({str(row.get("pdb_id")) for row in plan_rows}),
        "target_method_count": len(plan_rows),
        "missing_or_pending_count": len(missing_rows),
        "complete_structures": complete_structures,
        "status_counts": dict(sorted(status_counts.items())),
        "method_counts": dict(sorted(method_counts.items())),
        "next_missing_actions": missing_rows[:20],
    }


def _md_escape(value: Any) -> str:
    return str(value).replace("|", "\\|")


def _build_markdown(plan_rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# CD38 Benchmark Expansion Plan",
        "",
        "This report audits which CD38 structures and pocket methods are already covered, and what is still needed before expanding the benchmark panel.",
        "",
        "## Summary",
        "",
        f"- Target structures: `{summary['target_structure_count']}`",
        f"- Target structure-method pairs: `{summary['target_method_count']}`",
        f"- Missing or pending pairs: `{summary['missing_or_pending_count']}`",
        f"- Structure targets CSV: `{summary['structure_targets_csv']}`",
        f"- Manifest CSV: `{summary['manifest_csv']}`",
        f"- Panel CSV: `{summary['panel_csv']}`",
        "",
        "Status counts:",
        "",
    ]
    for status, count in summary["status_counts"].items():
        lines.append(f"- `{status}`: `{count}`")

    display_cols = [
        "pdb_id",
        "method",
        "status",
        "manifest_row_count",
        "result_count",
        "recommended_action",
    ]
    lines.extend(["", "## Plan Table", ""])
    lines.append("| " + " | ".join(display_cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(display_cols)) + " |")
    for row in plan_rows:
        lines.append("| " + " | ".join(_md_escape(row.get(col, "")) for col in display_cols) + " |")

    command_rows = [row for row in plan_rows if str(row.get("recommended_command") or "").strip()]
    lines.extend(["", "## Recommended Commands", ""])
    if not command_rows:
        lines.append("No direct commands are currently available; fill missing external output paths or ligand metadata first.")
    else:
        for row in command_rows:
            lines.extend(
                [
                    f"### {row['pdb_id']} {row['method']} ({row['status']})",
                    "",
                    "```powershell",
                    str(row["recommended_command"]),
                    "```",
                    "",
                ]
            )

    lines.extend(
        [
            "## Interpretation",
            "",
            "- `complete` means a result already exists in the summarized benchmark panel.",
            "- `manifest_ready` means the manifest has a row but the result directory is missing.",
            "- `needs_ligand_metadata` means the structure may be usable, but ligand chain/name or residue number still needs curation.",
            "- `needs_p2rank_output` and `needs_fpocket_output` mean the external tool output must be generated before this repository can benchmark it.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = _build_parser().parse_args()
    repo_root = _repo_root()
    targets_csv = Path(args.structure_targets_csv).expanduser().resolve()
    manifest_csv = Path(args.manifest_csv).expanduser().resolve()
    results_root = Path(args.results_root).expanduser().resolve()
    command_results_root = Path(args.results_root).expanduser()
    panel_csv = Path(args.panel_csv).expanduser().resolve() if args.panel_csv else results_root / "cd38_benchmark_panel.csv"
    out_dir = Path(args.out_dir).expanduser().resolve()

    targets = _read_csv(targets_csv)
    if not targets:
        raise FileNotFoundError(f"No target rows found: {targets_csv}")
    manifest_rows = _read_csv(manifest_csv)
    panel_rows = _read_csv(panel_csv)
    default_methods = [_normalize_method(method) for method in _split_csv(args.default_methods)]

    plan_rows = _build_plan_rows(
        targets=targets,
        manifest_rows=manifest_rows,
        panel_rows=panel_rows,
        default_methods=default_methods,
        repo_root=repo_root,
        results_root=command_results_root,
        near_threshold=float(args.near_threshold),
    )
    summary = _build_summary(plan_rows, args, panel_csv)

    out_dir.mkdir(parents=True, exist_ok=True)
    plan_csv = out_dir / "cd38_benchmark_expansion_plan.csv"
    missing_csv = out_dir / "cd38_benchmark_missing_actions.csv"
    summary_json = out_dir / "cd38_benchmark_expansion_summary.json"
    report_md = out_dir / "cd38_benchmark_expansion_plan.md"

    missing_rows = [row for row in plan_rows if row.get("status") not in {"complete", "target_disabled"}]
    _write_csv(plan_csv, plan_rows, PLAN_COLUMNS)
    _write_csv(missing_csv, missing_rows, PLAN_COLUMNS)
    summary_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    report_md.write_text(_build_markdown(plan_rows, summary), encoding="utf-8")

    print(f"Saved: {plan_csv}")
    print(f"Saved: {missing_csv}")
    print(f"Saved: {summary_json}")
    print(f"Saved: {report_md}")
    print(f"Target method pairs: {len(plan_rows)}")
    print(f"Missing or pending pairs: {len(missing_rows)}")


if __name__ == "__main__":
    main()
