from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROXY_FEATURES = [
    "pocket_hit_fraction",
    "catalytic_hit_fraction",
    "mouth_occlusion_score",
    "mouth_axis_block_fraction",
    "mouth_aperture_block_fraction",
    "mouth_min_clearance",
    "delta_pocket_occupancy_proxy",
    "pocket_block_volume_proxy",
    "substrate_overlap_score",
    "ligand_path_block_score",
    "ligand_path_block_fraction",
    "ligand_path_bottleneck_score",
    "ligand_path_exit_block_fraction",
    "ligand_path_min_clearance",
    "pocket_shape_residue_count",
    "pocket_shape_overwide_proxy",
    "pocket_shape_tightness_proxy",
]

ROW_FIELDS = [
    "nanobody_id",
    "conformer_id",
    "pose_id",
    "flag_count",
    "flag_ids",
    "recommended_action",
    *PROXY_FEATURES,
]

CANDIDATE_FIELDS = [
    "nanobody_id",
    "pose_count",
    "flagged_pose_count",
    "flagged_pose_fraction",
    "dominant_flags",
    "mean_pocket_hit_fraction",
    "mean_catalytic_hit_fraction",
    "mean_mouth_occlusion_score",
    "mean_ligand_path_block_score",
    "mean_ligand_path_exit_block_fraction",
    "mean_pocket_shape_overwide_proxy",
    "recommended_action",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit geometry proxy consistency from an existing pose_features.csv without changing scores."
    )
    parser.add_argument("--feature_csv", required=True, help="Input pose_features.csv.")
    parser.add_argument("--out_dir", default="geometry_proxy_audit", help="Output directory.")
    parser.add_argument("--high_threshold", type=float, default=0.65, help="High proxy threshold.")
    parser.add_argument("--low_threshold", type=float, default=0.35, help="Low proxy threshold.")
    parser.add_argument(
        "--overwide_threshold",
        type=float,
        default=0.55,
        help="Threshold for pocket_shape_overwide_proxy review flags.",
    )
    parser.add_argument(
        "--open_clearance_threshold",
        type=float,
        default=3.5,
        help="Clearance above this value suggests the path/mouth may still be open.",
    )
    parser.add_argument(
        "--disagreement_threshold",
        type=float,
        default=0.45,
        help="Large gap threshold between core blocking proxies.",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=30,
        help="Reserved for report previews; the detailed flagged pose CSV includes all flagged rows.",
    )
    return parser


def _read_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"feature_csv not found: {csv_path}")
    try:
        return pd.read_csv(csv_path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)


def _num(row: pd.Series, col: str) -> float:
    try:
        value = float(row.get(col, np.nan))
    except (TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def _finite(value: float) -> bool:
    return bool(np.isfinite(value))


def _is_high(value: float, threshold: float) -> bool:
    return _finite(value) and value >= float(threshold)


def _is_low(value: float, threshold: float) -> bool:
    return _finite(value) and value <= float(threshold)


def _flag_row(row: pd.Series, args: argparse.Namespace) -> list[str]:
    high = float(args.high_threshold)
    low = float(args.low_threshold)
    open_clearance = float(args.open_clearance_threshold)
    disagreement = float(args.disagreement_threshold)

    pocket_hit = _num(row, "pocket_hit_fraction")
    catalytic_hit = _num(row, "catalytic_hit_fraction")
    mouth_occ = _num(row, "mouth_occlusion_score")
    mouth_axis = _num(row, "mouth_axis_block_fraction")
    mouth_aperture = _num(row, "mouth_aperture_block_fraction")
    mouth_clearance = _num(row, "mouth_min_clearance")
    path_block = _num(row, "ligand_path_block_score")
    path_fraction = _num(row, "ligand_path_block_fraction")
    path_bottleneck = _num(row, "ligand_path_bottleneck_score")
    path_exit = _num(row, "ligand_path_exit_block_fraction")
    path_clearance = _num(row, "ligand_path_min_clearance")
    substrate_overlap = _num(row, "substrate_overlap_score")
    overwide = _num(row, "pocket_shape_overwide_proxy")

    flags: list[str] = []
    mouth_high = any(_is_high(value, high) for value in [mouth_occ, mouth_axis, mouth_aperture])
    path_high = any(_is_high(value, high) for value in [path_block, path_fraction, path_bottleneck, path_exit])
    path_open = (
        _is_low(path_exit, low)
        or _is_low(path_block, low)
        or (_finite(path_clearance) and path_clearance >= open_clearance)
    )
    mouth_open = _finite(mouth_clearance) and mouth_clearance >= open_clearance

    if _is_high(overwide, float(args.overwide_threshold)):
        flags.append("pocket_shape_overwide")

    if mouth_high and path_open:
        flags.append("mouth_high_but_ligand_path_open")

    if path_high and _is_low(pocket_hit, low):
        flags.append("path_high_but_pocket_contact_low")

    if path_high and _is_low(catalytic_hit, low):
        flags.append("path_high_but_catalytic_contact_low")

    if _is_high(pocket_hit, high) and _is_high(catalytic_hit, high) and not mouth_high and not path_high:
        flags.append("contact_high_but_blocking_proxy_low")

    if mouth_high and mouth_open:
        flags.append("mouth_block_high_but_clearance_open")

    if path_high and _finite(path_clearance) and path_clearance >= open_clearance:
        flags.append("path_block_high_but_clearance_open")

    core_values = [
        value
        for value in [pocket_hit, catalytic_hit, mouth_occ, path_block, substrate_overlap]
        if _finite(value)
    ]
    if len(core_values) >= 3 and (max(core_values) - min(core_values)) >= disagreement:
        flags.append("core_proxy_disagreement")

    return flags


def _feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for col in PROXY_FEATURES:
        if col not in df.columns:
            rows.append(
                {
                    "feature": col,
                    "present": False,
                    "finite_count": 0,
                    "missing_fraction": 1.0,
                    "mean": "",
                    "p50": "",
                    "p95": "",
                    "min": "",
                    "max": "",
                }
            )
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        finite = values[np.isfinite(values.to_numpy(dtype=np.float64))]
        if finite.empty:
            rows.append(
                {
                    "feature": col,
                    "present": True,
                    "finite_count": 0,
                    "missing_fraction": float(values.isna().mean()),
                    "mean": "",
                    "p50": "",
                    "p95": "",
                    "min": "",
                    "max": "",
                }
            )
            continue
        arr = finite.to_numpy(dtype=np.float64)
        rows.append(
            {
                "feature": col,
                "present": True,
                "finite_count": int(finite.shape[0]),
                "missing_fraction": float(values.isna().mean()),
                "mean": float(np.nanmean(arr)),
                "p50": float(np.nanpercentile(arr, 50)),
                "p95": float(np.nanpercentile(arr, 95)),
                "min": float(np.nanmin(arr)),
                "max": float(np.nanmax(arr)),
            }
        )
    return pd.DataFrame(rows)


def _recommended_action(flags: list[str]) -> str:
    flag_set = set(flags)
    if not flags:
        return "No proxy consistency issue detected."
    if "pocket_shape_overwide" in flag_set:
        return "Review pocket boundary or method consensus before trusting blocking score."
    if any(flag.endswith("_clearance_open") or flag == "mouth_high_but_ligand_path_open" for flag in flag_set):
        return "Inspect mouth/path geometry; score may be driven by a partial blocker while an escape path remains open."
    if "path_high_but_pocket_contact_low" in flag_set or "path_high_but_catalytic_contact_low" in flag_set:
        return "Check chain assignment, ligand template alignment, and pocket/catalytic residue definitions."
    if "contact_high_but_blocking_proxy_low" in flag_set:
        return "Treat as binding/contact evidence rather than clear pocket blocking evidence."
    return "Review geometry inputs and feature distributions before interpreting this pose."


def _flagged_rows(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        flags = _flag_row(row, args)
        if not flags:
            continue
        record: dict[str, Any] = {}
        for col in ROW_FIELDS:
            if col == "flag_count":
                record[col] = len(flags)
            elif col == "flag_ids":
                record[col] = ";".join(flags)
            elif col == "recommended_action":
                record[col] = _recommended_action(flags)
            elif col in df.columns:
                value = row.get(col)
                if isinstance(value, (np.integer, np.floating)):
                    value = float(value)
                record[col] = value
            else:
                record[col] = ""
        records.append(record)
    if not records:
        return pd.DataFrame(columns=ROW_FIELDS)
    out = pd.DataFrame(records)
    sort_cols = [col for col in ["flag_count", "pocket_shape_overwide_proxy", "mouth_occlusion_score"] if col in out.columns]
    if sort_cols:
        out = out.sort_values(by=sort_cols, ascending=[False] * len(sort_cols), na_position="last")
    return out.reset_index(drop=True)


def _candidate_audit(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if "nanobody_id" not in df.columns:
        return pd.DataFrame(columns=CANDIDATE_FIELDS)
    flag_records: list[tuple[str, list[str]]] = []
    for _, row in df.iterrows():
        flag_records.append((str(row.get("nanobody_id") or ""), _flag_row(row, args)))

    flag_df = pd.DataFrame(
        {
            "nanobody_id": [item[0] for item in flag_records],
            "flag_ids": [";".join(item[1]) for item in flag_records],
            "flag_count": [len(item[1]) for item in flag_records],
        }
    )
    merged = pd.concat([df.reset_index(drop=True), flag_df[["flag_ids", "flag_count"]]], axis=1)
    rows: list[dict[str, Any]] = []
    for nanobody_id, group in merged.groupby("nanobody_id", dropna=False):
        pose_count = int(len(group))
        flagged = group.loc[pd.to_numeric(group["flag_count"], errors="coerce").fillna(0) > 0]
        counter: Counter[str] = Counter()
        for text in flagged.get("flag_ids", pd.Series([], dtype=object)).fillna("").astype(str):
            for flag in [part.strip() for part in text.split(";") if part.strip()]:
                counter[flag] += 1
        dominant = ";".join([flag for flag, _ in counter.most_common(4)])
        record: dict[str, Any] = {
            "nanobody_id": nanobody_id,
            "pose_count": pose_count,
            "flagged_pose_count": int(len(flagged)),
            "flagged_pose_fraction": float(len(flagged) / max(pose_count, 1)),
            "dominant_flags": dominant,
            "recommended_action": _recommended_action(list(counter.keys())),
        }
        for col in [
            "pocket_hit_fraction",
            "catalytic_hit_fraction",
            "mouth_occlusion_score",
            "ligand_path_block_score",
            "ligand_path_exit_block_fraction",
            "pocket_shape_overwide_proxy",
        ]:
            out_col = f"mean_{col}"
            if col in group.columns:
                values = pd.to_numeric(group[col], errors="coerce")
                record[out_col] = float(values.mean()) if values.notna().any() else ""
            else:
                record[out_col] = ""
        rows.append(record)
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=CANDIDATE_FIELDS)
    return out.sort_values(
        by=["flagged_pose_fraction", "flagged_pose_count"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)


def _summary(
    df: pd.DataFrame,
    feature_summary: pd.DataFrame,
    flagged_rows: pd.DataFrame,
    candidate_audit: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, Any]:
    missing_features = [
        str(row["feature"])
        for _, row in feature_summary.iterrows()
        if not bool(row.get("present"))
    ]
    flag_counts: Counter[str] = Counter()
    if not flagged_rows.empty and "flag_ids" in flagged_rows.columns:
        for text in flagged_rows["flag_ids"].fillna("").astype(str):
            for flag in [part.strip() for part in text.split(";") if part.strip()]:
                flag_counts[flag] += 1

    row_count = int(len(df))
    flagged_pose_count = int(
        min(int(flagged_rows.shape[0]), row_count)
    )
    if not candidate_audit.empty and "flagged_pose_count" in candidate_audit.columns:
        flagged_pose_count = int(pd.to_numeric(candidate_audit["flagged_pose_count"], errors="coerce").fillna(0).sum())
    flagged_fraction = float(flagged_pose_count / max(row_count, 1))
    if row_count <= 0:
        audit_status = "FAIL"
    elif flagged_fraction >= 0.30:
        audit_status = "WARN"
    elif missing_features and len(missing_features) >= 8:
        audit_status = "WARN"
    else:
        audit_status = "PASS"

    next_actions: list[str] = []
    if "pocket_shape_overwide" in flag_counts:
        next_actions.append("Review high pocket_shape_overwide_proxy rows; consider method consensus or tighter pocket definition.")
    if any(flag in flag_counts for flag in ["mouth_high_but_ligand_path_open", "path_block_high_but_clearance_open"]):
        next_actions.append("Inspect mouth/path rows where blocking is high but a clearance/open-path proxy remains permissive.")
    if any(flag in flag_counts for flag in ["path_high_but_pocket_contact_low", "path_high_but_catalytic_contact_low"]):
        next_actions.append("Check chain assignment, ligand template alignment, and pocket/catalytic residue files.")
    if not next_actions:
        next_actions.append("No immediate proxy-consistency blocker detected; keep using these rows as QC context, not as proof of physical blocking.")

    return {
        "feature_csv": str(Path(args.feature_csv).expanduser().resolve()),
        "row_count": row_count,
        "candidate_count": int(df["nanobody_id"].nunique()) if "nanobody_id" in df.columns else 0,
        "audit_status": audit_status,
        "flagged_pose_count": flagged_pose_count,
        "flagged_pose_fraction": flagged_fraction,
        "flag_counts": dict(flag_counts),
        "missing_proxy_features": missing_features,
        "thresholds": {
            "high_threshold": float(args.high_threshold),
            "low_threshold": float(args.low_threshold),
            "overwide_threshold": float(args.overwide_threshold),
            "open_clearance_threshold": float(args.open_clearance_threshold),
            "disagreement_threshold": float(args.disagreement_threshold),
        },
        "next_actions": next_actions,
    }


def _md(value: Any) -> str:
    return str(value).replace("|", "\\|")


def _build_report(summary: dict[str, Any], feature_summary: pd.DataFrame, flagged_rows: pd.DataFrame, candidate_audit: pd.DataFrame) -> str:
    lines = [
        "# Geometry Proxy Audit",
        "",
        "This report checks whether static geometry proxies are internally consistent. It does not change Rule, ML, or consensus scores.",
        "",
        "## Summary",
        "",
        f"- Feature CSV: `{summary['feature_csv']}`",
        f"- Audit status: `{summary['audit_status']}`",
        f"- Pose rows: `{summary['row_count']}`",
        f"- Candidate count: `{summary['candidate_count']}`",
        f"- Flagged pose rows: `{summary['flagged_pose_count']}`",
        f"- Flagged pose fraction: `{summary['flagged_pose_fraction']:.2%}`",
        "",
        "## Main Flags",
        "",
    ]
    flag_counts = summary.get("flag_counts") or {}
    if flag_counts:
        lines.extend(["| Flag | Count |", "| --- | ---: |"])
        for flag, count in sorted(flag_counts.items(), key=lambda item: (-int(item[1]), str(item[0]))):
            lines.append(f"| {_md(flag)} | {int(count)} |")
    else:
        lines.append("- No proxy-consistency flags were detected.")

    lines.extend(["", "## Next Actions", ""])
    for action in summary.get("next_actions", []):
        lines.append(f"- {action}")

    lines.extend(
        [
            "",
            "## Feature Coverage",
            "",
            "| Feature | Present | Finite Rows | Mean | P95 | Max |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in feature_summary.iterrows():
        lines.append(
            f"| {_md(row.get('feature', ''))} | {_md(row.get('present', ''))} | {_md(row.get('finite_count', ''))} | {_md(_fmt(row.get('mean', '')))} | {_md(_fmt(row.get('p95', '')))} | {_md(_fmt(row.get('max', '')))} |"
        )

    lines.extend(["", "## Top Flagged Poses", ""])
    if flagged_rows.empty:
        lines.append("- No flagged pose rows.")
    else:
        cols = [col for col in ["nanobody_id", "conformer_id", "pose_id", "flag_count", "flag_ids", "recommended_action"] if col in flagged_rows.columns]
        lines.extend(["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"])
        for _, row in flagged_rows.loc[:, cols].head(12).iterrows():
            lines.append("| " + " | ".join(_md(row.get(col, "")) for col in cols) + " |")

    lines.extend(["", "## Top Candidate-Level Audit", ""])
    if candidate_audit.empty:
        lines.append("- Candidate-level audit is unavailable because `nanobody_id` is missing.")
    else:
        cols = [col for col in ["nanobody_id", "pose_count", "flagged_pose_count", "flagged_pose_fraction", "dominant_flags"] if col in candidate_audit.columns]
        lines.extend(["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"])
        for _, row in candidate_audit.loc[:, cols].head(12).iterrows():
            values = []
            for col in cols:
                value = row.get(col, "")
                if col == "flagged_pose_fraction" and isinstance(value, (float, np.floating)):
                    value = f"{float(value):.2%}"
                values.append(_md(value))
            lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(number):
        return ""
    return f"{number:.4f}"


def main() -> None:
    args = _build_parser().parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _read_csv(args.feature_csv)
    feature_summary = _feature_summary(df)
    flagged = _flagged_rows(df, args)
    candidate_audit = _candidate_audit(df, args)
    summary = _summary(df, feature_summary, flagged, candidate_audit, args)

    feature_summary_csv = out_dir / "geometry_proxy_feature_summary.csv"
    flagged_rows_csv = out_dir / "geometry_proxy_flagged_poses.csv"
    candidate_audit_csv = out_dir / "geometry_proxy_candidate_audit.csv"
    summary_json = out_dir / "geometry_proxy_audit_summary.json"
    report_md = out_dir / "geometry_proxy_audit_report.md"

    feature_summary.to_csv(feature_summary_csv, index=False)
    flagged.to_csv(flagged_rows_csv, index=False)
    candidate_audit.to_csv(candidate_audit_csv, index=False)
    summary.update(
        {
            "outputs": {
                "feature_summary_csv": str(feature_summary_csv),
                "flagged_rows_csv": str(flagged_rows_csv),
                "candidate_audit_csv": str(candidate_audit_csv),
                "summary_json": str(summary_json),
                "report_md": str(report_md),
            }
        }
    )
    summary_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    report_md.write_text(_build_report(summary, feature_summary, flagged, candidate_audit), encoding="utf-8")

    print(f"Saved: {feature_summary_csv}")
    print(f"Saved: {flagged_rows_csv}")
    print(f"Saved: {candidate_audit_csv}")
    print(f"Saved: {summary_json}")
    print(f"Saved: {report_md}")
    print(f"Audit status: {summary['audit_status']}")
    print(f"Flagged poses: {summary['flagged_pose_count']}/{summary['row_count']}")


if __name__ == "__main__":
    main()
