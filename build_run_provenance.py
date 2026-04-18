"""Build a reproducibility card for one completed recommended pipeline run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any


DEFAULT_HASH_MAX_MB = 100.0
DEFAULT_PACKAGES = [
    "numpy",
    "pandas",
    "biopython",
    "torch",
    "streamlit",
]

PATH_REFERENCE_COLUMNS = ["pdb_path", "pocket_file", "catalytic_file", "ligand_file"]

INPUT_FILE_MANIFEST_COLUMNS = [
    "source_table",
    "source_table_path",
    "data_row_number",
    "csv_line_number",
    "nanobody_id",
    "conformer_id",
    "pose_id",
    "column",
    "value_source",
    "original_value",
    "resolved_path",
    "relative_to_table_dir",
    "relative_to_repo_root",
    "exists",
    "type",
    "size_bytes",
    "sha256",
    "hash_status",
    "status",
]


def _now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _safe_relpath(path: Path, base: Path) -> str | None:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except Exception:
        return None


def _sha256_file(path: Path, max_bytes: int) -> tuple[str | None, str]:
    if not path.exists():
        return None, "missing"
    if not path.is_file():
        return None, "not_file"
    size = int(path.stat().st_size)
    if size > max_bytes:
        return None, f"skipped_size_gt_{max_bytes}"

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest(), "ok"


def _sha256_required(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Cannot hash missing file: {path}")
    if not path.is_file():
        raise ValueError(f"Cannot hash non-file path: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(
    *,
    group: str,
    label: str,
    path_value: Any,
    out_dir: Path,
    repo_root: Path,
    max_hash_bytes: int,
) -> dict[str, Any]:
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = (out_dir / path).resolve()
    exists = path.exists()
    kind = "dir" if exists and path.is_dir() else "file" if exists else "missing"
    sha256, hash_status = _sha256_file(path, max_hash_bytes)
    rel_out = _safe_relpath(path, out_dir)
    rel_repo = _safe_relpath(path, repo_root)
    size_bytes = int(path.stat().st_size) if exists and path.is_file() else None
    return {
        "group": str(group),
        "label": str(label),
        "path": str(path),
        "relative_to_out_dir": rel_out,
        "relative_to_repo_root": rel_repo,
        "exists": bool(exists),
        "type": kind,
        "size_bytes": size_bytes,
        "sha256": sha256,
        "hash_status": hash_status,
    }


def _dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for record in records:
        key = (str(record.get("group")), str(record.get("path")))
        if key in seen:
            continue
        seen.add(key)
        out.append(record)
    return out


def _collect_input_records(
    *,
    summary: dict[str, Any],
    summary_path: Path,
    out_dir: Path,
    repo_root: Path,
    max_hash_bytes: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for key in ["input_csv", "feature_csv"]:
        value = summary.get(key)
        if value:
            records.append(
                _file_record(
                    group="input",
                    label=key,
                    path_value=value,
                    out_dir=out_dir,
                    repo_root=repo_root,
                    max_hash_bytes=max_hash_bytes,
                )
            )
    records.append(
        _file_record(
            group="input",
            label="recommended_pipeline_summary_json",
            path_value=summary_path,
            out_dir=out_dir,
            repo_root=repo_root,
            max_hash_bytes=max_hash_bytes,
        )
    )
    return _dedupe_records(records)


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    if not path.exists() or not path.is_file():
        return [], []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            return list(reader.fieldnames or []), [dict(row) for row in reader]
    except UnicodeDecodeError:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return list(reader.fieldnames or []), [dict(row) for row in reader]


def _clean_reference_value(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def _resolve_input_reference_path(value: str, table_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (table_path.parent / path).resolve()


def _input_reference_status(*, column: str, value: str, exists: bool, kind: str, hash_status: str) -> str:
    if not value:
        return "empty_required" if column == "pdb_path" else "empty_optional"
    if not exists:
        return "missing"
    if kind != "file":
        return "not_file"
    if hash_status != "ok":
        return hash_status
    return "ok"


def _input_file_reference_record(
    *,
    source_table: str,
    table_path: Path,
    data_row_number: int,
    row: dict[str, Any],
    column: str,
    value_source: str,
    original_value: str,
    repo_root: Path,
    max_hash_bytes: int,
) -> dict[str, Any]:
    resolved_path: Path | None = None
    if original_value:
        resolved_path = _resolve_input_reference_path(original_value, table_path)
    exists = bool(resolved_path and resolved_path.exists())
    kind = "dir" if exists and resolved_path and resolved_path.is_dir() else "file" if exists else "missing"
    sha256, hash_status = _sha256_file(resolved_path, max_hash_bytes) if resolved_path else (None, "empty")
    size_bytes = int(resolved_path.stat().st_size) if exists and resolved_path and resolved_path.is_file() else None
    return {
        "source_table": source_table,
        "source_table_path": str(table_path),
        "data_row_number": int(data_row_number),
        "csv_line_number": int(data_row_number) + 1,
        "nanobody_id": _clean_reference_value(row.get("nanobody_id")),
        "conformer_id": _clean_reference_value(row.get("conformer_id")),
        "pose_id": _clean_reference_value(row.get("pose_id")),
        "column": column,
        "value_source": value_source,
        "original_value": original_value,
        "resolved_path": "" if resolved_path is None else str(resolved_path),
        "relative_to_table_dir": "" if resolved_path is None else (_safe_relpath(resolved_path, table_path.parent) or ""),
        "relative_to_repo_root": "" if resolved_path is None else (_safe_relpath(resolved_path, repo_root) or ""),
        "exists": bool(exists),
        "type": kind,
        "size_bytes": size_bytes,
        "sha256": sha256,
        "hash_status": hash_status,
        "status": _input_reference_status(column=column, value=original_value, exists=exists, kind=kind, hash_status=hash_status),
    }


def _collect_input_file_references(
    *,
    summary: dict[str, Any],
    repo_root: Path,
    max_hash_bytes: int,
) -> list[dict[str, Any]]:
    input_csv = _clean_reference_value(summary.get("input_csv"))
    feature_csv = _clean_reference_value(summary.get("feature_csv"))
    source_table = "input_csv" if input_csv else "feature_csv" if feature_csv else ""
    table_value = input_csv or feature_csv
    if not table_value:
        return []
    table_path = Path(table_value).expanduser().resolve()
    fieldnames, rows = _read_csv_rows(table_path)
    if not rows:
        return []

    defaults = summary.get("input_file_defaults") if isinstance(summary.get("input_file_defaults"), dict) else {}
    default_by_column = {
        "pocket_file": _clean_reference_value(defaults.get("default_pocket_file")),
        "catalytic_file": _clean_reference_value(defaults.get("default_catalytic_file")),
        "ligand_file": _clean_reference_value(defaults.get("default_ligand_file")),
    }

    records: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows, start=1):
        for column in PATH_REFERENCE_COLUMNS:
            row_has_column = column in fieldnames
            row_value = _clean_reference_value(row.get(column)) if row_has_column else ""
            default_value = default_by_column.get(column, "")
            value_source = "row_value"
            effective_value = row_value
            if column != "pdb_path" and not effective_value and default_value:
                value_source = f"default_{column}"
                effective_value = default_value
            if not effective_value and not (column == "pdb_path" and source_table == "input_csv"):
                continue
            records.append(
                _input_file_reference_record(
                    source_table=source_table,
                    table_path=table_path,
                    data_row_number=row_number,
                    row=row,
                    column=column,
                    value_source=value_source,
                    original_value=effective_value,
                    repo_root=repo_root,
                    max_hash_bytes=max_hash_bytes,
                )
            )
    return records


def _write_input_file_manifest_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INPUT_FILE_MANIFEST_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in INPUT_FILE_MANIFEST_COLUMNS})


def _input_file_manifest_summary(records: list[dict[str, Any]], manifest_csv: Path, manifest_hash: str) -> dict[str, Any]:
    unique_paths = {str(record.get("resolved_path") or "") for record in records if record.get("resolved_path")}
    status_counts: dict[str, int] = {}
    column_counts: dict[str, int] = {}
    for record in records:
        status = str(record.get("status") or "")
        column = str(record.get("column") or "")
        status_counts[status] = status_counts.get(status, 0) + 1
        column_counts[column] = column_counts.get(column, 0) + 1
    return {
        "manifest_csv": str(manifest_csv),
        "manifest_hash": manifest_hash,
        "reference_count": int(len(records)),
        "unique_path_count": int(len(unique_paths)),
        "missing_count": int(status_counts.get("missing", 0)),
        "empty_required_count": int(status_counts.get("empty_required", 0)),
        "hash_skipped_count": int(
            sum(count for status, count in status_counts.items() if status.startswith("skipped_"))
        ),
        "status_counts": status_counts,
        "column_counts": column_counts,
    }


def _collect_artifact_records(
    *,
    summary: dict[str, Any],
    out_dir: Path,
    repo_root: Path,
    max_hash_bytes: int,
) -> list[dict[str, Any]]:
    artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
    records: list[dict[str, Any]] = []
    for key, value in artifacts.items():
        if not value:
            continue
        records.append(
            _file_record(
                group="artifact",
                label=str(key),
                path_value=value,
                out_dir=out_dir,
                repo_root=repo_root,
                max_hash_bytes=max_hash_bytes,
            )
        )
    return _dedupe_records(records)


def _collect_source_records(
    *,
    summary: dict[str, Any],
    out_dir: Path,
    repo_root: Path,
    max_hash_bytes: int,
) -> list[dict[str, Any]]:
    paths: list[Path] = [
        repo_root / "run_recommended_pipeline.py",
        repo_root / "pipeline_runner_common.py",
        repo_root / "runtime_dependency_utils.py",
        repo_root / "app_metadata.py",
        repo_root / "requirements.txt",
    ]
    commands = summary.get("commands") if isinstance(summary.get("commands"), list) else []
    for item in commands:
        if not isinstance(item, dict):
            continue
        command = item.get("command") if isinstance(item.get("command"), list) else []
        for part in command:
            text = str(part)
            if not text.lower().endswith(".py"):
                continue
            path = Path(text).expanduser()
            if not path.is_absolute():
                path = (repo_root / path).resolve()
            paths.append(path)

    records = [
        _file_record(
            group="source",
            label=path.name,
            path_value=path,
            out_dir=out_dir,
            repo_root=repo_root,
            max_hash_bytes=max_hash_bytes,
        )
        for path in paths
        if path.exists()
    ]
    return _dedupe_records(records)


def _stable_hash(payload: Any) -> str:
    data = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _build_integrity_signature(
    *,
    created_at: str,
    json_path: Path,
    md_path: Path,
    manifest_path: Path,
    input_file_manifest_path: Path,
    parameter_hash: str,
    file_manifest_hash: str,
    input_file_manifest_hash: str,
) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for key, path in {
        "run_provenance_card_json": json_path,
        "run_provenance_card_md": md_path,
        "run_artifact_manifest_csv": manifest_path,
        "run_input_file_manifest_csv": input_file_manifest_path,
    }.items():
        files[key] = {
            "path": path.name,
            "sha256": _sha256_required(path),
            "size_bytes": int(path.stat().st_size),
        }
    payload = {
        "files": files,
        "provenance_hashes": {
            "parameter_hash": parameter_hash,
            "file_manifest_hash": file_manifest_hash,
            "input_file_manifest_hash": input_file_manifest_hash,
        },
    }
    return {
        "schema_version": 1,
        "created_at": created_at,
        "algorithm": "sha256",
        "signature_type": "integrity_seal_without_private_key",
        "files": files,
        "provenance_hashes": payload["provenance_hashes"],
        "signature_payload_hash": _stable_hash(payload),
        "notes": (
            "This is a reproducibility integrity seal for accidental change detection. "
            "It is not a private-key cryptographic signature."
        ),
    }


def _package_versions(packages: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for package in packages:
        try:
            version = metadata.version(package)
            status = "ok"
        except metadata.PackageNotFoundError:
            version = None
            status = "missing"
        out.append({"package": package, "version": version, "status": status})
    return out


def _run_git(repo_root: Path, args: list[str]) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        return int(proc.returncode), (proc.stdout or "").strip(), (proc.stderr or "").strip()
    except Exception as exc:
        return 1, "", str(exc)


def _git_info(repo_root: Path) -> dict[str, Any]:
    rc_commit, commit, err_commit = _run_git(repo_root, ["rev-parse", "HEAD"])
    rc_branch, branch, err_branch = _run_git(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"])
    rc_status, status_text, err_status = _run_git(repo_root, ["status", "--short"])
    status_lines = [line for line in status_text.splitlines() if line.strip()]
    return {
        "available": rc_commit == 0,
        "commit": commit if rc_commit == 0 else None,
        "branch": branch if rc_branch == 0 else None,
        "dirty": bool(status_lines),
        "status_count": len(status_lines),
        "untracked_count": sum(1 for line in status_lines if line.startswith("??")),
        "tracked_dirty_count": sum(1 for line in status_lines if not line.startswith("??")),
        "status_sample": status_lines[:25],
        "errors": {
            "commit": err_commit if rc_commit != 0 else None,
            "branch": err_branch if rc_branch != 0 else None,
            "status": err_status if rc_status != 0 else None,
        },
    }


def _parameter_payload(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "start_mode": summary.get("start_mode"),
        "label_col": summary.get("label_col"),
        "label_valid_count": summary.get("label_valid_count"),
        "label_class_count": summary.get("label_class_count"),
        "label_compare_possible": summary.get("label_compare_possible"),
        "calibration_possible": summary.get("calibration_possible"),
        "ranking_config": summary.get("ranking_config"),
        "suggestion_config": summary.get("suggestion_config"),
        "experiment_plan_config": summary.get("experiment_plan_config"),
        "strategy_seed": summary.get("strategy_seed"),
        "runtime_dependencies": summary.get("runtime_dependencies"),
        "commands": summary.get("commands"),
    }


def _write_manifest_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group",
        "label",
        "path",
        "relative_to_out_dir",
        "relative_to_repo_root",
        "exists",
        "type",
        "size_bytes",
        "sha256",
        "hash_status",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in fieldnames})


def _format_size(value: Any) -> str:
    try:
        size = int(value)
    except (TypeError, ValueError):
        return "N/A"
    units = ["B", "KB", "MB", "GB"]
    out = float(size)
    unit = units[0]
    for unit in units:
        if out < 1024.0 or unit == units[-1]:
            break
        out /= 1024.0
    return f"{out:.1f} {unit}" if unit != "B" else f"{int(out)} B"


def _record_table(records: list[dict[str, Any]], limit: int = 40) -> list[str]:
    if not records:
        return ["暂无记录。"]
    lines = ["| Group | Label | Exists | Size | SHA256 |", "|---|---|---:|---:|---|"]
    for record in records[:limit]:
        digest = str(record.get("sha256") or record.get("hash_status") or "")
        if len(digest) > 16:
            digest = digest[:16] + "..."
        lines.append(
            "| "
            + " | ".join(
                [
                    str(record.get("group") or ""),
                    str(record.get("label") or ""),
                    str(bool(record.get("exists"))),
                    _format_size(record.get("size_bytes")),
                    digest,
                ]
            )
            + " |"
        )
    if len(records) > limit:
        lines.append(f"\n已省略 {len(records) - limit} 条，完整清单见 `run_artifact_manifest.csv`。")
    return lines


def _build_markdown(card: dict[str, Any]) -> str:
    run = card.get("run") if isinstance(card.get("run"), dict) else {}
    git = card.get("git") if isinstance(card.get("git"), dict) else {}
    environment = card.get("environment") if isinstance(card.get("environment"), dict) else {}
    records = card.get("file_manifest") if isinstance(card.get("file_manifest"), list) else []
    input_records = [record for record in records if isinstance(record, dict) and record.get("group") == "input"]
    artifact_records = [record for record in records if isinstance(record, dict) and record.get("group") == "artifact"]
    source_records = [record for record in records if isinstance(record, dict) and record.get("group") == "source"]
    commands = card.get("commands") if isinstance(card.get("commands"), list) else []
    input_file_manifest = (
        card.get("input_file_manifest") if isinstance(card.get("input_file_manifest"), dict) else {}
    )

    lines = [
        "# Run Provenance Card",
        "",
        "该文件用于复现实验和审计结果来源；它不改变任何模型训练、排序或评分。",
        "",
        "## Run",
        "",
        f"- Created at: `{card.get('created_at', 'N/A')}`",
        f"- Start mode: `{run.get('start_mode', 'N/A')}`",
        f"- Output dir: `{run.get('out_dir', 'N/A')}`",
        f"- Summary JSON: `{run.get('summary_json', 'N/A')}`",
        f"- Parameter hash: `{card.get('parameter_hash', 'N/A')}`",
        f"- File manifest hash: `{card.get('file_manifest_hash', 'N/A')}`",
        "",
        "## Environment",
        "",
        f"- Python: `{environment.get('python_version', 'N/A')}`",
        f"- Executable: `{environment.get('python_executable', 'N/A')}`",
        f"- Platform: `{environment.get('platform', 'N/A')}`",
        f"- Repo branch: `{git.get('branch', 'N/A')}`",
        f"- Repo commit: `{git.get('commit', 'N/A')}`",
        f"- Git dirty: `{git.get('dirty', 'N/A')}` ({git.get('status_count', 'N/A')} changed entries)",
        "",
        "## Inputs",
        "",
        *_record_table(input_records),
        "",
        "## Input File References",
        "",
        f"- Manifest CSV: `{input_file_manifest.get('manifest_csv', 'N/A')}`",
        f"- Manifest hash: `{input_file_manifest.get('manifest_hash', 'N/A')}`",
        f"- References: `{input_file_manifest.get('reference_count', 0)}`",
        f"- Unique paths: `{input_file_manifest.get('unique_path_count', 0)}`",
        f"- Missing paths: `{input_file_manifest.get('missing_count', 0)}`",
        f"- Empty required pdb_path rows: `{input_file_manifest.get('empty_required_count', 0)}`",
        "",
        "## Output Artifacts",
        "",
        *_record_table(artifact_records),
        "",
        "## Source Files",
        "",
        *_record_table(source_records),
        "",
        "## Commands",
        "",
        "| Step | RC | Sec |",
        "|---|---:|---:|",
    ]
    if commands:
        for item in commands:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"| {item.get('name', 'unknown')} | {int(item.get('returncode', 0) or 0)} | "
                f"{float(item.get('elapsed_sec', 0.0) or 0.0):.2f} |"
            )
    else:
        lines.append("| N/A | N/A | N/A |")

    packages = environment.get("packages") if isinstance(environment.get("packages"), list) else []
    lines.extend(["", "## Dependency Versions", "", "| Package | Version | Status |", "|---|---|---|"])
    for item in packages:
        if not isinstance(item, dict):
            continue
        lines.append(f"| {item.get('package')} | {item.get('version') or 'N/A'} | {item.get('status')} |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- 如果 `Git dirty=True`，说明本次运行使用了未提交或未跟踪的本地代码；复现时应同时保留源码快照或提交记录。",
            "- 如果某个大文件 hash 被跳过，JSON/CSV 中会记录 `hash_status`，可按需提高 `--hash_max_mb` 后重建卡片。",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def build_run_provenance(
    *,
    summary_json: str | Path,
    out_dir: str | Path | None = None,
    repo_root: str | Path | None = None,
    hash_max_mb: float = DEFAULT_HASH_MAX_MB,
) -> dict[str, Any]:
    summary_path = Path(summary_json).expanduser().resolve()
    if not summary_path.exists():
        raise FileNotFoundError(f"summary_json not found: {summary_path}")
    summary = _read_json(summary_path)
    run_out_dir = Path(str(summary.get("out_dir") or summary_path.parent)).expanduser().resolve()
    output_dir = Path(out_dir).expanduser().resolve() if out_dir is not None else run_out_dir / "provenance"
    repo = Path(repo_root).expanduser().resolve() if repo_root is not None else Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    max_hash_bytes = int(max(0.0, float(hash_max_mb)) * 1024 * 1024)

    input_records = _collect_input_records(
        summary=summary,
        summary_path=summary_path,
        out_dir=run_out_dir,
        repo_root=repo,
        max_hash_bytes=max_hash_bytes,
    )
    artifact_records = _collect_artifact_records(
        summary=summary,
        out_dir=run_out_dir,
        repo_root=repo,
        max_hash_bytes=max_hash_bytes,
    )
    source_records = _collect_source_records(
        summary=summary,
        out_dir=run_out_dir,
        repo_root=repo,
        max_hash_bytes=max_hash_bytes,
    )
    manifest_records = _dedupe_records([*input_records, *artifact_records, *source_records])
    parameter_payload = _parameter_payload(summary)
    input_file_manifest_path = output_dir / "run_input_file_manifest.csv"
    input_file_reference_records = _collect_input_file_references(
        summary=summary,
        repo_root=repo,
        max_hash_bytes=max_hash_bytes,
    )
    input_file_manifest_hash = _stable_hash(
        [
            {
                "source_table": record.get("source_table"),
                "data_row_number": record.get("data_row_number"),
                "nanobody_id": record.get("nanobody_id"),
                "conformer_id": record.get("conformer_id"),
                "pose_id": record.get("pose_id"),
                "column": record.get("column"),
                "value_source": record.get("value_source"),
                "original_value": record.get("original_value"),
                "relative_to_table_dir": record.get("relative_to_table_dir"),
                "relative_to_repo_root": record.get("relative_to_repo_root"),
                "exists": record.get("exists"),
                "type": record.get("type"),
                "size_bytes": record.get("size_bytes"),
                "sha256": record.get("sha256"),
                "hash_status": record.get("hash_status"),
                "status": record.get("status"),
            }
            for record in input_file_reference_records
        ]
    )

    card = {
        "created_at": _now_text(),
        "schema_version": 1,
        "run": {
            "summary_json": str(summary_path),
            "out_dir": str(run_out_dir),
            "start_mode": summary.get("start_mode"),
            "input_csv": summary.get("input_csv"),
            "feature_csv": summary.get("feature_csv"),
            "label_col": summary.get("label_col"),
            "label_valid_count": summary.get("label_valid_count"),
            "label_class_count": summary.get("label_class_count"),
            "ranking_config": summary.get("ranking_config"),
        },
        "parameter_payload": parameter_payload,
        "parameter_hash": _stable_hash(parameter_payload),
        "file_manifest": manifest_records,
        "file_manifest_hash": _stable_hash(
            [
                {
                    "group": record.get("group"),
                    "label": record.get("label"),
                    "relative_to_out_dir": record.get("relative_to_out_dir"),
                    "relative_to_repo_root": record.get("relative_to_repo_root"),
                    "size_bytes": record.get("size_bytes"),
                    "sha256": record.get("sha256"),
                    "hash_status": record.get("hash_status"),
                }
                for record in manifest_records
            ]
        ),
        "input_file_manifest": _input_file_manifest_summary(
            input_file_reference_records,
            input_file_manifest_path,
            input_file_manifest_hash,
        ),
        "environment": {
            "python_version": sys.version.replace("\n", " "),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "packages": _package_versions(DEFAULT_PACKAGES),
        },
        "git": _git_info(repo),
        "commands": summary.get("commands") if isinstance(summary.get("commands"), list) else [],
    }

    json_path = output_dir / "run_provenance_card.json"
    md_path = output_dir / "run_provenance_card.md"
    manifest_path = output_dir / "run_artifact_manifest.csv"
    integrity_path = output_dir / "run_provenance_integrity.json"
    json_path.write_text(json.dumps(card, ensure_ascii=True, indent=2), encoding="utf-8")
    md_path.write_text(_build_markdown(card), encoding="utf-8")
    _write_manifest_csv(manifest_path, manifest_records)
    _write_input_file_manifest_csv(input_file_manifest_path, input_file_reference_records)
    integrity_payload = _build_integrity_signature(
        created_at=str(card["created_at"]),
        json_path=json_path,
        md_path=md_path,
        manifest_path=manifest_path,
        input_file_manifest_path=input_file_manifest_path,
        parameter_hash=str(card["parameter_hash"]),
        file_manifest_hash=str(card["file_manifest_hash"]),
        input_file_manifest_hash=input_file_manifest_hash,
    )
    integrity_path.write_text(json.dumps(integrity_payload, ensure_ascii=True, indent=2), encoding="utf-8")

    return {
        "created_at": card["created_at"],
        "summary_json": str(summary_path),
        "out_dir": str(output_dir),
        "parameter_hash": card["parameter_hash"],
        "file_manifest_hash": card["file_manifest_hash"],
        "input_file_manifest_hash": input_file_manifest_hash,
        "signature_payload_hash": integrity_payload["signature_payload_hash"],
        "artifact_count": len(artifact_records),
        "input_count": len(input_records),
        "input_file_reference_count": len(input_file_reference_records),
        "source_count": len(source_records),
        "artifacts": {
            "run_provenance_card_json": str(json_path),
            "run_provenance_card_md": str(md_path),
            "run_artifact_manifest_csv": str(manifest_path),
            "run_input_file_manifest_csv": str(input_file_manifest_path),
            "run_provenance_integrity_json": str(integrity_path),
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a reproducibility provenance card for a pipeline run")
    parser.add_argument("--summary_json", required=True, help="Path to recommended_pipeline_summary.json")
    parser.add_argument("--out_dir", default=None, help="Output directory for provenance files")
    parser.add_argument("--repo_root", default=None, help="Repository root used for source hashes and git info")
    parser.add_argument("--hash_max_mb", type=float, default=DEFAULT_HASH_MAX_MB)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = build_run_provenance(
        summary_json=args.summary_json,
        out_dir=args.out_dir,
        repo_root=args.repo_root,
        hash_max_mb=float(args.hash_max_mb),
    )
    print(f"Run provenance card: {result['artifacts']['run_provenance_card_md']}")
    print(f"Run provenance integrity seal: {result['artifacts']['run_provenance_integrity_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
