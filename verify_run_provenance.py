from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _stable_hash(payload: Any) -> str:
    data = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_record_path(path_text: Any, signature_dir: Path) -> Path:
    path = Path(str(path_text or "")).expanduser()
    if not path.is_absolute():
        path = signature_dir / path
    return path.resolve()


def _verify_file_records(signature: dict[str, Any], signature_dir: Path) -> list[dict[str, Any]]:
    files = signature.get("files") if isinstance(signature.get("files"), dict) else {}
    rows: list[dict[str, Any]] = []
    for key, record in sorted(files.items()):
        if not isinstance(record, dict):
            rows.append(
                {
                    "key": str(key),
                    "path": "",
                    "expected_sha256": "",
                    "actual_sha256": "",
                    "expected_size_bytes": "",
                    "actual_size_bytes": "",
                    "status": "invalid_signature_record",
                }
            )
            continue
        path = _resolve_record_path(record.get("path"), signature_dir)
        expected_sha = str(record.get("sha256") or "")
        expected_size = record.get("size_bytes")
        actual_sha = _sha256_file(path)
        actual_size = int(path.stat().st_size) if path.exists() and path.is_file() else None
        status = "ok"
        if actual_sha is None:
            status = "missing"
        elif expected_sha and actual_sha != expected_sha:
            status = "hash_mismatch"
        elif expected_size is not None and actual_size != int(expected_size):
            status = "size_mismatch"
        rows.append(
            {
                "key": str(key),
                "path": str(path),
                "expected_sha256": expected_sha,
                "actual_sha256": actual_sha or "",
                "expected_size_bytes": expected_size,
                "actual_size_bytes": actual_size,
                "status": status,
            }
        )
    return rows


def _expected_payload_hash(signature: dict[str, Any]) -> str:
    payload = {
        "files": signature.get("files") if isinstance(signature.get("files"), dict) else {},
        "provenance_hashes": signature.get("provenance_hashes")
        if isinstance(signature.get("provenance_hashes"), dict)
        else {},
    }
    return _stable_hash(payload)


def _build_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Run Provenance Integrity Verification",
        "",
        f"- Signature JSON: `{summary['signature_json']}`",
        f"- Verified at: `{summary['verified_at']}`",
        f"- Overall status: `{summary['overall_status']}`",
        f"- Files checked: `{summary['file_count']}`",
        f"- Failed files: `{summary['failed_file_count']}`",
        f"- Signature payload status: `{summary['signature_payload_status']}`",
        "",
        "## File Checks",
        "",
        "| Key | Status | Expected SHA256 | Actual SHA256 |",
        "| --- | --- | --- | --- |",
    ]
    for row in summary.get("file_checks", []):
        expected = str(row.get("expected_sha256") or "")
        actual = str(row.get("actual_sha256") or "")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("key") or ""),
                    str(row.get("status") or ""),
                    expected[:16] + "..." if len(expected) > 16 else expected,
                    actual[:16] + "..." if len(actual) > 16 else actual,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Summary JSON: `{summary['outputs']['summary_json']}`",
            f"- Report MD: `{summary['outputs']['report_md']}`",
            "",
        ]
    )
    return "\n".join(lines)


def verify_run_provenance(
    *,
    signature_json: str | Path,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    signature_path = Path(signature_json).expanduser().resolve()
    if not signature_path.exists():
        raise FileNotFoundError(f"signature_json not found: {signature_path}")
    signature = _read_json(signature_path)
    output_dir = Path(out_dir).expanduser().resolve() if out_dir is not None else signature_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    file_checks = _verify_file_records(signature, signature_path.parent)
    expected_payload_hash = str(signature.get("signature_payload_hash") or "")
    actual_payload_hash = _expected_payload_hash(signature)
    signature_payload_status = "ok" if expected_payload_hash == actual_payload_hash else "signature_payload_hash_mismatch"
    failed_file_count = sum(1 for row in file_checks if row.get("status") != "ok")
    overall_status = "ok" if failed_file_count == 0 and signature_payload_status == "ok" else "failed"

    summary_json = output_dir / "run_provenance_integrity_verification.json"
    report_md = output_dir / "run_provenance_integrity_verification.md"
    summary = {
        "verified_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "signature_json": str(signature_path),
        "overall_status": overall_status,
        "file_count": int(len(file_checks)),
        "failed_file_count": int(failed_file_count),
        "signature_payload_status": signature_payload_status,
        "expected_signature_payload_hash": expected_payload_hash,
        "actual_signature_payload_hash": actual_payload_hash,
        "file_checks": file_checks,
        "outputs": {
            "summary_json": str(summary_json),
            "report_md": str(report_md),
        },
    }
    _write_json(summary_json, summary)
    report_md.write_text(_build_report(summary), encoding="utf-8")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify a run_provenance_integrity.json seal.")
    parser.add_argument("--signature_json", required=True, help="Path to run_provenance_integrity.json")
    parser.add_argument("--out_dir", default=None, help="Output directory for verification summary/report")
    parser.add_argument("--strict", action="store_true", help="Return non-zero when verification fails")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    summary = verify_run_provenance(signature_json=args.signature_json, out_dir=args.out_dir)
    print(f"Saved: {summary['outputs']['summary_json']}")
    print(f"Saved: {summary['outputs']['report_md']}")
    print(f"Overall status: {summary['overall_status']}")
    if args.strict and summary["overall_status"] != "ok":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
