from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

from app_metadata import APP_NAME, APP_RELEASE_CHANNEL, APP_RELEASE_DATE, APP_VERSION


def _compute_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_portable_release(
    *,
    bundle_dir: str | Path,
    zip_path: str | Path,
) -> tuple[Path, Path]:
    bundle_dir = Path(bundle_dir).expanduser().resolve()
    zip_path = Path(zip_path).expanduser().resolve()
    manifest_path = zip_path.with_suffix(".manifest.json")

    if not bundle_dir.exists():
        raise FileNotFoundError(f"Portable bundle directory not found: {bundle_dir}")
    if not (bundle_dir / "ML_Local_App.exe").exists():
        raise FileNotFoundError(f"Portable launcher missing: {bundle_dir / 'ML_Local_App.exe'}")
    if not (bundle_dir / "app").exists():
        raise FileNotFoundError(f"Portable app directory missing: {bundle_dir / 'app'}")

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    if manifest_path.exists():
        manifest_path.unlink()

    entry_rows: list[dict[str, Any]] = []
    root_parent = bundle_dir.parent

    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for file_path in sorted(bundle_dir.rglob("*")):
            if not file_path.is_file():
                continue
            arcname = file_path.relative_to(root_parent)
            zf.write(file_path, arcname=str(arcname).replace("\\", "/"))
            entry_rows.append(
                {
                    "relative_path": str(arcname).replace("\\", "/"),
                    "size_bytes": int(file_path.stat().st_size),
                }
            )

    manifest = {
        "release_type": "ml_portable_zip_release",
        "app_name": APP_NAME,
        "app_version": APP_VERSION,
        "release_channel": APP_RELEASE_CHANNEL,
        "release_date": APP_RELEASE_DATE,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "bundle_dir": str(bundle_dir),
        "zip_path": str(zip_path),
        "zip_size_bytes": int(zip_path.stat().st_size),
        "zip_sha256": _compute_sha256(zip_path),
        "entry_count": len(entry_rows),
        "entries": entry_rows,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return zip_path, manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a distributable zip from the portable ML app bundle")
    parser.add_argument("--bundle_dir", default="portable_dist/ML_Portable", help="Portable bundle directory")
    parser.add_argument("--zip_path", default="portable_dist/ML_Portable_release.zip", help="Output zip path")
    args = parser.parse_args(argv)

    zip_path, manifest_path = build_portable_release(
        bundle_dir=args.bundle_dir,
        zip_path=args.zip_path,
    )
    print(f"Portable release zip created: {zip_path}")
    print(f"Release manifest: {manifest_path}")
    print(f"App version: {APP_VERSION}")
    print(f"ZIP size bytes: {zip_path.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
