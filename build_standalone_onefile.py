from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from app_metadata import APP_NAME, APP_RELEASE_CHANNEL, APP_RELEASE_DATE, APP_VERSION
from build_portable_bundle import build_portable_app_dir, _safe_rmtree


def _compute_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_builder_python(repo_root: Path) -> str:
    candidate = repo_root / ".venv" / "Scripts" / "python.exe"
    if candidate.exists():
        return str(candidate)
    return sys.executable


def build_standalone_onefile(
    *,
    repo_root: str | Path,
    dist_dir: str | Path,
    work_dir: str | Path,
    output_name: str = "ML_Local_App_Standalone",
) -> tuple[Path, Path]:
    repo_root = Path(repo_root).expanduser().resolve()
    dist_dir = Path(dist_dir).expanduser().resolve()
    work_dir = Path(work_dir).expanduser().resolve()
    staging_app_dir = work_dir / "embedded_app" / "app"
    python_executable = _resolve_builder_python(repo_root)
    launcher_script = repo_root / "ml_desktop_launcher.py"
    icon_path = repo_root / "assets" / "app_icon.ico"

    if not launcher_script.exists():
        raise FileNotFoundError(f"Launcher script not found: {launcher_script}")

    if work_dir.exists():
        _safe_rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    dist_dir.mkdir(parents=True, exist_ok=True)

    embedded_entries = build_portable_app_dir(
        repo_root=repo_root,
        app_dir=staging_app_dir,
    )

    command = [
        python_executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--windowed",
        "--name",
        output_name,
        "--distpath",
        str(dist_dir),
        "--workpath",
        str(work_dir / "pyinstaller_work"),
        "--specpath",
        str(work_dir / "pyinstaller_spec"),
        "--add-data",
        f"{staging_app_dir};app",
        str(launcher_script),
    ]
    if icon_path.exists():
        command.extend(["--icon", str(icon_path)])

    subprocess.run(
        command,
        cwd=str(repo_root),
        check=True,
    )

    exe_path = dist_dir / f"{output_name}.exe"
    if not exe_path.exists():
        raise FileNotFoundError(f"Standalone exe was not generated: {exe_path}")

    manifest_path = exe_path.with_suffix(".manifest.json")
    embedded_app_size = sum(path.stat().st_size for path in staging_app_dir.rglob("*") if path.is_file())
    manifest = {
        "release_type": "ml_single_file_standalone_bundle",
        "app_name": APP_NAME,
        "app_version": APP_VERSION,
        "release_channel": APP_RELEASE_CHANNEL,
        "release_date": APP_RELEASE_DATE,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "repo_root": str(repo_root),
        "embedded_app_prefix": "app/",
        "runtime_extraction_mode": "pyinstaller_onefile_meipass",
        "output_exe": str(exe_path),
        "output_exe_size_bytes": int(exe_path.stat().st_size),
        "output_exe_sha256": _compute_sha256(exe_path),
        "embedded_app_size_bytes": int(embedded_app_size),
        "embedded_entry_count": len(embedded_entries),
        "embedded_entries": embedded_entries,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    _safe_rmtree(work_dir)
    return exe_path, manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a single-file standalone launcher bundle for the ML local app")
    parser.add_argument("--repo_root", default=".", help="Repository root")
    parser.add_argument("--dist_dir", default="portable_dist/standalone_onefile", help="Standalone exe output directory")
    parser.add_argument("--work_dir", default="build/standalone_onefile_build", help="Temporary staging/work directory")
    parser.add_argument("--output_name", default="ML_Local_App_Standalone", help="Standalone exe name")
    args = parser.parse_args(argv)

    exe_path, manifest_path = build_standalone_onefile(
        repo_root=args.repo_root,
        dist_dir=args.dist_dir,
        work_dir=args.work_dir,
        output_name=args.output_name,
    )
    print(f"Standalone exe created: {exe_path}")
    print(f"Standalone manifest: {manifest_path}")
    print(f"App version: {APP_VERSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
