from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


def _run_validation(
    *,
    exe_path: Path,
    host_repo_root: Path | None = None,
    report_dir: Path,
    timeout_sec: int = 180,
) -> Path:
    if not exe_path.exists():
        raise FileNotFoundError(f"Standalone exe not found: {exe_path}")

    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"standalone_validation_{timestamp}.json"
    latest_report_path = report_dir / "standalone_validation_latest.json"

    with tempfile.TemporaryDirectory(prefix="ml_standalone_validate_") as tmp_dir:
        sandbox_root = Path(tmp_dir).resolve()
        exe_copy = sandbox_root / exe_path.name
        payload_path = sandbox_root / "standalone_selftest_payload.json"
        shutil.copy2(exe_path, exe_copy)

        completed = subprocess.run(
            [str(exe_copy), "--selftest-json", str(payload_path)],
            cwd=str(sandbox_root),
            timeout=max(30, int(timeout_sec)),
            check=False,
        )

        payload: dict[str, Any] = {}
        if payload_path.exists():
            payload = json.loads(payload_path.read_text(encoding="utf-8"))

        checks: list[dict[str, Any]] = []

        def add_check(name: str, ok: bool, detail: str) -> None:
            checks.append({"name": name, "ok": bool(ok), "detail": detail})

        add_check("process_exit_code_zero", completed.returncode == 0, f"returncode={completed.returncode}")
        add_check("selftest_payload_exists", payload_path.exists(), f"payload_path={payload_path}")

        repo_root = str(payload.get("repo_root") or "")
        repo_root_source = str(payload.get("repo_root_source") or "")
        meipass_dir = str(payload.get("meipass_dir") or "")
        python_executable = str(payload.get("python_executable") or "")
        candidate_dirs = payload.get("candidate_dirs") if isinstance(payload.get("candidate_dirs"), list) else []

        add_check("repo_root_source_is_meipass_app", repo_root_source == "meipass_app", f"repo_root_source={repo_root_source}")
        add_check(
            "repo_root_under_meipass_app",
            bool(meipass_dir) and repo_root == str(Path(meipass_dir) / "app"),
            f"meipass_dir={meipass_dir}; repo_root={repo_root}",
        )
        add_check(
            "python_executable_under_embedded_venv",
            bool(meipass_dir) and python_executable.startswith(str(Path(meipass_dir) / "app" / ".venv" / "Scripts")),
            f"python_executable={python_executable}",
        )
        add_check(
            "launcher_dependencies_ok",
            bool(payload.get("launcher_dependency_ok")),
            f"launcher_dependency_ok={payload.get('launcher_dependency_ok')}",
        )
        add_check(
            "app_script_exists",
            bool(payload.get("app_script_exists")),
            f"app_script_exists={payload.get('app_script_exists')}",
        )
        add_check(
            "candidate_dirs_include_meipass",
            bool(meipass_dir) and str(Path(meipass_dir)) in [str(item) for item in candidate_dirs],
            f"candidate_dirs={candidate_dirs}",
        )
        if host_repo_root is not None:
            add_check(
                "repo_root_not_equal_host_repo_root",
                repo_root.lower() != str(host_repo_root).lower(),
                f"repo_root={repo_root}; host_repo_root={host_repo_root}",
            )

        report = {
            "validated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "exe_path": str(exe_path),
            "sandbox_exe_path": str(exe_copy),
            "sandbox_root": str(sandbox_root),
            "payload_path": str(payload_path),
            "returncode": int(completed.returncode),
            "payload": payload,
            "checks": checks,
            "ok": all(bool(item.get("ok")) for item in checks),
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        latest_report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the standalone onefile launcher outside the repo tree")
    parser.add_argument("--exe_path", default="portable_dist/standalone_onefile/ML_Local_App_Standalone.exe", help="Path to standalone exe")
    parser.add_argument("--host_repo_root", default=".", help="Host repo root used to ensure validation is not falling back to source tree")
    parser.add_argument("--report_dir", default="portable_dist/standalone_onefile_validation", help="Directory for validation reports")
    parser.add_argument("--timeout_sec", type=int, default=180, help="Timeout for standalone selftest process")
    args = parser.parse_args(argv)

    report_path = _run_validation(
        exe_path=Path(args.exe_path).expanduser().resolve(),
        host_repo_root=Path(args.host_repo_root).expanduser().resolve(),
        report_dir=Path(args.report_dir).expanduser().resolve(),
        timeout_sec=int(args.timeout_sec),
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    print(f"Standalone validation report: {report_path}")
    print(f"Standalone validation ok: {report.get('ok')}")
    return 0 if bool(report.get("ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
