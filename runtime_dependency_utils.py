from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


_DEPENDENCY_REGISTRY: dict[str, dict[str, str]] = {
    "streamlit": {
        "package_name": "streamlit",
        "import_name": "streamlit",
        "display_name": "Streamlit",
        "stage_key": "launcher_ui",
        "stage_label": "桌面启动 / 本地交互界面",
    },
    "torch": {
        "package_name": "torch",
        "import_name": "torch",
        "display_name": "PyTorch",
        "stage_key": "train_pose_model",
        "stage_label": "ML 训练 / pose 预测",
    },
    "biopython": {
        "package_name": "biopython",
        "import_name": "Bio",
        "display_name": "Biopython",
        "stage_key": "build_feature_table",
        "stage_label": "PDB 解析 / 特征构建",
    },
}


def _normalize_python_for_check(python_executable: str | None) -> str:
    if python_executable is None:
        return str(Path(sys.executable).resolve())

    candidate = Path(str(python_executable)).expanduser().resolve()
    if candidate.name.lower() == "pythonw.exe":
        console_python = candidate.with_name("python.exe")
        if console_python.exists():
            return str(console_python)
    return str(candidate)


def _build_dependency_specs(keys: list[str]) -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    for key in keys:
        payload = _DEPENDENCY_REGISTRY.get(str(key))
        if payload is None:
            raise KeyError(f"Unknown dependency key: {key}")
        specs.append(dict(payload))
    return specs


def get_pipeline_runtime_dependency_specs(start_mode: str) -> list[dict[str, str]]:
    mode = str(start_mode or "input_csv").strip().lower()
    keys = ["torch"]
    if mode == "input_csv":
        keys.insert(0, "biopython")
    return _build_dependency_specs(keys)


def get_launcher_runtime_dependency_specs() -> list[dict[str, str]]:
    return _build_dependency_specs(["streamlit"])


def check_runtime_dependencies(
    specs: list[dict[str, str]],
    *,
    python_executable: str | None = None,
    timeout_sec: float = 20.0,
) -> dict[str, Any]:
    checked_python = _normalize_python_for_check(python_executable)
    report: dict[str, Any] = {
        "ok": True,
        "checked_python_executable": checked_python,
        "checked_dependencies": [dict(item) for item in specs],
        "missing_dependencies": [],
        "error_message": None,
        "check_method": "importlib.util.find_spec",
    }
    if not specs:
        return report

    probe_script = (
        "import importlib.util, json, sys\n"
        "specs = json.loads(sys.argv[1])\n"
        "missing = []\n"
        "for item in specs:\n"
        "    import_name = str(item.get('import_name') or '').strip()\n"
        "    if not import_name:\n"
        "        missing_item = dict(item)\n"
        "        missing_item['error'] = 'missing import_name'\n"
        "        missing.append(missing_item)\n"
        "        continue\n"
        "    if importlib.util.find_spec(import_name) is None:\n"
        "        missing_item = dict(item)\n"
        "        missing_item['error'] = f'module {import_name!r} not found'\n"
        "        missing.append(missing_item)\n"
        "print(json.dumps({'missing': missing}, ensure_ascii=True))\n"
    )

    try:
        proc = subprocess.run(
            [checked_python, "-c", probe_script, json.dumps(specs, ensure_ascii=True)],
            text=True,
            capture_output=True,
            timeout=max(1.0, float(timeout_sec)),
            check=False,
        )
    except subprocess.TimeoutExpired:
        report["ok"] = False
        report["error_message"] = f"Dependency probe timed out after {float(timeout_sec):.1f}s."
        return report
    except Exception as exc:
        report["ok"] = False
        report["error_message"] = f"Dependency probe failed: {exc}"
        return report

    if proc.returncode != 0:
        stderr_tail = "\n".join((proc.stderr or "").splitlines()[-12:]).strip()
        report["ok"] = False
        report["error_message"] = (
            f"Dependency probe exited with rc={proc.returncode}."
            + (f" STDERR: {stderr_tail}" if stderr_tail else "")
        )
        return report

    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        report["ok"] = False
        report["error_message"] = f"Dependency probe returned invalid JSON: {exc}"
        return report

    missing = payload.get("missing") if isinstance(payload, dict) and isinstance(payload.get("missing"), list) else []
    report["missing_dependencies"] = missing
    report["ok"] = not bool(missing)
    return report


def format_missing_dependency_summary(missing_dependencies: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for item in missing_dependencies:
        package_name = str(item.get("package_name") or item.get("display_name") or "unknown")
        stage_label = str(item.get("stage_label") or item.get("stage_key") or "unknown stage")
        parts.append(f"{package_name} ({stage_label})")
    return ", ".join(parts)


__all__ = [
    "check_runtime_dependencies",
    "format_missing_dependency_summary",
    "get_launcher_runtime_dependency_specs",
    "get_pipeline_runtime_dependency_specs",
]
