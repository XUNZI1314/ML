from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any
from tkinter import BOTH, LEFT, RIGHT, X, Button, Entry, Frame, Label, StringVar, Tk, messagebox

from app_metadata import APP_NAME, APP_RELEASE_CHANNEL, APP_VERSION
from runtime_dependency_utils import (
    check_runtime_dependencies,
    format_missing_dependency_summary,
    get_launcher_runtime_dependency_specs,
)


def _get_meipass_dir() -> Path | None:
    value = getattr(sys, "_MEIPASS", None)
    if not value:
        return None
    try:
        return Path(str(value)).resolve()
    except Exception:
        return None


def _candidate_start_dirs() -> list[Path]:
    candidates: list[Path] = []

    meipass_dir = _get_meipass_dir()
    exe_dir = Path(sys.executable).resolve().parent
    script_dir = Path(__file__).resolve().parent
    cwd_dir = Path.cwd().resolve()

    base_dirs = [path for path in [meipass_dir, exe_dir, script_dir, cwd_dir] if path is not None]
    for path in base_dirs:
        if path not in candidates:
            candidates.append(path)
        portable_app_dir = path / "app"
        if portable_app_dir not in candidates:
            candidates.append(portable_app_dir)
    return candidates


def _locate_repo_root() -> tuple[Path, str]:
    markers = ["local_ml_app.py", "run_recommended_pipeline.py", "requirements.txt"]
    meipass_dir = _get_meipass_dir()
    if meipass_dir is not None:
        packaged_app_dir = meipass_dir / "app"
        if all((packaged_app_dir / marker).exists() for marker in markers):
            return packaged_app_dir, "meipass_app"

    exe_dir = Path(sys.executable).resolve().parent
    portable_app_dir = exe_dir / "app"
    if all((portable_app_dir / marker).exists() for marker in markers):
        return portable_app_dir, "portable_app_dir"

    for start in _candidate_start_dirs():
        current = start
        for candidate in [current] + list(current.parents):
            if all((candidate / marker).exists() for marker in markers):
                return candidate, "ancestor_search"
    raise FileNotFoundError("Could not locate repo root containing local_ml_app.py and requirements.txt.")


def _find_repo_root() -> Path:
    repo_root, _ = _locate_repo_root()
    return repo_root


def _resolve_python_executable(repo_root: Path) -> str:
    venv_scripts = repo_root / ".venv" / "Scripts"
    for name in ["pythonw.exe", "python.exe"]:
        candidate = venv_scripts / name
        if candidate.exists():
            return str(candidate)

    for name in ["pythonw.exe", "python.exe", "python", "py"]:
        found = shutil.which(name)
        if found:
            return str(found)

    raise FileNotFoundError("Could not locate a usable Python executable.")


def _resolve_icon_path(repo_root: Path) -> Path | None:
    candidate = repo_root / "assets" / "app_icon.ico"
    return candidate if candidate.exists() else None


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_streamlit(url: str, timeout_sec: float = 60.0) -> bool:
    deadline = time.time() + max(1.0, float(timeout_sec))
    health_url = url.rstrip("/") + "/_stcore/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=2.0) as response:
                if int(getattr(response, "status", 200)) < 500:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def _open_local_path(path: Path) -> None:
    target = path.expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"Path not found: {target}")
    os.startfile(str(target))  # type: ignore[attr-defined]


def _show_error(title: str, text: str) -> None:
    try:
        root = Tk()
        root.withdraw()
        messagebox.showerror(title, text)
        root.destroy()
    except Exception:
        print(f"{title}: {text}", file=sys.stderr)


def _build_streamlit_command(
    python_executable: str,
    app_script: Path,
    port: int,
) -> list[str]:
    return [
        python_executable,
        "-m",
        "streamlit",
        "run",
        str(app_script),
        "--server.headless",
        "true",
        "--server.port",
        str(int(port)),
        "--browser.gatherUsageStats",
        "false",
        "--server.fileWatcherType",
        "none",
    ]


def _check_launcher_runtime_dependencies(python_executable: str) -> dict[str, Any]:
    return check_runtime_dependencies(
        get_launcher_runtime_dependency_specs(),
        python_executable=python_executable,
    )


def _raise_if_launcher_dependencies_missing(python_executable: str) -> dict[str, Any]:
    dependency_report = _check_launcher_runtime_dependencies(python_executable)
    error_message = str(dependency_report.get("error_message") or "").strip()
    if error_message:
        raise RuntimeError(f"Launcher dependency precheck failed: {error_message}")

    missing_dependencies = (
        dependency_report.get("missing_dependencies")
        if isinstance(dependency_report.get("missing_dependencies"), list)
        else []
    )
    if missing_dependencies:
        summary_text = format_missing_dependency_summary(missing_dependencies)
        raise RuntimeError(
            "Current runtime is missing required Python packages for the launcher: "
            f"{summary_text}. Install requirements first, then retry."
        )
    return dependency_report


def _collect_selftest_payload() -> dict[str, Any]:
    repo_root, repo_root_source = _locate_repo_root()
    python_executable = _resolve_python_executable(repo_root)
    dependency_report = _check_launcher_runtime_dependencies(python_executable)
    missing_dependencies = (
        dependency_report.get("missing_dependencies")
        if isinstance(dependency_report.get("missing_dependencies"), list)
        else []
    )
    app_script = repo_root / "local_ml_app.py"
    icon_path = _resolve_icon_path(repo_root)
    meipass_dir = _get_meipass_dir()
    payload = {
        "app_name": APP_NAME,
        "app_version": APP_VERSION,
        "release_channel": APP_RELEASE_CHANNEL,
        "repo_root": str(repo_root),
        "repo_root_source": repo_root_source,
        "python_executable": str(python_executable),
        "meipass_dir": None if meipass_dir is None else str(meipass_dir),
        "launcher_dependency_ok": bool(dependency_report.get("ok")),
        "launcher_missing_dependencies": format_missing_dependency_summary(missing_dependencies) if missing_dependencies else "",
        "app_script_exists": bool(app_script.exists()),
        "icon_path": None if icon_path is None else str(icon_path),
        "streamlit_batch": str(repo_root / "start_local_app.bat"),
        "candidate_dirs": [str(path) for path in _candidate_start_dirs()],
        "cwd": str(Path.cwd().resolve()),
        "exe_path": str(Path(sys.executable).resolve()),
        "ok": bool(dependency_report.get("ok")) and bool(app_script.exists()),
        "error_message": "",
    }
    return payload


def _write_selftest_json(path_text: str, payload: dict[str, Any]) -> None:
    target = Path(str(path_text)).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _print_selftest_payload(payload: dict[str, Any]) -> None:
    print(f"app_name={payload.get('app_name')}")
    print(f"app_version={payload.get('app_version')}")
    print(f"release_channel={payload.get('release_channel')}")
    print(f"repo_root={payload.get('repo_root')}")
    print(f"repo_root_source={payload.get('repo_root_source')}")
    print(f"python_executable={payload.get('python_executable')}")
    print(f"meipass_dir={payload.get('meipass_dir')}")
    print(f"launcher_dependency_ok={payload.get('launcher_dependency_ok')}")
    print(f"launcher_missing_dependencies={payload.get('launcher_missing_dependencies')}")
    print(f"app_script_exists={payload.get('app_script_exists')}")
    print(f"icon_path={payload.get('icon_path')}")
    print(f"streamlit_batch={payload.get('streamlit_batch')}")
    print("candidate_dirs=" + " | ".join([str(item) for item in payload.get("candidate_dirs") or []]))


def run_selftest(json_path: str = "") -> int:
    try:
        payload = _collect_selftest_payload()
    except Exception as exc:
        payload = {
            "app_name": APP_NAME,
            "app_version": APP_VERSION,
            "release_channel": APP_RELEASE_CHANNEL,
            "repo_root": "",
            "repo_root_source": "",
            "python_executable": "",
            "meipass_dir": None,
            "launcher_dependency_ok": False,
            "launcher_missing_dependencies": "",
            "app_script_exists": False,
            "icon_path": None,
            "streamlit_batch": "",
            "candidate_dirs": [str(path) for path in _candidate_start_dirs()],
            "cwd": str(Path.cwd().resolve()),
            "exe_path": str(Path(sys.executable).resolve()),
            "ok": False,
            "error_message": str(exc),
        }

    if json_path:
        _write_selftest_json(json_path, payload)
    _print_selftest_payload(payload)
    return 0 if bool(payload.get("ok")) else 1


class DesktopLauncherApp:
    def __init__(self, repo_root: Path, python_executable: str) -> None:
        self.repo_root = repo_root
        self.python_executable = python_executable
        self.app_script = repo_root / "local_ml_app.py"
        self.port = _pick_free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.proc: subprocess.Popen[str] | None = None
        self.log_path = repo_root / "local_app_runs" / "desktop_launcher_runtime.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.window = Tk()
        self.status_var = StringVar(master=self.window, value="正在启动本地界面...")
        self.note_var = StringVar(
            master=self.window,
            value="首次启动通常需要几秒；启动成功后会自动在浏览器中打开本地界面。",
        )
        self.url_var = StringVar(master=self.window, value=self.url)
        self.colors = {
            "window_bg": "#f3f6fb",
            "card_bg": "#ffffff",
            "card_border": "#d7deea",
            "text": "#0f172a",
            "muted": "#526277",
            "accent": "#0b57d0",
            "status_info": "#1d4ed8",
            "status_success": "#166534",
            "status_warning": "#92400e",
            "status_error": "#b91c1c",
        }
        self.status_label: Label | None = None
        self.window.title(f"{APP_NAME} v{APP_VERSION}")
        self.window.geometry("720x420")
        self.window.resizable(False, False)
        self.window.protocol("WM_DELETE_WINDOW", self.stop_and_close)
        self._apply_window_icon()
        self._build_ui()
        self._set_status(self.status_var.get(), tone="info")
        self._center_window()

    def _apply_window_icon(self) -> None:
        icon_path = _resolve_icon_path(self.repo_root)
        if icon_path is None:
            return
        try:
            self.window.iconbitmap(default=str(icon_path))
        except Exception:
            pass

    def _build_ui(self) -> None:
        self.window.configure(bg=self.colors["window_bg"], padx=18, pady=18)

        container = Frame(self.window, bg=self.colors["window_bg"])
        container.pack(fill=BOTH, expand=True)

        header = Frame(container, bg=self.colors["window_bg"])
        header.pack(fill=X)
        Label(
            header,
            text=f"{APP_NAME} 桌面启动器",
            font=("Microsoft YaHei UI", 16, "bold"),
            bg=self.colors["window_bg"],
            fg=self.colors["text"],
        ).pack(anchor="w")
        Label(
            header,
            text=f"版本 {APP_VERSION} | 通道 {APP_RELEASE_CHANNEL}",
            justify=LEFT,
            bg=self.colors["window_bg"],
            fg=self.colors["muted"],
        ).pack(anchor="w", pady=(5, 0))
        Label(
            header,
            text="建议直接保留这个窗口，启动完成后会自动打开浏览器中的本地软件界面。",
            wraplength=660,
            justify=LEFT,
            bg=self.colors["window_bg"],
            fg=self.colors["muted"],
        ).pack(anchor="w", pady=(8, 0))

        status_card = self._create_card(
            container,
            title="启动状态",
            subtitle="如果首次打开较慢，一般是 Python 环境和 Streamlit 正在完成初始化。",
        )
        self.status_label = Label(
            status_card,
            textvariable=self.status_var,
            wraplength=630,
            justify=LEFT,
            font=("Microsoft YaHei UI", 11, "bold"),
            bg=self.colors["card_bg"],
            fg=self.colors["status_info"],
        )
        self.status_label.pack(anchor="w", fill=X)

        url_card = self._create_card(
            container,
            title="访问地址",
            subtitle="浏览器没有自动弹出时，直接复制下面的地址打开即可。",
        )
        url_row = Frame(url_card, bg=self.colors["card_bg"])
        url_row.pack(fill=X)
        Entry(
            url_row,
            textvariable=self.url_var,
            state="readonly",
            relief="flat",
            readonlybackground="#eef4ff",
            fg=self.colors["accent"],
            font=("Consolas", 10),
        ).pack(side=LEFT, fill=X, expand=True, ipady=7)

        url_actions = Frame(url_card, bg=self.colors["card_bg"])
        url_actions.pack(fill=X, pady=(10, 0))
        Button(url_actions, text="打开浏览器", command=self.open_browser, width=14).pack(side=LEFT, padx=(0, 8))
        Button(url_actions, text="复制地址", command=self.copy_url, width=14).pack(side=LEFT)

        path_card = self._create_card(
            container,
            title="常用位置",
            subtitle="这里只保留最常用的目录入口，方便你快速定位运行结果和日志。",
        )
        Label(
            path_card,
            text=f"项目目录: {self.repo_root}",
            wraplength=630,
            justify=LEFT,
            bg=self.colors["card_bg"],
            fg=self.colors["text"],
        ).pack(anchor="w")
        Label(
            path_card,
            text=f"运行日志: {self.log_path}",
            wraplength=630,
            justify=LEFT,
            bg=self.colors["card_bg"],
            fg=self.colors["muted"],
        ).pack(anchor="w", pady=(6, 0))

        path_actions = Frame(path_card, bg=self.colors["card_bg"])
        path_actions.pack(fill=X, pady=(10, 0))
        Button(path_actions, text="打开项目目录", command=self.open_repo_root, width=14).pack(side=LEFT, padx=(0, 8))
        Button(path_actions, text="打开运行产物", command=self.open_run_root, width=14).pack(side=LEFT, padx=(0, 8))
        Button(path_actions, text="打开日志", command=self.open_launcher_log, width=14).pack(side=LEFT)

        footer = Frame(container, bg=self.colors["window_bg"])
        footer.pack(fill=X, pady=(16, 0))
        Label(
            footer,
            textvariable=self.note_var,
            wraplength=500,
            justify=LEFT,
            bg=self.colors["window_bg"],
            fg=self.colors["muted"],
        ).pack(side=LEFT, fill=BOTH, expand=True)
        Button(footer, text="停止并退出", command=self.stop_and_close, width=16).pack(side=RIGHT)

    def _create_card(self, parent: Frame, title: str, subtitle: str) -> Frame:
        card = Frame(
            parent,
            bg=self.colors["card_bg"],
            highlightbackground=self.colors["card_border"],
            highlightthickness=1,
        )
        card.pack(fill=X, pady=(14, 0))
        Label(
            card,
            text=title,
            font=("Microsoft YaHei UI", 11, "bold"),
            bg=self.colors["card_bg"],
            fg=self.colors["text"],
        ).pack(anchor="w", padx=14, pady=(12, 0))
        Label(
            card,
            text=subtitle,
            wraplength=630,
            justify=LEFT,
            bg=self.colors["card_bg"],
            fg=self.colors["muted"],
        ).pack(anchor="w", padx=14, pady=(4, 0))

        body = Frame(card, bg=self.colors["card_bg"])
        body.pack(fill=X, padx=14, pady=(10, 12))
        return body

    def _center_window(self) -> None:
        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        x_pos = max((self.window.winfo_screenwidth() - width) // 2, 0)
        y_pos = max((self.window.winfo_screenheight() - height) // 2 - 30, 0)
        self.window.geometry(f"{width}x{height}+{x_pos}+{y_pos}")

    def _set_status(self, text: str, tone: str = "info") -> None:
        palette = {
            "info": self.colors["status_info"],
            "success": self.colors["status_success"],
            "warning": self.colors["status_warning"],
            "error": self.colors["status_error"],
        }
        self.status_var.set(text)
        if self.status_label is not None:
            self.status_label.configure(fg=palette.get(tone, self.colors["status_info"]))

    def _set_note(self, text: str) -> None:
        self.note_var.set(text)

    def start(self) -> None:
        command = _build_streamlit_command(self.python_executable, self.app_script, self.port)
        log_file = self.log_path.open("a", encoding="utf-8")
        log_file.write(f"\n==== launcher start {time.strftime('%Y-%m-%d %H:%M:%S')} ====\n")
        log_file.write("CMD: " + " ".join(command) + "\n")
        log_file.flush()

        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        self.proc = subprocess.Popen(
            command,
            cwd=str(self.repo_root),
            stdout=log_file,
            stderr=log_file,
            text=True,
            creationflags=creationflags,
        )

        threading.Thread(target=self._wait_until_ready, daemon=True).start()
        self.window.after(1000, self._poll_process)
        self.window.mainloop()

    def _wait_until_ready(self) -> None:
        if _wait_for_streamlit(self.url):
            self.window.after(0, lambda: self._set_status("本地界面已启动，可以在浏览器中使用。", tone="success"))
            self.window.after(0, lambda: self._set_note("浏览器应已自动打开；如果没有自动弹出，可以复制上方地址手动访问。"))
            self.window.after(0, self.open_browser)
        else:
            self.window.after(0, lambda: self._set_status("本地界面启动超时，请查看运行日志。", tone="warning"))
            self.window.after(0, lambda: self._set_note("如果长时间没有响应，优先点击“打开日志”查看启动报错。"))

    def _poll_process(self) -> None:
        if self.proc is None:
            return
        rc = self.proc.poll()
        if rc is None:
            self.window.after(1000, self._poll_process)
            return
        tone = "info" if rc == 0 else "warning"
        self._set_status(f"本地界面已退出，返回码: {rc}", tone=tone)
        self._set_note("如果是误关闭，可以重新双击启动器再次打开。")

    def open_browser(self) -> None:
        webbrowser.open(self.url, new=1)

    def copy_url(self) -> None:
        self.window.clipboard_clear()
        self.window.clipboard_append(self.url)
        self.window.update()
        self._set_note("访问地址已复制到剪贴板。")

    def open_repo_root(self) -> None:
        _open_local_path(self.repo_root)

    def open_run_root(self) -> None:
        run_root = self.repo_root / "local_app_runs"
        run_root.mkdir(parents=True, exist_ok=True)
        _open_local_path(run_root)

    def open_launcher_log(self) -> None:
        self.log_path.touch(exist_ok=True)
        _open_local_path(self.log_path)

    def stop_and_close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self._set_status("正在停止本地界面...", tone="warning")
            self._set_note("正在关闭本地服务，请稍候。")
            self.proc.terminate()
            try:
                self.proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.window.destroy()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Desktop launcher for the local ML Streamlit app")
    parser.add_argument("--selftest", action="store_true", help="Print resolved paths and exit")
    parser.add_argument("--selftest-json", default="", help="Write selftest payload to a JSON file and exit")
    args = parser.parse_args(argv)

    if args.selftest or str(args.selftest_json or "").strip():
        return run_selftest(str(args.selftest_json or "").strip())

    try:
        repo_root = _find_repo_root()
        python_executable = _resolve_python_executable(repo_root)
        _raise_if_launcher_dependencies_missing(python_executable)
        app = DesktopLauncherApp(repo_root=repo_root, python_executable=python_executable)
        app.start()
        return 0
    except Exception as exc:
        _show_error("ML Local App", str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
