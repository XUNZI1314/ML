from __future__ import annotations

import argparse
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
from tkinter import BOTH, LEFT, RIGHT, Button, Label, StringVar, Tk, messagebox

from app_metadata import APP_NAME, APP_RELEASE_CHANNEL, APP_VERSION


def _candidate_start_dirs() -> list[Path]:
    candidates: list[Path] = []

    exe_dir = Path(sys.executable).resolve().parent
    script_dir = Path(__file__).resolve().parent
    cwd_dir = Path.cwd().resolve()

    for path in [exe_dir, script_dir, cwd_dir]:
        if path not in candidates:
            candidates.append(path)
        portable_app_dir = path / "app"
        if portable_app_dir not in candidates:
            candidates.append(portable_app_dir)
    return candidates


def _find_repo_root() -> Path:
    markers = ["local_ml_app.py", "run_recommended_pipeline.py", "requirements.txt"]
    exe_dir = Path(sys.executable).resolve().parent
    portable_app_dir = exe_dir / "app"
    if all((portable_app_dir / marker).exists() for marker in markers):
        return portable_app_dir

    for start in _candidate_start_dirs():
        current = start
        for candidate in [current] + list(current.parents):
            if all((candidate / marker).exists() for marker in markers):
                return candidate
    raise FileNotFoundError("Could not locate repo root containing local_ml_app.py and requirements.txt.")


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


def run_selftest() -> int:
    repo_root = _find_repo_root()
    python_executable = _resolve_python_executable(repo_root)
    app_script = repo_root / "local_ml_app.py"
    icon_path = _resolve_icon_path(repo_root)
    candidates = [str(path) for path in _candidate_start_dirs()]
    print(f"app_name={APP_NAME}")
    print(f"app_version={APP_VERSION}")
    print(f"release_channel={APP_RELEASE_CHANNEL}")
    print(f"repo_root={repo_root}")
    print(f"python_executable={python_executable}")
    print(f"app_script_exists={app_script.exists()}")
    print(f"icon_path={icon_path}")
    print(f"streamlit_batch={repo_root / 'start_local_app.bat'}")
    print("candidate_dirs=" + " | ".join(candidates))
    return 0


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
        self.status_var = StringVar(value="正在启动本地界面...")
        self.url_var = StringVar(value=self.url)
        self.window = Tk()
        self.window.title(f"{APP_NAME} v{APP_VERSION}")
        self.window.geometry("580x250")
        self.window.resizable(False, False)
        self.window.protocol("WM_DELETE_WINDOW", self.stop_and_close)
        self._apply_window_icon()
        self._build_ui()

    def _apply_window_icon(self) -> None:
        icon_path = _resolve_icon_path(self.repo_root)
        if icon_path is None:
            return
        try:
            self.window.iconbitmap(default=str(icon_path))
        except Exception:
            pass

    def _build_ui(self) -> None:
        self.window.configure(padx=18, pady=18)

        Label(self.window, text=f"{APP_NAME} 桌面启动器", font=("Microsoft YaHei UI", 14, "bold")).pack(anchor="w")
        Label(
            self.window,
            text=f"版本 {APP_VERSION} | 通道 {APP_RELEASE_CHANNEL}",
            justify=LEFT,
        ).pack(anchor="w", pady=(6, 2))
        Label(self.window, text=f"项目目录: {self.repo_root}", wraplength=500, justify=LEFT).pack(anchor="w", pady=(10, 2))
        Label(self.window, textvariable=self.status_var, wraplength=500, justify=LEFT).pack(anchor="w", pady=(4, 2))
        Label(self.window, textvariable=self.url_var, fg="#0b57d0", wraplength=500, justify=LEFT).pack(anchor="w", pady=(2, 12))

        top_row = Button(self.window, text="打开浏览器", command=self.open_browser, width=16)
        top_row.pack(side=LEFT, padx=(0, 8))
        Button(self.window, text="打开项目目录", command=self.open_repo_root, width=16).pack(side=LEFT, padx=(0, 8))
        Button(self.window, text="打开运行产物目录", command=self.open_run_root, width=18).pack(side=LEFT)

        bottom_bar = Label(
            self.window,
            text="关闭此窗口会同时停止本地 Streamlit 服务。",
            wraplength=500,
            justify=LEFT,
        )
        bottom_bar.pack(anchor="w", pady=(22, 10), fill=BOTH)

        Button(self.window, text="停止并退出", command=self.stop_and_close, width=18).pack(side=RIGHT)

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
            self.window.after(0, lambda: self.status_var.set("本地界面已启动，可以在浏览器中使用。"))
            self.window.after(0, self.open_browser)
        else:
            self.window.after(0, lambda: self.status_var.set("本地界面启动超时，请查看运行日志。"))

    def _poll_process(self) -> None:
        if self.proc is None:
            return
        rc = self.proc.poll()
        if rc is None:
            self.window.after(1000, self._poll_process)
            return
        self.status_var.set(f"本地界面已退出，返回码: {rc}")

    def open_browser(self) -> None:
        webbrowser.open(self.url, new=1)

    def open_repo_root(self) -> None:
        _open_local_path(self.repo_root)

    def open_run_root(self) -> None:
        run_root = self.repo_root / "local_app_runs"
        run_root.mkdir(parents=True, exist_ok=True)
        _open_local_path(run_root)

    def stop_and_close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.status_var.set("正在停止本地界面...")
            self.proc.terminate()
            try:
                self.proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.window.destroy()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Desktop launcher for the local ML Streamlit app")
    parser.add_argument("--selftest", action="store_true", help="Print resolved paths and exit")
    args = parser.parse_args(argv)

    if args.selftest:
        return run_selftest()

    try:
        repo_root = _find_repo_root()
        python_executable = _resolve_python_executable(repo_root)
        app = DesktopLauncherApp(repo_root=repo_root, python_executable=python_executable)
        app.start()
        return 0
    except Exception as exc:
        _show_error("ML Local App", str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
