@echo off
cd /d "%~dp0"
setlocal

set "PYTHONW_EXE=%CD%\.venv\Scripts\pythonw.exe"
set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"

if exist "%PYTHONW_EXE%" goto launch_launcher_pythonw
if exist "%PYTHON_EXE%" goto launch_launcher_python

where pythonw >nul 2>nul
if not errorlevel 1 (
    set "PYTHONW_EXE=pythonw"
    goto launch_launcher_pythonw
)

where python >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_EXE=python"
    goto launch_launcher_python
)

echo Could not find a usable Python runtime.
echo Please create .venv first or install Python.
pause
exit /b 1

:launch_launcher_pythonw
start "" "%PYTHONW_EXE%" ml_desktop_launcher.py
exit /b 0

:launch_launcher_python
start "" "%PYTHON_EXE%" ml_desktop_launcher.py
exit /b 0
