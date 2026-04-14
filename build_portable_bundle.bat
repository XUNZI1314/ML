@echo off
setlocal

cd /d "%~dp0"

set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

echo Rebuilding desktop launcher exe first...
call build_desktop_app.bat
if errorlevel 1 exit /b 1

echo Building portable directory bundle...
"%PYTHON_EXE%" build_portable_bundle.py --repo_root . --bundle_dir portable_dist\ML_Portable --launcher_exe dist\ML_Local_App.exe
if errorlevel 1 (
    echo Portable bundle build failed.
    exit /b 1
)

echo.
echo Portable bundle ready: portable_dist\ML_Portable
echo Double-click portable_dist\ML_Portable\ML_Local_App.exe to start.
exit /b 0
