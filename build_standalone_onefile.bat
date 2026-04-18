@echo off
setlocal

cd /d "%~dp0"

set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

echo [1/5] Checking PyInstaller...
"%PYTHON_EXE%" -m PyInstaller --version >nul 2>nul
if errorlevel 1 (
    echo PyInstaller not found. Installing into current environment...
    "%PYTHON_EXE%" -m pip install pyinstaller
    if errorlevel 1 (
        echo Failed to install PyInstaller.
        exit /b 1
    )
)

echo [2/5] Running launcher selftest...
"%PYTHON_EXE%" ml_desktop_launcher.py --selftest
if errorlevel 1 (
    echo Selftest failed.
    exit /b 1
)

echo [3/5] Building standalone onefile bundle...
"%PYTHON_EXE%" build_standalone_onefile.py --repo_root . --dist_dir portable_dist\standalone_onefile --work_dir build\standalone_onefile_build --output_name ML_Local_App_Standalone
if errorlevel 1 (
    echo Standalone onefile build failed.
    exit /b 1
)

echo [4/5] Validating standalone onefile bundle...
"%PYTHON_EXE%" validate_standalone_onefile.py --exe_path portable_dist\standalone_onefile\ML_Local_App_Standalone.exe --host_repo_root . --report_dir portable_dist\standalone_onefile_validation
if errorlevel 1 (
    echo Standalone onefile validation failed.
    exit /b 1
)

echo [5/5] Standalone output ready.
echo   portable_dist\standalone_onefile\ML_Local_App_Standalone.exe
echo   portable_dist\standalone_onefile\ML_Local_App_Standalone.manifest.json
echo   portable_dist\standalone_onefile_validation\standalone_validation_latest.json
exit /b 0
