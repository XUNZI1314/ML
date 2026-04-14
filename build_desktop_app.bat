@echo off
setlocal

cd /d "%~dp0"

set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

echo [1/3] Checking PyInstaller...
"%PYTHON_EXE%" -m PyInstaller --version >nul 2>nul
if errorlevel 1 (
    echo PyInstaller not found. Installing into current environment...
    "%PYTHON_EXE%" -m pip install pyinstaller
    if errorlevel 1 (
        echo Failed to install PyInstaller.
        exit /b 1
    )
)

echo [2/3] Running launcher selftest...
"%PYTHON_EXE%" ml_desktop_launcher.py --selftest
if errorlevel 1 (
    echo Selftest failed.
    exit /b 1
)

echo [3/3] Building desktop executable...
"%PYTHON_EXE%" -m PyInstaller ^
    --noconfirm ^
    --clean ^
    --onefile ^
    --windowed ^
    --icon assets\app_icon.ico ^
    --name ML_Local_App ^
    ml_desktop_launcher.py

if errorlevel 1 (
    echo Build failed.
    exit /b 1
)

echo.
echo Build complete: dist\ML_Local_App.exe
echo You should keep the exe inside this repo tree so it can find local_ml_app.py and .venv.
"%PYTHON_EXE%" -c "from app_metadata import APP_NAME, APP_VERSION, APP_RELEASE_CHANNEL; print(f'{APP_NAME} version {APP_VERSION} ({APP_RELEASE_CHANNEL})')"
exit /b 0
