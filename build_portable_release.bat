@echo off
setlocal

cd /d "%~dp0"

set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

echo Rebuilding portable bundle first...
call build_portable_bundle.bat
if errorlevel 1 exit /b 1

echo Building distributable zip release...
"%PYTHON_EXE%" build_portable_release.py --bundle_dir portable_dist\ML_Portable --zip_path portable_dist\ML_Portable_release.zip
if errorlevel 1 (
    echo Portable release zip build failed.
    exit /b 1
)

echo.
echo Portable release ready:
echo   portable_dist\ML_Portable_release.zip
echo   portable_dist\ML_Portable_release.manifest.json
exit /b 0
