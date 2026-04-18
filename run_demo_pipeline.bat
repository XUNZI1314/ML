@echo off
setlocal
cd /d "%~dp0"

echo Running ML demo pipeline...
python run_demo_pipeline.py --out_dir demo_outputs --demo_dir demo_data

echo.
echo Demo run finished. Open demo_outputs\DEMO_README.md for the output guide.
pause
