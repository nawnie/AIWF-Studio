@echo off
setlocal
cd /d "%~dp0"

if exist "venv\Scripts\python.exe" (
    set "AIWF_PYTHON=venv\Scripts\python.exe"
) else (
    set "AIWF_PYTHON=python"
)

echo Starting AIWF Studio Second GUI preview...
echo.
"%AIWF_PYTHON%" -m aiwf.second_gui %*

if errorlevel 1 (
    echo.
    echo Second GUI exited with an error.
    pause
)
