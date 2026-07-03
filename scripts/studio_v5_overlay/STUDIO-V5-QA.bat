@echo off
setlocal EnableExtensions
rem This script lives in scripts\studio_v5_overlay\; the repo root is two levels up.
set "AIWF_ROOT=%~dp0..\..\"
cd /d "%AIWF_ROOT%"
if not defined PYTHON set "PYTHON=python"
if not defined VENV_DIR set "VENV_DIR=%AIWF_ROOT%venv"
if exist "%VENV_DIR%\Scripts\python.exe" set "PYTHON=%VENV_DIR%\Scripts\python.exe"
set "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1"

"%PYTHON%" -m pytest -q ^
  tests\individual_tests\test_rife.py ^
  tests\individual_tests\test_video_lab.py ^
  tests\individual_tests\test_studio_v4_update.py ^
  tests\individual_tests\test_studio_v5_labs.py ^
  tests\individual_tests\test_studio_v5_ui_contracts.py ^
  tests\individual_tests\test_image_workflow_service.py ^
  tests\individual_tests\test_audio_lab_runner.py ^
  tests\individual_tests\test_engine_supervisor.py ^
  tests\individual_tests\test_wan_sampler_policy.py
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if "%EXIT_CODE%"=="0" (
  echo Studio v5 targeted QA passed.
) else (
  echo Studio v5 targeted QA failed with exit code %EXIT_CODE%.
)
pause
endlocal & exit /b %EXIT_CODE%
