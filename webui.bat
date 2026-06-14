@echo off
setlocal EnableExtensions

rem AIWF Studio bootstrap — prepares the venv, then starts the Gradio UI (python launch.py -> aiwf.app).
set "AIWF_ROOT=%~dp0"
cd /d "%AIWF_ROOT%"

if not defined PYTHON set "PYTHON=python"
if not defined VENV_DIR set "VENV_DIR=%AIWF_ROOT%venv"
if exist "%VENV_DIR%\Scripts\python.exe" set "PYTHON=%VENV_DIR%\Scripts\python.exe"

rem Optional one-off flags for this session only. Saved GPU/network/theme options live in launch.json
rem (Settings -> Launch profile) and are applied automatically on every start.
rem Recommended for RTX 4070 Ti SUPER Wan I2V (also the app defaults when unset):
rem   set "COMMANDLINE_ARGS=--opt-sdp-attention"
if not defined COMMANDLINE_ARGS set "COMMANDLINE_ARGS="

"%PYTHON%" "%AIWF_ROOT%launch.py" %COMMANDLINE_ARGS% %*
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" pause
endlocal & exit /b %EXIT_CODE%