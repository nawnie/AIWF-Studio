@echo off
setlocal EnableExtensions

rem AIWF Studio Pro - production-oriented React/FastAPI app.
set "AIWF_ROOT=%~dp0"
cd /d "%AIWF_ROOT%"

if not defined PYTHON set "PYTHON=python"
if not defined VENV_DIR set "VENV_DIR=%AIWF_ROOT%venv"
if exist "%VENV_DIR%\Scripts\python.exe" set "PYTHON=%VENV_DIR%\Scripts\python.exe"
if not defined COMMANDLINE_ARGS set "COMMANDLINE_ARGS=--autolaunch"

"%PYTHON%" "%AIWF_ROOT%launch_pro.py" %COMMANDLINE_ARGS% %*
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" pause
endlocal & exit /b %EXIT_CODE%
