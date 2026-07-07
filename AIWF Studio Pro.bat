@echo off
setlocal EnableExtensions

rem AIWF Studio Pro - production-oriented React/FastAPI app.
rem Uses the saved backend profile by default: diffusers, sdcpp, or onnx.
rem The profile launcher also checks whether the Pro frontend needs a patch/build before launch.
set "AIWF_ROOT=%~dp0"
cd /d "%AIWF_ROOT%"

if not defined PYTHON set "PYTHON=python"
if not defined VENV_DIR set "VENV_DIR=%AIWF_ROOT%venv"
if exist "%VENV_DIR%\Scripts\python.exe" set "PYTHON=%VENV_DIR%\Scripts\python.exe"
if /I "%~1"=="--terminal" goto visible_terminal

set "PYTHONW=%PYTHON%"
if exist "%VENV_DIR%\Scripts\pythonw.exe" set "PYTHONW=%VENV_DIR%\Scripts\pythonw.exe"
start "AIWF Studio Pro" "%PYTHONW%" "%AIWF_ROOT%launch_backend_profile.py" %COMMANDLINE_ARGS% %*
endlocal & exit /b 0

:visible_terminal
"%PYTHON%" "%AIWF_ROOT%launch_backend_profile.py" %COMMANDLINE_ARGS% %*
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" pause
endlocal & exit /b %EXIT_CODE%
