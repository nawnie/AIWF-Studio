@echo off
setlocal EnableExtensions

rem AIWF Studio Gradio Lab - full Gradio workspace for WIP features.
set "AIWF_ROOT=%~dp0"
cd /d "%AIWF_ROOT%"

if not defined PYTHON set "PYTHON=python"
if not defined VENV_DIR set "VENV_DIR=%AIWF_ROOT%venv"
if exist "%VENV_DIR%\Scripts\python.exe" set "PYTHON=%VENV_DIR%\Scripts\python.exe"
if not defined COMMANDLINE_ARGS set "COMMANDLINE_ARGS=--autolaunch"
if /I "%~1"=="--terminal" goto visible_terminal

set "PYTHONW=%PYTHON%"
if exist "%VENV_DIR%\Scripts\pythonw.exe" set "PYTHONW=%VENV_DIR%\Scripts\pythonw.exe"
start "AIWF Studio Gradio Lab" "%PYTHONW%" "%AIWF_ROOT%launch_gradio.py" %COMMANDLINE_ARGS% %*
endlocal & exit /b 0

:visible_terminal
"%PYTHON%" "%AIWF_ROOT%launch_gradio.py" %COMMANDLINE_ARGS% %*
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" pause
endlocal & exit /b %EXIT_CODE%
