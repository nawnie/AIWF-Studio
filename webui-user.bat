@echo off
rem Double-click this file to start AIWF Studio (Gradio web UI at http://localhost:7860 by default).
rem
rem Customize below if needed:
rem   VENV_DIR  — path to the project virtualenv (default: .\venv)
rem   COMMANDLINE_ARGS — extra flags for this session only (e.g. --listen --port 8188)
rem
rem Persistent launch options (GPU mode, port, API, theme, …) are stored in launch.json.
rem Edit them in the app: Settings -> Launch profile -> Save launch options, then restart.

set "VENV_DIR=%~dp0venv"
set "COMMANDLINE_ARGS="

call "%~dp0webui.bat" %*