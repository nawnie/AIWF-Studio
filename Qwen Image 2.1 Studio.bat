@echo off
setlocal EnableExtensions

rem Qwen Image 2.1 Studio - PySide6 desktop app on top of a running ComfyUI (127.0.0.1:8188).
rem Generate / edit with up to 10 references / RGBA / LoRA stack / train LoRAs (ai-toolkit).
rem Bootstrap once: powershell -ExecutionPolicy Bypass -File scripts\bootstrap_qwen21.ps1
set "AIWF_ROOT=%~dp0"
cd /d "%AIWF_ROOT%"

set "PYTHON=%AIWF_ROOT%engines\qwen_image_2_1\.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=%AIWF_ROOT%venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
"%PYTHON%" "%AIWF_ROOT%engines\qwen_image_2_1\app.py" %*
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" pause
endlocal & exit /b %EXIT_CODE%
