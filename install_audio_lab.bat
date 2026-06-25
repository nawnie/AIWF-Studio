@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if not defined PYTHON set "PYTHON=python"
if exist "%~dp0venv\Scripts\python.exe" set "PYTHON=%~dp0venv\Scripts\python.exe"
"%PYTHON%" "%~dp0scripts\bootstrap_audio_lab.py" --repo "%~dp0"
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" (echo Audio Lab engine is ready.) else (echo Audio Lab installation failed with exit code %RC%.)
pause
endlocal & exit /b %RC%
