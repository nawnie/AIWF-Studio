@echo off
setlocal EnableExtensions

set "AIWF_ROOT=%~dp0"
cd /d "%AIWF_ROOT%"

powershell -NoProfile -ExecutionPolicy Bypass -File "%AIWF_ROOT%scripts\install_aiwf_studio.ps1" %*
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" pause
endlocal & exit /b %EXIT_CODE%
