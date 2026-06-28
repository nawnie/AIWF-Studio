@echo off
cd /d "%~dp0\.."

set "PY=engines\ltx\.venv\Scripts\python.exe"
set "WORKER=engines\ltx\worker.py"
set "REQ=scripts\ltx_smoketest_request.json"
set "LOG=scripts\ltx_smoketest.log"

if not exist "%PY%" (
    echo Could not find %PY% > "%LOG%"
    pause
    exit /b 1
)

mkdir "outputs\ltx-videos" 2>nul

echo ============================================================ > "%LOG%"
echo LTX 2.3 generation smoke test - %DATE% %TIME% >> "%LOG%"
echo ============================================================ >> "%LOG%"

"%PY%" "%WORKER%" "%REQ%" >> "%LOG%" 2>&1
echo. >> "%LOG%"
echo Finished with exit code %ERRORLEVEL% at %DATE% %TIME% >> "%LOG%"

echo Done. See %CD%\%LOG%
pause
