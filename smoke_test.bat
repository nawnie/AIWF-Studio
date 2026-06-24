@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

set "PY=venv\Scripts\python.exe"
set "SCRIPT=scripts\smoke_backend.py"
set "LOG=smoke_test.log"

if not exist "%PY%" (
    echo Could not find %PY% - run this from the AIWF_Studio folder with the venv set up.
    pause
    exit /b 1
)

echo ============================================================ > "%LOG%"
echo AIWF Studio backend smoke test - %DATE% %TIME% >> "%LOG%"
echo ============================================================ >> "%LOG%"

echo Running smoke test. Each engine/checkpoint runs as its own process,
echo so a hard crash in one does NOT stop the rest. Logging to %LOG%.
echo.

rem ---------------------------------------------------------------
rem IMAGE LANE - one isolated process per selected checkpoint
rem ---------------------------------------------------------------
echo. >> "%LOG%"
echo ---- IMAGE LANE ---- >> "%LOG%"
echo [IMAGE] enumerating checkpoints...

set "IMG_COUNT=0"
set "IMG_FAIL=0"
for /f "usebackq delims=" %%I in (`"%PY%" "%SCRIPT%" --enumerate-checkpoints 2^>^>"%LOG%"`) do (
    set /a IMG_COUNT+=1
    echo [IMAGE] testing checkpoint: %%I
    echo. >> "%LOG%"
    echo --- checkpoint %%I --- >> "%LOG%"
    "%PY%" "%SCRIPT%" --checkpoint "%%I" >> "%LOG%" 2>&1
    if errorlevel 1 (
        set /a IMG_FAIL+=1
        echo [IMAGE]   FAIL - see %LOG%
    ) else (
        echo [IMAGE]   PASS
    )
)
echo [IMAGE] done: !IMG_COUNT! checkpoint(s), !IMG_FAIL! failure(s)
echo.

rem ---------------------------------------------------------------
rem VIDEO LANE - one isolated process per Wan VAE, then one for the
rem real generation pass
rem ---------------------------------------------------------------
echo. >> "%LOG%"
echo ---- VIDEO LANE (Wan VAE checks) ---- >> "%LOG%"
echo [VIDEO] enumerating Wan VAE files...

set "VAE_COUNT=0"
set "VAE_FAIL=0"
for /f "usebackq delims=" %%V in (`"%PY%" "%SCRIPT%" --enumerate-vae 2^>^>"%LOG%"`) do (
    set /a VAE_COUNT+=1
    echo [VIDEO] testing VAE: %%V
    echo. >> "%LOG%"
    echo --- vae %%V --- >> "%LOG%"
    "%PY%" "%SCRIPT%" --vae "%%V" >> "%LOG%" 2>&1
    if errorlevel 1 (
        set /a VAE_FAIL+=1
        echo [VIDEO]   FAIL - see %LOG%
    ) else (
        echo [VIDEO]   PASS
    )
)
echo [VIDEO] done: !VAE_COUNT! VAE file(s), !VAE_FAIL! failure(s)
echo.

echo ---- VIDEO LANE (real generation pass) ---- >> "%LOG%"
echo [VIDEO] running Wan generation pass (Q4/Q5 only)...
"%PY%" "%SCRIPT%" --video-gen >> "%LOG%" 2>&1
set "GEN_RC=%ERRORLEVEL%"
if "%GEN_RC%"=="0" (
    echo [VIDEO]   PASS or skipped - see %LOG%
) else (
    echo [VIDEO]   FAIL - see %LOG%
)

echo.
echo ============================================================
echo Smoke test finished. Full log: %CD%\%LOG%
echo   Image: !IMG_COUNT! checkpoint(s), !IMG_FAIL! failure(s)
echo   Video VAE: !VAE_COUNT! file(s), !VAE_FAIL! failure(s)
echo   Video generation exit code: %GEN_RC%
echo ============================================================
pause
