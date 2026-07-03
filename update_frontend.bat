@echo off
setlocal
title AIWF Studio - Update Frontend
echo ============================================
echo  AIWF Studio - Frontend update
echo ============================================
echo.

rem Run from the repo root no matter where this .bat was double-clicked from.
cd /d "%~dp0frontend"
if errorlevel 1 (
    echo [ERROR] Could not find the frontend folder next to this script.
    pause
    exit /b 1
)

where npm >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Node.js / npm was not found on PATH.
    echo Install Node.js LTS from https://nodejs.org and run this again.
    pause
    exit /b 1
)

echo [1/2] Installing dependencies (npm install)...
call npm install
if errorlevel 1 (
    echo.
    echo [ERROR] npm install failed. See the output above.
    pause
    exit /b 1
)

echo.
echo [2/2] Building the Pro UI (npm run build)...
call npm run build
if errorlevel 1 (
    echo.
    echo [ERROR] Frontend build failed. See the output above.
    pause
    exit /b 1
)

echo.
echo ============================================
echo  Frontend updated successfully.
echo  The new build is in frontend\dist and is
echo  served automatically the next time you
echo  start (or restart) AIWF Studio Pro.
echo ============================================
echo.
pause
endlocal
