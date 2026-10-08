@echo off
setlocal EnableExtensions

rem AIWF Studio Pro - loopback-only profile (added 2026-10-06).
rem
rem Same app and launcher chain as "AIWF Studio Pro.bat", but the server binds
rem 127.0.0.1 only, even when launch.json has "listen": true. Nothing saved is
rem changed: launch.json, mobile pairing and auth settings stay as they are.
rem
rem Studio Flow (the unified workspace) reaches Dataset Studio, ReTrain and
rem Qwen Chat over loopback only. Start those apps with their own launchers;
rem Studio Flow shows each one as connected or not running.
rem
rem Optional overrides (defaults shown):
rem   AIWF_DATASET_STUDIO_URL      http://127.0.0.1:8796
rem   AIWF_RETRAIN_API_URL         http://127.12.6.3:8787
rem   AIWF_QWEN_CHAT_URL           http://127.0.0.1:8080
set "AIWF_PRO_LOOPBACK_ONLY=1"
call "%~dp0AIWF Studio Pro.bat" %*
endlocal & exit /b %ERRORLEVEL%
