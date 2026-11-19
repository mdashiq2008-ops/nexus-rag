@echo off
setlocal
cd /d "%~dp0"

start "NEXUS Backend" /D "%~dp0" cmd /k "python -m uvicorn api:app --host 127.0.0.1 --port 8000"
timeout /t 3 /nobreak >nul
start "NEXUS Frontend" /D "%~dp0frontend" cmd /k "npm.cmd run dev -- --host 127.0.0.1 --port 5173"
timeout /t 5 /nobreak >nul
start "" "http://127.0.0.1:5173/"
