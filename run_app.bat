@echo off
setlocal

:: Default Configuration
set API_PORT=8000
set FRONTEND_PORT=4321
set API_HOST=0.0.0.0

:: Load Backend .env if exists
if exist "DNA-Design-Web\backend\.env" (
    for /f "usebackq tokens=1* delims==" %%a in ("DNA-Design-Web\backend\.env") do (
        if "%%a"=="API_PORT" set API_PORT=%%b
        if "%%a"=="API_HOST" set API_HOST=%%b
    )
)

:: Load Frontend .env if exists
if exist "DNA-Design-Web\.env" (
    for /f "usebackq tokens=1* delims==" %%a in ("DNA-Design-Web\.env") do (
        if "%%a"=="PORT" set FRONTEND_PORT=%%b
    )
)

echo ----------------------------------------
echo   DNA Design App - Quick Start Script
echo ----------------------------------------

:: 1. Kill existing processes (optional, might fail if not running)
echo [1/3] Cleaning up existing processes...
taskkill /F /IM python.exe /T >nul 2>&1
taskkill /F /IM node.exe /T >nul 2>&1

:: 2. Start Backend
echo [2/3] Starting Backend Server (Port %API_PORT%)...
cd DNA-Design-Web\backend
start "DNA Backend" python -m uvicorn app.main:app --reload --host %API_HOST% --port %API_PORT%

:: Wait a moment
timeout /t 3 /nobreak >nul

:: 3. Start Frontend
echo [3/3] Starting Frontend Server (Port %FRONTEND_PORT%)...
cd ..
start "DNA Frontend" npm run dev -- --port %FRONTEND_PORT%

echo ----------------------------------------
echo   App is running!
echo   Frontend: http://localhost:%FRONTEND_PORT%
echo   Backend:  http://localhost:%API_PORT%
echo   Close the opened command windows to stop.
echo ----------------------------------------
pause
