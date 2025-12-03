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

:: 0. Check Dependencies
echo [0/3] Checking dependencies...

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python not found. Please install Python.
    pause
    exit /b 1
)

:: Check R
R --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Warning: R is not found in PATH. VAE/PriVAE sampling may fail.
    echo Please install R and add it to your PATH.
)

:: Setup Virtual Environment
set VENV_DIR=.venv
if not exist "%VENV_DIR%" (
    echo Creating virtual environment in %VENV_DIR%...
    python -m venv %VENV_DIR%
    if %errorlevel% neq 0 (
        echo Error: Failed to create virtual environment.
        pause
        exit /b 1
    )
)

set PYTHON_CMD=%VENV_DIR%\Scripts\python.exe
set PIP_CMD=%VENV_DIR%\Scripts\pip.exe

:: Check Backend Dependencies
"%PYTHON_CMD%" -c "import uvicorn" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing backend dependencies...
    "%PIP_CMD%" install -r DNA-Design-Web\backend\requirements.txt
    if %errorlevel% neq 0 (
        echo Error: Failed to install backend dependencies.
        pause
        exit /b 1
    )
)

:: Check Node modules
if not exist "DNA-Design-Web\node_modules" (
    echo Installing frontend dependencies...
    cd DNA-Design-Web
    call npm install
    cd ..
)

:: 1. Kill existing processes (optional, might fail if not running)
echo [1/3] Cleaning up existing processes...
taskkill /F /IM python.exe /T >nul 2>&1
taskkill /F /IM node.exe /T >nul 2>&1

:: 2. Start Backend
echo [2/3] Starting Backend Server (Port %API_PORT%)...
cd DNA-Design-Web\backend
start "DNA Backend" ..\..\%PYTHON_CMD% -m uvicorn app.main:app --reload --host %API_HOST% --port %API_PORT%

:: Wait a moment
timeout /t 3 /nobreak >nul

:: 3. Start Frontend
echo [3/3] Starting Frontend Server (Port %FRONTEND_PORT%)...
cd ..
start "DNA Frontend" npm run dev -- --port %FRONTEND_PORT%

echo ----------------------------------------
echo   DNA Design System Initialized Successfully
echo ----------------------------------------
echo   * Frontend Interface:  http://localhost:%FRONTEND_PORT%
echo   * Backend API Server:  http://localhost:%API_PORT%
echo.
echo   System is ready for use.
echo   Close the opened command windows to shut down services.
echo ----------------------------------------
pause
