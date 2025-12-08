#!/bin/bash

# Function to handle script exit
cleanup() {
    echo ""
    echo "Stopping servers..."
    kill $(jobs -p) 2>/dev/null
    
    echo "Cleaning up generated files..."
    # PriVAE cleanup
    rm -rf "PriVAE/DNA/data-for-sampling/samples-"* 2>/dev/null
    rm -f "PriVAE/DNA/data-for-sampling/processed-data-files/"* 2>/dev/null
    
    # VAE cleanup
    rm -rf "VAE-Ag-DNA-design (VAE)/data-for-sampling/past-samples-with-info/samples-"* 2>/dev/null
    rm -f "VAE-Ag-DNA-design (VAE)/data-for-sampling/processed-data-files/"* 2>/dev/null
    
    # Classifier cleanup
    rm -f "Ag-DNA-design (Classifier)/predictions.csv" 2>/dev/null
    
    echo "Cleanup complete."
    exit
}

# Trap Ctrl+C and kill background processes
trap cleanup SIGINT SIGTERM

echo "----------------------------------------"
echo "  DNA Design App - Quick Start Script"
echo "----------------------------------------"

# Load environment variables if they exist
if [ -f DNA-Design-Web/backend/.env ]; then
    export $(grep -v '^#' DNA-Design-Web/backend/.env | xargs)
fi
if [ -f DNA-Design-Web/.env ]; then
    export $(grep -v '^#' DNA-Design-Web/.env | xargs)
fi

# Set defaults if not set
API_PORT=${API_PORT:-8000}
FRONTEND_PORT=${PORT:-4321}
API_HOST=${API_HOST:-0.0.0.0}

# Detect Python
if command -v python3 &>/dev/null; then
    PYTHON_CMD=python3
elif command -v python &>/dev/null; then
    PYTHON_CMD=python
else
    echo "Error: Python not found. Please install Python 3."
    exit 1
fi

echo "Using Python: $PYTHON_CMD"

# Check Python version (Require 3.12+)
PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 12 ]); then
    echo "Error: Python 3.12 or higher is required. Found version $PYTHON_VERSION"
    exit 1
fi

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 12 ]); then
    echo "Error: Python 3.12 or higher is required. Found Python $PYTHON_VERSION"
    exit 1
fi

# Check for R
if ! command -v R &>/dev/null; then
    echo "Warning: R is not found. VAE/PriVAE sampling may fail."
    echo "Please install R (e.g., sudo apt install r-base)."
else
    echo "Checking R packages..."
    # Ensure local library directory exists
    mkdir -p ~/R/library
    export R_LIBS_USER=~/R/library
    
    # Install required packages if missing
    Rscript -e '
    lib_path <- Sys.getenv("R_LIBS_USER")
    if (!dir.exists(lib_path)) dir.create(lib_path, recursive = TRUE)
    .libPaths(c(lib_path, .libPaths()))
    
    required_packages <- c("tmvtnorm", "corpcor", "gmm", "sandwich", "mvtnorm")
    installed <- installed.packages()[,"Package"]
    missing <- required_packages[!(required_packages %in% installed)]
    
    if (length(missing) > 0) {
        message("Installing missing R packages: ", paste(missing, collapse=", "))
        install.packages(missing, repos="https://cloud.r-project.org", lib=lib_path)
    } else {
        message("All required R packages are installed.")
    }
    '
fi

# Setup Virtual Environment
VENV_DIR=".venv"
SYSTEM_PYTHON=$PYTHON_CMD

# Check if venv exists and is valid
if [ ! -f "$VENV_DIR/bin/python" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    rm -rf "$VENV_DIR" # Remove broken venv if exists
    
    # Try to create venv with pip
    if ! $SYSTEM_PYTHON -m venv $VENV_DIR; then
        echo "Error: Failed to create virtual environment."
        echo "Please install python3-venv (e.g., sudo apt install python3-venv) or python3-full."
        exit 1
    fi
fi

# Use the virtual environment's Python
PYTHON_CMD="$VENV_DIR/bin/python"

# Check if pip exists in venv, if not try to bootstrap it
if ! $PYTHON_CMD -m pip --version >/dev/null 2>&1; then
    echo "Pip not found in virtual environment. Attempting to bootstrap..."
    # Try ensurepip
    if ! $PYTHON_CMD -m ensurepip >/dev/null 2>&1; then
        # If ensurepip fails (common on Debian/Ubuntu without python3-venv), try downloading get-pip.py
        echo "ensurepip failed. Downloading get-pip.py..."
        curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
        $PYTHON_CMD get-pip.py
        rm get-pip.py
    fi
fi

# Verify pip is now available
if ! $PYTHON_CMD -m pip --version >/dev/null 2>&1; then
    echo "Error: Failed to install pip in virtual environment."
    echo "Please install python3-venv and python3-pip on your system:"
    echo "sudo apt install python3-venv python3-pip"
    exit 1
fi

# Check for backend dependencies
echo "[0/3] Checking dependencies..."
if ! $PYTHON_CMD -c "import uvicorn" &>/dev/null; then
    echo "Installing backend dependencies..."
    
    # Upgrade pip in venv just in case
    $PYTHON_CMD -m pip install --upgrade pip

    if ! $PYTHON_CMD -m pip install -r DNA-Design-Web/backend/requirements.txt; then
        echo "Error: Failed to install backend dependencies."
        exit 1
    fi
fi

# Check for frontend dependencies
if [ ! -d "DNA-Design-Web/node_modules" ]; then
    echo "Installing frontend dependencies..."
    cd DNA-Design-Web
    npm install
    cd ..
fi

# 1. Kill existing processes
echo "[1/3] Cleaning up existing processes..."
fuser -k $API_PORT/tcp >/dev/null 2>&1
fuser -k $FRONTEND_PORT/tcp >/dev/null 2>&1
sleep 1

# 2. Start Backend
echo "[2/3] Starting Backend Server (Port $API_PORT)..."
cd DNA-Design-Web/backend

# Verify Python executable exists before running
if [ ! -f "../../$PYTHON_CMD" ]; then
    echo "Error: Python executable not found at ../../$PYTHON_CMD"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Use fastapi dev for development
../../$VENV_DIR/bin/fastapi dev app/main.py --host $API_HOST --port $API_PORT &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 2

# 3. Start Frontend
echo "[3/3] Starting Frontend Server (Port $FRONTEND_PORT)..."
cd ..
npm run dev -- --port $FRONTEND_PORT &
FRONTEND_PID=$!

echo "----------------------------------------"
echo "  DNA Design System Initialized Successfully"
echo "----------------------------------------"
echo "  • Frontend Interface:  http://localhost:$FRONTEND_PORT"
echo "  • Backend API Server:  http://localhost:$API_PORT"
echo ""
echo "  System is ready for use."
echo "  Press Ctrl+C to gracefully shut down all services."
echo "----------------------------------------"

# Wait for processes
wait
