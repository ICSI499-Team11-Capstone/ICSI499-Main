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

# 1. Kill existing processes
echo "[1/3] Cleaning up existing processes..."
fuser -k $API_PORT/tcp >/dev/null 2>&1
fuser -k $FRONTEND_PORT/tcp >/dev/null 2>&1
sleep 1

# 2. Start Backend
echo "[2/3] Starting Backend Server (Port $API_PORT)..."
cd DNA-Design-Web/backend
python3.14 -m uvicorn app.main:app --reload --host $API_HOST --port $API_PORT &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 2

# 3. Start Frontend
echo "[3/3] Starting Frontend Server (Port $FRONTEND_PORT)..."
cd ..
npm run dev -- --port $FRONTEND_PORT &
FRONTEND_PID=$!

echo "----------------------------------------"
echo "  App is running!"
echo "  Frontend: http://localhost:$FRONTEND_PORT"
echo "  Backend:  http://localhost:$API_PORT"
echo "  Press Ctrl+C to stop both servers."
echo "----------------------------------------"

# Wait for processes
wait
