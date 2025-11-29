#!/bin/bash

# Backend Startup Script
# This script starts the FastAPI server with uvicorn

echo "🚀 Starting Sports Truth Tracker Backend Server..."
echo "=================================================="

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Virtual environment not activated!"
    echo "Please run: source ../venv/bin/activate"
    exit 1
fi

# Check if .env file exists
if [ ! -f "../.env" ]; then
    echo "❌ .env file not found in parent directory!"
    echo "Please create .env file with required API keys"
    exit 1
fi

echo "✅ Virtual environment: $VIRTUAL_ENV"
echo "✅ Environment file: ../.env"
echo ""
echo "Starting server on http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo "=================================================="
echo ""

# Start the server with uvicorn
# --reload: Auto-reload on code changes (development mode)
# --host 0.0.0.0: Listen on all network interfaces
# --port 8000: Port number
python -m uvicorn server:app --reload --host 0.0.0.0 --port 8000
