#!/bin/bash

# Start and Test Caching Script
# This script starts the backend server and runs the caching tests

echo "=================================="
echo "Sports Truth Tracker - Cache Test"
echo "=================================="
echo ""

# Check if we're in the right directory
if [ ! -f "server.py" ]; then
    echo "❌ Error: server.py not found"
    echo "Please run this script from the new/backend directory:"
    echo "  cd new/backend"
    echo "  ./start_and_test.sh"
    exit 1
fi

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Error: Python not found"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

# Check if required packages are installed
echo "🔍 Checking dependencies..."
python -c "import fastapi, uvicorn, requests" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Error: Required packages not installed"
    echo "Please install dependencies:"
    echo "  pip install -r requirements.txt"
    exit 1
fi
echo "✅ Dependencies OK"
echo ""

# Start the server in the background
echo "🚀 Starting backend server..."
python server.py > server_test.log 2>&1 &
SERVER_PID=$!

# Wait for server to start
echo "⏳ Waiting for server to start..."
sleep 5

# Check if server is running
if ! ps -p $SERVER_PID > /dev/null; then
    echo "❌ Error: Server failed to start"
    echo "Check server_test.log for details:"
    echo ""
    tail -20 server_test.log
    exit 1
fi

# Check if server is responding
for i in {1..10}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ Server is running (PID: $SERVER_PID)"
        break
    fi
    if [ $i -eq 10 ]; then
        echo "❌ Error: Server not responding"
        kill $SERVER_PID 2>/dev/null
        exit 1
    fi
    sleep 1
done

echo ""
echo "=================================="
echo "Running Cache Tests"
echo "=================================="
echo ""

# Run the tests
python test_caching.py
TEST_RESULT=$?

echo ""
echo "=================================="
echo "Cleanup"
echo "=================================="
echo ""

# Stop the server
echo "🛑 Stopping server (PID: $SERVER_PID)..."
kill $SERVER_PID 2>/dev/null
sleep 2

# Force kill if still running
if ps -p $SERVER_PID > /dev/null 2>&1; then
    echo "⚠️  Force stopping server..."
    kill -9 $SERVER_PID 2>/dev/null
fi

echo "✅ Server stopped"
echo ""

# Show result
if [ $TEST_RESULT -eq 0 ]; then
    echo "=================================="
    echo "🎉 ALL TESTS PASSED!"
    echo "=================================="
    echo ""
    echo "Caching is working correctly."
    echo "You can now start the server normally:"
    echo "  python server.py"
else
    echo "=================================="
    echo "⚠️  SOME TESTS FAILED"
    echo "=================================="
    echo ""
    echo "Check the output above for details."
    echo "Server logs are in: server_test.log"
fi

echo ""

exit $TEST_RESULT
