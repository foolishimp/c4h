#!/bin/bash
# Script to test server/client mode in compatibility mode

set -e

# Define variables
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
PYTHONPATH=$REPO_ROOT
PORT=5555
SERVER_CONFIG="$REPO_ROOT/config/compat_system_config.yml"
CLIENT_CONFIG="$REPO_ROOT/tests/examples/config/jobs_coder_01.yml"
LOG_FILE="$REPO_ROOT/server_client_test.log"

# Load the API keys from .env file
if [ -f "$REPO_ROOT/.env" ]; then
    echo "Loading environment variables from $REPO_ROOT/.env" | tee -a $LOG_FILE
    set -a # automatically export all variables
    source "$REPO_ROOT/.env"
    set +a
else
    echo "WARNING: .env file not found at $REPO_ROOT/.env" | tee -a $LOG_FILE
fi

# Use the python from .pyenv where uvicorn is installed
PYTHON_CMD=/Users/jim/.pyenv/versions/3.11.5/bin/python

# Print environment setup
echo "===========================================" > $LOG_FILE
echo "ENVIRONMENT SETUP" >> $LOG_FILE
echo "REPO_ROOT: $REPO_ROOT" >> $LOG_FILE
echo "PYTHONPATH: $PYTHONPATH" >> $LOG_FILE
echo "PYTHON_CMD: $PYTHON_CMD" >> $LOG_FILE
echo "SERVER_CONFIG: $SERVER_CONFIG" >> $LOG_FILE
echo "CLIENT_CONFIG: $CLIENT_CONFIG" >> $LOG_FILE
echo "PORT: $PORT" >> $LOG_FILE
echo "API_KEY: ${ANTHROPIC_API_KEY:0:10}..." >> $LOG_FILE
echo "===========================================" >> $LOG_FILE
echo "" >> $LOG_FILE

# Print test information
echo "Starting server/client compatibility test..." | tee -a $LOG_FILE
echo "Using compatibility configuration: $SERVER_CONFIG" | tee -a $LOG_FILE
echo "Using client job configuration: $CLIENT_CONFIG" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Check if test project setup exists
if [ ! -d "$REPO_ROOT/tests/test_projects" ]; then
    echo "Setting up test projects..." | tee -a $LOG_FILE
    $REPO_ROOT/tests/setup/setup_test_projects.sh
    echo "Test projects set up successfully." | tee -a $LOG_FILE
    echo "" | tee -a $LOG_FILE
fi

# Function to cleanup on exit
cleanup() {
    echo "Cleaning up..." | tee -a $LOG_FILE
    # Kill server process if running
    if [ -n "$SERVER_PID" ]; then
        echo "Terminating server process (PID: $SERVER_PID)..." | tee -a $LOG_FILE
        kill $SERVER_PID 2>/dev/null || true
    fi
    echo "Cleanup complete." | tee -a $LOG_FILE
}

# Register cleanup function
trap cleanup EXIT

# Start server in background
echo "Starting server..." | tee -a $LOG_FILE
PYTHONPATH=$PYTHONPATH $PYTHON_CMD $REPO_ROOT/c4h_services/src/bootstrap/prefect_runner.py service -P $PORT --config $SERVER_CONFIG >> $LOG_FILE 2>&1 &
SERVER_PID=$!
echo "Server started with PID: $SERVER_PID" | tee -a $LOG_FILE

# Wait for server to initialize
echo "Waiting for server to initialize (5 seconds)..." | tee -a $LOG_FILE
sleep 5

# Check if server is running
if ps -p $SERVER_PID > /dev/null; then
    echo "Server is running." | tee -a $LOG_FILE
else
    echo "ERROR: Server failed to start. Check $LOG_FILE for details." | tee -a $LOG_FILE
    exit 1
fi

echo "" | tee -a $LOG_FILE
echo "Running client job..." | tee -a $LOG_FILE

# Run client job with reduced polling
echo "Running client job with reduced polling..." | tee -a $LOG_FILE
PYTHONPATH=$PYTHONPATH $PYTHON_CMD $REPO_ROOT/c4h_services/src/bootstrap/prefect_runner.py jobs -P $PORT --config $CLIENT_CONFIG --poll --poll-interval 10 --max-polls 12 | tee -a $LOG_FILE

# Capture client exit code
CLIENT_EXIT_CODE=${PIPESTATUS[0]}

echo "" | tee -a $LOG_FILE
if [ $CLIENT_EXIT_CODE -eq 0 ]; then
    echo "✅ Client job completed successfully!" | tee -a $LOG_FILE
    echo "Test PASSED" | tee -a $LOG_FILE
else
    echo "❌ Client job failed with exit code: $CLIENT_EXIT_CODE" | tee -a $LOG_FILE
    echo "Test FAILED" | tee -a $LOG_FILE
fi

echo "" | tee -a $LOG_FILE
echo "Test log saved to: $LOG_FILE" | tee -a $LOG_FILE

exit $CLIENT_EXIT_CODE