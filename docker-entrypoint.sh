#!/bin/bash
set -e

echo "Starting FastAPI (port 8000)..."
cd /app/src/api
uvicorn api:app --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!

# Wait for FastAPI to be ready
for i in $(seq 1 30); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "FastAPI ready."
        break
    fi
    sleep 1
done

echo "Starting Flask (port 5000)..."
cd /app
python src/api/app.py &
FLASK_PID=$!

echo "Both services running."
echo "  Flask  -> http://localhost:5000"
echo "  FastAPI -> http://localhost:8000"

# Wait for either process to exit
wait -n $FASTAPI_PID $FLASK_PID
exit $?
