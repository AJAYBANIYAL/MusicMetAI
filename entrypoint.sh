#!/bin/sh
set -e

# Start the FastAPI backend in the background.
# Uvicorn binds to 127.0.0.1:8000 — only reachable by Nginx inside the container.
cd /app/backend
uvicorn main:app --host 127.0.0.1 --port 8000 &

# Wait for the backend to be ready before starting Nginx.
# TensorFlow and other heavy ML imports can take well over 1 second to load,
# so we poll /health instead of using a fixed sleep.
MAX_ATTEMPTS=30
attempt=0
echo "Waiting for uvicorn to be ready..."
until curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1; do
    attempt=$((attempt + 1))
    if [ "$attempt" -ge "$MAX_ATTEMPTS" ]; then
        echo "ERROR: Backend did not become ready after ${MAX_ATTEMPTS} seconds. Aborting." >&2
        exit 1
    fi
    sleep 1
done
echo "Backend is ready (attempt ${attempt}/${MAX_ATTEMPTS}). Starting Nginx."

# Start Nginx in the foreground so it becomes PID 1's watched process.
# If Nginx exits, the container exits.
exec nginx -g "daemon off;"
