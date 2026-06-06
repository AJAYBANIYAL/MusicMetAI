#!/bin/sh
set -e

# Start the FastAPI backend in the background.
# Uvicorn binds to 127.0.0.1:8000 — only reachable by Nginx inside the container.
cd /app/backend
uvicorn main:app --host 127.0.0.1 --port 8000 &

# Give uvicorn a moment to start before Nginx begins proxying.
sleep 1

# Start Nginx in the foreground so it becomes PID 1's watched process.
# If Nginx exits, the container exits.
exec nginx -g "daemon off;"
