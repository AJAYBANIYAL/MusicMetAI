# ── Stage 1: Build the React frontend ────────────────────────────────────────
FROM node:22-alpine AS frontend-build

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

# ── Stage 2: Combined runtime image ──────────────────────────────────────────
# basic-pitch on Linux uses tflite-runtime for Python <3.11; 3.11+ pulls a broken TensorFlow pin.
FROM python:3.10-slim

# ── System dependencies ───────────────────────────────────────────────────────
# Install Nginx, ffmpeg, and libsndfile in one layer to keep the image lean.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        nginx \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python backend ────────────────────────────────────────────────────────────
WORKDIR /app/backend

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/main.py .

# ── Frontend (built static assets) ───────────────────────────────────────────
COPY --from=frontend-build /app/dist /usr/share/nginx/html

# ── Nginx configuration ───────────────────────────────────────────────────────
COPY nginx.conf /etc/nginx/conf.d/default.conf
# Remove the default Nginx site so our config is the only one active.
RUN rm -f /etc/nginx/sites-enabled/default

# ── Startup script ────────────────────────────────────────────────────────────
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Nginx (port 80) is the public-facing port; FastAPI listens internally on 8000.
EXPOSE 80

ENTRYPOINT ["/entrypoint.sh"]
