# ── Stage 1: Build frontend ────────────────────────────────────────────────
FROM node:20-alpine AS frontend-build

WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --silent
COPY frontend/ .
RUN npm run build

# ── Stage 2: Python backend ────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend
COPY backend/ backend/
COPY train.py .

# Copy built frontend
COPY --from=frontend-build /app/frontend/dist /app/frontend/dist

# Create data dirs
RUN mkdir -p data/uploads data/processed models

# Env defaults
ENV PYTHONUNBUFFERED=1
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
