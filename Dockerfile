FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV and Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Copy application code
COPY . .

# Create models directory if not present
RUN mkdir -p models

# Environment
ENV PORT=8080
ENV FLASK_ENV=production
ENV MODEL_DIR=models
ENV PYTHONUNBUFFERED=1

EXPOSE $PORT

# Production server: Gunicorn with 1 worker, 180s timeout, no --preload (PyTorch fork-safe)
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 1 --timeout 180 app:app"]
