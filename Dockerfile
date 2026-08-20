FROM node:22-alpine AS frontend-build

WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/tsconfig.json ./tsconfig.json
COPY frontend/src ./src
RUN npm run build

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV, Pillow, Tesseract OCR, and PDF rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Copy application code and the compiled typed browser module
COPY . .
ARG MODEL_BASE_URL=https://raw.githubusercontent.com/sathyadharma082010-source/nephroscan-ai/main/models
RUN mkdir -p models && set -eux; \
    for model in \
      kidney_stone_resnet18.pth \
      chest_pneumonia_resnet18.pth \
      brain_mri_resnet18.pth \
      heart_cardiomegaly_resnet18_improved.pth; do \
      if [ ! -s "models/$model" ]; then \
        curl --fail --location --retry 3 --silent --show-error "$MODEL_BASE_URL/$model" -o "models/$model"; \
      fi; \
    done
COPY --from=frontend-build /frontend/dist ./frontend/dist

# Create models directory if not present
RUN mkdir -p models

# Environment
ENV PORT=8080
ENV FLASK_ENV=production
ENV MODEL_DIR=models
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV MALLOC_ARENA_MAX=2

EXPOSE $PORT

# One worker keeps four in-memory PyTorch models within Railway Trial RAM.
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 1 --timeout 180 app:app"]
