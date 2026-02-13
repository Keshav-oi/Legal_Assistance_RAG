FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies for document loaders (PDF, DOCX, HTML parsing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install llama-cpp-python for CPU (Metal is not available inside Docker)
# For GPU support on Linux with NVIDIA, set CMAKE_ARGS="-DGGML_CUDA=on"
RUN pip install --no-cache-dir llama-cpp-python==0.3.4

# Copy application code
COPY app/ app/

# Create data directories (will be overridden by volume mounts)
RUN mkdir -p /app/data/documents /app/data/faiss_index /app/models

# Environment defaults (can be overridden in docker-compose.yml)
ENV BASE_DIR=/app \
    MODEL_DIR=/app/models \
    DOCUMENTS_DIR=/app/data/documents \
    FAISS_INDEX_DIR=/app/data/faiss_index \
    MODEL_N_GPU_LAYERS=0 \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000

EXPOSE 8000

# Health check against the API endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
