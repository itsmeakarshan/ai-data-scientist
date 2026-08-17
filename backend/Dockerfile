FROM python:3.11-slim

# System dependencies for OpenMP (LightGBM/XGBoost), build tools, and curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/backend/requirements.txt && \
    pip install --no-cache-dir greenlet>=3.0.0

# Copy application source code
COPY backend /app/backend
COPY scripts /app/scripts
COPY evaluation /app/evaluation
COPY alembic.ini /app/alembic.ini
COPY pyproject.toml /app/pyproject.toml

# Create directories
RUN mkdir -p /app/data/raw /app/data/processed /app/reports/artifacts /app/experiments/artifacts /app/mlruns

ENV PYTHONPATH=/app \
    PORT=8000 \
    HOST=0.0.0.0

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
