# syntax=docker/dockerfile:1
# Multi-stage build – runtime image stays slim.

# ---------------------------------------------------------------------------
# Stage 1: dependency installation
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: minimal runtime image
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Non-root user for security
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY app/ ./app/
COPY models.yaml ./

# NPU device access: /dev/accel must be mounted at runtime (--device /dev/accel)
# Models must be mounted read-only: -v /host/models:/models:ro

ENV CONFIG_PATH=models.yaml \
    NPU_DEVICE_STRING=NPU \
    LOG_LEVEL=INFO \
    OPENVINO_API_HOST=0.0.0.0 \
    OPENVINO_API_PORT=4647 \
    OPENVINO_API_THREAD_POOL_SIZE=4

USER appuser

EXPOSE 4647

# Single worker enforced: NPU context is per-process
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", "--port", "4647", "--workers", "1", "--no-access-log"]
