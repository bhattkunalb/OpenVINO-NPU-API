# syntax=docker/dockerfile:1
# ── Stage 1: build dependencies ──────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .

RUN pip install --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Intel NPU driver must be installed on the HOST; the container only needs
# the OpenVINO Python bindings.  Mount /dev/accel (or equivalent) at runtime.
WORKDIR /app

COPY --from=builder /install /usr/local
COPY app/ ./app/
COPY models.yaml .

ENV OPENVINO_API_MODEL_CONFIG=models.yaml \
    OPENVINO_API_DEVICE=NPU \
    OPENVINO_API_LOG_LEVEL=INFO \
    OPENVINO_API_THREAD_POOL_SIZE=4

EXPOSE 8000

# Single worker – NPU context is in-process; do not scale with multiple workers.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
