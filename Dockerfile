# syntax=docker/dockerfile:1
#
# Serving image for isr-restore: ONNX Runtime + FastAPI, NO PyTorch.
# Multi-stage build: the builder compiles/installs the wheels, the runtime stage only
# receives the finished virtual environment, which keeps the final image small.

# ----------------------------------------------------------------------------- builder
FROM python:3.15-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src

# Only the `serve` extra: numpy, OpenCV (headless), PyYAML, ONNX Runtime, FastAPI, uvicorn.
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install ".[serve]"

# ----------------------------------------------------------------------------- runtime
FROM python:3.15-slim AS runtime

# Run as an unprivileged user: a compromised API process must not be root in the container.
RUN useradd --create-home --uid 10001 appuser

COPY --from=builder /opt/venv /opt/venv
# The ONNX models are baked into the image: the image tag IS the model version.
COPY models/restoration.onnx models/detector.onnx /app/models/

ENV PATH="/opt/venv/bin:$PATH" \
    ISR_MODEL_DIR=/app/models \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
USER appuser
EXPOSE 8000

# Marks the container unhealthy when the models are missing (the endpoint returns 503).
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["uvicorn", "isr.api:app", "--host", "0.0.0.0", "--port", "8000"]
