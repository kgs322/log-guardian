# docker/Dockerfile
# -----------------------------
# Stage 1: builder (create wheels)
# -----------------------------
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Build deps for common scientific libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Resolve & cache dependencies
COPY requirements.txt .
RUN pip wheel --wheel-dir /wheels -r requirements.txt

# 2) Build your package wheel (uses pyproject.toml)
COPY pyproject.toml ./
COPY src ./src
RUN pip wheel --wheel-dir /wheels .

# -----------------------------
# Stage 2: runtime
# -----------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LOG_GUARDIAN_CONFIG="configs/prod.yaml" \
    APP_HOST="0.0.0.0" \
    APP_PORT="8000"

# Small runtime tools
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd -m -u 10001 appuser
WORKDIR /app

# Install from wheels (no network at runtime)
COPY --from=builder /wheels /wheels
RUN pip install --no-index --find-links=/wheels /wheels/*

# Project files (kept minimal; mounts will override in compose)
COPY configs ./configs
COPY src ./src
COPY scripts ./scripts
COPY models ./models
COPY data/samples ./data/samples
COPY .env.example ./.env.example
COPY README.md ./README.md

# Writable dirs
RUN mkdir -p data/raw data/processed models/artifacts logs \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Healthcheck (FastAPI root — change path if needed)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS "http://localhost:${APP_PORT}/" || exit 1

# Default: serve API (override CMD or LOG_GUARDIAN_CONFIG at runtime)
ENTRYPOINT ["bash", "-lc"]
CMD ["uvicorn log_guardian.api.app:app --host ${APP_HOST} --port ${APP_PORT}"]
