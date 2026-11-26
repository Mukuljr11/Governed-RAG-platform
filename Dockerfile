# Stage 1: build the virtual environment
FROM python:3.12.15-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=10 UV_CONCURRENT_DOWNLOADS=4
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Stage 2: small runtime image
FROM python:3.12.15-slim
RUN useradd --create-home appuser
RUN mkdir -p /data/uploads && chown -R appuser:appuser /data
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY app ./app
COPY alembic.ini ./
COPY migrations ./migrations
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
USER appuser
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]