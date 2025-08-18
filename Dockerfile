# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install UV package manager
RUN pip install uv

# Copy dependency files
COPY pyproject.toml ./

# Install dependencies
RUN uv pip install --system -e ".[prod]"

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Create necessary directories
RUN mkdir -p /app/data /app/logs

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/api/v1/health || exit 1

# Expose port
EXPOSE 8001

# Default command
CMD ["uvicorn", "src.dfs_prophet.main:app", "--host", "0.0.0.0", "--port", "8001"]

# Development stage
FROM base as development

# Switch back to root for development
USER root

# Install development dependencies
RUN uv pip install --system -e ".[dev]"

# Switch back to app user
USER app

# Development command with reload
CMD ["uvicorn", "src.dfs_prophet.main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]

# Production stage with multi-stage build
FROM base as production

# Copy only necessary files for production
COPY --chown=app:app src/ ./src/

# Production command with multiple workers
CMD ["gunicorn", "src.dfs_prophet.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8001"]
