# Multi-stage build for tokamak-rl v6.0
FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install dependencies
COPY pyproject.toml /tmp/
WORKDIR /tmp
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir .

# Production stage
FROM python:3.11-slim as production

# Create non-root user
RUN groupadd -r tokamak && useradd -r -g tokamak tokamak

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libopenblas0 \
    liblapack3 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY src/ /app/src/
COPY *.py /app/
COPY README.md LICENSE /app/

# Set ownership and permissions
RUN chown -R tokamak:tokamak /app \
    && chmod -R 755 /app

# Create data directories
RUN mkdir -p /app/data /app/logs /app/checkpoints \
    && chown -R tokamak:tokamak /app/data /app/logs /app/checkpoints

# Switch to non-root user
USER tokamak

# Environment variables
ENV PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TOKAMAK_ENV=production \
    TOKAMAK_LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "-m", "tokamak_rl.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
