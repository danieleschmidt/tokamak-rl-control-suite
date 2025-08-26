# Production Dockerfile for Tokamak RL Control Suite
FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# Production stage
FROM python:3.11-slim as production

# Create non-root user
RUN useradd --create-home --shell /bin/bash tokamak

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libopenblas0 \
    liblapack3 \
    && rm -rf /var/lib/apt/lists/*

# Copy application
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY src/ ./src/
COPY --chown=tokamak:tokamak . .

# Switch to non-root user
USER tokamak

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "from src.tokamak_rl import make_tokamak_env; make_tokamak_env()" || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "-m", "tokamak_rl.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
