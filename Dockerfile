# Multi-stage Dockerfile for Tokamak RL Control Suite
# Optimized for both development and production usage

# Stage 1: Base Python environment with system dependencies
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for scientific computing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libhdf5-dev \
    pkg-config \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd --gid 1000 tokamak && \
    useradd --uid 1000 --gid tokamak --shell /bin/bash --create-home tokamak

# Stage 2: Development environment
FROM base as development

# Install development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    htop \
    tree \
    && rm -rf /var/lib/apt/lists/*

# Install Python development dependencies
RUN pip install --no-cache-dir \
    pip-tools \
    pre-commit \
    jupyter \
    ipykernel

# Set working directory
WORKDIR /workspace

# Copy project files
COPY --chown=tokamak:tokamak . /workspace/

# Switch to non-root user
USER tokamak

# Install project in development mode
RUN pip install --no-cache-dir -e ".[dev,docs,mpi]"

# Install pre-commit hooks
RUN git init --initial-branch=main 2>/dev/null || true && \
    pre-commit install 2>/dev/null || true

# Expose Jupyter port
EXPOSE 8888

# Default command for development
CMD ["bash"]

# Stage 3: Production environment
FROM base as production

# Install only production dependencies
WORKDIR /app

# Copy project files
COPY --chown=tokamak:tokamak . /app/

# Switch to non-root user
USER tokamak

# Install project in production mode
RUN pip install --no-cache-dir -e ".[mpi]"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import tokamak_rl; print(tokamak_rl.__version__)" || exit 1

# Run tests to validate build
RUN python -m pytest tests/ -v --tb=short

# Default command for production
CMD ["python", "-m", "tokamak_rl.cli"]

# Stage 4: Documentation builder
FROM base as docs

WORKDIR /docs

# Install documentation dependencies
RUN pip install --no-cache-dir \
    sphinx \
    sphinx-rtd-theme \
    myst-parser \
    sphinx-autodoc-typehints

# Copy source and docs
COPY --chown=tokamak:tokamak . /docs/

USER tokamak

# Build documentation
RUN sphinx-build -b html docs/ docs/_build/html/

# Serve documentation
EXPOSE 8000
CMD ["python", "-m", "http.server", "8000", "--directory", "docs/_build/html/"]