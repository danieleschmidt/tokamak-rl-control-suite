#!/bin/bash
# Build script for Tokamak RL Control Suite
# Provides comprehensive build automation with multiple targets

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="tokamak-rl-control-suite"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-ghcr.io/username}"
VERSION="${VERSION:-$(cat pyproject.toml | grep version | cut -d'"' -f2)}"
BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    cat << EOF
Tokamak RL Control Suite Build Script

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    all                 Build all targets (default)
    clean              Clean build artifacts
    lint               Run code linting
    test               Run test suite
    type-check         Run type checking
    security           Run security scans
    docs               Build documentation
    package            Build Python package
    docker             Build Docker images
    docker-dev         Build development Docker image
    docker-prod        Build production Docker image
    docker-docs        Build documentation Docker image
    push               Push Docker images to registry
    release            Create release build (lint + test + package + docker)
    install            Install package locally
    install-dev        Install in development mode
    check              Run all checks (lint + test + type-check + security)

Options:
    -v, --verbose      Verbose output
    -h, --help         Show this help message
    --no-cache         Disable Docker build cache
    --parallel         Run tests in parallel

Examples:
    $0                 # Build everything
    $0 test            # Run tests only
    $0 docker --no-cache  # Build Docker images without cache
    $0 release         # Create full release build

Environment Variables:
    DOCKER_REGISTRY    Docker registry URL (default: ghcr.io/username)
    VERSION           Override version from pyproject.toml
    PYTHON_VERSION    Python version for Docker (default: 3.11)
    BUILD_PARALLEL    Number of parallel jobs (default: auto)

EOF
}

# Parse arguments
VERBOSE=false
NO_CACHE=""
PARALLEL=false
COMMAND="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        *)
            COMMAND="$1"
            shift
            ;;
    esac
done

# Set verbose output
if [[ "$VERBOSE" == "true" ]]; then
    set -x
fi

# Build functions
clean() {
    log_info "Cleaning build artifacts..."
    
    # Python build artifacts
    rm -rf build/
    rm -rf dist/
    rm -rf *.egg-info/
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    
    # Test artifacts
    rm -rf .coverage
    rm -rf htmlcov/
    rm -rf .tox/
    
    # Documentation
    rm -rf docs/_build/
    
    # Docker cleanup
    if command -v docker &> /dev/null; then
        log_info "Cleaning Docker artifacts..."
        docker system prune -f --volumes || true
        
        # Remove project images
        docker images --format "table {{.Repository}}:{{.Tag}}" | grep "$PROJECT_NAME" | xargs -r docker rmi || true
    fi
    
    log_success "Clean completed"
}

lint() {
    log_info "Running code linting..."
    
    # Check if tools are installed
    if ! command -v ruff &> /dev/null; then
        log_warning "ruff not found, installing..."
        pip install ruff
    fi
    
    if ! command -v black &> /dev/null; then
        log_warning "black not found, installing..."
        pip install black
    fi
    
    # Run formatters and linters
    log_info "Running black formatter..."
    black --check --diff src/ tests/ || {
        log_warning "Code formatting issues found. Run 'black src/ tests/' to fix."
        return 1
    }
    
    log_info "Running ruff linter..."
    ruff check src/ tests/
    
    log_info "Running import sorting check..."
    ruff check --select I src/ tests/
    
    log_success "Linting completed"
}

type_check() {
    log_info "Running type checking..."
    
    if ! command -v mypy &> /dev/null; then
        log_warning "mypy not found, installing..."
        pip install mypy
    fi
    
    # Run mypy
    mypy src/tokamak_rl/ --ignore-missing-imports
    
    log_success "Type checking completed"
}

test() {
    log_info "Running test suite..."
    
    # Set test options
    TEST_OPTS=""
    if [[ "$PARALLEL" == "true" ]]; then
        TEST_OPTS="-n auto"
    fi
    
    # Run tests with coverage
    python -m pytest tests/ \
        --cov=src/tokamak_rl \
        --cov-report=term-missing \
        --cov-report=html \
        --cov-report=xml \
        --junitxml=test-results.xml \
        -v $TEST_OPTS
    
    # Check coverage threshold
    COVERAGE=$(python -c "import xml.etree.ElementTree as ET; print(float(ET.parse('coverage.xml').getroot().attrib['line-rate']) * 100)" 2>/dev/null || echo "0")
    THRESHOLD=80
    
    if (( $(echo "$COVERAGE < $THRESHOLD" | bc -l) )); then
        log_warning "Coverage ${COVERAGE}% is below threshold ${THRESHOLD}%"
    else
        log_success "Coverage ${COVERAGE}% meets threshold ${THRESHOLD}%"
    fi
    
    log_success "Testing completed"
}

security() {
    log_info "Running security scans..."
    
    if ! command -v bandit &> /dev/null; then
        log_warning "bandit not found, installing..."
        pip install bandit
    fi
    
    if ! command -v safety &> /dev/null; then
        log_warning "safety not found, installing..."
        pip install safety
    fi
    
    # Run bandit security linter
    log_info "Running bandit security scan..."
    bandit -r src/ -f json -o bandit-report.json || {
        log_warning "Security issues found. Check bandit-report.json"
    }
    
    # Check dependencies for known vulnerabilities
    log_info "Checking dependencies for vulnerabilities..."
    safety check --json --output safety-report.json || {
        log_warning "Vulnerable dependencies found. Check safety-report.json"
    }
    
    log_success "Security scanning completed"
}

docs() {
    log_info "Building documentation..."
    
    if ! command -v sphinx-build &> /dev/null; then
        log_warning "sphinx not found, installing docs dependencies..."
        pip install -e ".[docs]"
    fi
    
    # Build docs
    sphinx-build -b html docs/ docs/_build/html/
    
    # Check for broken links
    sphinx-build -b linkcheck docs/ docs/_build/linkcheck/ || {
        log_warning "Broken links found in documentation"
    }
    
    log_success "Documentation built at docs/_build/html/"
}

package() {
    log_info "Building Python package..."
    
    # Clean previous builds
    rm -rf build/ dist/ *.egg-info/
    
    # Build package
    python -m build
    
    # Verify package
    python -m twine check dist/*
    
    log_success "Package built successfully"
    ls -la dist/
}

docker_build() {
    local target="$1"
    local tag="$2"
    local dockerfile="${3:-Dockerfile}"
    
    log_info "Building Docker image: $tag (target: $target)"
    
    docker build $NO_CACHE \
        --target "$target" \
        --tag "$tag" \
        --build-arg VERSION="$VERSION" \
        --build-arg BUILD_DATE="$BUILD_DATE" \
        --build-arg GIT_COMMIT="$GIT_COMMIT" \
        --file "$dockerfile" \
        .
        
    log_success "Docker image built: $tag"
}

docker() {
    log_info "Building all Docker images..."
    
    # Build development image
    docker_build "development" "${DOCKER_REGISTRY}/${PROJECT_NAME}:${VERSION}-dev"
    docker_build "development" "${DOCKER_REGISTRY}/${PROJECT_NAME}:latest-dev"
    
    # Build production image
    docker_build "production" "${DOCKER_REGISTRY}/${PROJECT_NAME}:${VERSION}"
    docker_build "production" "${DOCKER_REGISTRY}/${PROJECT_NAME}:latest"
    
    # Build docs image
    docker_build "docs" "${DOCKER_REGISTRY}/${PROJECT_NAME}:${VERSION}-docs"
    docker_build "docs" "${DOCKER_REGISTRY}/${PROJECT_NAME}:latest-docs"
    
    log_success "All Docker images built"
}

docker_dev() {
    docker_build "development" "${DOCKER_REGISTRY}/${PROJECT_NAME}:${VERSION}-dev"
    docker_build "development" "${DOCKER_REGISTRY}/${PROJECT_NAME}:latest-dev"
}

docker_prod() {
    docker_build "production" "${DOCKER_REGISTRY}/${PROJECT_NAME}:${VERSION}"
    docker_build "production" "${DOCKER_REGISTRY}/${PROJECT_NAME}:latest"
}

docker_docs() {
    docker_build "docs" "${DOCKER_REGISTRY}/${PROJECT_NAME}:${VERSION}-docs"
    docker_build "docs" "${DOCKER_REGISTRY}/${PROJECT_NAME}:latest-docs"
}

push() {
    log_info "Pushing Docker images to registry..."
    
    # Push all images
    docker push "${DOCKER_REGISTRY}/${PROJECT_NAME}:${VERSION}"
    docker push "${DOCKER_REGISTRY}/${PROJECT_NAME}:latest"
    docker push "${DOCKER_REGISTRY}/${PROJECT_NAME}:${VERSION}-dev"
    docker push "${DOCKER_REGISTRY}/${PROJECT_NAME}:latest-dev"
    docker push "${DOCKER_REGISTRY}/${PROJECT_NAME}:${VERSION}-docs"
    docker push "${DOCKER_REGISTRY}/${PROJECT_NAME}:latest-docs"
    
    log_success "Images pushed to registry"
}

install() {
    log_info "Installing package..."
    pip install .
    log_success "Package installed"
}

install_dev() {
    log_info "Installing package in development mode..."
    pip install -e ".[dev,docs,mpi]"
    log_success "Package installed in development mode"
}

check() {
    log_info "Running all checks..."
    lint
    type_check
    test
    security
    log_success "All checks completed"
}

release() {
    log_info "Creating release build..."
    
    # Run all checks
    check
    
    # Build artifacts
    docs
    package
    docker
    
    log_success "Release build completed"
    
    # Show summary
    echo
    log_info "Release Summary:"
    echo "  Version: $VERSION"
    echo "  Git Commit: $GIT_COMMIT"
    echo "  Build Date: $BUILD_DATE"
    echo "  Package: $(ls dist/*.whl | head -1)"
    echo "  Docker Images:"
    echo "    - ${DOCKER_REGISTRY}/${PROJECT_NAME}:${VERSION}"
    echo "    - ${DOCKER_REGISTRY}/${PROJECT_NAME}:${VERSION}-dev"
    echo "    - ${DOCKER_REGISTRY}/${PROJECT_NAME}:${VERSION}-docs"
}

all() {
    log_info "Running full build pipeline..."
    clean
    check
    docs
    package
    docker
    log_success "Full build completed"
}

# Execute command
case "$COMMAND" in
    all)
        all
        ;;
    clean)
        clean
        ;;
    lint)
        lint
        ;;
    test)
        test
        ;;
    type-check)
        type_check
        ;;
    security)
        security
        ;;
    docs)
        docs
        ;;
    package)
        package
        ;;
    docker)
        docker
        ;;
    docker-dev)
        docker_dev
        ;;
    docker-prod)
        docker_prod
        ;;
    docker-docs)
        docker_docs
        ;;
    push)
        push
        ;;
    install)
        install
        ;;
    install-dev)
        install_dev
        ;;
    check)
        check
        ;;
    release)
        release
        ;;
    help)
        show_help
        ;;
    *)
        log_error "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac

log_success "Build script completed successfully"