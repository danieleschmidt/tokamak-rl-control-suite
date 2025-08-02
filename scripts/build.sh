#!/bin/bash
set -euo pipefail

# Build script for tokamak-rl-control-suite
# Provides standardized building, testing, and deployment automation

# Default values
BUILD_TARGET="development"
DOCKER_REGISTRY=""
IMAGE_TAG="latest"
PUSH_IMAGE=false
RUN_TESTS=true
VERBOSE=false
CLEAN_BUILD=false
PLATFORM="linux/amd64"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
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

# Show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Build and deploy tokamak-rl-control-suite Docker images

OPTIONS:
    -t, --target TARGET       Build target (development|production|docs) [default: development]
    -r, --registry REGISTRY   Docker registry to push to
    --tag TAG                 Image tag [default: latest]
    --push                    Push image to registry after build
    --no-tests                Skip running tests during build
    --clean                   Clean build (no cache)
    --platform PLATFORM      Target platform [default: linux/amd64]
    -v, --verbose             Verbose output
    -h, --help                Show this help message

EXAMPLES:
    $0                                           # Build development image
    $0 -t production --tag v1.0.0              # Build production image with tag
    $0 -t production --push --registry myregistry.io  # Build and push to registry
    $0 --clean -v                               # Clean verbose build

TARGETS:
    development     Full development environment with tools
    production      Optimized production environment
    docs            Documentation builder and server
    all             Build all targets

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -t|--target)
                BUILD_TARGET="$2"
                shift 2
                ;;
            -r|--registry)
                DOCKER_REGISTRY="$2"
                shift 2
                ;;
            --tag)
                IMAGE_TAG="$2"
                shift 2
                ;;
            --push)
                PUSH_IMAGE=true
                shift
                ;;
            --no-tests)
                RUN_TESTS=false
                shift
                ;;
            --clean)
                CLEAN_BUILD=true
                shift
                ;;
            --platform)
                PLATFORM="$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

# Validate build target
validate_target() {
    case $BUILD_TARGET in
        development|production|docs|all)
            log_info "Building target: $BUILD_TARGET"
            ;;
        *)
            log_error "Invalid build target: $BUILD_TARGET"
            log_info "Valid targets: development, production, docs, all"
            exit 1
            ;;
    esac
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is required but not installed"
        exit 1
    fi
    
    # Check Docker Buildx (for multi-platform builds)
    if ! docker buildx version &> /dev/null; then
        log_warning "Docker Buildx not available, using legacy build"
    fi
    
    # Check if Docker is running
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Get image name
get_image_name() {
    local target=$1
    local base_name="tokamak-rl-control-suite"
    
    if [[ -n "$DOCKER_REGISTRY" ]]; then
        echo "${DOCKER_REGISTRY}/${base_name}:${target}-${IMAGE_TAG}"
    else
        echo "${base_name}:${target}-${IMAGE_TAG}"
    fi
}

# Build single target
build_target() {
    local target=$1
    local image_name=$(get_image_name "$target")
    
    log_info "Building $target target..."
    log_info "Image name: $image_name"
    
    # Prepare build arguments
    local build_args=""
    if [[ "$VERBOSE" == "true" ]]; then
        build_args="--progress=plain"
    fi
    
    if [[ "$CLEAN_BUILD" == "true" ]]; then
        build_args="$build_args --no-cache"
    fi
    
    # Build command
    local build_cmd="docker build"
    
    # Use buildx if available for multi-platform builds
    if docker buildx version &> /dev/null && [[ "$PLATFORM" != "linux/amd64" || "$target" == "all" ]]; then
        build_cmd="docker buildx build --platform $PLATFORM"
    fi
    
    # Execute build
    cd "$PROJECT_ROOT"
    
    if [[ "$target" == "all" ]]; then
        # Build all targets
        for t in development production docs; do
            local t_image_name=$(get_image_name "$t")
            log_info "Building $t..."
            
            $build_cmd \
                $build_args \
                --target "$t" \
                --tag "$t_image_name" \
                .
            
            log_success "Built $t_image_name"
        done
    else
        # Build single target
        $build_cmd \
            $build_args \
            --target "$target" \
            --tag "$image_name" \
            .
        
        log_success "Built $image_name"
    fi
}

# Run tests
run_tests() {
    if [[ "$RUN_TESTS" == "false" ]]; then
        log_info "Skipping tests (--no-tests specified)"
        return 0
    fi
    
    log_info "Running tests..."
    
    # Use test service from docker-compose
    cd "$PROJECT_ROOT"
    
    if [[ "$BUILD_TARGET" == "all" ]] || [[ "$BUILD_TARGET" == "development" ]]; then
        docker-compose run --rm tokamak-test
        log_success "Tests passed"
    else
        log_info "Skipping tests for $BUILD_TARGET target"
    fi
}

# Push image to registry
push_image() {
    if [[ "$PUSH_IMAGE" == "false" ]]; then
        log_info "Skipping image push"
        return 0
    fi
    
    if [[ -z "$DOCKER_REGISTRY" ]]; then
        log_error "Registry not specified, cannot push image"
        exit 1
    fi
    
    log_info "Pushing images to registry..."
    
    if [[ "$BUILD_TARGET" == "all" ]]; then
        for target in development production docs; do
            local image_name=$(get_image_name "$target")
            log_info "Pushing $image_name..."
            docker push "$image_name"
            log_success "Pushed $image_name"
        done
    else
        local image_name=$(get_image_name "$BUILD_TARGET")
        log_info "Pushing $image_name..."
        docker push "$image_name"
        log_success "Pushed $image_name"
    fi
}

# Security scan
security_scan() {
    log_info "Running security scan..."
    
    local image_name=$(get_image_name "$BUILD_TARGET")
    
    # Use trivy if available
    if command -v trivy &> /dev/null; then
        log_info "Scanning with Trivy..."
        trivy image --severity HIGH,CRITICAL "$image_name"
    else
        log_warning "Trivy not available, skipping security scan"
        log_info "Install Trivy for security scanning: https://github.com/aquasecurity/trivy"
    fi
}

# Generate build report
generate_build_report() {
    log_info "Generating build report..."
    
    local report_file="${PROJECT_ROOT}/build-report-$(date +%Y%m%d-%H%M%S).json"
    
    cat > "$report_file" << EOF
{
  "build_info": {
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "target": "$BUILD_TARGET",
    "tag": "$IMAGE_TAG",
    "platform": "$PLATFORM",
    "registry": "$DOCKER_REGISTRY",
    "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
    "git_branch": "$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
  },
  "images": [
EOF

    if [[ "$BUILD_TARGET" == "all" ]]; then
        for target in development production docs; do
            local image_name=$(get_image_name "$target")
            local image_id=$(docker images --format "{{.ID}}" --filter "reference=$image_name" | head -1)
            local image_size=$(docker images --format "{{.Size}}" --filter "reference=$image_name" | head -1)
            
            cat >> "$report_file" << EOF
    {
      "target": "$target",
      "name": "$image_name",
      "id": "$image_id",
      "size": "$image_size"
    },
EOF
        done
        # Remove trailing comma
        sed -i '$ s/,$//' "$report_file"
    else
        local image_name=$(get_image_name "$BUILD_TARGET")
        local image_id=$(docker images --format "{{.ID}}" --filter "reference=$image_name" | head -1)
        local image_size=$(docker images --format "{{.Size}}" --filter "reference=$image_name" | head -1)
        
        cat >> "$report_file" << EOF
    {
      "target": "$BUILD_TARGET",
      "name": "$image_name",
      "id": "$image_id",
      "size": "$image_size"
    }
EOF
    fi
    
    cat >> "$report_file" << EOF
  ]
}
EOF
    
    log_success "Build report saved to: $report_file"
}

# Main function
main() {
    log_info "Starting tokamak-rl-control-suite build process..."
    
    parse_args "$@"
    validate_target
    check_prerequisites
    
    # Start timer
    start_time=$(date +%s)
    
    # Build process
    build_target "$BUILD_TARGET"
    run_tests
    security_scan
    push_image
    generate_build_report
    
    # End timer
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    log_success "Build completed successfully in ${duration}s"
    
    # Show image information
    if [[ "$BUILD_TARGET" == "all" ]]; then
        log_info "Built images:"
        for target in development production docs; do
            local image_name=$(get_image_name "$target")
            docker images --filter "reference=$image_name" --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
        done
    else
        local image_name=$(get_image_name "$BUILD_TARGET")
        log_info "Built image:"
        docker images --filter "reference=$image_name" --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
    fi
}

# Run main function with all arguments
main "$@"