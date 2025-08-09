#!/bin/bash
set -e

# Tokamak RL Control Suite - Deployment Script
echo "🚀 Starting deployment of Tokamak RL Control Suite"

ENVIRONMENT=${1:-production}
ACTION=${2:-deploy}

echo "Environment: $ENVIRONMENT"
echo "Action: $ACTION"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Pre-deployment checks
pre_deployment_checks() {
    log_info "Running pre-deployment checks..."
    
    # Check if required tools are installed
    command -v docker >/dev/null 2>&1 || { log_error "Docker is required but not installed. Aborting."; exit 1; }
    command -v kubectl >/dev/null 2>&1 || { log_error "kubectl is required but not installed. Aborting."; exit 1; }
    command -v terraform >/dev/null 2>&1 || { log_error "Terraform is required but not installed. Aborting."; exit 1; }
    
    # Check cluster connectivity
    kubectl cluster-info >/dev/null 2>&1 || { log_error "Cannot connect to Kubernetes cluster. Aborting."; exit 1; }
    
    log_info "Pre-deployment checks passed ✅"
}

# Build and push Docker images
build_and_push() {
    log_info "Building and pushing Docker images..."
    
    # Build production image
    docker build -f deployment/docker/Dockerfile.prod -t tokamak-rl:$ENVIRONMENT .
    
    # Tag and push to registry
    docker tag tokamak-rl:$ENVIRONMENT your-registry.com/tokamak-rl:$ENVIRONMENT
    docker push your-registry.com/tokamak-rl:$ENVIRONMENT
    
    log_info "Docker images built and pushed ✅"
}

# Deploy infrastructure
deploy_infrastructure() {
    log_info "Deploying infrastructure with Terraform..."
    
    cd deployment/terraform
    terraform init
    terraform plan -var="environment=$ENVIRONMENT"
    terraform apply -var="environment=$ENVIRONMENT" -auto-approve
    cd ../..
    
    log_info "Infrastructure deployed ✅"
}

# Deploy application
deploy_application() {
    log_info "Deploying application to Kubernetes..."
    
    # Apply Kubernetes manifests
    kubectl apply -f deployment/kubernetes/deployment-$ENVIRONMENT.yaml
    
    # Wait for rollout to complete
    kubectl rollout status deployment/tokamak-rl-$ENVIRONMENT -n tokamak-rl
    
    # Apply ingress if production
    if [ "$ENVIRONMENT" == "production" ]; then
        kubectl apply -f deployment/kubernetes/ingress.yaml
    fi
    
    log_info "Application deployed ✅"
}

# Run health checks
health_checks() {
    log_info "Running health checks..."
    
    # Wait for pods to be ready
    kubectl wait --for=condition=ready pod -l app=tokamak-rl,environment=$ENVIRONMENT -n tokamak-rl --timeout=300s
    
    # Check service endpoints
    SERVICE_IP=$(kubectl get svc tokamak-rl-service-$ENVIRONMENT -n tokamak-rl -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    if [ ! -z "$SERVICE_IP" ]; then
        curl -f http://$SERVICE_IP/health || { log_error "Health check failed"; exit 1; }
        log_info "Health checks passed ✅"
    else
        log_warn "Service IP not available yet, skipping external health check"
    fi
}

# Rollback function
rollback() {
    log_warn "Rolling back deployment..."
    kubectl rollout undo deployment/tokamak-rl-$ENVIRONMENT -n tokamak-rl
    kubectl rollout status deployment/tokamak-rl-$ENVIRONMENT -n tokamak-rl
    log_info "Rollback completed ✅"
}

# Main deployment flow
case $ACTION in
    "deploy")
        pre_deployment_checks
        build_and_push
        deploy_infrastructure
        deploy_application
        health_checks
        log_info "🎉 Deployment completed successfully!"
        ;;
    "rollback")
        rollback
        ;;
    "infrastructure")
        deploy_infrastructure
        ;;
    "application")
        deploy_application
        health_checks
        ;;
    *)
        echo "Usage: $0 <environment> <action>"
        echo "Actions: deploy, rollback, infrastructure, application"
        exit 1
        ;;
esac
