#!/bin/bash
# Production deployment script

set -e

echo "🚀 Starting Tokamak RL production deployment..."

# Check prerequisites
echo "Checking prerequisites..."
command -v docker >/dev/null 2>&1 || { echo "Docker is required but not installed. Aborting." >&2; exit 1; }
command -v kubectl >/dev/null 2>&1 || { echo "kubectl is required but not installed. Aborting." >&2; exit 1; }

# Build and tag Docker image
echo "Building Docker image..."
docker build -f docker/Dockerfile -t tokamak-rl:latest .

# Deploy to Kubernetes
echo "Deploying to Kubernetes..."
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl apply -f kubernetes/hpa.yaml

# Wait for deployment to be ready
echo "Waiting for deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/tokamak-rl-api -n tokamak-rl-production

# Run health checks
echo "Running health checks..."
kubectl get pods -n tokamak-rl-production

echo "✅ Deployment completed successfully!"
