#!/bin/bash
set -e

NAMESPACE="tokamak-rl"

echo "🚀 Deploying Tokamak-RL to Kubernetes..."

# Apply manifests in order
echo "📁 Creating namespace..."
kubectl apply -f deployment_v6/k8s/namespace.yaml

echo "⚙️  Applying configuration..."
kubectl apply -f deployment_v6/k8s/configmap.yaml
kubectl apply -f deployment_v6/k8s/pvc.yaml

echo "🚀 Deploying application..."
kubectl apply -f deployment_v6/k8s/deployment.yaml
kubectl apply -f deployment_v6/k8s/service.yaml
kubectl apply -f deployment_v6/k8s/ingress.yaml

if [ -f "deployment_v6/k8s/hpa.yaml" ]; then
    echo "📈 Setting up autoscaling..."
    kubectl apply -f deployment_v6/k8s/hpa.yaml
fi

echo "⏳ Waiting for deployment to be ready..."
kubectl rollout status deployment/tokamak-rl -n $NAMESPACE --timeout=300s

echo "✅ Deployment completed successfully!"
echo "🌐 Application available at: https://tokamak-rl.terragonlabs.io"
