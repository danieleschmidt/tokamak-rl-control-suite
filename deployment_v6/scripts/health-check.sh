#!/bin/bash

NAMESPACE="tokamak-rl"

echo "🏥 Checking Tokamak-RL health..."

# Check pods
echo "📊 Pod status:"
kubectl get pods -n $NAMESPACE

# Check services
echo "🌐 Service status:"
kubectl get services -n $NAMESPACE

# Check ingress
echo "🚪 Ingress status:"
kubectl get ingress -n $NAMESPACE

# Check HPA (if exists)
if kubectl get hpa -n $NAMESPACE tokamak-rl-hpa >/dev/null 2>&1; then
    echo "📈 Autoscaler status:"
    kubectl get hpa -n $NAMESPACE
fi

echo "✅ Health check completed!"
