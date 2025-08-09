#!/bin/bash

# Tokamak RL Control Suite - Monitoring Script
echo "📊 System monitoring check"

NAMESPACE="tokamak-rl"
ENVIRONMENT=${1:-production}

# Check pod status
echo "🏃 Pod Status:"
kubectl get pods -n $NAMESPACE -l environment=$ENVIRONMENT

# Check resource usage
echo "📈 Resource Usage:"
kubectl top pods -n $NAMESPACE -l environment=$ENVIRONMENT

# Check recent logs for errors
echo "📋 Recent Error Logs:"
kubectl logs -n $NAMESPACE -l environment=$ENVIRONMENT --since=1h | grep -i error | tail -10

# Check service endpoints
echo "🌐 Service Status:"
kubectl get svc -n $NAMESPACE

# Check ingress status
if [ "$ENVIRONMENT" == "production" ]; then
    echo "🌍 Ingress Status:"
    kubectl get ingress -n $NAMESPACE
fi

# Check persistent volume usage
echo "💾 Storage Usage:"
kubectl get pvc -n $NAMESPACE

# Performance metrics
echo "⚡ Performance Metrics:"
curl -s http://localhost:9090/api/v1/query?query=up{job="tokamak-rl"} | jq -r '.data.result[].value[1]' 2>/dev/null || echo "Prometheus not accessible"

echo "✅ Monitoring check completed"
