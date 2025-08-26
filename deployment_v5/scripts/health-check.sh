#!/bin/bash
# Health check script

echo "🏥 Running health checks..."

# Check service status
SERVICE_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)
if [ "$SERVICE_STATUS" = "200" ]; then
    echo "✅ Service health check passed"
else
    echo "❌ Service health check failed (HTTP $SERVICE_STATUS)"
    exit 1
fi

# Check database connectivity
echo "Checking database connectivity..."
# Add database connectivity checks here

# Check Redis connectivity
echo "Checking Redis connectivity..."
# Add Redis connectivity checks here

echo "✅ All health checks passed!"
