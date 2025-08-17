#!/bin/bash
set -e

echo "🐳 Building Tokamak-RL Docker image..."
docker build -t tokamak-rl:v6.0 -f deployment_v6/docker/Dockerfile .

echo "🏷️  Tagging image..."
docker tag tokamak-rl:v6.0 tokamak-rl:latest

echo "✅ Build completed successfully!"
