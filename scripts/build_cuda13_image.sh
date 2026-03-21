#!/bin/bash
# Build and push Isaac Sim 6.0 CUDA 13.1+ base image to GHCR
# Usage: ./scripts/build_cuda13_image.sh [tag]

set -euo pipefail

TAG="${1:-cuda13.1-base}"
IMAGE_NAME="ghcr.io/explicitcontextualunderstanding/isaac-sim-6-${TAG}"
DOCKERFILE="Dockerfile.cuda13"

echo "=== Building Isaac Sim 6.0 CUDA 13.1+ Image ==="
echo "Image: ${IMAGE_NAME}:latest"
echo "Dockerfile: ${DOCKERFILE}"
echo ""

# Check prerequisites
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found"
    exit 1
fi

# Check if we can access Docker
if ! docker info &> /dev/null; then
    echo "❌ Docker daemon not accessible"
    exit 1
fi

# Check for GHCR auth
if [ -z "${GITHUB_TOKEN:-}" ]; then
    echo "⚠️  GITHUB_TOKEN not set - will try to use existing docker auth"
fi

echo "📦 Building image..."
docker build -f "${DOCKERFILE}" -t "${IMAGE_NAME}:latest" -t "${IMAGE_NAME}:$(date +%Y%m%d)" .

echo ""
echo "🔐 Authenticating to GHCR..."
if [ -n "${GITHUB_TOKEN:-}" ]; then
    echo "${GITHUB_TOKEN}" | docker login ghcr.io -u explicitcontextualunderstanding --password-stdin
else
    echo "⚠️  Using existing docker auth (if any)"
fi

echo ""
echo "🚀 Pushing to GHCR..."
docker push "${IMAGE_NAME}:latest"
docker push "${IMAGE_NAME}:$(date +%Y%m%d)"

echo ""
echo "✅ Build and push complete!"
echo ""
echo "Images available:"
echo "  ${IMAGE_NAME}:latest"
echo "  ${IMAGE_NAME}:$(date +%Y%m%d)"
echo ""
echo "Update your RunPod template with:"
echo "  ${IMAGE_NAME}:latest"
