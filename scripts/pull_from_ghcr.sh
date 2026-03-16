#!/bin/bash
# Isaac Sim 6.0 Container Pull Script
# Usage: ./scripts/pull_from_ghcr.sh
#
# Pulls the Isaac Sim 6.0 container from GHCR and runs it.

set -e

GHCR_IMAGE="ghcr.io/explicitcontextualunderstanding/isaac-sim-6:latest"

echo "=== Pulling Isaac Sim 6.0 from GHCR ==="
echo "Image: ${GHCR_IMAGE}"
echo ""

# Check if logged in to GHCR
if ! docker ghcr.io &>/dev/null; then
    echo "Not logged in to GHCR. Logging in..."
    echo "${GH_TOKEN}" | docker login ghcr.io -u explicitcontextualunderstanding --password-stdin
fi

# Pull the image
echo "Pulling image..."
docker pull "${GH_IMAGE}"

echo ""
echo "=== Image Pulled Successfully ==="
echo ""
echo "To run headless:"
echo "  docker run --gpus all \\"
echo "    -e ACCEPT_EULA=Y \\"
echo "    -e PRIVACY_CONSENT=Y \\"
echo "    -e LIVESTREAM=2 \\"
echo "    -e HEADLESS=1 \\"
echo "    -e OMNI_KIT_IP=0.0.0.0 \\"
echo "    ${GH_IMAGE}"
