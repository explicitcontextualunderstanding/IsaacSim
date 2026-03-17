#!/bin/bash
# Isaac Sim 6.0 Container Pull Script
# Usage: ./scripts/pull_from_ghcr.sh
#
# Pulls the Isaac Sim 6.0 container from GHCR and runs it.

set -e

GHCR_IMAGE="ghcr.io/explicitcontextualunderstanding/isaac-sim-6-dev:latest"

echo "=== Pulling Isaac Sim 6.0-dev from GHCR ==="
echo "Image: ${GHCR_IMAGE}"
echo ""

# Check if logged in to GHCR
if ! docker login ghcr.io &>/dev/null; then
    echo "Not logged in to GHCR. Logging in..."
    if [ -n "$GH_TOKEN" ]; then
        echo "${GH_TOKEN}" | docker login ghcr.io -u explicitcontextualunderstanding --password-stdin
    else
        echo "❌ Error: GH_TOKEN not found in environment."
        exit 1
    fi
fi

# Pull the image
echo "Pulling image..."
docker pull "${GHCR_IMAGE}"

echo ""
echo "=== Image Pulled Successfully ==="
echo ""
echo "To run headless with self-healing validation:"
echo "  docker run --gpus all --network host \\"
echo "    -e ACCEPT_EULA=Y \\"
echo "    -e PRIVACY_CONSENT=Y \\"
echo "    -e HEADLESS=1 \\"
echo "    -e ENABLE_CAMERAS=1 \\"
echo "    -e OMNI_KIT_IP=0.0.0.0 \\"
echo "    ${GHCR_IMAGE} \\"
echo "    bash -c \"/workspace/IsaacSim/scripts/validate_container.sh && sleep infinity\""
echo ""
echo "To access streaming UI (noVNC):"
echo "  Connect to: https://<your-pod-ip>.runpod.io:6080/vnc.html?resize=scale"
