#!/bin/bash
# img build script for Isaac Sim 6.0 container
# Usage: ./scripts/run_img_build.sh [--push] [--tar]
#
# img is a userspace container image builder that works without Docker daemon
# Download: https://github.com/genuinetools/img/releases
#
# Examples:
#   ./scripts/run_img_build.sh --push              # Build and push to GHCR
#   ./scripts/run_img_build.sh --tar isaac-sim-6.tar  # Build and save as tar for S3 upload

set -e

# Configuration
DOCKERFILE="tools/docker/Dockerfile"
CONTEXT="."
GHCR_ORG="${GHCR_ORG:-explicitcontextualunderstanding}"
IMAGE_NAME="${IMAGE_NAME:-isaac-sim-6}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
DESTINATION="ghcr.io/${GHCR_ORG}/${IMAGE_NAME}:${IMAGE_TAG}"
TAR_FILE=""

# Parse args
PUSH_FLAG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --push)
            PUSH_FLAG="1"
            shift
            ;;
        --tar)
            TAR_FILE="${2:-isaac-sim-6.tar}"
            shift 2
            ;;
        --org)
            GHCR_ORG="$2"
            DESTINATION="ghcr.io/${GHCR_ORG}/${IMAGE_NAME}:${IMAGE_TAG}"
            shift 2
            ;;
        --tag)
            IMAGE_TAG="$2"
            DESTINATION="ghcr.io/${GHCR_ORG}/${IMAGE_NAME}:${IMAGE_TAG}"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=== Isaac Sim 6.0 img Build ==="
echo "Destination: ${DESTINATION}"
echo "Dockerfile: ${DOCKERFILE}"
echo ""

# Check if img is available
if ! command -v img &> /dev/null; then
    echo "Installing img..."

    # Download img binary
    IMG_VERSION="v0.5.11"
    curl -sL "https://github.com/genuinetools/img/releases/download/${IMG_VERSION}/img-linux-amd64" \
        -o /usr/local/bin/img
    chmod +x /usr/local/bin/img
fi

echo "img version: $(img version 2>&1 | head -1)"

# Setup Docker config for GHCR
mkdir -p ~/.docker

if [ -n "${GH_PAT}" ]; then
    echo "Using GH_PAT for authentication"
    AUTH=$(echo -n "${GH_USER:-${GHCR_ORG}:${GH_PAT}" | base64)
    cat > ~/.docker/config.json <<EOF
{
    "auths": {
        "ghcr.io": {
            "auth": "${AUTH}"
        }
    }
}
EOF
else
    echo "Warning: GH_PAT not set - may not be able to push to GHCR"
    cat > ~/.docker/config.json <<EOF
{
    "auths": {}
}
EOF
fi

# Check CUDA version
echo ""
echo "CUDA Environment:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>/dev/null || echo "No GPU available"
echo ""

# Run img build
echo "Starting img build..."
echo "This may take 30-60 minutes depending on network and build cache..."
echo ""

if [ -n "${TAR_FILE}" ]; then
    # Build to local storage, then save as tar
    echo "Building to local storage..."
    img build \
        --dockerfile "${DOCKERFILE}" \
        --context "${CONTEXT}" \
        --destination "local://${IMAGE_NAME}:${IMAGE_TAG}" \
        --label "isaac-sim-version=6.0" \
        --label "built=$(date -Iseconds)"

    echo "Saving as tar archive..."
    img save "${IMAGE_NAME}:${IMAGE_TAG}" -o "${TAR_FILE}"

    echo ""
    echo "=== Build Complete ==="
    echo "Image saved to: ${TAR_FILE}"
    echo "Upload to S3 with: aws s3 cp ${TAR_FILE} s3://your-bucket/"
else
    # Build and push to registry
    img build \
        --dockerfile "${DOCKERFILE}" \
        --context "${CONTEXT}" \
        --destination "${DESTINATION}" \
        --label "isaac-sim-version=6.0" \
        --label "built=$(date -Iseconds)"

    if [ -n "${PUSH_FLAG}" ]; then
        echo ""
        echo "Push complete"
    fi

    echo ""
    echo "=== Build Complete ==="
    echo "Image: ${DESTINATION}"
fi
