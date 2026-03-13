#!/bin/bash
# Kaniko build script for Isaac Sim 6.0 container
# Usage: ./scripts/run_kaniko_build.sh [--push]

set -e

# Configuration
DOCKERFILE="tools/docker/Dockerfile"
CONTEXT="."
GHCR_ORG="${GHCR_ORG:-explicitcontextualunderstanding}"
IMAGE_NAME="${IMAGE_NAME:-isaac-sim-6}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
DESTINATION="ghcr.io/${GHCR_ORG}/${IMAGE_NAME}:${IMAGE_TAG}"

# Parse args
PUSH_FLAG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --push)
            PUSH_FLAG="--push"
            shift
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

echo "=== Isaac Sim 6.0 Kaniko Build ==="
echo "Destination: ${DESTINATION}"
echo "Dockerfile: ${DOCKERFILE}"
echo ""

# Check if Kaniko is available
if ! command -v /kaniko/executor &> /dev/null; then
    echo "Installing Kaniko..."

    # Download Kaniko
    KANIKO_VERSION="v1.23.0"
    wget -q -O /tmp/kaniko.tar.gz \
        "https://github.com/GoogleContainerTools/kaniko/releases/download/${KANIKO_VERSION}/kaniko_${KANIKO_VERSION}_linux_amd64.tar.gz"

    mkdir -p /usr/local/bin
    tar -xzf /tmp/kaniko.tar.gz -C /usr/local/bin
    rm /tmp/kaniko.tar.gz
fi

# Setup Docker config for GHCR
mkdir -p /kaniko/.docker

if [ -n "${GH_PAT}" ]; then
    echo "Using GH_PAT for authentication"
    AUTH=$(echo -n "${GH_USER:-${GHCR_ORG}}:${GH_PAT}" | base64)
    cat > /kaniko/.docker/config.json <<EOF
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
    cat > /kaniko/.docker/config.json <<EOF
{
    "auths": {}
}
EOF
fi

# Check CUDA version
echo "CUDA Environment:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo ""

# Run Kaniko build
echo "Starting Kaniko build..."
echo "This may take 30-60 minutes depending on network and build cache..."
echo ""

/kaniko/executor \
    --context "${CONTEXT}" \
    --dockerfile "${DOCKERFILE}" \
    --destination "${DESTINATION}" \
    --snapshot-mode=redo \
    --use-new-run \
    --compressed-caching=false \
    --build-arg CUDA_VERSION=12.6 \
    --build-arg UBUNTU_VERSION=24.04 \
    --build-arg COMPUTE_CAPABILITY=100 \
    ${PUSH_FLAG}

echo ""
echo "=== Build Complete ==="
echo "Image: ${DESTINATION}"
