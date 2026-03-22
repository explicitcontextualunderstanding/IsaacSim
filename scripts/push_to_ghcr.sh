#!/bin/bash
# Push Isaac Sim tar to GHCR as runnable OCI image
# Usage: ./scripts/push_to_ghcr.sh [tag]
# Requires: skopeo (installed via apt), GITHUB_TOKEN env var

set -e

TAG="${1:-runtime}"
TAR_FILE="/workspace/isaac-sim-6.tar"
GHCR_IMAGE="ghcr.io/explicitcontextualunderstanding/isaac-sim-6-${TAG}:latest"

echo "=== Isaac Sim GHCR Push Script ==="
echo "Target: ${GHCR_IMAGE}"

# Check prerequisites
if [ -z "$GITHUB_TOKEN" ]; then
    echo "❌ Error: GITHUB_TOKEN not set"
    echo "Set with: export GITHUB_TOKEN=ghp_xxxxxxxx"
    exit 1
fi

# Install AWS CLI if needed
if ! command -v aws &>/dev/null; then
    echo "📥 Installing AWS CLI..."
    apt-get update -qq
    apt-get install -y -qq curl unzip
    cd /tmp
    curl -sSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip -q awscliv2.zip
    ./aws/install --update
    rm -rf aws awscliv2.zip
    cd - >/dev/null
fi

if [ ! -f "$TAR_FILE" ]; then
    echo "❌ Error: Tar file not found at ${TAR_FILE}"
    echo "Downloading from S3..."
    aws s3 cp s3://isaac-sim-6-0-dev/isaac-sim-6.tar "$TAR_FILE"
fi

# Check tar size
TAR_SIZE=$(stat -c%s "$TAR_FILE" 2>/dev/null || stat -f%z "$TAR_FILE")
# Install coreutils for numfmt if on minimal system
if ! command -v numfmt &>/dev/null; then
    apt-get install -y -qq coreutils 2>/dev/null || true
fi
echo "📦 Tar size: $(numfmt --to=iec-i $TAR_SIZE 2>/dev/null || echo $TAR_SIZE bytes)"

# Install skopeo if needed
if ! command -v skopeo &>/dev/null; then
    echo "📥 Installing skopeo..."
    apt-get update -qq && apt-get install -y -qq skopeo
fi

# Login to GHCR via skopeo
echo "🔐 Authenticating to GHCR..."
skopeo login ghcr.io -u explicitcontextualunderstanding --password-stdin <<<"$GITHUB_TOKEN"

# Push tar as runnable OCI image
echo "🚀 Pushing to GHCR (this may take 10-15 min for 16GB)..."

# Create credentials file to avoid leaking token in process list
CREDS_FILE=$(mktemp)
trap 'rm -f "$CREDS_FILE"' EXIT
printf 'explicitcontextualunderstanding:%s\n' "$GITHUB_TOKEN" >"$CREDS_FILE"
chmod 600 "$CREDS_FILE"

skopeo copy \
    docker-archive:"$TAR_FILE" \
    docker://"$GHCR_IMAGE" \
    --dest-creds-file="$CREDS_FILE"

echo "✅ Push complete!"
echo ""
echo "Update your RunPod template to use:"
echo "  ${GHCR_IMAGE}"
echo ""
echo "Test with:"
echo "  docker pull ${GHCR_IMAGE}"
echo "  docker run --rm ${GHCR_IMAGE} nvidia-smi"
