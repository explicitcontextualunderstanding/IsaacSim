#!/bin/bash
# Assemble Docker Image from S3 Build Artifacts
# Downloads pre-compiled build from S3, creates Docker image, pushes to GHCR
# Run on: Any CPU instance with Docker (Vultr, AWS, GCP, local)

set -euo pipefail

# Configuration
S3_BUCKET="${S3_BUCKET:-isaac-sim-6-0-dev}"
S3_PREFIX="${S3_PREFIX:-builds}"
BUILD_TAG="${BUILD_TAG:-latest}"
GHCR_IMAGE="${GHCR_IMAGE:-ghcr.io/explicitcontextualunderstanding/isaac-sim-6:latest}"
GITHUB_USER="${GITHUB_USER:-explicitcontextualunderstanding}"
WORKSPACE="${WORKSPACE:-/tmp/isaac-assemble-$(date +%s)}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[ASSEMBLE]${NC} $*"; }
success() { echo -e "${GREEN}✓${NC} $*"; }
warn() { echo -e "${YELLOW}⚠${NC} $*"; }
error() { echo -e "${RED}✗${NC} $*"; }

usage() {
    cat << 'EOF'
Assemble Docker Image from S3 Build
====================================

Downloads pre-compiled Isaac Sim build from S3 and creates runnable Docker image.

Usage: ./assemble_image.sh [OPTIONS]

Options:
    --build-tag TAG      Build tag to download (default: latest)
    --s3-bucket BUCKET   S3 bucket name (default: isaac-sim-6-0-dev)
    --s3-prefix PREFIX   S3 prefix path (default: builds)
    --ghcr-image IMAGE   Target GHCR image (default: ghcr.io/.../isaac-sim-6:latest)
    --github-user USER   GitHub username (default: explicitcontextualunderstanding)
    --skip-push          Build only, don't push to GHCR
    --keep-workspace     Keep workspace directory after build

Environment:
    AWS_ACCESS_KEY_ID     Required for S3 download
    AWS_SECRET_ACCESS_KEY Required for S3 download
    GITHUB_TOKEN          Required for GHCR push

Examples:
    # Assemble latest build
    ./assemble_image.sh

    # Assemble specific build
    ./assemble_image.sh --build-tag 20260321-143022

    # Build only, don't push
    ./assemble_image.sh --skip-push
EOF
}

# Parse arguments
SKIP_PUSH=false
KEEP_WORKSPACE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --build-tag)
            BUILD_TAG="$2"
            shift 2
            ;;
        --s3-bucket)
            S3_BUCKET="$2"
            shift 2
            ;;
        --s3-prefix)
            S3_PREFIX="$2"
            shift 2
            ;;
        --ghcr-image)
            GHCR_IMAGE="$2"
            shift 2
            ;;
        --github-user)
            GITHUB_USER="$2"
            shift 2
            ;;
        --skip-push)
            SKIP_PUSH=true
            shift
            ;;
        --keep-workspace)
            KEEP_WORKSPACE=true
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Image Assembly from S3 Build Artifacts                   ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# ============================================
# PHASE 1: Prerequisites
# ============================================
log "[1/5] Checking prerequisites..."

# Check Docker
if ! command -v docker &>/dev/null; then
    error "Docker not installed"
    exit 1
fi

if ! docker info &>/dev/null; then
    error "Docker daemon not accessible (try: sudo usermod -aG docker \$USER && newgrp docker)"
    exit 1
fi
success "Docker available"

# Check AWS credentials
if [ -z "${AWS_ACCESS_KEY_ID:-}" ] || [ -z "${AWS_SECRET_ACCESS_KEY:-}" ]; then
    error "AWS credentials not set (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)"
    exit 1
fi

# Test S3 access
if ! aws s3 ls "s3://${S3_BUCKET}/${S3_PREFIX}/" &>/dev/null; then
    error "Cannot access S3: s3://${S3_BUCKET}/${S3_PREFIX}/"
    exit 1
fi
success "S3 access verified: ${S3_BUCKET}/${S3_PREFIX}"

# Check GitHub token (if pushing)
if [ "$SKIP_PUSH" = false ] && [ -z "${GITHUB_TOKEN:-}" ]; then
    error "GITHUB_TOKEN not set (required for GHCR push)"
    log "Use --skip-push to build only"
    exit 1
fi

# ============================================
# PHASE 2: Download from S3
# ============================================
log "[2/5] Downloading build artifacts from S3..."

mkdir -p "$WORKSPACE"
cd "$WORKSPACE"

S3_URL="s3://${S3_BUCKET}/${S3_PREFIX}/isaac-sim-build-${BUILD_TAG}.tar.gz"
MANIFEST_URL="s3://${S3_BUCKET}/${S3_PREFIX}/manifest-${BUILD_TAG}.json"

log "Downloading: ${S3_URL}"
if ! aws s3 cp "$S3_URL" "isaac-sim-build.tar.gz"; then
    error "Failed to download build from S3"
    log "Available builds:"
    aws s3 ls "s3://${S3_BUCKET}/${S3_PREFIX}/" | grep "isaac-sim-build" | tail -10
    exit 1
fi

# Download manifest if exists
if aws s3 ls "$MANIFEST_URL" &>/dev/null; then
    aws s3 cp "$MANIFEST_URL" "manifest.json"
    log "Build info:"
    cat manifest.json | python3 -m json.tool 2>/dev/null || cat manifest.json
fi

BUILD_SIZE=$(stat -c%s "isaac-sim-build.tar.gz" 2>/dev/null || stat -f%z "isaac-sim-build.tar.gz" 2>/dev/null || echo "0")
BUILD_SIZE_MB=$((BUILD_SIZE / 1024 / 1024))
success "Downloaded: ${BUILD_SIZE_MB}MB"

# ============================================
# PHASE 3: Create Dockerfile
# ============================================
log "[3/5] Creating Dockerfile..."

cat > Dockerfile.assemble << 'DOCKERFILE'
# Isaac Sim 6.0 - Assembled from S3 Build
FROM ubuntu:24.04

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV OMNI_KIT_ALLOW_ROOT=1
ENV ACCEPT_EULA=Y
ENV PRIVACY_CONSENT=Y

# Install runtime dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libvulkan1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcursor1 \
    libxi6 \
    libxrandr2 \
    libxt6 \
    libglu1-mesa \
    libegl1 \
    libgomp1 \
    libatomic1 \
    libopenexr-dev \
    libtbb-dev \
    rapidjson-dev \
    libbenchmark-dev \
    libgtest-dev \
    libjson-c-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -ms /bin/bash -l --uid 1234 isaac-sim -d /isaac-sim

# Copy and extract build
COPY isaac-sim-build.tar.gz /tmp/
RUN mkdir -p /isaac-sim && \
    tar xzf /tmp/isaac-sim-build.tar.gz -C /isaac-sim && \
    rm /tmp/isaac-sim-build.tar.gz && \
    chown -R isaac-sim:isaac-sim /isaac-sim

# Set up environment
ENV ISAAC_SIM_PATH=/isaac-sim/_build/linux-x86_64/release
ENV PATH=${ISAAC_SIM_PATH}:${PATH}
ENV PYTHONPATH=${ISAAC_SIM_PATH}:${PYTHONPATH}
ENV MIN_DRIVER_VERSION=570.169

WORKDIR /isaac-sim
USER isaac-sim

# Create config directory
RUN mkdir -p /isaac-sim/.nvidia-omniverse/config

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD bash -c 'test -d ${ISAAC_SIM_PATH} || exit 1'

ENTRYPOINT ["/bin/bash"]
CMD ["-c", "echo 'Isaac Sim 6.0 ready. Use ./python.sh to start.' && sleep infinity"]
DOCKERFILE

success "Dockerfile created"

# ============================================
# PHASE 4: Build Image
# ============================================
log "[4/5] Building Docker image..."

# Build
docker build \
    -f Dockerfile.assemble \
    -t "${GHCR_IMAGE}" \
    -t "isaac-sim-6:local" \
    .

success "Image built: ${GHCR_IMAGE}"
docker images | grep -E "isaac-sim|${GHCR_IMAGE}" | head -5

# ============================================
# PHASE 5: Push to GHCR
# ============================================
if [ "$SKIP_PUSH" = true ]; then
    log "[5/5] Skipping GHCR push (--skip-push)"
    success "Image built locally: isaac-sim-6:local"
else
    log "[5/5] Pushing to GHCR..."

    # Login
    if ! echo "$GITHUB_TOKEN" | docker login ghcr.io -u "$GITHUB_USER" --password-stdin; then
        error "GHCR login failed"
        exit 1
    fi

    # Push
    if docker push "$GHCR_IMAGE"; then
        success "Pushed to GHCR: ${GHCR_IMAGE}"
    else
        error "Push failed"
        exit 1
    fi
fi

# ============================================
# Cleanup
# ============================================
if [ "$KEEP_WORKSPACE" = false ]; then
    log "Cleaning up workspace..."
    cd /
    rm -rf "$WORKSPACE"
    success "Workspace cleaned"
else
    log "Workspace preserved: ${WORKSPACE}"
fi

# ============================================
# Summary
# ============================================
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  IMAGE ASSEMBLY COMPLETE ✓                                 ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Build Tag: ${BUILD_TAG}"
echo "Source: s3://${S3_BUCKET}/${S3_PREFIX}/isaac-sim-build-${BUILD_TAG}.tar.gz"
echo "Image: ${GHCR_IMAGE}"
if [ "$SKIP_PUSH" = true ]; then
    echo "Status: Built locally (not pushed)"
else
    echo "Status: Pushed to GHCR"
fi
echo ""
echo "Next: Deploy on RunPod"
echo "  runpodctl create pod --image ${GHCR_IMAGE} ..."
echo ""
