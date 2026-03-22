#!/bin/bash
# RunPod GPU Build Script
# Compile Isaac Sim source and upload to S3
# Run on: RunPod GPU instance (L40S, 4090, etc.)

set -euo pipefail

ISAAC_REPO="https://github.com/explicitcontextualunderstanding/IsaacSim.git"
WORKSPACE="/workspace"
ISAAC_DIR="${WORKSPACE}/IsaacSim"
BUILD_TAG="${BUILD_TAG:-$(date +%Y%m%d-%H%M%S)}"
S3_BUCKET="${S3_BUCKET:-isaac-sim-6-0-dev}"
S3_PREFIX="${S3_PREFIX:-builds}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[RUNPOD-BUILD]${NC} $*"; }
success() { echo -e "${GREEN}✓${NC} $*"; }
warn() { echo -e "${YELLOW}⚠${NC} $*"; }
error() { echo -e "${RED}✗${NC} $*"; }

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  RunPod GPU Build - Isaac Sim 6.0                        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# ============================================
# CHECK: RunPod Environment
# ============================================
log "Checking RunPod environment..."

if [ -z "${RUNPOD_POD_ID:-}" ]; then
    warn "Not running on RunPod (no RUNPOD_POD_ID)"
else
    log "RunPod Pod ID: $RUNPOD_POD_ID"
fi

if ! command -v nvidia-smi &>/dev/null; then
    error "nvidia-smi not found - GPU not available"
    exit 1
fi

GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1)
success "GPU: $GPU_INFO"

# ============================================
# PHASE 1: Setup
# ============================================
log "[1/5] Setting up environment..."

# Install dependencies
apt-get update -qq && apt-get install -y -qq \
    git git-lfs gcc-11 g++-11 awscli &&
    rm -rf /var/lib/apt/lists/*

# Set GCC 11
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 110
success "GCC 11 configured"

# Check AWS credentials
if [ -z "${AWS_ACCESS_KEY_ID:-}" ]; then
    error "AWS_ACCESS_KEY_ID not set"
    exit 1
fi

# Verify S3 access
if ! aws s3 ls "s3://${S3_BUCKET}/" &>/dev/null; then
    error "Cannot access S3 bucket: ${S3_BUCKET}"
    exit 1
fi
success "S3 access verified"

# ============================================
# PHASE 2: Clone
# ============================================
log "[2/5] Cloning repository..."

cd "$WORKSPACE"

if [ -d "$ISAAC_DIR/.git" ]; then
    log "Updating existing repo..."
    cd "$ISAAC_DIR"
    git fetch origin
    git reset --hard origin/main
else
    log "Cloning Isaac Sim..."
    git clone "$ISAAC_REPO" "$ISAAC_DIR"
    cd "$ISAAC_DIR"
fi

# Initialize submodules
git submodule update --init --recursive 2>/dev/null || warn "No submodules"

# Git LFS
git lfs install
if ! git lfs pull; then
    LFS_EXIT=$?
    error "Git LFS pull failed with exit code $LFS_EXIT"
    warn "Build may proceed but assets will be missing"
    warn "Manual fix: cd $ISAAC_DIR && git lfs pull"
fi

success "Repository ready at $ISAAC_DIR"

# ============================================
# PHASE 3: Build
# ============================================
log "[3/5] Building Isaac Sim (this takes 20-30 minutes)..."

cd "$ISAAC_DIR"

# Accept EULA
touch .eula_accepted

# Make scripts executable
chmod +x build.sh repo.sh 2>/dev/null || true

# Configure for parallel build (use all GPUs)
export CMAKE_BUILD_PARALLEL_LEVEL=36

# Build
log "Starting release build..."
if [ -f build.sh ]; then
    ./build.sh --release 2>&1 | tee "${WORKSPACE}/build.log"
else
    error "build.sh not found"
    exit 1
fi

success "Build complete"

# ============================================
# PHASE 4: Package
# ============================================
log "[4/5] Packaging build artifacts..."

mkdir -p "${WORKSPACE}/artifacts"

# Create tarball (exclude large unnecessary files)
tar czf "${WORKSPACE}/artifacts/isaac-sim-build-${BUILD_TAG}.tar.gz" \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='_build/packages/temp' \
    -C "$ISAAC_DIR" \
    _build/linux-x86_64/release/ 2>&1 | tee -a "${WORKSPACE}/build.log"
TAR_EXIT=${PIPESTATUS[0]}
if [ "$TAR_EXIT" -ne 0 ]; then
    error "tar failed with exit code $TAR_EXIT"
    exit 1
fi

BUILD_SIZE=$(stat -c%s "${WORKSPACE}/artifacts/isaac-sim-build-${BUILD_TAG}.tar.gz" 2>/dev/null || echo "0")
BUILD_SIZE_MB=$((BUILD_SIZE / 1024 / 1024))

success "Package created: ${BUILD_SIZE_MB}MB"

# Create manifest
cat >"${WORKSPACE}/artifacts/manifest-${BUILD_TAG}.json" <<MANIFEST
{
    "build_tag": "${BUILD_TAG}",
    "timestamp": "$(date -Iseconds)",
    "git_commit": "$(git rev-parse HEAD)",
    "git_branch": "$(git branch --show-current)",
    "size_bytes": ${BUILD_SIZE},
    "size_mb": ${BUILD_SIZE_MB},
    "gpu_info": "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)",
    "cuda_version": "$(nvcc --version | grep release | awk '{print $6}' | tr -d ',')",
    "gcc_version": "$(gcc -dumpversion)"
}
MANIFEST

success "Manifest created"

# ============================================
# PHASE 5: Upload
# ============================================
log "[5/5] Uploading to S3..."

# Upload build
aws s3 cp "${WORKSPACE}/artifacts/isaac-sim-build-${BUILD_TAG}.tar.gz" \
    "s3://${S3_BUCKET}/${S3_PREFIX}/isaac-sim-build-${BUILD_TAG}.tar.gz"

# Upload manifest
aws s3 cp "${WORKSPACE}/artifacts/manifest-${BUILD_TAG}.json" \
    "s3://${S3_BUCKET}/${S3_PREFIX}/manifest-${BUILD_TAG}.json"

# Update "latest" pointer
aws s3 cp "${WORKSPACE}/artifacts/isaac-sim-build-${BUILD_TAG}.tar.gz" \
    "s3://${S3_BUCKET}/${S3_PREFIX}/isaac-sim-build-latest.tar.gz"

success "Upload complete"

# ============================================
# SUMMARY
# ============================================
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  RUNPOD GPU BUILD COMPLETE ✓                               ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Build Tag: ${BUILD_TAG}"
echo "Size: ${BUILD_SIZE_MB}MB"
echo "S3 Location: s3://${S3_BUCKET}/${S3_PREFIX}/isaac-sim-build-${BUILD_TAG}.tar.gz"
echo ""
echo "Next: Run image assembly on external CPU instance"
echo "  ./scripts/assemble_image.sh --build-tag ${BUILD_TAG}"
echo ""
