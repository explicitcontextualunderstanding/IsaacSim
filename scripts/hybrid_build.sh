#!/bin/bash
# Hybrid Build Orchestrator: GPU Build → S3 → CPU Reassembly
# Usage: ./scripts/hybrid_build.sh [gpu-build|cpu-assemble|all]

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Defaults
S3_BUCKET="${S3_BUCKET:-isaac-sim-6-0-dev}"
S3_PREFIX="${S3_PREFIX:-builds}"
BUILD_TAG="${BUILD_TAG:-$(date +%Y%m%d-%H%M%S)}"
GHCR_IMAGE="${GHCR_IMAGE:-ghcr.io/explicitcontextualunderstanding/isaac-sim-6:latest}"
GITHUB_USER="${GITHUB_USER:-explicitcontextualunderstanding}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[HYBRID-BUILD]${NC} $*"; }
success() { echo -e "${GREEN}✓${NC} $*"; }
warn() { echo -e "${YELLOW}⚠${NC} $*"; }
error() { echo -e "${RED}✗${NC} $*"; }

usage() {
    cat <<'EOF'
Hybrid Build Orchestrator
=========================

Build Isaac Sim using GPU→S3→CPU workflow for RunPod compatibility.

Usage: ./scripts/hybrid_build.sh [COMMAND] [OPTIONS]

Commands:
  gpu-build       Phase 1: Compile on RunPod GPU, upload to S3
  cpu-assemble    Phase 2: Reassemble on external CPU CPU, push to GHCR
  all             Run full workflow (default)

Options:
  --s3-bucket     S3 bucket name (default: isaac-sim-6-0-dev)
  --s3-prefix     S3 prefix path (default: builds)
  --build-tag     Build tag/version (default: timestamp)
  --ghcr-image    Target GHCR image (default: ghcr.io/.../isaac-sim-6:latest)
  --skip-preflight Skip preflight checks

Environment:
  RUNPOD_API_KEY      RunPod API key
  AWS_ACCESS_KEY_ID   AWS credentials
  GITHUB_TOKEN        GitHub token for GHCR push

Examples:
  # Full workflow
  ./scripts/hybrid_build.sh all

  # Just GPU build phase
  ./scripts/hybrid_build.sh gpu-build --build-tag v6.0.0-rc22

  # CPU reassembly with specific build
  ./scripts/hybrid_build.sh cpu-assemble --build-tag 20260321-143022
EOF
}

# ============================================
# PHASE 1: GPU Build on RunPod
# ============================================
run_gpu_build() {
    log "=========================================="
    log "PHASE 1: GPU Build on RunPod"
    log "=========================================="

    # Check prerequisites
    if [ -z "${RUNPOD_API_KEY:-}" ]; then
        error "RUNPOD_API_KEY not set"
        exit 1
    fi

    if [ -z "${AWS_ACCESS_KEY_ID:-}" ] || [ -z "${AWS_SECRET_ACCESS_KEY:-}" ]; then
        error "AWS credentials not set"
        exit 1
    fi

    log "Provisioning RunPod GPU instance..."
    log "  GPU: 4x L40S (parallel compilation)"
    log "  Duration: ~25 minutes"
    log "  Estimated cost: ~$2.50"

    # Create build script for RunPod
    BUILD_SCRIPT=$(
        cat <<'BUILD_EOF'
#!/bin/bash
set -euo pipefail

cd /workspace

# Clone if needed
if [ ! -d IsaacSim/.git ]; then
    git clone https://github.com/explicitcontextualunderstanding/IsaacSim.git
fi

cd IsaacSim
git fetch origin
git checkout main
git pull origin main

# Configure build
export CMAKE_BUILD_PARALLEL_LEVEL=36
export OMNI_KIT_ACCEPT_EULA=YES

# Install dependencies
apt-get update && apt-get install -y gcc-11 g++-11 git-lfs awscli

# Set GCC 11
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 110

# Pull LFS assets
git lfs install
git lfs pull

# Build
mkdir -p /workspace/build-artifacts
./build.sh --release 2>&1 | tee /workspace/build-artifacts/build.log

# Package
log "Packaging build artifacts..."
tar czf /workspace/build-artifacts/isaac-sim-build.tar.gz \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    _build/linux-x86_64/release/

# Upload to S3
log "Uploading to S3..."
aws s3 cp /workspace/build-artifacts/isaac-sim-build.tar.gz \
s3://S3_BUCKET_PLACEHOLDER/S3_PREFIX_PLACEHOLDER/isaac-sim-build-BUILD_TAG_PLACEHOLDER.tar.gz

# Upload latest symlink
aws s3 cp /workspace/build-artifacts/isaac-sim-build.tar.gz \
s3://S3_BUCKET_PLACEHOLDER/S3_PREFIX_PLACEHOLDER/isaac-sim-build-latest.tar.gz

# Upload manifest
cat > /workspace/build-artifacts/manifest.json << MANIFEST
{
    "build_tag": "${BUILD_TAG}",
    "timestamp": "$(date -Iseconds)",
    "git_commit": "$(git rev-parse HEAD)",
    "size_bytes": $(stat -c%s /workspace/build-artifacts/isaac-sim-build.tar.gz),
    "gpu_info": "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
}
MANIFEST

aws s3 cp /workspace/build-artifacts/manifest.json \
s3://S3_BUCKET_PLACEHOLDER/S3_PREFIX_PLACEHOLDER/manifest-BUILD_TAG_PLACEHOLDER.json

echo "GPU Build Complete!"
BUILD_EOF
    )

    # Replace placeholders with actual values
    BUILD_SCRIPT="${BUILD_SCRIPT//S3_BUCKET_PLACEHOLDER/${S3_BUCKET}}"
    BUILD_SCRIPT="${BUILD_SCRIPT//S3_PREFIX_PLACEHOLDER/${S3_PREFIX}}"
    BUILD_SCRIPT="${BUILD_SCRIPT//BUILD_TAG_PLACEHOLDER/${BUILD_TAG}}"

    # Deploy to RunPod
    log "Creating RunPod build instance..."

    # Save build script
    echo "$BUILD_SCRIPT" >/tmp/runpod-build-script.sh

    log "Build script prepared. To execute:"
    log "  1. Provision RunPod: 4x L40S, 200GB disk"
    log "  2. Upload and run: /tmp/runpod-build-script.sh"
    log "  3. Monitor S3 for: s3://${S3_BUCKET}/${S3_PREFIX}/"

    success "GPU build phase prepared"
    return 0
}

# ============================================
# PHASE 2: CPU Reassembly on external CPU
# ============================================
run_cpu_assemble() {
    log "=========================================="
    log "PHASE 2: CPU Reassembly on external CPU"
    log "=========================================="

    # Check for assemble_image.sh
    ASSEMBLE_SCRIPT="${SCRIPT_DIR}/assemble_image.sh"
    if [ ! -f "$ASSEMBLE_SCRIPT" ]; then
        error "assemble_image.sh not found at ${ASSEMBLE_SCRIPT}"
        log "Clone the full repo to get all scripts"
        exit 1
    fi

    log "Executing assemble_image.sh..."
    log "  Build Tag: ${BUILD_TAG}"
    log "  S3 Bucket: ${S3_BUCKET}"
    log "  GHCR Image: ${GHCR_IMAGE}"

    # Run the assembly script
    bash "$ASSEMBLE_SCRIPT" \
        --build-tag "${BUILD_TAG}" \
        --s3-bucket "${S3_BUCKET}" \
        --s3-prefix "${S3_PREFIX}" \
        --ghcr-image "${GHCR_IMAGE}" \
        --github-user "${GITHUB_USER}"

    success "CPU reassembly complete"
    return 0
}

# ============================================
# MAIN
# ============================================
main() {
    COMMAND="${1:-all}"

    case $COMMAND in
    gpu-build)
        run_gpu_build
        ;;
    cpu-assemble)
        run_cpu_assemble
        ;;
    all)
        run_gpu_build && run_cpu_assemble
        log ""
        log "=========================================="
        log "FULL WORKFLOW CONFIGURED"
        log "=========================================="
        log ""
        log "Execute in order:"
        log "  1. Run 'gpu-build' on RunPod GPU"
        log "  2. Verify S3 upload complete"
        log "  3. Run 'cpu-assemble' on external CPU (external CPU/AWS/GCP/Local)"
        log "  4. Push to GHCR"
        ;;
    help | --help | -h)
        usage
        exit 0
        ;;
    *)
        error "Unknown command: $COMMAND"
        usage
        exit 1
        ;;
    esac
}

main "$@"
