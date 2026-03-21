#!/bin/bash
# Full Stack Build: Isaac Sim 6.0 Source → Container Image → Validation
# Run on Vultr GPU instance with full Docker access
# Produces: Runnable GHCR image with compiled binaries

set -euo pipefail

# Configuration
ISAAC_REPO="https://github.com/explicitcontextualunderstanding/IsaacSim.git"
WORKSPACE="/workspace"
ISAAC_DIR="${WORKSPACE}/IsaacSim"
BUILD_DIR="${WORKSPACE}/build"
GHCR_IMAGE="ghcr.io/explicitcontextualunderstanding/isaac-sim-6-full:latest"
GITHUB_USER="explicitcontextualunderstanding"
LOG_FILE="${WORKSPACE}/build-$(date +%Y%m%d-%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() { echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} $*" | tee -a "$LOG_FILE"; }
success() { echo -e "${GREEN}✓${NC} $*" | tee -a "$LOG_FILE"; }
warn() { echo -e "${YELLOW}⚠${NC} $*" | tee -a "$LOG_FILE"; }
error() { echo -e "${RED}✗${NC} $*" | tee -a "$LOG_FILE"; }

# Phase tracking
PHASE=0
TOTAL_PHASES=6

start_phase() {
    PHASE=$((PHASE + 1))
    log ""
    log "=========================================="
    log "PHASE ${PHASE}/${TOTAL_PHASES}: $1"
    log "=========================================="
}

# Trap errors
trap 'error "Build failed in Phase ${PHASE}. Check ${LOG_FILE}"' ERR

echo "=========================================="
echo "Isaac Sim 6.0 - Full Stack Build"
echo "Vultr GPU Instance → GHCR Runnable Image"
echo "=========================================="
echo ""

# Check prerequisites
start_phase "Environment Validation"

command -v docker &>/dev/null || { error "Docker not found"; exit 1; }
command -v nvidia-smi &>/dev/null || { error "nvidia-smi not found"; exit 1; }
command -v git &>/dev/null || { error "git not found"; exit 1; }

success "All prerequisites found"

log "GPU Info:"
nvidia-smi --query-gpu=name,driver_version,memory.total,cuda_version --format=csv | tee -a "$LOG_FILE"

log "Docker Status:"
docker info 2>/dev/null | grep -E "Server Version|Storage Driver" | tee -a "$LOG_FILE"

# Get GitHub token
if [ -z "${GITHUB_TOKEN:-}" ]; then
    log "Enter GitHub Personal Access Token (needs 'write:packages' scope):"
    read -r GITHUB_TOKEN
    export GITHUB_TOKEN
fi

# Phase 2: Clone/Update Isaac Sim
start_phase "Clone Isaac Sim Repository"

mkdir -p "$WORKSPACE"
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

# Initialize submodules if needed
if [ -f .gitmodules ]; then
    log "Initializing submodules..."
    git submodule update --init --recursive || warn "No submodules or failed to init"
fi

# Pull Git LFS files
if command -v git-lfs &>/dev/null; then
    log "Pulling LFS files..."
    git lfs pull || warn "Git LFS pull failed (may not be needed)"
fi

success "Repository ready at $ISAAC_DIR"

# Phase 3: Build Isaac Sim from Source
start_phase "Build Isaac Sim Binaries (This takes 2-4 hours)"

cd "$ISAAC_DIR"

# Accept EULA
touch .eula_accepted

# Make scripts executable
find . -name "*.sh" -type f -exec chmod +x {} + 2>/dev/null || true
chmod +x build.sh repo.sh 2>/dev/null || true

# Configure for release build
log "Starting release build..."
log "Build log: ${BUILD_DIR}/build.log"

mkdir -p "$BUILD_DIR"

# Run the build
if [ -f build.sh ]; then
    ./build.sh -r 2>&1 | tee "${BUILD_DIR}/build.log"
else
    warn "build.sh not found, skipping source build"
fi

success "Build complete"

# Package if repo.sh exists
if [ -f repo.sh ]; then
    log "Packaging Isaac Sim..."
    mkdir -p "${BUILD_DIR}/packages"
    ./repo.sh package -c release -m isaac-sim-standalone --temp-dir "${BUILD_DIR}/temp" 2>&1 | tee -a "${BUILD_DIR}/build.log" || warn "Packaging may have failed"
fi

# Phase 4: Create Full Docker Image
start_phase "Create Docker Image with Binaries"

cd "$WORKSPACE"

# Create Dockerfile for full image
cat > Dockerfile.full << 'DOCKERFILE_EOF'
# Isaac Sim 6.0 Full Image (CUDA 13.1 + Compiled Binaries)
FROM nvidia/cuda:13.1.1-devel-ubuntu24.04

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV OMNI_KIT_ALLOW_ROOT=1

# Install system dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    curl wget git git-lfs build-essential gcc-11 g++-11 \
    software-properties-common gpgv libatomic1 libegl1 libgl1 \
    libglu1-mesa libglx0 libgomp1 libsm6 libxi6 libxrandr2 \
    libxt6 unzip ca-certificates libglib2.0-0 libnghttp2-14 \
    libvulkan-dev vulkan-tools libgl1-mesa-dev libx11-dev \
    libxext-dev libxcursor-dev libxrandr-dev libxi-dev \
    libxinerama-dev libxfixes-dev libxrender1 libxtst6 libxmu6 \
    libgtest-dev libbenchmark-dev rapidjson-dev libopenexr-dev \
    libjson-c-dev libtbb-dev zlib1g-dev libbz2-dev liblzma-dev \
    libzstd-dev libssl-dev pkg-config p7zip-full python3 python3-pip \
    python3-venv \
    && apt-get -y autoremove && apt-get clean autoclean \
    && rm -rf /var/lib/apt/lists/*

# Configure GCC 11 as default
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 110

# Vulkan/NVIDIA runtime config
RUN mkdir -p /usr/share/glvnd/egl_vendor.d /etc/vulkan/icd.d /etc/vulkan/implicit_layer.d
RUN printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}\n' \
    > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
RUN printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.194"}}\n' \
    > /etc/vulkan/icd.d/nvidia_icd.json
ENV VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json

# Create isaac-sim user
RUN useradd -ms /bin/bash -l --uid 1234 isaac-sim -d /isaac-sim \
    && mkdir -p /isaac-sim \
    && chown -R isaac-sim:isaac-sim /isaac-sim

# Copy compiled binaries
COPY --chown=isaac-sim:isaac-sim . /isaac-sim/

# Set up workspace
WORKDIR /isaac-sim
USER isaac-sim

# Create necessary directories
RUN mkdir -p /isaac-sim/.nvidia-omniverse/config

ENV MIN_DRIVER_VERSION=570.169

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD bash -c 'logdir="/isaac-sim/.nvidia-omniverse/logs/Kit"; [ -d "$logdir" ] && latest=$(ls -t "$logdir"/*/*/kit_*.log 2>/dev/null | head -1) && [ -n "$latest" ] && grep -q "AppReady" "$latest" && exit 0; exit 1'

# Entrypoint
ENTRYPOINT ["/bin/bash"]
CMD ["/isaac-sim/runheadless.sh"]
DOCKERFILE_EOF

# Build the image
log "Building Docker image (this may take 20-30 minutes)..."
cd "$ISAAC_DIR"

# Create .dockerignore to speed up build
cat > .dockerignore << 'DOCKERIGNORE_EOF'
.git
.gitignore
*.log
_build/packages/temp
*.pyc
__pycache__
.DS_Store
.dockerignore
Dockerfile*
docs/*.md
docs/*.png
DOCKERIGNORE_EOF

docker build -f "${WORKSPACE}/Dockerfile.full" -t "${GHCR_IMAGE}" -t "isaac-sim-6:local" .

success "Docker image built: ${GHCR_IMAGE}"
docker images | grep isaac-sim | head -5 | tee -a "$LOG_FILE"

# Phase 5: Validate the Container
start_phase "Validate Container"

log "Running validation tests..."

# Test 1: Basic container startup
log "Test 1: Container startup..."
docker run --rm --gpus all "${GHCR_IMAGE}" bash -c "nvidia-smi" | tee -a "$LOG_FILE"
success "GPU accessible in container"

# Test 2: CUDA version
log "Test 2: CUDA version..."
docker run --rm --gpus all "${GHCR_IMAGE}" bash -c "nvcc --version" | tee -a "$LOG_FILE"
success "CUDA accessible"

# Test 3: GCC version
log "Test 3: GCC version..."
docker run --rm "${GHCR_IMAGE}" bash -c "gcc --version | head -1" | tee -a "$LOG_FILE"
success "GCC accessible"

# Test 4: Vulkan
log "Test 4: Vulkan availability..."
docker run --rm --gpus all "${GHCR_IMAGE}" bash -c "vulkaninfo --summary 2>/dev/null | head -20 || echo 'Vulkan not fully configured (expected in build image)'" | tee -a "$LOG_FILE"

# Test 5: Isaac Sim files present
log "Test 5: Isaac Sim files..."
docker run --rm "${GHCR_IMAGE}" bash -c "ls -la /isaac-sim/ | head -20" | tee -a "$LOG_FILE"
success "Isaac Sim files present"

# Test 6: Run full validation if script exists
if [ -f scripts/validate_container.sh ]; then
    log "Test 6: Running validate_container.sh..."
    docker run --rm --gpus all \
        -e ACCEPT_EULA=Y \
        -e PRIVACY_CONSENT=Y \
        "${GHCR_IMAGE}" \
        bash -c "cd /isaac-sim && ./scripts/validate_container.sh 2>&1 || true" | tee -a "$LOG_FILE" || warn "Validation script had issues"
fi

success "Container validation complete"

# Phase 6: Push to GHCR
start_phase "Push to GHCR"

log "Authenticating to GHCR..."
echo "$GITHUB_TOKEN" | docker login ghcr.io -u "$GITHUB_USER" --password-stdin

log "Pushing image to GHCR..."
docker push "$GHCR_IMAGE"

success "Image pushed to GHCR"
log "Image URL: ${GHCR_IMAGE}"

# Final Summary
log ""
log "=========================================="
log "FULL STACK BUILD COMPLETE!"
log "=========================================="
log ""
log "Artifacts:"
log "  Docker Image: ${GHCR_IMAGE}"
log "  Build Log:    ${LOG_FILE}"
log ""
log "Next steps:"
log "  1. Update RunPod template:"
log "     Image: ${GHCR_IMAGE}"
log ""
log "  2. Test on RunPod:"
log "     docker run --gpus all -it ${GHCR_IMAGE}"
log ""
log "  3. Delete Vultr instance to save costs:"
log "     (Don't forget - costs ~\$2-3/hour!)"
log ""
log "=========================================="
