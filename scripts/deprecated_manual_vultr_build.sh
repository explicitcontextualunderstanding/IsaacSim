#!/bin/bash
# Manual build script for Vultr GPU instance
# Run this on your Vultr instance with Docker access

set -euo pipefail

GHCR_IMAGE="ghcr.io/explicitcontextualunderstanding/isaac-sim-6-cuda13.1-base:latest"
GITHUB_USER="explicitcontextualunderstanding"

echo "=== Isaac Sim 6.0 CUDA 13.1+ Base Image Build ==="
echo ""

# Check prerequisites
echo "[1/6] Checking prerequisites..."
command -v docker &> /dev/null || { echo "❌ Docker not found"; exit 1; }
command -v nvidia-smi &> /dev/null || { echo "❌ nvidia-smi not found"; exit 1; }
docker info &> /dev/null || { echo "❌ Docker daemon not running"; exit 1; }

# Check CUDA version
echo ""
echo "[2/6] Checking CUDA version..."
nvcc --version | head -4 || echo "⚠️ nvcc not found - base image will install CUDA"
nvidia-smi | head -10

# Get GitHub token
echo ""
echo "[3/6] Setting up GitHub authentication..."
if [ -z "${GITHUB_TOKEN:-}" ]; then
    echo "Enter your GitHub Personal Access Token (needs 'write:packages' scope):"
    read -r GITHUB_TOKEN
    export GITHUB_TOKEN
fi

echo "🔐 Logging into GHCR..."
echo "$GITHUB_TOKEN" | docker login ghcr.io -u "$GITHUB_USER" --password-stdin

# Download Dockerfile
echo ""
echo "[4/6] Downloading Dockerfile..."
cat > Dockerfile.cuda13 << 'DOCKERFILE_EOF'
# Isaac Sim 6.0 Container with CUDA 13.1+
FROM nvidia/cuda:13.1.1-devel-ubuntu24.04

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

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
    && apt-get -y autoremove && apt-get clean autoclean \
    && rm -rf /var/lib/apt/lists/*

# Configure GCC 11 as default (Isaac Sim 6.0 requirement)
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 110

# Verify CUDA
RUN nvcc --version && gcc --version

# Vulkan/NVIDIA runtime config
RUN mkdir -p /usr/share/glvnd/egl_vendor.d /etc/vulkan/icd.d
RUN printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}\n' \
    > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
RUN printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.194"}}\n' \
    > /etc/vulkan/icd.d/nvidia_icd.json
ENV VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json

# Create non-root user
RUN useradd -ms /bin/bash -l --uid 1234 isaac-sim -d /isaac-sim \
    && mkdir -p /isaac-sim && chown -R isaac-sim:isaac-sim /isaac-sim

WORKDIR /isaac-sim
USER isaac-sim

RUN mkdir -p /isaac-sim/.nvidia-omniverse/config
ENV MIN_DRIVER_VERSION=570.169

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD bash -c 'exit 1'

ENTRYPOINT ["/bin/bash"]
CMD ["-c", "echo 'Isaac Sim 6.0 CUDA 13.1+ base ready. Mount binaries to /isaac-sim.' && sleep infinity"]
DOCKERFILE_EOF

echo "✅ Dockerfile created"

# Build image
echo ""
echo "[5/6] Building image (this may take 10-15 minutes)..."
docker build -f Dockerfile.cuda13 -t "${GHCR_IMAGE}" -t "isaac-sim-6-cuda13:local" .

echo ""
echo "✅ Image built successfully!"
docker images | grep isaac-sim

# Push to GHCR
echo ""
echo "[6/6] Pushing to GHCR..."
docker push "$GHCR_IMAGE"

echo ""
echo "=========================================="
echo "✅ BUILD AND PUSH COMPLETE!"
echo "=========================================="
echo ""
echo "Image URL: ${GHCR_IMAGE}"
echo ""
echo "Update your RunPod template with:"
echo "  Image: ${GHCR_IMAGE}"
echo ""
echo "Docker Command (in RunPod template):"
echo '  rm -rf /workspace/IsaacSim 2>/dev/null || true'
echo ""
echo "Then in your container, mount Isaac Sim:"
echo "  # Clone or extract Isaac Sim binaries to /isaac-sim"
echo "  git clone https://github.com/explicitcontextualunderstanding/IsaacSim.git /isaac-sim"
echo ""
echo "Test locally:"
echo "  docker run --gpus all -it ${GHCR_IMAGE} nvidia-smi"
