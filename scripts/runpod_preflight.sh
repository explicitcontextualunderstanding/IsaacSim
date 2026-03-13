#!/bin/bash
# Isaac Sim 6.0 Pre-Flight Script for RunPod
# Paste this into the RunPod "Docker Command" field to auto-setup on pod start
#
# This script:
# 1. Checks CUDA/Blackwell compatibility
# 2. Installs missing Isaac Sim build dependencies
# 3. Clones the Isaac Sim repo (if GH_TOKEN available)
# 4. Keeps container alive with sleep infinity

set -e

echo "=== Isaac Sim 6.0 Pre-Flight ==="

# Check CUDA version
echo "[1/5] Checking CUDA..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
    CUDA_VERSION=$(nvcc --version 2>/dev/null | grep release | awk '{print $5}' | tr -d ',')
    echo "   CUDA Version: ${CUDA_VERSION:-not found}"
else
    echo "   ERROR: nvidia-smi not found"
fi

# Check Ubuntu version
echo "[2/5] Checking OS..."
. /etc/os-release
echo "   Ubuntu ${VERSION_ID}"

# Install missing dependencies
echo "[3/5] Installing Isaac Sim build dependencies..."
apt-get update -qq
apt-get install -y --no-install-recommends \
    libvulkan-dev \
    vulkan-tools \
    libgl1-mesa-dev \
    libx11-dev \
    libxext-dev \
    libxcursor-dev \
    libxrandr-dev \
    libxi-dev \
    libxinerama-dev \
    libxfixes-dev \
    libgl1 \
    libglu1-mesa \
    libsm6 \
    libice6 \
    libxrender1 \
    libxtst6 \
    libxmu6 \
    libxmu6 \
    libglib2.0-0 \
    libgtest-dev \
    libbenchmark-dev \
    rapidjson-dev \
    libopenexr-dev \
    libjson-c-dev \
    libtbb-dev \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    libzstd-dev \
    libssl-dev \
    pkg-config \
    wget \
    curl \
    git \
    git-lfs \
    ca-certificates \
    unzip \
    7zip \
    7z \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install GCC 11 for Ubuntu 24.04 (Isaac Sim requires GCC 11, not 12+)
echo "   Setting up GCC 11..."
if [ -f /etc/os-release ] && . /etc/os-release && [ "$VERSION_ID" = "24.04" ]; then
    apt-get update -qq
    apt-get install -y --no-install-recommends gcc-11 g++-11
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 110
    echo "   GCC 11 configured as default"
fi

echo "   Dependencies installed"

# Setup GH credentials if available
echo "[4/5] Setting up GitHub access..."
if [ -n "$GH_TOKEN" ]; then
    echo "   GH_TOKEN found in environment"
    git config --global credential.helper store
    echo "https://${GH_TOKEN}@github.com" > ~/.git-credentials
    git config --global user.email "ci@runpod.io"
    git config --global user.name "RunPod CI"
else
    echo "   WARNING: GH_TOKEN not set - cannot clone private repos"
fi

# Clone Isaac Sim if GH available
echo "[5/5] Cloning Isaac Sim..."
if [ -n "$GH_TOKEN" ] && [ -d /workspace ]; then
    cd /workspace
    if [ ! -d IsaacSim ]; then
        echo "   Cloning explicitcontextualunderstanding/IsaacSim..."
        git clone https://github.com/explicitcontextualunderstanding/IsaacSim.git
    else
        echo "   IsaacSim already exists"
    fi
    cd IsaacSim
    echo "   Repo ready at $(pwd)"
else
    echo "   Skipping clone (no GH_TOKEN)"
fi

echo ""
echo "=== Pre-Flight Complete ==="
echo "To build Isaac Sim:"
echo "  cd /workspace/IsaacSim"
echo "  ./build.sh"
echo ""
echo "Keeping container alive..."
sleep infinity
