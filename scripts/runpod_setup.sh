#!/bin/bash
# Setup script for RunPod Isaac Sim build environment
# Run this first on a fresh RunPod instance

set -e

echo "=== RunPod Isaac Sim Build Environment Setup ==="

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (sudo)"
    exit 1
fi

# Install basic dependencies
echo "Installing dependencies..."
apt-get update
apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    python3 \
    python3-pip \
    python3-requests \
    docker.io || true

# Install Git LFS (needed for Isaac Sim repo)
echo "Installing Git LFS..."
apt-get install -y git-lfs
git lfs install

# Install GCC 11 (Strict requirement for Isaac Sim 6.0-dev)
echo "Setting up GCC 11..."
apt-get install -y gcc-11 g++-11
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 110
echo "✅ GCC 11 configured as default"

# Verify NVIDIA driver
echo "Checking NVIDIA driver..."
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# Check disk space
echo "Disk space:"
df -h /workspace 2>/dev/null || df -h /

# Check memory
echo "Memory:"
free -h

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Clone the Isaac Sim repo (if not already present):"
echo "   git clone https://github.com/${GH_ORG:-explicitcontextualunderstanding}/IsaacSim.git"
echo "   cd IsaacSim"
echo ""
echo "2. Run the automated build orchestrator:"
echo "   export RUNPOD_API_KEY='your-key'"
echo "   export NETWORK_VOLUME_ID='vol-xxx'"
echo "   python3 scripts/automated_build.py"
echo ""
echo "Or run the validation gate manually:"
echo "   ./scripts/validate_container.sh"
