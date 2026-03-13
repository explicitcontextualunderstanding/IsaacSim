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
    docker.io || true

# Install Git LFS (needed for Isaac Sim repo)
echo "Installing Git LFS..."
apt-get install -y git-lfs
git lfs install

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
echo "1. Clone the Isaac Sim repo:"
echo "   git clone https://github.com/${GH_ORG:-explicitcontextualunderstanding}/IsaacSim.git"
echo "   cd IsaacSim"
echo ""
echo "2. Set GitHub authentication:"
echo "   export GH_USER=your-username"
echo "   export GH_PAT=your-github-token"
echo ""
echo "3. Run the Kaniko build:"
echo "   ./scripts/run_kaniko_build.sh --push"
echo ""
echo "Or use the automated script:"
echo "   python3 scripts/automated_build.py"
