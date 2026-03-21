#!/bin/bash
# Launch RunPod instance for CUDA 13.1+ image building
# Uses Python runpod SDK to find and deploy appropriate GPU

set -euo pipefail

cd "$(dirname "$0")/.."

# Load RunPod API key from 1Password if not set
if [ -z "${RUNPOD_API_KEY:-}" ]; then
    export RUNPOD_API_KEY=$(python3 -c "
import subprocess
import os
os.chdir('/Users/kieranlal/workspace/nano2')
result = subprocess.run(
    ['bash', '-c', 'source scripts/op_api_key_wrapper.sh --print-export RUNPOD_API_KEY'],
    capture_output=True, text=True, timeout=60
)
if result.returncode == 0 and result.stdout.startswith('export RUNPOD_API_KEY='):
    print(result.stdout.split('=', 1)[1].strip().strip(\"'\\\"\"))
")
fi

if [ -z "${RUNPOD_API_KEY:-}" ]; then
    echo "❌ RUNPOD_API_KEY not set and could not be loaded from 1Password"
    exit 1
fi

echo "=== Launching CUDA 13.1+ Builder Pod ==="
echo "Looking for GPUs with CUDA 13.1+ support..."
echo ""

# Run Python script to launch pod
python3 << 'PYTHON_SCRIPT'
import os
import sys
import runpod
import time

runpod.api_key = os.environ['RUNPOD_API_KEY']

# CUDA 13.1+ capable GPUs (Blackwell and newer)
# RTX Pro 6000, RTX 5090, RTX 4090 (Ada), H100, H200
preferred_gpus = [
    "NVIDIA RTX PRO 6000 Blackwell",
    "NVIDIA RTX 5090",
    "NVIDIA RTX 4090",
    "NVIDIA H100",
    "NVIDIA H200",
    "NVIDIA A100",
]

print("Fetching available GPUs...")
gpus = runpod.get_gpus()

if not gpus:
    print("❌ No GPUs available")
    sys.exit(1)

# Filter for GPUs with sufficient VRAM and availability
candidates = []
for gpu in gpus:
    vram = gpu.get("memoryInGb", 0)
    stock = gpu.get("stockStatus", "Unknown")
    name = gpu.get("displayName", "")
    price = gpu.get("lowestPrice", {}).get("uninterruptablePrice") or 0
    
    if vram >= 24 and stock not in ["Out", "Low"]:
        candidates.append({
            "id": gpu.get("id"),
            "name": name,
            "vram": vram,
            "price": price,
            "stock": stock,
        })

if not candidates:
    print("❌ No GPUs with 24GB+ VRAM available")
    sys.exit(1)

# Sort by price (cheapest first)
candidates.sort(key=lambda x: x["price"] if x["price"] else float('inf'))

print("\n=== Available GPUs (24GB+ VRAM) ===")
for i, gpu in enumerate(candidates[:10]):
    price_str = f"${gpu['price']:.3f}/hr" if gpu['price'] else "price unknown"
    pref = "✓" if any(pref in gpu['name'] for pref in preferred_gpus) else " "
    print(f"{pref} {i+1}. {gpu['name']:35} {gpu['vram']:3}GB @ {price_str:12} ({gpu['stock']})")

# Find best GPU (prefer CUDA 13.1+ capable, then cheapest)
selected = None
for pref_name in preferred_gpus:
    for gpu in candidates:
        if pref_name in gpu['name']:
            selected = gpu
            break
    if selected:
        break

# Fallback to cheapest available
if not selected:
    selected = candidates[0]

print(f"\n=== Selected GPU ===")
print(f"Name: {selected['name']}")
print(f"VRAM: {selected['vram']}GB")
print(f"Price: ${selected['price']:.3f}/hr" if selected['price'] else "Price: unknown")

# Get network volume ID
network_volume_id = os.environ.get("NETWORK_VOLUME_ID")
if not network_volume_id:
    # Try to get from 1Password
    import subprocess
    result = subprocess.run(
        ["bash", "-c", "source /Users/kieranlal/workspace/nano2/scripts/op_api_key_wrapper.sh --print-export NETWORK_VOLUME_ID"],
        capture_output=True, text=True, timeout=60
    )
    if result.returncode == 0 and result.stdout.startswith("export NETWORK_VOLUME_ID="):
        network_volume_id = result.stdout.split("=", 1)[1].strip().strip("'\"")

print(f"\nNetwork Volume: {network_volume_id or 'Not configured'}")

# Create pod
pod_name = f"isaac-sim-6-cuda13-builder-{int(time.time())}"
print(f"\n🚀 Creating pod: {pod_name}")

# Docker command to build image
docker_command = """#!/bin/bash
set -e
echo "=== CUDA 13.1+ Image Builder ==="
cd /workspace/IsaacSim || cd /isaac-sim

# Check CUDA version
echo "CUDA Version:"
nvcc --version

# Check Docker
echo ""
echo "Docker status:"
docker info 2>/dev/null || echo "Docker not running"

# Build image if Dockerfile exists
if [ -f Dockerfile.cuda13 ]; then
    echo ""
    echo "Building CUDA 13.1+ image..."
    ./scripts/build_cuda13_image.sh
else
    echo "Dockerfile.cuda13 not found"
fi

echo ""
echo "Build complete. Keeping container alive..."
sleep infinity
"""

try:
    pod = runpod.create_pod(
        name=pod_name,
        image_name="docker:dind",  # Docker-in-Docker image
        gpu_type_id=selected['id'],
        gpu_count=1,
        container_disk_in_gb=150,
        volume_in_gb=100,
        docker_args=docker_command,
        env={
            "GITHUB_TOKEN": os.environ.get("GITHUB_TOKEN", ""),
            "NETWORK_VOLUME_ID": network_volume_id or "",
        },
    )
    
    print(f"\n✅ Pod created successfully!")
    print(f"Pod ID: {pod['id']}")
    print(f"Status: {pod.get('desiredStatus', 'Unknown')}")
    
    # Get pod details
    pod_details = runpod.get_pod(pod['id'])
    if pod_details:
        runtime = pod_details.get('runtime', {})
        if runtime:
            public_ip = runtime.get('publicIp', {})
            if public_ip:
                print(f"\n📡 Connect to pod:")
                print(f"  SSH: ssh root@{public_ip.get('ip', 'N/A')}")
    
    print(f"\n⏳ Pod is starting... This may take 2-3 minutes.")
    print(f"Monitor at: https://www.runpod.io/console/pods")
    print(f"\nTo check pod status:")
    print(f"  python3 -c \"import runpod; runpod.api_key='{os.environ['RUNPOD_API_KEY']}'; print(runpod.get_pod('{pod['id']}'))\"")
    
except Exception as e:
    print(f"\n❌ Failed to create pod: {e}")
    sys.exit(1)
PYTHON_SCRIPT
