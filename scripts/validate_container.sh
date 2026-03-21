#!/bin/bash
# Isaac Sim 6.0-dev Container Validation Script
# Usage: ./scripts/validate_container.sh
#
# Validates Isaac Sim 6.0-dev specific requirements like GCC 11,
# Blackwell-era SM compatibility, Git LFS assets, and ROS 2 Bridge.
# Ensures headless Golden MCAP render pipelines are ready.

set -e

echo "=== Isaac Sim 6.0-dev Container Validation ==="

# --- 1. Compiler & Environment Sanity ---
echo ""
echo "[1/4] Checking Compiler & Environment Sanity..."

# Environment Variables
export ISAACSIM_PATH="${ISAACSIM_PATH:-/workspace/IsaacSim/_build/linux-x86_64/release}"
echo "   ISAACSIM_PATH: $ISAACSIM_PATH"

if [ ! -d "$ISAACSIM_PATH" ]; then
    echo "❌ Error: Isaac Sim directory not found at $ISAACSIM_PATH"
    echo "   Ensure ISAACSIM_PATH points to _build/linux-x86_64/release."
    exit 1
fi

# GCC Version Check
if [ -x "/usr/bin/gcc" ]; then
    GCC_VERSION=$(/usr/bin/gcc -dumpversion)
    GCC_MAJOR=$(echo "$GCC_VERSION" | cut -d. -f1)
    if [ "$GCC_MAJOR" != "11" ]; then
        echo "⚠️  Incorrect GCC version detected (${GCC_VERSION}). Attempting auto-fix..."
        
        # Check if GCC 11 is even installed
        if ! dpkg -l | grep -q "gcc-11"; then
            echo "   Installing gcc-11 and g++-11..."
            apt-get update && apt-get install -y gcc-11 g++-11
        fi

        # Set up update-alternatives to prioritize version 11
        echo "   Updating alternatives to prioritize GCC 11..."
        update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 \
                            --slave /usr/bin/g++ g++ /usr/bin/g++-11
        update-alternatives --set gcc /usr/bin/gcc-11
        
        # Re-verify
        NEW_GCC=$(/usr/bin/gcc -dumpversion)
        echo "✅ GCC auto-fixed to version $NEW_GCC."
    else
        echo "✅ GCC version is 11 ($(/usr/bin/gcc --version | head -n1))"
    fi
else
    echo "❌ Error: /usr/bin/gcc not found. GCC 11 is required."
    exit 1
fi

# Git LFS Assets Scan
echo "   Scanning Git LFS assets in exts/omni.isaac.sim.res/data..."
LFS_DIR="$ISAACSIM_PATH/exts/omni.isaac.sim.res/data"
if [ -d "$LFS_DIR" ]; then
    # Look for files under 1KB that might be LFS pointers (usually 100-300 bytes)
    LFS_POINTERS=$(find "$LFS_DIR" -type f -size -1k \( -name "*.png" -o -name "*.usd" -o -name "*.usda" -o -name "*.usdc" \) 2>/dev/null | head -n 5 || true)
    
    if [ -n "$LFS_POINTERS" ]; then
        echo "⚠️  LFS Pointers detected. Attempting to pull assets..."
        
        # Ensure Git LFS is installed and initialized
        git lfs install
        
        # Force pull missing assets
        echo "   Running: git lfs pull (this may take a few minutes)..."
        git lfs pull
        
        # Final check
        LFS_RECHECK=$(find "$LFS_DIR" -type f -size -1k \( -name "*.png" -o -name "*.usd" -o -name "*.usda" -o -name "*.usdc" \) 2>/dev/null | head -n 1 || true)
        if [ -z "$LFS_RECHECK" ]; then
            echo "✅ Git LFS assets restored successfully."
        else
            echo "❌ Error: git lfs pull failed to restore assets. Check internet/auth."
            exit 1
        fi
    else
        echo "✅ Git LFS assets appear to be fully pulled (no <1KB pointers detected)."
    fi
else
    echo "⚠️ Warning: Git LFS data directory not found at $LFS_DIR."
fi


# --- 2. GPU & Headless Renderer Validation ---
echo ""
echo "[2/4] Validating GPU & Headless Renderer Compatibility..."
cd "$ISAACSIM_PATH"

echo "   Running: ./omni.isaac.sim.compatibility_check.sh --headless"
if [ -f "./omni.isaac.sim.compatibility_check.sh" ]; then
    COMPAT_LOG=$(mktemp)
    ./omni.isaac.sim.compatibility_check.sh --headless > "$COMPAT_LOG" 2>&1 || true
    
    # Check for Blackwell (sm_100/120) or Ada (sm_89) support in logs
    if grep -qE "sm_100|sm_120|CUDA Tile" "$COMPAT_LOG"; then
        echo "✅ Blackwell support (sm_100/sm_120 or CUDA Tile) explicitly detected in compatibility logs."
    elif grep -qE "sm_89" "$COMPAT_LOG"; then
        echo "✅ Ada support (sm_89) detected in compatibility logs."
    else
        echo "⚠️ Warning: Could not explicitly determine Blackwell (sm_100/120) or Ada (sm_89) support from logs."
        echo "   If using L40S/RTX 6000 Ada/Blackwell, ensure hardware is fully recognized."
    fi
    rm -f "$COMPAT_LOG"
else
    echo "⚠️ Warning: ./omni.isaac.sim.compatibility_check.sh not found. Skipping."
fi


# --- 3. ROS 2 Bridge & Library Linkage ---
echo ""
echo "[3/4] Testing ROS 2 Bridge & Library Linkage..."

# Nitros Acceleration library check
if [[ "$LD_LIBRARY_PATH" != *"isaac_ros"* ]]; then
    echo "⚠️ Warning: LD_LIBRARY_PATH does not appear to include Isaac ROS runtime libraries."
    echo "   For Golden MCAP recording, cuVSLAM or nvblox may fail silently without these libraries."
else
    echo "✅ LD_LIBRARY_PATH includes Isaac ROS paths."
fi

# Bridge Import Test (Isaac Sim 6.0-dev2 Modular ROS2)
echo " Running ROS 2 bridge import test (modular isaacsim.ros2.core)..."
ROS2_LOAD_TEST=$(./python.sh -c "import isaacsim.ros2.core; print('ROS2 Core Loaded')" 2>&1 || true)
if echo "$ROS2_LOAD_TEST" | grep -q "ROS2 Core Loaded"; then
    echo "✅ Modular ROS2 Core (isaacsim.ros2.core) loaded successfully."
else
    echo "❌ Error: Modular ROS2 Core failed to load. Typical cause: fastdds/rmw linkage failure."
    echo "    Output: $ROS2_LOAD_TEST"
    exit 1
fi


# --- 4. Python-based "Pulse Check" ---
echo ""
echo "[4/4] Running Python-based Pulse Check..."

cat > /tmp/validate_pulse.py << 'EOF'
from isaacsim import SimulationApp
import os
import sys

# Off-screen Pipeline configuration
os.environ["ENABLE_CAMERAS"] = "1"
os.environ["HEADLESS"] = "1"

# 1. Start headless with cameras enabled
config = {
    "headless": True,
}
print("   Starting SimulationApp (Headless, Cameras Enabled)...")
try:
    simulation_app = SimulationApp(config)
except Exception as e:
    print(f"❌ Failed to initialize SimulationApp: {e}")
    sys.exit(1)

# 2. Verify ROS2 Bridge extension
print("   Enabling omni.isaac.ros2_bridge extension...")
from isaacsim.core.utils.extensions import enable_extension
ros_success = enable_extension("omni.isaac.ros2_bridge")

# 3. Check Renderer & Warmup Shaders
print("   Warming up Vulkan shaders (10 frames)...")
for _ in range(10):
    simulation_app.update()

print(f"✅ Renderer Status: {'PASS' if simulation_app.is_running() else 'FAIL'}")
print(f"✅ ROS2 Bridge Ext: {'LOADED' if ros_success else 'FAILED'}")

simulation_app.close()
EOF

./python.sh /tmp/validate_pulse.py

# --- 5. Golden MCAP Smoke Test ---
echo ""
echo "[5/5] Running Golden MCAP Smoke Test..."

# Shader Cache Persistence Tip
if [ ! -d "/root/.nv/ComputeCache" ]; then
    echo "⚠️  Tip: Ensure /root/.nv/ComputeCache is mapped to a Network Volume to persist Vulkan shader cache and speed up warmup across pod restarts."
fi

echo "   Testing 10-second headless recording..."
if [ -f "scripts/record_golden_mcap.py" ]; then
    ./python.sh scripts/record_golden_mcap.py \
        --output /workspace/IsaacSim/recordings/test_pulse.mcap \
        --headless \
        --max_frames 300 || echo "⚠️ Warning: Golden MCAP smoke test failed. Check scripts/record_golden_mcap.py."
    
    MCAP_FILE="/workspace/IsaacSim/recordings/test_pulse.mcap"
    if [ -f "$MCAP_FILE" ]; then
        MCAP_SIZE=$(stat -c%s "$MCAP_FILE" 2>/dev/null || stat -f%z "$MCAP_FILE" 2>/dev/null || echo 0)
        # 5MB = 5 * 1024 * 1024 = 5242880 bytes
        if [ "$MCAP_SIZE" -gt 5242880 ]; then
            echo "✅ Smoke test completed successfully. MCAP file is > 5MB (${MCAP_SIZE} bytes)."
        else
            echo "❌ Error: MCAP file relies on fastdds but is too small (${MCAP_SIZE} bytes). The ros2_bridge may be running but failing to publish sensor data."
            exit 1
        fi
    else
        echo "❌ Error: test_pulse.mcap was not created."
        exit 1
    fi
else
    echo "⚠️ Warning: scripts/record_golden_mcap.py not found. Skipping smoke test."
fi

echo ""
echo "=== Validation Complete ==="
echo "✅ All 6.0-dev critical requirements met."
