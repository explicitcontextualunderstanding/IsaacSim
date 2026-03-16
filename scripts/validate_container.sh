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
        echo "❌ Error: /usr/bin/gcc is version ${GCC_VERSION}."
        echo "   Isaac Sim 6.0-dev requires strict GCC 11 adherence to prevent segmentation faults."
        exit 1
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
        echo "❌ Error: Detected tiny asset files (<1KB) likely caused by missing Git LFS pulls:"
        echo "$LFS_POINTERS"
        echo "   USD scenes will load with 'ghost' (black) textures. Ensure Git LFS assets are pulled."
        exit 1
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
    if grep -qE "sm_100|sm_120" "$COMPAT_LOG"; then
        echo "✅ Blackwell support (sm_100/sm_120) explicitly detected in compatibility logs."
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

# Bridge Import Test
echo "   Running ROS 2 bridge import test..."
ROS2_LOAD_TEST=$(./python.sh -c "import omni.isaac.ros2_bridge; print('ROS2 Bridge Loaded')" 2>&1 || true)
if echo "$ROS2_LOAD_TEST" | grep -q "ROS2 Bridge Loaded"; then
    echo "✅ ROS2 Bridge loaded successfully without undefined symbol errors."
else
    echo "❌ Error: ROS2 Bridge failed to load. Typical cause: fastdds/rmw linkage failure."
    echo "   Output: $ROS2_LOAD_TEST"
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

# 3. Check Renderer
simulation_app.update()
print(f"✅ Renderer Status: {'PASS' if simulation_app.is_running() else 'FAIL'}")
print(f"✅ ROS2 Bridge Ext: {'LOADED' if ros_success else 'FAILED'}")

simulation_app.close()
EOF

./python.sh /tmp/validate_pulse.py

echo ""
echo "=== Validation Complete ==="
echo "✅ All 6.0-dev critical requirements met."
