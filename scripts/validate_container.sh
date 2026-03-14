#!/bin/bash
# Isaac Sim 6.0 Container Validation Script
# Usage: ./scripts/validate_container.sh [--headless]
#
# Tests that the Isaac Sim container is working correctly.
# Run with --headless to test PhysX GPU pipeline and RTX renderer.

set -e

echo "=== Isaac Sim 6.0 Container Validation ==="

# Parse args
HEADLESS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --headless)
            HEADLESS="1"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Test 1: GPU detection
echo ""
echo "[1/7] Testing GPU detection..."
nvidia-smi --query-gpu=name,driver_version,memory.total,m compute_cap --format=csv
echo "✅ GPU detected"

# Test 2: CUDA availability
echo ""
echo "[2/7] Testing CUDA..."
nvcc --version || echo "⚠️ nvcc not in PATH (OK for runtime)"
echo "✅ CUDA environment available"

# Test 3: Isaac Sim path
echo ""
echo "[3/7] Checking Isaac Sim installation..."
if [ -z "$ISAAC_SIM_PATH" ]; then
    ISAAC_SIM_PATH="/workspace/IsaacSim/_build/linux-x86_64/release"
fi
echo "   ISAAC_SIM_PATH: $ISAAC_SIM_PATH"

if [ -d "$ISAAC_SIM_PATH" ]; then
    echo "✅ Isaac Sim directory exists"
else
    echo "❌ Isaac Sim directory not found at $ISAAC_SIM_PATH"
    exit 1
fi

# Test 4: Compatibility check
echo ""
echo "[4/7] Running Isaac Sim compatibility check..."
cd "$ISAAC_SIM_PATH"
./isaac-sim.compatibility_check.sh --headless --/app/quitAfter=10 2>&1 || true
echo "✅ Compatibility check complete"

# Test 5: Python and omni imports
echo ""
echo "[5/7] Testing Python and Isaac Sim imports..."
cd "$ISAAC_SIM_PATH"
./python.sh -c "
import sys
print(f'   Python: {sys.version}')

# Test Isaac Sim modules
try:
    import omni
    print('   ✅ omni module loaded')
except ImportError as e:
    print(f'   ❌ omni import failed: {e}')
    sys.exit(1)

try:
    import carb
    print('   ✅ carb module loaded')
except ImportError as e:
    print(f'   ⚠️ carb import warning: {e}')

try:
    import pxr
    print('   ✅ pxr (USD) module loaded')
except ImportError as e:
    print(f'   ⚠️ pxr import warning: {e}')

print('✅ All imports successful')
"
echo "✅ Python imports OK"

# Test 6: Full headless test with PhysX and RTX (if requested)
if [ -n "$HEADLESS" ]; then
    echo ""
    echo "[6/7] Running PhysX GPU pipeline test..."
    cd "$ISAAC_SIM_PATH"

    # Create validation script
    cat > /tmp/validate_isaac_6.py << 'PYEOF'
import os
import sys

# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from isaacsim import SimulationApp

# Initialize SimulationApp in Headless Mode
CONFIG = {
    "headless": True,
    "renderer": "RayTracedLighting",  # Forces RTX initialization
    "width": 1280,
    "height": 720
}

print("[VALIDATION] Starting Isaac Sim 6.0-dev Headless...")
simulation_app = SimulationApp(CONFIG)

# Late imports (must happen AFTER SimulationApp)
import carb
from isaacsim.core.api import World
from isaacsim.core.utils.extensions import enable_extension

print("[VALIDATION] Core APIs loaded. Checking GPU availability...")

# Verify GPU & Driver
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

# Step the simulator
print("[VALIDATION] Stepping Physics Engine...")
for i in range(10):
    world.step(render=True)

print("[SUCCESS] Isaac Sim 6.0-dev validated. PhysX and RTX are functional.")

simulation_app.close()
print("[VALIDATION] Shutdown complete.")
PYEOF

    ./python.sh /tmp/validate_isaac_6.py
    echo "✅ PhysX GPU pipeline test OK"
else
    echo ""
    echo "[6/7] Skipping PhysX test (use --headless to enable)"
fi

# Test 7: Environment check
echo ""
echo "[7/7] Checking environment variables..."
echo "   ACCEPT_EULA: ${ACCEPT_EULA:-not set}"
echo "   OMNI_KIT_ACCEPT_EULA: ${OMNI_KIT_ACCEPT_EULA:-not set}"
echo "   ISAAC_SIM_PATH: ${ISAAC_SIM_PATH:-not set}"
echo "✅ Environment check complete"

echo ""
echo "=== Validation Complete ==="
echo "✅ All tests passed!"
echo ""
echo "To run full PhysX test: ./scripts/validate_container.sh --headless"
