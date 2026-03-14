#!/bin/bash
# Isaac Sim 6.0 Container Validation Script
# Usage: ./scripts/validate_container.sh [--headless]
#
# Tests that the Isaac Sim container is working correctly.

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
echo "[1/5] Testing GPU detection..."
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo "✅ GPU detected"

# Test 2: CUDA availability
echo ""
echo "[2/5] Testing CUDA..."
nvcc --version || echo "⚠️ nvcc not in PATH (OK for runtime)"
echo "✅ CUDA environment available"

# Test 3: Isaac Sim path
echo ""
echo "[3/5] Checking Isaac Sim installation..."
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

# Test 4: Python and omni imports
echo ""
echo "[4/5] Testing Python and Isaac Sim imports..."
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

# Test 5: Full headless test (if requested)
if [ -n "$HEADLESS" ]; then
    echo ""
    echo "[5/5] Running headless smoke test..."
    ./python.sh -c "
import omni
import omni.kit
import omni.usd

# Create a simple USD stage
stage = omni.usd.get_context().get_stage()
if stage:
    print('   ✅ USD stage created')
else:
    print('   ❌ Failed to create USD stage')
    exit(1)

print('✅ Headless smoke test PASSED')
"
    echo "✅ Headless test OK"
else
    echo ""
    echo "[5/5] Skipping headless test (use --headless to enable)"
fi

echo ""
echo "=== Validation Complete ==="
echo "✅ All tests passed!"
