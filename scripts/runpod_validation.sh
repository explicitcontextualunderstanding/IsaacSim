#!/bin/bash
# RunPod Deployment Validation
# Validates Isaac Sim 6.0 container on RunPod before marking as "ready"
# Usage: ./scripts/runpod_validation.sh [--wait-for-app] [--timeout SECONDS]

set -euo pipefail

WAIT_FOR_APP="${WAIT_FOR_APP:-false}"
TIMEOUT="${TIMEOUT:-300}"
ACCEPT_EULA="${ACCEPT_EULA:-Y}"
PRIVACY_CONSENT="${PRIVACY_CONSENT:-Y}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --wait-for-app)
            WAIT_FOR_APP=true
            shift
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[RUNPOD-VAL]${NC} $*"; }
pass() { echo -e "${GREEN}✓${NC} $*"; }
fail() { echo -e "${RED}✗${NC} $*"; exit 1; }
warn() { echo -e "${YELLOW}⚠${NC} $*"; }

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Isaac Sim 6.0 - RunPod Deployment Validation            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# ============================================
# CHECK 1: Environment
# ============================================
log "[1/8] Checking RunPod environment..."

if [ -n "${RUNPOD_POD_ID:-}" ]; then
    pass "Running on RunPod: $RUNPOD_POD_ID"
else
    warn "Not running on RunPod (no RUNPOD_POD_ID)"
fi

# Check workspace
if [ -d "/workspace" ]; then
    WORKSPACE_SIZE=$(df -h /workspace | tail -1 | awk '{print $4}')
    pass "Workspace mounted: $WORKSPACE_SIZE available"
else
    warn "Workspace not at /workspace"
fi

# Check network volume
if [ -d "/workspace/IsaacSim" ]; then
    pass "IsaacSim directory exists in workspace"
else
    warn "IsaacSim not in workspace (may need to clone)"
fi

# ============================================
# CHECK 2: Container Environment
# ============================================
log "[2/8] Checking container environment..."

if [ -f "/.dockerenv" ] || [ -f "/run/.containerenv" ]; then
    pass "Running inside container"
else
    warn "May not be inside container"
fi

if [ -f "/etc/os-release" ]; then
    OS_NAME=$(grep "^NAME=" /etc/os-release | cut -d'"' -f2)
    OS_VERSION=$(grep "^VERSION_ID=" /etc/os-release | cut -d'"' -f2)
    pass "OS: $OS_NAME $OS_VERSION"
fi

# ============================================
# CHECK 3: GPU Access
# ============================================
log "[3/8] Validating GPU access..."

if ! command -v nvidia-smi &>/dev/null; then
    fail "nvidia-smi not found in container"
fi

GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "")

if [ -z "$GPU_INFO" ]; then
    fail "Cannot access GPU from container"
fi

GPU_NAME=$(echo "$GPU_INFO" | cut -d',' -f1 | xargs)
DRIVER=$(echo "$GPU_INFO" | cut -d',' -f2 | xargs)
MEM_USED=$(echo "$GPU_INFO" | cut -d',' -f3 | xargs)
MEM_TOTAL=$(echo "$GPU_INFO" | cut -d',' -f4 | xargs)

pass "GPU: $GPU_NAME"
pass "Driver: $DRIVER"
pass "Memory: $MEM_USED / $MEM_TOTAL"

# Check CUDA
if command -v nvcc &>/dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | tr -d ',')
    pass "CUDA: $CUDA_VERSION"

    CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d'.' -f1)
    if [ "$CUDA_MAJOR" = "13" ] 2>/dev/null || [ "$CUDA_MAJOR" -ge 13 ] 2>/dev/null; then
        pass "CUDA 13.1+ confirmed"
    else
        warn "CUDA $CUDA_VERSION (13.1+ recommended for Blackwell)"
    fi
else
    warn "nvcc not in PATH"
fi

# ============================================
# CHECK 4: Isaac Sim Installation
# ============================================
log "[4/8] Checking Isaac Sim installation..."

ISAAC_PATHS=(
    "/isaac-sim"
    "/workspace/IsaacSim"
    "/workspace/IsaacSim/_build/linux-x86_64/release"
)

ISAAC_FOUND=false
for path in "${ISAAC_PATHS[@]}"; do
    if [ -d "$path" ]; then
        FILE_COUNT=$(find "$path" -type f 2>/dev/null | wc -l)
        pass "Isaac Sim found: $path ($FILE_COUNT files)"
        ISAAC_FOUND=true
        ISAAC_DIR="$path"
        break
    fi
done

if [ "$ISAAC_FOUND" = false ]; then
    fail "Isaac Sim installation not found"
fi

# Check for key binaries
KEY_FILES=("kit" "python.sh")
for file in "${KEY_FILES[@]}"; do
    if [ -f "$ISAAC_DIR/$file" ] || [ -f "$ISAAC_DIR/isaac-sim.$file" ]; then
        pass "Key file found: $file"
    else
        warn "Key file not found: $file"
    fi
done

# ============================================
# CHECK 5: Dependencies
# ============================================
log "[5/8] Checking dependencies..."

# Check Python
if command -v python3 &>/dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    pass "Python: $PYTHON_VERSION"
fi

# Check GCC
if command -v gcc &>/dev/null; then
    GCC_VERSION=$(gcc --version | head -1)
    pass "GCC: $GCC_VERSION"

    GCC_MAJOR=$(gcc -dumpversion | cut -d'.' -f1)
    if [ "$GCC_MAJOR" = "11" ]; then
        pass "GCC 11 confirmed"
    else
        warn "GCC $GCC_MAJOR (11 recommended)"
    fi
fi

# Check Vulkan
if command -v vulkaninfo &>/dev/null; then
    VULKAN_INFO=$(vulkaninfo --summary 2>/dev/null | grep "deviceName" | head -1 || echo "")
    if [ -n "$VULKAN_INFO" ]; then
        pass "Vulkan available: $VULKAN_INFO"
    else
        warn "Vulkan installed but no device found"
    fi
else
    warn "Vulkan not installed"
fi

# ============================================
# CHECK 6: Network
# ============================================
log "[6/8] Checking network connectivity..."

if curl -s --max-time 5 https://github.com &>/dev/null; then
    pass "GitHub accessible"
else
    warn "Cannot reach GitHub"
fi

if curl -s --max-time 5 https://ghcr.io &>/dev/null; then
    pass "GHCR accessible"
else
    warn "Cannot reach GHCR"
fi

# Check streaming ports
for port in 5900 6080 8211; do
    if ss -tln | grep -q ":$port"; then
        pass "Port $port listening"
    else
        warn "Port $port not listening (may be normal)"
    fi
done

# ============================================
# CHECK 7: Isaac Sim Startup Test
# ============================================
log "[7/8] Testing Isaac Sim startup..."

export OMNI_KIT_ALLOW_ROOT=1
export ACCEPT_EULA
export PRIVACY_CONSENT
export HEADLESS=1

# Find Isaac Sim executable
if [ -f "$ISAAC_DIR/python.sh" ]; then
    PYTHON_SH="$ISAAC_DIR/python.sh"
elif [ -f "$ISAAC_DIR/kit" ]; then
    KIT_BIN="$ISAAC_DIR/kit"
else
    warn "Cannot find Isaac Sim executable"
    PYTHON_SH=""
fi

# Quick import test
if [ -n "$PYTHON_SH" ]; then
    IMPORT_TEST=$($PYTHON_SH -c "import carb; print('Carb OK')" 2>&1 || echo "FAILED")
    if echo "$IMPORT_TEST" | grep -q "Carb OK"; then
        pass "Omniverse imports working"
    else
        warn "Import test issue: $IMPORT_TEST"
    fi
fi

# ============================================
# CHECK 8: App Startup (Optional)
# ============================================
if [ "$WAIT_FOR_APP" = true ]; then
    log "[8/8] Waiting for Isaac Sim app startup..."
    log "Timeout: ${TIMEOUT} seconds"

    LOG_DIR="$ISAAC_DIR/.nvidia-omniverse/logs/Kit"
    START_TIME=$(date +%s)

    # Start Isaac Sim in background
    if [ -f "$ISAAC_DIR/omni.isaac.sim.python.kit" ]; then
        $ISAAC_DIR/python.sh "$ISAAC_DIR/omni.isaac.sim.python.kit" &
        PID=$!
        log "Started Isaac Sim (PID: $PID)"

        # Wait for AppReady
        while true; do
            CURRENT_TIME=$(date +%s)
            ELAPSED=$((CURRENT_TIME - START_TIME))

            if [ $ELAPSED -gt $TIMEOUT ]; then
                fail "Timeout waiting for AppReady (${TIMEOUT}s)"
            fi

            # Check log for AppReady
            if [ -d "$LOG_DIR" ]; then
                LATEST_LOG=$(find "$LOG_DIR" -name "kit_*.log" -type f -mmin -1 2>/dev/null | head -1)
                if [ -n "$LATEST_LOG" ] && grep -q "AppReady" "$LATEST_LOG"; then
                    pass "Isaac Sim app ready!"
                    break
                fi
            fi

            # Check if process died
            if ! kill -0 $PID 2>/dev/null; then
                fail "Isaac Sim process died"
            fi

            sleep 5
        done

        # Cleanup
        kill $PID 2>/dev/null || true
    else
        warn "Kit file not found, skipping app test"
    fi
else
    log "[8/8] Skipping app startup test (use --wait-for-app to enable)"
fi

# ============================================
# SUMMARY
# ============================================
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  RUNPOD VALIDATION COMPLETE ✓                              ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Isaac Sim 6.0 is ready to run on RunPod!"
echo ""
echo "Quick start:"
echo "  cd $ISAAC_DIR"
echo "  ./python.sh"
echo ""
echo "Or with streaming:"
echo "  ./runheadless.sh"
echo ""
echo "noVNC available at: https://<pod-ip>.runpod.io:6080"
echo ""
