#!/bin/bash
# Preflight Checks for RunPod Spot Instance
# Validates GPU, CUDA, token, network BEFORE provisioning expensive Vultr
# Usage: ./scripts/preflight_runpod.sh
# Cost: ~$0.50 for 10 minutes on RunPod spot vs $2.50/hr on Vultr

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

CHECKS_PASSED=0
CHECKS_FAILED=0
WARNINGS=0

declare -A CHECK_STATUS

log() { echo -e "${BLUE}[RUNPOD-PREFLIGHT]${NC} $*"; }
pass() { echo -e "${GREEN}✓${NC} $*"; CHECKS_PASSED=$((CHECKS_PASSED + 1)); }
fail() { echo -e "${RED}✗${NC} $*"; CHECKS_FAILED=$((CHECKS_FAILED + 1)); CHECK_STATUS["$1"]=failed; }
warn() { echo -e "${YELLOW}⚠${NC} $*"; WARNINGS=$((WARNINGS + 1)); }

print_summary() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    if [ $CHECKS_FAILED -eq 0 ]; then
        echo "║  RUNPOD PREFLIGHT PASSED ✓                                 ║"
        echo "╚════════════════════════════════════════════════════════════╝"
        echo ""
        echo "Passed: $CHECKS_PASSED | Warnings: $WARNINGS | Failed: $CHECKS_FAILED"
        echo ""
        echo "✅ Ready to provision Vultr for Docker build"
        echo ""
        return 0
    else
        echo "║  RUNPOD PREFLIGHT FAILED ✗                                 ║"
        echo "╚════════════════════════════════════════════════════════════╝"
        echo ""
        echo "Passed: $CHECKS_PASSED | Warnings: $WARNINGS | Failed: $CHECKS_FAILED"
        echo ""
        echo "❌ Fix issues before provisioning Vultr"
        echo ""
        return 1
    fi
}

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  RunPod Spot Preflight - Validate Before Vultr             ║"
echo "║  Cost: ~$0.50 (10 min) vs $2.50+/hr on Vultr                ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# ============================================
# CHECK 1: RunPod Environment
# ============================================
log "[1/10] Checking RunPod environment..."

if [ -n "${RUNPOD_POD_ID:-}" ]; then
    pass "Running on RunPod: $RUNPOD_POD_ID"
else
    warn "RUNPOD_POD_ID not set - may not be on RunPod"
fi

if [ -d "/workspace" ]; then
    WORKSPACE_FREE=$(df -BG /workspace 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G' || echo "0")
    pass "Workspace mounted: ${WORKSPACE_FREE}GB free"

    if [ "$WORKSPACE_FREE" -lt 100 ] 2>/dev/null; then
        warn "Low workspace space (${WORKSPACE_FREE}GB) - may need cleanup"
    else
        pass "Sufficient workspace space"
    fi
else
    fail "WORKSPACE" "Workspace not mounted at /workspace"
fi

# ============================================
# CHECK 2: Environment Variables
# ============================================
log "[2/10] Checking environment variables..."

if [ -z "${GITHUB_TOKEN:-}" ]; then
    fail "ENV_VARS" "GITHUB_TOKEN not set"
    echo ""
    echo "Set: export GITHUB_TOKEN=ghp_xxxxxxxx"
    echo "Create at: https://github.com/settings/tokens"
    echo "Required scope: write:packages"
else
    # Mask token for display
    TOKEN_PREVIEW="${GITHUB_TOKEN:0:8}...${GITHUB_TOKEN: -4}"
    pass "GITHUB_TOKEN set: $TOKEN_PREVIEW"
fi

if [ -z "${GITHUB_USER:-}" ]; then
    export GITHUB_USER="explicitcontextualunderstanding"
    warn "GITHUB_USER not set, using default: $GITHUB_USER"
else
    pass "GITHUB_USER: $GITHUB_USER"
fi

# ============================================
# CHECK 3: GitHub Token Validation
# ============================================
log "[3/10] Validating GitHub token..."

if [ -z "${GITHUB_TOKEN:-}" ]; then
    fail "GITHUB_API" "Cannot validate - GITHUB_TOKEN not set"
else
    # Check token format
    if [[ "$GITHUB_TOKEN" =~ ^ghp_[a-zA-Z0-9]{36}$ ]] || [[ "$GITHUB_TOKEN" =~ ^ghs_[a-zA-Z0-9]{36}$ ]]; then
        pass "Token format valid"
    else
        warn "Token format unusual (expected ghp_ or ghs_ prefix)"
    fi

    # Test API access
    GH_API_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" \
        -H "Authorization: Bearer $GITHUB_TOKEN" \
        -H "Accept: application/vnd.github+json" \
        https://api.github.com/user 2>/dev/null) || GH_API_RESPONSE="000"

    if [ "$GH_API_RESPONSE" == "200" ]; then
        GH_USER=$(curl -s \
            -H "Authorization: Bearer $GITHUB_TOKEN" \
            -H "Accept: application/vnd.github+json" \
            https://api.github.com/user 2>/dev/null | grep -o '"login": "[^"]*"' | cut -d'"' -f4 || echo "unknown")

        # Check token scopes
        GH_SCOPES=$(curl -s -I \
            -H "Authorization: Bearer $GITHUB_TOKEN" \
            -H "Accept: application/vnd.github+json" \
            https://api.github.com/user 2>/dev/null | grep -i "x-oauth-scopes" || echo "")

        if echo "$GH_SCOPES" | grep -qi "write:packages\|repo"; then
            pass "GitHub API accessible for user: $GH_USER"

            if echo "$GH_SCOPES" | grep -qi "write:packages"; then
                pass "Token has write:packages scope ✓"
                CHECK_STATUS["GITHUB_API"]=passed
            else
                warn "Token may lack write:packages scope"
                warn "Current scopes: $GH_SCOPES"
            fi
        else
            fail "GITHUB_SCOPE" "Token valid but missing write:packages scope"
            echo "Current scopes: $GH_SCOPES"
            echo "Required: write:packages"
        fi
    elif [ "$GH_API_RESPONSE" == "401" ]; then
        fail "GITHUB_API" "Token invalid or expired (HTTP 401)"
    elif [ "$GH_API_RESPONSE" == "000" ]; then
        fail "GITHUB_API" "Cannot reach GitHub API (network issue)"
    else
        fail "GITHUB_API" "GitHub API returned HTTP $GH_API_RESPONSE"
    fi
fi

# ============================================
# CHECK 4: NVIDIA GPU
# ============================================
log "[4/10] Checking NVIDIA GPU..."

if ! command -v nvidia-smi &>/dev/null; then
    fail "NVIDIA" "nvidia-smi not found"
else
    GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version,memory.total,cuda_version --format=csv,noheader 2>/dev/null || echo "")

    if [ -z "$GPU_INFO" ]; then
        fail "NVIDIA" "nvidia-smi found but no GPU detected"
    else
        GPU_NAME=$(echo "$GPU_INFO" | cut -d',' -f1 | xargs)
        DRIVER_VERSION=$(echo "$GPU_INFO" | cut -d',' -f2 | xargs)
        GPU_MEM=$(echo "$GPU_INFO" | cut -d',' -f3 | xargs)
        CUDA_DRIVER=$(echo "$GPU_INFO" | cut -d',' -f4 | xargs)

        pass "GPU: $GPU_NAME"
        pass "Driver: $DRIVER_VERSION"
        pass "Memory: $GPU_MEM"
        pass "CUDA (driver): $CUDA_DRIVER"

        # Check driver version for CUDA 13.1+
        DRIVER_MAJOR=$(echo "$DRIVER_VERSION" | cut -d'.' -f1)
        if [ "$DRIVER_MAJOR" -ge 570 ] 2>/dev/null; then
            pass "Driver supports CUDA 13.1+ (570+) ✓"
        else
            warn "Driver $DRIVER_VERSION may not support CUDA 13.1+ (needs 570+)"
        fi

        # Check GPU architecture
        case "$GPU_NAME" in
            *"Blackwell"*|*"H100"*|*"H200"*|*"RTX 50"*|*"RTX Pro 6000"*)
                pass "GPU supports CUDA 13.1+ (Blackwell/Hopper) ✓"
                CHECK_STATUS["GPU_COMPATIBLE"]=passed
                ;;
            *"RTX 4090"*|*"RTX 6000 Ada"*)
                warn "GPU is Ada architecture - may work with newer drivers"
                CHECK_STATUS["GPU_COMPATIBLE"]=warning
                ;;
            *"L40S"*|*"A100"*)
                warn "GPU may not support CUDA 13.1+ (check driver)"
                CHECK_STATUS["GPU_COMPATIBLE"]=warning
                ;;
        esac

        CHECK_STATUS["NVIDIA"]=passed
    fi
fi

# ============================================
# CHECK 5: CUDA Toolkit
# ============================================
log "[5/10] Checking CUDA toolkit..."

if ! command -v nvcc &>/dev/null; then
    fail "CUDA" "nvcc not found - CUDA toolkit not installed"
else
    NVCC_VERSION=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $6}' | tr -d ',' || echo "unknown")
    CUDA_MAJOR=$(echo "$NVCC_VERSION" | cut -d'.' -f1)

    pass "CUDA toolkit: $NVCC_VERSION"

    if [ "$CUDA_MAJOR" = "13" ] || [ "$CUDA_MAJOR" -ge 13 ] 2>/dev/null; then
        pass "CUDA 13.1+ available ✓"
        CHECK_STATUS["CUDA"]=passed
    else
        fail "CUDA_VERSION" "CUDA $NVCC_VERSION (13.1+ required for Blackwell)"
    fi
fi

# ============================================
# CHECK 6: Memory
# ============================================
log "[6/10] Checking memory..."

if command -v free &>/dev/null; then
    MEM_TOTAL=$(free -g | awk '/^Mem:/{print $2}')
    MEM_AVAILABLE=$(free -g | awk '/^Mem:/{print $7}')

    pass "Memory: ${MEM_TOTAL}GB total, ${MEM_AVAILABLE}GB available"

    if [ "$MEM_TOTAL" -ge 64 ] 2>/dev/null; then
        pass "Sufficient memory (64GB+) ✓"
        CHECK_STATUS["MEMORY"]=passed
    elif [ "$MEM_TOTAL" -ge 32 ] 2>/dev/null; then
        warn "Low memory (${MEM_TOTAL}GB) - build may be slow"
        CHECK_STATUS["MEMORY"]=warning
    else
        fail "MEMORY" "Insufficient memory (${MEM_TOTAL}GB, need 64GB+)"
    fi
else
    warn "Cannot check memory (free command not available)"
fi

# ============================================
# CHECK 7: Network Connectivity
# ============================================
log "[7/10] Checking network connectivity..."

NETWORK_OK=true

# Check GitHub
if curl -s --max-time 10 https://github.com &>/dev/null; then
    pass "github.com reachable"
else
    fail "NETWORK_GITHUB" "Cannot reach github.com"
    NETWORK_OK=false
fi

# Check GHCR
if curl -s --max-time 10 https://ghcr.io &>/dev/null; then
    pass "ghcr.io reachable"
else
    fail "NETWORK_GHCR" "Cannot reach ghcr.io"
    NETWORK_OK=false
fi

# Check NVIDIA registry
if curl -s --max-time 10 https://nvcr.io &>/dev/null; then
    pass "nvcr.io (NVIDIA registry) reachable"
else
    warn "Cannot reach nvcr.io"
fi

# Check GitHub API specifically
if [ "${CHECK_STATUS[GITHUB_API]:-}" = "passed" ]; then
    pass "GitHub API fully accessible"
fi

if [ "$NETWORK_OK" = true ]; then
    CHECK_STATUS["NETWORK"]=passed
fi

# ============================================
# CHECK 8: Git LFS (optional but recommended)
# ============================================
log "[8/10] Checking Git LFS..."

if command -v git-lfs &>/dev/null; then
    LFS_VERSION=$(git-lfs version 2>/dev/null | head -1 || echo "unknown")
    pass "Git LFS installed: $LFS_VERSION"

    # Test Git LFS auth (simplified check)
    if [ -n "${GITHUB_TOKEN:-}" ]; then
        pass "Git LFS token available"
    fi
else
    warn "Git LFS not installed (may be needed for assets)"
fi

# ============================================
# CHECK 9: Vultr Prerequisites Summary
# ============================================
log "[9/10] Checking Vultr requirements..."

VULTR_READY=true

# Must-have for Vultr
for check in GPU_COMPATIBLE NVIDIA CUDA MEMORY NETWORK; do
    if [ "${CHECK_STATUS[$check]:-}" != "passed" ] && [ "${CHECK_STATUS[$check]:-}" != "warning" ]; then
        fail "VULTR_READY" "$check not satisfied"
        VULTR_READY=false
    fi
done

if [ "$VULTR_READY" = true ]; then
    pass "All Vultr prerequisites satisfied ✓"
    CHECK_STATUS["VULTR_READY"]=passed
fi

# ============================================
# CHECK 10: Cost Estimate
# ============================================
log "[10/10] Cost estimate..."

echo ""
echo "Estimated costs for full build:"
echo "  • RunPod spot (validation):    ~\$0.50 (10 min)"
echo "  • Vultr GPU (build):           ~\$10-15 (4-6 hrs)"
echo "  • GHCR storage:                Free (public)"
echo "  • Total:                       ~\$10.50-15.50"
echo ""

if [ $CHECKS_FAILED -eq 0 ]; then
    echo "✅ Ready to provision Vultr"
    echo ""
    echo "Next steps:"
    echo "  1. Provision Vultr GPU instance (CUDA 13.1+ capable)"
    echo "  2. SSH into Vultr"
    echo "  3. Run: ./vultr_full_build.sh --skip-preflight"
else
    echo "❌ Fix failed checks before provisioning Vultr"
    echo ""
fi

# Print summary
print_summary
