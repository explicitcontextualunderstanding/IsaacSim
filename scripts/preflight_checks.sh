#!/bin/bash
# Preflight Checks for Isaac Sim 6.0 Build
# Fails fast on missing prerequisites
# Usage: ./scripts/preflight_checks.sh [--full-stack]

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

CHECKS_PASSED=0
CHECKS_FAILED=0
WARNINGS=0

# Track which checks passed for dependency logic
declare -A CHECK_STATUS

log() { echo -e "${BLUE}[PREFLIGHT]${NC} $*"; }
pass() {
    echo -e "${GREEN}✓${NC} $*"
    CHECKS_PASSED=$((CHECKS_PASSED + 1))
}
fail() {
    echo -e "${RED}✗${NC} $*"
    CHECKS_FAILED=$((CHECKS_FAILED + 1))
    CHECK_STATUS["$1"]=failed
}
warn() {
    echo -e "${YELLOW}⚠${NC} $*"
    WARNINGS=$((WARNINGS + 1))
}

fatal() {
    echo ""
    echo -e "${RED}╔════════════════════════════════════════════════════════════╗"
    echo -e "║ PREFLIGHT FAILED - Cannot proceed with build               ║"
    echo -e "╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Failed checks: $CHECKS_FAILED"
    echo "Passed checks: $CHECKS_PASSED"
    exit 1
}

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Isaac Sim 6.0 Build - Preflight Checks                    ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# ============================================
# Environment Detection
# ============================================
detect_environment() {
    # Check for RunPod first (most specific)
    if [ -n "${RUNPOD_POD_ID:-}" ]; then
        echo "runpod"
    # Check for GitHub Actions
    elif [ -n "${GITHUB_ACTIONS:-}" ]; then
        echo "github-actions"
    # Check for NVMe (before GPU - more specific)
    elif [ -d "/mnt/nvme" ]; then
        echo "local-nvme"
    # Check for GPU (generic local)
    elif command -v nvidia-smi &>/dev/null; then
        echo "local-gpu"
    else
        echo "local"
    fi
}

ENVIRONMENT=$(detect_environment)
log "Detected environment: $ENVIRONMENT"

# Storage mapping by environment
declare -A STORAGE_PATHS
case "$ENVIRONMENT" in
runpod)
    STORAGE_PATHS["network_volume"]="/workspace"
    STORAGE_PATHS["container_disk"]="/"
    ;;
github-actions)
    STORAGE_PATHS["github_runner"]="/"
    ;;
local-nvme)
    STORAGE_PATHS["nvme"]="/mnt/nvme"
    STORAGE_PATHS["root"]="/"
    ;;
local-gpu)
    STORAGE_PATHS["root"]="/"
    ;;
*)
    STORAGE_PATHS["root"]="/"
    ;;
esac

# ============================================
# CHECK 1: Environment Variables
# ============================================
log "[1/12] Checking environment variables..."

MISSING_VARS=()

# Check required vars
if [ -z "${GITHUB_TOKEN:-}" ]; then
    MISSING_VARS+=("GITHUB_TOKEN")
fi

if [ -z "${GITHUB_USER:-}" ]; then
    export GITHUB_USER="explicitcontextualunderstanding"
    warn "GITHUB_USER not set, using default: $GITHUB_USER"
fi

# Optional but recommended
if [ -z "${AWS_ACCESS_KEY_ID:-}" ] && [ -z "${AWS_PROFILE:-}" ]; then
    warn "AWS credentials not set - S3 operations may fail"
fi

if [ ${#MISSING_VARS[@]} -gt 0 ]; then
    fail "ENV_VARS" "Missing required environment variables: ${MISSING_VARS[*]}"
    echo ""
    echo "Set these variables:"
    for var in "${MISSING_VARS[@]}"; do
        case $var in
        GITHUB_TOKEN)
            echo "  export GITHUB_TOKEN=ghp_xxxxxxxx"
            echo "    Create at: https://github.com/settings/tokens"
            echo "    Required scope: write:packages"
            ;;
        esac
    done
else
    pass "Environment variables configured"
    CHECK_STATUS["ENV_VARS"]=passed
fi

# ============================================
# CHECK 2: GitHub Token Validation
# ============================================
log "[2/12] Validating GitHub token..."

if [ -z "${GITHUB_TOKEN:-}" ]; then
    fail "GITHUB_AUTH" "Cannot validate token - GITHUB_TOKEN not set"
else
    # Check token format
    if [[ ! "$GITHUB_TOKEN" =~ ^ghp_[a-zA-Z0-9]{36}$ ]] &&
        [[ ! "$GITHUB_TOKEN" =~ ^ghs_[a-zA-Z0-9]{36}$ ]] &&
        [[ ! "$GITHUB_TOKEN" =~ ^github_pat_[a-zA-Z0-9_]{22,}$ ]]; then
        warn "Token format looks unusual (should start with ghp_, ghs_, or github_pat_)"
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
            https://api.github.com/user 2>/dev/null | jq -r '.login' 2>/dev/null || echo "unknown")

        # Check token scopes
        GH_SCOPES=$(curl -s -I \
            -H "Authorization: Bearer $GITHUB_TOKEN" \
            -H "Accept: application/vnd.github+json" \
            https://api.github.com/user 2>/dev/null | grep -i "x-oauth-scopes" || echo "")

        if echo "$GH_SCOPES" | grep -q "write:packages" || echo "$GH_SCOPES" | grep -q "repo"; then
            pass "GitHub token valid for user: $GH_USER"
            if echo "$GH_SCOPES" | grep -q "write:packages"; then
                pass "Token has write:packages scope"
            else
                warn "Token may lack write:packages scope - GHCR push may fail"
            fi
            CHECK_STATUS["GITHUB_AUTH"]=passed
        else
            fail "GITHUB_AUTH" "Token valid but missing write:packages scope"
            echo "Current scopes: $GH_SCOPES"
            echo "Required: write:packages"
        fi
    elif [ "$GH_API_RESPONSE" == "401" ]; then
        fail "GITHUB_AUTH" "Token invalid or expired (HTTP 401)"
    elif [ "$GH_API_RESPONSE" == "000" ]; then
        fail "GITHUB_AUTH" "Cannot reach GitHub API (network issue)"
    else
        fail "GITHUB_AUTH" "GitHub API returned HTTP $GH_API_RESPONSE"
    fi
fi

# ============================================
# CHECK 3: Docker Daemon
# ============================================
log "[3/12] Checking Docker..."

if ! command -v docker &>/dev/null; then
    fail "DOCKER" "Docker not installed"
    echo "Install: curl -fsSL https://get.docker.com | sh"
else
    DOCKER_VERSION=$(docker --version 2>/dev/null | awk '{print $3}' | tr -d ',' || echo "unknown")

    if docker info &>/dev/null; then
        DOCKER_INFO=$(docker info --format '{{.ServerVersion}}' 2>/dev/null || echo "unknown")
        pass "Docker daemon running (v$DOCKER_VERSION)"
        CHECK_STATUS["DOCKER"]=passed
    else
        fail "DOCKER" "Docker installed but daemon not accessible"
        echo "Try: sudo systemctl start docker"
        echo "Or: sudo usermod -aG docker $USER && newgrp docker"
    fi
fi

# ============================================
# CHECK 4: NVIDIA GPU & Drivers
# ============================================
log "[4/12] Checking NVIDIA GPU..."

if ! command -v nvidia-smi &>/dev/null; then
    fail "NVIDIA" "nvidia-smi not found - NVIDIA drivers not installed"
else
    NVIDIA_SMI_OUTPUT=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo "")

    if [ -z "$NVIDIA_SMI_OUTPUT" ]; then
        fail "NVIDIA" "nvidia-smi found but no GPU detected"
    else
        GPU_NAME=$(echo "$NVIDIA_SMI_OUTPUT" | cut -d',' -f1 | xargs)
        DRIVER_VERSION=$(echo "$NVIDIA_SMI_OUTPUT" | cut -d',' -f2 | xargs)
        GPU_MEM=$(echo "$NVIDIA_SMI_OUTPUT" | cut -d',' -f3 | xargs)

        pass "GPU detected: $GPU_NAME ($GPU_MEM)"
        pass "Driver: $DRIVER_VERSION"

        # Check driver version for CUDA 13.1+
        DRIVER_MAJOR=$(echo "$DRIVER_VERSION" | cut -d'.' -f1)
        if ! [[ "$DRIVER_MAJOR" =~ ^[0-9]+$ ]]; then
            fail "NVIDIA" "Cannot parse driver version: $DRIVER_VERSION (major: '$DRIVER_MAJOR')"
            echo "  Expected format: XXX.YYY"
        else
            if [ "$DRIVER_MAJOR" -ge 570 ]; then
                pass "Driver supports CUDA 13.1+ (570+)"
            else
                warn "Driver $DRIVER_VERSION may not support CUDA 13.1+ (needs 570+)"
            fi
        fi

        CHECK_STATUS["NVIDIA"]=passed
    fi
fi

# ============================================
# CHECK 5: CUDA Toolkit
# ============================================
log "[5/12] Checking CUDA toolkit..."

if ! command -v nvcc &>/dev/null; then
    fail "CUDA" "nvcc not found - CUDA toolkit not installed"
    echo "Install: apt-get install nvidia-cuda-toolkit"
else
    NVCC_VERSION=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $6}' | tr -d ',' || echo "unknown")
    CUDA_MAJOR=$(echo "$NVCC_VERSION" | cut -d'.' -f1)

    pass "CUDA toolkit: $NVCC_VERSION"

    if [ "$CUDA_MAJOR" = "13" ] || [ "$CUDA_MAJOR" -ge 13 ] 2>/dev/null; then
        pass "CUDA 13.1+ available"
        CHECK_STATUS["CUDA"]=passed
    else
        warn "CUDA $NVCC_VERSION may not support sm_100 (Blackwell)"
        warn "Isaac Sim 6.0 requires CUDA 13.1+ for Blackwell GPUs"
        CHECK_STATUS["CUDA"]=warning
    fi
fi

# ============================================
# CHECK 6: Docker NVIDIA Runtime
# ============================================
log "[6/12] Checking Docker NVIDIA runtime..."

if [ "${CHECK_STATUS[DOCKER]:-}" != "passed" ] || [ "${CHECK_STATUS[NVIDIA]:-}" != "passed" ]; then
    warn "Skipping NVIDIA runtime check (prerequisites not met)"
else
    NVIDIA_DOCKER_TEST=$(docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi -L 2>&1 || echo "FAILED")

    if echo "$NVIDIA_DOCKER_TEST" | grep -q "GPU"; then
        pass "Docker NVIDIA runtime working"
        CHECK_STATUS["NVIDIA_DOCKER"]=passed
    else
        fail "NVIDIA_DOCKER" "Docker cannot access GPU"
        echo "Install nvidia-docker2:"
        echo "  distribution=$(
            . /etc/os-release
            echo $ID$VERSION_ID
        )"
        echo "  curl -s -L https://nvidia.github.io/nvidia-docker/\$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list"
        echo "  sudo apt-get update && sudo apt-get install -y nvidia-docker2"
        echo "  sudo systemctl restart docker"
    fi
fi

# ============================================
# CHECK 7: Disk Space
# ============================================
log "[7/12] Checking disk space..."

log "Checking storage locations for environment: $ENVIRONMENT"
for storage_type in "${!STORAGE_PATHS[@]}"; do
    path="${STORAGE_PATHS[$storage_type]}"
    if df -BG "$path" &>/dev/null; then
        DISK_INFO=$(df -BG "$path" 2>/dev/null | tail -1)
        AVAIL_GB=$(echo "$DISK_INFO" | awk '{print $4}' | tr -d 'G')

        echo "  $storage_type (${path}): ${AVAIL_GB}GB available"
    fi
done

# Determine primary storage for build (use network volume on RunPod, nvme locally)
PRIMARY_PATH="${STORAGE_PATHS[network_volume]:-${STORAGE_PATHS[nvme]:-${STORAGE_PATHS[github_runner]:-${STORAGE_PATHS[container_disk]:-${STORAGE_PATHS[root]:-/}}}}}"
AVAILABLE_GB=$(df -BG "$PRIMARY_PATH" 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G' || echo "0")

if [ "$AVAILABLE_GB" -ge 100 ] 2>/dev/null; then
    pass "Sufficient disk space (${AVAILABLE_GB}GB on primary: $PRIMARY_PATH)"
    CHECK_STATUS["DISK"]=passed
elif [ "$AVAILABLE_GB" -ge 50 ] 2>/dev/null; then
    warn "Low disk space (${AVAILABLE_GB}GB on $PRIMARY_PATH) - build may fail"
    warn "Isaac Sim build requires ~100GB"
    CHECK_STATUS["DISK"]=warning
else
    fail "DISK" "Insufficient disk space (${AVAILABLE_GB}GB on $PRIMARY_PATH, need 100GB+)"
fi

# ============================================
# CHECK 8: Memory
# ============================================
log "[8/12] Checking memory..."

if command -v free &>/dev/null; then
    MEM_TOTAL=$(free -g | awk '/^Mem:/{print $2}')
    MEM_AVAILABLE=$(free -g | awk '/^Mem:/{print $7}')

    pass "Memory: ${MEM_TOTAL}GB total, ${MEM_AVAILABLE}GB available"

    if [ "$MEM_TOTAL" -ge 64 ] 2>/dev/null; then
        pass "Sufficient memory (64GB+)"
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
# CHECK 9: Network Connectivity
# ============================================
log "[9/12] Checking network connectivity..."

NETWORK_OK=true

# Check GitHub
if ! curl -s --max-time 10 https://github.com &>/dev/null; then
    fail "NETWORK" "Cannot reach github.com"
    NETWORK_OK=false
else
    pass "github.com reachable"
fi

# Check GHCR
if ! curl -s --max-time 10 https://ghcr.io &>/dev/null; then
    fail "NETWORK" "Cannot reach ghcr.io"
    NETWORK_OK=false
else
    pass "ghcr.io reachable"
fi

# Check NVIDIA registry (for base images)
if ! curl -s --max-time 10 https://nvcr.io &>/dev/null; then
    warn "Cannot reach nvcr.io (NVIDIA registry)"
else
    pass "nvcr.io reachable"
fi

if [ "$NETWORK_OK" = true ]; then
    CHECK_STATUS["NETWORK"]=passed
fi

# ============================================
# CHECK 10: GHCR Authentication
# ============================================
log "[10/12] Checking GHCR authentication..."

if [ "${CHECK_STATUS[GITHUB_AUTH]:-}" != "passed" ] || [ "${CHECK_STATUS[DOCKER]:-}" != "passed" ]; then
    warn "Skipping GHCR auth check (prerequisites not met)"
else
    # Try to login
    if echo "$GITHUB_TOKEN" | docker login ghcr.io -u "$GITHUB_USER" --password-stdin &>/dev/null; then
        pass "GHCR authentication successful"

        # Try to check if we can see packages
        GHCR_TEST=$(curl -s -o /dev/null -w "%{http_code}" \
            -H "Authorization: Bearer $GITHUB_TOKEN" \
            https://ghcr.io/v2/ 2>/dev/null || echo "000")

        if [ "$GHCR_TEST" == "200" ] || [ "$GHCR_TEST" == "401" ]; then
            # 401 is OK - means auth worked but endpoint needs more
            pass "GHCR API accessible"
        fi

        CHECK_STATUS["GHCR"]=passed
    else
        fail "GHCR" "Cannot authenticate to GHCR"
        echo "Check:"
        echo "  1. Token has 'write:packages' scope"
        echo "  2. User '$GITHUB_USER' is correct"
        echo "  3. Package permissions granted at:"
        echo "     https://github.com/users/$GITHUB_USER/packages/container/isaac-sim-6/settings"
    fi
fi

# ============================================
# CHECK 11: S3 Build Artifact (for GH workflow)
# ============================================
log "[11/12] Checking S3 build artifacts..."

S3_BUCKET="${S3_BUCKET:-isaac-sim-6-0-dev}"
S3_PREFIX="${S3_PREFIX:-builds}"

# Check if AWS credentials are available
if [ -z "${AWS_ACCESS_KEY_ID:-}" ] && [ -z "${AWS_PROFILE:-}" ]; then
    warn "AWS credentials not set - cannot verify S3 artifacts"
else
    # List available builds
    S3_BUILDS=$(aws s3 ls "s3://${S3_BUCKET}/${S3_PREFIX}/" 2>/dev/null || echo "")

    if [ -z "$S3_BUILDS" ]; then
        warn "No builds found in s3://${S3_BUCKET}/${S3_PREFIX}/"
    else
        pass "S3 bucket accessible"

        # Check for specific build
        if echo "$S3_BUILDS" | grep -q "isaac-sim-build-.*\.tar\.gz"; then
            pass "Build artifacts found in S3"
            echo "$S3_BUILDS" | grep "isaac-sim-build" | head -5 | sed 's/^/    /'
        else
            warn "No isaac-sim-build-*.tar.gz found"
        fi
    fi
fi

# ============================================
# CHECK 12: GHCR Image Manifest (for RunPod)
# ============================================
log "[12/12] Checking GHCR image accessibility..."

if [ -z "${GITHUB_TOKEN:-}" ]; then
    warn "GITHUB_TOKEN not set - cannot verify GHCR image"
else
    # Test if we can access the specific image manifest
    GHCR_IMAGE="ghcr.io/explicitcontextualunderstanding/isaac-sim-6"

    MANIFEST_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" \
        -H "Authorization: Bearer $GITHUB_TOKEN" \
        -H "Accept: application/vnd.docker.distribution.manifest.v2+json" \
        "https://ghcr.io/v2/${GHCR_IMAGE}/manifests/latest" 2>/dev/null || echo "000")

    case $MANIFEST_RESPONSE in
    200)
        pass "GHCR image accessible: ${GHCR_IMAGE}:latest"
        CHECK_STATUS["GHCR_IMAGE"]=passed
        ;;
    401)
        warn "GHCR auth required - token may need 'read:packages' scope"
        echo "  Run: curl -sI -H 'Authorization: Bearer \$GITHUB_TOKEN' \\"
        echo "    'https://ghcr.io/v2/${GHCR_IMAGE}/manifests/latest'"
        CHECK_STATUS["GHCR_IMAGE"]=warning
        ;;
    404)
        warn "GHCR image not found: ${GHCR_IMAGE}:latest"
        CHECK_STATUS["GHCR_IMAGE"]=warning
        ;;
    *)
        warn "GHCR image check returned HTTP $MANIFEST_RESPONSE"
        CHECK_STATUS["GHCR_IMAGE"]=warning
        ;;
    esac
fi

# ============================================
# SUMMARY
# ============================================
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
if [ $CHECKS_FAILED -eq 0 ]; then
    echo "║  PREFLIGHT PASSED ✓                                        ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Passed: $CHECKS_PASSED | Warnings: $WARNINGS | Failed: $CHECKS_FAILED"
    echo ""
    echo "Ready to proceed with build."
    echo ""
    exit 0
else
    echo "║  PREFLIGHT FAILED ✗                                        ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Passed: $CHECKS_PASSED | Warnings: $WARNINGS | Failed: $CHECKS_FAILED"
    echo ""
    fatal
fi
