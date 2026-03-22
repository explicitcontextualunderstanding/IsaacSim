#!/bin/bash
# Mid-Flight Validation - Check build progress and catch issues early
# Usage: ./scripts/mid_flight_check.sh [phase]

set -euo pipefail

PHASE="${1:-check}"
ISAAC_DIR="${ISAAC_DIR:-/workspace/IsaacSim}"
BUILD_DIR="${ISAAC_DIR}/_build"
LOG_DIR="${BUILD_DIR}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[MID-FLIGHT]${NC} $*"; }
pass() { echo -e "${GREEN}✓${NC} $*"; }
fail() { echo -e "${RED}✗${NC} $*"; }
warn() { echo -e "${YELLOW}⚠${NC} $*"; }

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Isaac Sim Build - Mid-Flight Check                        ║"
echo "║  Phase: $PHASE"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

case $PHASE in
    post-clone)
        log "Checking repository after clone..."

        if [ ! -d "$ISAAC_DIR/.git" ]; then
            fail "Repository not cloned properly"
            exit 1
        fi

        cd "$ISAAC_DIR"
        COMMIT_COUNT=$(git rev-list --count HEAD 2>/dev/null || echo "0")
        pass "Repository cloned: $COMMIT_COUNT commits"

        # Check for critical files
        CRITICAL_FILES=("build.sh" "repo.sh" "premake5.lua" "package.toml")
        MISSING=0
        for file in "${CRITICAL_FILES[@]}"; do
            if [ ! -f "$file" ]; then
                fail "Missing critical file: $file"
                MISSING=$((MISSING + 1))
            fi
        done

        if [ $MISSING -eq 0 ]; then
            pass "All critical files present"
        else
            exit 1
        fi

        # Check Git LFS
        if [ -f .gitmodules ]; then
            LFS_FILES=$(git lfs ls-files 2>/dev/null | wc -l)
            if [ "$LFS_FILES" -gt 0 ]; then
                log "Git LFS files: $LFS_FILES"
                LFS_PULLED=$(find . -type f -size -1k -name "*.png" 2>/dev/null | wc -l)
                if [ "$LFS_PULLED" -gt 0 ]; then
                    warn "Some LFS files appear unpulled"
                    warn "Run: git lfs pull"
                else
                    pass "Git LFS files pulled"
                fi
            fi
        fi
        ;;

    post-build)
        log "Checking build artifacts..."

        if [ ! -d "$BUILD_DIR" ]; then
            fail "Build directory not found: $BUILD_DIR"
            exit 1
        fi

        # Check for build outputs
        BUILD_OUTPUTS=("$BUILD_DIR/linux-x86_64/release")
        for dir in "${BUILD_OUTPUTS[@]}"; do
            if [ -d "$dir" ]; then
                FILE_COUNT=$(find "$dir" -type f 2>/dev/null | wc -l)
                pass "Build output found: $dir ($FILE_COUNT files)"
            else
                fail "Build output missing: $dir"
                exit 1
            fi
        done

        # Check for error logs
        if [ -f "$LOG_DIR/build.log" ]; then
            ERROR_COUNT=$(grep -c "error:" "$LOG_DIR/build.log" 2>/dev/null || echo "0")
            if [ "$ERROR_COUNT" -gt 0 ]; then
                warn "Build log contains $ERROR_COUNT errors"
                grep "error:" "$LOG_DIR/build.log" | head -5
            else
                pass "No errors in build log"
            fi
        fi

        # Check binary sizes
        if [ -f "$BUILD_DIR/linux-x86_64/release/isaac-sim" ] || \
           [ -f "$BUILD_DIR/linux-x86_64/release/omni.isaac.sim" ]; then
            pass "Main binaries present"
        else
            warn "Main binaries not found - checking for kit files..."
            KIT_COUNT=$(find "$BUILD_DIR" -name "kit*.exe" -o -name "kit" 2>/dev/null | wc -l)
            if [ "$KIT_COUNT" -gt 0 ]; then
                pass "Kit binaries found ($KIT_COUNT)"
            else
                fail "No kit binaries found"
                exit 1
            fi
        fi
        ;;

    post-package)
        log "Checking package artifacts..."

        PACKAGE_DIR="${ISAAC_DIR}/_build/packages"

        if [ ! -d "$PACKAGE_DIR" ]; then
            fail "Package directory not found: $PACKAGE_DIR"
            exit 1
        fi

        # Look for package files
        PACKAGE_COUNT=$(find "$PACKAGE_DIR" -name "*.zip" -o -name "*.tar.gz" 2>/dev/null | wc -l)
        if [ "$PACKAGE_COUNT" -gt 0 ]; then
            pass "Packages found: $PACKAGE_COUNT"
            ls -lh "$PACKAGE_DIR" | grep -E "\.zip|\.tar" | head -10
        else
            fail "No packages found in $PACKAGE_DIR"
            exit 1
        fi

        # Check package sizes
        for pkg in $(find "$PACKAGE_DIR" -name "*.zip" -o -name "*.tar.gz" 2>/dev/null | head -5); do
            SIZE=$(du -h "$pkg" | cut -f1)
            pass "Package: $(basename $pkg) ($SIZE)"
        done
        ;;

    post-docker)
        log "Checking Docker image..."

        IMAGE_NAME="${IMAGE_NAME:-isaac-sim-6:latest}"

        if ! docker images | grep -q "isaac-sim"; then
            fail "Docker image not found"
            exit 1
        fi

        IMAGE_ID=$(docker images --format "{{.Repository}}:{{.Tag}}" | grep "isaac-sim" | head -1)
        pass "Image found: $IMAGE_ID"

        # Check image size
        SIZE=$(docker images --format "{{.Size}}" | grep -E "GB|MB" | head -1)
        pass "Image size: $SIZE"

        # Basic container test
        log "Testing container startup..."
        if docker run --rm "$IMAGE_ID" bash -c "nvidia-smi" &>/dev/null; then
            pass "Container GPU access working"
        else
            warn "Container GPU test inconclusive"
        fi

        if docker run --rm "$IMAGE_ID" bash -c "ls /isaac-sim/" &>/dev/null; then
            pass "Isaac Sim files accessible in container"
        else
            fail "Isaac Sim files not found in container"
            exit 1
        fi
        ;;

    pre-push)
        log "Pre-push validation..."

        IMAGE_NAME="${IMAGE_NAME:-isaac-sim-6:latest}"

        if ! docker images | grep -q "isaac-sim"; then
            fail "Image not found locally"
            exit 1
        fi

        # Check GHCR auth
        if ! docker info 2>/dev/null | grep -q "Username"; then
            warn "Not logged into Docker registry"
            warn "Run: echo \$GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin"
        else
            REGISTRY=$(docker info --format '{{.IndexServerAddress}}' 2>/dev/null || echo "")
            if echo "$REGISTRY" | grep -q "ghcr"; then
                pass "Logged into GHCR"
            else
                warn "May not be logged into GHCR"
            fi
        fi

        # Verify tag
        TAGS=$(docker images --format "{{.Tag}}" | grep -v "<none>" | head -5)
        if [ -n "$TAGS" ]; then
            pass "Image tagged:"
            echo "$TAGS" | sed 's/^/  - /'
        else
            warn "Image may not be properly tagged"
        fi
        ;;

    *)
        log "Running full mid-flight check..."
        $0 post-clone
        $0 post-build
        $0 post-package
        $0 post-docker
        $0 pre-push
        ;;
esac

echo ""
echo "Mid-flight check complete for phase: $PHASE"
