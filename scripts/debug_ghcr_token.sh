#!/bin/bash
# GHCR Token Debug Script
# Usage: export TOKEN="ghp_..."; ./scripts/debug_ghcr_token.sh
# Or: TOKEN="ghp_..." ./scripts/debug_ghcr_token.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=== GHCR Token Debug Script ==="
echo "Timestamp: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo ""

# Check 1: TOKEN is set
if [ -z "$TOKEN" ]; then
    echo -e "${RED}❌ FAIL: TOKEN environment variable is not set${NC}"
    echo "   Set it with: export TOKEN='ghp_YOUR_TOKEN'"
    exit 1
fi

echo -e "${GREEN}✅ TOKEN is set${NC}"
echo "   Length: ${#TOKEN} characters"
echo "   Prefix: ${TOKEN:0:4}"

# Validate token format
if [[ ! "$TOKEN" =~ ^ghp_[a-zA-Z0-9]{36}$ ]]; then
    echo -e "${YELLOW}⚠️  WARNING: Token format looks incorrect${NC}"
    echo "   Expected: ghp_ followed by 36 characters"
    echo "   Got: ${TOKEN:0:10}..."
fi

echo ""
echo "--- Check 1: GitHub API Access ---"
GH_USER=$(curl -s -H "Authorization: Bearer ${TOKEN}" \
    "https://api.github.com/user" | jq -r '.login // empty')

if [ -n "$GH_USER" ]; then
    echo -e "${GREEN}✅ Token is valid for GitHub API${NC}"
    echo "   User: $GH_USER"
else
    echo -e "${RED}❌ FAIL: Token rejected by GitHub API${NC}"
    echo "   The token may be expired or invalid"
    exit 1
fi

echo ""
echo "--- Check 2: Token Scopes ---"
SCOPES=$(curl -sI -H "Authorization: Bearer ${TOKEN}" \
    "https://api.github.com/user" | grep -i "x-oauth-scopes" | tr -d '\r')

if [ -n "$SCOPES" ]; then
    echo -e "${GREEN}✅ Token has scopes${NC}"
    echo "   $SCOPES"
    
    # Check for required scopes
    if echo "$SCOPES" | grep -q "read:packages"; then
        echo -e "${GREEN}✅ Has 'read:packages' scope${NC}"
    else
        echo -e "${RED}❌ MISSING: 'read:packages' scope${NC}"
        echo "   This scope is REQUIRED for GHCR access"
    fi
    
    if echo "$SCOPES" | grep -q "repo"; then
        echo -e "${GREEN}✅ Has 'repo' scope${NC}"
    else
        echo -e "${YELLOW}⚠️  WARNING: Missing 'repo' scope${NC}"
        echo "   May be needed for private package metadata"
    fi
else
    echo -e "${YELLOW}⚠️  Could not determine token scopes${NC}"
fi

echo ""
echo "--- Check 3: Package Registry Access ---"
PACKAGES=$(curl -s -H "Authorization: Bearer ${TOKEN}" \
    "https://api.github.com/user/packages?package_type=container" | jq -r '.[].name // empty')

if [ -n "$PACKAGES" ]; then
    echo -e "${GREEN}✅ Can list packages${NC}"
    echo "   Packages found:"
    echo "$PACKAGES" | head -5 | sed 's/^/     - /'
    
    if echo "$PACKAGES" | grep -q "isaac-sim-6"; then
        echo -e "${GREEN}✅ isaac-sim-6 package is accessible${NC}"
    else
        echo -e "${YELLOW}⚠️  isaac-sim-6 package not found in list${NC}"
        echo "   This may be normal if the package is owned by an org"
    fi
else
    echo -e "${YELLOW}⚠️  No packages found or access denied${NC}"
fi

echo ""
echo "--- Check 4: GHCR Manifest Access ---"
GHCR_STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "Authorization: Bearer ${TOKEN}" \
    "https://ghcr.io/v2/explicitcontextualunderstanding/isaac-sim-6/manifests/latest")

case "$GHCR_STATUS" in
    200)
        echo -e "${GREEN}✅ GHCR manifest accessible (HTTP 200)${NC}"
        echo "   The token can pull the image!"
        ;;
    401)
        echo -e "${RED}❌ FAIL: GHCR unauthorized (HTTP 401)${NC}"
        echo "   Token lacks 'read:packages' scope"
        echo "   OR package doesn't exist/isn't accessible"
        ;;
    403)
        echo -e "${RED}❌ FAIL: GHCR forbidden (HTTP 403)${NC}"
        echo "   Token has packages scope but access is denied"
        echo "   Possible causes:"
        echo "     - Token is expired"
        echo "     - Token owner doesn't have access to the package"
        echo "     - Package visibility issue"
        ;;
    404)
        echo -e "${YELLOW}⚠️  GHCR not found (HTTP 404)${NC}"
        echo "   The image doesn't exist or isn't accessible"
        ;;
    *)
        echo -e "${YELLOW}⚠️  GHCR returned HTTP $GHCR_STATUS${NC}"
        ;;
esac

echo ""
echo "--- Check 5: Package Visibility ---"
echo "   Testing unauthenticated access..."
ANON_STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
    "https://ghcr.io/v2/explicitcontextualunderstanding/isaac-sim-6/manifests/latest")

if [ "$ANON_STATUS" = "401" ]; then
    echo -e "${GREEN}✅ Package exists and is private (as expected)${NC}"
elif [ "$ANON_STATUS" = "200" ]; then
    echo -e "${YELLOW}⚠️  Package is PUBLIC${NC}"
    echo "   This shouldn't happen for a private EULA-protected image"
else
    echo "   Unauthenticated access: HTTP $ANON_STATUS"
fi

echo ""
echo "=== Summary ==="

if [ "$GHCR_STATUS" = "200" ]; then
    echo -e "${GREEN}✅ Token is READY for RunPod deployment${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Update RunPod Registry Credential:"
    echo "   - Settings → Registry Credentials → Delete old GHCR-IsaacSim"
    echo "   - Create new: ghcr.io, username: explicitcontextualunderstanding"
    echo "   - Token: ${TOKEN:0:10}..."
    echo ""
    echo "2. Create pod:"
    echo "   runpod pod create \\"
    echo "     --image ghcr.io/explicitcontextualunderstanding/isaac-sim-6:latest \\"
    echo "     --gpu-type 'NVIDIA L40S' \\"
    echo "     --container-disk-size 50"
else
    echo -e "${RED}❌ Token is NOT ready for GHCR${NC}"
    echo ""
    echo "Common fixes:"
    echo "1. Regenerate token with 'read:packages' scope:"
    echo "   https://github.com/settings/tokens/new"
    echo ""
    echo "2. Verify token isn't expired:"
    echo "   https://github.com/settings/tokens"
    echo ""
    echo "3. Alternative: Use S3 download instead:"
    echo "   aws s3 cp s3://isaac-sim-6-0-dev/isaac-sim-6.tar ."
fi

echo ""
echo "=== End of Report ==="
