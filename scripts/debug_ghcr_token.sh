#!/bin/bash
# GHCR Token Debug Script
# Usage: export TOKEN="ghp_..."; ./scripts/debug_ghcr_token.sh
# Or: TOKEN="ghp_..." ./scripts/debug_ghcr_token.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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
SCOPE_RESPONSE=$(curl -sI -H "Authorization: Bearer ${TOKEN}" \
    "https://api.github.com/user")
SCOPES=$(echo "$SCOPE_RESPONSE" | grep -i "x-oauth-scopes" | tr -d '\r')

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
    
    # NEW: Check if write:packages exists (for org-owned packages)
    if echo "$SCOPES" | grep -q "write:packages"; then
        echo -e "${GREEN}✅ Has 'write:packages' scope${NC}"
        echo "   This may help with org-owned packages"
    else
        echo -e "${BLUE}ℹ️  INFO: Missing 'write:packages' scope${NC}"
        echo "   May be needed for certain org configurations"
    fi
else
    echo -e "${YELLOW}⚠️  Could not determine token scopes${NC}"
fi

# NEW: Check 2.5 - SAML Authorization Status
echo ""
echo "--- Check 2.5: SAML/SSO Authorization ---"
echo "   Checking if token requires SSO authorization..."

# Get organizations for the user
ORGS=$(curl -s -H "Authorization: Bearer ${TOKEN}" \
    "https://api.github.com/user/orgs" | jq -r '.[].login // empty')

if [ -n "$ORGS" ]; then
    echo "   User belongs to organizations:"
    echo "$ORGS" | head -5 | sed 's/^/     - /'
    
    # Check each org for SAML enforcement
    for ORG in $(echo "$ORGS" | head -3); do
        SAML_CHECK=$(curl -s -o /dev/null -w "%{http_code}" \
            -H "Authorization: Bearer ${TOKEN}" \
            "https://api.github.com/orgs/${ORG}/repos" 2>/dev/null || echo "000")
        
        if [ "$SAML_CHECK" = "403" ]; then
            echo -e "${YELLOW}⚠️  Organization '$ORG' may require SSO authorization${NC}"
            echo "   Token needs SSO enablement for this org"
        fi
    done
else
    echo "   User does not belong to any organizations"
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

# NEW: Check 3.5 - Package Ownership and Collaborators
echo ""
echo "--- Check 3.5: Package Ownership & Collaborators ---"

echo "   Fetching package details..."
PACKAGE_DETAILS=$(curl -s -H "Authorization: Bearer ${TOKEN}" \
    "https://api.github.com/user/packages/container/isaac-sim-6" 2>/dev/null || echo "")

if [ -n "$PACKAGE_DETAILS" ]; then
    OWNER_TYPE=$(echo "$PACKAGE_DETAILS" | jq -r '.owner.type // "unknown"')
    OWNER_LOGIN=$(echo "$PACKAGE_DETAILS" | jq -r '.owner.login // "unknown"')
    VISIBILITY=$(echo "$PACKAGE_DETAILS" | jq -r '.visibility // "unknown"')
    
    echo "   Package owner type: $OWNER_TYPE"
    echo "   Package owner: $OWNER_LOGIN"
    echo "   Package visibility: $VISIBILITY"
    
    if [ "$OWNER_TYPE" = "Organization" ]; then
        echo -e "${YELLOW}⚠️  WARNING: Package is organization-owned${NC}"
        echo "   Classic PATs may have limited access to org packages"
        echo "   Consider:"
        echo "     1. Ensure token has SSO authorization for org"
        echo "     2. Check org PAT policy settings"
        echo "     3. Try Fine-Grained PAT with explicit repo access"
    elif [ "$OWNER_TYPE" = "User" ]; then
        echo -e "${GREEN}✅ Package is user-owned${NC}"
    fi
    
    # Check if user has direct collaborator access
    # Note: The collaborators API may not work for user-owned packages or may require different permissions
    COLLAB_API_URL="https://api.github.com/user/packages/container/isaac-sim-6/collaborators"
    COLLAB_RESPONSE=$(curl -s -H "Authorization: Bearer ${TOKEN}" "$COLLAB_API_URL" 2>/dev/null || echo "")
    
    # Check if response is valid JSON array before parsing
    if [ -n "$COLLAB_RESPONSE" ] && echo "$COLLAB_RESPONSE" | jq -e 'type == "array"' > /dev/null 2>&1; then
        COLLABORATORS=$(echo "$COLLAB_RESPONSE" | jq -r '.[].login // empty')
        
        if [ -n "$COLLABORATORS" ]; then
            echo "   Package collaborators:"
            echo "$COLLABORATORS" | head -5 | sed 's/^/     - /'
            
            if echo "$COLLABORATORS" | grep -q "$GH_USER"; then
                echo -e "${GREEN}✅ User is listed as collaborator${NC}"
            else
                echo -e "${YELLOW}⚠️  WARNING: User not in collaborator list${NC}"
            fi
        else
            echo "   No collaborators found (empty list)"
        fi
    else
        echo "   ℹ️  Collaborator list not available (normal for user-owned packages)"
    fi
else
    echo -e "${YELLOW}⚠️  Could not fetch package details${NC}"
fi

echo ""
echo "--- Check 4: GHCR Manifest Access ---"
echo "   Testing manifest access (this is the critical test)..."

# Use verbose mode to capture headers
GHCR_RESPONSE=$(curl -s -w "\n%{http_code}\n" \
    -H "Authorization: Bearer ${TOKEN}" \
    "https://ghcr.io/v2/explicitcontextualunderstanding/isaac-sim-6/manifests/latest" 2>&1)

GHCR_STATUS=$(echo "$GHCR_RESPONSE" | tail -1)

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
        echo ""
        echo -e "${YELLOW}   Possible causes:${NC}"
        echo "   1. Package is org-owned and token lacks SSO authorization"
        echo "   2. Package collaborator access not granted"
        echo "   3. Organization PAT policy blocks Classic tokens"
        echo "   4. Fine-Grained PAT required for this repository"
        echo "   5. Package visibility settings prevent access"
        echo ""
        echo -e "${BLUE}   Diagnostic steps:${NC}"
        echo "   1. Check: https://github.com/settings/tokens"
        echo "      Look for 'Enable SSO' button next to your token"
        echo "   2. Check package settings:"
        echo "      https://github.com/users/explicitcontextualunderstanding/packages/container/package/isaac-sim-6"
        echo "   3. Try Fine-Grained PAT:"
        echo "      https://github.com/settings/personal-access-tokens/new"
        ;;
    404)
        echo -e "${YELLOW}⚠️  GHCR not found (HTTP 404)${NC}"
        echo "   The image doesn't exist or isn't accessible"
        ;;
    *)
        echo -e "${YELLOW}⚠️  GHCR returned HTTP $GHCR_STATUS${NC}"
        ;;
esac

# NEW: Check 4.5 - Verbose Headers for Debugging
echo ""
echo "--- Check 4.5: Verbose Headers (for support) ---"
echo "   Fetching detailed response headers..."

VERBOSE_RESPONSE=$(curl -s -I \
    -H "Authorization: Bearer ${TOKEN}" \
    "https://ghcr.io/v2/explicitcontextualunderstanding/isaac-sim-6/manifests/latest" 2>&1)

# Extract key headers
REQUEST_ID=$(echo "$VERBOSE_RESPONSE" | grep -i "x-github-request-id" | head -1 | tr -d '\r')
WWW_AUTH=$(echo "$VERBOSE_RESPONSE" | grep -i "www-authenticate" | head -1 | tr -d '\r')
CONTENT_TYPE=$(echo "$VERBOSE_RESPONSE" | grep -i "content-type" | head -1 | tr -d '\r')

if [ -n "$REQUEST_ID" ]; then
    echo "   $REQUEST_ID"
fi

if [ -n "$WWW_AUTH" ]; then
    echo "   WWW-Authenticate: $WWW_AUTH"
    echo -e "${YELLOW}   (This header indicates authentication challenge type)${NC}"
fi

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

# NEW: Check 6 - Token Type Detection & Fine-Grained PAT Info
echo ""
echo "--- Check 6: Token Type Analysis ---"

if [[ "$TOKEN" =~ ^ghp_[a-zA-Z0-9]{36}$ ]]; then
    echo -e "${BLUE}ℹ️  Token Type: Classic Personal Access Token (PAT)${NC}"
    echo "   Prefix: ghp_"
    echo ""
    echo -e "${YELLOW}   Note: Classic PATs may have limitations:${NC}"
    echo "   - May require SSO authorization for org access"
    echo "   - May be blocked by org Fine-Grained PAT policy"
    echo "   - Org-owned packages may need explicit collaborator access"
    echo ""
    echo "   Try creating a Fine-Grained PAT if this token fails:"
    echo "   https://github.com/settings/personal-access-tokens/new"
elif [[ "$TOKEN" =~ ^ghs_[a-zA-Z0-9]{35,}$ ]]; then
    echo -e "${BLUE}ℹ️  Token Type: GitHub App Installation Token${NC}"
    echo "   Prefix: ghs_"
    echo ""
    echo -e "${YELLOW}   Note: App tokens are permission snapshots:${NC}"
    echo "   - Generated when App permissions were granted"
    echo "   - Do NOT inherit permission changes retroactively"
    echo "   - Must regenerate after App permission updates"
elif [[ "$TOKEN" =~ ^gho_[a-zA-Z0-9]{35,}$ ]]; then
    echo -e "${YELLOW}⚠️  Token Type: OAuth Token${NC}"
    echo "   Prefix: gho_"
    echo ""
    echo -e "${RED}   WARNING: OAuth tokens do NOT support package access!${NC}"
    echo "   These tokens are for user authentication only"
    echo "   Use a Classic PAT (ghp_) or App token (ghs_) instead"
else
    echo -e "${YELLOW}⚠️  Unknown token format${NC}"
    echo "   Unrecognized prefix: ${TOKEN:0:4}"
fi

echo ""
echo "--- Check 6.5: gh CLI Token Fallback ---"
echo "   Checking if gh CLI is available..."

if command -v gh &> /dev/null; then
    echo -e "${GREEN}✅ gh CLI is installed${NC}"
    
    # Check if gh is authenticated (macOS compatible, no -P flag)
    GH_USER=$(gh auth status 2>&1 | grep "Logged in to github.com as" | sed 's/.*as //' | awk '{print $1}' || echo "")
    
    if [ -n "$GH_USER" ]; then
        echo "   gh CLI is authenticated as: $GH_USER"
        
        # Get gh's token
        GH_TOKEN=$(gh auth token 2>/dev/null || echo "")
        
        if [ -n "$GH_TOKEN" ]; then
            echo "   Token prefix: ${GH_TOKEN:0:4}..."
            
            # Check if it's a PAT or OAuth
            if [[ "$GH_TOKEN" =~ ^ghp_ ]]; then
                echo -e "${GREEN}✅ gh token is a Classic PAT (ghp_)${NC}"
                echo "   This token may work for GHCR!"
                
                # Offer to test it
                echo ""
                echo -e "${BLUE}   To test gh's token, run:${NC}"
                echo "      export TOKEN=\"\$(gh auth token)\""
                echo "      ./scripts/debug_ghcr_token.sh"
                
            elif [[ "$GH_TOKEN" =~ ^gho_ ]]; then
                echo -e "${YELLOW}⚠️  gh token is OAuth (gho_)${NC}"
                echo "   OAuth tokens do NOT work for GHCR package access"
                echo "   You need a Classic PAT with read:packages scope"
                
            elif [[ "$GH_TOKEN" =~ ^ghs_ ]]; then
                echo -e "${YELLOW}⚠️  gh token is GitHub App token (ghs_)${NC}"
                echo "   App tokens may work but are permission snapshots"
                echo "   Ensure App has 'read:packages' permission"
            fi
            
            # Quick test of gh token
            echo ""
            echo "   Testing gh token against GHCR..."
            GH_TEST=$(curl -s -o /dev/null -w "%{http_code}" \
                -H "Authorization: Bearer ${GH_TOKEN}" \
                "https://ghcr.io/v2/explicitcontextualunderstanding/isaac-sim-6/manifests/latest")
            
            if [ "$GH_TEST" = "200" ]; then
                echo -e "${GREEN}✅ gh token WORKS for GHCR!${NC}"
                echo ""
                echo "   To use gh's token:"
                echo "      export TOKEN=\"\$(gh auth token)\""
            else
                echo -e "${YELLOW}⚠️  gh token failed (HTTP $GH_TEST)${NC}"
                echo "   gh token likely lacks 'read:packages' scope"
            fi
        else
            echo -e "${YELLOW}⚠️  Could not retrieve gh token${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  gh CLI is not authenticated${NC}"
        echo "   Run: gh auth login"
    fi
else
    echo -e "${YELLOW}⚠️  gh CLI is not installed${NC}"
    echo "   Install from: https://cli.github.com/"
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
    echo "Most Likely Causes (in order):"
    echo ""
    echo "1. ${YELLOW}Organization SAML/SSO${NC} - Token not authorized for org"
    echo "   Fix: https://github.com/settings/tokens → Enable SSO"
    echo ""
    echo "2. ${YELLOW}Package Collaborator Access${NC} - Not listed as reader"
    echo "   Fix: https://github.com/users/explicitcontextualunderstanding/packages/container/package/isaac-sim-6"
    echo "   → Package Settings → Manage Access"
    echo ""
    echo "3. ${YELLOW}Org PAT Policy${NC} - Classic PATs blocked"
    echo "   Fix: Create Fine-Grained PAT:"
    echo "   https://github.com/settings/personal-access-tokens/new"
    echo "   → Select repo: isaac-sim-6 → Grant packages:read"
    echo ""
    echo "4. ${YELLOW}Missing write:packages${NC} - Org requires write scope"
    echo "   Fix: Regenerate token with BOTH read:packages AND write:packages"
    echo ""
    echo "Fallback: Use S3 download instead:"
    echo "   aws s3 cp s3://isaac-sim-6-0-dev/isaac-sim-6.tar ."
fi

echo ""
echo "=== End of Report ==="
echo ""
echo -e "${BLUE}For support, include the following in your ticket:${NC}"
echo "   - HTTP Status: $GHCR_STATUS"
[ -n "$REQUEST_ID" ] && echo "   - $REQUEST_ID"
echo "   - Token prefix: ${TOKEN:0:4}..."
echo "   - User: $GH_USER"
[ -n "$OWNER_TYPE" ] && echo "   - Package owner type: $OWNER_TYPE"
