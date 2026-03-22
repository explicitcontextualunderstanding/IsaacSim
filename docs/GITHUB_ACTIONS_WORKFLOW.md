# GitHub Actions Image Assembly

## Overview

Assemble Isaac Sim Docker images using **GitHub Actions** - free, fast, and no external CPU needed.

```
RunPod GPU (Compile) → S3 → GitHub Actions (Assemble) → GHCR → RunPod (Deploy)
     25 min, ~$2.50         Free, ~10 min                  Run on demand
```

## Why GitHub Actions?

| Aspect | GitHub Actions | Vultr/AWS | Local |
|--------|---------------|-----------|-------|
| **Cost** | **Free** ✅ | ~$0.15 | Free |
| **Setup** | None | Provision instance | Docker install |
| **Speed** | ~10 min | ~5 min | ~5 min |
| **Automation** | ✅ Native | Manual | Manual |
| **Integration** | ✅ Git-based triggers | API calls | None |

## ⚠️ Resource Limits (Free Tier)

GitHub Actions free tier has constraints that **may impact** Isaac Sim assembly:

| Limit | Value | Impact |
|-------|-------|--------|
| **Storage** | 500MB (actions/cache) | ✅ Sufficient (extracted build ~400MB) |
| **Artifact retention** | 90 days | ✅ Image in GHCR, not artifacts |
| **Job timeout** | 6 hours | ✅ Assembly ~10 min, well under limit |
| **Concurrent jobs** | 20 (Linux) | ✅ Single job |
| **Disk space** | ~14GB SSD | ⚠️ **Tight** - 4GB build + base image |
| **Network** | Unmetered | ✅ No egress charges |

### Disk Space Warning

**Free tier runners have ~14GB disk:**
- Ubuntu base image: ~2GB
- Isaac Sim build: ~4GB extracted
- Docker layers: ~4GB
- Working space: ~4GB
- **Total**: ~14GB (at limit!)

**Mitigations:**
1. Use `ubuntu:22.04` base (smaller than 24.04)
2. Use `--squash` during build (reduces layers)
3. Clean up with `docker system prune -f`
4. Skip local layer caching if disk low

### If Free Tier Fails

| Symptom | Cause | Solution |
|-----------|-------|----------|
| "No space left" | Disk full | Use larger runner or external CPU |
| "timeout" | Job too slow | Splits steps, use caching |
| "rate limit" | Too many runs | Wait or use GitHub Team ($4/mo) |

**Paid Options (With Funds):**

| Option | Cost | Specs | Best For |
|--------|------|-------|----------|
| **GitHub Larger Runners** | $0.008/min (2-core) to $0.64/min (64-core) | Up to 64 cores, 128GB RAM, 2040GB SSD | Frequent builds, automation |
| **Vultr CPU** | $0.03/hr (1 vCPU) to $0.48/hr (8 vCPU) | 1-8 vCPU, 4-32GB RAM, 10-40GB disk | One-off builds, full control |
| **AWS EC2 t3.large spot** | ~$0.03/hr | 2 vCPU, 8GB RAM | Reliable, well-known |
| **Self-hosted runner** | Hardware cost only | Your own specs | Unlimited, private |

### Recommendation

**For occasional builds**: Vultr CPU (~$0.15 for 5 min) - simplest, full Docker access  
**For automated CI**: GitHub Larger Runners - integrated, metered billing  

**Your choice depends on:**
- Build frequency (once vs daily)
- Need for automation (manual vs triggered)
- Preference for integration (GitHub native vs external)

## Setup

### 1. Add Secrets

Go to **Settings → Secrets → Actions** in your GitHub repo:

| Secret | Value | Description |
|--------|-------|-------------|
| `AWS_ACCESS_KEY_ID` | AKIAXXXXXXXXXXXXXXXX | S3 read access |
| `AWS_SECRET_ACCESS_KEY` | xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx | S3 secret key |
| `GITHUB_TOKEN` | Auto-generated | GHCR push (auto) |

### 2. S3 IAM Policy

Ensure the AWS user has this policy:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::isaac-sim-6-0-dev",
                "arn:aws:s3:::isaac-sim-6-0-dev/*"
            ]
        }
    ]
}
```

## Usage

### Manual Trigger

```bash
# Trigger via GitHub CLI
gh workflow run assemble-image.yml \
  --repo explicitcontextualunderstanding/IsaacSim \
  --ref main \
  -f build_tag=20260321-143022 \
  -f push_to_ghcr=true

# Or via web UI:
# Actions → "Assemble Isaac Sim Image" → "Run workflow"
```

### Automatic Trigger (Optional)

Set up S3 EventBridge to trigger on new builds:

```yaml
# Add to assemble-image.yml
on:
  repository_dispatch:
    types: [s3-build-complete]
```

Then in `runpod_build.sh`, add after upload:

```bash
# Trigger GitHub Actions
curl -X POST \
  -H "Authorization: token $GH_WORKFLOW_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/repos/OWNER/REPO/dispatches \
  -d "{\"event_type\":\"s3-build-complete\",\"client_payload\":{\"build_tag\":\"$BUILD_TAG\"}}"
```

## Workflow Options

| Input | Default | Description |
|-------|---------|-------------|
| `build_tag` | `latest` | Which S3 build to assemble |
| `s3_bucket` | `isaac-sim-6-0-dev` | S3 bucket name |
| `s3_prefix` | `builds` | S3 prefix path |
| `push_to_ghcr` | `true` | Push to GHCR after build |

## Complete Workflow

```bash
# Step 1: Build on RunPod GPU
./scripts/runpod_build.sh
# → Uploads to S3

# Step 2: Assemble via GitHub Actions (automatic or manual)
gh workflow run assemble-image.yml -f build_tag=20260321-143022
# → Pushes to GHCR

# Step 3: Deploy on RunPod
runpodctl create pod --image ghcr.io/.../isaac-sim-6:20260321-143022
```

## Cost Comparison

| Phase | Provider | Cost | Time |
|-------|----------|------|------|
| Compile | RunPod 4x L40S | ~$2.50 | 25 min |
| Store | S3 | ~$0.10/GB/mo | - |
| Assemble | **GitHub Actions** | **Free** | ~10 min |
| **Total Build** | | **~$2.60** | **~35 min** |

## Troubleshooting

### "Access Denied" to S3

- Check IAM policy has correct bucket ARN
- Verify secrets are set in GitHub
- Ensure S3 object exists: `aws s3 ls s3://bucket/builds/`

### "Permission denied" to GHCR

- `GITHUB_TOKEN` is auto-generated (no need to set manually)
- Ensure repo has Packages write permissions
- Check image name matches: `ghcr.io/OWNER/isaac-sim-6`

### Build too slow

- GitHub Actions has layer caching (`cache-from: type=gha`)
- First build: ~10 min
- Subsequent builds: ~3-5 min (with cache)

## Security Notes

- AWS credentials have **read-only** S3 access
- No long-lived tokens (GitHub auto-generates)
- S3 bucket should be **private**
- GHCR image is **public** (adjust visibility if needed)
