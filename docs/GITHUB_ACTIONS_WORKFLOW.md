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
