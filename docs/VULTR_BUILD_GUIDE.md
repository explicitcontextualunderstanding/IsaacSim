# Vultr Build Guide (LEGACY) - Use Hybrid Workflow Instead

> ⚠️ **DEPRECATED**: This guide is for **legacy full builds on Vultr only** (~$10-15, 4-6 hours).
>
> **RECOMMENDED**: Use [Hybrid GPU→S3→CPU Workflow](HYBRID_BUILD_WORKFLOW.md) instead (~$2.65, 30 min).
>
> This document is kept for reference. For new builds, use hybrid approach.

## Quick Decision

| Scenario | Use This | Cost | Time |
|----------|----------|------|------|
| **New Build** | [Hybrid Workflow](HYBRID_BUILD_WORKFLOW.md) | ~$2.65 | 30 min |
| **Emergency/Debug** | This guide (full Vultr) | ~$10-15 | 4-6 hours |
| **CPU Reassembly Only** | Hybrid Phase 3 | ~$0.20 | 5 min |

## Legacy Architecture (Full Vultr Build)

```mermaid
flowchart TB
    subgraph Build["🔨 Build Phase (Vultr)"]
        V[Vultr GPU Instance<br/>CUDA 13.1+ Builder]
        D[Docker Build]
        V --> D
    end

    subgraph Registry["📦 Container Registry"]
        GHCR[GHCR.io<br/>Isaac Sim Images]
    end

    subgraph Runtime["🚀 Runtime Phase (RunPod)"]
        RP[RunPod GPU Pod]
        NV[(Network Volume<br/>Persistent Storage)]
        RP --> NV
    end

    subgraph Storage["💾 Object Storage"]
        S3[S3 Bucket<br/>isaac-sim-6-0-dev]
    end

    D -->|docker push| GHCR
    GHCR -->|docker pull| RP
    S3 -.->|Large binaries| RP
    S3 -.->|Backup/Cache| V
```

## Prerequisites
- Vultr GPU instance with Docker access
- GitHub Personal Access Token with `write:packages` scope
- CUDA 13.1+ capable GPU (RTX Pro 6000, RTX 5090, H100, etc.)

## Quick Start

### 1. SSH into your Vultr instance
```bash
ssh root@<your-vultr-ip>
```

### 2. Download and run the build script

**Prerequisites:** GitHub token with `write:packages` scope ([create token](https://github.com/settings/tokens))

```mermaid
sequenceDiagram
    participant U as User
    participant V as Vultr Instance
    participant D as Docker
    participant G as GHCR.io

    U->>V: SSH connect
    V->>V: Install Docker (if needed)
    U->>V: Download build script
    U->>V: Set GITHUB_TOKEN env var
    V->>V: Run manual_vultr_build.sh
    V->>D: docker build -f Dockerfile.cuda13
    D->>V: Build complete (~10-15 min)
    V->>G: docker push ghcr.io/...
    G-->>V: Image pushed ✅
    V-->>U: Build complete
    U->>V: Terminate instance (important!)
```

```bash
# Set your GitHub token (for GHCR push)
export GITHUB_TOKEN=ghp_xxxxxxxx

# Run build
curl -fsSL https://raw.githubusercontent.com/explicitcontextualunderstanding/IsaacSim/main/scripts/manual_vultr_build.sh | bash
```

Or manually:
```bash
# Copy the Dockerfile.cuda13 from this repo
# Copy scripts/manual_vultr_build.sh
chmod +x manual_vultr_build.sh
export GITHUB_TOKEN=ghp_xxxxxxxx
./manual_vultr_build.sh
```

### 3. Verify the image
```bash
# Test locally
docker run --gpus all -it ghcr.io/explicitcontextualunderstanding/isaac-sim-6-cuda13.1-base:latest nvidia-smi

# Check CUDA version
docker run --gpus all -it ghcr.io/explicitcontextualunderstanding/isaac-sim-6-cuda13.1-base:latest nvcc --version
```

## Manual Build (Alternative)

If the script doesn't work:

```bash
# 1. Login to GHCR
export GITHUB_TOKEN=ghp_your_token_here
echo "$GITHUB_TOKEN" | docker login ghcr.io -u explicitcontextualunderstanding --password-stdin

# 2. Build
docker build -f Dockerfile.cuda13 -t ghcr.io/explicitcontextualunderstanding/isaac-sim-6-cuda13.1-base:latest .

# 3. Push
docker push ghcr.io/explicitcontextualunderstanding/isaac-sim-6-cuda13.1-base:latest
```

## Service Roles & Data Flow

```mermaid
flowchart LR
    subgraph Sources["Data Sources"]
        S3Bin[Isaac Sim Binaries<br/>16.7GB tar]
        GHImg[Container Image<br/>~1-2GB layers]
    end

    subgraph Access["Access Methods"]
        AWS[AWS CLI<br/>aws s3 cp]
        Docker[Docker CLI<br/>docker pull/push]
    end

    subgraph Target["Runtime Targets"]
        Vultr[Vultr GPU<br/>Build Environment]
        RunPod[RunPod Pod<br/>Execution Environment]
    end

    S3Bin -->|Large artifacts| AWS
    GHImg -->|Container images| Docker
    AWS -->|Pull binaries| Vultr
    AWS -->|Pull binaries| RunPod
    Docker -->|Push images| Vultr
    Docker -->|Pull images| RunPod
```

### When to Use Each Service

| Service | Best For | Avoid For | Cost |
|---------|----------|-----------|------|
| **Vultr** | Building images with full Docker access | Running workloads 24/7 | ~$1.50-3/hr |
| **GHCR.io** | Storing container images | Large binary files (>2GB) | Free (public) |
| **S3** | Large artifacts, backups | Container images | ~$0.023/GB/mo |
| **Network Volume** | Runtime persistence, caches | Long-term storage | ~$0.07/GB/mo |

## Cost Optimization

```mermaid
pie title Monthly Cost Breakdown (Typical Usage)
    "RunPod Spot GPU" : 65
    "Network Volume (50GB)" : 10
    "S3 Storage (20GB)" : 2
    "Vultr Build (1hr)" : 1
    "GHCR (Public)" : 0
```

**Typical Monthly Costs:**
- **Vultr always-on**: ~$850/mo ❌
- **Vultr (build only)** + RunPod spot: ~$50-100/mo ✅

## RunPod Template Update

After successful build, update your RunPod template:

| Field | Value |
|-------|-------|
| Template ID | `hx1b4w5i60` |
| Image | `ghcr.io/explicitcontextualunderstanding/isaac-sim-6-cuda13.1-base:latest` |
| Docker Command | `rm -rf /workspace/IsaacSim 2>/dev/null \|\| true` |
| Network Volume | `chemical_lavender_lamprey` (xssve1bbu4) |

## Troubleshooting

### CUDA version check
```bash
nvidia-smi  # Should show CUDA 13.1+ capable driver (570.169+)
nvcc --version  # Should show 13.1+
```

### Docker build fails
- Ensure Docker daemon is running: `systemctl status docker`
- Check disk space: `df -h`
- Try with `--no-cache` flag

### GHCR push fails
- Verify token has `write:packages` scope: `gh auth status`
- Check package visibility at: https://github.com/users/explicitcontextualunderstanding/packages

## Complete Workflow

```mermaid
flowchart TB
    subgraph Phase1["Phase 1: Build"]
        A[Developer Machine] -->|git push| B[GitHub Fork]
        C[Vultr GPU] -->|docker build| D[Base Image]
        D -->|docker push| E[GHCR.io]
    end

    subgraph Phase2["Phase 2: Deploy"]
        F[RunPod Template] -->|references| E
        G[RunPod Pod] -->|pulls image| E
        G -->|mounts| H[(Network Volume)]
        G -->|pulls| I[S3 Binaries]
    end

    subgraph Phase3["Phase 3: Run"]
        G -->|executes| J[Isaac Sim 6.0]
        J -->|streaming| K[noVNC Client]
        J -->|logs| H
    end

    Phase1 -.->|image ready| Phase2
    Phase2 -.->|pod running| Phase3
```

## Troubleshooting Flow

```mermaid
flowchart TD
    Start[Build Fails] --> CheckDriver{Driver 570+?}
    CheckDriver -->|No| UpdateDriver[Update NVIDIA Driver]
    CheckDriver -->|Yes| CheckDocker{Docker Running?}

    CheckDocker -->|No| StartDocker[systemctl start docker]
    CheckDocker -->|Yes| CheckDisk{Disk Space?}

    CheckDisk -->|<20GB| CleanDisk[docker system prune -a]
    CheckDisk -->|OK| CheckAuth{GHCR Auth?}

    CheckAuth -->|No| Auth[docker login ghcr.io]
    CheckAuth -->|Yes| CheckToken{Token Scope?}

    CheckToken -->|Missing| RegenToken[Create token with<br/>write:packages scope]
    CheckToken -->|OK| Retry[Retry Build]

    UpdateDriver --> Retry
    StartDocker --> Retry
    CleanDisk --> Retry
    Auth --> Retry
    RegenToken --> Retry
```

## CUDA 13.1+ GPU Compatibility

| GPU | CUDA Support | Available on Vultr |
|-----|-------------|-------------------|
| RTX Pro 6000 Blackwell | 13.1+ | ✓ |
| RTX 5090 | 13.1+ | ✓ |
| RTX 4090 | 12.0+ | ✓ |
| H100 | 13.1+ | ✓ |
| H200 | 13.1+ | ✓ |
| L40S | 12.1 max | ✗ |

Note: RTX 4090 may work with CUDA 13.1+ depending on driver version.
