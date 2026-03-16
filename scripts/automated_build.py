#!/usr/bin/env python3
"""
Isaac Sim 6.0 Container Build Orchestrator

Builds Isaac Sim 6.0 container on RunPod and optionally pushes to GHCR.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

try:
    import runpod
    import paramiko
except ImportError:
    print("Installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "runpod", "paramiko"])
    import runpod
    import paramiko


# Configuration
# Priority: 1) Environment variable, 2) 1Password via wrapper

OP_WRAPPER = os.path.expanduser("~/workspace/nano2/scripts/op_api_key_wrapper.sh")

def get_secret_from_1password(var_name):
    """Fetch secret from 1Password using the wrapper script."""
    try:
        result = subprocess.run(
            ["bash", "-c", f"source {OP_WRAPPER} --print-export {var_name}"],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0 and result.stdout:
            # Parse: export VAR=VALUE
            line = result.stdout.strip()
            if line.startswith("export "):
                _, _, value = line.partition("=")
                return value.strip().strip("'").strip('"')
    except Exception as e:
        print(f"⚠️ Failed to get {var_name} from 1Password: {e}")
    return None


def get_config():
    """Get configuration from env or 1Password."""
    config = {}

    # RUNPOD_API_KEY
    config["RUNPOD_API_KEY"] = os.environ.get("RUNPOD_API_KEY")
    if not config["RUNPOD_API_KEY"]:
        config["RUNPOD_API_KEY"] = get_secret_from_1password("RUNPOD_API_KEY")

    # NETWORK_VOLUME_ID
    config["NETWORK_VOLUME_ID"] = os.environ.get("NETWORK_VOLUME_ID")
    if not config["NETWORK_VOLUME_ID"]:
        config["NETWORK_VOLUME_ID"] = get_secret_from_1password("NETWORK_VOLUME_ID")

    # GITHUB_TOKEN (for GHCR push)
    config["GITHUB_TOKEN"] = os.environ.get("GITHUB_TOKEN")
    if not config["GITHUB_TOKEN"]:
        # Try GH_RUNNER_TOKEN as fallback
        config["GITHUB_TOKEN"] = os.environ.get("GH_RUNNER_TOKEN")
        if not config["GITHUB_TOKEN"]:
            config["GITHUB_TOKEN"] = get_secret_from_1password("GH_RUNNER_TOKEN")

    # DATA_CENTER - with fallback list
    config["DATA_CENTER"] = os.environ.get("DATA_CENTER")
    if not config["DATA_CENTER"]:
        config["DATA_CENTER"] = get_secret_from_1password("DATA_CENTER")

    # Fallback data centers if primary is exhausted
    # Fallback data centers - skip US-NC-1 if it's exhausted
    config["FALLBACK_DCS"] = os.environ.get("RUNPOD_FALLBACK_DCS", "US-CA-1,CA-MTL-1,EU-CZ-1,EU-RO-1,AP-SE-1").split(",")

    # CLOUD_TYPE (SECURE or COMMUNITY)
    config["CLOUD_TYPE"] = os.environ.get("RUNPOD_CLOUD_TYPE", "SECURE")

    return config


def verify_identity():
    """Verify GitHub identity matches expected user."""
    result = subprocess.run(
        ["gh", "api", "user", "--jq", ".login"],
        capture_output=True,
        text=True
    )
    user = result.stdout.strip()
    expected = "explicitcontextualunderstanding"
    if user != expected:
        raise RuntimeError(f"Wrong GitHub user: {user}. Expected: {expected}")
    print(f"✅ Verified identity: {user}")
    return user


def find_available_regions(min_vram_gb=24):
    """Query which regions have GPUs available using GraphQL."""
    import requests

    # Query all GPUs and filter in Python
    query = """
    query {
      gpuTypes {
        id
        displayName
        memoryInGb
        stockStatus
        nodeGroupDatacenters {
          id
          name
        }
      }
    }
    """

    try:
        response = requests.post(
            "https://api.runpod.io/graphql",
            headers={"Authorization": f"Bearer {runpod.api_key}"},
            json={"query": query},
            timeout=30
        )

        if response.status_code != 200:
            print(f"⚠️ Region query failed: {response.status_code}")
            return None

        data = response.json()
        if "errors" in data:
            print(f"⚠️ GraphQL error: {data['errors'][0]['message']}")
            return None

        gpus = data.get("data", {}).get("gpuTypes", [])

        # Collect all datacenter + GPU combinations
        region_gpu_map = {}
        for gpu in gpus:
            vram = gpu.get("memoryInGb", 0)
            stock = gpu.get("stockStatus", "Unknown")
            if vram >= min_vram_gb and stock != "Out":
                datacenters = gpu.get("nodeGroupDatacenters", [])
                for dc in datacenters:
                    dc_id = dc.get("id")
                    if dc_id:
                        if dc_id not in region_gpu_map:
                            region_gpu_map[dc_id] = []
                        region_gpu_map[dc_id].append({
                            "gpu": gpu.get("displayName"),
                            "vram": vram,
                            "stock": stock
                        })

        return region_gpu_map

    except Exception as e:
        print(f"⚠️ Region query failed: {e}")
        return None


def find_best_gpu(min_vram_gb=24, min_ram_gb=64):
    """Find cheapest GPU meeting requirements, prioritizing regions with availability."""
    # First, try to find which regions have GPUs
    print("\n🌍 Scanning for available regions...")
    region_map = find_available_regions(min_vram_gb)

    if region_map:
        print("📍 Regions with available GPUs:")
        for dc, gpus in sorted(region_map.items(), key=lambda x: -len(x[1])):
            best = max(gpus, key=lambda g: g["vram"])
            print(f"   {dc}: {best['gpu']} ({best['vram']}GB) - stock: {best['stock']}")

    # Get all GPUs from runpod SDK
    all_gpus = runpod.get_gpus()

    if not all_gpus:
        print("❌ No GPUs available")
        return None

    # Filter by VRAM requirement
    candidates = []
    for gpu in all_gpus:
        vram = gpu.get("memoryInGb", 0)
        if vram >= min_vram_gb:
            price = gpu.get("lowestPrice", {}).get("uninterruptablePrice") or 0
            stock = gpu.get("stockStatus", "Unknown")
            candidates.append({
                "id": gpu.get("id"),
                "display": gpu.get("displayName"),
                "vram": vram,
                "price": price,
                "stock": stock,
            })

    if not candidates:
        print(f"❌ No GPUs with {min_vram_gb}GB+ VRAM available")
        return None

    # Sort by price (cheapest first)
    candidates.sort(key=lambda x: x["price"] if x["price"] else float('inf'))

    print(f"\n📊 GPUs meeting requirements (VRAM >= {min_vram_gb}GB):")
    for i, gpu in enumerate(candidates[:10]):  # Top 10
        price_str = f"${gpu['price']}/hr" if gpu['price'] else "price unknown"
        print(f"   {i+1}. {gpu['display']} - {gpu['vram']}GB VRAM @ {price_str} (stock: {gpu['stock']})")

    # Attach region info if available
    if region_map:
        for gpu in candidates:
            gpu["regions"] = region_map

    return candidates


def create_pod(config, gpu_info, data_center=None, cloud=None, spot=False):
    """Create RunPod pod with resources based on GPU specs."""
    gpu_type_id = gpu_info["id"]
    vram = gpu_info.get("vram", 24)

    # Scale CPU/RAM based on VRAM (compilation needs ~2x VRAM for temp space)
    min_ram_gb = max(64, vram * 2)  # At least 64GB or 2x VRAM
    min_vcpu = min(48, max(16, vram * 2))  # 2 vCPUs per GB VRAM, capped

    dc = data_center or config.get("DATA_CENTER") or "US-NC-1"
    instance_type = "SPOT" if spot else cloud or "SECURE"
    print(f"🚀 Deploying {instance_type} worker with {gpu_type_id} ({vram}GB VRAM) in {dc}")
    print(f"   Requesting: {min_ram_gb}GB RAM, {min_vcpu} vCPUs")

    # CUDA 13.1 + Ubuntu 24.04 for Blackwell (sm_100) support
    pod_args = {
        "name": "Isaac6-Source-Build",
        "image_name": "nvidia/cuda:13.1.1-devel-ubuntu24.04",
        "gpu_type_id": gpu_type_id,
        "gpu_count": 1,
        "container_disk_in_gb": 100,
        "min_vcpu_count": min_vcpu,
        "min_memory_in_gb": min_ram_gb,
        "volume_mount_path": "/workspace",
        "support_public_ip": True,
        "command": ["sleep", "infinity"],
    }

    # Add data center if provided
    if dc:
        pod_args["data_center_id"] = dc

    # Add network volume if provided (critical for spot - data persists on volume)
    if config.get("NETWORK_VOLUME_ID"):
        pod_args["network_volume_id"] = config["NETWORK_VOLUME_ID"]

    if spot:
        # Spot instances - will use interruptable pricing if available
        pod = runpod.create_pod(**pod_args)
    elif cloud:
        pod_args["cloud_type"] = cloud
        pod = runpod.create_pod(**pod_args)
    else:
        pod = runpod.create_pod(**pod_args)

    return pod


def create_pod_with_backoff(config, gpu, data_center=None, cloud=None, spot=False, max_retries=2):
    """Create pod with exponential backoff on rate limit."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return create_pod(config, gpu, data_center=data_center, cloud=cloud, spot=spot)
        except Exception as e:
            last_error = str(e)
            # Check if it's a retryable error
            retryable = any(x in last_error.lower() for x in [
                "no longer any instances available",
                "something went wrong",
                "try again later",
                "rate limit",
                "too many requests"
            ])
            if retryable and attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 2  # 2, 4 seconds
                print(f"   ⏳ {last_error[:40]}... retry in {wait_time}s...")
                time.sleep(wait_time)
                continue

    # All retries exhausted - return None so we continue to next GPU/region
    if last_error:
        print(f"   ❌ {gpu['id']} in {data_center}: {last_error[:40]}...")
    return None


def create_pod_with_fallback(config, available_gpus):
    """Try creating pod with fallback: spot first, then on-demand, across regions."""
    primary_dc = config.get("DATA_CENTER") or "US-NC-1"
    fallback_dcs = config.get("FALLBACK_DCS", [])

    # Prioritize non-NC regions, put US-NC-1 last
    all_dcs = fallback_dcs + [primary_dc] if primary_dc not in fallback_dcs else fallback_dcs
    data_centers = [d for d in all_dcs if d != "US-NC-1"] + ["US-NC-1"]

    # Try SPOT instances first (cheaper, more available)
    print("\n🔄 Trying SPOT instances (60% cheaper)...")
    for dc in data_centers:
        print(f"   Trying {dc}...")
        for gpu in available_gpus:
            pod = create_pod_with_backoff(config, gpu, data_center=dc, spot=True)
            if pod:
                return pod
            # If None, continue to next GPU/region

    # Fall back to on-demand
    cloud_types = ["SECURE", "COMMUNITY"]
    for cloud in cloud_types:
        for dc in data_centers:
            print(f"\n🔄 Trying {cloud} cloud in {dc} (on-demand)...")
            for gpu in available_gpus:
                pod = create_pod_with_backoff(config, gpu, data_center=dc, cloud=cloud)
                if pod:
                    return pod
                # If None, continue to next GPU/region

    raise RuntimeError("No GPUs available")


def wait_for_ssh(pod, timeout=120):
    """Wait for SSH to become available."""
    # Try multiple field names for IP address
    ip_address = pod.get('ipAddress', {})
    if isinstance(ip_address, dict):
        address = ip_address.get('address', pod.get('ip', pod.get('clusterIp')))
    else:
        address = pod.get('ip', pod.get('clusterIp'))

    # Public IP - check machine level
    public_ip = pod.get('publicIp') or pod.get('runpodIp')

    port = 22

    print("⏳ Waiting for SSH...")
    print(f"   Address: {address}")
    print(f"   Public IP: {public_ip}")
    print(f"   SSH: ssh root@{public_ip or address}")

    for attempt in range(timeout // 10):
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(address, port=port, username='root', timeout=10)
            print(f"✅ SSH available at {address}")
            print(f"   Connect with: ssh root@{public_ip}")
            return ssh
        except Exception as e:
            print(f"  Attempt {attempt + 1}: {e}")
            time.sleep(10)

    raise RuntimeError(f"SSH not available after {timeout}s")


def execute_command(ssh, cmd):
    """Execute command on remote host."""
    print(f"🛠️ {cmd}")
    stdin, stdout, stderr = ssh.exec_command(f"bash -l -c '{cmd}'")

    # Stream stdout
    for line in stdout:
        print(line.strip())

    # Check stderr for errors
    stderr_output = stderr.read().decode()
    if stderr_output and "error" in stderr_output.lower():
        print(f"⚠️ STDERR: {stderr_output}")

    exit_code = stdout.channel.recv_exit_status()
    return exit_code


def run_validation_gate(ssh_client):
    """
    Executes the self-healing validation script on the remote RunPod instance.
    """
    print("🚀 Triggering Phase 0: Self-Healing Validation Gate...")
    
    # Run the script with a generous timeout for potential Git LFS pulls
    stdin, stdout, stderr = ssh_client.exec_command(
        "/workspace/IsaacSim/scripts/validate_container.sh", 
        timeout=900  # 15-minute window for LFS and shader warmup
    )
    
    # Stream output in real-time for visibility
    for line in stdout:
        print(f"[Remote]: {line.strip()}")
        
    exit_status = stdout.channel.recv_exit_status()
    if exit_status != 0:
        error_msg = stderr.read().decode().strip()
        print(f"❌ Validation Failed with exit code {exit_status}: {error_msg}")
        raise RuntimeError("Pod failed validation gate. Terminating to save costs.")
    
    print("✅ Validation Passed. Proceeding to production task.")


def run_build(ssh, github_user):
    """Run Isaac Sim build commands."""
    build_commands = [
        # Set EULA acceptance
        "export OMNI_KIT_ACCEPT_EULA=YES",
        # Streaming: bind to 0.0.0.0 for external access
        "export OMNI_KIT_IP=0.0.0.0",
        "export OMNI_KIT_STREAM_PORT=6080",

        # Clone/update Isaac Sim source
        "cd /workspace && git clone https://github.com/NVIDIA-Omniverse/IsaacSim.git || true",
        "cd /workspace/IsaacSim && git checkout v6.0.0-dev && git pull",

        # Build
        "./build.sh --skip-compiler-version-check -j 36",

        # Package container
        "./repo.sh package_container --app isaacsim.exp.full.kit --platform linux/amd64",
    ]

    for cmd in build_commands:
        exit_code = execute_command(ssh, cmd)
        if exit_code != 0:
            print(f"❌ Command failed with exit code {exit_code}")
            return False

    return True


def push_to_ghcr(ssh, github_user):
    """Push container to GHCR."""
    # Authenticate
    execute_command(ssh, f"echo $GITHUB_TOKEN | docker login ghcr.io -u {github_user} --password-stdin")

    # Get image tag
    execute_command(ssh, "docker images --format '{{.Repository}}:{{.Tag}}' | head -1")

    # Tag and push
    # Note: Adjust based on actual image name from package_container
    execute_command(ssh, f"docker tag isaac-sim-6-dev ghcr.io/{github_user}/isaac-sim-6-dev:latest")
    execute_command(ssh, f"docker push ghcr.io/{github_user}/isaac-sim-6-dev:latest")


def dry_run():
    """Verify credentials without spinning up a server."""
    print("🔍 Running dry-run to verify credentials...\n")

    # Fetch config from env or 1Password
    config = get_config()

    errors = []

    # Check RunPod API key
    if not config.get("RUNPOD_API_KEY"):
        errors.append("RUNPOD_API_KEY not set")
    else:
        print(f"✅ RUNPOD_API_KEY: {'*' * len(config['RUNPOD_API_KEY'])}")

    # Check Network Volume ID
    if not config.get("NETWORK_VOLUME_ID"):
        errors.append("NETWORK_VOLUME_ID not set")
    else:
        print(f"✅ NETWORK_VOLUME_ID: {config['NETWORK_VOLUME_ID']}")

    # Check GitHub token
    if not config.get("GITHUB_TOKEN"):
        print("⚠️ GITHUB_TOKEN not set (GHCR push disabled)")
    else:
        print(f"✅ GITHUB_TOKEN: {'*' * len(config['GITHUB_TOKEN'])}")

    # Check Data Center
    if config.get("DATA_CENTER"):
        print(f"✅ DATA_CENTER: {config['DATA_CENTER']}")

    # Verify GitHub identity
    try:
        result = subprocess.run(
            ["gh", "api", "user", "--jq", ".login"],
            capture_output=True,
            text=True
        )
        user = result.stdout.strip()
        if user != "explicitcontextualunderstanding":
            errors.append(f"Wrong GitHub user: {user}. Expected: explicitcontextualunderstanding")
        else:
            print(f"✅ GitHub identity: {user}")
    except Exception as e:
        errors.append(f"Failed to verify GitHub identity: {e}")

    print()
    if errors:
        print("❌ Dry-run failed:")
        for err in errors:
            print(f"   - {err}")
        sys.exit(1)
    else:
        print("✅ All credentials verified!")
        print("\nTo run the build:")
        print("   export RUNPOD_API_KEY='your-key'")
        print("   export NETWORK_VOLUME_ID='vol-xxx'")
        print("   export PUSH_TO_GHCR=true  # optional")
        print("   python scripts/automated_build.py")
        sys.exit(0)


def preflight(config, github_user):
    """Run preflight test with minimal GPU to validate infrastructure."""
    print("🧪 Running preflight test with minimal GPU...")

    # Set API key
    runpod.api_key = config["RUNPOD_API_KEY"]

    # Create minimal pod (RTX 4060)
    print("🚀 Deploying preflight pod (L40S, 8 vCPU, 16GB)...")

    pod = runpod.create_pod(
        name="isaac-preflight",
        image_name="alpine:latest",
        gpu_type_id="NVIDIA L40S",
        gpu_count=1,
        container_disk_in_gb=30,
        min_vcpu_count=8,
        min_memory_in_gb=16,
        support_public_ip=True,
        data_center_id=config.get("DATA_CENTER", "US-NC-1"),
        command=["sleep", "infinity"],
    )

    print(f"📦 Preflight pod created: {pod['id']}")
    print(f"   Internal IP: {pod.get('ip', 'N/A')}")
    print(f"   Public IP: {pod.get('publicIp', 'N/A')}")
    if pod.get('publicIp'):
        print(f"   SSH: ssh root@{pod.get('publicIp')}")

    try:
        ssh = wait_for_ssh(pod)

        # Build minimal alpine container
        print("📦 Building minimal alpine container...")

        # Create Dockerfile
        dockerfile = """FROM alpine:latest
RUN apk add --no-cache curl
CMD ["echo", "preflight-success"]
"""
        execute_command(ssh, f"echo '{dockerfile}' > /tmp/Dockerfile")

        # Build and tag
        execute_command(ssh, "cd /tmp && docker build -t preflight-test:latest -f Dockerfile .")

        # Login to GHCR
        if config.get("GITHUB_TOKEN"):
            execute_command(ssh, f"echo $GITHUB_TOKEN | docker login ghcr.io -u {github_user} --password-stdin")
        else:
            print("⚠️ No GITHUB_TOKEN, skipping GHCR push")
            return

        # Tag and push
        execute_command(ssh, f"docker tag preflight-test:latest ghcr.io/{github_user}/preflight-test:latest")
        execute_command(ssh, f"docker push ghcr.io/{github_user}/preflight-test:latest")

        print("✅ Preflight complete!")

    finally:
        print("🧹 Terminating preflight pod...")
        runpod.terminate_pod(pod['id'])


def main():
    """Main entry point."""
    # Check for dry-run flag
    if "--dry-run" in sys.argv or "-n" in sys.argv:
        dry_run()
        sys.exit(0)

    # Check for preflight flag
    if "--preflight" in sys.argv:
        config = get_config()
        github_user = verify_identity()
        preflight(config, github_user)
        sys.exit(0)

    # Fetch config from env or 1Password
    config = get_config()

    if not config.get("RUNPOD_API_KEY"):
        print("❌ RUNPOD_API_KEY not set")
        print("   Set env var or configure in ~/.1password-secrets.env")
        sys.exit(1)

    runpod.api_key = config["RUNPOD_API_KEY"]

    # Verify GitHub identity
    github_user = verify_identity()

    # Check if build-only mode (accept any GPU for building)
    build_only = os.environ.get("BUILD_ONLY", "false").lower() == "true"
    if build_only:
        print("\n🔧 BUILD_ONLY mode: Accepting any available GPU for compilation")
        print("   (Just need nvidia-container-toolkit, not Blackwell)")
        min_vram = 8  # Any GPU will work for build
    else:
        min_vram = 24  # Need 24GB+ for runtime

    # Check GPU availability and select best option
    print(f"\n🔍 Checking GPU availability (min {min_vram}GB VRAM)...")
    available_gpus = find_best_gpu(min_vram_gb=min_vram)
    if not available_gpus:
        print("❌ No suitable GPUs available. Try again later.")
        sys.exit(1)

    # Create pod with fallback
    pod = create_pod_with_fallback(config, available_gpus)
    print(f"📦 Pod created: {pod['id']}")

    try:
        ssh = wait_for_ssh(pod)

        # Get public IP
        public_ip = pod.get('publicIp', pod.get('ip'))
        print(f"\n✅ Streaming access:")
        print(f"   noVNC: https://{public_ip}.runpod.io:6080/vnc.html?resize=scale")

        # Phase 0: Self-Healing Validation Gate
        try:
            run_validation_gate(ssh)
        except Exception as e:
            print(f"\n❌ Validation Gate Error: {e}")
            return  # Terminate pod immediately

        # Run build
        success = run_build(ssh, github_user)

        if success:
            print("\n✅ Build complete!")

            # Optional: Push to GHCR
            if os.environ.get("PUSH_TO_GHCR", "").lower() == "true":
                push_to_ghcr(ssh, github_user)
        else:
            print("\n❌ Build failed")

    finally:
        print("🧹 Terminating Pod...")
        runpod.terminate_pod(pod['id'])


if __name__ == "__main__":
    main()
