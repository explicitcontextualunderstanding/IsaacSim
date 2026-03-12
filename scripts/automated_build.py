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
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
NETWORK_VOLUME_ID = os.environ.get("NETWORK_VOLUME_ID", "")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")


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


def create_pod():
    """Create RunPod pod with RTX 5090 GPUs and proper port configuration."""
    print("🚀 Deploying 3x L40S build worker...")

    pod = runpod.create_pod(
        name="Isaac6-Source-Build",
        gpu_type_id="NVIDIA L40S",
        gpu_count=3,
        network_volume_id=NETWORK_VOLUME_ID,
        container_disk_in_gb=200,
        min_vcpu=48,
        memory="300GB",
        volume_mount_path="/workspace",
        enable_public_ip=True,
        exposed_ports=[
            {"port": 6080, "protocol": "tcp"},   # noVNC
            {"port": 8000, "protocol": "tcp"},   # Isaac Sim streaming
            {"port": 49100, "protocol": "tcp"},  # WebRTC signaling
            {"port": 47998, "protocol": "udp"},  # WebRTC video - CRITICAL
        ]
    )

    return pod


def wait_for_ssh(pod, timeout=120):
    """Wait for SSH to become available."""
    print("⏳ Waiting for SSH...")
    address = pod.get('ip', pod.get('internalIp'))
    port = 22

    for attempt in range(timeout // 10):
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(address, port=port, username='root', timeout=10)
            print(f"✅ SSH available at {address}")
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

    errors = []

    # Check RunPod API key
    if not RUNPOD_API_KEY:
        errors.append("RUNPOD_API_KEY not set")
    else:
        print(f"✅ RUNPOD_API_KEY: {'*' * len(RUNPOD_API_KEY)}")

    # Check Network Volume ID
    if not NETWORK_VOLUME_ID:
        errors.append("NETWORK_VOLUME_ID not set")
    else:
        print(f"✅ NETWORK_VOLUME_ID: {NETWORK_VOLUME_ID}")

    # Check GitHub token
    if not GITHUB_TOKEN:
        errors.append("GITHUB_TOKEN not set (needed for GHCR push)")
    else:
        print(f"✅ GITHUB_TOKEN: {'*' * len(GITHUB_TOKEN)}")

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


def main():
    """Main entry point."""
    # Check for dry-run flag
    if "--dry-run" in sys.argv or "-n" in sys.argv:
        dry_run()

    if not RUNPOD_API_KEY:
        print("❌ RUNPOD_API_KEY not set")
        sys.exit(1)

    runpod.api_key = RUNPOD_API_KEY

    # Verify GitHub identity
    github_user = verify_identity()

    # Create pod
    pod = create_pod()
    print(f"📦 Pod created: {pod['id']}")

    try:
        ssh = wait_for_ssh(pod)

        # Get public IP
        public_ip = pod.get('publicIp', pod.get('ip'))
        print(f"\n✅ Streaming access:")
        print(f"   noVNC: https://{public_ip}.runpod.io:6080/vnc.html?resize=scale")

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
