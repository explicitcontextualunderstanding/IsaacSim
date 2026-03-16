#!/usr/bin/env bash
set -euo pipefail

# validate_container.sh
# Purpose: Comprehensive validation for Isaac Sim 6.0-dev container.
# Performs GCC check, Git LFS asset scan, ISAACSIM_PATH sanity, headless renderer check,
# ROS2 bridge import test, and LD_LIBRARY_PATH inspection. Outputs a JSON summary.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}" || true

SUMMARY_FILE="/tmp/validate_container_summary.json"
rm -f "$SUMMARY_FILE"

# Helpers
log(){ echo "[validate] $*" >&2; }
json_escape(){ python - <<PY
import json,sys
print(json.dumps(sys.stdin.read()))
PY
}

# 1) GCC Version Check (must be 11.x)
GCC_OK=false
GCC_VERSION_STR=""
if command -v gcc >/dev/null 2>&1; then
  GCC_VERSION_STR=$(gcc --version | head -n1 || true)
  if echo "$GCC_VERSION_STR" | grep -E "\b11(\.|$)" >/dev/null 2>&1; then
    GCC_OK=true
  fi
fi
log "GCC: $GCC_VERSION_STR"

# 2) Git LFS pointer scan (files <1KB under exts/omni.isaac.sim.res/data)
LFS_ISSUES=false
LFS_LIST=""
LFS_DIRS=("exts/omni.isaac.sim.res/data" "exts/omni.isaac.sim.res/data/materials")
for d in "${LFS_DIRS[@]}"; do
  if [[ -d "$d" ]]; then
    # Find regular files smaller than 1k
    mapfile -t smallfiles < <(find "$d" -type f -size -1024c -print 2>/dev/null || true)
    if [[ ${#smallfiles[@]} -gt 0 ]]; then
      LFS_ISSUES=true
      LFS_LIST+="$d: ${smallfiles[*]}\n"
    fi
  fi
done
if [[ "$LFS_ISSUES" == true ]]; then
  log "Git LFS pointer files detected (likely missing LFS pull)"
fi

# 3) ISAACSIM_PATH sanity
ISAACSIM_PATH="${ISAACSIM_PATH:-_build/linux-x86_64/release}"
ISAACSIM_PATH_ABS="$(realpath "$ISAACSIM_PATH" 2>/dev/null || true)"
ISAACSIM_OK=false
if [[ -n "$ISAACSIM_PATH_ABS" && -d "$ISAACSIM_PATH_ABS" ]]; then
  # Check for SimulationApp or kit files
  if [[ -f "$ISAACSIM_PATH_ABS/SimulationApp" || -d "$ISAACSIM_PATH_ABS/kit" ]]; then
    ISAACSIM_OK=true
  fi
fi
log "ISAACSIM_PATH -> $ISAACSIM_PATH_ABS (exists: $ISAACSIM_OK)"

# 4) Headless compatibility script
COMPAT_OK=false
COMPAT_OUTPUT=""
COMPAT_CMD_CANDIDATES=("./omni.isaac.sim.compatibility_check.sh" "./isaac-sim.compatibility_check.sh" "./scripts/validate_headless_compat.sh")
for c in "${COMPAT_CMD_CANDIDATES[@]}"; do
  if [[ -x "$c" ]]; then
    log "Running compatibility check: $c --headless"
    if OUTPUT=$($c --headless 2>&1 || true); then
      COMPAT_OUTPUT="$OUTPUT"
      if echo "$OUTPUT" | grep -E "sm_120|sm_100" >/dev/null 2>&1; then
        COMPAT_OK=true
      fi
    else
      COMPAT_OUTPUT="$OUTPUT"
    fi
    break
  fi
done

# 5) ROS2 Bridge import test (best-effort via ./python.sh)
ROS_BRIDGE_OK=false
ROS_BRIDGE_OUTPUT=""
if [[ -x "./python.sh" ]]; then
  ROS_BRIDGE_OUTPUT=$(./python.sh -c "import importlib,sys
try:
  import omni.isaac.ros2_bridge
  print('ROS2_BRIDGE_OK')
except Exception as e:
  print('ROS2_BRIDGE_FAIL', e)
  sys.exit(2)" 2>&1 || true)
  if echo "$ROS_BRIDGE_OUTPUT" | grep -q 'ROS2_BRIDGE_OK'; then
    ROS_BRIDGE_OK=true
  fi
else
  # Fallback: try system python
  ROS_BRIDGE_OUTPUT=$(python3 -c "import importlib,sys
try:
  import omni.isaac.ros2_bridge
  print('ROS2_BRIDGE_OK')
except Exception as e:
  print('ROS2_BRIDGE_FAIL', e)
  sys.exit(2)" 2>&1 || true)
  if echo "$ROS_BRIDGE_OUTPUT" | grep -q 'ROS2_BRIDGE_OK'; then
    ROS_BRIDGE_OK=true
  fi
fi
log "ROS2 Bridge load: ${ROS_BRIDGE_OK}"

# 6) LD_LIBRARY_PATH includes Isaac/ROS runtime libraries (best-effort)
LD_OK=false
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  if echo "$LD_LIBRARY_PATH" | grep -q "isaac\|ros2\|lib"; then
    LD_OK=true
  fi
fi
# Also check for libraries under ISAACSIM_PATH
if [[ "$LD_OK" == false && -n "$ISAACSIM_PATH_ABS" ]]; then
  if ls "$ISAACSIM_PATH_ABS"/lib* >/dev/null 2>&1; then
    LD_OK=true
  fi
fi

# 7) Minimal pulse check: attempt to start a headless SimulationApp (best-effort, may fail on CI)
PULSE_OK=false
PULSE_OUTPUT=""
PULSE_PY=$(mktemp -t validate_pulse_XXXX.py)
cat > "$PULSE_PY" <<'PY'
import sys
ok=False
try:
    # Try a few import variants used in different Isaac Sim builds
    try:
        from isaacsim import SimulationApp
        print('IMPORT: isaacsim.SimulationApp')
    except Exception:
        try:
            from omni.isaac.sim import SimulationApp
            print('IMPORT: omni.isaac.sim.SimulationApp')
        except Exception:
            print('IMPORT_FAIL')
            sys.exit(2)
    # Try to start headless app
    try:
        cfg={"headless": True}
        app = SimulationApp(cfg)
        print('SIM_STARTED')
        app.close()
        ok=True
    except Exception as e:
        print('SIM_FAIL', e)
        sys.exit(2)
except Exception as e:
    print('PULSE_ERROR', e)
    sys.exit(2)
if ok:
    sys.exit(0)
PY

if [[ -x "./python.sh" ]]; then
  PULSE_OUTPUT=$(./python.sh "$PULSE_PY" 2>&1 || true)
else
  PULSE_OUTPUT=$(python3 "$PULSE_PY" 2>&1 || true)
fi
if echo "$PULSE_OUTPUT" | grep -q 'SIM_STARTED'; then
  PULSE_OK=true
fi
rm -f "$PULSE_PY"

# Compose JSON summary
jq -n \
  --arg gcc_version "$GCC_VERSION_STR" \
  --argjson gcc_ok $([[ "$GCC_OK" == true ]] && echo true || echo false) \
  --arg lfs_issues "$LFS_ISSUES" \
  --arg lfs_list "$LFS_LIST" \
  --arg isaac_path "$ISAACSIM_PATH_ABS" \
  --argjson isaac_ok $([[ "$ISAACSIM_OK" == true ]] && echo true || echo false) \
  --argjson compat_ok $([[ "$COMPAT_OK" == true ]] && echo true || echo false) \
  --arg compat_output "$COMPAT_OUTPUT" \
  --argjson ros_bridge_ok $([[ "$ROS_BRIDGE_OK" == true ]] && echo true || echo false) \
  --arg ros_output "$ROS_BRIDGE_OUTPUT" \
  --argjson ld_ok $([[ "$LD_OK" == true ]] && echo true || echo false) \
  --argjson pulse_ok $([[ "$PULSE_OK" == true ]] && echo true || echo false) \
  --arg pulse_output "$PULSE_OUTPUT" \
  '{
    gcc_version: $gcc_version,
    gcc_ok: $gcc_ok,
    git_lfs_issues: ($lfs_issues == "true"),
    git_lfs_list: $lfs_list,
    isaac_path: $isaac_path,
    isaac_path_ok: $isaac_ok,
    headless_compat_ok: $compat_ok,
    headless_compat_output: $compat_output,
    ros2_bridge_ok: $ros_bridge_ok,
    ros2_bridge_output: $ros_output,
    ld_library_ok: $ld_ok,
    pulse_check_ok: $pulse_ok,
    pulse_check_output: $pulse_output
  }' > "$SUMMARY_FILE"

cat "$SUMMARY_FILE"

# Exit non-zero if any critical checks failed
CRITICAL_FAIL=false
if [[ "$GCC_OK" != true ]]; then CRITICAL_FAIL=true; fi
if [[ "$ISAACSIM_OK" != true ]]; then CRITICAL_FAIL=true; fi
if [[ "$COMPAT_OK" != true ]]; then CRITICAL_FAIL=true; fi
if [[ "$ROS_BRIDGE_OK" != true ]]; then CRITICAL_FAIL=true; fi
if [[ "$CRITICAL_FAIL" == true ]]; then
  log "CRITICAL failures detected — see JSON summary above"
  exit 2
fi

log "Validation completed — all critical checks passed"
exit 0
