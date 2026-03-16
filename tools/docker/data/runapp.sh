#! /bin/sh

# Optional flags from env (only add if set)
EXTRA_FLAGS=""
[ -n "${OMNI_SERVER}" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --/persistent/isaac/asset_root/default=${OMNI_SERVER}"

/isaac-sim/license.sh && /isaac-sim/privacy.sh && /isaac-sim/isaac-sim.sh \
  --merge-config="/isaac-sim/config/open_endpoint.toml" \
  $EXTRA_FLAGS \
  "$@"
