#! /bin/sh

# Optional flags from env (only add each if set); pass before "$@" so docker run args can override
EXTRA_FLAGS=""
[ -n "${OMNI_SERVER}" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --/persistent/isaac/asset_root/default=${OMNI_SERVER}"
[ -n "${ISAACSIM_HOST}" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --/exts/omni.kit.livestream.app/primaryStream/publicIp=${ISAACSIM_HOST}"
[ -n "${ISAACSIM_SIGNAL_PORT}" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --/exts/omni.kit.livestream.app/primaryStream/signalPort=${ISAACSIM_SIGNAL_PORT}"
[ -n "${ISAACSIM_STREAM_PORT}" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --/exts/omni.kit.livestream.app/primaryStream/streamPort=${ISAACSIM_STREAM_PORT}"

/isaac-sim/license.sh && /isaac-sim/privacy.sh && /isaac-sim/isaac-sim.streaming.sh \
    --merge-config="/isaac-sim/config/open_endpoint.toml" \
    $EXTRA_FLAGS \
    "$@"
