#!/bin/bash

PRIVACY_EMAIL="${PRIVACY_EMAIL:-user@example.com}"  # Allow override via environment

docker run --name isaac-sim --rm -it --gpus all --network=host \
  --entrypoint bash \
  -e ACCEPT_EULA=Y \
  -e OMNI_ENV_PRIVACY_CONSENT=Y \
  -e OMNI_ENV_PRIVACY_USERID="${PRIVACY_EMAIL}" \
  isaac-sim-docker:latest \
  "$@"
