#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e
SCRIPT_DIR=$(dirname ${BASH_SOURCE})

shopt -s globstar
args=""

# Check if we are running in a docker container
if [ -f /.dockerenv ]; then
    echo "running as root"
    args="$args --allow-root"

    # Check for vulkan in docker container
    if [[ -f "${SCRIPT_DIR}/vulkan_check.sh" ]]; then
      ${SCRIPT_DIR}/vulkan_check.sh
    fi
fi

if [[ -z "${DISPLAY}" ]]; then
    echo "running headless"
    args="$args --no-window"
fi

pushd $SCRIPT_DIR/tests/
for f in tests-omni.isaac.benchmarks*.sh; do
    echo "Executing Test: $f"
    bash "$f" $args $@
done
for f in tests-standalone_benchmarks*.sh; do
    echo "Executing Test: $f"
    bash "$f" $args $@
done
popd