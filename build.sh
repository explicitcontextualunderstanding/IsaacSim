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

SCRIPT_DIR=$(dirname ${BASH_SOURCE})

# Check EULA acceptance first
"${SCRIPT_DIR}/tools/eula_check.sh"
EULA_STATUS=$?

if [ $EULA_STATUS -ne 0 ]; then
    echo "Error: NVIDIA Software License Agreement and Product-Specific Terms for NVIDIA Omniverse must be accepted to proceed."
    exit 1
fi

set -e

# Check compiler version and fail early if it is not GCC 11.
# This prevents "silent killer" runtime failures (segfaults / undefined symbols) caused by
# building with GCC 12/13 on Ubuntu 24.04.
SKIP_COMPILER_VERSION_CHECK=0
NEW_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--skip-compiler-version-check" ]]; then
        SKIP_COMPILER_VERSION_CHECK=1
    else
        NEW_ARGS+=("$arg")
    fi
done

if [[ "$SKIP_COMPILER_VERSION_CHECK" -eq 0 ]]; then
    if command -v gcc >/dev/null 2>&1; then
        GCC_MAJOR=$(gcc -dumpversion | cut -d. -f1)
        if [[ "$GCC_MAJOR" != "11" ]]; then
            echo "\nERROR: Isaac Sim requires GCC 11 on Ubuntu 24.04."
            echo "       Detected GCC: $(gcc --version | head -n1)"
            echo "       Fix: install gcc-11/g++-11 and make them the default via update-alternatives."
            echo "       Or rerun with --skip-compiler-version-check (unsupported).\n"
            exit 1
        fi
    else
        echo "\nERROR: gcc not found. Please install gcc-11 and g++-11.\n"
        exit 1
    fi
fi

source "$SCRIPT_DIR/repo.sh" build "${NEW_ARGS[@]}" || exit $?
