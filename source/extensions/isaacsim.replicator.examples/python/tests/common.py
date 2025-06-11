# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
from collections import Counter

import numpy as np


# Check the contents of a folder against expected extension counts e.g expected_counts={png: 3, json: 3, npy: 3}
def validate_folder_contents(path: str, expected_counts: dict[str, int], include_subfolders: bool = False) -> bool:
    print(f"Validating folder contents: {path}")
    if not os.path.exists(path) or not os.path.isdir(path):
        print(f"  Folder does not exist: {path}")
        return False

    print(f"  Expected counts: {expected_counts}")
    all_files = []
    if include_subfolders:
        for root, _, files in os.walk(path):
            all_files.extend(os.path.join(root, f) for f in files)
    else:
        all_files = os.listdir(path)
    print(f"  All files in directory: {all_files}")
    file_counts = Counter(f.split(".")[-1].lower() for f in all_files if "." in f)
    print(f"  File counts: {file_counts}")

    success = all(file_counts.get(ext.lower(), 0) == count for ext, count in expected_counts.items())
    return success


# Helper function to print numpy histogram data
def print_diff_histogram(diff_array, num_bins=10):
    print(f"\t\tDiff histogram ({num_bins} bins):")
    hist_counts, bin_edges = np.histogram(diff_array, bins=num_bins)
    if np.any(hist_counts):
        for i in range(len(hist_counts)):
            bin_start = bin_edges[i]
            bin_end = bin_edges[i + 1]
            print(f"\t\t  [{bin_start:6.1f} - {bin_end:6.1f}): {hist_counts[i]}")
    else:
        print("\t\t  No differences")
