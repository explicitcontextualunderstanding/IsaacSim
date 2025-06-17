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
import os
from glob import glob

NVIDIA_PROPRIETARY_TEXT = [
    # First format
    """NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto.  Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.""",
    # Second format
    """NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
property and proprietary rights in and to this material, related
documentation and any modifications thereto. Any use, reproduction,
disclosure or distribution of this material and related documentation
without an express license agreement from NVIDIA CORPORATION or
its affiliates is strictly prohibited.""",
]

MIT_LICENSE_TEXT = [
    # First paragraph
    """Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:""",
    # Key phrases that indicate MIT license
    "Permission is hereby granted, free of charge",
    'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND',
    "INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY",
    "FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT",
]


def find_license_file(link_path, package_name, config_tags=None):
    """Search for license files in common locations.

    Args:
        link_path: Base path to search for license files
        package_name: Name of the package to find licenses for
        config_tags: Dictionary of configuration tags to replace in the link path (e.g.,
            {'config': 'release', 'platform_target': 'linux-x86_64', 'platform_target_abi': 'manylinux_2_35_x86_64'})

    Returns:
        Union[dict, list, str, None]: License information found, or None if no license found
    """
    if config_tags is None:
        config_tags = {"config": "release", "platform_target": "linux-x86_64", "platform_target_abi": "linux-x86_64"}

    # Replace variables in link_path
    for tag, value in config_tags.items():
        link_path = link_path.replace(f"${{{tag}}}", value)

    # Make path absolute if relative, starting from project root
    if link_path.startswith("../"):
        # Get the project root (parent of deps directory)
        script_dir = os.path.dirname(os.path.abspath(__file__))  # tools/check_externals
        project_root = os.path.dirname(os.path.dirname(script_dir))  # Go up two levels to project root
        link_path = os.path.abspath(os.path.join(project_root, link_path.lstrip("../")))

    all_matches = set()
    package_specific_matches = set()
    root_license_matches = set()

    # First pass: categorize files without scanning content
    possible_patterns = [
        f"PACKAGE-LICENSES/{package_name}-LICENSE*",  # Package-specific licenses first
        f"PACKAGE-LICENSES/{package_name.lower()}-LICENSE*",
        "LICENSE*",  # Root licenses next
        "*.LICENSE",
        "*.license",
        "COPYING*",
        "PACKAGE-LICENSES/*LICENSE*",  # Generic package licenses last
        "licenses/LICENSE*",
        "LICENSES/*",
    ]
    for pattern in possible_patterns:
        full_pattern = os.path.join(link_path, pattern)
        matches = glob(full_pattern, recursive=True)
        matches = [m for m in matches if os.path.isfile(m)]
        for match in matches:
            rel_path = os.path.relpath(match)

            # Skip PIP-packages-LICENSES.txt and duplicates
            if os.path.basename(match) == "PIP-packages-LICENSES.txt":
                continue
            if rel_path in all_matches:
                continue

            # Categorize the file
            if package_name.lower() in os.path.basename(match).lower():
                package_specific_matches.add(rel_path)
            elif os.path.basename(match) == "LICENSE.txt":
                # Check if this is directly in the package directory (not in a subdirectory)
                match_dir = os.path.dirname(os.path.abspath(match))
                package_dir = os.path.abspath(link_path)
                if match_dir == package_dir:
                    root_license_matches.add(rel_path)
                else:
                    all_matches.add(rel_path)
            else:
                all_matches.add(rel_path)

    # Second pass: scan content of relevant files
    spdx_matches = []
    nvidia_proprietary_matches = []
    mit_license_matches = []

    files_to_scan = package_specific_matches or root_license_matches or all_matches
    for rel_path in files_to_scan:
        try:
            with open(rel_path, "r", encoding="utf-8") as f:
                content = f.read()
                content_normalized = content.replace("\n", " ").strip()
                if any(text.replace("\n", " ").strip() in content_normalized for text in NVIDIA_PROPRIETARY_TEXT):
                    nvidia_proprietary_matches.append(rel_path)
                if any(indicator.replace("\n", " ").strip() in content_normalized for indicator in MIT_LICENSE_TEXT):
                    mit_license_matches.append(rel_path)
                if "SPDX-License-Identifier:" in content:
                    spdx_line = next(line for line in content.splitlines() if "SPDX-License-Identifier:" in line)
                    spdx_type = spdx_line.split("SPDX-License-Identifier:", 1)[1].strip()
                    spdx_matches.append((rel_path, spdx_type))
        except Exception as e:
            pass

    # Return results in priority order
    if len(nvidia_proprietary_matches) == 1:
        return {"type": "NVIDIA proprietary", "location": nvidia_proprietary_matches[0]}
    elif len(nvidia_proprietary_matches) > 1:
        pass

    if len(mit_license_matches) == 1:
        return {"type": "MIT", "location": mit_license_matches[0]}
    elif len(mit_license_matches) > 1:
        pass

    if len(spdx_matches) == 1:
        return {"type": spdx_matches[0][1], "location": spdx_matches[0][0]}
    elif len(spdx_matches) > 1:
        pass

    # If no special licenses found, return file lists in priority order
    if package_specific_matches:
        return sorted(package_specific_matches)
    if root_license_matches:
        return [next(iter(sorted(root_license_matches)))]
    return sorted(all_matches) if all_matches else None
