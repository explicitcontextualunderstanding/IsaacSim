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
import argparse
import fnmatch
import json
import os
import platform
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from glob import glob

# Add the root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import packmanapi

from tools.check_externals.download_all_kits import get_package_combinations
from tools.check_externals.license_finder import find_license_file


def detect_platform():
    """Detect the current platform and return appropriate platform strings.

    Returns:
        tuple: (platform_abi, platform_target) strings for the current system.

    Raises:
        RuntimeError: If running on an unsupported platform.
    """
    system = platform.system().lower()
    if system == "linux":
        return "manylinux_2_35_x86_64", "linux-x86_64"
    elif system == "windows":
        return "windows-x86_64", "windows-x86_64"
    else:
        raise RuntimeError(f"Unsupported platform: {system}")


def should_filter_package(platform_filter, pkg):
    """Check if a package should be filtered out based on platform filter.

    The package is filtered out only if the platform_filter matches ALL platforms
    listed in the package's platforms attribute.

    Args:
        platform_filter: String to filter out platforms
        pkg: Package element with platforms and version attributes

    Returns:
        bool: True if the package should be filtered out, False otherwise
    """
    if not platform_filter:
        return False

    # Check version string for platform filter
    version = pkg.get("version", "")
    if platform_filter.lower() in version.lower():
        return True

    # Check platforms attribute
    platforms = pkg.get("platforms", "")
    if not platforms:
        return False

    # Split platforms by whitespace and check if platform_filter matches ALL of them
    platform_list = platforms.split()
    if not platform_list:
        return False

    # Only filter out if the platform_filter matches ALL platforms
    return all(platform_filter.lower() in platform.lower() for platform in platform_list)


def gather_package_data(files, config, platform_abi, platform_target, package=None, platform_filter=None):
    """Gather package data from XML files.

    Args:
        files: List of XML files to process
        config: Config to check (i.e. release or debug)
        platform_abi: Platform ABI string
        platform_target: Platform target string
        package: Optional specific package to check
        platform_filter: Optional string to filter out platforms (e.g., 'linux' or 'aarch64')

    Returns:
        tuple: (results_by_file, deps_count_by_file, json_output, full_package_listing)
    """
    results_by_file = defaultdict(lambda: defaultdict(list))
    deps_count_by_file = defaultdict(lambda: defaultdict(int))
    json_output = defaultdict(dict)
    full_package_listing = defaultdict(lambda: defaultdict(list))
    imported_packages = {}  # Store package info from imported files
    filtered_package_names = set()  # Track which package names were filtered

    def substitute_variables(path):
        """Substitute variables in a path string.

        Args:
            path: Path string that may contain variables

        Returns:
            str: Path with variables substituted
        """
        path = path.replace("${config}", config)
        path = path.replace("${platform_target}", platform_target)
        path = path.replace("${platform_target_abi}", platform_abi)
        path = path.replace("${platform}", platform_target)
        return path

    def process_imported_file(import_path, filters):
        """Process an imported file and store its package information.

        Args:
            import_path: Path to the imported file
            filters: List of package names to include from the imported file
        """
        try:
            import_tree = ET.parse(import_path)
            import_root = import_tree.getroot()

            # Store package info from imported file
            for dependency in import_root.findall(".//dependency"):
                name = dependency.get("name")
                if not name:
                    continue

                # Get package name if it exists
                package_elem = dependency.find("package")
                package_name = package_elem.get("name") if package_elem is not None else None

                # Only store packages that match the filters
                if filters:
                    # Substitute variables in name for comparison
                    substituted_name = substitute_variables(name)
                    substituted_package_name = substitute_variables(package_name) if package_name else None

                    if not any(
                        substitute_variables(filter_name) in substituted_name
                        or (substituted_package_name and substitute_variables(filter_name) in substituted_package_name)
                        for filter_name in filters
                    ):
                        continue

                # Check if this package is for our platform
                platforms = dependency.get("platforms", "")
                package_elements = dependency.findall("package")

                # If there's exactly one package and it has no platform info, accept it
                if len(package_elements) == 1 and not platforms:
                    filtered_package_names.add(substituted_name)

                    # Get version from the package element
                    version = package_elements[0].get("version", "")
                    # Replace variables in version string
                    version = substitute_variables(version)

                    imported_packages[substituted_name] = {
                        "name": package_name if package_name else substituted_name,
                        "version": version,
                        "license_file": find_license_file(dependency.get("linkPath", ""), substituted_name),
                    }
                    continue

                # Otherwise, check platform compatibility
                if platforms:
                    platform_list = platforms.split()
                    # Skip if none of the platforms match our target
                    if not any(platform_target.lower() in platform.lower() for platform in platform_list):
                        continue

                filtered_package_names.add(substituted_name)

                # Look for version in nested package element first
                if package_elem is not None:
                    version = package_elem.get("version", "")
                else:
                    version = dependency.get("version", "")

                # Replace variables in version string
                version = substitute_variables(version)

                imported_packages[substituted_name] = {
                    "name": package_name if package_name else substituted_name,
                    "version": version,
                    "license_file": find_license_file(dependency.get("linkPath", ""), substituted_name),
                }
        except Exception as e:
            print(f"Warning: Failed to process imported file {import_path}: {e}")

    for file in files:
        tree = ET.parse(file)
        root = tree.getroot()
        file_dir = os.path.dirname(os.path.abspath(file))

        print(
            f"\nChecking {file} with config: {config}, platform_abi: {platform_abi}, platform_target: {platform_target}"
        )

        # Process imports first
        for import_elem in root.findall(".//import"):
            import_path = import_elem.get("path")
            if not import_path:
                continue

            # Substitute variables in import path
            import_path = substitute_variables(import_path)

            # Resolve import path relative to the file being scanned
            import_path = os.path.normpath(os.path.join(file_dir, import_path))

            # Get filters if any
            filters = []
            for filter_elem in import_elem.findall("filter"):
                include = filter_elem.get("include")
                if include:
                    # Substitute variables in filter include value
                    include = substitute_variables(include)
                    filters.append(include)

            # Process the imported file
            process_imported_file(import_path, filters)

        # Process dependencies
        total_deps = 0
        full_package_listing[file][config] = []

        for dependency in root.findall(".//dependency"):
            name = dependency.get("name")
            if not name:
                continue

            # Substitute variables in name
            name = substitute_variables(name)

            # Skip if this package matches our filter
            if should_filter_package(platform_filter, dependency):
                continue

            total_deps += 1

            # Use imported package info if available, otherwise use local info
            if name in imported_packages:
                package_info = imported_packages[name]
            else:
                # Look for version in nested package element first
                package_elem = dependency.find("package")
                if package_elem is not None:
                    version = package_elem.get("version", "")
                else:
                    version = dependency.get("version", "")

                # Replace variables in version string
                version = substitute_variables(version)

                package_info = {
                    "name": package_elem.get("name") if package_elem is not None else name,
                    "version": version,
                    "license_file": find_license_file(dependency.get("linkPath", ""), name),
                }

            if not package or name == package:
                full_package_listing[file][config].append(package_info)

        deps_count_by_file[file][config] = total_deps

        _, results = packmanapi.verify(
            file,
            exclude_local=True,
            remotes=["cloudfront"],
            tokens={"config": config, "platform_target": platform_target, "platform_target_abi": platform_target},
            platform=platform_target,
            tags={"public": "true"},
        )

        results_by_file[file][config] = results

        # Prepare JSON output for this file and config
        json_output[file][config] = {
            "problem_packages": [
                {
                    "name": result[1].name,
                    "version": result[1].version,
                }
                for result in results
            ],
        }

    return results_by_file, deps_count_by_file, json_output, full_package_listing


def write_output_files(results_by_file, deps_count_by_file, json_output, full_package_listing, configs):
    """Write gathered data to output files and print summary.

    Args:
        results_by_file: Dict of verification results by file and config
        deps_count_by_file: Dict of dependency counts by file and config
        json_output: Dict of problem packages by file and config
        full_package_listing: Dict of all packages by file and config
        configs: List of configs that were checked

    Returns:
        int: Total number of issues found
    """
    # Print summary and count total issues
    total_issues = 0
    total_deps = 0
    print("\nSummary of issues:")
    print("-" * 80)
    for file in sorted(results_by_file.keys()):
        print(f"\n{file}:")
        for config in configs:
            issues = results_by_file[file][config]
            total_issues += len(issues)
            total_deps += deps_count_by_file[file][config]
            print(
                f"  {config}: {len(issues)} issue{'s' if len(issues) != 1 else ''} out of {deps_count_by_file[file][config]} dependencies"
            )

    print(f"\nTotal issues found: {total_issues} across {total_deps} total dependencies")

    # Write JSON outputs
    with open("packman_verification_results.json", "w") as f:
        json.dump(json_output, f, indent=2)

    with open("packman_full_results.json", "w") as f:
        json.dump(full_package_listing, f, indent=2)

    # Write CSV output with alphabetically sorted entries
    with open("packman_full_results.csv", "w") as f:
        f.write("Package Name,Version,Public/Private,License Files,License Type\n")

        # Create a dictionary to track unique package entries
        unique_packages = {}

        for file in full_package_listing:
            for config in configs:
                private_packages = {pkg["name"] for pkg in json_output[file][config]["problem_packages"]}

                for package in full_package_listing[file][config]:
                    name = package["name"]
                    version = package["version"]
                    is_public = "private" if name in private_packages else "public"

                    # Create a key for the package based on its identifying information
                    package_key = (name, version)

                    # Store package info if we haven't seen it before
                    if package_key not in unique_packages:
                        license_info = package.get("license_file")
                        if not license_info:
                            license_files = ""
                            license_type = ""
                        elif isinstance(license_info, list):
                            license_files = ";".join(license_info)
                            license_type = ""
                        elif isinstance(license_info, dict):
                            license_files = license_info["location"]
                            license_type = license_info.get("type", "")
                        else:
                            license_files = str(license_info)
                            license_type = ""

                        unique_packages[package_key] = {
                            "name": name,
                            "version": version,
                            "public_private": is_public,
                            "license_files": license_files,
                            "license_type": license_type,
                        }

        # Sort packages by name first, then by version
        sorted_packages = sorted(unique_packages.values(), key=lambda x: (x["name"].lower(), x["version"].lower()))

        # Write sorted packages to CSV
        for package_info in sorted_packages:
            f.write(
                f"{package_info['name']},{package_info['version']},"
                f"{package_info['public_private']},{package_info['license_files']},"
                f"{package_info['license_type']}\n"
            )

    print("\nDetailed results written to packman_verification_results.json")
    print("Full package listing written to packman_full_results.json")
    print("CSV summary written to packman_full_results.csv")

    return total_issues


def merge_verification_results(*results_list):
    """Merge multiple verification results into one.

    Args:
        *results_list: List of (results_by_file, deps_count_by_file, json_output, full_package_listing) tuples

    Returns:
        tuple: Merged (results_by_file, deps_count_by_file, json_output, full_package_listing)
    """
    merged_results = defaultdict(lambda: defaultdict(list))
    merged_deps = defaultdict(lambda: defaultdict(int))
    merged_json = defaultdict(dict)
    merged_listing = defaultdict(lambda: defaultdict(list))

    for results in results_list:
        results_by_file, deps_count_by_file, json_output, full_package_listing = results

        # Merge results_by_file
        for file, configs in results_by_file.items():
            for config, results in configs.items():
                merged_results[file][config].extend(results)

        # Merge deps_count
        for file, configs in deps_count_by_file.items():
            for config, count in configs.items():
                merged_deps[file][config] += count

        # Merge JSON output
        for file, configs in json_output.items():
            for config, data in configs.items():
                if config not in merged_json[file]:
                    merged_json[file][config] = data
                else:
                    merged_json[file][config]["problem_packages"].extend(data["problem_packages"])

        # Merge package listings
        for file, configs in full_package_listing.items():
            for config, packages in configs.items():
                merged_listing[file][config].extend(packages)

    return merged_results, merged_deps, merged_json, merged_listing


def verify_externals(xml_files=None, package=None, platform_abi=None, platform_target=None, platform_filter=None):
    """Verify external dependencies and their licenses.

    Args:
        xml_files: List of packman XML files to scan. If None, scans all files in ./deps/*.packman.xml
        package: Specific package name to check licenses for. If None, checks all packages.
        platform_abi: Platform ABI string (e.g., 'manylinux_2_35_x86_64'). If None, detects from system.
        platform_target: Platform target string (e.g., 'linux-x86_64'). If None, detects from system.
        platform_filter: String to filter out platforms (e.g., 'linux' will skip all linux platforms)

    Returns:
        int: Number of issues found (non-zero indicates verification failure)
    """
    if platform_abi is None or platform_target is None:
        platform_abi, platform_target = detect_platform()

    # Get list of files to process
    if xml_files:
        files = [f for f in xml_files if os.path.exists(f)]
        if not files:
            raise FileNotFoundError(f"No specified files found")
    else:
        files = glob("./deps/*.packman.xml")

    # Get combinations from kit-sdk.packman.xml
    _, _, combinations = get_package_combinations("./deps/kit-sdk.packman.xml", platform_filter)

    # Filter combinations based on platform and config if specified
    if platform_filter or platform_target:
        filtered_combinations = []
        for combo in combinations:
            if (
                platform_filter
                and combo.get("platform_target")
                and platform_filter.lower() in combo.get("platform_target").lower()
            ):
                continue
            if platform_target and combo.get("platform_target") != platform_target:
                continue
            if platform_filter or platform_target:
                filtered_combinations.append(combo)
        combinations = filtered_combinations

    if not combinations:
        print("No matching combinations found for the specified platform/config")
        return 1

    # Gather results for each combination
    all_results = []
    for combo in combinations:
        print(f"\nVerifying with combination: {combo}")
        results = gather_package_data(
            files,
            combo["config"],
            combo.get("platform_target_abi", platform_abi),
            combo.get("platform_target", platform_target),
            package,
            platform_filter,
        )
        all_results.append(results)

    # Merge results
    merged_results = merge_verification_results(*all_results)

    # Write output and get total issues
    configs = list(set(combo["config"] for combo in combinations))
    return write_output_files(*merged_results, configs)


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="Check packman external dependencies and licenses.")
    parser.add_argument(
        "--file",
        "-f",
        help="Specific packman XML file to scan. If not provided, scans all files in ./deps/*.packman.xml",
    )
    parser.add_argument(
        "--package", "-p", help="Specific package name to check licenses for. If not provided, checks all packages."
    )
    parser.add_argument(
        "--skip-platform", "-s", help="Skip platforms containing this string (e.g., 'linux' or 'aarch64')"
    )
    parser.add_argument(
        "--platform", "-t", help="Explicitly set platform target (e.g., 'linux-x86_64' or 'windows-x86_64')"
    )
    parser.add_argument("--config", "-c", help="Explicitly set config (e.g., 'release' or 'debug')")
    args = parser.parse_args()

    # Get platform info
    platform_abi = None
    platform_target = args.platform
    if platform_target is None:
        platform_abi, platform_target = detect_platform()

    # Get list of files to process
    if args.file:
        files = [args.file]
        if not os.path.exists(args.file):
            raise FileNotFoundError(f"Specified file not found: {args.file}")
    else:
        files = glob("./deps/*.packman.xml")
        if not files:
            raise FileNotFoundError("No packman XML files found in ./deps/")

    # Get combinations from kit-sdk.packman.xml
    _, _, combinations = get_package_combinations("./deps/kit-sdk.packman.xml", args.skip_platform)

    # Filter combinations based on platform and config if specified
    if args.platform or args.config:
        filtered_combinations = []
        for combo in combinations:
            if args.platform and combo.get("platform_target") != args.platform:
                continue
            if args.config and combo.get("config") != args.config:
                continue
            filtered_combinations.append(combo)
        combinations = filtered_combinations

    if not combinations:
        print("No matching combinations found for the specified platform/config")
        return 1

    # Gather results for each combination
    all_results = []
    for combo in combinations:
        print(f"\nVerifying with combination: {combo}")
        results = gather_package_data(
            files,
            combo["config"],
            combo.get("platform_target_abi", platform_abi),
            combo.get("platform_target", platform_target),
            args.package,
            args.skip_platform,
        )
        all_results.append(results)

    # Merge results
    merged_results = merge_verification_results(*all_results)

    # Write output and get total issues
    configs = list(set(combo["config"] for combo in combinations))
    return write_output_files(*merged_results, configs)


if __name__ == "__main__":
    main()
