# SPDX-FileCopyrightText: Copyright (c) 2018-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
import sys
import textwrap

VSCODE_SETTINGS_TEMPLATE = """
{
    "editor.rulers": [120],

    // Enables python language server (seems to work slightly better than jedi)
    "python.languageServer": "Pylance",
    "python.jediEnabled": false,

    // Those paths are automatically filled by isaacsim (see: 'python -m isaacsim --help')
    "python.defaultInterpreterPath": "PYTHON.DEFAULTINTERPRETERPATH",
    "python.analysis.extraPaths": [
        PYTHON.ANALYSIS.EXTRAPATHS
    ],

    // Use "black" as a formatter
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "120"],

    // Use "flake8" for linting
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
}
"""


def generate_vscode_settings():
    def _get_paths(base_path):
        paths = []
        if os.path.isdir(base_path):
            for folder in os.listdir(base_path):
                folder_path = os.path.join(base_path, folder)
                if os.path.isdir(folder_path):
                    paths.append(folder_path)
        return paths

    try:
        import omni.kit_app  # importing 'omni.kit_app' will bootstrap kernel

        kit_path = os.path.dirname(os.path.abspath(os.path.realpath(omni.kit_app.__file__)))
    except ModuleNotFoundError:
        print("Unable to find 'omniverse-kit' package")
        exit()

    # get extensions paths
    extensions_paths = []
    # - omniverse-kit
    folder_path = os.path.join(kit_path, "kernel", "py")
    if os.path.isdir(folder_path):
        extensions_paths.append(folder_path)
    for folder in ["exts", "extscore"]:
        extensions_paths.extend(_get_paths(os.path.join(kit_path, folder)))
    # - isaacsim
    isaacsim_path = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
    for folder in ["exts", "extscache", "extsDeprecated", "extsUser"]:
        extensions_paths.extend(_get_paths(os.path.join(isaacsim_path, folder)))

    cwd = os.getcwd()
    vscode_settings_path = os.path.join(cwd, ".vscode", "settings.json")
    # check if .vscode/settings.json exists
    if os.path.exists(vscode_settings_path):
        overwrite = input("VS Code settings already exists. Overwrite? (y/N): ")
        if overwrite.lower() not in ["y", "yes"]:
            print("Cancelled: VS Code settings not overwritten")
            return

    # update 'python.defaultInterpreterPath'
    template = VSCODE_SETTINGS_TEMPLATE[:]
    template = template.replace("PYTHON.DEFAULTINTERPRETERPATH", sys.executable)

    # update 'python.analysis.extraPaths'
    content = "\n".join([f'"{path}",' for path in extensions_paths])
    content = textwrap.indent(content, prefix=" " * 8)[8:]
    template = template.replace("PYTHON.ANALYSIS.EXTRAPATHS", content)

    # create .vscode/settings.json
    os.makedirs(os.path.join(cwd, ".vscode"), exist_ok=True)
    with open(vscode_settings_path, "w") as f:
        f.write(template)
    print("VS Code settings generated at", vscode_settings_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generate-vscode-settings", default=False, action="store_true", help="Generate VS Code settings."
    )
    args, _ = parser.parse_known_args()

    if args.generate_vscode_settings:
        generate_vscode_settings()
