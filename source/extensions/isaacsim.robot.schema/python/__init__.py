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
"""Plugin registration helpers for robot schema modules."""

import os

from pxr import Plug


def _register_plugin_path(path: str) -> None:
    """Register USD plugins located at a given path.

    Args:
        path: Directory path containing a plugInfo.json file.
    """
    # Check if plugins at this path are already registered to avoid duplicate registration
    # which can occur when the module is reloaded
    pluginfo_path = os.path.join(path, "plugInfo.json")
    if not os.path.exists(pluginfo_path):
        return

    try:
        import json

        with open(pluginfo_path) as f:
            # Strip comment lines (lines starting with #) for JSON parsing
            lines = f.readlines()
            json_lines = [line for line in lines if not line.strip().startswith("#")]
            json_content = "".join(json_lines)
            data = json.loads(json_content)

        # Get plugin names that would be registered
        plugin_names_to_register = set()
        if "Plugins" in data:
            for plugin_info in data["Plugins"]:
                if "Name" in plugin_info:
                    plugin_names_to_register.add(plugin_info["Name"])

        if not plugin_names_to_register:
            return

        # Check if these plugins are already registered
        registry = Plug.Registry()
        all_plugins = registry.GetAllPlugins()
        registered_plugin_names = {plugin.name for plugin in all_plugins}

        # If all plugins are already registered, skip to avoid re-registration errors
        if plugin_names_to_register.issubset(registered_plugin_names):
            return

    except Exception:
        # If we can't check registration status, proceed with registration attempt
        pass

    # Attempt registration
    result = Plug.Registry().RegisterPlugins(path)
    if not result:
        import carb

        carb.log_error(f"No plugins found at path {path}")


def _register_plugins(ext_path: str):
    """Register robot schema plugin resources for an extension path.

    Args:
        ext_path: Extension root path to scan for plugin resources.
    """
    _register_plugin_path(os.path.join(ext_path, "usd", "schema", "isaac", "robot_schema"))

    plugin_path = os.path.join(ext_path, "plugins", "plugins")
    _register_plugin_path(os.path.join(plugin_path, "isaacSensorSchema", "resources"))
    _register_plugin_path(os.path.join(plugin_path, "rangeSensorSchema", "resources"))


# carb.tokens.get_tokens_interface().resolve("${isaacsim.robot.schema}") can not be resolved by sphinx
ext_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_register_plugins(ext_path)


from . import robot_schema  # type: ignore[attr-defined]
