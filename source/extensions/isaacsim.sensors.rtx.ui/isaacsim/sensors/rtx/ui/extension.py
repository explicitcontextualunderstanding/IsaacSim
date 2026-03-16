# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Extension module for RTX sensor UI integration in Isaac Sim."""


import gc
from pathlib import Path

import omni.ext
import omni.kit.actions.core
import omni.kit.commands
from isaacsim.core.utils.stage import get_next_free_path
from isaacsim.gui.components.menu import create_submenu
from isaacsim.sensors.rtx import SUPPORTED_LIDAR_CONFIGS
from omni.kit.menu.utils import add_menu_items, remove_menu_items
from pxr import Tf


class Extension(omni.ext.IExt):
    """Extension class for the isaacsim.sensors.rtx.ui extension.

    This extension provides UI integration for creating RTX sensors in Isaac Sim. It registers menu items
    and actions that allow users to create RTX Lidar and RTX Radar sensors directly from the Create menu
    and viewport context menus.

    The extension supports multiple RTX Lidar sensor configurations from various vendors, organizing them
    into vendor-specific submenus. It also provides RTX Radar sensor creation capabilities. All sensors
    can be created either through the main Create menu or via right-click context menus in the viewport.

    When sensors are created, they are automatically placed at the next available path and can be parented
    to selected prims in the scene.
    """

    def on_startup(self, ext_id: str):
        """Initializes the extension by registering sensor creation actions and menu items.

        Sets up RTX Lidar and RTX Radar sensor creation actions, builds the sensor menu hierarchy,
        and adds menu items to both the Create menu and Isaac context menu.

        Args:
            ext_id: The extension identifier.
        """
        self._ext_id = ext_id
        self._ext_name = omni.ext.get_extension_name(ext_id)
        self._registered_actions = []

        icon_dir = omni.kit.app.get_app().get_extension_manager().get_extension_path_by_module(__name__)
        sensor_icon_path = str(Path(icon_dir).joinpath("data/sensor.svg"))

        action_registry = omni.kit.actions.core.get_action_registry()

        rtx_lidar_vendor_dict = {}
        for config in SUPPORTED_LIDAR_CONFIGS:
            # Assume paths are of the form "/Isaac/Sensors/<Vendor>/<Sensor>/<Sensor>.usd"
            config_path = Path(config)
            vendor_name = config_path.parts[3]
            sensor_name = config_path.stem.replace("_", " ")
            if sensor_name.startswith(vendor_name):
                # Remove the vendor prefix from the sensor name
                # Note: assumes there is a single character (eg. underscore) between the vendor and sensor name
                sensor_name = sensor_name[len(vendor_name) + 1 :]

            # Register an action for this lidar sensor.
            action_id = f"create_lidar_{config_path.stem}"
            sensor_config = config_path.stem
            action_registry.register_action(
                self._ext_name,
                action_id,
                lambda *_, sn=sensor_name, sc=sensor_config: self._create_lidar(sn, sc),
                description=f"Create {vendor_name} {sensor_name} RTX Lidar sensor",
            )
            self._registered_actions.append(action_id)

            if vendor_name not in rtx_lidar_vendor_dict:
                rtx_lidar_vendor_dict[vendor_name] = []
            rtx_lidar_vendor_dict[vendor_name].append(
                {
                    "name": sensor_name,
                    "onclick_action": (self._ext_name, action_id),
                }
            )

        # Sort the vendors by name and arrange them into a list.
        rtx_lidar_vendor_list = []
        for vendor_name in sorted(rtx_lidar_vendor_dict.keys()):
            rtx_lidar_vendor_list.append({"name": {vendor_name: rtx_lidar_vendor_dict[vendor_name]}})

        # Register an action for the RTX Radar sensor.
        radar_action_id = "create_rtx_radar"
        action_registry.register_action(
            self._ext_name,
            radar_action_id,
            lambda *_: self._create_radar(),
            description="Create RTX Radar sensor",
        )
        self._registered_actions.append(radar_action_id)

        # Compose the RTX Lidar menu dictionary.
        rtx_lidar_menu_dict = {"name": {"RTX Lidar": rtx_lidar_vendor_list}}

        # Wrap into a top-level Sensors dictionary (like in the camera extension).
        sensors_menu_dict = {
            "name": {
                "Sensors": [
                    rtx_lidar_menu_dict,
                    {"name": "RTX Radar", "onclick_action": (self._ext_name, radar_action_id)},
                ]
            },
            "glyph": sensor_icon_path,
        }

        # Convert the dictionary to a menu and add it.
        self._menu_items = create_submenu(sensors_menu_dict)
        add_menu_items(self._menu_items, "Create")

        # Add a menu item to the Isaac Sim in context menus.
        context_menu_dict = {
            "name": {
                "Isaac": [
                    sensors_menu_dict,
                ],
            },
            "glyph": sensor_icon_path,
        }

        self._viewport_create_menu = omni.kit.context_menu.add_menu(context_menu_dict, "CREATE")

    def on_shutdown(self):
        """Cleans up extension resources by removing menu items and deregistering actions.

        Removes all registered menu items from the Create menu, deregisters all sensor creation
        actions from the action registry, and performs garbage collection.
        """
        remove_menu_items(self._menu_items, "Create")
        self._viewport_create_menu = None

        # Deregister all registered actions.
        action_registry = omni.kit.actions.core.get_action_registry()
        for action_id in self._registered_actions:
            action_registry.deregister_action(self._ext_name, action_id)
        self._registered_actions.clear()

        gc.collect()

    def _get_stage_and_path(self) -> str | None:
        """Gets the currently selected prim path from the USD stage.

        Returns:
            The path of the last selected prim, or None if no prims are selected.
        """
        selectedPrims = omni.usd.get_context().get_selection().get_selected_prim_paths()

        if len(selectedPrims) > 0:
            curr_prim = selectedPrims[-1]
        else:
            curr_prim = None
        return curr_prim

    def _create_lidar(self, sensor_name, sensor_config):
        """Creates an RTX Lidar sensor at the selected location.

        Generates a unique path based on the sensor name and executes the IsaacSensorCreateRtxLidar
        command to create the sensor prim.

        Args:
            sensor_name: The display name of the lidar sensor.
            sensor_config: The configuration identifier for the lidar sensor.
        """
        selected_prim = self._get_stage_and_path()
        prim_path = get_next_free_path("/" + Tf.MakeValidIdentifier(sensor_name), None)
        omni.kit.commands.execute(
            "IsaacSensorCreateRtxLidar", path=prim_path, parent=selected_prim, config=sensor_config
        )

    def _create_radar(self):
        """Creates an RTX Radar sensor at the selected location.

        Executes the IsaacSensorCreateRtxRadar command to create the radar sensor prim
        at a fixed path under the selected parent prim.
        """
        selected_prim = self._get_stage_and_path()
        omni.kit.commands.execute("IsaacSensorCreateRtxRadar", path="/RtxRadar", parent=selected_prim)
