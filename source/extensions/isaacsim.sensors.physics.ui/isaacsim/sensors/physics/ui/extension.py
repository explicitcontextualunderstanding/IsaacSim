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
"""UI extension that adds physics sensors to menus."""

import gc
from pathlib import Path
from typing import Any

import omni.ext
import omni.kit.actions.core
import omni.kit.commands
from isaacsim.core.experimental.prims import XformPrim
from isaacsim.gui.components.menu import create_submenu
from omni.kit.menu.utils import add_menu_items, remove_menu_items
from pxr import Gf


class Extension(omni.ext.IExt):
    """Extension that adds physics sensors to create menus."""

    def on_startup(self, ext_id: str):
        """Register sensor menu items when the extension starts.

        Args:
            ext_id: Extension identifier provided by the extension manager.
        """
        self._ext_id = ext_id
        self._ext_name = omni.ext.get_extension_name(ext_id)
        self._registered_actions = []

        icon_dir = omni.kit.app.get_app().get_extension_manager().get_extension_path_by_module(__name__)
        sensor_icon_path = str(Path(icon_dir).joinpath("data/sensor.svg"))

        # Register actions for each sensor menu item.
        action_registry = omni.kit.actions.core.get_action_registry()

        action_registry.register_action(
            self._ext_name,
            "create_contact_sensor",
            lambda *_: self._add_contact_sensor(),
            description="Create a contact sensor",
        )
        self._registered_actions.append("create_contact_sensor")

        action_registry.register_action(
            self._ext_name,
            "create_imu_sensor",
            lambda *_: self._add_imu_sensor(),
            description="Create an IMU sensor",
        )
        self._registered_actions.append("create_imu_sensor")

        # Build menu dictionary structure
        sensors_menu_dict = {
            "name": {
                "Sensors": [
                    {
                        "name": "Contact Sensor",
                        "onclick_action": (self._ext_name, "create_contact_sensor"),
                    },
                    {
                        "name": "Imu Sensor",
                        "onclick_action": (self._ext_name, "create_imu_sensor"),
                    },
                ]
            },
            "glyph": sensor_icon_path,
        }

        # Convert the dictionary to a menu and add it
        self._menu_items = create_submenu(sensors_menu_dict)
        add_menu_items(self._menu_items, "Create")

        # Add sensor to context menu
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
        """Remove menu items and clean up on shutdown."""
        remove_menu_items(self._menu_items, "Create")
        self._viewport_create_menu = None

        # Deregister all registered actions.
        action_registry = omni.kit.actions.core.get_action_registry()
        for action_id in self._registered_actions:
            action_registry.deregister_action(self._ext_name, action_id)
        self._registered_actions.clear()

        gc.collect()

    def _get_stage_and_path(self) -> str | None:
        """Get the selected prim path from the stage.

        Returns:
            The selected prim path, or None if nothing is selected.
        """
        selectedPrims = omni.usd.get_context().get_selection().get_selected_prim_paths()

        if len(selectedPrims) > 0:
            curr_prim = selectedPrims[-1]
        else:
            curr_prim = None
        return curr_prim

    def _add_contact_sensor(self, *args: Any, **kargs: Any):
        """Create a contact sensor under the current selection.

        Args:
            *args: Additional positional arguments from the menu callback.
            **kargs: Additional keyword arguments from the menu callback.
        """
        result, prim = omni.kit.commands.execute(
            "IsaacSensorExperimentalCreateContactSensor",
            path="/Contact_Sensor",
            parent=self._get_stage_and_path(),
            min_threshold=0.0,
            max_threshold=100000.0,
            color=Gf.Vec4f(1, 0, 0, 1),
            radius=-1,
            translation=Gf.Vec3d(0, 0, 0),
        )

    def _add_imu_sensor(self, *args: Any, **kargs: Any):
        """Create an IMU sensor under the current selection.

        Args:
            *args: Additional positional arguments from the menu callback.
            **kargs: Additional keyword arguments from the menu callback.
        """
        result, prim = omni.kit.commands.execute(
            "IsaacSensorExperimentalCreateImuSensor",
            path="/Imu_Sensor",
            parent=self._get_stage_and_path(),
            translation=Gf.Vec3d(0, 0, 0),
        )
        if result and prim:
            # Make lidar invisible on stage as camera
            XformPrim(str(prim.GetPath())).set_visibilities([False])
