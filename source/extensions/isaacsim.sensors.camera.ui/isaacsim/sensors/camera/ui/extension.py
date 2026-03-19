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

"""Extension for the isaacsim.sensors.camera.ui extension that provides UI integration for camera and depth sensor creation."""


import gc
from pathlib import Path

import omni.ext
import omni.kit.actions.core
import omni.kit.commands
from isaacsim.core.utils.prims import create_prim
from isaacsim.core.utils.stage import get_next_free_path
from isaacsim.gui.components.menu import create_submenu
from isaacsim.sensors.camera import SingleViewDepthSensorAsset
from isaacsim.storage.native import get_assets_root_path
from omni.kit.menu.utils import add_menu_items, remove_menu_items


class Extension(omni.ext.IExt):
    """Extension for the isaacsim.sensors.camera.ui extension that provides UI integration for camera and depth sensor creation.

    This extension adds menu items to the Create menu and context menus that allow users to create various camera and depth sensor prims in the USD stage. It supports sensors from multiple vendors including Orbbec, Leopard Imaging, RealSense, Sensing, SICK, and Stereolabs.

    The extension automatically registers actions for each supported sensor type and creates a hierarchical menu structure organized by vendor. For depth sensors, it creates specialized SingleViewDepthSensorAsset instances with proper initialization. For regular camera sensors, it creates standard Xform prims with the appropriate USD reference.

    Supported sensor vendors and models include:
    - Orbbec: Gemini 2, FemtoMega, Gemini 335, Gemini 335L (all depth sensors)
    - Leopard Imaging: Hawk, Owl
    - RealSense: D455, D457, D555 (all depth sensors)
    - Sensing: Multiple SG series models with various configurations
    - SICK: Inspector83x
    - Stereolabs: ZED_X (depth sensor)

    The extension provides both main menu integration under Create > Sensors > Camera and Depth Sensors and context menu integration accessible via right-click in the viewport under Isaac > Sensors.
    """

    # Define sensors data organized by vendor and sensor name
    SENSORS = {
        "Orbbec": {
            "Orbbec Gemini 2": {
                "prim_prefix": "/Gemini2",
                "usd_path": "/Isaac/Sensors/Orbbec/Gemini2/orbbec_gemini2_v1.0.usd",
                "is_depth_sensor": True,
            },
            "Orbbec FemtoMega": {
                "prim_prefix": "/Femto",
                "usd_path": "/Isaac/Sensors/Orbbec/FemtoMega/orbbec_femtomega_v1.0.usd",
                "is_depth_sensor": True,
            },
            "Orbbec Gemini 335": {
                "prim_prefix": "/Gemini335",
                "usd_path": "/Isaac/Sensors/Orbbec/Gemini335/orbbec_gemini_335.usd",
                "is_depth_sensor": True,
            },
            "Orbbec Gemini 335L": {
                "prim_prefix": "/Gemini335L",
                "usd_path": "/Isaac/Sensors/Orbbec/Gemini335L/orbbec_gemini_335L.usd",
                "is_depth_sensor": True,
            },
        },
        "Leopard Imaging": {
            "Hawk": {"prim_prefix": "/Hawk", "usd_path": "/Isaac/Sensors/LeopardImaging/Hawk/hawk_v1.1_nominal.usd"},
            "Owl": {"prim_prefix": "/Owl", "usd_path": "/Isaac/Sensors/LeopardImaging/Owl/owl.usd"},
        },
        "RealSense": {
            "Realsense D455": {
                "prim_prefix": "/RealsenseD455",
                "usd_path": "/Isaac/Sensors/RealSense/D455/rsd455.usd",
                "is_depth_sensor": True,
            },
            "Realsense D457": {
                "prim_prefix": "/RealsenseD457",
                "usd_path": "/Isaac/Sensors/RealSense/D457/rsd457.usd",
                "is_depth_sensor": True,
            },
            "Realsense D555": {
                "prim_prefix": "/RealsenseD555",
                "usd_path": "/Isaac/Sensors/RealSense/D555/rsd555.usd",
                "is_depth_sensor": True,
            },
        },
        "Sensing": {
            "Sensing SG2-AR0233C-5200-G2A-H100F1A": {
                "prim_prefix": "/SG2_AR0233C_5200_G2A_H100F1A",
                "usd_path": "/Isaac/Sensors/Sensing/SG2/H100F1A/SG2-AR0233C-5200-G2A-H100F1A.usd",
            },
            "Sensing SG2-OX03CC-5200-GMSL2-H60YA": {
                "prim_prefix": "/SG2_OX03CC_5200_GMSL2_H60YA",
                "usd_path": "/Isaac/Sensors/Sensing/SG2/H60YA/Camera_SG2_OX03CC_5200_GMSL2_H60YA.usd",
            },
            "Sensing SG3-ISX031C-GMSL2F-H190XA": {
                "prim_prefix": "/SG3_ISX031C_GMSL2F_H190XA",
                "usd_path": "/Isaac/Sensors/Sensing/SG3/H190XA/SG3S-ISX031C-GMSL2F-H190XA.usd",
            },
            "Sensing SG5-IMX490C-5300-GMSL2-H110SA": {
                "prim_prefix": "/SG5_IMX490C_5300_GMSL2_H110SA",
                "usd_path": "/Isaac/Sensors/Sensing/SG5/H100SA/SG5-IMX490C-5300-GMSL2-H110SA.usd",
            },
            "Sensing SG8S-AR0820C-5300-G2A-H120YA": {
                "prim_prefix": "/SG8_AR0820C_5300_G2A_H120YA",
                "usd_path": "/Isaac/Sensors/Sensing/SG8/H120YA/SG8S-AR0820C-5300-G2A-H120YA.usd",
            },
            "Sensing SG8S-AR0820C-5300-G2A-H30YA": {
                "prim_prefix": "/SG8_AR0820C_5300_G2A_H30YA",
                "usd_path": "/Isaac/Sensors/Sensing/SG8/H30YA/SG8S-AR0820C-5300-G2A-H30YA.usd",
            },
            "Sensing SG8S-AR0820C-5300-G2A-H60SA": {
                "prim_prefix": "/SG8_AR0820C_5300_G2A_H60SA",
                "usd_path": "/Isaac/Sensors/Sensing/SG8/H60SA/SG8S-AR0820C-5300-G2A-H60SA.usd",
            },
        },
        "SICK": {
            "Inspector83x": {
                "prim_prefix": "/Inspector83x",
                "usd_path": "/Isaac/Sensors/SICK/Inspector83x/SICK_Inspector83x.usd",
            },
        },
        "Stereolabs": {
            "ZED_X": {
                "prim_prefix": "/ZED_X",
                "usd_path": "/Isaac/Sensors/Stereolabs/ZED_X/ZED_X.usdc",
                "is_depth_sensor": True,
            }
        },
    }
    """Dictionary containing sensor configurations organized by vendor and sensor name.

The structure maps vendor names to their sensor models, where each sensor model contains
configuration data including prim prefix, USD asset path, and optional depth sensor flag.
Used to dynamically generate menu items and actions for creating sensor prims in the scene."""

    def on_startup(self, ext_id: str):
        """Initializes the extension by setting up sensor creation actions and menu items.

        Args:
            ext_id: The extension identifier.
        """
        self._ext_id = ext_id
        self._ext_name = omni.ext.get_extension_name(ext_id)
        self._registered_actions = []

        icon_dir = omni.kit.app.get_app().get_extension_manager().get_extension_path_by_module(__name__)

        action_registry = omni.kit.actions.core.get_action_registry()

        # Build menu structure based on SENSORS dictionary
        vendor_dicts = {}
        for vendor, sensors in self.SENSORS.items():
            sensor_items = []
            for sensor_name, sensor_data in sensors.items():
                prim_prefix = sensor_data["prim_prefix"]
                usd_path = sensor_data["usd_path"]

                # Register an action for this camera sensor.
                action_id = "create_camera_" + sensor_name.lower().replace(" ", "_").replace("-", "_")
                if sensor_data.get("is_depth_sensor", False):
                    action_fn = lambda *_, pp=prim_prefix, up=usd_path: self._create_depth_sensor(pp, up)
                else:
                    action_fn = lambda *_, pp=prim_prefix, up=usd_path: create_prim(
                        prim_path=get_next_free_path(pp, None),
                        prim_type="Xform",
                        usd_path=get_assets_root_path() + up,
                    )
                action_registry.register_action(
                    self._ext_name,
                    action_id,
                    action_fn,
                    description=f"Create {sensor_name} camera sensor",
                )
                self._registered_actions.append(action_id)

                sensor_items.append({"name": sensor_name, "onclick_action": (self._ext_name, action_id)})

            vendor_dicts[vendor] = {"name": {vendor: sensor_items}}

        # Create the menu structure
        camera_and_depth_sensors_dict = {
            "name": {
                "Camera and Depth Sensors": [
                    vendor_dicts.get("Orbbec", {}),
                    vendor_dicts.get("Leopard Imaging", {}),
                    vendor_dicts.get("RealSense", {}),
                    vendor_dicts.get("Sensing", {}),
                    vendor_dicts.get("SICK", {}),
                    vendor_dicts.get("Stereolabs", {}),
                ]
            }
        }

        sensors_menu_dict = {
            "name": {
                "Sensors": [
                    camera_and_depth_sensors_dict,
                ]
            },
            "glyph": str(Path(icon_dir).joinpath("data/sensor.svg")),
        }

        self._menu_items = create_submenu(sensors_menu_dict)
        add_menu_items(self._menu_items, "Create")

        # add sensor to context menu
        context_menu_dict = {
            "name": {
                "Isaac": [
                    sensors_menu_dict,
                ],
            },
            "glyph": str(Path(icon_dir).joinpath("data/robot.svg")),
        }

        self._viewport_create_menu = omni.kit.context_menu.add_menu(context_menu_dict, "CREATE")

    def on_shutdown(self):
        """Cleans up the extension by removing menu items and deregistering actions."""
        remove_menu_items(self._menu_items, "Create")
        self._viewport_create_menu = None

        # Deregister all registered actions.
        action_registry = omni.kit.actions.core.get_action_registry()
        for action_id in self._registered_actions:
            action_registry.deregister_action(self._ext_name, action_id)
        self._registered_actions.clear()

        gc.collect()

    def _get_stage_and_path(self):
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

    def _create_depth_sensor(self, prim_prefix: str, usd_path: str):
        """Creates and initializes a depth sensor asset at the next available path.

        Args:
            prim_prefix: The prefix for the prim path.
            usd_path: The USD asset path for the depth sensor.
        """
        depth_sensor = SingleViewDepthSensorAsset(
            prim_path=get_next_free_path(prim_prefix, None),
            asset_path=get_assets_root_path() + usd_path,
        )
        depth_sensor.initialize()
