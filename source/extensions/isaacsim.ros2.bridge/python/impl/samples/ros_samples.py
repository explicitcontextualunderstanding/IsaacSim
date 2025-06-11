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


import asyncio
import os
import weakref

import carb
import omni.ext
import omni.ui as ui
import omni.usd
from isaacsim.examples.browser import get_instance as get_browser_instance
from isaacsim.gui.components.ui_utils import setup_ui_headers
from isaacsim.storage.native import get_assets_root_path


class Extension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        self._ext_id = ext_id

        nova_carter_name = "Nova Carter"
        get_browser_instance().register_example(
            name=nova_carter_name,
            execute_entrypoint=self.build_window,
            ui_hook=lambda a=weakref.proxy(self): a.build_ui(
                nova_carter_name, "/Isaac/Samples/ROS2/Scenario/carter_warehouse_navigation.usd"
            ),
            category="ROS2/Navigation",
        )

        nova_carter_joint_states_name = "Nova Carter Joint States"
        get_browser_instance().register_example(
            name=nova_carter_joint_states_name,
            execute_entrypoint=self.build_window,
            ui_hook=lambda a=weakref.proxy(self): a.build_ui(
                nova_carter_joint_states_name,
                "/Isaac/Samples/ROS2/Scenario/carter_warehouse_navigation_joint_states.usd",
            ),
            category="ROS2/Navigation",
        )

        iw_hub_name = "iw_hub"
        get_browser_instance().register_example(
            name=iw_hub_name,
            execute_entrypoint=self.build_window,
            ui_hook=lambda a=weakref.proxy(self): a.build_ui(
                iw_hub_name, "/Isaac/Samples/ROS2/Scenario/iw_hub_warehouse_navigation.usd"
            ),
            category="ROS2/Navigation",
        )

        sample_scene_name = "Sample Scene"
        get_browser_instance().register_example(
            name=sample_scene_name,
            execute_entrypoint=self.build_window,
            ui_hook=lambda a=weakref.proxy(self): a.build_ui(
                sample_scene_name, "/Isaac/Samples/ROS2/Scenario/carter_warehouse_apriltags_worker.usd"
            ),
            category="ROS2/Isaac ROS",
        )

        perceptor_scene_name = "Perceptor Scene"
        get_browser_instance().register_example(
            name=perceptor_scene_name,
            execute_entrypoint=self.build_window,
            ui_hook=lambda a=weakref.proxy(self): a.build_ui(
                perceptor_scene_name, "/Isaac/Samples/ROS2/Scenario/perceptor_navigation.usd"
            ),
            category="ROS2/Isaac ROS",
        )

        hospital_scene_name = "Hospital Scene"
        get_browser_instance().register_example(
            name=hospital_scene_name,
            execute_entrypoint=self.build_window,
            ui_hook=lambda a=weakref.proxy(self): a.build_ui(
                hospital_scene_name, "/Isaac/Samples/ROS2/Scenario/multiple_robot_carter_hospital_navigation.usd"
            ),
            category="ROS2/Navigation/Multiple Robots",
        )

        office_scene_name = "Office Scene"
        get_browser_instance().register_example(
            name=office_scene_name,
            execute_entrypoint=self.build_window,
            ui_hook=lambda a=weakref.proxy(self): a.build_ui(
                office_scene_name, "/Isaac/Samples/ROS2/Scenario/multiple_robot_carter_office_navigation.usd"
            ),
            category="ROS2/Navigation/Multiple Robots",
        )

    def build_window(self):
        pass

    def build_ui(self, name, file_path):

        # check if ros2 bridge is enabled before proceeding
        extension_enabled = omni.kit.app.get_app().get_extension_manager().is_extension_enabled("isaacsim.ros2.bridge")
        if not extension_enabled:
            msg = "ROS2 Bridge is not enabled. Please enable the extension to use this feature."
            carb.log_error(msg)
        else:
            overview = "This sample demonstrates how to use ROS2 Navigation packages with Isaac Sim. \n\n The Environment Loaded already contains the OmniGraphs needed to connect with ROS2."
            self._main_stack = ui.VStack(spacing=5, height=0)
            with self._main_stack:
                setup_ui_headers(
                    self._ext_id,
                    file_path=os.path.abspath(__file__),
                    title=name,
                    overview=overview,
                    info_collapsed=False,
                )
                ui.Button(
                    "Load Sample Scene", clicked_fn=lambda a=weakref.proxy(self): a._on_environment_setup(file_path)
                )

    def _on_environment_setup(self, stage_path):
        async def load_stage(path):
            await omni.usd.get_context().open_stage_async(path)

        self._assets_root_path = get_assets_root_path()
        if self._assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            return
        scenario_path = self._assets_root_path + stage_path

        asyncio.ensure_future(load_stage(scenario_path))

    def on_shutdown(self):
        get_browser_instance().deregister_example(name="Carter", category="ROS2")
        get_browser_instance().deregister_example(name="iw_hub", category="ROS2")
        get_browser_instance().deregister_example(name="Sample Scene", category="Isaac ROS")
        get_browser_instance().deregister_example(name="Perceptor Scene", category="Isaac ROS")
        get_browser_instance().deregister_example(name="Hospital Scene", category="ROS2")
        get_browser_instance().deregister_example(name="Office Scene", category="ROS2")
