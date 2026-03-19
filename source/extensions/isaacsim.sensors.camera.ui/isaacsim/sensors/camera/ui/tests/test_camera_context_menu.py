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

import omni.kit.app
import omni.kit.ui_test as ui_test
import omni.usd
from isaacsim.core.utils.stage import clear_stage
from isaacsim.test.utils import MenuUITestCase, count_menu_items, get_all_menu_paths


class TestCameraContextMenu(MenuUITestCase):
    async def test_camera_context_menu_count(self):
        """Test that all the Camera and Depth Sensor menu items are added correctly.

        Expected items based on extension definition:
            RealSense: 3
            Orbbec: 4
            Leopard Imaging: 2
            Sensing: 7
            SICK: 1
            Stereolabs: 1
        Total expected = 18.
        """
        viewport_context_menu = await self.get_viewport_context_menu()
        self.assertIsNotNone(viewport_context_menu, "Failed to get viewport context menu")

        camera_menu_dict = viewport_context_menu["Create"]["Isaac"]["Sensors"]["Camera and Depth Sensors"]
        n_items = count_menu_items(camera_menu_dict)
        expected_items = 3 + 4 + 2 + 7 + 1 + 1  # equal to 18 based on extension definition.

        self.assertEqual(
            n_items,
            expected_items,
            f"The number of items in the Camera and Depth Sensors menu ({n_items}) does not match the expected ({expected_items})",
        )

    async def test_camera_sensors_context_menu_click(self):
        """Test the Camera and Depth Sensors are added to stage context menus correctly."""
        viewport_context_menu = await self.get_viewport_context_menu()
        self.assertIsNotNone(viewport_context_menu, "Failed to get viewport context menu")

        camera_viewport_menu_dict = viewport_context_menu["Create"]["Isaac"]["Sensors"]["Camera and Depth Sensors"]
        all_menu_paths = get_all_menu_paths(camera_viewport_menu_dict)

        for test_path in all_menu_paths:
            full_test_path = "Create/Isaac/Sensors/Camera and Depth Sensors/" + test_path

            clear_stage()
            await self.wait_n_frames(2)

            await self.get_viewport_context_menu()
            await ui_test.select_context_menu(full_test_path, offset=ui_test.Vec2(10, 10))

            await self.wait_for_stage_loading()
            await self.wait_n_frames(50)

            # Check if there is more than one Camera prim on stage
            stage = omni.usd.get_context().get_stage()
            n_cameras = sum(1 for prim in stage.TraverseAll() if prim.GetTypeName() == "Camera")

            self.assertGreater(
                n_cameras, 1, f"There are {n_cameras} Camera prims on stage for {full_test_path}, expected at least 1."
            )
