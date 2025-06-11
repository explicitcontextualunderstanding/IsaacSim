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
"""
Test file for Camera and Depth sensors added via context menus.
This test file simulates context menu clicks (instead of using main menu clicks)
to create Camera sensors and then verifies that a sensor prim is created on stage.
"""

import asyncio

import carb
import omni.kit.app
import omni.kit.test
import omni.kit.ui_test as ui_test
import omni.timeline
import omni.usd
from isaacsim.core.utils.stage import clear_stage, create_new_stage
from omni.ui.tests.test_base import OmniUiTest


class TestCameraContextMenu(OmniUiTest):
    async def setUp(self):
        # Create a new stage for testing.
        result = create_new_stage()
        self.assertTrue(result)
        await omni.kit.app.get_app().next_update_async()

    async def tearDown(self):
        await omni.kit.app.get_app().next_update_async()
        # Wait until stage loading finishes if necessary.
        while omni.usd.get_context().get_stage_loading_status()[2] > 0:
            await omni.kit.app.get_app().next_update_async()
        await super().tearDown()

    async def test_camera_context_menu_count(self):
        """
        Test that all the Camera and Depth Sensor menu items are added correctly.
        Expected items based on extension definition:
            Intel: 1
            Orbbec: 4
            Leopard Imaging: 2
            Sensing: 7
            Stereolabs: 1
        Total expected = 15.
        """
        # Open the Stage window context menu with retry mechanism
        max_attempts = 5
        retry_delay = 0.5
        viewport_context_menu = None

        for attempt in range(max_attempts):
            try:
                viewport_window = ui_test.find("Viewport")
                await viewport_window.right_click()
                viewport_context_menu = await ui_test.get_context_menu()
                break  # Success, exit the loop
            except Exception as e:
                if attempt < max_attempts - 1:  # Don't sleep on the last attempt
                    carb.log_warn(f"Attempt {attempt+1} failed to get context menu: {str(e)}. Retrying...")
                    await omni.kit.app.get_app().next_update_async()
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 1.5  # Exponential backoff
                else:
                    carb.log_error(f"Failed to get context menu after {max_attempts} attempts: {str(e)}")
                    raise  # Re-raise the last exception if all attempts failed

        self.assertIsNotNone(viewport_context_menu, "Failed to get viewport context menu")
        camera_menu_dict = viewport_context_menu["Create"]["Isaac"]["Sensors"]["Camera and Depth Sensors"]

        def count_menu_items(menu_dict):
            count = 0
            for key, item in menu_dict.items():
                if isinstance(item, dict):
                    count += count_menu_items(item)
                else:
                    count += len(item)
            return count

        n_items = count_menu_items(camera_menu_dict)
        expected_items = 1 + 4 + 2 + 7 + 1  # equal to 15 based on extension definition.
        self.assertEqual(
            n_items,
            expected_items,
            f"The number of items in the Camera and Depth Sensors menu ({n_items}) does not match the expected ({expected_items})",
        )

    async def test_camera_sensors_context_menu_click(self):
        """
        Test that selecting a sensor from the Viewport context menu creates a sensor prim.
        In this test we directly select the 'Intel Realsense D455' sensor.
        """
        # Clear the stage before testing.
        clear_stage()
        await omni.kit.app.get_app().next_update_async()
        await omni.kit.app.get_app().next_update_async()

        # Open the Viewport window and get the context menu with retry mechanism
        max_attempts = 5
        retry_delay = 0.5
        viewport_context_menu = None

        for attempt in range(max_attempts):
            try:
                viewport_window = ui_test.find("Viewport")
                await viewport_window.right_click()
                viewport_context_menu = await ui_test.get_context_menu()
                break  # Success, exit the loop
            except Exception as e:
                if attempt < max_attempts - 1:  # Don't sleep on the last attempt
                    carb.log_warn(f"Attempt {attempt+1} failed to get context menu: {str(e)}. Retrying...")
                    await omni.kit.app.get_app().next_update_async()
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 1.5  # Exponential backoff
                else:
                    carb.log_error(f"Failed to get context menu after {max_attempts} attempts: {str(e)}")
                    raise  # Re-raise the last exception if all attempts failed

        self.assertIsNotNone(viewport_context_menu, "Failed to get viewport context menu")

        # Construct the full menu path for the 'Intel Realsense D455' sensor.
        full_test_path = "Create/Isaac/Sensors/Camera and Depth Sensors/Intel/Intel Realsense D455"

        # Select the sensor from the context menu.
        await ui_test.select_context_menu(full_test_path)
        for _ in range(20):
            await omni.kit.app.get_app().next_update_async()

        # Verify that a prim was created with a path starting with '/Realsense'.
        stage = omni.usd.get_context().get_stage()
        sensor_prims = [prim for prim in stage.TraverseAll() if prim.GetPath().pathString.startswith("/Realsense")]
        self.assertGreater(
            len(sensor_prims),
            1,
            f"realsense sensor not loaded correctly",
        )
