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

import carb
import numpy as np
import omni.kit

# NOTE:
#   omni.kit.test - std python's unittest module with additional wrapping to add suport for async/await tests
#   For most things refer to unittest docs: https://docs.python.org/3/library/unittest.html
import omni.kit.test
from isaacsim.core.utils.stage import is_stage_loading, update_stage_async

# Import extension python module we are testing with absolute import path, as if we are external user (other extension)
from isaacsim.examples.interactive.robo_factory import RoboFactory


class TestRoboFactoryExampleExtension(omni.kit.test.AsyncTestCase):

    # Before running each test
    async def setUp(self):
        self._sample = RoboFactory()
        self._sample.set_world_settings(physics_dt=1.0 / 60.0, stage_units_in_meters=1.0)
        await self._sample.load_world_async()
        await update_stage_async()
        while is_stage_loading():
            await update_stage_async()
        return

    # After running each test
    async def tearDown(self):
        # In some cases the test will end before the asset is loaded, in this case wait for assets to load
        while is_stage_loading():
            print("tearDown, assets still loading, waiting to finish...")
            await asyncio.sleep(1.0)
        await self._sample.clear_async()
        await update_stage_async()
        self._sample = None
        pass

    # Run all functions with simulation enabled
    async def test_stacking(self):
        """Test that stacking task successfully stacks cubes at the target position."""
        await self._sample.reset_async()
        await update_stage_async()
        await self._sample._on_start_stacking_event_async()
        await update_stage_async()
        # Run simulation for enough frames to complete stacking
        # Task 0 has offset [0, -3, 0], stack target is [0.5, 0.5, 0.12]
        # Final stack position should be [0.5, -2.5, 0.12] (x, y coordinates: [0.5, -2.5])
        for i in range(1750):
            await update_stage_async()

        # Get cube names and observations from the first stacking task
        stacking_task = self._sample._stackings[0]

        # Check if stacking is complete (optional - for debugging)
        is_done = stacking_task.is_done()
        if not is_done:
            print(f"Warning: Stacking task may not be complete. is_done() = {is_done}")

        cube_names = stacking_task.get_cube_names()
        task_observations = stacking_task.get_observations()

        # Verify all cubes are stacked at the target position (x, y coordinates)
        # Task 0 has offset [0, -3, 0], stack target is [0.5, 0.5, 0.12]
        # Final stack position should be [0.5, -2.5, 0.12] (x, y coordinates: [0.5, -2.5])
        # Note: Using tolerance of 0.02 to account for physics simulation variations
        expected_stack_position_xy = np.array([0.5, -2.5])  # [0.5, 0.5] + offset [0, -3]
        for cube_name in cube_names:
            cube_position = task_observations[cube_name]["position"]
            cube_position_xy = cube_position[0:2] if len(cube_position) >= 2 else cube_position

            # Check if position is close to expected (with tolerance for physics variations)
            is_close = np.isclose(cube_position_xy, expected_stack_position_xy, atol=0.02).all()
            self.assertTrue(
                is_close,
                f"Cube {cube_name} not at expected stack position. Got position {cube_position} (xy: {cube_position_xy}), expected xy: {expected_stack_position_xy}",
            )
        pass

    async def test_reset(self):
        await self._sample.reset_async()
        await update_stage_async()
        await update_stage_async()
        await self._sample.reset_async()
        await update_stage_async()
        await update_stage_async()
        pass
