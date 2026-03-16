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

"""Tests for holonomic (mecanum) controller."""

import numpy as np
import omni.kit.test
from isaacsim.robot.experimental.wheeled_robots.controllers.holonomic_controller import HolonomicController


class TestHolonomicController(omni.kit.test.AsyncTestCase):
    """Tests for HolonomicController kinematics."""

    async def setUp(self):
        """Set up test fixtures."""

    # ----------------------------------------------------------------------
    async def tearDown(self):
        """Tear down test fixtures."""

    # ----------------------------------------------------------------------

    async def test_holonomic_drive(self):
        """Test holonomic forward() output length and finite values."""
        wheel_radius = [0.04, 0.04, 0.04]
        wheel_orientations = [[0, 0, 0, 1], [0.866, 0, 0, -0.5], [0.866, 0, 0, 0.5]]
        wheel_positions = [
            [-0.0980432, 0.000636773, -0.050501],
            [0.0493475, -0.084525, -0.050501],
            [0.0495291, 0.0856937, -0.050501],
        ]
        mecanum_angles = [90, 90, 90]
        velocity_command = [1.0, 1.0, 0.1]

        controller = HolonomicController(
            wheel_radius=wheel_radius,
            wheel_positions=wheel_positions,
            wheel_orientations=wheel_orientations,
            mecanum_angles=mecanum_angles,
        )
        actions = controller.forward(velocity_command)
        self.assertEqual(len(actions), controller.num_wheels, "Output length should match number of wheels")
        self.assertTrue(np.all(np.isfinite(actions)), "All action values should be finite")
