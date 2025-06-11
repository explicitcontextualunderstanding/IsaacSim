# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Literal

import isaacsim.core.experimental.utils.stage as stage_utils
import omni.kit.commands
import omni.kit.test
from isaacsim.core.experimental.objects import Capsule, Cone, Cube, Cylinder, Shape, Sphere


async def populate_stage(max_num_prims: int, operation: Literal["wrap", "create"]) -> None:
    # create new stage
    await stage_utils.create_new_stage_async()
    # define prims
    if operation == "wrap":
        for i in range(max_num_prims):
            omni.kit.commands.execute(
                "CreatePrimWithDefaultXform",
                prim_type="SphereLight",
                prim_path=f"/World/A_{i}",
                attributes={"inputs:intensity": 30000},
                select_new_prim=False,
            )


class TestShape(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        """Method called to prepare the test fixture"""
        super().setUp()

    async def tearDown(self):
        """Method called immediately after the test method has been called"""
        super().tearDown()

    # --------------------------------------------------------------------

    async def test_fetch_instances(self):
        await stage_utils.create_new_stage_async()
        # create shapes
        Capsule("/World/shape_01")
        Cone("/World/shape_02")
        Cube("/World/shape_03")
        Cylinder("/World/shape_04")
        Sphere("/World/shape_05")
        # fetch instances
        instances = Shape.fetch_instances(
            [
                "/World",
                "/World/shape_01",
                "/World/shape_02",
                "/World/shape_03",
                "/World/shape_04",
                "/World/shape_05",
            ]
        )
        # check
        self.assertEqual(len(instances), 6)
        self.assertIsNone(instances[0])
        self.assertIsInstance(instances[1], Capsule)
        self.assertIsInstance(instances[2], Cone)
        self.assertIsInstance(instances[3], Cube)
        self.assertIsInstance(instances[4], Cylinder)
        self.assertIsInstance(instances[5], Sphere)
