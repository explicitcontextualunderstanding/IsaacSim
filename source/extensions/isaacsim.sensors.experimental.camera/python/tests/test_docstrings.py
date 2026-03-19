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

import isaacsim.core.experimental.utils.stage as stage_utils
import isaacsim.test.docstring
from isaacsim.sensors.experimental.camera import (
    CameraSensor,
    SingleViewDepthCameraSensor,
    TiledCameraSensor,
    draw_annotator_data_to_image,
)
from pxr import UsdGeom


class TestExtensionDocstrings(isaacsim.test.docstring.AsyncDocTestCase):
    async def setUp(self):
        """Method called to prepare the test fixture"""
        super().setUp()
        # create new stage
        await stage_utils.create_new_stage_async()
        stage_utils.define_prim(f"/World", "Xform")

    async def tearDown(self):
        """Method called immediately after the test method has been called"""
        super().tearDown()

    # --------------------------------------------------------------------

    async def test_camera_sensor_docstrings(self):
        # define prim
        stage_utils.define_prim(f"/World/prim_0", "Camera")
        # test case
        await self.assertDocTests(CameraSensor)

    async def test_single_view_depth_camera_sensor_docstrings(self):
        # define prim
        stage_utils.define_prim(f"/World/prim_0", "Camera")
        # test case
        await self.assertDocTests(SingleViewDepthCameraSensor)

    async def test_tiled_camera_sensor_docstrings(self):
        # define prims
        for i in range(3):
            prim = stage_utils.define_prim(f"/World/prim_{i}", "Camera")
            UsdGeom.Xformable(prim).AddTranslateOp().Set((i * 1.0, 0.0, 0.0))
            UsdGeom.Xformable(prim).AddRotateXYZOp().Set((0.0, 0.0, 0.0))
        # test case
        await self.assertDocTests(TiledCameraSensor)

    async def test_draw_annotator_data_to_image_docstrings(self):
        self.assertDocTest(draw_annotator_data_to_image)
