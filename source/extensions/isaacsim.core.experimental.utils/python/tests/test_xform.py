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

import isaacsim.core.experimental.utils.backend as backend_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import isaacsim.core.experimental.utils.xform as xform_utils
import numpy as np
import omni.kit.test


class TestXform(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        """Method called to prepare the test fixture"""
        super().setUp()
        # create new stage
        await stage_utils.create_new_stage_async()

    async def tearDown(self):
        """Method called immediately after the test method has been called"""
        super().tearDown()

    # --------------------------------------------------------------------

    async def test_world_and_local_pose(self):
        def _check_pose(pose, position, orientation, *, rtol: float = 1e-03, atol: float = 1e-05):
            np.testing.assert_allclose(pose[0].numpy(), position, rtol=rtol, atol=atol)
            np.testing.assert_allclose(pose[1].numpy(), orientation, rtol=rtol, atol=atol)

        path_a = "/World/A"
        path_b = "/World/A/B"
        stage_utils.define_prim(path_a)
        stage_utils.define_prim(path_b)
        # test cases
        with backend_utils.use_backend("usdrt"):
            # set local poses (A: euler-XYZ:90,0,0 | B: euler-XYZ:0,-90,00)
            xform_utils.set_local_pose(path_a, translation=[1.0, 2.0, 3.0], orientation=[0.7071, 0.7071, 0.0, 0.0])
            xform_utils.set_local_pose(path_b, translation=[-6.0, -5.0, -4.0], orientation=[0.7071, 0.0, -0.7071, 0])
            # get local poses
            _check_pose(xform_utils.get_local_pose(path_a), [1.0, 2.0, 3.0], [0.7071, 0.7071, 0.0, 0.0])
            _check_pose(xform_utils.get_local_pose(path_b), [-6.0, -5.0, -4.0], [0.7071, 0.0, -0.7071, 0])
            # get world poses
            _check_pose(xform_utils.get_world_pose(path_a), [1.0, 2.0, 3.0], [0.7071, 0.7071, 0.0, 0.0])
            _check_pose(xform_utils.get_world_pose(path_b), [-5.0, 6.0, -2.0], [0.5, 0.5, -0.5, -0.5])
            # set world poses (B: euler-XYZ:0,0,180)
            xform_utils.set_world_pose(path_b, position=[4.0, 5.0, 6.0], orientation=[0.0, 0.0, 0.0, 1.0])
            # get local poses
            _check_pose(xform_utils.get_local_pose(path_b), [3.0, 3.0, -3.0], [0.0, 0.0, 0.7071, 0.7071])
