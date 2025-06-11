# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np

# Import extension python module we are testing with absolute import path, as if we are external user (other extension)
import omni.kit.test
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.stage import (
    add_reference_to_stage,
    create_new_stage_async,
    get_current_stage,
    update_stage_async,
)
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.robot_setup.grasp_editor import GraspSpec, import_grasps_from_file
from isaacsim.storage.native import get_assets_root_path_async
from pxr import Sdf, UsdLux, UsdPhysics


class TestGraspImporter(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        ext_manager = omni.kit.app.get_app().get_extension_manager()
        ext_id = ext_manager.get_enabled_extension_id("isaacsim.robot_setup.grasp_editor")
        extension_path = ext_manager.get_extension_path(ext_id)
        self._grasp_file = extension_path + "/data/robotiq_rubix_grasp.yaml"

        await create_new_stage_async()
        await update_stage_async()
        await self._create_light()

        set_camera_view(eye=[1.6, 1.3, 1.0], target=[0, -0.3, 0], camera_prim_path="/OmniverseKit_Persp")

        asset_root_path = await get_assets_root_path_async()

        gripper_usd_path = asset_root_path + "/Isaac/Robots/Robotiq/2F-85/Robotiq_2F_85_flattened.usd"
        self._gripper_path = "/gripper"
        add_reference_to_stage(gripper_usd_path, self._gripper_path)
        self._gripper_xform = SingleXFormPrim(self._gripper_path)
        self._gripper_xform.set_world_pose(np.array([0.0, 0.2, 0.0]), np.array([0.8, 0.0, 0.2, 0.0]))

        fixed_joint_path = self._gripper_path + "/FixedJoint"

        stage = get_current_stage()
        fixed_joint = UsdPhysics.FixedJoint.Define(stage, fixed_joint_path)
        fixed_joint.GetBody1Rel().SetTargets([self._gripper_path + "/Robotiq_2F_85/base_link"])

        self._cube_path = "/cube/RubikCube"
        add_reference_to_stage(asset_root_path + "/Isaac/Props/Rubiks_Cube/rubiks_cube.usd", self._cube_path)

        self._cube_xform = SingleXFormPrim(self._cube_path)
        self._cube_xform.set_world_pose(np.array([1.0, 0.0, 0.0]))

        self._grasp_spec = import_grasps_from_file(self._grasp_file)

    def assertAlmostEqual(self, a, b, msg=""):
        # overriding method because it doesn't support iterables
        a = np.array(a)
        b = np.array(b)
        self.assertFalse(np.any(abs((a[a != np.array(None)] - b[b != np.array(None)])) > 1e-3), msg)

    async def _create_light(self):
        sphereLight = UsdLux.SphereLight.Define(get_current_stage(), Sdf.Path("/World/SphereLight"))
        sphereLight.CreateRadiusAttr(2)
        sphereLight.CreateIntensityAttr(100000)
        SingleXFormPrim(str(sphereLight.GetPath().pathString)).set_world_pose([6.5, 0, 12])

    async def tearDown(self):
        while omni.usd.get_context().get_stage_loading_status()[2] > 0:
            print("tearDown, assets still loading, waiting to finish...")
            await asyncio.sleep(1.0)
        await update_stage_async()

    async def test_accessors(self):
        self.assertTrue(self._grasp_spec.get_grasp_names() == ["grasp_0", "grasp_1"])

        d = self._grasp_spec.get_grasp_dicts()
        self.assertTrue(d["grasp_1"]["confidence"] == 1.0)
        self.assertAlmostEqual(
            d["grasp_1"]["position"], [-0.13256364870261714, -0.06671887519456805, -0.14897155140648582]
        )
        self.assertAlmostEqual(d["grasp_1"]["orientation"]["w"], 0.8342422246932983)
        self.assertAlmostEqual(
            d["grasp_1"]["orientation"]["xyz"], [0.1769535725646071, -0.42840731993728665, 0.2986545025937998]
        )
        self.assertAlmostEqual(d["grasp_1"]["cspace_position"]["finger_joint"], 0.1758938431739807)
        self.assertAlmostEqual(d["grasp_1"]["pregrasp_cspace_position"]["finger_joint"], 0.1758938431739807)

        self.assertTrue(d["grasp_0"] == self._grasp_spec.get_grasp_dict_by_name("grasp_0"))
        self.assertTrue(d["grasp_1"] == self._grasp_spec.get_grasp_dict_by_name("grasp_1"))

    # The two tests below do not numerically test the correctness of the `compute()` functions.
    # Refer to `test_grasp_editor_subframes()` for a mroe rigorous test.
    # These tests simply store golden values of positions that looked visually correct when compared
    # to pictures of the grasps saved to the `isaac_grasp` file.
    async def test_compute_gripper_pose(self):
        rb_trans, rb_quat = self._cube_xform.get_world_pose()

        pos, orient = self._grasp_spec.compute_gripper_pose_from_rigid_body_pose("grasp_0", rb_trans, rb_quat)
        self.assertAlmostEqual(pos, [0.87771687, -0.06043073, 0.08953029])
        self.assertAlmostEqual(orient, [0.90047985, -0.09601596, 0.27552128, 0.32249805])

        pos, orient = self._grasp_spec.compute_gripper_pose_from_rigid_body_pose("grasp_1", rb_trans, rb_quat)
        self.assertAlmostEqual(pos, [0.86743635, -0.06671888, -0.14897155])
        self.assertAlmostEqual(orient, [0.83424222, 0.17695357, -0.42840732, 0.2986545])

        # This line was used to visually verify that pos, orient are correct for both grasps
        # self._gripper_xform.set_world_pose(pos, orient)

    async def test_compute_rigid_body_pose(self):
        gripper_trans, gripper_quat = self._gripper_xform.get_world_pose()

        pos, orient = self._grasp_spec.compute_rigid_body_pose_from_gripper_pose("grasp_0", gripper_trans, gripper_quat)
        self.assertAlmostEqual(pos, [0.14131404, 0.16882488, -0.07536021])
        self.assertAlmostEqual(orient, [0.94041753, 0.01493187, -0.04889638, -0.3361563])

        pos, orient = self._grasp_spec.compute_rigid_body_pose_from_gripper_pose("grasp_1", gripper_trans, gripper_quat)
        self._cube_xform.set_world_pose(pos, orient)
        self.assertAlmostEqual(pos, [0.16610557, 0.17034578, -0.12548552])
        self.assertAlmostEqual(orient, [0.70542973, -0.24410456, 0.61794968, -0.24681987])

        # This line was used to visually verify that pos, orient are correct for both grasps
        # self._cube_xform.set_world_pose(pos, orient)
