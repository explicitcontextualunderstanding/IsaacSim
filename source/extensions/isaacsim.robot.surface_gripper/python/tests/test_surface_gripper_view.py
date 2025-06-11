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

import omni.kit.commands

# NOTE:
#   omni.kit.test - std python's unittest module with additional wrapping to add suport for async/await tests
#   For most things refer to unittest docs: https://docs.python.org/3/library/unittest.html
import omni.kit.test
from isaacsim.core.utils.physics import simulate_async
from isaacsim.robot.surface_gripper import GripperView

# Import extension python module we are testing with absolute import path, as if we are external user (other extension)
# from isaacsim.robot.surface_gripper._surface_gripper import Surface_Gripper, Surface_Gripper_Properties
from pxr import Gf, Sdf, UsdGeom, UsdPhysics
from usd.schema.isaac import robot_schema


# Having a test class dervived from omni.kit.test.AsyncTestCase declared on the root of module will make it auto-discoverable by omni.kit.test
class TestSurfaceGripperView(omni.kit.test.AsyncTestCase):
    async def createRigidCube(self, boxActorPath, mass, scale, position, rotation, color):
        p = Gf.Vec3f(position[0], position[1], position[2])
        orientation = Gf.Quatf(rotation[3], rotation[0], rotation[1], rotation[2])
        color = Gf.Vec3f(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)
        size = 1.0
        scale = Gf.Vec3f(scale[0], scale[1], scale[2])

        cubeGeom = UsdGeom.Cube.Define(self.stage, boxActorPath)
        cubePrim = self.stage.GetPrimAtPath(boxActorPath)
        cubeGeom.CreateSizeAttr(size)
        cubeGeom.AddTranslateOp().Set(p)
        cubeGeom.AddOrientOp().Set(orientation)
        cubeGeom.AddScaleOp().Set(scale)
        cubeGeom.CreateDisplayColorAttr().Set([color])
        await omni.kit.app.get_app().next_update_async()
        UsdPhysics.CollisionAPI.Apply(cubePrim)
        rigid_api = UsdPhysics.RigidBodyAPI.Apply(cubePrim)

        if mass > 0:
            massAPI = UsdPhysics.MassAPI.Apply(cubePrim)
            massAPI.CreateMassAttr(mass)
        else:
            rigid_api.CreateKinematicEnabledAttr(True)

        await omni.kit.app.get_app().next_update_async()
        UsdPhysics.CollisionAPI(cubePrim)

    # Helper for setting up the physics stage
    async def setup_physics(self):
        # Set Up Physics scene
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        self._scene = UsdPhysics.Scene.Define(self.stage, Sdf.Path("/physicsScene"))
        self._scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        self._scene.CreateGravityMagnitudeAttr().Set(9.81)

    # Helper for setting up the surface gripper
    async def setup_gripper(self, count):
        for i in range(count):
            env_path = "/env" + str(i)
            box0_path = env_path + "/box0"
            box1_path = env_path + "/box1"
            box0_props = [box0_path, 0.0, [1, 1, 2.0], [-0.50, 0, 1.00], [0, 0, 0, 1], [80, 80, 255]]
            box1_props = [box1_path, 1.0, [0.1, 0.1, 0.1], [0.06, 0, 2.04], [0, 0, 0, 1], [255, 80, 80]]
            d6FixedJoint_path = Sdf.Path(box0_path + "/d6FixedJoint")

            surface_gripper = omni.kit.commands.execute(
                "CreateSurfaceGripper",
                prim_path=env_path,
            )

            gripper_path = env_path + "/SurfaceGripper"
            gripper_prim = self.stage.GetPrimAtPath(gripper_path)
            attachment_points_rel = gripper_prim.GetRelationship(robot_schema.Relations.ATTACHMENT_POINTS.name)
            attachment_points_rel.SetTargets([d6FixedJoint_path])

            await self.createRigidCube(*box0_props)
            await self.createRigidCube(*box1_props)

            joint_prim = UsdPhysics.Joint.Define(self.stage, d6FixedJoint_path)
            joint_prim.CreateBody0Rel().SetTargets([box0_path])
            joint_prim.CreateBody1Rel().SetTargets([box1_path])

        self.gripper_view = GripperView(
            paths="/env*/SurfaceGripper",
        )

    # Before running each test
    async def setUp(self):
        await omni.usd.get_context().new_stage_async()
        self.stage = omni.usd.get_context().get_stage()
        self._timeline = omni.timeline.get_timeline_interface()

        pass

    # After running each test
    async def tearDown(self):
        await omni.kit.app.get_app().next_update_async()
        pass

    async def test_initialize_surface_gripper(self):

        await self.setup_physics()
        await self.setup_gripper(2)
        self._timeline.play()
        await simulate_async(0.125)
        await omni.kit.app.get_app().next_update_async()
        pass

    async def test_get_surface_gripper_status(self):
        gripper_count = 2
        await self.setup_physics()
        await self.setup_gripper(2)
        self._timeline.play()
        await simulate_async(0.125)
        await omni.kit.app.get_app().next_update_async()

        # set status of the grippers and make sure we can retrieve it after a step
        status_exp = ["Open", "Open"]
        status_values = [-0.5, -0.5]
        self.gripper_view.apply_gripper_action(status_values)

        await simulate_async(0.125)
        await omni.kit.app.get_app().next_update_async()

        status = self.gripper_view.get_surface_gripper_status()
        for i in range(gripper_count):
            self.assertEqual(status[i], status_exp[i])

        # set status of the grippers and make sure we can retrieve them without a step
        status_exp = ["Open", "Closing"]
        status_values = [-0.5, 0.5]
        self.gripper_view.apply_gripper_action(status_values)

        status = self.gripper_view.get_surface_gripper_status()
        for i in range(gripper_count):
            self.assertEqual(status[i], status_exp[i])

        # set status of only some grippers
        status_new_exp = ["Open", "Closing"]
        status_values = [0.5, 0.5]
        changed_gripper_indices = [1]
        self.gripper_view.apply_gripper_action(status_values, changed_gripper_indices)

        status = self.gripper_view.get_surface_gripper_status()
        self.assertEqual(status[0], status_new_exp[0])
        self.assertEqual(status[1], status_new_exp[1])

        pass

    async def test_surface_gripper_properties(self):
        gripper_count = 2
        await self.setup_physics()
        await self.setup_gripper(gripper_count)
        self._timeline.play()
        await simulate_async(0.125)
        await omni.kit.app.get_app().next_update_async()

        # set all properties of the grippers and make sure we can retrieve them after a step
        max_grip_distance_exp = [1.0, 1.0]
        coaxial_force_limit_exp = [2.0, 2.0]
        shear_force_limit_exp = [0.3, 0.5]
        retry_interval_exp = [0.1, 0.2]
        self.gripper_view.set_surface_gripper_properties(
            max_grip_distance_exp, coaxial_force_limit_exp, shear_force_limit_exp, retry_interval_exp
        )

        await simulate_async(0.125)
        await omni.kit.app.get_app().next_update_async()

        max_grip_distance, coaxial_force_limit, shear_force_limit, retry_interval = (
            self.gripper_view.get_surface_gripper_properties()
        )
        for i in range(gripper_count):
            self.assertAlmostEqual(max_grip_distance[i], max_grip_distance_exp[i])
            self.assertAlmostEqual(coaxial_force_limit[i], coaxial_force_limit_exp[i])
            self.assertAlmostEqual(shear_force_limit[i], shear_force_limit_exp[i])
            self.assertAlmostEqual(retry_interval[i], retry_interval_exp[i])

        # set all properties of the grippers and make sure we can retrieve them without a step
        max_grip_distance_exp = [2.0, 2.0]
        coaxial_force_limit_exp = [1.0, 1.0]
        shear_force_limit_exp = [0.03, 0.1]
        retry_interval_exp = [0.7, 0.8]
        self.gripper_view.set_surface_gripper_properties(
            max_grip_distance_exp, coaxial_force_limit_exp, shear_force_limit_exp, retry_interval_exp
        )

        max_grip_distance, coaxial_force_limit, shear_force_limit, retry_interval = (
            self.gripper_view.get_surface_gripper_properties()
        )
        for i in range(gripper_count):
            self.assertAlmostEqual(max_grip_distance[i], max_grip_distance_exp[i])
            self.assertAlmostEqual(coaxial_force_limit[i], coaxial_force_limit_exp[i])
            self.assertAlmostEqual(shear_force_limit[i], shear_force_limit_exp[i])
            self.assertAlmostEqual(retry_interval[i], retry_interval_exp[i])

        # set only some properties of the grippers
        max_grip_distance_exp = [0.0, 0.8]
        shear_force_limit_exp = [0.2, 0.5]
        self.gripper_view.set_surface_gripper_properties(
            max_grip_distance=max_grip_distance_exp, shear_force_limit=shear_force_limit_exp
        )

        max_grip_distance, coaxial_force_limit, shear_force_limit, retry_interval = (
            self.gripper_view.get_surface_gripper_properties()
        )
        for i in range(gripper_count):
            self.assertAlmostEqual(max_grip_distance[i], max_grip_distance_exp[i])
            self.assertAlmostEqual(coaxial_force_limit[i], coaxial_force_limit_exp[i])
            self.assertAlmostEqual(shear_force_limit[i], shear_force_limit_exp[i])
            self.assertAlmostEqual(retry_interval[i], retry_interval_exp[i])

        # set properties of only some grippers
        max_grip_distance_new_exp = [0.0, 0.52]
        coaxial_force_limit_new_exp = [0.0, 0.23]
        shear_force_limit_new_exp = [0.0, 0.07]
        retry_interval_new_exp = [0.0, 0.3]
        changed_gripper_indices = [1]
        self.gripper_view.set_surface_gripper_properties(
            max_grip_distance_new_exp,
            coaxial_force_limit_new_exp,
            shear_force_limit_new_exp,
            retry_interval_new_exp,
            changed_gripper_indices,
        )

        max_grip_distance, coaxial_force_limit, shear_force_limit, retry_interval = (
            self.gripper_view.get_surface_gripper_properties()
        )
        self.assertAlmostEqual(max_grip_distance[0], max_grip_distance_exp[0])
        self.assertAlmostEqual(coaxial_force_limit[0], coaxial_force_limit_exp[0])
        self.assertAlmostEqual(shear_force_limit[0], shear_force_limit_exp[0])
        self.assertAlmostEqual(retry_interval[0], retry_interval_exp[0])
        self.assertAlmostEqual(max_grip_distance[1], max_grip_distance_new_exp[1])
        self.assertAlmostEqual(coaxial_force_limit[1], coaxial_force_limit_new_exp[1])
        self.assertAlmostEqual(shear_force_limit[1], shear_force_limit_new_exp[1])
        self.assertAlmostEqual(retry_interval[1], retry_interval_new_exp[1])

        pass
