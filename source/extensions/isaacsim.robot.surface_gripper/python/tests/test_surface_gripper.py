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

import os

import isaacsim.robot.surface_gripper._surface_gripper as surface_gripper
import omni.kit.commands

# NOTE:
#   omni.kit.test - std python's unittest module with additional wrapping to add suport for async/await tests
#   For most things refer to unittest docs: https://docs.python.org/3/library/unittest.html
import omni.kit.test
import omni.kit.usd
import omni.physics.tensors
from isaacsim.core.api import World
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.core.utils.physics import simulate_async
from isaacsim.core.utils.stage import (
    add_reference_to_stage,
    create_new_stage_async,
    get_current_stage,
    update_stage_async,
)
from pxr import Gf, PhysxSchema
from usd.schema.isaac import robot_schema


# Having a test class dervived from omni.kit.test.AsyncTestCase declared on the root of module will make it auto-discoverable by omni.kit.test
class TestSurfaceGripper(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        World.clear_instance()
        await create_new_stage_async()
        self._world = World(stage_units_in_meters=1.0)
        await self._world.initialize_simulation_context_async()
        self._stage = get_current_stage()
        self._gripper_interface = surface_gripper.acquire_surface_gripper_interface()
        self._timeline = omni.timeline.get_timeline_interface()
        await update_stage_async()
        await self.load_gantry_scene()
        await update_stage_async()

    async def tearDown(self):
        await update_stage_async()
        self._timeline.stop()
        while omni.usd.get_context().get_stage_loading_status()[2] > 0:
            await simulate_async(1.0)
        await update_stage_async()

    async def test_create_surface_gripper(self):
        gripper_prim_path = "/World/SurfaceGripper"
        robot_schema.CreateSurfaceGripper(self._stage, gripper_prim_path)
        await update_stage_async()
        gripper_prim = self._stage.GetPrimAtPath(gripper_prim_path)
        self.assertTrue(gripper_prim.IsValid())

    async def test_configure_surface_gripper(self):
        gripper_prim_path = "/World/SurfaceGripper"
        gripper_joints_prim_path = "/World/Surface_Gripper_Joints"
        robot_schema.CreateSurfaceGripper(self._stage, gripper_prim_path)
        await self.configure_surface_gripper(gripper_prim_path, gripper_joints_prim_path)
        gripper_prim = self._stage.GetPrimAtPath(gripper_prim_path)
        self.assertTrue(gripper_prim.GetAttribute(robot_schema.Attributes.MAX_GRIP_DISTANCE.name).IsValid())
        self.assertTrue(gripper_prim.GetAttribute(robot_schema.Attributes.COAXIAL_FORCE_LIMIT.name).IsValid())
        self.assertTrue(gripper_prim.GetAttribute(robot_schema.Attributes.SHEAR_FORCE_LIMIT.name).IsValid())
        self.assertTrue(gripper_prim.GetAttribute(robot_schema.Attributes.RETRY_INTERVAL.name).IsValid())

    async def test_close_open_close_surface_gripper(self):
        gripper_prim_path = "/World/SurfaceGripper"
        gripper_joints_prim_path = "/World/Surface_Gripper_Joints"
        robot_schema.CreateSurfaceGripper(self._stage, gripper_prim_path)
        await self.configure_surface_gripper(gripper_prim_path, gripper_joints_prim_path)
        gripper_prim = self._stage.GetPrimAtPath(gripper_prim_path)
        self._timeline.play()

        await self.update_joint_target_positions(0.0, 0.0, 0.145)
        await simulate_async(1.0)

        self._gripper_interface.close_gripper(gripper_prim_path)
        await simulate_async(1.0)
        self.assertEqual(self._gripper_interface.get_gripper_status(gripper_prim_path), "Closed")
        gripped_object_list = gripper_prim.GetRelationship(robot_schema.Relations.GRIPPED_OBJECTS.name).GetTargets()
        self.assertEqual(len(gripped_object_list), 1)
        self.assertEqual(gripped_object_list[0], "/World/Boxes/Cube_28")

        await update_stage_async()
        await update_stage_async()
        self._gripper_interface.open_gripper(gripper_prim_path)
        await update_stage_async()
        self.assertEqual(self._gripper_interface.get_gripper_status(gripper_prim_path), "Open")
        gripped_object_list = gripper_prim.GetRelationship(robot_schema.Relations.GRIPPED_OBJECTS.name).GetTargets()
        self.assertEqual(len(gripped_object_list), 0)

        await update_stage_async()
        await update_stage_async()
        self._gripper_interface.close_gripper(gripper_prim_path)
        await simulate_async(1.0)
        self.assertEqual(self._gripper_interface.get_gripper_status(gripper_prim_path), "Closed")
        gripped_object_list = gripper_prim.GetRelationship(robot_schema.Relations.GRIPPED_OBJECTS.name).GetTargets()
        self.assertEqual(len(gripped_object_list), 1)
        self.assertEqual(gripped_object_list[0], "/World/Boxes/Cube_28")

    async def test_multi_object_close(self):
        gripper_prim_path = "/World/SurfaceGripper"
        gripper_joints_prim_path = "/World/Surface_Gripper_Joints"
        robot_schema.CreateSurfaceGripper(self._stage, gripper_prim_path)
        await self.configure_surface_gripper(gripper_prim_path, gripper_joints_prim_path)
        gripper_prim = self._stage.GetPrimAtPath(gripper_prim_path)
        self._timeline.play()

        await self.update_joint_target_positions(0.175, 0.0, 0.145)
        await simulate_async(1.0)

        expected_gripped_object_list = ["/World/Boxes/Cube_20", "/World/Boxes/Cube_24"]
        self._gripper_interface.close_gripper(gripper_prim_path)
        await simulate_async(1.0)
        self.assertEqual(self._gripper_interface.get_gripper_status(gripper_prim_path), "Closed")
        await update_stage_async()
        gripped_object_list = gripper_prim.GetRelationship(robot_schema.Relations.GRIPPED_OBJECTS.name).GetTargets()
        self.assertEqual(len(gripped_object_list), 2)
        self.assertTrue(sorted(set(gripped_object_list)) == sorted(set(expected_gripped_object_list)))

        await update_stage_async()
        await update_stage_async()
        self._gripper_interface.open_gripper(gripper_prim_path)
        await update_stage_async()
        self.assertEqual(self._gripper_interface.get_gripper_status(gripper_prim_path), "Open")
        gripped_object_list = gripper_prim.GetRelationship(robot_schema.Relations.GRIPPED_OBJECTS.name).GetTargets()
        self.assertEqual(len(gripped_object_list), 0)

        await update_stage_async()
        await update_stage_async()
        self._gripper_interface.close_gripper(gripper_prim_path)
        await simulate_async(1.0)
        self.assertEqual(self._gripper_interface.get_gripper_status(gripper_prim_path), "Closed")
        gripped_object_list = gripper_prim.GetRelationship(robot_schema.Relations.GRIPPED_OBJECTS.name).GetTargets()
        self.assertEqual(len(gripped_object_list), 2)
        self.assertTrue(sorted(set(gripped_object_list)) == sorted(set(expected_gripped_object_list)))

    async def test_close_threshold(self):
        gripper_prim_path = "/World/SurfaceGripper"
        gripper_joints_prim_path = "/World/Surface_Gripper_Joints"
        robot_schema.CreateSurfaceGripper(self._stage, gripper_prim_path)
        await self.configure_surface_gripper(gripper_prim_path, gripper_joints_prim_path)
        gripper_prim = self._stage.GetPrimAtPath(gripper_prim_path)
        self._timeline.play()

        await self.update_joint_target_positions(0.0, 0.0, 0.125)
        await simulate_async(1.0)

        self._gripper_interface.close_gripper(gripper_prim_path)
        await simulate_async(1.0)
        self.assertEqual(self._gripper_interface.get_gripper_status(gripper_prim_path), "Open")
        gripped_object_list = gripper_prim.GetRelationship(robot_schema.Relations.GRIPPED_OBJECTS.name).GetTargets()
        self.assertEqual(len(gripped_object_list), 0)

        await update_stage_async()
        await update_stage_async()
        self._gripper_interface.open_gripper(gripper_prim_path)
        await update_stage_async()
        self.assertEqual(self._gripper_interface.get_gripper_status(gripper_prim_path), "Open")
        gripped_object_list = gripper_prim.GetRelationship(robot_schema.Relations.GRIPPED_OBJECTS.name).GetTargets()
        self.assertEqual(len(gripped_object_list), 0)

        await update_stage_async()
        await update_stage_async()
        self._gripper_interface.close_gripper(gripper_prim_path)
        await simulate_async(1.0)
        self.assertEqual(self._gripper_interface.get_gripper_status(gripper_prim_path), "Open")
        gripped_object_list = gripper_prim.GetRelationship(robot_schema.Relations.GRIPPED_OBJECTS.name).GetTargets()
        self.assertEqual(len(gripped_object_list), 0)

    async def test_retry_interval(self):
        gripper_prim_path = "/World/SurfaceGripper"
        gripper_joints_prim_path = "/World/Surface_Gripper_Joints"
        robot_schema.CreateSurfaceGripper(self._stage, gripper_prim_path)
        await self.configure_surface_gripper(gripper_prim_path, gripper_joints_prim_path)
        gripper_prim = self._stage.GetPrimAtPath(gripper_prim_path)
        self._timeline.play()

        await self.update_joint_target_positions(0.0, 0.0, 0.125)
        await simulate_async(1.0)

        self._gripper_interface.close_gripper(gripper_prim_path)
        await self.update_joint_target_positions(0.0, 0.0, 0.145)
        self.assertEqual(self._gripper_interface.get_gripper_status(gripper_prim_path), "Closing")
        gripped_object_list = gripper_prim.GetRelationship(robot_schema.Relations.GRIPPED_OBJECTS.name).GetTargets()
        self.assertEqual(len(gripped_object_list), 0)

        await simulate_async(1.0)

        self.assertEqual(self._gripper_interface.get_gripper_status(gripper_prim_path), "Closed")
        gripped_object_list = gripper_prim.GetRelationship(robot_schema.Relations.GRIPPED_OBJECTS.name).GetTargets()
        self.assertEqual(len(gripped_object_list), 1)
        self.assertEqual(gripped_object_list[0], "/World/Boxes/Cube_28")

    async def test_shear_break_forces(self):
        gripper_prim_path = "/World/SurfaceGripper"
        gripper_joints_prim_path = "/World/Surface_Gripper_Joints"
        robot_schema.CreateSurfaceGripper(self._stage, gripper_prim_path)
        await self.configure_surface_gripper(gripper_prim_path, gripper_joints_prim_path)
        gripper_prim = self._stage.GetPrimAtPath(gripper_prim_path)
        self._timeline.play()

        await self.update_joint_target_positions(0.0, 0.0, 0.145)
        await simulate_async(1.0)

        self._gripper_interface.close_gripper(gripper_prim_path)
        await simulate_async(1.0)

        # Max shear force is 5
        # Box 28 has mass of 0.2
        box_28_prim_path = "/World/Boxes/Cube_28"
        box_28_prim = self._stage.GetPrimAtPath(box_28_prim_path)
        forceApi = PhysxSchema.PhysxForceAPI.Apply(box_28_prim)
        forceApi.GetForceAttr().Set(Gf.Vec3f(0.0, -4, 0.0))
        await update_stage_async()
        self.assertEqual(self._gripper_interface.get_gripper_status(gripper_prim_path), "Closed")
        gripped_object_list = gripper_prim.GetRelationship(robot_schema.Relations.GRIPPED_OBJECTS.name).GetTargets()
        self.assertEqual(len(gripped_object_list), 1)
        self.assertEqual(gripped_object_list[0], box_28_prim_path)

        forceApi.GetForceAttr().Set(Gf.Vec3f(0.0, -500, 0.0))
        await update_stage_async()
        await simulate_async(1.0)
        self.assertEqual(self._gripper_interface.get_gripper_status(gripper_prim_path), "Open")
        gripped_object_list = gripper_prim.GetRelationship(robot_schema.Relations.GRIPPED_OBJECTS.name).GetTargets()
        self.assertEqual(len(gripped_object_list), 0)

    async def test_coaxial_break_force(self):
        gripper_prim_path = "/World/SurfaceGripper"
        gripper_joints_prim_path = "/World/Surface_Gripper_Joints"
        robot_schema.CreateSurfaceGripper(self._stage, gripper_prim_path)
        await self.configure_surface_gripper(gripper_prim_path, gripper_joints_prim_path)
        gripper_prim = self._stage.GetPrimAtPath(gripper_prim_path)
        self._timeline.play()

        await self.update_joint_target_positions(0.0, 0.0, 0.145)
        await simulate_async(1.0)

        self._gripper_interface.close_gripper(gripper_prim_path)
        await simulate_async(1.0)

        # Max coaxial force is 0.005
        # Box 28 has mass of 0.2
        box_28_prim_path = "/World/Boxes/Cube_28"
        box_28_prim = self._stage.GetPrimAtPath(box_28_prim_path)
        forceApi = PhysxSchema.PhysxForceAPI.Apply(box_28_prim)
        forceApi.GetForceAttr().Set(Gf.Vec3f(0.0, 0.0, -0.003))
        await update_stage_async()
        self.assertEqual(self._gripper_interface.get_gripper_status(gripper_prim_path), "Closed")
        gripped_object_list = gripper_prim.GetRelationship(robot_schema.Relations.GRIPPED_OBJECTS.name).GetTargets()
        self.assertEqual(len(gripped_object_list), 1)
        self.assertEqual(gripped_object_list[0], box_28_prim_path)

        await self.update_joint_target_positions(0.0, 0.0, 0.0)
        await simulate_async(1.0)

        forceApi.GetForceAttr().Set(Gf.Vec3f(0.0, 0.0, -1000))
        await update_stage_async()
        await simulate_async(2.0)
        self.assertEqual(self._gripper_interface.get_gripper_status(gripper_prim_path), "Open")
        gripped_object_list = gripper_prim.GetRelationship(robot_schema.Relations.GRIPPED_OBJECTS.name).GetTargets()
        self.assertEqual(len(gripped_object_list), 0)

    async def test_multi_gripper_scene(self):
        # First gripper
        gripper_prim_path = "/World/SurfaceGripper"
        gripper_joints_prim_path = "/World/Surface_Gripper_Joints"
        robot_schema.CreateSurfaceGripper(self._stage, gripper_prim_path)
        await self.configure_surface_gripper(gripper_prim_path, gripper_joints_prim_path)
        gripper_prim = self._stage.GetPrimAtPath(gripper_prim_path)

        # Second gripper
        gripper_prim_path_01 = "/World/SurfaceGripper_01"
        gripper_joints_prim_path_01 = "/World/Surface_Gripper_Joints_01"
        robot_schema.CreateSurfaceGripper(self._stage, gripper_prim_path_01)
        await self.configure_surface_gripper(gripper_prim_path_01, gripper_joints_prim_path_01)
        gripper_prim_01 = self._stage.GetPrimAtPath(gripper_prim_path_01)

        self._timeline.play()

        # Move grippers down to touch the boxes
        # First gripper should be "Closing" (Some gripper joints are off the box)
        # Second gripper should be "Closed"
        await self.update_joint_target_positions(0.0, 0.05, 0.15)
        await simulate_async(1.0)

        # First gripper
        self._gripper_interface.close_gripper(gripper_prim_path)
        await simulate_async(1.0)
        self.assertEqual(self._gripper_interface.get_gripper_status(gripper_prim_path), "Closing")
        gripped_object_list = gripper_prim.GetRelationship(robot_schema.Relations.GRIPPED_OBJECTS.name).GetTargets()
        self.assertEqual(len(gripped_object_list), 1)
        self.assertEqual(gripped_object_list[0], "/World/Boxes/Cube_28")

        await update_stage_async()
        await update_stage_async()
        self._gripper_interface.open_gripper(gripper_prim_path)
        await update_stage_async()
        self.assertEqual(self._gripper_interface.get_gripper_status(gripper_prim_path), "Open")
        gripped_object_list = gripper_prim.GetRelationship(robot_schema.Relations.GRIPPED_OBJECTS.name).GetTargets()
        self.assertEqual(len(gripped_object_list), 0)

        await update_stage_async()
        await update_stage_async()
        self._gripper_interface.close_gripper(gripper_prim_path)
        await simulate_async(1.0)
        self.assertEqual(self._gripper_interface.get_gripper_status(gripper_prim_path), "Closing")
        gripped_object_list = gripper_prim.GetRelationship(robot_schema.Relations.GRIPPED_OBJECTS.name).GetTargets()
        self.assertEqual(len(gripped_object_list), 1)
        self.assertEqual(gripped_object_list[0], "/World/Boxes/Cube_28")

        # Second gripper
        await update_stage_async()
        await update_stage_async()
        self._gripper_interface.close_gripper(gripper_prim_path_01)
        await simulate_async(1.0)
        self.assertEqual(self._gripper_interface.get_gripper_status(gripper_prim_path_01), "Closed")
        gripped_object_list = gripper_prim_01.GetRelationship(robot_schema.Relations.GRIPPED_OBJECTS.name).GetTargets()
        self.assertEqual(len(gripped_object_list), 1)
        self.assertEqual(gripped_object_list[0], "/World/Boxes/Cube_30")

        await update_stage_async()
        await update_stage_async()
        self._gripper_interface.open_gripper(gripper_prim_path_01)
        await update_stage_async()
        self.assertEqual(self._gripper_interface.get_gripper_status(gripper_prim_path_01), "Open")
        gripped_object_list = gripper_prim_01.GetRelationship(robot_schema.Relations.GRIPPED_OBJECTS.name).GetTargets()
        self.assertEqual(len(gripped_object_list), 0)

        await update_stage_async()
        await update_stage_async()
        self._gripper_interface.close_gripper(gripper_prim_path_01)
        await simulate_async(1.0)
        self.assertEqual(self._gripper_interface.get_gripper_status(gripper_prim_path_01), "Closed")
        gripped_object_list = gripper_prim_01.GetRelationship(robot_schema.Relations.GRIPPED_OBJECTS.name).GetTargets()
        self.assertEqual(len(gripped_object_list), 1)
        self.assertEqual(gripped_object_list[0], "/World/Boxes/Cube_30")

    async def load_gantry_scene(self):
        usd_path = os.path.abspath(
            os.path.join(
                get_extension_path_from_name("isaacsim.robot.surface_gripper"), "data", "SurfaceGripper_gantry.usda"
            )
        )
        add_reference_to_stage(usd_path, "/World")
        self._stage.SetDefaultPrim(self._stage.GetPrimAtPath("/World"))
        await update_stage_async()

    async def configure_surface_gripper(self, gripper_prim_path, gripper_joints_prim_path):
        gripper_prim = self._stage.GetPrimAtPath(gripper_prim_path)
        attachment_points_rel = gripper_prim.GetRelationship(robot_schema.Relations.ATTACHMENT_POINTS.name)
        gripper_joints = [p.GetPath() for p in self._stage.GetPrimAtPath(gripper_joints_prim_path).GetChildren()]
        attachment_points_rel.SetTargets(gripper_joints)
        gripper_prim.GetAttribute(robot_schema.Attributes.MAX_GRIP_DISTANCE.name).Set(0.02)
        gripper_prim.GetAttribute(robot_schema.Attributes.COAXIAL_FORCE_LIMIT.name).Set(0.005)
        gripper_prim.GetAttribute(robot_schema.Attributes.SHEAR_FORCE_LIMIT.name).Set(5)
        gripper_prim.GetAttribute(robot_schema.Attributes.RETRY_INTERVAL.name).Set(1.0)
        await update_stage_async()

    async def update_joint_target_positions(self, joint_x_target, joint_y_target, joint_z_target):
        joint_x = self._stage.GetPrimAtPath("/World/Joints/x_joint")
        joint_x.GetAttribute("drive:linear:physics:targetPosition").Set(joint_x_target)
        await update_stage_async()
        joint_y = self._stage.GetPrimAtPath("/World/Joints/y_joint")
        joint_y.GetAttribute("drive:linear:physics:targetPosition").Set(joint_y_target)
        await update_stage_async()
        joint_z = self._stage.GetPrimAtPath("/World/Joints/z_joint")
        joint_z.GetAttribute("drive:linear:physics:targetPosition").Set(joint_z_target)
        await update_stage_async()
