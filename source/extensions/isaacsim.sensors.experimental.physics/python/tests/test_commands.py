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

"""Tests for sensor creation commands.

These tests verify that IsaacSensorExperimentalCreateImuSensor and IsaacSensorExperimentalCreateContactSensor
commands create prims at the correct paths, especially when the stage has a default prim.

The key behavior being tested is that sensor paths should NOT be prepended with the
stage's default prim - they should be created exactly where specified relative to the
parent prim.
"""

import carb
import isaacsim.core.experimental.utils.prim as prim_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import omni.isaac.IsaacSensorSchema as IsaacSensorSchema
import omni.kit.commands
import omni.kit.test
import omni.usd
from isaacsim.core.experimental.objects import Cube
from isaacsim.storage.native import get_assets_root_path_async
from pxr import Gf, PhysxSchema, UsdPhysics

from .common import setup_ant_scene


class TestSensorCommands(omni.kit.test.AsyncTestCase):
    """Test sensor creation commands for correct path handling."""

    async def setUp(self):
        """Set up test environment before each test."""
        self._assets_root_path = await get_assets_root_path_async()
        if self._assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            return

    async def tearDown(self):
        """Clean up after each test."""
        stage_utils.close_stage()
        await omni.kit.app.get_app().next_update_async()

    async def _create_empty_stage_with_world(self):
        """Create an empty stage with /World as root - no default prim."""
        await stage_utils.create_new_stage_async()
        stage = omni.usd.get_context().get_stage()
        # Create /World but do NOT set it as default prim
        stage_utils.define_prim("/World", type_name="Xform")
        await omni.kit.app.get_app().next_update_async()
        return stage

    async def _create_stage_with_default_prim(self):
        """Create a stage with a default prim set (simulating loaded USD asset).

        This simulates loading an asset like ant_colored.usd where /Ant is the default prim.
        """
        await stage_utils.create_new_stage_async()
        stage = omni.usd.get_context().get_stage()
        # Create /DefaultRoot and set it as default prim
        default_prim = stage_utils.define_prim("/DefaultRoot", type_name="Xform")
        stage.SetDefaultPrim(default_prim)
        # Create /World separately (not under default prim)
        stage_utils.define_prim("/World", type_name="Xform")
        await omni.kit.app.get_app().next_update_async()
        return stage

    # ==================== IMU Sensor Command Tests ====================

    async def test_imu_sensor_path_no_default_prim(self):
        """Test IMU sensor is created at correct path when no default prim exists."""
        stage = await self._create_empty_stage_with_world()

        # Create a cube to attach the sensor to
        cube = Cube("/World/Cube", sizes=[1.0])
        UsdPhysics.RigidBodyAPI.Apply(cube.prims[0])
        await omni.kit.app.get_app().next_update_async()

        # Create IMU sensor
        expected_path = "/World/Cube/Imu_Sensor"
        success, prim = omni.kit.commands.execute(
            "IsaacSensorExperimentalCreateImuSensor",
            path="/Imu_Sensor",
            parent="/World/Cube",
        )

        self.assertTrue(success, "Command should succeed")
        self.assertIsNotNone(prim, "Prim should be created")

        # Verify path is exactly as expected
        created_path = prim.GetPath().pathString
        self.assertEqual(created_path, expected_path, f"Sensor should be at {expected_path}, got {created_path}")

        # Verify prim exists at expected path
        self.assertTrue(
            prim_utils.get_prim_at_path(expected_path).IsValid(),
            f"Prim should exist at {expected_path}",
        )

    async def test_imu_sensor_path_with_default_prim(self):
        """Test IMU sensor path is NOT prepended with default prim.

        This is the critical test that would have caught the prepend_default_prim bug.
        When a stage has a default prim (like /DefaultRoot), the sensor should still
        be created at the specified parent path (/World/Cube/...), NOT at
        /DefaultRoot/World/Cube/...
        """
        stage = await self._create_stage_with_default_prim()

        # Verify default prim is set
        self.assertTrue(stage.HasDefaultPrim(), "Stage should have a default prim")
        default_prim_path = stage.GetDefaultPrim().GetPath().pathString
        self.assertEqual(default_prim_path, "/DefaultRoot", "Default prim should be /DefaultRoot")

        # Create a cube at /World/Cube (NOT under default prim)
        cube = Cube("/World/Cube", sizes=[1.0])
        UsdPhysics.RigidBodyAPI.Apply(cube.prims[0])
        await omni.kit.app.get_app().next_update_async()

        # Create IMU sensor
        expected_path = "/World/Cube/Imu_Sensor"
        wrong_path = "/DefaultRoot/World/Cube/Imu_Sensor"  # This is what happens with the bug

        success, prim = omni.kit.commands.execute(
            "IsaacSensorExperimentalCreateImuSensor",
            path="/Imu_Sensor",
            parent="/World/Cube",
        )

        self.assertTrue(success, "Command should succeed")
        self.assertIsNotNone(prim, "Prim should be created")

        created_path = prim.GetPath().pathString

        # Critical assertion: path should NOT be under default prim
        self.assertFalse(
            created_path.startswith("/DefaultRoot/"),
            f"Sensor path should NOT start with default prim. Got: {created_path}",
        )

        # Verify correct path
        self.assertEqual(
            created_path,
            expected_path,
            f"Sensor should be at {expected_path}, got {created_path}. "
            f"Bug: prepend_default_prim may be True instead of False.",
        )

        # Verify prim does NOT exist at wrong path
        self.assertFalse(
            prim_utils.get_prim_at_path(wrong_path).IsValid(),
            f"Prim should NOT exist at {wrong_path}",
        )

    async def test_imu_sensor_with_ant_scene(self):
        """Test IMU sensor path when loading a USD asset with default prim.

        This test uses the actual ant scene which sets /Ant as the default prim,
        simulating the real-world scenario that exposed the bug.
        """
        # Load ant scene - this sets /Ant as default prim
        ant_config = await setup_ant_scene()
        stage = stage_utils.get_current_stage()

        # Verify /Ant is the default prim
        self.assertTrue(stage.HasDefaultPrim(), "Ant stage should have default prim")
        default_prim_path = stage.GetDefaultPrim().GetPath().pathString
        self.assertEqual(default_prim_path, "/Ant", "Default prim should be /Ant")

        # Create a separate structure not under /Ant
        stage_utils.define_prim("/World", type_name="Xform")
        cube = Cube("/World/TestCube", sizes=[1.0])
        UsdPhysics.RigidBodyAPI.Apply(cube.prims[0])
        await omni.kit.app.get_app().next_update_async()

        # Create sensor
        expected_path = "/World/TestCube/Imu_Sensor"
        success, prim = omni.kit.commands.execute(
            "IsaacSensorExperimentalCreateImuSensor",
            path="/Imu_Sensor",
            parent="/World/TestCube",
        )

        self.assertTrue(success)
        created_path = prim.GetPath().pathString

        # Should NOT be under /Ant
        self.assertFalse(
            created_path.startswith("/Ant/"),
            f"Sensor should NOT be under /Ant. Got: {created_path}",
        )
        self.assertEqual(created_path, expected_path)

    async def test_imu_sensor_nested_hierarchy(self):
        """Test IMU sensor with deeply nested parent hierarchy."""
        stage = await self._create_stage_with_default_prim()

        # Create nested hierarchy
        stage_utils.define_prim("/World/Level1", type_name="Xform")
        stage_utils.define_prim("/World/Level1/Level2", type_name="Xform")
        cube = Cube("/World/Level1/Level2/Cube", sizes=[1.0])
        UsdPhysics.RigidBodyAPI.Apply(cube.prims[0])
        await omni.kit.app.get_app().next_update_async()

        expected_path = "/World/Level1/Level2/Cube/DeepSensor"
        success, prim = omni.kit.commands.execute(
            "IsaacSensorExperimentalCreateImuSensor",
            path="/DeepSensor",
            parent="/World/Level1/Level2/Cube",
        )

        self.assertTrue(success)
        self.assertEqual(prim.GetPath().pathString, expected_path)

    async def test_imu_sensor_attributes(self):
        """Test IMU sensor attributes are set correctly."""
        stage = await self._create_empty_stage_with_world()

        cube = Cube("/World/Cube", sizes=[1.0])
        UsdPhysics.RigidBodyAPI.Apply(cube.prims[0])
        await omni.kit.app.get_app().next_update_async()

        # Create sensor with custom attributes
        success, prim = omni.kit.commands.execute(
            "IsaacSensorExperimentalCreateImuSensor",
            path="/CustomImu",
            parent="/World/Cube",
            translation=Gf.Vec3d(1.0, 2.0, 3.0),
            orientation=Gf.Quatd(0.707, 0.707, 0.0, 0.0),
            linear_acceleration_filter_size=5,
            angular_velocity_filter_size=3,
            orientation_filter_size=7,
        )

        self.assertTrue(success)

        # Verify attributes
        self.assertEqual(prim.GetLinearAccelerationFilterWidthAttr().Get(), 5)
        self.assertEqual(prim.GetAngularVelocityFilterWidthAttr().Get(), 3)
        self.assertEqual(prim.GetOrientationFilterWidthAttr().Get(), 7)

        # Verify prim type
        usd_prim = prim.GetPrim()
        self.assertTrue(usd_prim.IsA(IsaacSensorSchema.IsaacImuSensor))

    async def test_imu_sensor_enabled_attribute(self):
        """Test that IMU sensor has enabled attribute set to True."""
        stage = await self._create_empty_stage_with_world()

        cube = Cube("/World/Cube", sizes=[1.0])
        UsdPhysics.RigidBodyAPI.Apply(cube.prims[0])
        await omni.kit.app.get_app().next_update_async()

        success, prim = omni.kit.commands.execute(
            "IsaacSensorExperimentalCreateImuSensor",
            path="/Sensor",
            parent="/World/Cube",
        )

        self.assertTrue(success)

        # Verify enabled attribute via base sensor schema
        base_sensor = IsaacSensorSchema.IsaacBaseSensor(prim)
        self.assertTrue(base_sensor.GetEnabledAttr().Get())

    # ==================== Contact Sensor Command Tests ====================

    async def test_contact_sensor_path_no_default_prim(self):
        """Test contact sensor is created at correct path when no default prim exists."""
        stage = await self._create_empty_stage_with_world()

        cube = Cube("/World/Cube", sizes=[1.0])
        UsdPhysics.RigidBodyAPI.Apply(cube.prims[0])
        UsdPhysics.CollisionAPI.Apply(cube.prims[0])
        await omni.kit.app.get_app().next_update_async()

        expected_path = "/World/Cube/Contact_Sensor"
        success, prim = omni.kit.commands.execute(
            "IsaacSensorExperimentalCreateContactSensor",
            path="/Contact_Sensor",
            parent="/World/Cube",
        )

        self.assertTrue(success)
        self.assertIsNotNone(prim)
        self.assertEqual(prim.GetPath().pathString, expected_path)

    async def test_contact_sensor_path_with_default_prim(self):
        """Test contact sensor path is NOT prepended with default prim."""
        stage = await self._create_stage_with_default_prim()

        cube = Cube("/World/Cube", sizes=[1.0])
        UsdPhysics.RigidBodyAPI.Apply(cube.prims[0])
        UsdPhysics.CollisionAPI.Apply(cube.prims[0])
        await omni.kit.app.get_app().next_update_async()

        expected_path = "/World/Cube/Contact_Sensor"
        success, prim = omni.kit.commands.execute(
            "IsaacSensorExperimentalCreateContactSensor",
            path="/Contact_Sensor",
            parent="/World/Cube",
        )

        self.assertTrue(success)
        created_path = prim.GetPath().pathString

        self.assertFalse(
            created_path.startswith("/DefaultRoot/"),
            f"Contact sensor path should NOT start with default prim. Got: {created_path}",
        )
        self.assertEqual(created_path, expected_path)

    async def test_contact_sensor_attributes(self):
        """Test contact sensor attributes are set correctly."""
        stage = await self._create_empty_stage_with_world()

        cube = Cube("/World/Cube", sizes=[1.0])
        UsdPhysics.RigidBodyAPI.Apply(cube.prims[0])
        UsdPhysics.CollisionAPI.Apply(cube.prims[0])
        await omni.kit.app.get_app().next_update_async()

        success, prim = omni.kit.commands.execute(
            "IsaacSensorExperimentalCreateContactSensor",
            path="/CustomContact",
            parent="/World/Cube",
            min_threshold=10.0,
            max_threshold=5000.0,
            color=Gf.Vec4f(1.0, 0.0, 0.0, 1.0),
            radius=0.5,
        )

        self.assertTrue(success)

        # Verify attributes
        threshold = prim.GetThresholdAttr().Get()
        self.assertAlmostEqual(threshold[0], 10.0, places=5)
        self.assertAlmostEqual(threshold[1], 5000.0, places=5)
        self.assertAlmostEqual(prim.GetRadiusAttr().Get(), 0.5, places=5)

        # Verify parent has contact report API
        parent_prim = stage.GetPrimAtPath("/World/Cube")
        self.assertTrue(parent_prim.HasAPI(PhysxSchema.PhysxContactReportAPI))

    async def test_contact_sensor_requires_parent(self):
        """Test that contact sensor command requires a parent prim."""
        stage = await self._create_empty_stage_with_world()

        # Try to create without parent
        success, prim = omni.kit.commands.execute(
            "IsaacSensorExperimentalCreateContactSensor",
            path="/Contact_Sensor",
            parent=None,
        )

        # Should fail gracefully
        self.assertIsNone(prim)

    # ==================== Path Edge Cases ====================

    async def test_path_with_trailing_slash(self):
        """Test that trailing slashes in parent path are handled correctly."""
        stage = await self._create_empty_stage_with_world()

        cube = Cube("/World/Cube", sizes=[1.0])
        UsdPhysics.RigidBodyAPI.Apply(cube.prims[0])
        await omni.kit.app.get_app().next_update_async()

        # Parent path with trailing slash
        expected_path = "/World/Cube/Sensor"
        success, prim = omni.kit.commands.execute(
            "IsaacSensorExperimentalCreateImuSensor",
            path="/Sensor",
            parent="/World/Cube/",  # Note trailing slash
        )

        self.assertTrue(success)
        # Should handle trailing slash and not create //
        self.assertEqual(prim.GetPath().pathString, expected_path)
        self.assertNotIn("//", prim.GetPath().pathString)

    async def test_path_with_leading_slash(self):
        """Test that leading slashes in path are handled correctly."""
        stage = await self._create_empty_stage_with_world()

        cube = Cube("/World/Cube", sizes=[1.0])
        UsdPhysics.RigidBodyAPI.Apply(cube.prims[0])
        await omni.kit.app.get_app().next_update_async()

        expected_path = "/World/Cube/Sensor"
        success, prim = omni.kit.commands.execute(
            "IsaacSensorExperimentalCreateImuSensor",
            path="Sensor",  # No leading slash
            parent="/World/Cube",
        )

        self.assertTrue(success)
        self.assertEqual(prim.GetPath().pathString, expected_path)

    async def test_unique_path_generation(self):
        """Test that duplicate sensor names get unique paths."""
        stage = await self._create_empty_stage_with_world()

        cube = Cube("/World/Cube", sizes=[1.0])
        UsdPhysics.RigidBodyAPI.Apply(cube.prims[0])
        await omni.kit.app.get_app().next_update_async()

        # Create first sensor
        success1, prim1 = omni.kit.commands.execute(
            "IsaacSensorExperimentalCreateImuSensor",
            path="/Sensor",
            parent="/World/Cube",
        )

        # Create second sensor with same name
        success2, prim2 = omni.kit.commands.execute(
            "IsaacSensorExperimentalCreateImuSensor",
            path="/Sensor",
            parent="/World/Cube",
        )

        self.assertTrue(success1)
        self.assertTrue(success2)

        # Paths should be different
        path1 = prim1.GetPath().pathString
        path2 = prim2.GetPath().pathString
        self.assertNotEqual(path1, path2, "Duplicate sensors should get unique paths")
        self.assertEqual(path1, "/World/Cube/Sensor")
        # Second one should have a suffix like _01
        self.assertTrue(
            path2.startswith("/World/Cube/Sensor"),
            f"Second sensor should be under /World/Cube/Sensor*, got {path2}",
        )

    # ==================== Undo Tests ====================

    async def test_imu_sensor_undo(self):
        """Test that IMU sensor creation can be undone."""
        stage = await self._create_empty_stage_with_world()

        cube = Cube("/World/Cube", sizes=[1.0])
        UsdPhysics.RigidBodyAPI.Apply(cube.prims[0])
        await omni.kit.app.get_app().next_update_async()

        sensor_path = "/World/Cube/Sensor"

        # Create sensor
        success, prim = omni.kit.commands.execute(
            "IsaacSensorExperimentalCreateImuSensor",
            path="/Sensor",
            parent="/World/Cube",
        )

        self.assertTrue(success)
        self.assertTrue(prim_utils.get_prim_at_path(sensor_path).IsValid())

        # Undo
        omni.kit.undo.undo()
        await omni.kit.app.get_app().next_update_async()

        # Sensor should be removed
        self.assertFalse(
            prim_utils.get_prim_at_path(sensor_path).IsValid(),
            "Sensor should be removed after undo",
        )
