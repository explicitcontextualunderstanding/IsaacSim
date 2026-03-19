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

# NOTE:
#   omni.kit.test - std python's unittest module with additional wrapping to add suport for async/await tests
#   For most things refer to unittest docs: https://docs.python.org/3/library/unittest.html
from __future__ import annotations

import asyncio
import math

import carb
import carb.tokens
import isaacsim.core.experimental.utils.prim as prim_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import isaacsim.core.experimental.utils.transform as transform_utils
import numpy as np
import omni.kit.commands
import omni.kit.test
import omni.timeline
from isaacsim.core.experimental.objects import Cube
from isaacsim.core.experimental.prims import Articulation, GeomPrim, RigidPrim, XformPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.sensors.experimental.physics import ImuSensorBackend, IMUSensorReading
from isaacsim.storage.native import get_assets_root_path_async
from pxr import Gf, UsdGeom, UsdUtils

from .common import (
    ANGLE_TOLERANCE_DEG,
    ANGULAR_VEL_TOLERANCE,
    CM_GRAVITY,
    EARTH_GRAVITY,
    GRAVITY_TOLERANCE,
    MOON_GRAVITY,
    ORIENTATION_TOLERANCE,
    setup_ant_scene,
    step_simulation,
)


# Having a test class dervived from omni.kit.test.AsyncTestCase declared on the root of module will make it auto-discoverable by omni.kit.test
class TestIMUSensor(omni.kit.test.AsyncTestCase):
    # Before running each test
    async def setUp(self):
        self._sensor_rate = 60
        self._imu_backends: dict[str, ImuSensorBackend] = {}
        self._assets_root_path = await get_assets_root_path_async()
        if self._assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            return
        self._timeline = omni.timeline.get_timeline_interface()
        self._ant_config = None

    def _get_imu_backend(self, prim_path: str) -> ImuSensorBackend:
        """Get or create a cached ImuSensorBackend for the given path."""
        if prim_path not in self._imu_backends:
            self._imu_backends[prim_path] = ImuSensorBackend(prim_path)
        return self._imu_backends[prim_path]

    async def _setup_ant(self, physics_rate=60):
        """Load the ant scene and configure ant-specific test data."""
        self._ant_config = await setup_ant_scene(physics_rate)
        self._stage = stage_utils.get_current_stage()
        await omni.kit.app.get_app().next_update_async()
        self.ant = XformPrim("/Ant", reset_xform_op_properties=True)

    # Convenience properties for ant configuration
    @property
    def leg_paths(self):
        return self._ant_config.leg_paths

    @property
    def sphere_path(self):
        return self._ant_config.sphere_path

    @property
    def sensor_offsets(self):
        return self._ant_config.imu_sensor_offsets

    @property
    def sensor_quatd(self):
        return self._ant_config.sensor_quatd

    async def _setup_simple_articulation(self, physics_rate=60):
        """Load the simple articulation scene for articulation-based tests."""
        self.pivot_path = "/Articulation/CenterPivot"
        self.slider_path = "/Articulation/Slider"
        self.arm_path = "/Articulation/Arm"

        # load nucleus asset
        await stage_utils.open_stage_async(
            self._assets_root_path + "/Isaac/Robots/IsaacSim/SimpleArticulation/simple_articulation.usd"
        )

        await omni.kit.app.get_app().next_update_async()
        await omni.kit.app.get_app().next_update_async()
        self._stage = stage_utils.get_current_stage()
        stage_utils.set_stage_units(meters_per_unit=1.0)
        SimulationManager.setup_simulation(dt=1.0 / physics_rate)

    # After running each test
    async def tearDown(self):
        if self._timeline.is_playing():
            self._timeline.stop()
        SimulationManager.invalidate_physics()
        await omni.kit.app.get_app().next_update_async()
        while omni.usd.get_context().get_stage_loading_status()[2] > 0:
            # print("tearDown, assets still loading, waiting to finish...")
            await asyncio.sleep(1.0)
        await omni.kit.app.get_app().next_update_async()

    async def _add_sensor_prims(self):
        """Helper to add IMU sensors to ant legs and sphere. Requires ant to be loaded."""
        for i in range(4):
            await omni.kit.app.get_app().next_update_async()
            result, sensor = omni.kit.commands.execute(
                "IsaacSensorExperimentalCreateImuSensor",
                path="/sensor",
                parent=self.leg_paths[i],
                translation=self.sensor_offsets[i],
                orientation=self.sensor_quatd[i],
            )
            self.assertTrue(result)
            self.assertIsNotNone(sensor)
            # Add sensor on body sphere
            await omni.kit.app.get_app().next_update_async()
            result, sensor = omni.kit.commands.execute(
                "IsaacSensorExperimentalCreateImuSensor",
                path="/sensor",
                parent=self.sphere_path,
                translation=self.sensor_offsets[4],
                orientation=self.sensor_quatd[4],
            )
            self.assertTrue(result)
            self.assertIsNotNone(sensor)

    async def test_add_sensor_prim(self):
        await self._setup_ant()
        await self._add_sensor_prims()

    async def test_orientation_imu(self):
        await self._setup_simple_articulation()

        result, sensor = omni.kit.commands.execute(
            "IsaacSensorExperimentalCreateImuSensor",
            path="/arm_imu",
            parent=self.arm_path,
            translation=Gf.Vec3d(0, 0, 0),
            orientation=Gf.Quatd(1, 0, 0, 0),
        )
        self.assertTrue(result)
        self.assertIsNotNone(sensor)

        self._timeline.play()

        await omni.kit.app.get_app().next_update_async()

        articulation = Articulation("/Articulation")
        await omni.kit.app.get_app().next_update_async()
        articulation.set_dof_gains(np.ones(articulation.num_dofs) * 1e8, np.ones(articulation.num_dofs) * 1e8)

        angle = 0
        for i in range(70):
            articulation.set_dof_positions(np.array([math.radians(angle), 0.5]))

            await omni.kit.app.get_app().next_update_async()
            await omni.kit.app.get_app().next_update_async()

            r = self._get_imu_backend(self.arm_path + "/arm_imu").get_sensor_reading()
            euler = transform_utils.quaternion_to_euler_angles(
                np.array([r.orientation_x, r.orientation_y, r.orientation_z, r.orientation_w]),
                degrees=True,
            )
            orientation = euler.numpy()[0]

            expected_angle = angle % 360
            if angle >= 180:
                expected_angle = angle - 360

            self.assertAlmostEqual(orientation, expected_angle, delta=ANGLE_TOLERANCE_DEG)
            angle += 5

    async def test_ang_vel_imu(self):
        await self._setup_simple_articulation()

        result, sensor = omni.kit.commands.execute(
            "IsaacSensorExperimentalCreateImuSensor",
            path="/slider_imu",
            parent=self.slider_path,
            translation=Gf.Vec3d(0, 0, 0),
            orientation=Gf.Quatd(1, 0, 0, 0),
        )
        self.assertTrue(result)
        self.assertIsNotNone(sensor)

        self._timeline.play()

        await omni.kit.app.get_app().next_update_async()

        articulation = Articulation("/Articulation")
        await omni.kit.app.get_app().next_update_async()

        angular_velocity_list = [x * 30 for x in range(0, 20)]

        for x in angular_velocity_list:
            articulation.set_dof_velocities(np.array([math.radians(x), 0]))

            await omni.kit.app.get_app().next_update_async()
            angular_velocity_z = (
                self._get_imu_backend(self.slider_path + "/slider_imu").get_sensor_reading().angular_velocity_z
            )
            # with sensor frequency = physics rate, all should be the same
            self.assertAlmostEqual(angular_velocity_z, math.radians(x), delta=ANGULAR_VEL_TOLERANCE)

            articulation.set_dof_positions(np.array([0, 0]))
            articulation.set_dof_velocities(np.array([0, 0]))
            articulation.set_dof_efforts(np.array([0, 0]))

    async def test_lin_acc_imu(self):
        """Ensure linear acceleration magnitudes align with applied efforts."""
        await self._setup_simple_articulation()

        result, sensor = omni.kit.commands.execute(
            "IsaacSensorExperimentalCreateImuSensor",
            path="/slider_imu",
            parent=self.slider_path,
            translation=Gf.Vec3d(0, 0, 0),
            orientation=Gf.Quatd(1, 0, 0, 0),
            linear_acceleration_filter_size=10,
            angular_velocity_filter_size=10,
            orientation_filter_size=10,
        )
        self.assertTrue(result)
        self.assertIsNotNone(sensor)

        # await self.test_add_arm_imu()
        result, sensor = omni.kit.commands.execute(
            "IsaacSensorExperimentalCreateImuSensor",
            path="/arm_imu",
            parent=self.arm_path,
            translation=Gf.Vec3d(0, 0, 0),
            orientation=Gf.Quatd(1, 0, 0, 0),
            linear_acceleration_filter_size=10,
            angular_velocity_filter_size=10,
            orientation_filter_size=10,
        )
        self.assertTrue(result)
        self.assertIsNotNone(sensor)

        self._timeline.play()

        await omni.kit.app.get_app().next_update_async()
        articulation = Articulation("/Articulation")
        await omni.kit.app.get_app().next_update_async()
        articulation.set_dof_gains(np.zeros(articulation.num_dofs), np.ones(articulation.num_dofs))

        x = 0
        for i in range(60):

            articulation.set_dof_efforts(np.array([math.radians(x), 0]))
            await omni.kit.app.get_app().next_update_async()
            slider_reading = self._get_imu_backend(self.slider_path + "/slider_imu").get_sensor_reading()
            slider_magnitude = np.linalg.norm(
                [slider_reading.linear_acceleration_x, slider_reading.linear_acceleration_y]
            )
            arm_reading = self._get_imu_backend(self.arm_path + "/arm_imu").get_sensor_reading()
            arm_magnitude = np.linalg.norm([arm_reading.linear_acceleration_x, arm_reading.linear_acceleration_y])
            self.assertGreaterEqual(slider_magnitude, arm_magnitude)

            x += 1000

    async def test_gravity_m(self):
        await self._setup_ant()
        await self._add_sensor_prims()
        self.ant.set_world_poses(positions=[0, 0, 1])
        UsdGeom.SetStageMetersPerUnit(self._stage, 1.0)

        await omni.kit.app.get_app().next_update_async()

        self._timeline.play()
        for i in range(20):
            await omni.kit.app.get_app().next_update_async()
            backend = self._get_imu_backend(self.sphere_path + "/sensor")
            sensor_reading = backend.get_sensor_reading()
            sensor_reading_no_gravity = backend.get_sensor_reading(read_gravity=False)
        self.assertAlmostEqual(sensor_reading.linear_acceleration_z, 0, delta=GRAVITY_TOLERANCE)
        self.assertAlmostEqual(sensor_reading_no_gravity.linear_acceleration_z, -EARTH_GRAVITY, delta=GRAVITY_TOLERANCE)

        for i in range(100):
            await omni.kit.app.get_app().next_update_async()
            backend = self._get_imu_backend(self.sphere_path + "/sensor")
            sensor_reading = backend.get_sensor_reading()
            sensor_reading_no_gravity = backend.get_sensor_reading(read_gravity=False)
        self.assertAlmostEqual(sensor_reading.linear_acceleration_z, EARTH_GRAVITY, delta=GRAVITY_TOLERANCE)
        self.assertAlmostEqual(sensor_reading_no_gravity.linear_acceleration_z, 0, delta=GRAVITY_TOLERANCE)

    async def test_gravity_moon_m(self):
        await self._setup_ant()
        await self._add_sensor_prims()
        self.ant.set_world_poses(positions=[0, 0, 1])
        SimulationManager.get_physics_scenes()[0].set_gravity(Gf.Vec3f(0.0, 0.0, -MOON_GRAVITY))
        UsdGeom.SetStageMetersPerUnit(self._stage, 1.0)

        await omni.kit.app.get_app().next_update_async()

        self._timeline.play()
        for i in range(20):
            await omni.kit.app.get_app().next_update_async()
            sensor_reading = self._get_imu_backend(self.sphere_path + "/sensor").get_sensor_reading()
        self.assertAlmostEqual(sensor_reading.linear_acceleration_z, 0, delta=GRAVITY_TOLERANCE)
        for i in range(200):
            await omni.kit.app.get_app().next_update_async()
            sensor_reading = self._get_imu_backend(self.sphere_path + "/sensor").get_sensor_reading()
        self.assertAlmostEqual(sensor_reading.linear_acceleration_z, MOON_GRAVITY, delta=GRAVITY_TOLERANCE)

    async def test_gravity_cm(self):
        await self._setup_ant()
        await self._add_sensor_prims()

        UsdGeom.SetStageMetersPerUnit(self._stage, 0.01)
        SimulationManager.get_physics_scenes()[0].set_gravity(Gf.Vec3f(0.0, 0.0, -CM_GRAVITY))

        await omni.kit.app.get_app().next_update_async()

        await omni.kit.app.get_app().next_update_async()
        self._timeline.play()
        for i in range(100):
            await omni.kit.app.get_app().next_update_async()
            sensor_reading = self._get_imu_backend(self.sphere_path + "/sensor").get_sensor_reading()
        self.assertAlmostEqual(sensor_reading.linear_acceleration_z, CM_GRAVITY, delta=GRAVITY_TOLERANCE)

    async def test_stop_start(self):
        await self._setup_ant()
        await self._add_sensor_prims()

        await omni.kit.app.get_app().next_update_async()

        self._timeline.play()
        await step_simulation(0.5)

        init_reading = self._get_imu_backend(self.sphere_path + "/sensor").get_sensor_reading()

        self._timeline.stop()
        await omni.kit.app.get_app().next_update_async()

        self._timeline.play()
        await step_simulation(0.5)
        sensor_reading = self._get_imu_backend(self.sphere_path + "/sensor").get_sensor_reading()

        self.assertAlmostEqual(
            sensor_reading.linear_acceleration_x, init_reading.linear_acceleration_x, delta=ORIENTATION_TOLERANCE
        )
        self.assertAlmostEqual(
            sensor_reading.linear_acceleration_y, init_reading.linear_acceleration_y, delta=ORIENTATION_TOLERANCE
        )
        self.assertAlmostEqual(
            sensor_reading.linear_acceleration_z, init_reading.linear_acceleration_z, delta=ORIENTATION_TOLERANCE
        )

    async def test_no_physics_scene(self):
        await stage_utils.open_stage_async(self._assets_root_path + "/Isaac/Environments/Grid/default_environment.usd")
        await omni.kit.app.get_app().next_update_async()
        self._stage = stage_utils.get_current_stage()
        await omni.kit.app.get_app().next_update_async()
        cube_path = "/new_cube"
        Cube(cube_path, sizes=1.0, positions=[0.0, 0.0, 2.0])
        GeomPrim(cube_path, apply_collision_apis=True)
        RigidPrim(cube_path, masses=[1.0])

        await omni.kit.app.get_app().next_update_async()
        result, sensor = omni.kit.commands.execute(
            "IsaacSensorExperimentalCreateImuSensor",
            path="/sensor",
            parent=cube_path,
        )

        await omni.kit.app.get_app().next_update_async()

        self._timeline.play()
        for i in range(20):
            await omni.kit.app.get_app().next_update_async()
            sensor_reading = self._get_imu_backend(cube_path + "/sensor").get_sensor_reading()
        self.assertAlmostEqual(sensor_reading.linear_acceleration_z, 0, delta=GRAVITY_TOLERANCE)
        for i in range(100):
            await omni.kit.app.get_app().next_update_async()
            sensor_reading = self._get_imu_backend(cube_path + "/sensor").get_sensor_reading()
        self.assertAlmostEqual(sensor_reading.linear_acceleration_z, EARTH_GRAVITY, delta=GRAVITY_TOLERANCE)
        self._timeline.stop()

    async def test_rolling_average_attributes(self):
        """Verify larger filter windows reduce sensor output variance."""
        await self._setup_ant(physics_rate=400)
        await omni.kit.app.get_app().next_update_async()

        result, sensor = omni.kit.commands.execute(
            "IsaacSensorExperimentalCreateImuSensor",
            path="/sphere_imu_1",
            parent=self.sphere_path,
            translation=Gf.Vec3d(0, 0, 0),
            orientation=Gf.Quatd(1, 0, 0, 0),
            linear_acceleration_filter_size=1,
            angular_velocity_filter_size=1,
            orientation_filter_size=1,
        )
        self.assertTrue(result)
        self.assertIsNotNone(sensor)

        low_rolling_avg_size_reading = np.zeros(
            6,
        )
        high_rolling_avg_size_reading = np.zeros(
            6,
        )

        self._timeline.play()
        # wait for the ant to settle down
        for i in range(200):
            await omni.kit.app.get_app().next_update_async()

        # test 1, when the rolling average is 1, should expect larger fluctuations
        for i in range(50):
            await omni.kit.app.get_app().next_update_async()
            sensor_reading = self._get_imu_backend(self.sphere_path + "/sphere_imu_1").get_sensor_reading()

            if not sensor_reading.is_valid:
                continue

            readings = np.array(
                [
                    sensor_reading.linear_acceleration_x,
                    sensor_reading.linear_acceleration_y,
                    sensor_reading.linear_acceleration_z,
                    sensor_reading.angular_velocity_x,
                    sensor_reading.angular_velocity_y,
                    sensor_reading.angular_velocity_z,
                ]
            )

            low_rolling_avg_size_reading = np.vstack((low_rolling_avg_size_reading, readings))

        # Ensure we collected enough valid readings (more than just the initial zeros row)
        self.assertGreater(
            low_rolling_avg_size_reading.shape[0], 1, "No valid sensor readings collected for low filter"
        )

        low_rolling_avg_size_1th_percentile = np.percentile(low_rolling_avg_size_reading, 1, axis=0)
        low_rolling_avg_size_99th_percentile = np.percentile(low_rolling_avg_size_reading, 99, axis=0)

        low_rolling_avg_size_diff = np.subtract(
            low_rolling_avg_size_99th_percentile, low_rolling_avg_size_1th_percentile
        )

        # test 2, when the rolling average is 20, should expect lower fluctuations
        sensor.CreateLinearAccelerationFilterWidthAttr().Set(20)
        sensor.CreateAngularVelocityFilterWidthAttr().Set(20)
        sensor.CreateOrientationFilterWidthAttr().Set(20)

        await omni.kit.app.get_app().next_update_async()

        for i in range(50):
            await omni.kit.app.get_app().next_update_async()
            sensor_reading = self._get_imu_backend(self.sphere_path + "/sphere_imu_1").get_sensor_reading()

            if not sensor_reading.is_valid:
                continue

            readings = np.array(
                [
                    sensor_reading.linear_acceleration_x,
                    sensor_reading.linear_acceleration_y,
                    sensor_reading.linear_acceleration_z,
                    sensor_reading.angular_velocity_x,
                    sensor_reading.angular_velocity_y,
                    sensor_reading.angular_velocity_z,
                ]
            )

            high_rolling_avg_size_reading = np.vstack((high_rolling_avg_size_reading, readings))

        # Ensure we collected enough valid readings (more than just the initial zeros row)
        self.assertGreater(
            high_rolling_avg_size_reading.shape[0], 1, "No valid sensor readings collected for high filter"
        )

        high_rolling_avg_size_1th_percentile = np.percentile(high_rolling_avg_size_reading, 1, axis=0)
        high_rolling_avg_size_99th_percentile = np.percentile(high_rolling_avg_size_reading, 99, axis=0)

        high_rolling_avg_size_diff = np.subtract(
            high_rolling_avg_size_99th_percentile, high_rolling_avg_size_1th_percentile
        )

        # low rolling average size is expected to have larger variation than with high rolling average size
        for i in range(len(high_rolling_avg_size_diff)):
            self.assertGreaterEqual(low_rolling_avg_size_diff[i], high_rolling_avg_size_diff[i])

    async def test_sensor_latest_data(self):
        await self._setup_ant()
        await self._add_sensor_prims()
        result, sensor = omni.kit.commands.execute(
            "IsaacSensorExperimentalCreateImuSensor",
            path="/custom_sensor",
            parent=self.sphere_path,
            translation=self.sensor_offsets[4],
            orientation=self.sensor_quatd[4],
        )
        self.assertTrue(result)
        self.assertIsNotNone(sensor)

        await omni.kit.app.get_app().next_update_async()
        self._timeline.play()
        await omni.kit.app.get_app().next_update_async()

        old_time = -1
        for i in range(10):
            await omni.kit.app.get_app().next_update_async()
            latest_sensor_reading = self._get_imu_backend(self.sphere_path + "/custom_sensor").get_sensor_reading()
            self.assertTrue(latest_sensor_reading.time > old_time)
            old_time = latest_sensor_reading.time

    async def test_wrong_sensor_path(self):
        await self._setup_ant()
        await self._add_sensor_prims()
        await omni.kit.app.get_app().next_update_async()
        self._timeline.play()
        # give it some time to reach the ground first
        await omni.kit.app.get_app().next_update_async()

        for i in range(10):  # Simulate for 10 steps
            await omni.kit.app.get_app().next_update_async()
            latest_sensor_reading = self._get_imu_backend(self.sphere_path + "/wrong_sensor").get_sensor_reading()

            self.assertFalse(latest_sensor_reading.is_valid)
            self.assertEqual(latest_sensor_reading.time, 0)

    async def test_change_buffer_size(self):
        """Ensure changing filter widths still yields valid readings."""
        await self._setup_ant()
        await self._add_sensor_prims()
        result, sensor = omni.kit.commands.execute(
            "IsaacSensorExperimentalCreateImuSensor",
            path="/custom_sensor",
            parent=self.sphere_path,
            translation=self.sensor_offsets[4],
            orientation=self.sensor_quatd[4],
        )
        self.assertTrue(result)
        self.assertIsNotNone(sensor)

        await omni.kit.app.get_app().next_update_async()
        self._timeline.play()
        await omni.kit.app.get_app().next_update_async()

        imu_sensor = prim_utils.get_prim_at_path(self.sphere_path + "/custom_sensor")
        imu_sensor.GetAttribute("linearAccelerationFilterWidth").Set(5)
        imu_sensor.GetAttribute("angularVelocityFilterWidth").Set(5)
        imu_sensor.GetAttribute("orientationFilterWidth").Set(5)
        await omni.kit.app.get_app().next_update_async()

        reading = self._get_imu_backend(self.sphere_path + "/custom_sensor").get_sensor_reading()
        self.assertTrue(reading.is_valid)

    async def test_imu_rigidbody_grandparent(self):
        """Validate IMU readings through nested transform hierarchy changes."""
        await self._setup_ant()
        Cube("/World/Cube", sizes=1.0, positions=[10.0, 0.0, 0.0])
        GeomPrim("/World/Cube", apply_collision_apis=True)
        RigidPrim("/World/Cube", masses=[1.0])

        stage_utils.define_prim("/World/Cube/xform", "Xform")
        XformPrim("/World/Cube/xform", translations=[10.0, 0.0, 0.0], reset_xform_op_properties=True)

        result, sensor = omni.kit.commands.execute(
            "IsaacSensorExperimentalCreateImuSensor",
            path="/custom_sensor",
            parent="/World/Cube/xform",
            translation=self.sensor_offsets[4],
            orientation=self.sensor_quatd[4],
        )

        await omni.kit.app.get_app().next_update_async()
        self._timeline.play()
        await step_simulation(0.5)
        custom_reading = self._get_imu_backend("/World/Cube/xform/custom_sensor").get_sensor_reading()

        self.assertAlmostEqual(custom_reading.linear_acceleration_z, EARTH_GRAVITY, delta=GRAVITY_TOLERANCE)

        self._timeline.stop()
        await omni.kit.app.get_app().next_update_async()

        # Rotate the parent cube about y by -90 degree
        # The x axis points upward
        xform_prim = XformPrim("/World/Cube", reset_xform_op_properties=True)
        xform_prim.set_local_poses(orientations=[0.70711, 0.0, -0.70711, 0.0])
        self._timeline.play()
        await omni.kit.app.get_app().next_update_async()

        await step_simulation(0.5)
        custom_reading = self._get_imu_backend("/World/Cube/xform/custom_sensor").get_sensor_reading()
        self.assertAlmostEqual(custom_reading.linear_acceleration_x, EARTH_GRAVITY, delta=GRAVITY_TOLERANCE)

        # rotated -90 degress abouty, check if this is correct
        # note: (-0.70711, 0 0.70711, 0) and (0.70711, 0, -0.70711, 0) represent the same angle
        self.assertAlmostEqual(abs(custom_reading.orientation_w), 0.70711, delta=ORIENTATION_TOLERANCE)
        self.assertAlmostEqual(custom_reading.orientation_x, 0.0, delta=ORIENTATION_TOLERANCE)
        self.assertAlmostEqual(abs(custom_reading.orientation_y), 0.70711, delta=ORIENTATION_TOLERANCE)
        self.assertAlmostEqual(custom_reading.orientation_z, 0.0, delta=ORIENTATION_TOLERANCE)
        self.assertAlmostEqual(custom_reading.orientation_w, -custom_reading.orientation_y, delta=ORIENTATION_TOLERANCE)

    async def test_invalid_imu(self):
        # goal is to make sure an invalid imu doesn't crash the sim
        result, sensor = omni.kit.commands.execute(
            "IsaacSensorExperimentalCreateImuSensor",
            path="/sensor",
            parent="/World",
            translation=Gf.Vec3d(0, 0, 0),
            orientation=Gf.Quatd(1, 0, 0, 0),
        )
        SimulationManager.setup_simulation(dt=1.0 / 60.0)
        self._timeline.play()
        await step_simulation(0.1)
        self._timeline.stop()

    async def test_is_imu_sensor(self):
        """Test is_imu_sensor returns correct values for valid sensor, invalid prim, and non-existent prim."""
        await stage_utils.create_new_stage_async()
        SimulationManager.setup_simulation(dt=1.0 / 60.0)

        # Create a cube with rigid body
        Cube("/World/Cube", sizes=1.0, positions=[0.0, 0.0, 1.0])
        GeomPrim("/World/Cube", apply_collision_apis=True)
        RigidPrim("/World/Cube", masses=[1.0])
        await omni.kit.app.get_app().next_update_async()

        # Create an IMU sensor on the cube
        result, sensor = omni.kit.commands.execute(
            "IsaacSensorExperimentalCreateImuSensor",
            path="/imu_sensor",
            parent="/World/Cube",
            translation=Gf.Vec3d(0, 0, 0),
            orientation=Gf.Quatd(1, 0, 0, 0),
        )
        self.assertTrue(result)
        self.assertIsNotNone(sensor)
        await omni.kit.app.get_app().next_update_async()

        stage = stage_utils.get_current_stage()

        # Before simulation starts, sensor should not be registered with the manager
        self.assertFalse(self._timeline.is_playing())

        # Start the timeline to register sensors with the manager
        self._timeline.play()
        await omni.kit.app.get_app().next_update_async()

        # After simulation starts, valid IMU sensor path should return True
        prim = stage.GetPrimAtPath("/World/Cube/imu_sensor")
        self.assertTrue(prim.IsValid() and prim.GetTypeName() == "IsaacImuSensor")

        # Test invalid prim (cube itself, not a sensor) - should return False
        prim = stage.GetPrimAtPath("/World/Cube")
        self.assertFalse(prim.GetTypeName() == "IsaacImuSensor")

        # Test non-existent prim path - should return False
        prim = stage.GetPrimAtPath("/World/NonExistent/sensor")
        self.assertFalse(prim.IsValid())

    async def test_is_sensor_reading_defaults(self):
        """Verify IMUSensorReading default construction values."""
        reading = IMUSensorReading()
        self.assertEqual(reading.time, 0.0)
        self.assertEqual(reading.linear_acceleration_x, 0.0)
        self.assertEqual(reading.linear_acceleration_y, 0.0)
        self.assertEqual(reading.linear_acceleration_z, 0.0)
        self.assertEqual(reading.angular_velocity_x, 0.0)
        self.assertEqual(reading.angular_velocity_y, 0.0)
        self.assertEqual(reading.angular_velocity_z, 0.0)
        self.assertFalse(reading.is_valid)

    async def test_invalid_prim_sensor_reading(self):
        """Test that get_sensor_reading returns invalid/zero data for non-sensor prims.

        This test replicates the scenario from test_invalid_imu_sensor_ogn where the
        IMU prim path points to a Cube (not an actual IMU sensor). The sensor backend
        should return invalid readings with zeros, not gravity data.
        """
        await stage_utils.create_new_stage_async()
        SimulationManager.setup_simulation(dt=1.0 / 60.0)

        # Create cube with rigid body - NO ground plane, cube will be in free fall
        Cube("/World/Cube", sizes=1.0, positions=[0.0, 0.0, 10.0])
        GeomPrim("/World/Cube", apply_collision_apis=True)
        RigidPrim("/World/Cube", masses=[1.0])

        # Also create a valid IMU sensor for comparison
        result, sensor = omni.kit.commands.execute(
            "IsaacSensorExperimentalCreateImuSensor",
            path="/imu_sensor",
            parent="/World/Cube",
        )
        self.assertTrue(result)
        await omni.kit.app.get_app().next_update_async()

        # Start simulation - cube is in free fall
        self._timeline.play()
        await step_simulation(0.5)

        # Query sensor reading for the CUBE path (not the sensor) - should be invalid
        cube_reading = self._get_imu_backend("/World/Cube").get_sensor_reading()

        # The reading should be invalid or have zero acceleration values
        # This matches the expected behavior in test_invalid_imu_sensor_ogn
        self.assertFalse(
            cube_reading.is_valid,
            f"Reading for non-sensor prim should be invalid, got is_valid={cube_reading.is_valid}",
        )
        self.assertEqual(
            cube_reading.linear_acceleration_z,
            0.0,
            f"linear_acceleration_z for non-sensor prim should be 0.0, got {cube_reading.linear_acceleration_z}",
        )
        self.assertEqual(cube_reading.linear_acceleration_x, 0.0)
        self.assertEqual(cube_reading.linear_acceleration_y, 0.0)
        self.assertEqual(cube_reading.angular_velocity_x, 0.0)
        self.assertEqual(cube_reading.angular_velocity_y, 0.0)
        self.assertEqual(cube_reading.angular_velocity_z, 0.0)

        # Verify the valid sensor in free fall - IMU should read ~0 acceleration
        # (In free fall, both the sensor and its reference frame fall together)
        sensor_reading = self._get_imu_backend("/World/Cube/imu_sensor").get_sensor_reading()
        self.assertTrue(sensor_reading.is_valid)
        self.assertAlmostEqual(
            sensor_reading.linear_acceleration_z,
            0.0,
            delta=GRAVITY_TOLERANCE,
            msg=f"In free fall, IMU should read ~0 acceleration, got {sensor_reading.linear_acceleration_z}",
        )

    async def test_rigid_prim_velocities_during_free_fall(self):
        """Test that RigidPrim.get_velocities() returns correct data during free fall simulation.

        This test verifies that the RigidPrim API correctly reports velocities during physics
        simulation. A falling cube should have increasing negative Z velocity due to gravity.
        """
        await stage_utils.create_new_stage_async()
        SimulationManager.setup_simulation(dt=1.0 / 60.0)

        # Create a cube in free fall (starting at z=10, no ground plane)
        Cube("/World/Cube", sizes=1.0, positions=[0.0, 0.0, 10.0])
        GeomPrim("/World/Cube", apply_collision_apis=True)
        RigidPrim("/World/Cube", masses=[1.0])

        await omni.kit.app.get_app().next_update_async()

        # Create RigidPrim wrapper to get velocities
        rigid_prim = RigidPrim("/World/Cube")

        # Start simulation
        self._timeline.play()

        # Step a few times to let physics initialize
        await step_simulation(0.1)

        # Check that physics tensor entity is valid
        is_valid = rigid_prim.is_physics_tensor_entity_valid()
        self.assertTrue(
            is_valid,
            "RigidPrim.is_physics_tensor_entity_valid() should return True during simulation",
        )

        # Get velocities - in free fall, the cube should have non-zero negative Z velocity
        linear_vel, _ = rigid_prim.get_velocities()

        linear_velocity_z = float(np.array(linear_vel[0])[2])

        # After 0.1s of free fall: v = g*t ≈ 9.81 * 0.1 ≈ -0.981 m/s (negative because falling)
        # Allow some tolerance for simulation startup
        self.assertLess(
            linear_velocity_z,
            -0.5,
            f"In free fall, Z velocity should be negative (falling), got {linear_velocity_z}",
        )

        # Continue simulation and verify velocity increases
        await step_simulation(0.2)

        linear_vel2, _ = rigid_prim.get_velocities()
        linear_velocity_z2 = float(np.array(linear_vel2[0])[2])

        # After 0.3s total: v ≈ 9.81 * 0.3 ≈ -2.94 m/s
        self.assertLess(
            linear_velocity_z2,
            linear_velocity_z,
            f"Velocity should increase in magnitude during free fall: {linear_velocity_z2} should be < {linear_velocity_z}",
        )

    async def test_reader_reinitialize_during_play(self):
        """Force reader.initialize() while IMU is active, then step physics.

        Reproduces a crash where ImuSensorImpl holds a stale
        IRigidBodyDataView pointer after the reader destroys all views.
        """
        await stage_utils.create_new_stage_async()
        await omni.kit.app.get_app().next_update_async()
        stage_utils.set_stage_units(meters_per_unit=1.0)
        SimulationManager.setup_simulation(dt=1.0 / 60.0)

        cube_path = "/World/Cube"
        Cube(cube_path, sizes=1.0, positions=[0.0, 0.0, 2.0])
        GeomPrim(cube_path, apply_collision_apis=True)
        RigidPrim(cube_path, masses=[1.0])

        result, sensor = omni.kit.commands.execute(
            "IsaacSensorExperimentalCreateImuSensor", path="/imu_sensor", parent=cube_path
        )
        self.assertTrue(result)
        sensor_path = cube_path + "/imu_sensor"
        await omni.kit.app.get_app().next_update_async()

        from isaacsim.core.experimental.prims import _prims_reader

        reader = _prims_reader.acquire_prim_data_reader_interface()
        try:
            self._timeline.play()
            await step_simulation(0.25)

            backend = ImuSensorBackend(sensor_path)
            reading = backend.get_sensor_reading()
            self.assertTrue(reading.is_valid)

            stage = omni.usd.get_context().get_stage()
            stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
            gen_before = reader.get_generation()

            reader.initialize(stage_id, -1)
            self.assertGreater(reader.get_generation(), gen_before)

            await step_simulation(0.25)

            reading_after = backend.get_sensor_reading()
            carb.log_info(f"Post-reinit IMU reading valid={reading_after.is_valid}")
        finally:
            _prims_reader.release_prim_data_reader_interface(reader)

    async def test_multiple_reader_reinitializations(self):
        """Reinitialize the reader several times in rapid succession while IMU is active."""
        await stage_utils.create_new_stage_async()
        await omni.kit.app.get_app().next_update_async()
        stage_utils.set_stage_units(meters_per_unit=1.0)
        SimulationManager.setup_simulation(dt=1.0 / 60.0)

        cube_path = "/World/Cube"
        Cube(cube_path, sizes=1.0, positions=[0.0, 0.0, 2.0])
        GeomPrim(cube_path, apply_collision_apis=True)
        RigidPrim(cube_path, masses=[1.0])

        result, sensor = omni.kit.commands.execute(
            "IsaacSensorExperimentalCreateImuSensor", path="/imu_sensor", parent=cube_path
        )
        self.assertTrue(result)
        sensor_path = cube_path + "/imu_sensor"
        await omni.kit.app.get_app().next_update_async()

        from isaacsim.core.experimental.prims import _prims_reader

        reader = _prims_reader.acquire_prim_data_reader_interface()
        try:
            self._timeline.play()
            await step_simulation(0.25)

            backend = ImuSensorBackend(sensor_path)
            self.assertTrue(backend.get_sensor_reading().is_valid)

            stage = omni.usd.get_context().get_stage()
            stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()

            for _ in range(3):
                reader.initialize(stage_id, -1)
                await step_simulation(0.1)
        finally:
            _prims_reader.release_prim_data_reader_interface(reader)
