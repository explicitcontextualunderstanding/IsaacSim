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
"""Unit tests for isaacsim.physics.newton.tensors rigid body view."""

import isaacsim.physics.newton
import isaacsim.physics.newton.tensors
import numpy as np
import omni.kit.app
import omni.kit.test
import omni.timeline
import omni.usd
import warp as wp
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.utils.stage import create_new_stage_async
from pxr import Gf, UsdGeom, UsdPhysics


async def wait_for_stage_loading():
    """Wait until USD stage loading is complete."""
    while omni.usd.get_context().get_stage_loading_status()[2] > 0:
        await omni.kit.app.get_app().next_update_async()


class TestNewtonRigidBodyView(omni.kit.test.AsyncTestCase):
    """Tests for Newton rigid body view tensor API."""

    async def setUp(self):
        """Set up test environment with rigid bodies."""
        self.use_gpu = True
        self.wp_device = "cuda:0" if self.use_gpu else "cpu"

        await create_new_stage_async()
        self.stage = omni.usd.get_context().get_stage()

        # Create physics scene
        scene_path = "/PhysicsScene"
        scene = UsdPhysics.Scene.Define(self.stage, scene_path)
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr(9.81)

        # Create multiple rigid bodies (cubes)
        self.num_bodies = 3
        self.body_paths = []
        for i in range(self.num_bodies):
            body_path = f"/World/Cube_{i}"
            self.body_paths.append(body_path)

            # Create cube geometry
            cube = UsdGeom.Cube.Define(self.stage, body_path)
            cube.GetSizeAttr().Set(0.5)
            cube.AddTranslateOp().Set(Gf.Vec3f(i * 2.0, 0.0, 1.0))

            # Add rigid body physics
            prim = self.stage.GetPrimAtPath(body_path)
            UsdPhysics.RigidBodyAPI.Apply(prim)
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MassAPI.Apply(prim)

            # Set mass and inertia
            mass_api = UsdPhysics.MassAPI(prim)
            mass_api.CreateMassAttr(1.0)
            # Inertia for uniform cube: I = (1/6) * m * s^2
            cube_size = 0.5
            cube_mass = 1.0
            cube_inertia = (1.0 / 6.0) * cube_mass * cube_size * cube_size
            mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(cube_inertia, cube_inertia, cube_inertia))

        await wait_for_stage_loading()

        success = SimulationManager.switch_physics_engine("newton")
        self.assertTrue(success, "Failed to switch to Newton physics backend")
        await omni.kit.app.get_app().next_update_async()

        self.timeline = omni.timeline.get_timeline_interface()
        self.timeline.play()
        await omni.kit.app.get_app().next_update_async()

        self.newton_stage = isaacsim.physics.newton.acquire_stage()
        self.sim = isaacsim.physics.newton.tensors.create_simulation_view(
            frontend_name="warp", stage_id=-1, newton_stage=self.newton_stage
        )

    async def tearDown(self):
        """Clean up after test."""
        self.timeline.pause()
        await omni.kit.app.get_app().next_update_async()
        await omni.usd.get_context().close_stage_async()

    async def test_rigid_body_view_creation(self):
        """Test creating rigid body view and basic properties."""
        bodies = self.sim.create_rigid_body_view("/World/Cube_*")

        self.assertIsNotNone(bodies)
        self.assertEqual(bodies.count, self.num_bodies, f"Should have {self.num_bodies} rigid bodies")

    async def test_body_paths_and_names(self):
        """Test rigid body paths and names match expected cube structure."""
        bodies = self.sim.create_rigid_body_view("/World/Cube_*")

        # Verify body paths are correct absolute USD paths
        body_paths = bodies.body_paths
        self.assertEqual(len(body_paths), self.num_bodies, f"Should have {self.num_bodies} body paths")
        for i, path in enumerate(body_paths):
            self.assertTrue(path.startswith("/World/Cube_"), f"Path should start with /World/Cube_, got: {path}")
            self.assertIn("Cube_", path, f"Path should contain 'Cube_', got: {path}")

        # Verify body names are extracted correctly
        body_names = bodies.body_names
        self.assertEqual(len(body_names), self.num_bodies, f"Should have {self.num_bodies} body names")
        for name in body_names:
            self.assertIn("Cube_", name, f"Name should contain 'Cube_', got: {name}")

    async def test_transforms(self):
        """Test transforms have correct format and values match scene setup."""
        bodies = self.sim.create_rigid_body_view("/World/Cube_*")
        indices = wp.from_numpy(np.arange(bodies.count, dtype=np.int32), dtype=wp.int32, device=self.wp_device)

        initial_transforms = bodies.get_transforms().numpy()
        self.assertEqual(initial_transforms.shape, (bodies.count, 7), "Transform shape should be (count, 7)")

        # Transforms are [x, y, z, qx, qy, qz, qw] - verify quaternions are normalized
        for i in range(bodies.count):
            quat = initial_transforms[i, 3:7]
            quat_norm = np.linalg.norm(quat)
            self.assertAlmostEqual(quat_norm, 1.0, places=3, msg=f"Quaternion for body {i} should be normalized")

        # Verify initial positions match scene setup (cubes at x=0, 2, 4, y=0, z=1)
        for i in range(bodies.count):
            self.assertAlmostEqual(initial_transforms[i, 1], 0.0, places=2, msg=f"Body {i} y should be ~0")
            self.assertAlmostEqual(initial_transforms[i, 2], 1.0, places=2, msg=f"Body {i} z should be ~1")

        # Modify transforms and verify round-trip
        height_offset = 2.0
        modified_transforms = initial_transforms.copy()
        modified_transforms[:, 2] += height_offset

        transform_wp = wp.from_numpy(modified_transforms, dtype=wp.float32, device=self.wp_device)
        bodies.set_transforms(transform_wp, indices)

        updated_transforms = bodies.get_transforms().numpy()
        for i in range(bodies.count):
            expected_z = initial_transforms[i, 2] + height_offset
            self.assertAlmostEqual(
                updated_transforms[i, 2],
                expected_z,
                places=4,
                msg=f"Body {i} z should be {expected_z}, got {updated_transforms[i, 2]}",
            )

    async def test_velocities(self):
        """Test velocities format and verify set values are applied."""
        bodies = self.sim.create_rigid_body_view("/World/Cube_*")
        indices = wp.from_numpy(np.arange(bodies.count, dtype=np.int32), dtype=wp.int32, device=self.wp_device)

        initial_vel = bodies.get_velocities().numpy()
        self.assertEqual(initial_vel.shape, (bodies.count, 6), "Velocity shape should be (count, 6)")

        # Velocities are [vx, vy, vz, wx, wy, wz] - linear then angular
        # Note: Bodies may have some velocity from gravity during initial simulation step

        # Set specific velocities
        target_linear_x = 2.0
        target_angular_z = 1.0
        modified_vel = np.zeros((bodies.count, 6), dtype=np.float32)
        modified_vel[:, 0] = target_linear_x  # vx
        modified_vel[:, 5] = target_angular_z  # wz

        vel_wp = wp.from_numpy(modified_vel, dtype=wp.float32, device=self.wp_device)
        bodies.set_velocities(vel_wp, indices)

        # Verify set values immediately
        immediate_vel = bodies.get_velocities().numpy()
        for i in range(bodies.count):
            self.assertAlmostEqual(
                immediate_vel[i, 0], target_linear_x, places=4, msg=f"Body {i} vx should be {target_linear_x}"
            )
            self.assertAlmostEqual(
                immediate_vel[i, 5], target_angular_z, places=4, msg=f"Body {i} wz should be {target_angular_z}"
            )

    async def test_accelerations(self):
        """Test getting rigid body accelerations."""
        bodies = self.sim.create_rigid_body_view("/World/Cube_*")

        accelerations = bodies.get_accelerations()
        self.assertIsNotNone(accelerations, "Accelerations should not be None")

        accelerations_np = accelerations.numpy() if hasattr(accelerations, "numpy") else np.array(accelerations)
        self.assertEqual(
            accelerations_np.shape,
            (bodies.count, 6),
            f"Accelerations shape should be ({bodies.count}, 6)",
        )

    async def test_masses(self):
        """Test masses match scene setup and can be modified."""
        bodies = self.sim.create_rigid_body_view("/World/Cube_*")
        indices = wp.from_numpy(np.arange(bodies.count, dtype=np.int32), dtype=wp.int32, device=self.wp_device)

        initial_masses = bodies.get_masses().numpy()

        # All cubes were created with mass=1.0
        for i in range(bodies.count):
            self.assertAlmostEqual(
                initial_masses[i, 0], 1.0, places=2, msg=f"Body {i} mass should be ~1.0, got {initial_masses[i, 0]}"
            )

        # Double all masses and verify
        scale_factor = 2.0
        modified_masses = initial_masses * scale_factor
        masses_wp = wp.from_numpy(modified_masses, dtype=wp.float32, device=self.wp_device)
        bodies.set_masses(masses_wp, indices)

        updated_masses = bodies.get_masses().numpy()
        for i in range(bodies.count):
            expected_mass = initial_masses[i, 0] * scale_factor
            self.assertAlmostEqual(
                updated_masses[i, 0],
                expected_mass,
                places=4,
                msg=f"Body {i} mass should be {expected_mass}, got {updated_masses[i, 0]}",
            )

    async def test_inv_masses(self):
        """Test inverse masses are consistent with masses (inv_mass = 1/mass)."""
        bodies = self.sim.create_rigid_body_view("/World/Cube_*")

        masses = bodies.get_masses().numpy()
        inv_masses = bodies.get_inv_masses().numpy()

        # Inverse mass should equal 1/mass for each body
        for i in range(bodies.count):
            if masses[i, 0] > 1e-6:
                expected_inv = 1.0 / masses[i, 0]
                self.assertAlmostEqual(
                    inv_masses[i, 0],
                    expected_inv,
                    places=4,
                    msg=f"inv_mass[{i}]={inv_masses[i, 0]} should be 1/mass={expected_inv}",
                )

    async def test_coms(self):
        """Test centers of mass have valid format and can be modified."""
        bodies = self.sim.create_rigid_body_view("/World/Cube_*")
        indices = wp.from_numpy(np.arange(bodies.count, dtype=np.int32), dtype=wp.int32, device=self.wp_device)

        initial_coms = bodies.get_coms().numpy()
        self.assertEqual(initial_coms.shape, (bodies.count, 7), "COMs shape should be (count, 7)")

        # COMs format is [x, y, z, qx, qy, qz, qw]
        # For uniform cubes, COM position should be at geometric center (0, 0, 0) in local frame
        # and quaternion should be normalized
        for i in range(bodies.count):
            # Check COM position is near origin (local frame)
            com_pos = initial_coms[i, 0:3]
            self.assertTrue(
                np.allclose(com_pos, [0, 0, 0], atol=0.1), f"Body {i} COM position should be near origin, got {com_pos}"
            )

            # Check quaternion is normalized
            quat = initial_coms[i, 3:7]
            quat_norm = np.linalg.norm(quat)
            self.assertAlmostEqual(quat_norm, 1.0, places=3, msg=f"Body {i} COM quaternion should be normalized")

        # Modify COM positions and verify
        offset = np.array([0.05, 0.0, 0.0])
        modified_coms = initial_coms.copy()
        modified_coms[:, 0:3] += offset

        coms_wp = wp.from_numpy(modified_coms, dtype=wp.float32, device=self.wp_device)
        bodies.set_coms(coms_wp, indices)

        updated_coms = bodies.get_coms().numpy()
        for i in range(bodies.count):
            expected_pos = initial_coms[i, 0:3] + offset
            self.assertTrue(
                np.allclose(updated_coms[i, 0:3], expected_pos, atol=1e-5),
                f"Body {i} COM position should be {expected_pos}, got {updated_coms[i, 0:3]}",
            )

    async def test_inertias(self):
        """Test inertias have correct format and diagonal elements for uniform cubes."""
        bodies = self.sim.create_rigid_body_view("/World/Cube_*")
        indices = wp.from_numpy(np.arange(bodies.count, dtype=np.int32), dtype=wp.int32, device=self.wp_device)

        initial_inertias = bodies.get_inertias().numpy()
        self.assertEqual(initial_inertias.shape, (bodies.count, 9), "Inertias shape should be (count, 9)")

        # Inertia is flattened 3x3: [Ixx, Ixy, Ixz, Iyx, Iyy, Iyz, Izx, Izy, Izz]
        # For uniform cubes, diagonal elements should be equal (Ixx = Iyy = Izz)
        # and off-diagonal elements should be ~0
        for i in range(bodies.count):
            Ixx, Ixy, Ixz = initial_inertias[i, 0], initial_inertias[i, 1], initial_inertias[i, 2]
            Iyx, Iyy, Iyz = initial_inertias[i, 3], initial_inertias[i, 4], initial_inertias[i, 5]
            Izx, Izy, Izz = initial_inertias[i, 6], initial_inertias[i, 7], initial_inertias[i, 8]

            # Diagonal should be positive
            self.assertGreater(Ixx, 0, f"Body {i} Ixx should be positive")
            self.assertGreater(Iyy, 0, f"Body {i} Iyy should be positive")
            self.assertGreater(Izz, 0, f"Body {i} Izz should be positive")

            # For uniform cube, Ixx ≈ Iyy ≈ Izz
            self.assertAlmostEqual(Ixx, Iyy, places=3, msg=f"Body {i}: Ixx should equal Iyy for uniform cube")
            self.assertAlmostEqual(Iyy, Izz, places=3, msg=f"Body {i}: Iyy should equal Izz for uniform cube")

            # Off-diagonal should be ~0 for axis-aligned cube
            self.assertAlmostEqual(Ixy, 0, places=3, msg=f"Body {i}: Ixy should be ~0")
            self.assertAlmostEqual(Ixz, 0, places=3, msg=f"Body {i}: Ixz should be ~0")

        # Scale inertias and verify round-trip
        scale_factor = 1.5
        modified_inertias = initial_inertias * scale_factor
        inertias_wp = wp.from_numpy(modified_inertias, dtype=wp.float32, device=self.wp_device)
        bodies.set_inertias(inertias_wp, indices)

        updated_inertias = bodies.get_inertias().numpy()
        self.assertTrue(
            np.allclose(updated_inertias, modified_inertias, atol=1e-5),
            "Inertias should match scaled values",
        )

    async def test_inv_inertias(self):
        """Test inverse inertias are consistent with inertias."""
        bodies = self.sim.create_rigid_body_view("/World/Cube_*")

        inertias = bodies.get_inertias().numpy()
        inv_inertias = bodies.get_inv_inertias().numpy()

        self.assertEqual(inv_inertias.shape, (bodies.count, 9), "Inverse inertias shape should be (count, 9)")

        # For diagonal inertia tensor, inverse diagonal should be 1/I
        # Check diagonal elements: indices 0 (Ixx), 4 (Iyy), 8 (Izz)
        for i in range(bodies.count):
            for diag_idx in [0, 4, 8]:
                if inertias[i, diag_idx] > 1e-6:
                    expected_inv = 1.0 / inertias[i, diag_idx]
                    self.assertAlmostEqual(
                        inv_inertias[i, diag_idx],
                        expected_inv,
                        places=3,
                        msg=f"Body {i} inv_inertia[{diag_idx}] should be 1/inertia",
                    )

    async def test_disable_simulations(self):
        """Test getting disable simulation flags."""
        bodies = self.sim.create_rigid_body_view("/World/Cube_*")

        disable_sims = bodies.get_disable_simulations()
        self.assertIsNotNone(disable_sims, "Disable simulations should not be None")

    async def test_disable_gravities(self):
        """Test getting disable gravity flags."""
        bodies = self.sim.create_rigid_body_view("/World/Cube_*")

        disable_gravities = bodies.get_disable_gravities()
        self.assertIsNotNone(disable_gravities, "Disable gravities should not be None")

    async def test_apply_forces_and_torques(self):
        """Test applying forces and torques to rigid bodies."""
        bodies = self.sim.create_rigid_body_view("/World/Cube_*")
        indices = wp.from_numpy(np.arange(bodies.count, dtype=np.int32), dtype=wp.int32, device=self.wp_device)

        # Create force and torque arrays
        forces = np.zeros((bodies.count, 3), dtype=np.float32)
        forces[:, 2] = 100.0  # Apply upward force

        torques = np.zeros((bodies.count, 3), dtype=np.float32)
        torques[:, 2] = 10.0  # Apply torque around z-axis

        # Apply forces and torques
        forces_wp = wp.from_numpy(forces, dtype=wp.float32, device=self.wp_device)
        torques_wp = wp.from_numpy(torques, dtype=wp.float32, device=self.wp_device)

        # This should not raise an error
        bodies.apply_forces_and_torques_at_position(forces_wp, torques_wp, None, indices, is_global=True)

        # Step simulation to apply forces
        await omni.kit.app.get_app().next_update_async()

    async def test_apply_forces_at_position(self):
        """Test applying forces at specific positions."""
        bodies = self.sim.create_rigid_body_view("/World/Cube_*")
        indices = wp.from_numpy(np.arange(bodies.count, dtype=np.int32), dtype=wp.int32, device=self.wp_device)

        # Create force and position arrays
        forces = np.zeros((bodies.count, 3), dtype=np.float32)
        forces[:, 2] = 50.0  # Apply upward force

        positions = np.zeros((bodies.count, 3), dtype=np.float32)
        positions[:, 0] = 0.1  # Offset position to create torque

        # Apply forces at positions
        forces_wp = wp.from_numpy(forces, dtype=wp.float32, device=self.wp_device)
        positions_wp = wp.from_numpy(positions, dtype=wp.float32, device=self.wp_device)

        # This should not raise an error
        bodies.apply_forces_and_torques_at_position(forces_wp, None, positions_wp, indices, is_global=True)

        # Step simulation to apply forces
        await omni.kit.app.get_app().next_update_async()

    async def test_simulation_stepping(self):
        """Test bodies fall under gravity during simulation."""
        bodies = self.sim.create_rigid_body_view("/World/Cube_*")

        initial_transforms = bodies.get_transforms().numpy()
        initial_z_positions = initial_transforms[:, 2].copy()

        # Step simulation multiple times
        num_steps = 10
        for _ in range(num_steps):
            await omni.kit.app.get_app().next_update_async()

        current_transforms = bodies.get_transforms().numpy()
        current_z_positions = current_transforms[:, 2]

        # All bodies should have fallen (z decreased) due to gravity
        for i in range(bodies.count):
            self.assertLess(
                current_z_positions[i],
                initial_z_positions[i],
                f"Body {i} should have fallen: initial z={initial_z_positions[i]}, current z={current_z_positions[i]}",
            )

        # Verify significant fall (at least 0.1m after 10 steps with gravity)
        avg_fall = np.mean(initial_z_positions - current_z_positions)
        self.assertGreater(avg_fall, 0.1, f"Average fall distance should be > 0.1m, got {avg_fall}")

    async def test_check_method(self):
        """Test the check method returns correct state."""
        bodies = self.sim.create_rigid_body_view("/World/Cube_*")

        result = bodies.check()
        self.assertTrue(result, "check() should return True for valid rigid body view")
