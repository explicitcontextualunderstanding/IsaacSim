# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Test suite for batch_compute_collider_transforms function."""

import isaacsim.robot_motion.cumotion as cu_mg
import numpy as np
import omni.kit.test
import warp as wp


class TestBatchColliderTransforms(omni.kit.test.AsyncTestCase):
    """Test suite for batch collider transform computation."""

    async def setUp(self):
        pass

    async def tearDown(self):
        pass

    # ============================================================================
    # Test batch_compute_collider_transforms
    # ============================================================================

    async def test_batch_compute_collider_transforms_identity(self):
        """Test batch collider transform computation with identity transforms."""
        # Identity transforms everywhere - output should equal input
        position_base_to_world = wp.array([[0.0, 0.0, 0.0]], dtype=wp.float32)
        quaternion_base_to_world = wp.array([[1.0, 0.0, 0.0, 0.0]], dtype=wp.float32)

        # 2 objects
        positions_world_to_object = wp.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=wp.float32)
        quaternions_world_to_object = wp.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=wp.float32)

        # 5 colliders: 2 for object 0, 3 for object 1
        positions_object_to_collider = wp.array(
            [
                [0.1, 0.0, 0.0],  # Object 0, collider 0
                [0.2, 0.0, 0.0],  # Object 0, collider 1
                [0.3, 0.0, 0.0],  # Object 1, collider 0
                [0.4, 0.0, 0.0],  # Object 1, collider 1
                [0.5, 0.0, 0.0],  # Object 1, collider 2
            ],
            dtype=wp.float32,
        )
        quaternions_object_to_collider = wp.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            dtype=wp.float32,
        )

        num_colliders_per_object = [2, 3]

        # Compute transforms
        output = cu_mg.impl.utils.batch_compute_collider_transforms(
            position_base_to_world=position_base_to_world,
            quaternion_base_to_world=quaternion_base_to_world,
            positions_world_to_object=positions_world_to_object,
            quaternions_world_to_object=quaternions_world_to_object,
            positions_object_to_collider=positions_object_to_collider,
            quaternions_object_to_collider=quaternions_object_to_collider,
            num_colliders_per_object=num_colliders_per_object,
        )

        # Expected: T_base_collider = T_base_world * T_world_object * T_object_collider
        # With identity base and rotations, this is just position addition
        expected_positions_base = np.array(
            [
                [1.1, 0.0, 0.0],  # Object 0 at [1,0,0] + collider at [0.1,0,0]
                [1.2, 0.0, 0.0],  # Object 0 at [1,0,0] + collider at [0.2,0,0]
                [2.3, 0.0, 0.0],  # Object 1 at [2,0,0] + collider at [0.3,0,0]
                [2.4, 0.0, 0.0],  # Object 1 at [2,0,0] + collider at [0.4,0,0]
                [2.5, 0.0, 0.0],  # Object 1 at [2,0,0] + collider at [0.5,0,0]
            ]
        )

        # For identity base frame, world and base transforms should be the same
        expected_positions_world = expected_positions_base.copy()

        self.assertTrue(np.allclose(output.positions_base_to_collider.numpy(), expected_positions_base, atol=1e-5))
        self.assertTrue(np.allclose(output.positions_world_to_collider.numpy(), expected_positions_world, atol=1e-5))

        # All quaternions should remain identity
        self.assertTrue(np.allclose(output.quaternions_base_to_collider.numpy()[:, 0], 1.0, atol=1e-5))  # w component
        self.assertTrue(np.allclose(output.quaternions_world_to_collider.numpy()[:, 0], 1.0, atol=1e-5))  # w component

    async def test_batch_compute_collider_transforms_with_base_translation(self):
        """Test with translated base frame."""
        # Base frame translated by [-1, 0, 0]
        position_base_to_world = wp.array([[-1.0, 0.0, 0.0]], dtype=wp.float32)
        quaternion_base_to_world = wp.array([[1.0, 0.0, 0.0, 0.0]], dtype=wp.float32)

        # 1 object at world origin
        positions_world_to_object = wp.array([[0.0, 0.0, 0.0]], dtype=wp.float32)
        quaternions_world_to_object = wp.array([[1.0, 0.0, 0.0, 0.0]], dtype=wp.float32)

        # 1 collider at [1, 0, 0] relative to object
        positions_object_to_collider = wp.array([[1.0, 0.0, 0.0]], dtype=wp.float32)
        quaternions_object_to_collider = wp.array([[1.0, 0.0, 0.0, 0.0]], dtype=wp.float32)

        num_colliders_per_object = [1]

        output = cu_mg.impl.utils.batch_compute_collider_transforms(
            position_base_to_world=position_base_to_world,
            quaternion_base_to_world=quaternion_base_to_world,
            positions_world_to_object=positions_world_to_object,
            quaternions_world_to_object=quaternions_world_to_object,
            positions_object_to_collider=positions_object_to_collider,
            quaternions_object_to_collider=quaternions_object_to_collider,
            num_colliders_per_object=num_colliders_per_object,
        )

        # Collider is at [1,0,0] in world frame
        expected_position_world = np.array([[1.0, 0.0, 0.0]])
        self.assertTrue(np.allclose(output.positions_world_to_collider.numpy(), expected_position_world, atol=1e-5))

        # In base frame collider is at [0, 0, 0]
        expected_position_base = np.array([[0.0, 0.0, 0.0]])
        self.assertTrue(np.allclose(output.positions_base_to_collider.numpy(), expected_position_base, atol=1e-5))

    async def test_batch_compute_collider_transforms_with_rotation(self):
        """Test with 90-degree rotation about Z-axis."""
        # Base frame rotated 90 degrees about Z-axis
        position_base_to_world = wp.array([[0.0, 0.0, 0.0]], dtype=wp.float32)
        quaternion_base_to_world = wp.array([[np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]], dtype=wp.float32)

        # 1 object at [1, 0, 0] in world
        positions_world_to_object = wp.array([[1.0, 0.0, 0.0]], dtype=wp.float32)
        quaternions_world_to_object = wp.array([[1.0, 0.0, 0.0, 0.0]], dtype=wp.float32)

        # 1 collider at origin relative to object
        positions_object_to_collider = wp.array([[0.0, 0.0, 0.0]], dtype=wp.float32)
        quaternions_object_to_collider = wp.array([[1.0, 0.0, 0.0, 0.0]], dtype=wp.float32)

        num_colliders_per_object = [1]

        output = cu_mg.impl.utils.batch_compute_collider_transforms(
            position_base_to_world=position_base_to_world,
            quaternion_base_to_world=quaternion_base_to_world,
            positions_world_to_object=positions_world_to_object,
            quaternions_world_to_object=quaternions_world_to_object,
            positions_object_to_collider=positions_object_to_collider,
            quaternions_object_to_collider=quaternions_object_to_collider,
            num_colliders_per_object=num_colliders_per_object,
        )

        # World frame: collider at [1, 0, 0]
        expected_position_world = np.array([[1.0, 0.0, 0.0]])
        expected_quaternion_world = np.array([[1.0, 0.0, 0.0, 0.0]])

        # Base frame: point at [1, 0, 0] rotated 90 degrees about Z becomes [0, 1, 0]
        expected_position_base = np.array([[0.0, 1.0, 0.0]])
        expected_quaternion_base = np.array([[np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]])

        self.assertTrue(np.allclose(output.positions_world_to_collider.numpy(), expected_position_world, atol=1e-5))
        self.assertTrue(np.allclose(output.quaternions_world_to_collider.numpy(), expected_quaternion_world, atol=1e-5))

        self.assertTrue(np.allclose(output.positions_base_to_collider.numpy(), expected_position_base, atol=1e-5))
        self.assertTrue(np.allclose(output.quaternions_base_to_collider.numpy(), expected_quaternion_base, atol=1e-5))

    async def test_batch_compute_collider_transforms_complex_scenario(self):
        """Test with multiple objects, colliders, and complex transforms."""
        # Base frame: translated and rotated
        position_base_to_world = wp.array([[-1.0, 0.0, 0.0]], dtype=wp.float32)
        # 90 degree rotation about Y-axis
        quaternion_base_to_world = wp.array([[np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0]], dtype=wp.float32)

        # 2 objects
        positions_world_to_object = wp.array(
            [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=wp.float32  # Object 0  # Object 1
        )
        quaternions_world_to_object = wp.array(
            [
                [1.0, 0.0, 0.0, 0.0],  # Object 0: no rotation
                [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)],  # Object 1: 90 deg about Z
            ],
            dtype=wp.float32,
        )

        # 3 colliders: 1 for object 0, 2 for object 1
        positions_object_to_collider = wp.array(
            [
                [0.0, 0.0, 0.0],  # Object 0, collider 0 (at object origin)
                [1.0, 0.0, 0.0],  # Object 1, collider 0
                [0.0, 1.0, 0.0],  # Object 1, collider 1
            ],
            dtype=wp.float32,
        )
        quaternions_object_to_collider = wp.array(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=wp.float32
        )

        num_colliders_per_object = [1, 2]

        output = cu_mg.impl.utils.batch_compute_collider_transforms(
            position_base_to_world=position_base_to_world,
            quaternion_base_to_world=quaternion_base_to_world,
            positions_world_to_object=positions_world_to_object,
            quaternions_world_to_object=quaternions_world_to_object,
            positions_object_to_collider=positions_object_to_collider,
            quaternions_object_to_collider=quaternions_object_to_collider,
            num_colliders_per_object=num_colliders_per_object,
        )

        # Verify we got 3 output transforms
        self.assertEqual(output.positions_base_to_collider.shape[0], 3)
        self.assertEqual(output.quaternions_base_to_collider.shape[0], 3)
        self.assertEqual(output.positions_world_to_collider.shape[0], 3)
        self.assertEqual(output.quaternions_world_to_collider.shape[0], 3)

        # Verify all quaternions are normalized (both base and world)
        for i in range(3):
            quat_base = output.quaternions_base_to_collider.numpy()[i]
            quat_base_norm = np.linalg.norm(quat_base)
            self.assertTrue(np.isclose(quat_base_norm, 1.0, atol=1e-5))

            quat_world = output.quaternions_world_to_collider.numpy()[i]
            quat_world_norm = np.linalg.norm(quat_world)
            self.assertTrue(np.isclose(quat_world_norm, 1.0, atol=1e-5))

    async def test_batch_compute_collider_transforms_single_collider(self):
        """Test edge case with single object and single collider."""
        position_base_to_world = wp.array([[0.0, 0.0, 0.0]], dtype=wp.float32)
        quaternion_base_to_world = wp.array([[1.0, 0.0, 0.0, 0.0]], dtype=wp.float32)

        positions_world_to_object = wp.array([[1.0, 2.0, 3.0]], dtype=wp.float32)
        quaternions_world_to_object = wp.array([[1.0, 0.0, 0.0, 0.0]], dtype=wp.float32)

        positions_object_to_collider = wp.array([[0.5, 0.5, 0.5]], dtype=wp.float32)
        quaternions_object_to_collider = wp.array([[1.0, 0.0, 0.0, 0.0]], dtype=wp.float32)

        num_colliders_per_object = [1]

        output = cu_mg.impl.utils.batch_compute_collider_transforms(
            position_base_to_world=position_base_to_world,
            quaternion_base_to_world=quaternion_base_to_world,
            positions_world_to_object=positions_world_to_object,
            quaternions_world_to_object=quaternions_world_to_object,
            positions_object_to_collider=positions_object_to_collider,
            quaternions_object_to_collider=quaternions_object_to_collider,
            num_colliders_per_object=num_colliders_per_object,
        )

        # With identity base frame, world and base positions should be the same
        expected_position = np.array([[1.5, 2.5, 3.5]])
        self.assertTrue(np.allclose(output.positions_base_to_collider.numpy(), expected_position, atol=1e-5))
        self.assertTrue(np.allclose(output.positions_world_to_collider.numpy(), expected_position, atol=1e-5))
