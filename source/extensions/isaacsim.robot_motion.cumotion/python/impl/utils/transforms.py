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

from __future__ import annotations

import cumotion
import numpy as np
import warp as wp

# ============================================================================
# Warp Kernel for Batch Collider Transform Computation
# ============================================================================


@wp.struct
class ColliderBatchTransformOutput:
    """Output structure for batch collider transform computation.

    Attributes:
        positions_base_to_collider: Collider positions in base frame, shape (M, 3).
        quaternions_base_to_collider: Collider quaternions in base frame (w, x, y, z), shape (M, 4).
        positions_world_to_collider: Collider positions in world frame, shape (M, 3).
        quaternions_world_to_collider: Collider quaternions in world frame (w, x, y, z), shape (M, 4).
    """

    positions_base_to_collider: wp.array(ndim=2, dtype=wp.float32)
    quaternions_base_to_collider: wp.array(ndim=2, dtype=wp.float32)
    positions_world_to_collider: wp.array(ndim=2, dtype=wp.float32)
    quaternions_world_to_collider: wp.array(ndim=2, dtype=wp.float32)


@wp.kernel
def compute_collider_transforms_kernel(
    # World-to-base transform (single transform, constant for all)
    position_base_to_world: wp.array(dtype=wp.float32, ndim=2),  # Shape: (1, 3)
    quaternion_base_to_world: wp.array(dtype=wp.float32, ndim=2),  # Shape: (1, 4) - (w, x, y, z)
    # Object transforms (N objects)
    positions_world_to_object: wp.array(dtype=wp.float32, ndim=2),  # Shape: (N, 3)
    quaternions_world_to_object: wp.array(dtype=wp.float32, ndim=2),  # Shape: (N, 4) - (w, x, y, z)
    # Collider transforms relative to their parent objects (M total colliders)
    positions_object_to_collider: wp.array(dtype=wp.float32, ndim=2),  # Shape: (M, 3)
    quaternions_object_to_collider: wp.array(dtype=wp.float32, ndim=2),  # Shape: (M, 4) - (w, x, y, z)
    # Mapping from collider index to object index
    collider_to_object_indices: wp.array(dtype=wp.int32),  # Shape: (M,)
    # Outputs: collider transforms in world frame
    output_transforms: ColliderBatchTransformOutput,
):
    """Compute transforms for all colliders: T_base_collider = T_base_world * T_world_object * T_object_collider.

    Each thread handles one collider and computes its transform in the base frame
    by composing three transforms:
    1. Base-to-world transform (constant for all)
    2. World-to-object transform (varies per object)
    3. Object-to-collider transform (varies per collider, several colliders per object)
    """
    collider_idx = wp.tid()

    # Get which object this collider belongs to
    object_idx = collider_to_object_indices[collider_idx]

    uniform_scale = wp.vec3(1.0, 1.0, 1.0)

    # Read base-to-world transform (constant, stored in first row)
    t_base_world = wp.vec3(position_base_to_world[0, 0], position_base_to_world[0, 1], position_base_to_world[0, 2])
    q_base_world = wp.quat(
        quaternion_base_to_world[0, 1],  # x
        quaternion_base_to_world[0, 2],  # y
        quaternion_base_to_world[0, 3],  # z
        quaternion_base_to_world[0, 0],  # w
    )
    transform_base_world_matrix = wp.transform_compose(
        t_base_world,
        q_base_world,
        uniform_scale,
    )
    transform_base_world = wp.transform_from_matrix(transform_base_world_matrix)

    # Read world-to-object transform for this object
    t_world_object = wp.vec3(
        positions_world_to_object[object_idx, 0],
        positions_world_to_object[object_idx, 1],
        positions_world_to_object[object_idx, 2],
    )
    q_world_object = wp.quat(
        quaternions_world_to_object[object_idx, 1],  # x
        quaternions_world_to_object[object_idx, 2],  # y
        quaternions_world_to_object[object_idx, 3],  # z
        quaternions_world_to_object[object_idx, 0],  # w
    )
    transform_world_object_matrix = wp.transform_compose(
        t_world_object,
        q_world_object,
        uniform_scale,
    )
    transform_world_object = wp.transform_from_matrix(transform_world_object_matrix)

    # Read object-to-collider transform for this collider
    t_object_collider = wp.vec3(
        positions_object_to_collider[collider_idx, 0],
        positions_object_to_collider[collider_idx, 1],
        positions_object_to_collider[collider_idx, 2],
    )
    q_object_collider = wp.quat(
        quaternions_object_to_collider[collider_idx, 1],  # x
        quaternions_object_to_collider[collider_idx, 2],  # y
        quaternions_object_to_collider[collider_idx, 3],  # z
        quaternions_object_to_collider[collider_idx, 0],  # w
    )
    transform_object_collider_matrix = wp.transform_compose(
        t_object_collider,
        q_object_collider,
        uniform_scale,
    )
    transform_object_collider = wp.transform_from_matrix(transform_object_collider_matrix)

    # Multiply the transforms: T_base_collider = T_base_world * T_world_object * T_object_collider
    transform_world_to_collider = wp.transform_multiply(transform_world_object, transform_object_collider)
    position_world_collider_out = wp.transform_get_translation(transform_world_to_collider)
    quaternion_world_collider_out = wp.transform_get_rotation(transform_world_to_collider)

    # Write output (convert back to Isaac Sim format: w, x, y, z)
    output_transforms.positions_world_to_collider[collider_idx, 0] = position_world_collider_out[0]
    output_transforms.positions_world_to_collider[collider_idx, 1] = position_world_collider_out[1]
    output_transforms.positions_world_to_collider[collider_idx, 2] = position_world_collider_out[2]

    output_transforms.quaternions_world_to_collider[collider_idx, 0] = quaternion_world_collider_out[3]  # w
    output_transforms.quaternions_world_to_collider[collider_idx, 1] = quaternion_world_collider_out[0]  # x
    output_transforms.quaternions_world_to_collider[collider_idx, 2] = quaternion_world_collider_out[1]  # y
    output_transforms.quaternions_world_to_collider[collider_idx, 3] = quaternion_world_collider_out[2]  # z

    transform_base_to_collider = wp.transform_multiply(transform_base_world, transform_world_to_collider)
    position_base_collider_out = wp.transform_get_translation(transform_base_to_collider)
    quaternion_base_collider_out = wp.transform_get_rotation(transform_base_to_collider)

    # Write output (convert back to Isaac Sim format: w, x, y, z)
    output_transforms.positions_base_to_collider[collider_idx, 0] = position_base_collider_out[0]
    output_transforms.positions_base_to_collider[collider_idx, 1] = position_base_collider_out[1]
    output_transforms.positions_base_to_collider[collider_idx, 2] = position_base_collider_out[2]

    output_transforms.quaternions_base_to_collider[collider_idx, 0] = quaternion_base_collider_out[3]  # w
    output_transforms.quaternions_base_to_collider[collider_idx, 1] = quaternion_base_collider_out[0]  # x
    output_transforms.quaternions_base_to_collider[collider_idx, 2] = quaternion_base_collider_out[1]  # y
    output_transforms.quaternions_base_to_collider[collider_idx, 3] = quaternion_base_collider_out[2]  # z


def batch_compute_collider_transforms(
    position_base_to_world: wp.array,  # Shape: (1, 3) or (3,)
    quaternion_base_to_world: wp.array,  # Shape: (1, 4) or (4,) - (w, x, y, z)
    positions_world_to_object: wp.array,  # Shape: (N, 3)
    quaternions_world_to_object: wp.array,  # Shape: (N, 4) - (w, x, y, z)
    positions_object_to_collider: wp.array,  # Shape: (M, 3)
    quaternions_object_to_collider: wp.array,  # Shape: (M, 4) - (w, x, y, z)
    num_colliders_per_object: list[int],  # Length N, ith element = number of colliders for object i
    device: wp.Device | None = None,
) -> ColliderBatchTransformOutput:
    """Batch compute all collider transforms in base frame using a Warp kernel.

    Computes T_base_collider = T_base_world * T_world_object * T_object_collider
    for all colliders in parallel on the GPU.

    Args:
        position_base_to_world: Single transform position, shape (1, 3) or (3,).
        quaternion_base_to_world: Single transform quaternion (w, x, y, z), shape (1, 4) or (4,).
        positions_world_to_object: Object positions in world frame, shape (N, 3).
        quaternions_world_to_object: Object quaternions in world frame (w, x, y, z), shape (N, 4).
        positions_object_to_collider: Collider positions relative to objects, shape (M, 3).
            All colliders are flattened - first all colliders of object 0, then object 1, etc.
        quaternions_object_to_collider: Collider quaternions relative to objects, shape (M, 4).
        num_colliders_per_object: List of length N where element i is the number of
            colliders for object i.
        device: Device to run the kernel on. Defaults to None.

    Returns:
        Structure containing positions and quaternions for colliders in both base and world frames.

    Example:

        .. code-block:: python

            # 2 objects: first has 2 colliders, second has 3 colliders
            num_colliders_per_object = [2, 3]
            positions_object_to_collider = wp.array([
                [0.1, 0.0, 0.0],  # Object 0, collider 0
                [0.2, 0.0, 0.0],  # Object 0, collider 1
                [0.3, 0.0, 0.0],  # Object 1, collider 0
                [0.4, 0.0, 0.0],  # Object 1, collider 1
                [0.5, 0.0, 0.0],  # Object 1, collider 2
            ], dtype=wp.float32)
    """

    # Ensure inputs are 2D
    if position_base_to_world.ndim == 1:
        position_base_to_world = position_base_to_world.reshape((1, 3))
    if quaternion_base_to_world.ndim == 1:
        quaternion_base_to_world = quaternion_base_to_world.reshape((1, 4))

    # Total number of colliders
    total_colliders = int(sum(num_colliders_per_object))

    # Build collider-to-object index mapping
    collider_to_object_indices_list = []
    for object_idx, num_colliders in enumerate(num_colliders_per_object):
        collider_to_object_indices_list.extend([object_idx] * int(num_colliders))

    collider_to_object_indices = wp.array(collider_to_object_indices_list, dtype=wp.int32, device=device)

    # Allocate output arrays
    output = ColliderBatchTransformOutput()
    output.positions_base_to_collider = wp.zeros((total_colliders, 3), dtype=wp.float32, device=device)
    output.quaternions_base_to_collider = wp.zeros((total_colliders, 4), dtype=wp.float32, device=device)
    output.positions_world_to_collider = wp.zeros((total_colliders, 3), dtype=wp.float32, device=device)
    output.quaternions_world_to_collider = wp.zeros((total_colliders, 4), dtype=wp.float32, device=device)

    # Launch kernel - one thread per collider
    wp.launch(
        kernel=compute_collider_transforms_kernel,
        dim=total_colliders,
        inputs=[
            position_base_to_world,
            quaternion_base_to_world,
            positions_world_to_object,
            quaternions_world_to_object,
            positions_object_to_collider,
            quaternions_object_to_collider,
            collider_to_object_indices,
        ],
        outputs=[output],
        device=device,
    )

    return output


def _position_quaternion_to_cumotion_pose(
    position: np.ndarray,
    quaternion: np.ndarray,
):
    """Convert position and quaternion arrays to a cuMotion Pose3 object.

    Args:
        position: Position array of size 3.
        quaternion: Quaternion array of size 4 in (w, x, y, z) format.

    Returns:
        cuMotion Pose3 object.

    Raises:
        ValueError: If position size is not 3.
        ValueError: If quaternion size is not 4.
    """

    if position.size != 3:
        raise ValueError("Position must be of size 3.")

    if quaternion.size != 4:
        raise ValueError("Quaternion must be of size 4 (w, x, y, z).")

    position = position.flatten()
    quaternion = quaternion.flatten()

    return cumotion.Pose3(translation=position, rotation=cumotion.Rotation3(*quaternion))


def _to_numpy(input_array: list[float] | wp.array | np.ndarray):
    """Convert input to numpy array.

    Args:
        input_array: Input data as list, warp array, or numpy array.

    Returns:
        Numpy array.
    """
    if isinstance(input_array, wp.array):
        return input_array.numpy()
    return np.array(input_array)


def isaac_sim_to_cumotion_pose(
    position_world_to_target: wp.array | np.ndarray | list[float],
    orientation_world_to_target: wp.array | np.ndarray | list[float],
    position_world_to_base: wp.array | np.ndarray | list[float] | None = None,
    orientation_world_to_base: wp.array | np.ndarray | list[float] | None = None,
) -> cumotion.Pose3:
    """Convert Isaac Sim pose to cuMotion pose in the robot base frame.

    Transforms a pose from Isaac Sim world frame to cuMotion's robot base frame.
    If base frame transform is not provided, returns the pose in world frame.

    Args:
        position_world_to_target: Target position in world frame [x, y, z].
        orientation_world_to_target: Target orientation in world frame as quaternion [w, x, y, z].
        position_world_to_base: Robot base position in world frame. Defaults to None (origin).
        orientation_world_to_base: Robot base orientation in world frame as quaternion [w, x, y, z].
            Defaults to None (identity).

    Returns:
        cuMotion Pose3 object representing the target pose in robot base frame.

    Example:

        .. code-block:: python

            pose = isaac_sim_to_cumotion_pose(
                position_world_to_target=[1.0, 2.0, 3.0],
                orientation_world_to_target=[1.0, 0.0, 0.0, 0.0],
                position_world_to_base=[0.5, 0.5, 0.0],
                orientation_world_to_base=[1.0, 0.0, 0.0, 0.0]
            )
    """
    position_world_to_target = _to_numpy(position_world_to_target)
    orientation_world_to_target = _to_numpy(orientation_world_to_target)

    transform_world_to_target = _position_quaternion_to_cumotion_pose(
        position=position_world_to_target, quaternion=orientation_world_to_target
    )

    if (position_world_to_base is None) and (orientation_world_to_base is None):
        return transform_world_to_target

    if position_world_to_base is not None:
        position_world_to_base = _to_numpy(position_world_to_base)
    else:
        position_world_to_base = np.zeros(
            [
                3,
            ]
        )

    if orientation_world_to_base is not None:
        orientation_world_to_base = _to_numpy(orientation_world_to_base)
    else:
        orientation_world_to_base = np.array([1.0, 0.0, 0.0, 0.0])

    transform_world_to_base = _position_quaternion_to_cumotion_pose(
        position=position_world_to_base, quaternion=orientation_world_to_base
    )

    # convert to the base-frame, which is the frame used in cumotion:
    return transform_world_to_base.inverse() * transform_world_to_target


def cumotion_to_isaac_sim_pose(
    pose_base_to_target: cumotion.Pose3,
    position_world_to_base: wp.array | np.ndarray | list[float] | None = None,
    orientation_world_to_base: wp.array | np.ndarray | list[float] | None = None,
) -> tuple[wp.array, wp.array]:
    """Convert cuMotion pose in base frame to Isaac Sim pose in world frame.

    Transforms a pose from cuMotion's robot base frame to Isaac Sim world frame.
    If base frame transform is not provided, assumes identity transform.

    Args:
        pose_base_to_target: Target pose in robot base frame.
        position_world_to_base: Robot base position in world frame. Defaults to None (origin).
        orientation_world_to_base: Robot base orientation in world frame as quaternion [w, x, y, z].
            Defaults to None (identity).

    Returns:
        Tuple of (position, quaternion) as warp arrays where position has shape (3,)
        and quaternion has shape (4,) in (w, x, y, z) format.

    Example:

        .. code-block:: python

            position, quaternion = cumotion_to_isaac_sim_pose(
                pose_base_to_target,
                position_world_to_base=[0.5, 0.5, 0.0],
                orientation_world_to_base=[1.0, 0.0, 0.0, 0.0]
            )
    """
    if position_world_to_base is not None:
        position_world_to_base = _to_numpy(position_world_to_base)
    else:
        position_world_to_base = np.zeros(
            [
                3,
            ]
        )

    if orientation_world_to_base is not None:
        orientation_world_to_base = _to_numpy(orientation_world_to_base)
    else:
        orientation_world_to_base = np.array([1.0, 0.0, 0.0, 0.0])

    transform_world_to_base = _position_quaternion_to_cumotion_pose(
        position=position_world_to_base, quaternion=orientation_world_to_base
    )

    pose_world_to_target = transform_world_to_base * pose_base_to_target

    translation = wp.from_numpy(pose_world_to_target.translation, dtype=wp.float32)
    rotation = wp.array(
        [
            pose_world_to_target.rotation.w(),
            pose_world_to_target.rotation.x(),
            pose_world_to_target.rotation.y(),
            pose_world_to_target.rotation.z(),
        ],
        dtype=wp.float32,
    )
    return translation, rotation


def isaac_sim_to_cumotion_translation(
    position_world_to_target: wp.array | np.ndarray | list[float],
    position_world_to_base: wp.array | np.ndarray | list[float] | None = None,
    orientation_world_to_base: wp.array | np.ndarray | list[float] | None = None,
) -> np.ndarray:
    """Convert Isaac Sim position to cuMotion position in robot base frame.

    Transforms a position vector from Isaac Sim world frame to cuMotion's robot base frame.

    Args:
        position_world_to_target: Target position in world frame [x, y, z].
        position_world_to_base: Robot base position in world frame. Defaults to None (origin).
        orientation_world_to_base: Robot base orientation in world frame as quaternion [w, x, y, z].
            Defaults to None (identity).

    Returns:
        Position vector in robot base frame as numpy array of shape (3,).

    Raises:
        ValueError: If position_world_to_target is not size 3.

    Example:

        .. code-block:: python

            position_base = isaac_sim_to_cumotion_translation(
                position_world_to_target=[1.0, 2.0, 3.0],
                position_world_to_base=[0.5, 0.5, 0.0],
                orientation_world_to_base=[1.0, 0.0, 0.0, 0.0]
            )
    """
    if position_world_to_base is not None:
        position_world_to_base = _to_numpy(position_world_to_base)
    else:
        position_world_to_base = np.zeros(
            [
                3,
            ]
        )

    if orientation_world_to_base is not None:
        orientation_world_to_base = _to_numpy(orientation_world_to_base)
    else:
        orientation_world_to_base = np.array([1.0, 0.0, 0.0, 0.0])

    transform_world_to_base = _position_quaternion_to_cumotion_pose(
        position=position_world_to_base, quaternion=orientation_world_to_base
    )

    position_world_to_target = _to_numpy(position_world_to_target)

    if position_world_to_target.size != 3:
        raise ValueError("Input position_world_to_target must be an array of size 3, equal to (x, y, z)")

    position_world_to_target = np.reshape(position_world_to_target, shape=[3, 1])

    transform_base_to_target = (
        transform_world_to_base.inverse().rotation.matrix() @ position_world_to_target
        + np.reshape(transform_world_to_base.inverse().translation, shape=[3, 1])
    )
    return transform_base_to_target.flatten()


def cumotion_to_isaac_sim_translation(
    position_base_to_target: np.ndarray,
    position_world_to_base: wp.array | np.ndarray | list[float] | None = None,
    orientation_world_to_base: wp.array | np.ndarray | list[float] | None = None,
) -> wp.array:
    """Convert cuMotion position to Isaac Sim position in world frame.

    Transforms a position vector from cuMotion's robot base frame to Isaac Sim world frame.

    Args:
        position_base_to_target: Target position in robot base frame [x, y, z].
        position_world_to_base: Robot base position in world frame. Defaults to None (origin).
        orientation_world_to_base: Robot base orientation in world frame as quaternion [w, x, y, z].
            Defaults to None (identity).

    Returns:
        Position vector in world frame as warp array of shape (3,).

    Raises:
        ValueError: If position_base_to_target is not size 3.

    Example:

        .. code-block:: python

            position_world = cumotion_to_isaac_sim_translation(
                position_base_to_target=np.array([0.5, 1.0, 0.5]),
                position_world_to_base=[0.5, 0.5, 0.0],
                orientation_world_to_base=[1.0, 0.0, 0.0, 0.0]
            )
    """

    if position_world_to_base is not None:
        position_world_to_base = _to_numpy(position_world_to_base)
    else:
        position_world_to_base = np.zeros(
            [
                3,
            ]
        )

    if orientation_world_to_base is not None:
        orientation_world_to_base = _to_numpy(orientation_world_to_base)
    else:
        orientation_world_to_base = np.array([1.0, 0.0, 0.0, 0.0])

    if position_base_to_target.size != 3:
        raise ValueError("Input position_base_to_target must be an array of size 3, equal to (x, y, z)")

    position_base_to_target = np.reshape(position_base_to_target, shape=[3, 1])
    transform_world_to_base = _position_quaternion_to_cumotion_pose(
        position=position_world_to_base, quaternion=orientation_world_to_base
    )

    position_world_to_target = transform_world_to_base.rotation.matrix() @ position_base_to_target + np.reshape(
        transform_world_to_base.translation, shape=[3, 1]
    )
    return wp.from_numpy(position_world_to_target.flatten(), dtype=wp.float32)


def isaac_sim_to_cumotion_rotation(
    orientation_world_to_target: wp.array | np.ndarray | list[float],
    orientation_world_to_base: wp.array | np.ndarray | list[float] | None = None,
) -> cumotion.Rotation3:
    """Convert Isaac Sim orientation to cuMotion rotation in robot base frame.

    Transforms an orientation from Isaac Sim world frame to cuMotion's robot base frame.

    Args:
        orientation_world_to_target: Target orientation in world frame as quaternion [w, x, y, z].
        orientation_world_to_base: Robot base orientation in world frame as quaternion [w, x, y, z].
            Defaults to None (identity).

    Returns:
        cuMotion Rotation3 object representing the orientation in robot base frame.

    Raises:
        ValueError: If orientation is not size 4.

    Example:

        .. code-block:: python

            rotation_base = isaac_sim_to_cumotion_rotation(
                orientation_world_to_target=[1.0, 0.0, 0.0, 0.0],
                orientation_world_to_base=[1.0, 0.0, 0.0, 0.0]
            )
    """

    if orientation_world_to_base is not None:
        orientation_world_to_base = _to_numpy(orientation_world_to_base)
    else:
        orientation_world_to_base = np.array([1.0, 0.0, 0.0, 0.0])

    orientation_world_to_target = _to_numpy(orientation_world_to_target)

    if orientation_world_to_target.size != 4:
        raise ValueError("Input orientation must be of size 4, equal to the quaternion (w, x, y, z).")

    orientation_world_to_target = orientation_world_to_target.flatten()
    rotation_world_to_base = cumotion.Rotation3(*orientation_world_to_base)

    # convert to the base-frame, which is the frame used in cumotion:
    return rotation_world_to_base.inverse() * cumotion.Rotation3(*orientation_world_to_target)


def cumotion_to_isaac_sim_rotation(
    orientation_base_to_target: cumotion.Rotation3,
    position_world_to_base: wp.array | np.ndarray | list[float] | None = None,
    orientation_world_to_base: wp.array | np.ndarray | list[float] | None = None,
) -> wp.array:
    """Convert cuMotion rotation to Isaac Sim orientation in world frame.

    Transforms an orientation from cuMotion's robot base frame to Isaac Sim world frame.

    Args:
        orientation_base_to_target: Target orientation in robot base frame.
        position_world_to_base: Robot base position in world frame (unused, kept for consistency).
            Defaults to None.
        orientation_world_to_base: Robot base orientation in world frame as quaternion [w, x, y, z].
            Defaults to None (identity).

    Returns:
        Orientation as warp array quaternion [w, x, y, z] of shape (4,).

    Example:

        .. code-block:: python

            quaternion = cumotion_to_isaac_sim_rotation(
                orientation_base_to_target,
                orientation_world_to_base=[1.0, 0.0, 0.0, 0.0]
            )
    """

    if orientation_world_to_base is not None:
        orientation_world_to_base = _to_numpy(orientation_world_to_base)
    else:
        orientation_world_to_base = np.array([1.0, 0.0, 0.0, 0.0])

    rotation_world_to_base = cumotion.Rotation3(*orientation_world_to_base)
    rotation_world_to_target = rotation_world_to_base * orientation_base_to_target

    # retun as a wp.array:
    return wp.array(
        [
            rotation_world_to_target.w(),
            rotation_world_to_target.x(),
            rotation_world_to_target.y(),
            rotation_world_to_target.z(),
        ],
        dtype=wp.float32,
    )
