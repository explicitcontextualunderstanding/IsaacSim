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

from dataclasses import dataclass
from typing import Literal

import cumotion
import isaacsim.robot_motion.experimental.motion_generation as mg
import numpy as np
import warp as wp
from isaacsim.core.experimental.materials import OmniPbrMaterial
from isaacsim.core.experimental.objects import Capsule, Cube, Sphere
from isaacsim.core.experimental.prims import XformPrim

from .utils import (
    ColliderBatchTransformOutput,
    batch_compute_collider_transforms,
    cumotion_to_isaac_sim_pose,
    isaac_sim_to_cumotion_pose,
)


@dataclass
class _CumotionCollider:
    """Collision geometry representation for cuMotion.

    Represents a single collision primitive in the cuMotion world. Note that a full
    parent object can be modeled by several smaller collision primitives. Each collider has its own
    transform relative to the parent object origin.

    Attributes:
        obstacle_handle: Handle to the collision obstacle in cuMotion's C++ memory.
        transform_object_to_collider: Transform from the parent object frame to this
            collider's frame. Defaults to identity.
        debug_prim_path: USD prim path for debug visualization. Defaults to None.

    Note:
        The world-to-collider transform is computed as:
        transform_world_to_collider = transform_world_to_object * transform_object_to_collider
    """

    obstacle_handle: cumotion.World.ObstacleHandle

    # Which indices are these particular transforms stored at?
    transform_object_to_collider_index: int

    # A prim path. This is only used if the user has enabled debugging visualizations.
    debug_prim_path: str | None = None


def _vectorize_cumotion_collider_data(
    cumotion_colliders: list[_CumotionCollider],
) -> tuple[list[cumotion.World.ObstacleHandle], list[str | None], int]:
    """Extract obstacle handles and debug prim paths from a list of _CumotionCollider objects.

    Args:
        cumotion_colliders: List of _CumotionCollider objects representing collision geometries.

    Returns:
        Tuple containing:
            - List of obstacle handles
            - List of debug prim paths (may contain None values)
            - Number of colliders
    """
    # The list of geometries which collectively model this obstacle:
    obstacle_handles = [c.obstacle_handle for c in cumotion_colliders]

    # The transforms from object frame --> collider geometry frame for each
    # geometry which model the obstacle.
    debug_prim_paths = [c.debug_prim_path for c in cumotion_colliders]
    n_colliders = len(cumotion_colliders)
    return obstacle_handles, debug_prim_paths, n_colliders


class CollisionData:
    """Container for collision data associated with a USD prim.

    Stores the transform and collision geometries for a single USD prim that has
    been added to the cuMotion world for collision checking.

    Args:
        transform_world_to_object: Transform from world frame to the object frame.
        cumotion_colliders: List of collision primitives representing this object.

    Attributes:
        transform_world_to_object: Current transform of the object in world space.
        cumotion_colliders: List of collision primitives for this object.
    """

    def __init__(
        self,
        transform_world_to_object_index: int,
        obstacle_handle_starting_index: int,
        n_colliders: int,
    ):
        # Store the indices of where the data for this object begins:
        self.transform_world_to_object_index = transform_world_to_object_index
        self.obstacle_handle_starting_index = obstacle_handle_starting_index
        self.n_colliders = n_colliders
        self.obstacle_handle_ending_index = obstacle_handle_starting_index + n_colliders


class CumotionWorldInterface(mg.WorldInterface):
    """World interface for cuMotion collision checking and planning.

    This class provides the bridge between Isaac Sim's USD scene representation and
    cuMotion's collision world. It manages obstacle registration, updates, and
    coordinate frame transformations between Isaac Sim (world frame) and cuMotion
    (robot base frame).

    Args:
        world_to_robot_base: Transform from world frame to robot base frame. Defaults
            to None (identity transform). This is a tuple of position and quaternion,
            where the quaternion is in the form (w, x, y, z).
        visualize_debug_prims: Whether to create visual debug primitives for collision
            geometry. Defaults to False.
        visual_debug_prim_rgb: RGB color for debug visualization. Defaults to None (red).
        visual_debug_prim_alpha: Alpha transparency for debug visualization. Defaults to 0.3.

    Attributes:
        world_view: cuMotion world view for collision queries.

    Example:

        .. code-block:: python

            world_interface = CumotionWorldInterface(
                visualize_debug_prims=True,
                visual_debug_prim_rgb=[0.0, 1.0, 0.0],
                visual_debug_prim_alpha=0.5
            )
    """

    def __init__(
        self,
        world_to_robot_base: tuple[wp.array, wp.array] | None = None,
        visualize_debug_prims: bool = False,
        visual_debug_enabled_prim_rgb: list[float] | None = None,
        visual_debug_disabled_prim_rgb: list[float] | None = None,
        visual_debug_prim_alpha: float = 0.3,
    ):

        self._world: cumotion.World = cumotion.create_world()
        self.__world_view = self._world.add_world_view()
        self.__world_inspector = cumotion.create_world_inspector(self.__world_view)
        # want to be able to manipulate our internal world objects according to changes
        # which occur to each prim:
        self._prim_path_to_collision_data: dict[str, CollisionData] = {}

        # Transform of the robot Base Frame --> World Frame. It is useful to store
        # in this way so we don't have to do as much matrix inversion. Assume identity.
        self._position_base_to_world: wp.array = wp.zeros(
            shape=[
                3,
            ],
            dtype=wp.float32,
        )
        self._quaternion_base_to_world: wp.array = wp.array([1.0, 0.0, 0.0, 0.0], dtype=wp.float32)

        if world_to_robot_base is not None:
            world_to_robot_base_cumotion = isaac_sim_to_cumotion_pose(world_to_robot_base[0], world_to_robot_base[1])
            base_to_world_cumotion = world_to_robot_base_cumotion.inverse()
            self._position_base_to_world, self._quaternion_base_to_world = cumotion_to_isaac_sim_pose(
                base_to_world_cumotion
            )

        # Storing Transforms from world frame --> object frames.
        self._position_world_to_objects: np.ndarray = np.empty((0, 3))
        self._quaternion_world_to_objects: np.ndarray = np.empty((0, 4))

        # Storing Transforms from object frames --> collision geometry frames.
        self._position_objects_to_colliders: np.ndarray = np.empty((0, 3))
        self._quaternion_objects_to_colliders: np.ndarray = np.empty((0, 4))

        # storing all of the obstacle data in pre-built arrays:
        self._all_obstacle_handles: list[cumotion.World.ObstacleHandle] = []
        self._all_debug_prim_paths: list[str] = []
        self._n_colliders_in_each_object: np.ndarray = np.empty([0], dtype=np.int32)

        # if we want to do debug visuals, set those up here:
        # Do we want to draw debug sphere for mesh files?
        self._visualize_debug_prims = visualize_debug_prims

        if not visualize_debug_prims:
            return

        # Set up the "disabled obstacle" material
        if visual_debug_disabled_prim_rgb is None:
            visual_debug_disabled_prim_rgb = [0.0, 1.0, 0.0]
        self._visual_debug_disabled_prim_material = OmniPbrMaterial(
            paths="/CumotionDebug/DisabledMaterial",
        )
        self._visual_debug_disabled_prim_material.set_input_values(
            "diffuse_color_constant", visual_debug_disabled_prim_rgb
        )
        self._visual_debug_disabled_prim_material.set_input_values("enable_opacity", [True])
        self._visual_debug_disabled_prim_material.set_input_values("opacity_constant", [visual_debug_prim_alpha])

        # Set up the "enabled obstacle" material
        if visual_debug_enabled_prim_rgb is None:
            visual_debug_enabled_prim_rgb = [1.0, 0.0, 0.0]
        self._visual_debug_enabled_prim_material = OmniPbrMaterial(
            paths="/CumotionDebug/EnabledMaterial",
        )
        self._visual_debug_enabled_prim_material.set_input_values(
            "diffuse_color_constant", visual_debug_enabled_prim_rgb
        )
        self._visual_debug_enabled_prim_material.set_input_values("enable_opacity", [True])
        self._visual_debug_enabled_prim_material.set_input_values("opacity_constant", [visual_debug_prim_alpha])

    def _set_debug_material(self, debug_visual, enabled):
        """Set the debug material for a visual prim based on enabled state.

        Args:
            debug_visual: Visual prim object to apply material to.
            enabled: Whether the obstacle is enabled (True) or disabled (False).
        """
        if enabled:
            debug_visual.apply_visual_materials(self._visual_debug_enabled_prim_material)
            return
        debug_visual.apply_visual_materials(self._visual_debug_disabled_prim_material)

    @property
    def world_view(self):
        """Get the world_view associated with the cumotion.World object"""
        return self.__world_view

    def add_spheres(
        self,
        prim_paths: list[str],
        radii: wp.array,
        scales: wp.array,
        safety_tolerances: wp.array,
        poses: tuple[wp.array, wp.array],
        enabled_array: wp.array,
    ):
        positions, quaternions = poses
        positions_np = positions.numpy()
        quaternions_np = quaternions.numpy()
        scales_np = scales.numpy()
        radii_np = radii.numpy()
        safety_tolerances_np = safety_tolerances.numpy()
        enabled_np = enabled_array.numpy()

        if np.any(radii_np <= 0.0):
            raise ValueError(f"Radius values must be positive in cuMotion.")

        # For accuracy, we need to make sure that the scales are uniform. Otherwise,
        # these are ellipsoids, and should not be included in cumotion as a sphere.
        for scale in scales_np:
            if not np.allclose(scale, scale[0]) or np.any(scale <= 0.0):
                raise ValueError(f"Scale on sphere objects must be uniform and positive in cuMotion.")

        if not (
            len(prim_paths)
            == len(radii_np)
            == len(positions_np)
            == len(quaternions_np)
            == len(scales_np)
            == len(safety_tolerances_np)
            == len(enabled_np)
        ):
            raise ValueError(f"All input arrays must have the same length in cuMotion.")

        if len(prim_paths) == 0:
            raise ValueError(f"No prim paths provided to add_sphere call.")

        prim_set = set(prim_paths)
        if not prim_set.isdisjoint(set(self._prim_path_to_collision_data.keys())):
            raise ValueError(f"Tried to add a sphere with a prim path that already exists in the world.")

        if len(prim_paths) != len(prim_set):
            raise ValueError(f"Attempted to add duplicate prim paths in cuMotion.")

        for i in range(len(prim_paths)):
            prim_path = prim_paths[i]
            radius = radii_np[i]
            position = positions_np[i]
            quaternion = quaternions_np[i]
            scale = scales_np[i]
            safety_tolerance = safety_tolerances_np[i]
            enabled = enabled_np[i]

            # Makes sure that the entire sphere is always covered.
            radius = float(np.max(scale) * radius.item() + safety_tolerance.item())

            # use a sphere object:
            obstacle: cumotion.Obstacle = cumotion.create_obstacle(cumotion.Obstacle.Type.SPHERE)
            obstacle.set_attribute(cumotion.Obstacle.Attribute.RADIUS, radius)

            debug_prim_path = None
            if self._visualize_debug_prims:
                debug_prim_path = self._debug_collision_prim_name_generate(original_prim_name=prim_path, i_geometry=0)
                sphere = Sphere(paths=debug_prim_path, radii=radius)
                self._set_debug_material(sphere, enabled.item())

            transform_world_to_object_index = self._extend_world_to_objects_matrix(position, quaternion)

            # Set the obstacle in the world:
            obstacle_handle = self._world.add_obstacle(obstacle)
            collision_data = self._extend_collider_data(
                transform_world_to_object_index=transform_world_to_object_index,
                cumotion_colliders=[
                    _CumotionCollider(
                        obstacle_handle=obstacle_handle,
                        debug_prim_path=debug_prim_path,
                        transform_object_to_collider_index=self._extend_objects_to_colliders_matrix(
                            position=np.array([0.0, 0.0, 0.0]), quaternion=np.array([1.0, 0.0, 0.0, 0.0])
                        ),
                    )
                ],
            )
            self._prim_path_to_collision_data[prim_path] = collision_data

            # cuMotion will throw an error if you try to enable a collider
            # which is already enabled.
            if not enabled.item():
                self._update_prim_enabled_value(prim_path=prim_path, enabled=False)

        self._update_prim_world_to_object_transforms(
            prim_paths=prim_paths,
            poses=(positions, quaternions),
        )

    def add_cubes(
        self,
        prim_paths: list[str],
        sizes: wp.array,
        scales: wp.array,
        safety_tolerances: wp.array,
        poses: tuple[wp.array, wp.array],
        enabled_array: wp.array,
    ):
        positions, quaternions = poses
        positions_np = positions.numpy()
        quaternions_np = quaternions.numpy()
        scales_np = scales.numpy()
        sizes_np = sizes.numpy()
        safety_tolerances_np = safety_tolerances.numpy()
        enabled_np = enabled_array.numpy()

        if np.any(sizes_np < 0.0):
            raise ValueError(f"Size values must be non-negative in cuMotion.")

        if np.any(scales_np <= 0.0):
            raise ValueError(f"Scale values for cube must be positive in cuMotion.")

        if not (
            len(prim_paths)
            == len(sizes_np)
            == len(positions_np)
            == len(quaternions_np)
            == len(scales_np)
            == len(safety_tolerances_np)
            == len(enabled_np)
        ):
            raise ValueError(f"All input arrays must have the same length in cuMotion.")

        if len(prim_paths) == 0:
            raise ValueError(f"No prim paths provided to add_cube call.")

        prim_set = set(prim_paths)
        if not prim_set.isdisjoint(set(self._prim_path_to_collision_data.keys())):
            raise ValueError(f"Tried to add a cube with a prim path that already exists in the world.")

        if len(prim_paths) != len(prim_set):
            raise ValueError(f"Attempted to add duplicate prim paths in cuMotion.")

        for i in range(len(prim_paths)):
            prim_path = prim_paths[i]
            size = sizes_np[i]
            position = positions_np[i]
            quaternion = quaternions_np[i]
            scale = scales_np[i]
            safety_tolerance = safety_tolerances_np[i]
            enabled = enabled_np[i]

            transform_world_to_object_index = self._extend_world_to_objects_matrix(position, quaternion)

            obstacle: cumotion.Obstacle = cumotion.create_obstacle(cumotion.Obstacle.Type.CUBOID)
            side_lengths = np.array(
                [
                    float(scale[0] * size.item() + safety_tolerance.item()),
                    float(scale[1] * size.item() + safety_tolerance.item()),
                    float(scale[2] * size.item() + safety_tolerance.item()),
                ]
            )
            obstacle.set_attribute(
                cumotion.Obstacle.Attribute.SIDE_LENGTHS,
                side_lengths,
            )

            debug_prim_path = None
            if self._visualize_debug_prims:
                # create the debug visual prim:
                debug_prim_path = self._debug_collision_prim_name_generate(
                    original_prim_name=prim_path,
                    i_geometry=0,
                )
                visual_cube = Cube(
                    paths=debug_prim_path,
                    sizes=1.0,
                    scales=side_lengths,
                )
                self._set_debug_material(visual_cube, enabled.item())

            # Set the obstacle in the world:
            obstacle_handle = self._world.add_obstacle(obstacle)
            collision_data = self._extend_collider_data(
                transform_world_to_object_index=transform_world_to_object_index,
                cumotion_colliders=[
                    _CumotionCollider(
                        obstacle_handle=obstacle_handle,
                        debug_prim_path=debug_prim_path,
                        transform_object_to_collider_index=self._extend_objects_to_colliders_matrix(
                            position=np.array([0.0, 0.0, 0.0]), quaternion=np.array([1.0, 0.0, 0.0, 0.0])
                        ),
                    )
                ],
            )
            self._prim_path_to_collision_data[prim_path] = collision_data

            # cuMotion will throw an error if you try to enable a collider
            # which is already enabled.
            if not enabled.item():
                self._update_prim_enabled_value(prim_path=prim_path, enabled=False)

        self._update_prim_world_to_object_transforms(
            prim_paths=prim_paths,
            poses=(positions, quaternions),
        )

    def add_triangulated_meshes(
        self,
        prim_paths: list[str],
        points: list[wp.array],
        face_vertex_indices: list[wp.array],
        scales: wp.array,
        safety_tolerances: wp.array,
        poses: tuple[wp.array, wp.array],
        enabled_array: wp.array,
    ):
        def _to_cumotion_sphere_collider(prim_name, i_sphere, sphere, enabled):
            """Convert a collision sphere to a _CumotionCollider object.

            Args:
                prim_name: Name of the prim this collider belongs to.
                i_sphere: Index of this sphere within the mesh decomposition.
                sphere: Collision sphere object with radius and center attributes.
                enabled: Whether the collider should be enabled initially.

            Returns:
                _CumotionCollider object representing the sphere collider.
            """
            # use a sphere object:
            obstacle: cumotion.Obstacle = cumotion.create_obstacle(cumotion.Obstacle.Type.SPHERE)
            obstacle.set_attribute(cumotion.Obstacle.Attribute.RADIUS, sphere.radius)

            # Set the obstacle in the world:
            obstacle_handle = self._world.add_obstacle(obstacle)

            # Store the object-to-collider transform (sphere center offset)
            transform_object_to_collider_index = self._extend_objects_to_colliders_matrix(
                position=sphere.center, quaternion=np.array([1.0, 0.0, 0.0, 0.0])
            )

            # For debugging - we can optionally draw collision spheres:
            debug_prim_path = None
            if self._visualize_debug_prims:
                # Get a unique prim name for this sphere:
                debug_prim_path = self._debug_collision_prim_name_generate(
                    original_prim_name=prim_name,
                    i_geometry=i_sphere,
                )

                # Create the sphere:
                sphere_core_object = Sphere(paths=debug_prim_path)
                sphere_core_object.set_radii(sphere.radius, indices=0)
                self._set_debug_material(sphere_core_object, enabled)

            return _CumotionCollider(
                obstacle_handle=obstacle_handle,
                transform_object_to_collider_index=transform_object_to_collider_index,
                debug_prim_path=debug_prim_path,
            )

        positions, quaternions = poses
        positions_np = positions.numpy()
        quaternions_np = quaternions.numpy()
        enabled_array_np = enabled_array.numpy()
        safety_tolerances_np = safety_tolerances.numpy()
        scales_np = scales.numpy()

        if not (
            len(prim_paths)
            == len(points)
            == len(face_vertex_indices)
            == len(positions_np)
            == len(quaternions_np)
            == len(scales)
            == len(safety_tolerances_np)
            == len(enabled_array_np)
        ):
            raise ValueError(f"All input arrays must have the same length in cuMotion.")

        if len(prim_paths) == 0:
            raise ValueError(f"No prim paths provided to add_triangulated_meshes call.")

        prim_set = set(prim_paths)
        if not prim_set.isdisjoint(set(self._prim_path_to_collision_data.keys())):
            raise ValueError(f"Tried to add a triangulated mesh with a prim path that already exists in the world.")

        if len(prim_paths) != len(prim_set):
            raise ValueError(f"Attempted to add duplicate prim paths in cuMotion.")

        for i in range(len(prim_paths)):
            prim_path = prim_paths[i]
            points_array = points[i]
            face_vertex_indices_array = face_vertex_indices[i]
            position = positions_np[i]
            quaternion = quaternions_np[i]
            scale = scales_np[i]
            safety_tolerance = safety_tolerances_np[i]
            enabled = enabled_array_np[i]

            if points_array.shape[1] != 3:
                raise ValueError(
                    f"Points must have 3 elements (x, y, z) in add_triangulated_meshes call. Got {points_array.shape[1]} elements."
                )

            if face_vertex_indices_array.shape[1] != 3:
                raise ValueError(
                    f"Face vertex indices must have 3 elements (i1, i2, i3) in add_triangulated_meshes call. Got {face_vertex_indices_array.shape[1]} elements."
                )

            # Apply scaling to the input points:
            points_scaled = (np.diag(scale) @ points_array.numpy().T).T

            transform_world_to_object_index = self._extend_world_to_objects_matrix(position, quaternion)
            collision_spheres = cumotion.generate_collision_spheres(
                vertices=points_scaled,
                triangles=face_vertex_indices_array.numpy(),
                max_overshoot=safety_tolerance.item(),
            )

            collision_data = self._extend_collider_data(
                transform_world_to_object_index=transform_world_to_object_index,
                cumotion_colliders=[
                    _to_cumotion_sphere_collider(prim_path, i_sphere, sphere, enabled.item())
                    for i_sphere, sphere in enumerate(collision_spheres)
                ],
            )

            self._prim_path_to_collision_data[prim_path] = collision_data

            # cuMotion will throw an error if you try to enable a collider
            # which is already enabled.
            if not enabled.item():
                self._update_prim_enabled_value(prim_path=prim_path, enabled=False)

        self._update_prim_world_to_object_transforms(
            prim_paths=prim_paths,
            poses=(positions, quaternions),
        )

    def add_planes(
        self,
        prim_paths: list[str],
        axes: list[Literal["X", "Y", "Z"]],
        lengths: wp.array,
        widths: wp.array,
        scales: wp.array,
        safety_tolerances: wp.array,
        poses: tuple[wp.array, wp.array],
        enabled_array: wp.array,
    ):
        positions, quaternions = poses
        positions_np = positions.numpy()
        quaternions_np = quaternions.numpy()
        enabled_array_np = enabled_array.numpy()
        safety_tolerances_np = safety_tolerances.numpy()
        scales_np = scales.numpy()
        if not (
            len(prim_paths)
            == len(axes)
            == len(positions_np)
            == len(quaternions_np)
            == len(scales_np)
            == len(safety_tolerances_np)
            == len(enabled_array_np)
        ):
            raise ValueError(f"All input arrays must have the same length in cuMotion.")

        if len(prim_paths) == 0:
            raise ValueError(f"No prim paths provided to add_planes call.")

        prim_set = set(prim_paths)
        if not prim_set.isdisjoint(set(self._prim_path_to_collision_data.keys())):
            raise ValueError(f"Tried to add a plane with a prim path that already exists in the world.")

        if len(prim_paths) != len(prim_set):
            raise ValueError(f"Attempted to add duplicate prim paths in cuMotion.")

        for i in range(len(prim_paths)):
            prim_path = prim_paths[i]
            axis = axes[i]
            position = positions_np[i]
            quaternion = quaternions_np[i]
            safety_tolerance = safety_tolerances_np[i]
            enabled = enabled_array_np[i]

            transform_world_to_object_index = self._extend_world_to_objects_matrix(position, quaternion)

            # This collider will be a cuboid type in cumotion:
            obstacle: cumotion.Obstacle = cumotion.create_obstacle(cumotion.Obstacle.Type.CUBOID)

            # TODO: a bit arbitrary?
            LARGE_NUMBER = 50_000.0
            SMALL_NUMBER = 1e-6 + safety_tolerance.item()

            side_lengths = np.array([LARGE_NUMBER, LARGE_NUMBER, SMALL_NUMBER])

            obstacle.set_attribute(
                cumotion.Obstacle.Attribute.SIDE_LENGTHS,
                side_lengths,
            )

            # Determine the rotation needed to align the capsule with the specified axis
            # cuMotion capsules are aligned along the Z-axis by default
            # If axis is "Z", use identity transform
            # If axis is "X", rotate 90 degrees about Y-axis
            # If axis is "Y", rotate 90 degrees about X-axis
            if axis == "Z":
                collider_position = np.array([0.0, 0.0, 0.0])
                collider_quaternion = np.array([1.0, 0.0, 0.0, 0.0])
            elif axis == "X":
                # 90 degree rotation about Y-axis: [cos(90/2), 0, sin(90/2), 0] (w, x, y, z)
                collider_position = np.array([0.0, 0.0, 0.0])
                collider_quaternion = np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0])
            elif axis == "Y":
                # 90 degree rotation about X-axis: [cos(90/2), sin(90/2), 0, 0] (w, x, y, z)
                collider_position = np.array([0.0, 0.0, 0.0])
                collider_quaternion = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])
            else:
                raise ValueError(f"Invalid axis: {axis}. Expected 'X', 'Y', or 'Z'.")

            debug_prim_path = None
            if self._visualize_debug_prims:
                debug_prim_path = self._debug_collision_prim_name_generate(original_prim_name=prim_path, i_geometry=0)
                visual_box = Cube(paths=debug_prim_path, sizes=1.0, scales=side_lengths)
                self._set_debug_material(visual_box, enabled.item())

            obstacle_handle = self._world.add_obstacle(obstacle)

            collision_data = self._extend_collider_data(
                transform_world_to_object_index=transform_world_to_object_index,
                cumotion_colliders=[
                    _CumotionCollider(
                        obstacle_handle=obstacle_handle,
                        transform_object_to_collider_index=self._extend_objects_to_colliders_matrix(
                            position=collider_position, quaternion=collider_quaternion
                        ),
                        debug_prim_path=debug_prim_path,
                    )
                ],
            )

            self._prim_path_to_collision_data[prim_path] = collision_data

            # cuMotion will throw an error if you try to enable a collider
            # which is already enabled.
            if not enabled.item():
                self._update_prim_enabled_value(
                    prim_path=prim_path,
                    enabled=False,
                )

        self._update_prim_world_to_object_transforms(
            prim_paths=prim_paths,
            poses=(positions, quaternions),
        )

    def add_capsules(
        self,
        prim_paths: list[str],
        axes: list[Literal["X", "Y", "Z"]],
        radii: wp.array,
        lengths: wp.array,
        scales: wp.array,
        safety_tolerances: wp.array,
        poses: tuple[wp.array, wp.array],
        enabled_array: wp.array,
    ):
        positions, quaternions = poses
        positions_np = positions.numpy()
        quaternions_np = quaternions.numpy()
        enabled_array_np = enabled_array.numpy()
        safety_tolerances_np = safety_tolerances.numpy()
        scales_np = scales.numpy()
        radii_np = radii.numpy()
        lengths_np = lengths.numpy()

        # For accuracy, we need to make sure that the scales are uniform. Otherwise,
        # these would be ellipsoid capsules, which are not supported in cuMotion.
        for scale in scales_np:
            if not np.allclose(scale, scale[0]) or np.any(scale <= 0.0):
                raise ValueError(f"Scale on capsule objects must be uniform and positive in cuMotion.")

        if np.any(lengths_np <= 0.0):
            raise ValueError(f"Length values must be positive in cuMotion.")

        if np.any(radii_np <= 0.0):
            raise ValueError(f"Radius values must be positive in cuMotion.")

        if not (
            len(prim_paths)
            == len(axes)
            == len(radii_np)
            == len(lengths_np)
            == len(positions_np)
            == len(quaternions_np)
            == len(scales_np)
            == len(safety_tolerances_np)
            == len(enabled_array_np)
        ):
            raise ValueError(f"All input arrays must have the same length in cuMotion.")

        if len(prim_paths) == 0:
            raise ValueError(f"No prim paths provided to add_capsules call.")

        prim_set = set(prim_paths)
        if not prim_set.isdisjoint(set(self._prim_path_to_collision_data.keys())):
            raise ValueError(f"Tried to add a capsule with a prim path that already exists in the world.")

        if len(prim_paths) != len(prim_set):
            raise ValueError(f"Attempted to add duplicate prim paths in cuMotion.")

        for i in range(len(prim_paths)):
            prim_path = prim_paths[i]
            axis = axes[i]
            radius = radii_np[i]
            length = lengths_np[i]
            position = positions_np[i]
            quaternion = quaternions_np[i]
            scale = scales_np[i]
            safety_tolerance = safety_tolerances_np[i]
            enabled = enabled_array_np[i]

            transform_world_to_object_index = self._extend_world_to_objects_matrix(position, quaternion)

            # This collider will be a cylinder-type in cumotion (which is actually a capsule):
            obstacle: cumotion.Obstacle = cumotion.create_obstacle(cumotion.Obstacle.Type.CAPSULE)

            # If the scaling isn't uniform, we cannot treat this as a capsule.
            if not np.allclose(scale, scale[0]):
                raise ValueError(f"Scale on capsule objects must be uniform in cuMotion.")

            # Apply scale and safety tolerance to radius and height
            scaled_radius = float(scale[0] * radius.item() + safety_tolerance.item())
            scaled_height = float(scale[0] * length.item() + safety_tolerance.item())

            obstacle.set_attribute(cumotion.Obstacle.Attribute.RADIUS, scaled_radius)
            obstacle.set_attribute(cumotion.Obstacle.Attribute.HEIGHT, scaled_height)

            # Determine the rotation needed to align the capsule with the specified axis
            # cuMotion capsules are aligned along the Z-axis by default
            # If axis is "Z", use identity transform
            # If axis is "X", rotate 90 degrees about Y-axis
            # If axis is "Y", rotate 90 degrees about X-axis
            if axis == "Z":
                collider_position = np.array([0.0, 0.0, 0.0])
                collider_quaternion = np.array([1.0, 0.0, 0.0, 0.0])
            elif axis == "X":
                # 90 degree rotation about Y-axis: [cos(90/2), 0, sin(90/2), 0] (w, x, y, z)
                collider_position = np.array([0.0, 0.0, 0.0])
                collider_quaternion = np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0])
            elif axis == "Y":
                # 90 degree rotation about X-axis: [cos(90/2), sin(90/2), 0, 0] (w, x, y, z)
                collider_position = np.array([0.0, 0.0, 0.0])
                collider_quaternion = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])
            else:
                raise ValueError(f"Invalid axis: {axis}. Expected 'X', 'Y', or 'Z'.")

            debug_prim_path = None
            if self._visualize_debug_prims:
                debug_prim_path = self._debug_collision_prim_name_generate(original_prim_name=prim_path, i_geometry=0)
                visual_capsule = Capsule(
                    paths=debug_prim_path,
                    radii=scaled_radius,
                    heights=scaled_height,
                )
                self._set_debug_material(visual_capsule, enabled.item())

            obstacle_handle = self._world.add_obstacle(obstacle)

            collision_data = self._extend_collider_data(
                transform_world_to_object_index=transform_world_to_object_index,
                cumotion_colliders=[
                    _CumotionCollider(
                        obstacle_handle=obstacle_handle,
                        transform_object_to_collider_index=self._extend_objects_to_colliders_matrix(
                            position=collider_position, quaternion=collider_quaternion
                        ),
                        debug_prim_path=debug_prim_path,
                    )
                ],
            )

            self._prim_path_to_collision_data[prim_path] = collision_data

            # cuMotion will throw an error if you try to enable a collider
            # which is already enabled.
            if not enabled.item():
                self._update_prim_enabled_value(
                    prim_path=prim_path,
                    enabled=False,
                )

        self._update_prim_world_to_object_transforms(
            prim_paths=prim_paths,
            poses=(positions, quaternions),
        )

    def add_oriented_bounding_boxes(
        self,
        prim_paths: list[str],
        centers: wp.array,
        rotations: wp.array,
        half_side_lengths: wp.array,
        scales: wp.array,
        safety_tolerances: wp.array,
        poses: tuple[wp.array, wp.array],
        enabled_array: wp.array,
    ):
        positions, quaternions = poses
        positions_np = positions.numpy()
        quaternions_np = quaternions.numpy()
        enabled_array_np = enabled_array.numpy()
        safety_tolerances_np = safety_tolerances.numpy()
        scales_np = scales.numpy()

        centers_np = centers.numpy()
        rotations_np = rotations.numpy()
        half_side_lengths_np = half_side_lengths.numpy()

        for scale in scales_np:
            if not np.allclose(scale, scale[0]) or np.any(scale <= 0.0):
                raise ValueError(f"Scale on oriented bounding boxes must be uniform and positive in cuMotion.")

        if np.any(half_side_lengths_np <= 0.0):
            raise ValueError(f"Half side length values must be positive in cuMotion.")

        if not (
            len(prim_paths)
            == positions_np.shape[0]
            == quaternions_np.shape[0]
            == centers_np.shape[0]
            == rotations_np.shape[0]
            == half_side_lengths_np.shape[0]
            == scales_np.shape[0]
            == safety_tolerances_np.shape[0]
            == enabled_array_np.shape[0]
        ):
            raise ValueError(f"All input arrays must have the same length in cuMotion.")

        if len(prim_paths) == 0:
            raise ValueError(f"No prim paths provided to add_oriented_bounding_boxes call.")

        prim_set = set(prim_paths)
        if not prim_set.isdisjoint(set(self._prim_path_to_collision_data.keys())):
            raise ValueError(
                f"Tried to add an oriented bounding box with a prim path that already exists in the world."
            )

        if len(prim_paths) != len(prim_set):
            raise ValueError(f"Attempted to add duplicate prim paths in cuMotion.")

        for i in range(len(prim_paths)):
            prim_path = prim_paths[i]
            position = positions_np[i]
            quaternion = quaternions_np[i]
            center = centers_np[i]
            rotation_quaternion = rotations_np[i]
            half_side_length = half_side_lengths_np[i]
            scale = scales_np[i]
            safety_tolerance = safety_tolerances_np[i]
            enabled = enabled_array_np[i]

            transform_world_to_object_index = self._extend_world_to_objects_matrix(position, quaternion)

            # This collider will be a cuboid-type in cumotion:
            obstacle: cumotion.Obstacle = cumotion.create_obstacle(cumotion.Obstacle.Type.CUBOID)
            side_lengths = np.array(
                [
                    float(scale[0] * 2 * half_side_length[0] + safety_tolerance.item()),
                    float(scale[1] * 2 * half_side_length[1] + safety_tolerance.item()),
                    float(scale[2] * 2 * half_side_length[2] + safety_tolerance.item()),
                ]
            )
            obstacle.set_attribute(cumotion.Obstacle.Attribute.SIDE_LENGTHS, side_lengths)

            # Store the constant offset between the OBB-frame and the object-frame:
            # The rotation is already a quaternion (w, x, y, z)
            collider_translation = scale * center

            debug_prim_path = None
            if self._visualize_debug_prims:
                debug_prim_path = self._debug_collision_prim_name_generate(original_prim_name=prim_path, i_geometry=0)
                visual_cube = Cube(
                    paths=debug_prim_path,
                    sizes=1.0,
                    scales=side_lengths,
                )
                self._set_debug_material(visual_cube, enabled.item())

            obstacle_handle = self._world.add_obstacle(obstacle)

            collision_data = self._extend_collider_data(
                transform_world_to_object_index=transform_world_to_object_index,
                cumotion_colliders=[
                    _CumotionCollider(
                        obstacle_handle=obstacle_handle,
                        transform_object_to_collider_index=self._extend_objects_to_colliders_matrix(
                            position=collider_translation, quaternion=rotation_quaternion
                        ),
                        debug_prim_path=debug_prim_path,
                    )
                ],
            )

            self._prim_path_to_collision_data[prim_path] = collision_data

            # cuMotion will throw an error if you try to enable a collider
            # which is already enabled.
            if not enabled.item():
                self._update_prim_enabled_value(
                    prim_path=prim_path,
                    enabled=False,
                )

        self._update_prim_world_to_object_transforms(
            prim_paths=prim_paths,
            poses=(positions, quaternions),
        )

    def update_obstacle_transforms(self, prim_paths: list[str], poses: tuple[wp.array, wp.array]):
        """Update the world-to-object transforms for tracked obstacles.

        Updates the poses of obstacles that have already been added to the world.

        Args:
            prim_paths: List of prim paths identifying the obstacles to update.
            poses: Tuple of (positions, quaternions) where positions has shape (N, 3)
                and quaternions has shape (N, 4) in (w, x, y, z) format.

        Raises:
            RuntimeError: If any prim paths are not currently tracked.
            ValueError: If input arrays have mismatched lengths or are empty.
        """
        if not self._validate_prim_paths(prim_paths):
            raise RuntimeError("Not all prims being updated are currently tracked by the cumotion world.")

        positions, quaternions = poses
        if not (len(prim_paths) == positions.shape[0] == quaternions.shape[0]):
            raise ValueError(f"All input arrays must have the same length in cuMotion.")

        if len(prim_paths) == 0:
            raise ValueError(f"No prim paths provided to update_obstacle_transforms call.")

        self._update_prim_world_to_object_transforms(prim_paths, poses)

    def update_obstacle_enables(self, prim_paths: list[str], enabled_array: wp.array):
        """Update the enabled/disabled state of tracked obstacles.

        Enables or disables collision checking for obstacles that have already been
        added to the world. Disabled obstacles are ignored during collision checking.

        Args:
            prim_paths: List of prim paths identifying the obstacles to update.
            enabled_array: Warp array of boolean values indicating enabled state,
                shape (N,) where N is the number of prim paths.

        Raises:
            RuntimeError: If any prim paths are not currently tracked.
            ValueError: If input arrays have mismatched lengths or are empty.
        """
        if not self._validate_prim_paths(prim_paths):
            raise RuntimeError("Not all prims being updated are currently tracked by the cumotion world.")

        if not (len(prim_paths) == enabled_array.shape[0]):
            raise ValueError(f"All input arrays must have the same length in cuMotion.")

        if len(prim_paths) == 0:
            raise ValueError(f"No prim paths provided to update_obstacle_enables call.")

        for i in range(len(prim_paths)):
            prim_path = prim_paths[i]
            enabled = enabled_array.numpy()[i].item()
            self._update_prim_enabled_value(prim_path, enabled)

    def get_world_to_robot_base_transform(self) -> tuple[wp.array, wp.array]:
        """Get the transform from world frame to robot base frame.

        Returns the current transform that maps coordinates from the Isaac Sim world
        frame to the cuMotion robot base frame.

        Returns:
            Tuple of (position, quaternion) as warp arrays where position has shape (3,)
            and quaternion has shape (4,) in (w, x, y, z) format.
        """
        # Invert the stored pose:
        transform_base_to_world = isaac_sim_to_cumotion_pose(
            self._position_base_to_world, self._quaternion_base_to_world
        )
        return cumotion_to_isaac_sim_pose(transform_base_to_world.inverse())

    def update_world_to_robot_root_transforms(self, poses: tuple[wp.array, wp.array]):
        """Update the transform from world frame to robot base frame.

        cuMotion plans relative to the robot base frame. This method updates the base
        frame pose and recomputes all collider transforms in the base frame.

        Args:
            poses: Tuple of (positions, quaternions) where positions has shape (1, 3)
                and quaternions has shape (1, 4) in (w, x, y, z) format.

        Raises:
            ValueError: If the number of transforms is not exactly 1.
            ValueError: If positions don't have 3 elements.
            ValueError: If quaternions don't have 4 elements.
        """

        positions, quaternions = poses
        if not (positions.shape[0] == quaternions.shape[0] == 1):
            raise ValueError("cuMotion only works with a single robot.")

        if positions.shape[1] != 3:
            raise ValueError(
                f"Positions must have 3 elements (x, y, z) in update_world_to_robot_root_transforms call. Got {positions.shape[1]} elements."
            )

        if quaternions.shape[1] != 4:
            raise ValueError(
                f"Quaternions must have 4 elements (w, x, y, z) in update_world_to_robot_root_transforms call. Got {quaternions.shape[1]} elements."
            )

        self._update_world_to_robot_root_transforms(poses)

    def _update_prim_world_to_object_transforms(
        self,
        prim_paths: list[str],
        poses: tuple[wp.array, wp.array],
    ):
        """Update world-to-object transforms and recompute all collider poses.

        This is an internal method that updates the stored transforms and propagates
        the changes to all colliders. It recomputes collider poses in both base frame
        (for cuMotion collision checking) and world frame (for debug visualization if enabled).
        Input validation should be performed before calling.

        Args:
            prim_paths: List of prim paths identifying the obstacles to update.
            poses: Tuple of (positions, quaternions) where positions has shape (N, 3)
                and quaternions has shape (N, 4) in (w, x, y, z) format.
        """
        # SAFETY: the input data is validated before calling this function.
        collision_data_array = [self._prim_path_to_collision_data[p] for p in prim_paths]

        # Store the new values of the positions and quaternions:
        positions, quaternions = poses

        positions_np = positions.numpy()
        quaternions_np = quaternions.numpy()

        # Use list comprehensions and preallocated arrays - MUCH faster than np.append/concatenate
        num_objects = len(collision_data_array)

        # Preallocate arrays with exact size needed
        transform_world_to_object_indices = np.zeros(num_objects, dtype=np.int32)
        n_colliders_per_object = np.zeros(num_objects, dtype=np.int32)

        # Fill arrays using vectorized operations
        for i, collision_data in enumerate(collision_data_array):
            transform_world_to_object_indices[i] = collision_data.transform_world_to_object_index
            n_colliders_per_object[i] = collision_data.n_colliders

        # Build obstacle handles list using slicing from the global array
        obstacle_handles = []
        if self._visualize_debug_prims:
            debug_prim_names = []

        for collision_data in collision_data_array:
            start_idx = collision_data.obstacle_handle_starting_index
            end_idx = collision_data.obstacle_handle_ending_index
            obstacle_handles.extend(self._all_obstacle_handles[start_idx:end_idx])
            if self._visualize_debug_prims:
                debug_prim_names.extend(self._all_debug_prim_paths[start_idx:end_idx])

        # Set the new positions and quaternions into our state:
        self._position_world_to_objects[transform_world_to_object_indices, :] = positions_np
        self._quaternion_world_to_objects[transform_world_to_object_indices, :] = quaternions_np

        # Build collider transform arrays by directly slicing from the global arrays
        total_colliders = int(np.sum(n_colliders_per_object))
        position_objects_to_colliders = np.empty((total_colliders, 3), dtype=np.float32)
        quaternion_objects_to_colliders = np.empty((total_colliders, 4), dtype=np.float32)

        collider_offset = 0
        for collision_data in collision_data_array:
            start_idx = collision_data.obstacle_handle_starting_index
            end_idx = collision_data.obstacle_handle_ending_index
            n = collision_data.n_colliders

            # Copy the transform data for this object's colliders
            position_objects_to_colliders[collider_offset : collider_offset + n, :] = (
                self._position_objects_to_colliders[start_idx:end_idx, :]
            )
            quaternion_objects_to_colliders[collider_offset : collider_offset + n, :] = (
                self._quaternion_objects_to_colliders[start_idx:end_idx, :]
            )

            collider_offset += n

        # batch update all of the transforms (world --> collider frames):
        output_batch: ColliderBatchTransformOutput = batch_compute_collider_transforms(
            position_base_to_world=self._position_base_to_world,
            quaternion_base_to_world=self._quaternion_base_to_world,
            positions_world_to_object=wp.from_numpy(positions_np, dtype=wp.float32),
            quaternions_world_to_object=wp.from_numpy(quaternions_np, dtype=wp.float32),
            positions_object_to_collider=wp.from_numpy(position_objects_to_colliders, dtype=wp.float32),
            quaternions_object_to_collider=wp.from_numpy(quaternion_objects_to_colliders, dtype=wp.float32),
            num_colliders_per_object=n_colliders_per_object,
        )

        positions_base_to_colliders_np = output_batch.positions_base_to_collider.numpy()
        quaternions_base_to_colliders_np = output_batch.quaternions_base_to_collider.numpy()

        # Write these results to the cumotion colliders:
        total_colliders = positions_base_to_colliders_np.shape[0]
        for obstacle_handle, transform_index in zip(obstacle_handles, range(total_colliders)):

            pose = cumotion.Pose3(
                rotation=cumotion.Rotation3(*quaternions_base_to_colliders_np[transform_index, :]),
                translation=positions_base_to_colliders_np[transform_index, :],
            )

            # set the base-frame pose:
            self._world.set_pose(obstacle_handle, pose)

        if self._visualize_debug_prims:
            # write the input world poses to the debug prims:
            xform_prim = XformPrim(
                paths=debug_prim_names,
            )

            # note: we are not supposed to write directly to world poses.
            xform_prim.set_local_poses(
                translations=output_batch.positions_world_to_collider,
                orientations=output_batch.quaternions_world_to_collider,
            )

    def _update_world_to_robot_root_transforms(
        self,
        pose: tuple[wp.array, wp.array],
    ):
        """Update the robot base frame transform and recompute all collider poses.

        This is an internal method that updates the base frame transform and propagates
        the changes to all colliders. Input validation should be performed before calling.

        Args:
            pose: Tuple of (position, quaternion) where position has shape (1, 3)
                and quaternion has shape (1, 4) in (w, x, y, z) format.
        """
        # SAFETY: the input data is validated before calling this function.

        # Store the new values of the positions and quaternions:
        position, quaternion = pose
        pose_world_to_base_cumotion = isaac_sim_to_cumotion_pose(position, quaternion)
        self._position_base_to_world, self._quaternion_base_to_world = cumotion_to_isaac_sim_pose(
            pose_world_to_base_cumotion.inverse()
        )

        # batch update all of the transforms (world --> collider frames):
        output_batch: ColliderBatchTransformOutput = batch_compute_collider_transforms(
            position_base_to_world=self._position_base_to_world,
            quaternion_base_to_world=self._quaternion_base_to_world,
            positions_world_to_object=wp.from_numpy(self._position_world_to_objects, dtype=wp.float32),
            quaternions_world_to_object=wp.from_numpy(self._quaternion_world_to_objects, dtype=wp.float32),
            positions_object_to_collider=wp.from_numpy(self._position_objects_to_colliders, dtype=wp.float32),
            quaternions_object_to_collider=wp.from_numpy(self._quaternion_objects_to_colliders, dtype=wp.float32),
            num_colliders_per_object=self._n_colliders_in_each_object,
        )

        positions_base_to_colliders_np = output_batch.positions_base_to_collider.numpy()
        quaternions_base_to_colliders_np = output_batch.quaternions_base_to_collider.numpy()

        # Write these results to the cumotion colliders:
        for obstacle_handle, transform_index in zip(
            self._all_obstacle_handles, range(positions_base_to_colliders_np.shape[0])
        ):

            pose = cumotion.Pose3(
                rotation=cumotion.Rotation3(*quaternions_base_to_colliders_np[transform_index, :]),
                translation=positions_base_to_colliders_np[transform_index, :],
            )

            # set the base-frame pose:
            self._world.set_pose(obstacle_handle, pose)

    def _update_prim_enabled_value(self, prim_path: str, enabled: bool):
        """Update the enabled state for all colliders associated with a prim.

        Args:
            prim_path: Prim path identifying the obstacle.
            enabled: Whether to enable (True) or disable (False) the obstacle.
        """
        collision_data = self._prim_path_to_collision_data[prim_path]
        set_enable_function = self._world.enable_obstacle if enabled else self._world.disable_obstacle
        startidx = collision_data.obstacle_handle_starting_index
        endidx = collision_data.obstacle_handle_ending_index
        obstacle_handles = self._all_obstacle_handles[startidx:endidx]

        if self._visualize_debug_prims:
            debug_paths = self._all_debug_prim_paths[startidx:endidx]

        self.__world_view.update()

        for i in range(len(obstacle_handles)):
            obstacle_handle = obstacle_handles[i]
            if self.__world_inspector.is_enabled(obstacle_handle) == enabled:
                # nothing to do, we are already in this enabled state:
                continue
            set_enable_function(obstacle_handle)

            if self._visualize_debug_prims:
                self._set_debug_material(XformPrim(debug_paths[i]), enabled)

    def _debug_collision_prim_name_generate(self, original_prim_name: str, i_geometry: int) -> str:
        """Generate a unique debug prim path for collision visualization.

        Args:
            original_prim_name: Original prim path name.
            i_geometry: Index of the geometry part (for multi-part obstacles).

        Returns:
            Generated debug prim path in the format "/CumotionDebug/{original_name}/Part{i}".
        """
        if original_prim_name.startswith("/"):
            original_prim_name = original_prim_name[1:]
        altered_prim_name = f"/CumotionDebug/{original_prim_name}/Part{i_geometry}"
        return altered_prim_name

    def _validate_prim_paths(self, prim_paths: list[str]) -> bool:
        """Validate that all prim paths are currently tracked.

        Args:
            prim_paths: List of prim paths to validate.

        Returns:
            True if all prim paths are tracked, False otherwise.
        """
        if not set(prim_paths).issubset(self._prim_path_to_collision_data.keys()):
            return False
        return True

    def _extend_world_to_objects_matrix(self, position: np.ndarray, quaternion: np.ndarray) -> int:
        """Add a new world-to-object transform to the internal storage arrays.

        Args:
            position: Position array of shape (3,) or (1, 3).
            quaternion: Quaternion array of shape (4,) or (1, 4) in (w, x, y, z) format.

        Returns:
            Index where the transform is stored in the internal arrays.
        """
        self._position_world_to_objects = np.concatenate(
            [self._position_world_to_objects, np.reshape(position, shape=[-1, 3])]
        )

        self._quaternion_world_to_objects = np.concatenate(
            [self._quaternion_world_to_objects, np.reshape(quaternion, shape=[-1, 4])]
        )

        # Returns the index where this pose is stored:
        return len(self._position_world_to_objects) - 1

    def _extend_objects_to_colliders_matrix(self, position: np.ndarray, quaternion: np.ndarray) -> int:
        """Add a new object-to-collider transform to the internal storage arrays.

        Args:
            position: Position array of shape (3,) or (1, 3).
            quaternion: Quaternion array of shape (4,) or (1, 4) in (w, x, y, z) format.

        Returns:
            Index where the transform is stored in the internal arrays.
        """
        self._position_objects_to_colliders = np.concatenate(
            [self._position_objects_to_colliders, np.reshape(position, shape=[-1, 3])]
        )

        self._quaternion_objects_to_colliders = np.concatenate(
            [self._quaternion_objects_to_colliders, np.reshape(quaternion, shape=[-1, 4])]
        )

        # Returns the index where this pose is stored:
        return len(self._position_objects_to_colliders) - 1

    def _extend_collider_data(
        self, cumotion_colliders: list[_CumotionCollider], transform_world_to_object_index: int
    ) -> CollisionData:
        """Add collision data for a new object to the internal storage.

        Args:
            cumotion_colliders: List of _CumotionCollider objects representing the object's collision geometries.
            transform_world_to_object_index: Index of the world-to-object transform in the internal arrays.

        Returns:
            CollisionData object containing indices and metadata for the added object.
        """
        (obstacle_handles, debug_prim_paths, n_colliders) = _vectorize_cumotion_collider_data(cumotion_colliders)

        obstacle_handle_starting_index = len(self._all_obstacle_handles)

        self._all_obstacle_handles.extend(obstacle_handles)
        self._all_debug_prim_paths.extend(debug_prim_paths)

        self._n_colliders_in_each_object = np.append(self._n_colliders_in_each_object, n_colliders)

        collision_data = CollisionData(
            transform_world_to_object_index=transform_world_to_object_index,
            obstacle_handle_starting_index=obstacle_handle_starting_index,
            n_colliders=n_colliders,
        )

        return collision_data
