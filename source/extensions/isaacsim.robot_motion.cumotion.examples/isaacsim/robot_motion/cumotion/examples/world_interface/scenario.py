# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Literal

from isaacsim.core.experimental.objects import Cone, Cube, Cylinder, Mesh
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim
from isaacsim.core.experimental.utils import prim as prim_utils
from isaacsim.core.experimental.utils import stage as stage_utils
from isaacsim.robot_motion.cumotion import CumotionWorldInterface
from isaacsim.robot_motion.experimental.motion_generation import (
    ObstacleConfiguration,
    ObstacleStrategy,
    SceneQuery,
    TrackableApi,
    WorldBinding,
)


class CumotionWorldInterfaceExample:
    """Example demonstrating CumotionWorldInterface.

    This example shows how to:
    - Set up a CumotionWorldInterface with WorldBinding
    - Discover obstacles using SceneQuery
    - Configure obstacle representations
    - Synchronize world state
    """

    def __init__(self):
        self._world_binding = None
        self._update_style: Literal["synchronize_transforms", "synchronize_properties", "synchronize"] = (
            "synchronize_transforms"  # default to transforms only
        )
        self._tracked_collision_api: Literal["physics", "motion_generation"] = "physics"  # default to physics

    def set_update_style(self, style: Literal["synchronize_transforms", "synchronize_properties", "synchronize"]):
        """Set the update style for world binding synchronization."""
        if style not in ["synchronize_transforms", "synchronize_properties", "synchronize"]:
            raise ValueError(
                f"Invalid update style: {style}. Must be one of: synchronize_transforms, synchronize_properties, synchronize"
            )
        self._update_style = style

    def set_tracked_collision_api(self, api: Literal["physics", "motion_generation"]):
        """Set the tracked collision API."""
        if api not in ["physics", "motion_generation"]:
            raise ValueError(f"Invalid collision API: {api}. Must be one of: physics, motion_generation")
        self._tracked_collision_api = api

    def load_example_assets(self):
        """Load robot assets to the stage."""
        obstacle_path = "/World/obstacle"

        # Create cube geometry
        cube = Cube(obstacle_path, sizes=0.1, positions=[0.4, 0.0, 0.65])

        # Apply collision APIs
        GeomPrim(obstacle_path, apply_collision_apis=True)
        RigidPrim(obstacle_path, masses=[1.0])
        return cube

    def _cleanup_debug_prims(self):
        """Delete all prims under 'CumotionDebug' to clean up old debug visualization."""
        # Find all prims that have "CumotionDebug" in their path
        debug_prim_paths = prim_utils.find_matching_prim_paths(".*CumotionDebug.*", traverse=True)

        if not debug_prim_paths:
            return

        # Filter to only root-level prims (ones whose parent is not in the list)
        # Deleting a parent automatically deletes all its children
        debug_prim_paths_set = set(debug_prim_paths)
        root_prim_paths = [path for path in debug_prim_paths if path.rsplit("/", 1)[0] not in debug_prim_paths_set]

        # Delete only the root prims
        for prim_path in root_prim_paths:
            try:
                stage_utils.delete_prim(prim_path)
            except ValueError:
                # Prim may have already been deleted or doesn't exist, skip
                pass

    def setup(self):
        """Set up the world interface."""
        # Clean up old world binding and debug prims before setting up
        # Set old world binding to None to allow garbage collection
        if self._world_binding is not None:
            # Clean up debug prims from the old world binding
            self._cleanup_debug_prims()
            # Set to None to allow garbage collection of GPU resources
            self._world_binding = None

        # Clean up any remaining debug prims
        self._cleanup_debug_prims()

        # Find all objects close to the origin:
        scene_query = SceneQuery()
        objects = scene_query.get_prims_in_aabb(
            search_box_origin=[0.0, 0.0, 0.0],
            search_box_minimum=[-100.0, -100.0, -100.0],
            search_box_maximum=[100.0, 100.0, 100.0],
            tracked_api=TrackableApi.PHYSICS_COLLISION,
        )
        print("Objects: ", objects)

        # HERE: try changing the representation to TRIANGULATED_MESH,
        # or, try changing the safety tolerance to 0.05, 0.1, 0.2, etc.
        obstacle_strategy = ObstacleStrategy()
        obstacle_strategy.set_default_safety_tolerance(0.06)
        obstacle_strategy.set_default_configuration(Mesh, ObstacleConfiguration("obb", 0.01))
        obstacle_strategy.set_default_configuration(Cone, ObstacleConfiguration("obb", 0.01))
        obstacle_strategy.set_default_configuration(Cylinder, ObstacleConfiguration("obb", 0.01))

        # Create a world binding, debug visualizations enabled:
        self._world_binding = WorldBinding(
            world_interface=CumotionWorldInterface(visualize_debug_prims=True),
            obstacle_strategy=obstacle_strategy,
            tracked_prims=objects,
            tracked_collision_api=TrackableApi.PHYSICS_COLLISION,
        )

        # initialize the objects into the cumotion world:
        self._world_binding.initialize()

    def reset(self):
        self.setup()

    def update(self, dt: float):
        """Use different styles of updating the world binding"""
        if self._world_binding is None:
            return

        if self._update_style == "synchronize_transforms":
            self._world_binding.synchronize_transforms()
        elif self._update_style == "synchronize_properties":
            self._world_binding.synchronize_properties()
        elif self._update_style == "synchronize":
            self._world_binding.synchronize()
        else:
            raise ValueError(f"Invalid update style: {self._update_style}")
