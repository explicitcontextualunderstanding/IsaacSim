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

import numpy as np
import warp as wp
from isaacsim.core.experimental.objects import Cone, Cube, Cylinder, Mesh
from isaacsim.core.experimental.prims import Articulation, GeomPrim
from isaacsim.core.experimental.utils import prim as prim_utils
from isaacsim.core.experimental.utils import stage as stage_utils
from isaacsim.robot_motion.cumotion import (
    CumotionWorldInterface,
    GraphBasedMotionPlanner,
    load_cumotion_supported_robot,
)
from isaacsim.robot_motion.experimental.motion_generation import (
    ObstacleConfiguration,
    ObstacleStrategy,
    SceneQuery,
    TrackableApi,
    WorldBinding,
)


class FrankaGraphPlannerExample:
    """Example demonstrating graph-based motion planning with cuMotion.

    This class provides methods for planning and executing paths:
    - :meth:`plan_to_cspace_target`: Plan to a configuration space target
    - :meth:`plan_to_task_space_target`: Plan to a task-space target (end-effector pose)
    - :meth:`setup_world_and_planner`: Set up world binding and graph-based motion planner
    - :meth:`update`: Execute the planned trajectory over time
    """

    def __init__(self):
        self._articulation = None
        self._trajectory = None
        self._trajectory_time = 0.0
        self._target = None
        self._cumotion_robot = None
        self._robot_prim_path = None
        self._controlled_dof_indices = None
        self._q_initial = None
        self._first_trajectory = True

    def _fetch_initial_position(self):
        if self._first_trajectory:
            self._first_trajectory = False
        else:
            self._q_initial = (
                self._articulation.get_dof_positions(dof_indices=self._controlled_dof_indices)
                .numpy()
                .flatten()
                .astype(np.float64)
            )

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

    def setup_world_and_planner(self):
        """Set up world binding and graph-based motion planner (shared by both target types)."""
        # Load robot configuration
        robot_config = load_cumotion_supported_robot("franka")

        # Create world interface - this owns all world state
        scene_query = SceneQuery()

        # Update world interface with robot base transform
        robot_base_positions, robot_base_orientations = self._articulation.get_world_poses()

        # Get all of the objects surrounding the robot:
        # Convert robot_base_positions from (1, 3) to (3,) array for search_box_origin
        search_origin = robot_base_positions.numpy()[0] if robot_base_positions.shape[0] > 0 else [0.0, 0.0, 0.0]
        objects = scene_query.get_prims_in_aabb(
            search_box_origin=search_origin,
            search_box_minimum=[-10.0, -10.0, -10.0],
            search_box_maximum=[10.0, 10.0, 10.0],
            tracked_api=TrackableApi.PHYSICS_COLLISION,
            exclude_prim_paths=[self._robot_prim_path, "/World/target"],  # don't include the franka itself or target
        )

        print("Objects: ", objects)

        obstacle_strategy = ObstacleStrategy()
        obstacle_strategy.set_default_configuration(Mesh, ObstacleConfiguration("obb", 0.01))
        obstacle_strategy.set_default_configuration(Cone, ObstacleConfiguration("obb", 0.01))
        obstacle_strategy.set_default_configuration(Cylinder, ObstacleConfiguration("obb", 0.01))

        world_binding = WorldBinding(
            world_interface=CumotionWorldInterface(visualize_debug_prims=True),
            obstacle_strategy=obstacle_strategy,
            tracked_prims=objects,
            tracked_collision_api=TrackableApi.PHYSICS_COLLISION,
        )
        print(f"tracked collision prims: {objects}")

        # populate the world:
        world_binding.initialize()

        world_binding.get_world_interface().update_world_to_robot_root_transforms(
            poses=(robot_base_positions, robot_base_orientations)
        )

        # update any out of date transforms before planning:
        world_binding.synchronize_transforms()

        # Create graph-based motion planner
        planner = GraphBasedMotionPlanner(
            cumotion_robot=robot_config,
            cumotion_world_interface=world_binding.get_world_interface(),
        )

        return robot_config, world_binding, planner

    def plan_to_cspace_target(self, q_target=None):
        """Plan a path to a C-space target.

        Args:
            q_target: Target configuration. If None, uses default modified configuration.

        Returns:
            str: Error message if planning failed, None if successful.
        """
        # Clean up old debug prims before planning
        self._cleanup_debug_prims()

        robot_config, world_binding, planner = self.setup_world_and_planner()

        self._fetch_initial_position()

        # Create target configuration
        if q_target is None:
            # Default target configuration by modifying default
            q_target = robot_config.robot_description.default_cspace_configuration()
            q_target[0] = np.pi / 2
            q_target[1] = -np.pi / 3  # Modify joint 1 for a different target pose
        else:
            # Ensure q_target is a numpy array
            q_target = np.array(q_target)

        # Plan path
        path = planner.plan_to_cspace_target(self._q_initial, q_target)

        if path is None:
            return "Planning failed: Unable to find a valid path.\n\nCommon issues:\n  - Start/goal configurations outside joint limits\n  - Start/goal configurations in collision\n  - Insufficient planning iterations"

        # Convert path to trajectory using minimal-time joint trajectory
        robot_joint_space = self._articulation.dof_names
        controlled_joint_names = robot_config.controlled_joint_names

        # Use reasonable velocity and acceleration limits for Franka
        max_velocities = np.array([2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5])  # rad/s
        max_accelerations = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])  # rad/s²

        # Convert Path to minimal-time trajectory
        self._trajectory = path.to_minimal_time_joint_trajectory(
            max_velocities=max_velocities,
            max_accelerations=max_accelerations,
            robot_joint_space=robot_joint_space,
            active_joints=controlled_joint_names,
        )

        self._trajectory_time = 0.0
        return None

    def plan_to_task_space_target(self):
        """Plan a path to a task-space target (from target cube position).

        Returns:
            str: Error message if planning failed, None if successful.
        """
        # Clean up old debug prims before planning
        self._cleanup_debug_prims()

        robot_config, world_binding, planner = self.setup_world_and_planner()

        self._fetch_initial_position()

        # Get target cube pose in world frame
        target_positions, target_orientations = self._target.get_world_poses()
        target_position_world = target_positions.numpy()[0]
        target_orientation_world = target_orientations.numpy()[0]  # quaternion wxyz

        # Plan to pose target (position and orientation are in world frame)
        path = planner.plan_to_pose_target(
            q_initial=self._q_initial,
            position=target_position_world,
            orientation=target_orientation_world,
        )

        if path is None:
            return "Planning failed: Unable to find a valid path.\n\nCommon issues:\n  - Start configuration outside joint limits or in collision\n  - Target pose unreachable\n  - Insufficient planning iterations"

        # Convert path to trajectory using minimal-time joint trajectory
        robot_joint_space = self._articulation.dof_names
        controlled_joint_names = robot_config.controlled_joint_names

        # Use reasonable velocity and acceleration limits for Franka
        max_velocities = np.array([2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5])  # rad/s
        max_accelerations = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])  # rad/s²

        # Convert Path to minimal-time trajectory
        self._trajectory = path.to_minimal_time_joint_trajectory(
            max_velocities=max_velocities,
            max_accelerations=max_accelerations,
            robot_joint_space=robot_joint_space,
            active_joints=controlled_joint_names,
        )

        self._trajectory_time = 0.0
        return None

    def update(self, step: float):
        """Update trajectory execution on each physics step."""
        if self._trajectory is None:
            return

        # Sample trajectory at current time
        target_state = self._trajectory.get_target_state(self._trajectory_time)

        if target_state is not None and target_state.joints.positions is not None:
            # NOTE: usually you would set targets - for the purpose of
            # demonstrating a trajectory, we write directly to the physics
            # joint positions of the robot.
            self._articulation.set_dof_positions(
                positions=target_state.joints.positions, dof_indices=target_state.joints.position_indices
            )

        # Advance trajectory time
        self._trajectory_time += step

        # Clamp to trajectory duration
        if self._trajectory_time >= self._trajectory.duration:
            self._trajectory_time = self._trajectory.duration
