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

import isaacsim.robot_motion.experimental.motion_generation as mg
import numpy as np
import warp as wp
from isaacsim.core.experimental.objects import Cone, Cube, Cylinder, Mesh
from isaacsim.core.experimental.prims import Articulation, GeomPrim, XformPrim
from isaacsim.core.experimental.utils import prim as prim_utils
from isaacsim.core.experimental.utils import stage as stage_utils
from isaacsim.core.experimental.utils import transform as transform_utils
from isaacsim.core.experimental.utils.stage import add_reference_to_stage
from isaacsim.robot_motion.cumotion import (
    CumotionWorldInterface,
    RmpFlowController,
    load_cumotion_supported_robot,
)
from isaacsim.robot_motion.experimental.motion_generation import (
    ObstacleConfiguration,
    ObstacleStrategy,
    SceneQuery,
    TrackableApi,
    WorldBinding,
)
from isaacsim.storage.native import get_assets_root_path


class FrankaRmpFlowExample:
    """Example demonstrating RMPflow controller with cuMotion integration.

    This example shows how to:
    - Set up a cuMotion world interface with obstacles
    - Create an RMPflow controller
    - Use the controller to follow a target while avoiding obstacles
    """

    def __init__(self):
        self._controller = None
        self._articulation = None
        self._target = None
        self._world_binding = None
        self._controlled_joint_names = None

    def load_example_assets(self):
        """Load robot, target, and obstacle assets to the stage."""
        self._robot_prim_path = "/panda"
        path_to_robot_usd = get_assets_root_path() + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"

        add_reference_to_stage(path_to_robot_usd, self._robot_prim_path)
        self._articulation = Articulation(self._robot_prim_path)

        self._target = Cube(paths="/World/target", sizes=0.04, positions=[0.5, 0.0, 0.25])

        # Create fixed cube obstacle
        obstacle_path = "/World/obstacle"
        obstacle_size = 0.05
        obstacle_position = np.array([0.4, 0.0, 0.65])

        # Create cube geometry
        cube = Cube(obstacle_path, sizes=obstacle_size, positions=[obstacle_position])

        # Apply collision APIs
        GeomPrim(obstacle_path, apply_collision_apis=True)

        return self._articulation, self._target, cube

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
        """Set up the RMPflow controller and world interface."""
        # Clean up old world binding and debug prims before setting up
        if self._world_binding is not None:
            # Clean up debug prims from the old world binding
            self._cleanup_debug_prims()
            # Set to None to allow garbage collection of GPU resources
            self._world_binding = None

        # Clean up any remaining debug prims
        self._cleanup_debug_prims()

        # Load robot configuration
        robot_config = load_cumotion_supported_robot("franka")

        # Create scene query to discover obstacles
        scene_query = SceneQuery()

        # Get robot base transform
        robot_base_positions, robot_base_orientations = self._articulation.get_world_poses()

        # Get all objects surrounding the robot
        search_origin = robot_base_positions.numpy()[0] if robot_base_positions.shape[0] > 0 else [0.0, 0.0, 0.0]
        objects = scene_query.get_prims_in_aabb(
            search_box_origin=search_origin,
            search_box_minimum=[-10.0, -10.0, -10.0],
            search_box_maximum=[10.0, 10.0, 10.0],
            tracked_api=TrackableApi.PHYSICS_COLLISION,
            exclude_prim_paths=[self._robot_prim_path],  # don't include the franka itself
        )

        # Set up obstacle strategy
        obstacle_strategy = ObstacleStrategy()
        obstacle_strategy.set_default_configuration(Mesh, ObstacleConfiguration("obb", 0.01))
        obstacle_strategy.set_default_configuration(Cone, ObstacleConfiguration("obb", 0.01))
        obstacle_strategy.set_default_configuration(Cylinder, ObstacleConfiguration("obb", 0.01))

        # Create world binding
        self._world_binding = WorldBinding(
            world_interface=CumotionWorldInterface(),
            obstacle_strategy=obstacle_strategy,
            tracked_prims=objects,
            tracked_collision_api=TrackableApi.PHYSICS_COLLISION,
        )

        # Populate the world
        self._world_binding.initialize()

        # Update world interface with robot base transform
        self._world_binding.get_world_interface().update_world_to_robot_root_transforms(
            poses=(robot_base_positions, robot_base_orientations)
        )

        # Synchronize transforms before using controller
        self._world_binding.synchronize_transforms()

        # Get robot joint and site names
        self._robot_joint_space = self._articulation.dof_names
        self._robot_site_space = robot_config.robot_description.tool_frame_names()
        self._tool_frame = self._robot_site_space[0]

        # Create RMPflow controller
        self._controller = RmpFlowController(
            cumotion_robot=robot_config,
            cumotion_world_interface=self._world_binding.get_world_interface(),
            robot_joint_space=self._robot_joint_space,
            robot_site_space=self._robot_site_space,
            tool_frame=self._tool_frame,
        )

        # Store controlled joint names for update()
        self._controlled_joint_names = robot_config.controlled_joint_names

        # Set initial target position
        quat = transform_utils.euler_angles_to_quaternion([0, np.pi, 0]).numpy()
        self._target.set_world_poses(positions=np.array([[0.5, 0, 0.7]]), orientations=np.array([quat]))

        self._sim_time = 0.0
        self._controller_reset = False

    def update(self, step: float):
        """Update controller on each physics step."""
        if self._controller is None or self._world_binding is None:
            return

        # Get current robot state
        controlled_joint_names = self._controlled_joint_names

        estimated_state = mg.RobotState(
            joints=mg.JointState.from_name(
                robot_joint_space=self._robot_joint_space,
                positions=(self._robot_joint_space, self._articulation.get_dof_positions()),
                velocities=(self._robot_joint_space, self._articulation.get_dof_velocities()),
            )
        )

        # Get target pose
        target_positions, target_orientations = self._target.get_world_poses()

        # Create setpoint with target site
        setpoint_state = mg.RobotState(
            sites=mg.SpatialState.from_name(
                spatial_space=self._robot_site_space,
                positions=([self._tool_frame], target_positions),
                orientations=([self._tool_frame], target_orientations),
            ),
        )

        # Update world interface to track robot base movements
        self._world_binding.get_world_interface().update_world_to_robot_root_transforms(
            self._articulation.get_world_poses()
        )

        # Synchronize transforms
        self._world_binding.synchronize_transforms()

        if not self._controller_reset:
            self._controller_reset = self._controller.reset(estimated_state, setpoint_state, self._sim_time)

        # Get desired state from controller
        desired_state = self._controller.forward(estimated_state, setpoint_state, self._sim_time)

        # Apply action to articulation
        if desired_state is not None and desired_state.joints.positions is not None:
            self._articulation.set_dof_position_targets(
                positions=desired_state.joints.positions,
                dof_indices=desired_state.joints.position_indices,
            )

        self._sim_time += step

    def reset(self):
        """Reset the example."""
        self.setup()
