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

import carb
import cumotion
import numpy as np
import warp as wp
from isaacsim.core.experimental.prims import Articulation, XformPrim
from isaacsim.core.experimental.utils import prim as prim_utils
from isaacsim.core.experimental.utils import stage as stage_utils
from isaacsim.core.experimental.utils.stage import add_reference_to_stage
from isaacsim.robot_motion.cumotion import (
    TrajectoryGenerator,
    load_cumotion_supported_robot,
)
from isaacsim.robot_motion.cumotion.impl.utils import (
    cumotion_to_isaac_sim_pose,
    isaac_sim_to_cumotion_pose,
)
from isaacsim.storage.native import get_assets_root_path


class UR10TrajectoryGeneratorExample:
    """Example demonstrating trajectory generation with cuMotion.

    This example shows how to:
    - Generate trajectories from C-space waypoints
    - Generate trajectories from path specifications
    - Execute trajectories on a robot
    """

    def __init__(self):
        self._articulation = None
        self._trajectory = None
        self._trajectory_time = 0.0
        self._robot_joint_space = None
        self._controlled_joint_names = None
        self._robot_config = None
        self._generator = None

    def load_example_assets(self):
        """Load robot assets to the stage."""
        robot_prim_path = "/ur10"
        path_to_robot_usd = get_assets_root_path() + "/Isaac/Robots/UniversalRobots/ur10/ur10.usd"

        add_reference_to_stage(path_to_robot_usd, robot_prim_path)
        self._articulation = Articulation(robot_prim_path)

        return self._articulation

    def setup(self):
        """Set up the trajectory generator (called on initialization)."""
        # Load robot configuration
        robot_config = load_cumotion_supported_robot("ur10")

        # Get robot joint names
        self._robot_joint_space = self._articulation.dof_names
        self._controlled_joint_names = robot_config.controlled_joint_names

        # Store robot config and generator for use in setup methods
        self._robot_config = robot_config
        self._generator = TrajectoryGenerator(
            cumotion_robot=robot_config,
            robot_joint_space=self._robot_joint_space,
        )

        self._tool_frame_name = "ee_link"

    def setup_cspace_trajectory(self):
        """Set up C-space trajectory from waypoints."""
        c_space_points = np.array(
            [
                [-0.41, 0.5, -2.36, -1.28, 5.13, -4.71],
                [-1.43, 1.0, -2.58, -1.53, 6.0, -4.74],
                [-2.83, 0.34, -2.11, -1.38, 1.26, -4.71],
                [-0.41, 0.5, -2.36, -1.28, 5.13, -4.71],  # Return to initial
            ]
        )

        # Visualize c-space targets in task space
        kinematics = self._robot_config.kinematics
        robot_base_positions, robot_base_orientations = self._articulation.get_world_poses()
        for i, point in enumerate(c_space_points):
            # Compute forward kinematics to get end effector pose
            pose_base_to_ee = kinematics.pose(point, self._tool_frame_name)

            # Convert to world frame for visualization
            position_world, quaternion_world = cumotion_to_isaac_sim_pose(
                pose_base_to_target=pose_base_to_ee,
                position_world_to_base=robot_base_positions,
                orientation_world_to_base=robot_base_orientations,
            )

            # Add frame prim for visualization
            frame_path = f"/visualized_frames/target_{i}"
            add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", frame_path)
            frame = XformPrim(frame_path, reset_xform_op_properties=True)
            frame.set_world_poses(
                positions=np.array([position_world.numpy()], dtype=np.float32),
                orientations=np.array([quaternion_world.numpy()], dtype=np.float32),
            )
            frame.set_local_scales(np.array([[0.04, 0.04, 0.04]], dtype=np.float32))

        self._trajectory = self._generator.generate_trajectory_from_cspace_waypoints(waypoints=c_space_points)
        self._trajectory_time = 0.0

    def setup_taskspace_trajectory(self):
        """Set up task-space trajectory from path specification."""
        # Get robot base transform directly from articulation
        robot_base_positions, robot_base_orientations = self._articulation.get_world_poses()

        task_space_position_targets = np.array(
            [[0.3, -0.3, 0.1], [0.3, 0.3, 0.1], [0.3, 0.3, 0.5], [0.3, -0.3, 0.5], [0.3, -0.3, 0.1]]
        )
        task_space_orientation_targets = np.tile(np.array([0, 1, 0, 0]), (5, 1))

        # Visualize task-space targets
        for i, (position, orientation) in enumerate(zip(task_space_position_targets, task_space_orientation_targets)):
            frame_path = f"/visualized_frames/target_{i}"
            add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", frame_path)
            frame = XformPrim(frame_path, reset_xform_op_properties=True)
            frame.set_world_poses(
                positions=np.array([position], dtype=np.float32),
                orientations=np.array([orientation], dtype=np.float32),
            )
            frame.set_local_scales(np.array([[0.04, 0.04, 0.04]], dtype=np.float32))

        # Create task-space path spec with first waypoint
        initial_pose = isaac_sim_to_cumotion_pose(
            position_world_to_target=task_space_position_targets[0],
            orientation_world_to_target=task_space_orientation_targets[0],
            position_world_to_base=robot_base_positions,
            orientation_world_to_base=robot_base_orientations,
        )
        path_spec = cumotion.create_task_space_path_spec(initial_pose)

        # Add linear path segments for remaining waypoints
        for i in range(1, len(task_space_position_targets)):
            target_pose = isaac_sim_to_cumotion_pose(
                position_world_to_target=task_space_position_targets[i],
                orientation_world_to_target=task_space_orientation_targets[i],
                position_world_to_base=robot_base_positions,
                orientation_world_to_base=robot_base_orientations,
            )
            path_spec.add_linear_path(target_pose)

        # Generate trajectory from path spec
        self._trajectory = self._generator.generate_trajectory_from_path_specification(
            path_specification=path_spec, tool_frame_name=self._tool_frame_name
        )
        self._trajectory_time = 0.0

    def setup_hybrid_trajectory(self):
        """Set up hybrid trajectory combining C-space and task-space paths."""
        # Get tool frame name
        initial_c_space_robot_pose = np.array([0, 0, 0, 0, 0, 0])

        # Create composite path spec
        composite_spec = cumotion.create_composite_path_spec(initial_c_space_robot_pose)

        #############################################################################
        # Demonstrate all the available movements in a taskspace path spec:

        # Convert angle-axis rotations to quaternions (w, x, y, z format)
        angle0 = np.pi / 2
        axis0 = np.array([1.0, 0.0, 0.0])
        w0 = np.cos(angle0 / 2)
        xyz0 = np.sin(angle0 / 2) * axis0
        quat0 = np.array([w0, xyz0[0], xyz0[1], xyz0[2]])

        t0 = np.array([0.3, -0.1, 0.3])
        pose0 = isaac_sim_to_cumotion_pose(position_world_to_target=t0, orientation_world_to_target=quat0)
        task_space_spec = cumotion.create_task_space_path_spec(pose0)

        # Add path linearly interpolating between r0,r1 and t0,t1
        t1 = np.array([0.3, -0.1, 0.5])
        angle1 = np.pi / 3
        axis1 = np.array([1, 0, 0])
        w1 = np.cos(angle1 / 2)
        xyz1 = np.sin(angle1 / 2) * axis1
        quat1 = np.array([w1, xyz1[0], xyz1[1], xyz1[2]])
        pose1 = isaac_sim_to_cumotion_pose(position_world_to_target=t1, orientation_world_to_target=quat1)
        task_space_spec.add_linear_path(pose1)

        # Add pure translation. Constant rotation is assumed
        # Extract base-frame translation from pose (since base=world, this matches t0)
        task_space_spec.add_translation(pose0.translation)

        # Add pure rotation.
        task_space_spec.add_rotation(pose0.rotation)

        # Add three-point arc with constant orientation.
        t2 = np.array([0.3, 0.3, 0.3])
        midpoint = np.array([0.3, 0, 0.5])
        task_space_spec.add_three_point_arc(t2, midpoint, constant_orientation=True)

        # Add three-point arc with tangent orientation.
        task_space_spec.add_three_point_arc(t0, midpoint, constant_orientation=False)

        # Add three-point arc with orientation target.
        pose2 = isaac_sim_to_cumotion_pose(position_world_to_target=t2, orientation_world_to_target=quat1)
        task_space_spec.add_three_point_arc_with_orientation_target(pose2, midpoint)

        # Add tangent arc with constant orientation. Tangent arcs are circles that connect two points
        task_space_spec.add_tangent_arc(t0, constant_orientation=True)

        # Add tangent arc with tangent orientation.
        task_space_spec.add_tangent_arc(t2, constant_orientation=False)

        # Add tangent arc with orientation target.
        task_space_spec.add_tangent_arc_with_orientation_target(pose0)

        ###################################################
        # Demonstrate the usage of a c_space path spec:
        c_space_spec = cumotion.create_cspace_path_spec(initial_c_space_robot_pose)
        c_space_waypoint = np.array([0, 0.5, -2.0, -1.28, 5.13, -4.71])
        c_space_spec.add_cspace_waypoint(c_space_waypoint)

        ##############################################################
        # Combine the two path specs together into a composite spec:

        # Specify how to connect initial_c_space and task_space points with transition_mode option
        transition_mode = cumotion.CompositePathSpec.TransitionMode.FREE
        composite_spec.add_task_space_path_spec(task_space_spec, transition_mode)

        transition_mode = cumotion.CompositePathSpec.TransitionMode.FREE
        composite_spec.add_cspace_path_spec(c_space_spec, transition_mode)

        # Generate trajectory from composite path spec
        self._trajectory = self._generator.generate_trajectory_from_path_specification(
            path_specification=composite_spec, tool_frame_name=self._tool_frame_name
        )
        if self._trajectory is None:
            carb.log_warn(
                "No trajectory could be computed from composite path spec. The path may contain unreachable poses."
            )
        self._trajectory_time = 0.0

    def update(self, step: float):
        """Update trajectory execution on each physics step."""
        if self._trajectory is None:
            return

        # Get current trajectory state
        desired_state = self._trajectory.get_target_state(self._trajectory_time)

        if desired_state is not None:
            # Apply to robot using position targets
            self._articulation.set_dof_positions(
                positions=desired_state.joints.positions,
                dof_indices=desired_state.joints.position_indices,
            )

        # Advance trajectory time
        self._trajectory_time += step

        # Loop trajectory by resetting time when it reaches duration
        if self._trajectory_time >= self._trajectory.duration:
            self._trajectory_time = 0.0

    def reset(self):
        """Reset the example."""
        # Delete any visualized frames
        prim = prim_utils.get_prim_at_path("/visualized_frames")
        if prim and prim.IsValid():
            stage_utils.delete_prim("/visualized_frames")

        self._trajectory = None
        self._trajectory_time = 0.0
