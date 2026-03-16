# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Interactive path planning controller module for robotic motion generation using RRT algorithms and trajectory optimization."""


import os
from typing import Optional

import carb
import isaacsim.core.api.objects
import isaacsim.robot_motion.motion_generation.interface_config_loader as interface_config_loader
import numpy as np
import omni.kit
from isaacsim.core.api.controllers.base_controller import BaseController
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot_motion.motion_generation import ArticulationTrajectory
from isaacsim.robot_motion.motion_generation.lula import RRT
from isaacsim.robot_motion.motion_generation.lula.trajectory_generator import LulaCSpaceTrajectoryGenerator
from isaacsim.robot_motion.motion_generation.path_planner_visualizer import PathPlannerVisualizer
from isaacsim.robot_motion.motion_generation.path_planning_interface import PathPlanner


class PathPlannerController(BaseController):
    """A controller that uses path planning algorithms to generate motion plans for robotic articulations.

    This controller integrates RRT-based path planning with trajectory generation to enable robots to navigate
    from their current configuration to target end-effector poses while avoiding obstacles. It converts RRT
    waypoints into smooth trajectories using spline-based interpolation and executes them as articulation
    actions.

    The controller maintains state between planning cycles, using the final position of the previous plan as
    the starting point for new plans. It supports dynamic obstacle addition and removal during operation.

    Args:
        name: Name identifier for the controller instance.
        path_planner_visualizer: Visualizer that provides access to the robot articulation and path planner.
        cspace_trajectory_generator: Generator that converts RRT waypoints into smooth cspace trajectories.
        physics_dt: Physics simulation timestep for trajectory timing.
        rrt_interpolation_max_dist: Maximum distance between interpolated waypoints when converting RRT plans
            to trajectories.
    """

    def __init__(
        self,
        name: str,
        path_planner_visualizer: PathPlannerVisualizer,
        cspace_trajectory_generator: LulaCSpaceTrajectoryGenerator,
        physics_dt=1 / 60.0,
        rrt_interpolation_max_dist=0.01,
    ):
        BaseController.__init__(self, name)

        self._robot = path_planner_visualizer.get_robot_articulation()

        self._cspace_trajectory_generator = cspace_trajectory_generator
        self._path_planner = path_planner_visualizer.get_path_planner()
        self._path_planner_visualizer = path_planner_visualizer

        self._last_solution = None
        self._action_sequence = None

        self._physics_dt = physics_dt
        self._rrt_interpolation_max_dist = rrt_interpolation_max_dist

    def _convert_rrt_plan_to_trajectory(self, rrt_plan):
        """Converts RRT path waypoints to a smooth articulation trajectory using spline interpolation.

        Args:
            rrt_plan: The RRT path plan containing waypoints to convert.

        Returns:
            Action sequence for executing the planned trajectory.
        """
        # This example uses the LulaCSpaceTrajectoryGenerator to convert RRT waypoints to a cspace trajectory.
        # In general this is not theoretically guaranteed to work since the trajectory generator uses spline-based
        # interpolation and RRT only guarantees that the cspace position of the robot can be linearly interpolated between
        # waypoints.  For this example, we verified experimentally that a dense interpolation of cspace waypoints with a maximum
        # l2 norm of .01 between waypoints leads to a good enough approximation of the RRT path by the trajectory generator.

        interpolated_path = self._path_planner_visualizer.interpolate_path(rrt_plan, self._rrt_interpolation_max_dist)
        trajectory = self._cspace_trajectory_generator.compute_c_space_trajectory(interpolated_path)
        art_trajectory = ArticulationTrajectory(self._robot, trajectory, self._physics_dt)

        return art_trajectory.get_action_sequence()

    def _make_new_plan(
        self, target_end_effector_position: np.ndarray, target_end_effector_orientation: Optional[np.ndarray] = None
    ) -> None:
        """Generates a new RRT path plan to reach the target end effector pose.

        Args:
            target_end_effector_position: Target position for the end effector.
            target_end_effector_orientation: Target orientation for the end effector.
        """
        self._path_planner.set_end_effector_target(target_end_effector_position, target_end_effector_orientation)
        self._path_planner.update_world()

        path_planner_visualizer = PathPlannerVisualizer(self._robot, self._path_planner)
        active_joints = path_planner_visualizer.get_active_joints_subset()
        if self._last_solution is None:
            start_pos = active_joints.get_joint_positions()
        else:
            start_pos = self._last_solution

        self._path_planner.set_max_iterations(5000)
        self._rrt_plan = self._path_planner.compute_path(start_pos, np.array([]))

        if self._rrt_plan is None or len(self._rrt_plan) <= 1:
            carb.log_warn("No plan could be generated to target pose: " + str(target_end_effector_position))
            self._action_sequence = []
            return

        print(len(self._rrt_plan))

        self._action_sequence = self._convert_rrt_plan_to_trajectory(self._rrt_plan)
        self._last_solution = self._action_sequence[-1].joint_positions

    def forward(
        self, target_end_effector_position: np.ndarray, target_end_effector_orientation: Optional[np.ndarray] = None
    ) -> ArticulationAction:
        """Computes the next action in the motion plan toward the target end effector pose.

        Args:
            target_end_effector_position: Target position for the end effector.
            target_end_effector_orientation: Target orientation for the end effector.

        Returns:
            The next articulation action in the planned sequence.
        """
        if self._action_sequence is None:
            # This will only happen the first time the forward function is used
            self._make_new_plan(target_end_effector_position, target_end_effector_orientation)

        if len(self._action_sequence) == 0:
            # The plan is completed; return null action to remain in place
            return ArticulationAction()

        if len(self._action_sequence) == 1:
            final_positions = self._action_sequence[0].joint_positions
            return ArticulationAction(
                final_positions, np.zeros_like(final_positions), joint_indices=self._action_sequence[0].joint_indices
            )

        return self._action_sequence.pop(0)

    def add_obstacle(self, obstacle: isaacsim.core.api.objects, static: bool = False) -> None:
        """Adds an obstacle to the path planning environment.

        Args:
            obstacle: The obstacle object to add to path planning.
            static: Whether the obstacle is static or dynamic.
        """
        self._path_planner.add_obstacle(obstacle, static)

    def remove_obstacle(self, obstacle: isaacsim.core.api.objects) -> None:
        """Removes an obstacle from the path planning environment.

        Args:
            obstacle: The obstacle object to remove from path planning.
        """
        self._path_planner.remove_obstacle(obstacle)

    def reset(self):
        """Resets the controller state by clearing the current plan and last solution."""
        # PathPlannerController will make one plan per reset
        self._path_planner.reset()
        self._action_sequence = None
        self._last_solution = None

    def get_path_planner(self) -> PathPlanner:
        """The underlying path planner instance used for motion planning.

        Returns:
            The path planner instance.
        """
        return self._path_planner


class FrankaRrtController(PathPlannerController):
    """A specialized path planning controller for Franka robots using RRT (Rapidly-exploring Random Tree) algorithm.

    This controller extends PathPlannerController to provide motion planning capabilities specifically
    tailored for Franka robotic arms. It automatically configures RRT path planning with conservative
    collision spheres and trajectory generation optimized for Franka robots.

    The controller uses inflated collision spheres for safe path planning and includes jerk and
    acceleration limits to ensure generated trajectories can be followed accurately by the simulated robot.
    It loads Franka-specific configuration files and sets up the necessary components for robust motion
    planning in cluttered environments.

    Args:
        name: Unique identifier for the controller instance.
        robot_articulation: The Franka robot articulation to be controlled.
    """

    def __init__(
        self,
        name,
        robot_articulation: SingleArticulation,
    ):
        ext_manager = omni.kit.app.get_app().get_extension_manager()
        ext_id = ext_manager.get_enabled_extension_id("isaacsim.examples.interactive")
        examples_extension_path = ext_manager.get_extension_path(ext_id)

        # Load default RRT config files stored in the isaacsim.robot_motion.motion_generation extension
        rrt_config = interface_config_loader.load_supported_path_planner_config("Franka", "RRT")

        # Replace the default robot description file with a copy that has inflated collision spheres
        rrt_config["robot_description_path"] = os.path.join(
            examples_extension_path,
            "isaacsim",
            "examples",
            "interactive",
            "path_planning",
            "path_planning_example_assets",
            "franka_conservative_spheres_robot_description.yaml",
        )
        rrt = RRT(**rrt_config)

        # Create a trajectory generator to convert RRT cspace waypoints to trajectories
        cspace_trajectory_generator = LulaCSpaceTrajectoryGenerator(
            rrt_config["robot_description_path"], rrt_config["urdf_path"]
        )

        # It is important that the Robot Description File includes optional Jerk and Acceleration limits so that the generated trajectory
        # can be followed closely by the simulated robot Articulation
        for i in range(len(rrt.get_active_joints())):
            assert cspace_trajectory_generator._lula_kinematics.has_c_space_acceleration_limit(i)
            assert cspace_trajectory_generator._lula_kinematics.has_c_space_jerk_limit(i)

        visualizer = PathPlannerVisualizer(robot_articulation, rrt)

        PathPlannerController.__init__(self, name, visualizer, cspace_trajectory_generator)
