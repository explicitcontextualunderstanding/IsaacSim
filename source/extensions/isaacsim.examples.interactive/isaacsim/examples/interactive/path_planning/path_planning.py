# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Interactive path planning example with Franka robot using RRT algorithms for obstacle avoidance."""


from isaacsim.examples.interactive.base_sample import BaseSample

from .path_planning_controller import FrankaRrtController
from .path_planning_task import FrankaPathPlanningTask


class PathPlanning(BaseSample):
    """Interactive path planning example using a Franka robot with RRT (Rapidly-exploring Random Tree) controller.

    This class demonstrates robotic path planning capabilities by implementing an interactive simulation where
    a Franka robot navigates around obstacles to reach target positions. The example uses RRT algorithms
    for motion planning and provides interactive controls for adding/removing obstacles, following targets,
    and logging simulation data.

    Key features:
    - Franka robot with disabled gravity for controlled movement
    - RRT-based path planning controller for obstacle avoidance
    - Interactive obstacle management (add/remove walls)
    - Target following with real-time path computation
    - Data logging capabilities for joint positions and target states
    - Custom PD controller gains for smooth motion execution

    The simulation setup includes:
    - A FrankaPathPlanningTask that manages the robot, target, and obstacles
    - A FrankaRrtController that computes collision-free paths
    - Physics callbacks for real-time simulation updates
    - Interactive UI controls for user interaction

    The path planning process works by:
    1. Setting up the scene with robot, target, and initial obstacles
    2. Using RRT algorithms to compute collision-free paths to targets
    3. Executing planned motions using articulation controllers
    4. Dynamically updating paths when obstacles or targets change

    This example is useful for understanding robotic motion planning concepts, RRT algorithms,
    and interactive simulation development in Isaac Sim.
    """

    def __init__(self):
        super().__init__()
        self._controller = None
        self._articulation_controller = None

    def setup_scene(self):
        """Sets up the scene by adding the Franka path planning task to the world."""
        world = self.get_world()
        world.add_task(FrankaPathPlanningTask("Plan To Target Task"))
        return

    async def setup_pre_reset(self):
        """Prepares the scene for reset by removing physics callbacks and resetting the controller."""
        world = self.get_world()
        if world.physics_callback_exists("sim_step"):
            world.remove_physics_callback("sim_step")
        self._controller.reset()
        return

    def world_cleanup(self):
        """Cleans up the world by resetting the controller reference."""
        self._controller = None
        return

    async def setup_post_load(self):
        """Sets up the scene after loading by initializing the Franka robot, controller, and articulation controller."""
        self._franka_task = list(self._world.get_current_tasks().values())[0]
        self._task_params = self._franka_task.get_params()
        my_franka = self._world.scene.get_object(self._task_params["robot_name"]["value"])
        my_franka.disable_gravity()
        self._controller = FrankaRrtController(name="franka_rrt_controller", robot_articulation=my_franka)
        self._articulation_controller = my_franka.get_articulation_controller()
        return

    async def _on_follow_target_event_async(self):
        """Handles the follow target event by passing world state to controller and starting physics simulation."""
        world = self.get_world()
        self._pass_world_state_to_controller()
        await world.play_async()
        if not world.physics_callback_exists("sim_step"):
            world.add_physics_callback("sim_step", self._on_follow_target_simulation_step)

    def _pass_world_state_to_controller(self):
        """Passes the current world state including obstacles to the path planning controller."""
        self._controller.reset()
        for wall in self._franka_task.get_obstacles():
            self._controller.add_obstacle(wall)

    def _on_follow_target_simulation_step(self, step_size):
        """Handles each simulation step during target following by computing and applying control actions.

        Args:
            step_size: The simulation step size.
        """
        observations = self._world.get_observations()
        actions = self._controller.forward(
            target_end_effector_position=observations[self._task_params["target_name"]["value"]]["position"],
            target_end_effector_orientation=observations[self._task_params["target_name"]["value"]]["orientation"],
        )
        kps, kds = self._franka_task.get_custom_gains()
        self._articulation_controller.set_gains(kps, kds)
        self._articulation_controller.apply_action(actions)
        return

    def _on_add_wall_event(self):
        """Handles the add wall event by creating a new obstacle in the current task."""
        world = self.get_world()
        current_task = list(world.get_current_tasks().values())[0]
        cube = current_task.add_obstacle()
        return

    def _on_remove_wall_event(self):
        """Handles the remove wall event by removing an obstacle from the current task."""
        world = self.get_world()
        current_task = list(world.get_current_tasks().values())[0]
        obstacle_to_delete = current_task.get_obstacle_to_delete()
        current_task.remove_obstacle()
        return

    def _on_logging_event(self, val):
        """Handles the logging event by starting or pausing data logging based on the input value.

        Args:
            val: Boolean value to start (True) or pause (False) logging.
        """
        world = self.get_world()
        data_logger = world.get_data_logger()
        if not world.get_data_logger().is_started():
            robot_name = self._task_params["robot_name"]["value"]
            target_name = self._task_params["target_name"]["value"]

            def frame_logging_func(tasks, scene):
                return {
                    "joint_positions": scene.get_object(robot_name).get_joint_positions().tolist(),
                    "applied_joint_positions": scene.get_object(robot_name)
                    .get_applied_action()
                    .joint_positions.tolist(),
                    "target_position": scene.get_object(target_name).get_world_pose()[0].tolist(),
                }

            data_logger.add_data_frame_logging_func(frame_logging_func)
        if val:
            data_logger.start()
        else:
            data_logger.pause()
        return

    def _on_save_data_event(self, log_path):
        """Saves logged data to the specified path and resets the data logger.

        Args:
            log_path: Path where the logged data will be saved.
        """
        world = self.get_world()
        data_logger = world.get_data_logger()
        data_logger.save(log_path=log_path)
        data_logger.reset()
        return
