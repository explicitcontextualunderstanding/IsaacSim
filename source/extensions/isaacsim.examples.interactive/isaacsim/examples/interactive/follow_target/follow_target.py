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

"""Interactive sample demonstrating robotic arm target following with RMP Flow control in Isaac Sim."""


from isaacsim.examples.interactive.base_sample import BaseSample
from isaacsim.robot.manipulators.examples.franka.controllers.rmpflow_controller import RMPFlowController
from isaacsim.robot.manipulators.examples.franka.tasks import FollowTarget as FollowTargetTask


class FollowTarget(BaseSample):
    """Interactive sample demonstrating robotic arm target following with RMP Flow control.

    This class implements a complete Isaac Sim example where a Franka robotic manipulator uses RMP Flow
    controller to follow a target object in 3D space. The sample provides real-time control capabilities,
    obstacle management, and data logging functionality for robotics research and development.

    Key features include:

    - Real-time target following using RMP Flow motion planning
    - Dynamic obstacle addition and removal during simulation
    - Comprehensive data logging of joint positions and target states
    - Interactive control through physics callbacks
    - Automatic scene setup with Franka robot and target objects

    The sample integrates with Isaac Sim's physics simulation loop to provide smooth, responsive robot
    motion while avoiding obstacles. The RMP Flow controller generates joint commands based on target
    position and orientation, enabling precise end-effector tracking.

    Data logging captures joint positions, applied commands, and target poses for analysis and
    debugging. Obstacles can be dynamically added to test collision avoidance capabilities.
    """

    def __init__(self):
        super().__init__()
        self._controller = None
        self._articulation_controller = None

    def setup_scene(self):
        """Sets up the scene by adding the FollowTarget task to the world."""
        world = self.get_world()
        world.add_task(FollowTargetTask())
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
        """Sets up the controller and articulation controller after the world has loaded."""
        self._franka_task = list(self._world.get_current_tasks().values())[0]
        self._task_params = self._franka_task.get_params()
        my_franka = self._world.scene.get_object(self._task_params["robot_name"]["value"])
        self._controller = RMPFlowController(name="target_follower_controller", robot_articulation=my_franka)
        self._articulation_controller = my_franka.get_articulation_controller()
        return

    async def _on_follow_target_event_async(self, val):
        """Handles the follow target event by starting or stopping the simulation and physics callbacks.

        Args:
            val: Whether to start following the target.
        """
        world = self.get_world()
        if val:
            await world.play_async()
            world.add_physics_callback("sim_step", self._on_follow_target_simulation_step)
        else:
            world.remove_physics_callback("sim_step")
        return

    def _on_follow_target_simulation_step(self, step_size):
        """Executes a simulation step for target following by computing and applying control actions.

        Args:
            step_size: The simulation step size.
        """
        observations = self._world.get_observations()
        actions = self._controller.forward(
            target_end_effector_position=observations[self._task_params["target_name"]["value"]]["position"],
            target_end_effector_orientation=observations[self._task_params["target_name"]["value"]]["orientation"],
        )
        self._articulation_controller.apply_action(actions)
        return

    def _on_add_obstacle_event(self):
        """Handles adding an obstacle to the scene and controller."""
        world = self.get_world()
        current_task = list(world.get_current_tasks().values())[0]
        cube = current_task.add_obstacle()
        self._controller.add_obstacle(cube)
        return

    def _on_remove_obstacle_event(self):
        """Handles removing an obstacle from the scene and controller."""
        world = self.get_world()
        current_task = list(world.get_current_tasks().values())[0]
        obstacle_to_delete = current_task.get_obstacle_to_delete()
        self._controller.remove_obstacle(obstacle_to_delete)
        current_task.remove_obstacle()
        return

    def _on_logging_event(self, val):
        """Handles data logging events by starting or pausing the data logger.

        Args:
            val: Whether to start or pause logging.
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
        """Handles saving logged data to the specified path and resets the data logger.

        Args:
            log_path: The file path where the logged data will be saved.
        """
        world = self.get_world()
        data_logger = world.get_data_logger()
        data_logger.save(log_path=log_path)
        data_logger.reset()
        return
