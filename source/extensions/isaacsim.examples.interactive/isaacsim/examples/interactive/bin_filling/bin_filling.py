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

"""Interactive demonstration of robotic bin filling using a UR10 manipulator with pick and place operations."""


import numpy as np
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.examples.interactive.base_sample import BaseSample
from isaacsim.robot.manipulators.examples.universal_robots.controllers.pick_place_controller import PickPlaceController
from isaacsim.robot.manipulators.examples.universal_robots.tasks import BinFilling as BinFillingTask


class BinFilling(BaseSample):
    """Interactive demonstration of robotic bin filling using a UR10 manipulator.

    This example showcases an autonomous bin filling task where a UR10 robotic arm picks up objects
    and places them into a designated bin. The demonstration includes:

    - Setting up a UR10 manipulator with gripper
    - Implementing pick and place operations using a dedicated controller
    - Managing object spawning and placement during simulation
    - Handling task completion and reset functionality

    The robot uses visual observations to determine picking and placing positions, automatically
    pausing to add cubes when appropriate, and stopping when the task is complete. The example
    inherits from BaseSample to provide standard interactive sample functionality including
    scene setup, event handling, and cleanup operations.

    The demonstration runs asynchronously and can be reset to repeat the bin filling process.
    """

    def __init__(self):
        super().__init__()
        self._controller = None
        self._articulation_controller = None
        self._added_cubes = False

    def setup_scene(self):
        """Sets up the bin filling task in the world scene.

        Adds the BinFillingTask to the world with the name "bin_filling".
        """
        world = self.get_world()
        world.add_task(BinFillingTask(name="bin_filling"))
        return

    async def setup_post_load(self):
        """Configures the bin filling task components after scene loading.

        Initializes the UR10 task, retrieves task parameters, sets up the pick and place controller
        with the robot's gripper, and configures the articulation controller.
        """
        self._ur10_task = self._world.get_task(name="bin_filling")
        self._task_params = self._ur10_task.get_params()
        my_ur10 = self._world.scene.get_object(self._task_params["robot_name"]["value"])
        self._controller = PickPlaceController(
            name="pick_place_controller", gripper=my_ur10.gripper, robot_articulation=my_ur10
        )
        self._articulation_controller = my_ur10.get_articulation_controller()
        return

    def _on_fill_bin_physics_step(self, step_size: float):
        """Handles physics step calculations for the bin filling process.

        Executes the pick and place controller, manages cube addition when the controller reaches event 6,
        and pauses the world when the filling task is complete.

        Args:
            step_size: The physics simulation step size.
        """
        observations = self._world.get_observations()
        actions = self._controller.forward(
            picking_position=observations[self._task_params["bin_name"]["value"]]["position"],
            placing_position=observations[self._task_params["bin_name"]["value"]]["target_position"],
            current_joint_positions=observations[self._task_params["robot_name"]["value"]]["joint_positions"],
            end_effector_offset=np.array([0, -0.098, 0.03]),
            end_effector_orientation=euler_angles_to_quat(np.array([np.pi, 0, np.pi / 2.0])),
        )
        if not self._added_cubes and self._controller.get_current_event() == 6 and not self._controller.is_paused():
            self._controller.pause()
            self._ur10_task.add_cubes(cubes_number=20)
            self._added_cubes = True
        if self._controller.is_done():
            self._world.pause()
        self._articulation_controller.apply_action(actions)
        return

    async def on_fill_bin_event_async(self):
        """Starts the bin filling simulation by registering physics callbacks and beginning world playback.

        Adds the physics step callback for bin filling and starts asynchronous world simulation.
        """
        world = self.get_world()
        world.add_physics_callback("sim_step", self._on_fill_bin_physics_step)
        await world.play_async()
        return

    async def setup_pre_reset(self):
        """Prepares the scene for reset by cleaning up physics callbacks and controller state.

        Removes the physics step callback if it exists, resets the controller, and clears the
        added cubes flag.
        """
        world = self.get_world()
        if world.physics_callback_exists("sim_step"):
            world.remove_physics_callback("sim_step")
        self._controller.reset()
        self._added_cubes = False
        return

    def world_cleanup(self):
        """Cleans up world resources by clearing controller references and resetting state flags."""
        self._controller = None
        self._added_cubes = False
        return
