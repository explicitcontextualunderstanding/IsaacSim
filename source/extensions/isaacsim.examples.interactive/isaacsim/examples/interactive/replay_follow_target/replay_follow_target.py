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

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
from isaacsim.core.experimental.materials import PreviewSurfaceMaterial
from isaacsim.core.experimental.objects import Cube, DistantLight
from isaacsim.core.experimental.prims import GeomPrim
from isaacsim.core.rendering_manager import ViewportManager
from isaacsim.core.simulation_manager import SimulationEvent, SimulationManager
from isaacsim.examples.base.base_sample_experimental import BaseSample
from isaacsim.robot.manipulators.examples.franka.franka_experimental import FrankaExperimental
from isaacsim.storage.native import get_assets_root_path


class ReplayFollowTarget(BaseSample):
    """Interactive sample that demonstrates replaying recorded robot trajectories and scene data.

    This class creates a simulation environment with a Franka robotic manipulator and a target cube,
    and provides functionality to replay previously recorded trajectory data. It supports two replay modes:
    trajectory-only replay (robot movements) and full scene replay (robot movements with target positions).

    The simulation environment includes:
    - A Franka robot manipulator positioned at /World/robot
    - A red target cube positioned at /World/TargetCube
    - A ground plane environment for physics simulation
    - Distant lighting for visualization

    During replay, the system loads data from a log file and steps through the recorded frames,
    applying joint positions to the robot and optionally moving the target cube to match the recorded
    scene state. The replay is synchronized with the physics simulation timeline.

    The class inherits from BaseSample and follows the standard Isaac Sim sample lifecycle with
    setup_scene, setup_post_load, setup_post_reset, and cleanup methods.
    """

    def __init__(self):
        super().__init__()
        self._robot = None
        self._target_cube = None
        self._robot_path = "/World/robot"
        self._target_path = "/World/TargetCube"
        self._data_logger = None
        self._current_time_step_index = 0
        self._physics_callback_id = None

    def setup_scene(self):
        """Set up the scene with Franka robot and target cube."""
        # Add ground plane environment for physics simulation
        stage = stage_utils.get_current_stage()
        ground_plane = stage_utils.add_reference_to_stage(
            usd_path=get_assets_root_path() + "/Isaac/Environments/Grid/default_environment.usd",
            path="/World/ground",
        )

        # Add distant light (only if it doesn't exist)
        if not stage.GetPrimAtPath("/World/DistantLight"):
            light = DistantLight("/World/DistantLight")
            light.set_intensities(300)

        # Create Franka robot (one line as requested)
        self._robot = FrankaExperimental(robot_path=self._robot_path, create_robot=True)

        # Create target cube
        target_position = [0.5, 0.0, 0.3]
        visual_material = PreviewSurfaceMaterial("/Visual_materials/target_red")
        visual_material.set_input_values("diffuseColor", [1.0, 0.0, 0.0])

        cube_shape = Cube(
            paths=self._target_path,
            positions=target_position,
            sizes=[1.0],
            scales=[0.03, 0.03, 0.03],
        )
        cube_shape.apply_visual_materials(visual_material)
        self._target_cube = GeomPrim(paths=cube_shape.paths)

        # Initialize data logger (using old API for now since replay depends on it)
        from isaacsim.core.api.loggers import DataLogger

        self._data_logger = DataLogger()

    async def setup_post_load(self):
        """Called after the scene is loaded."""
        # Set camera view
        ViewportManager.set_camera_view(eye=[1.5, 1.5, 1.5], target=[0.01, 0.01, 0.01], camera="/OmniverseKit_Persp")

        # Reset time step index
        self._current_time_step_index = 0

    async def setup_pre_reset(self):
        """Called before world reset."""
        # Deregister physics callbacks
        if self._physics_callback_id is not None:
            SimulationManager.deregister_callback(self._physics_callback_id)
            self._physics_callback_id = None

        # Reset time step index
        self._current_time_step_index = 0

    async def setup_post_reset(self):
        """Called after world reset."""
        # Reset robot to default pose
        if self._robot:
            self._robot.reset_to_default_pose()

        # Reset time step index
        self._current_time_step_index = 0

    async def setup_post_clear(self):
        """Called after clearing the scene."""
        # Deregister physics callbacks
        if self._physics_callback_id is not None:
            SimulationManager.deregister_callback(self._physics_callback_id)
            self._physics_callback_id = None

        self._robot = None
        self._target_cube = None
        self._data_logger = None
        self._current_time_step_index = 0

    def physics_cleanup(self):
        """Clean up physics resources."""
        if self._physics_callback_id is not None:
            SimulationManager.deregister_callback(self._physics_callback_id)
            self._physics_callback_id = None

    async def _on_replay_trajectory_event_async(self, data_file):
        """Load and replay trajectory data.

        Args:
            data_file: Path to the trajectory data file to load and replay.
        """
        self._data_logger.load(log_path=data_file)
        self._current_time_step_index = 0

        # Start timeline playback
        app_utils.play()
        await app_utils.update_app_async()

        # Register physics callback
        self._physics_callback_id = SimulationManager.register_callback(
            self._on_replay_trajectory_step, event=SimulationEvent.PHYSICS_POST_STEP
        )

    async def _on_replay_scene_event_async(self, data_file):
        """Load and replay scene data (robot + target).

        Args:
            data_file: Path to the scene data file to load and replay.
        """
        self._data_logger.load(log_path=data_file)
        self._current_time_step_index = 0

        # Start timeline playback
        app_utils.play()
        await app_utils.update_app_async()

        # Register physics callback
        self._physics_callback_id = SimulationManager.register_callback(
            self._on_replay_scene_step, event=SimulationEvent.PHYSICS_POST_STEP
        )

    def _on_replay_trajectory_step(self, dt, context):
        """Physics callback for replaying trajectory (robot only).

        Args:
            dt: Time delta for the physics step.
            context: Physics context for the callback.
        """
        if self._data_logger is None or self._robot is None:
            return

        if self._current_time_step_index < self._data_logger.get_num_of_data_frames():
            data_frame = self._data_logger.get_data_frame(data_frame_index=self._current_time_step_index)

            # Apply joint positions to robot using experimental API
            joint_positions = np.array(data_frame.data["applied_joint_positions"])
            if joint_positions.ndim == 1:
                joint_positions = joint_positions.reshape(1, -1)

            # Use set_dof_position_targets (experimental API)
            self._robot.set_dof_position_targets(joint_positions)

            self._current_time_step_index += 1
        else:
            # Replay complete, deregister callback
            if self._physics_callback_id is not None:
                SimulationManager.deregister_callback(self._physics_callback_id)
                self._physics_callback_id = None

    def _on_replay_scene_step(self, dt, context):
        """Physics callback for replaying scene (robot + target).

        Args:
            dt: Time delta for the physics step.
            context: Physics context for the callback.
        """
        if self._data_logger is None or self._robot is None or self._target_cube is None:
            return

        if self._current_time_step_index < self._data_logger.get_num_of_data_frames():
            data_frame = self._data_logger.get_data_frame(data_frame_index=self._current_time_step_index)

            # Apply joint positions to robot using experimental API
            joint_positions = np.array(data_frame.data["applied_joint_positions"])
            if joint_positions.ndim == 1:
                joint_positions = joint_positions.reshape(1, -1)

            # Use set_dof_position_targets (experimental API)
            self._robot.set_dof_position_targets(joint_positions)

            # Move target cube to recorded position
            target_position = np.array(data_frame.data["target_position"])
            if target_position.ndim == 1:
                target_position = target_position.reshape(1, -1)

            # Get current orientation (keep it unchanged)
            _, current_orientation = self._target_cube.get_world_poses()
            self._target_cube.set_world_poses(positions=target_position, orientations=current_orientation)

            self._current_time_step_index += 1
        else:
            # Replay complete, deregister callback
            if self._physics_callback_id is not None:
                SimulationManager.deregister_callback(self._physics_callback_id)
                self._physics_callback_id = None
