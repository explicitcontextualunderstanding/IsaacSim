# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


"""Provides pre-configured robot implementations for mobility generation scenarios in Isaac Sim."""


import math
import os
from typing import List, Tuple, Union

import numpy as np
import torch

# isaacsim.core.api
from isaacsim.core.api.robots.robot import Robot as _Robot

# isaacsim.core.prims
from isaacsim.core.prims import Articulation as _ArticulationView
from isaacsim.core.prims import SingleXFormPrim as XFormPrim
from isaacsim.core.utils.prims import get_prim_at_path

# isaacsim.core.utils
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.replicator.mobility_gen.impl.camera import MobilityGenCamera
from isaacsim.replicator.mobility_gen.impl.common import Buffer, Module

# isaacsim.replicator.mobility_gen.examples
from isaacsim.replicator.mobility_gen.impl.robot import ROBOTS, MobilityGenRobot
from isaacsim.replicator.mobility_gen.impl.types import Pose2d
from isaacsim.replicator.mobility_gen.impl.utils.global_utils import get_world, join_sdf_paths
from isaacsim.replicator.mobility_gen.impl.utils.prim_utils import (
    prim_rotate_x,
    prim_rotate_y,
    prim_rotate_z,
    prim_translate,
)
from isaacsim.replicator.mobility_gen.impl.utils.registry import Registry
from isaacsim.replicator.mobility_gen.impl.utils.stage_utils import stage_add_camera
from isaacsim.robot.policy.examples.robots import SpotFlatTerrainPolicy

# isaacsim.robot.policy.examples
from isaacsim.robot.policy.examples.robots.h1 import H1FlatTerrainPolicy
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController

# isaacsim.robot.wheeled_robots
from isaacsim.robot.wheeled_robots.robots import WheeledRobot as _WheeledRobot
from isaacsim.storage.native import get_assets_root_path

# this package
from .misc import HawkCamera


class WheeledMobilityGenRobot(MobilityGenRobot):
    """A wheeled robot implementation for mobility generation scenarios.

    This class extends MobilityGenRobot to provide specific functionality for wheeled robots using differential
    steering control. It integrates the robot's physical representation, articulation system, and control logic
    to enable autonomous navigation and movement in simulation environments.

    The class manages the robot's wheel actions through a DifferentialController, allowing for precise control
    of linear and angular velocities. It supports optional front camera integration for visual perception
    capabilities during navigation tasks.

    Args:
        prim_path: USD prim path where the robot is located in the stage.
        robot: The wheeled robot instance providing the physical robot interface.
        articulation_view: Articulation view for managing the robot's joint system.
        controller: Differential controller for wheel-based movement control.
        front_camera: Optional camera module for front-facing visual perception.
    """

    # Wheeled robot parameters
    wheel_dof_names: List[str]
    usd_url: str
    chassis_subpath: str
    wheel_radius: float
    wheel_base: float

    def __init__(
        self,
        prim_path: str,
        robot: _WheeledRobot,
        articulation_view: _ArticulationView,
        controller: DifferentialController,
        front_camera: Module | None = None,
    ):
        super().__init__(
            prim_path=prim_path, robot=robot, articulation_view=articulation_view, front_camera=front_camera
        )
        self.controller = controller
        self.robot = robot

    @classmethod
    def build(cls, prim_path: str) -> "WheeledRobot":
        """Creates and configures a wheeled robot instance with necessary components.

        Constructs a wheeled robot by creating the robot prim, articulation view, differential controller,
        and front camera, then adds them to the world scene.

        Args:
            prim_path: USD prim path where the robot will be created.

        Returns:
            The configured wheeled robot instance.
        """

        world = get_world()

        robot = world.scene.add(
            _WheeledRobot(prim_path, wheel_dof_names=cls.wheel_dof_names, create_robot=True, usd_path=cls.usd_url)
        )

        view = _ArticulationView(join_sdf_paths(prim_path, cls.chassis_subpath))

        world.scene.add(view)

        controller = DifferentialController(name="controller", wheel_radius=cls.wheel_radius, wheel_base=cls.wheel_base)

        camera = cls.build_front_camera(prim_path)

        return cls(prim_path=prim_path, robot=robot, articulation_view=view, controller=controller, front_camera=camera)

    def write_action(self, step_size: float):
        """Applies wheel actions to the robot based on current controller commands.

        Uses the differential controller to convert the stored action into wheel commands
        and applies them to the robot's wheels.

        Args:
            step_size: Time step size for the action application.
        """
        self.robot.apply_wheel_actions(self.controller.forward(command=self.action.get_value()))


class PolicyMobilityGenRobot(MobilityGenRobot):
    """Base class for policy-driven mobility generation robots.

    This class provides a foundation for robots that use reinforcement learning policies for locomotion control,
    such as humanoid and quadruped robots. It integrates with Isaac Sim's articulation system and supports
    various robot policies for terrain navigation and movement control.

    The class handles the setup and management of policy-controlled robots within the mobility generation
    framework, including articulation management, policy integration, and camera attachment for perception.

    Args:
        prim_path: USD path where the robot prim is located in the stage.
        robot: The robot instance that provides the physical representation and control interface.
        articulation_view: View object for managing the robot's articulated joints and degrees of freedom.
        controller: Policy controller that determines robot movement and behavior. Supports both H1 humanoid
            and Spot quadruped terrain navigation policies.
        front_camera: Optional camera module attached to the robot for perception and data collection.
    """

    usd_url: str
    articulation_path: str

    def __init__(
        self,
        prim_path: str,
        robot: _Robot,
        articulation_view: _ArticulationView,
        controller: Union[H1FlatTerrainPolicy, SpotFlatTerrainPolicy],
        front_camera: Module | None = None,
    ):
        super().__init__(prim_path, robot, articulation_view, front_camera)
        self.controller = controller

    @classmethod
    def build_policy(cls, prim_path: str):
        """Builds the policy controller for the robot.

        Args:
            prim_path: USD prim path for the robot.
        """
        raise NotImplementedError

    @classmethod
    def build(cls, prim_path: str):
        """Creates and configures a policy-controlled mobility generation robot instance.

        Args:
            prim_path: USD prim path for the robot.

        Returns:
            The configured robot instance.
        """
        stage = get_current_stage()
        world = get_world()

        add_reference_to_stage(usd_path=cls.usd_url, prim_path=prim_path)

        robot = _Robot(prim_path=prim_path)

        world.scene.add(robot)

        # Articulation
        view = _ArticulationView(join_sdf_paths(prim_path, cls.articulation_path))

        world.scene.add(view)

        # Controller
        controller = cls.build_policy(prim_path)

        prim = get_prim_at_path(prim_path)
        prim_translate(prim, (0, 0, cls.z_offset))

        camera = cls.build_front_camera(prim_path)

        return cls(prim_path=prim_path, robot=robot, articulation_view=view, controller=controller, front_camera=camera)

    def write_action(self, step_size: float):
        """Applies the current action to the robot using the policy controller.

        Args:
            step_size: Time step size for the action.
        """
        action = self.action.get_value()
        command = np.array([action[0], 0.0, action[1]])
        command = torch.from_numpy(command)
        self.controller.forward(step_size, command)

    def set_pose_2d(self, pose: Pose2d):
        """Sets the robot's 2D pose and reinitializes the controller.

        Args:
            pose: The 2D pose to set for the robot.
        """
        super().set_pose_2d(pose)
        self.controller.initialize()


@ROBOTS.register()
class JetbotRobot(WheeledMobilityGenRobot):
    """A wheeled robot implementation for the Jetbot platform in mobility generation scenarios.

    This class provides a complete implementation of the NVIDIA Jetbot robot for use in mobility generation
    workflows. It inherits from WheeledMobilityGenRobot and includes pre-configured parameters for physics
    simulation, camera setup, occupancy mapping, and control behaviors.

    The robot features differential drive control with two wheels, a front-facing Hawk camera for perception,
    and support for various control modes including keyboard, gamepad, random actions, and path following.
    It includes built-in collision detection and occupancy mapping capabilities optimized for the Jetbot's
    compact form factor.

    Key characteristics:
    - Wheel base: 0.1125 meters
    - Wheel radius: 0.03 meters
    - Physics timestep: 0.005 seconds
    - Occupancy map radius: 0.25 meters
    - Maximum linear velocity: 0.25 m/s (keyboard/gamepad)
    - Maximum angular velocity: 1.0 rad/s (keyboard/gamepad)

    The robot automatically configures its USD asset path, joint names, camera positioning, and control
    parameters. It supports chase camera functionality with configurable offset and tilt angle for
    visualization purposes.
    """

    physics_dt: float = 0.005
    """Time step for physics simulation in seconds."""

    z_offset: float = 0.1
    """Vertical offset applied to the robot's initial position."""

    chase_camera_base_path = "chassis"
    """Base path for the chase camera attachment point."""
    chase_camera_x_offset: float = -0.5
    """X-axis offset for the chase camera position relative to the base path."""
    chase_camera_z_offset: float = 0.5
    """Z-axis offset for the chase camera position relative to the base path."""
    chase_camera_tilt_angle: float = 60.0
    """Tilt angle for the chase camera in degrees."""

    occupancy_map_radius: float = 0.25
    """Radius for occupancy map detection around the robot."""
    occupancy_map_z_min: float = 0.05
    """Minimum height for occupancy map detection."""
    occupancy_map_z_max: float = 0.5
    """Maximum height for occupancy map detection."""
    occupancy_map_cell_size: float = 0.05
    """Cell size for the occupancy map grid."""
    occupancy_map_collision_radius: float = 0.25
    """Collision radius used for occupancy map calculations."""

    front_camera_base_path = "chassis/rgb_camera/front_hawk"
    """Path to the front camera's base attachment point."""
    front_camera_rotation = (0.0, 0.0, 0.0)
    """Rotation values (x, y, z) for the front camera orientation."""
    front_camera_translation = (0.0, 0.0, 0.0)
    """Translation values (x, y, z) for the front camera position."""
    front_camera_type = HawkCamera

    keyboard_linear_velocity_gain: float = 0.25
    """Gain multiplier for linear velocity when using keyboard controls."""
    keyboard_angular_velocity_gain: float = 1.0
    """Gain multiplier for angular velocity when using keyboard controls."""

    gamepad_linear_velocity_gain: float = 0.25
    """Gain multiplier for linear velocity when using gamepad controls."""
    gamepad_angular_velocity_gain: float = 1.0
    """Gain multiplier for angular velocity when using gamepad controls."""

    random_action_linear_velocity_range: Tuple[float, float] = (-0.3, 0.25)
    """Range (min, max) for random linear velocity actions."""
    random_action_angular_velocity_range: Tuple[float, float] = (-0.75, 0.75)
    """Range (min, max) for random angular velocity actions."""
    random_action_linear_acceleration_std: float = 1.0
    """Standard deviation for random linear acceleration noise."""
    random_action_angular_acceleration_std: float = 5.0
    """Standard deviation for random angular acceleration noise."""
    random_action_grid_pose_sampler_grid_size: float = 5.0
    """Grid size for random pose sampling."""

    path_following_speed: float = 0.25
    """Target speed for path following behavior."""
    path_following_angular_gain: float = 1.0
    """Gain for angular corrections during path following."""
    path_following_stop_distance_threshold: float = 0.5
    """Distance threshold for stopping during path following."""
    path_following_forward_angle_threshold = math.pi / 4
    """Angle threshold for forward direction during path following."""
    path_following_target_point_offset_meters: float = 1.0
    """Distance offset for target point selection during path following."""

    wheel_dof_names: List[str] = ["left_wheel_joint", "right_wheel_joint"]
    """Names of the wheel degree-of-freedom joints."""
    usd_url: str = get_assets_root_path() + "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd"
    """URL path to the USD file containing the robot's 3D model."""

    chassis_subpath: str = "chassis"
    """Subpath to the chassis component within the robot's USD structure."""
    wheel_base: float = 0.1125
    """Distance between the front and rear axles of the robot."""
    wheel_radius: float = 0.03
    """Radius of the robot's wheels."""


@ROBOTS.register()
class CarterRobot(WheeledMobilityGenRobot):
    """A mobility generation robot implementation for the NVIDIA Nova Carter wheeled robot.

    This class provides a specialized configuration of the Nova Carter robot for mobility generation
    scenarios. It inherits from WheeledMobilityGenRobot and includes pre-configured parameters for
    physics simulation, camera setup, occupancy mapping, and various control modes including keyboard,
    gamepad, random actions, and path following.

    The robot features differential drive control with a wheelbase of 0.413 meters and wheel radius
    of 0.14 meters. It includes a front-facing Hawk camera for perception and supports chase camera
    views for visualization. The configuration includes collision detection parameters and occupancy
    mapping capabilities for navigation planning.

    Key features include support for multiple input modalities (keyboard, gamepad), autonomous
    behaviors (random exploration, path following), and configurable velocity limits and acceleration
    parameters for different operational scenarios.
    """

    physics_dt: float = 0.005
    """Physics simulation timestep in seconds."""

    z_offset: float = 0.25
    """Vertical offset applied to the robot's initial position."""

    chase_camera_base_path = "chassis_link"
    """Base path for the chase camera attachment point."""
    chase_camera_x_offset: float = -1.5
    """X-axis offset for the chase camera position."""
    chase_camera_z_offset: float = 0.8
    """Z-axis offset for the chase camera position."""
    chase_camera_tilt_angle: float = 60.0
    """Tilt angle for the chase camera in degrees."""

    occupancy_map_radius: float = 0.55
    """Radius for occupancy map generation around the robot."""
    occupancy_map_z_min: float = 0.1
    """Minimum Z height for occupancy map detection."""
    occupancy_map_z_max: float = 0.62
    """Maximum Z height for occupancy map detection."""
    occupancy_map_cell_size: float = 0.05
    """Cell size for the occupancy map grid."""
    occupancy_map_collision_radius: float = 0.5
    """Collision radius used for occupancy map calculations."""

    front_camera_base_path = "chassis_link/sensors/front_hawk/front_hawk"
    """Base path for the front camera attachment point."""
    front_camera_rotation = (0.0, 0.0, 0.0)
    """Rotation values for the front camera in degrees (x, y, z)."""
    front_camera_translation = (0.0, 0.0, 0.0)
    """Translation offset for the front camera position (x, y, z)."""
    front_camera_type = HawkCamera

    keyboard_linear_velocity_gain: float = 1.0
    """Gain multiplier for keyboard linear velocity input."""
    keyboard_angular_velocity_gain: float = 1.0
    """Gain multiplier for keyboard angular velocity input."""

    gamepad_linear_velocity_gain: float = 1.0
    """Gain multiplier for gamepad linear velocity input."""
    gamepad_angular_velocity_gain: float = 1.0
    """Gain multiplier for gamepad angular velocity input."""

    random_action_linear_velocity_range: Tuple[float, float] = (-0.3, 1.0)
    """Range of linear velocities for random action generation (min, max)."""
    random_action_angular_velocity_range: Tuple[float, float] = (-0.75, 0.75)
    """Range of angular velocities for random action generation (min, max)."""
    random_action_linear_acceleration_std: float = 5.0
    """Standard deviation for linear acceleration noise in random actions."""
    random_action_angular_acceleration_std: float = 5.0
    """Standard deviation for angular acceleration noise in random actions."""
    random_action_grid_pose_sampler_grid_size: float = 5.0
    """Grid size for random pose sampling."""

    path_following_speed: float = 1.0
    """Target speed for path following behavior."""
    path_following_angular_gain: float = 1.0
    """Angular gain for path following control."""
    path_following_stop_distance_threshold: float = 0.5
    """Distance threshold for stopping during path following."""
    path_following_forward_angle_threshold = math.pi / 4
    """Angle threshold for forward direction during path following."""
    path_following_target_point_offset_meters: float = 1.0
    """Offset distance for target point selection in path following."""

    wheel_dof_names: List[str] = ["joint_wheel_left", "joint_wheel_right"]
    """Names of the wheel degree-of-freedom joints."""
    usd_url: str = get_assets_root_path() + "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd"
    """USD file path for the robot asset."""
    chassis_subpath: str = "chassis_link"
    """Subpath to the chassis component within the robot USD."""
    wheel_base = 0.413
    """Distance between the robot's wheel axles."""
    wheel_radius = 0.14
    """Radius of the robot's wheels."""


@ROBOTS.register()
class H1Robot(PolicyMobilityGenRobot):
    """A humanoid robot implementation for mobility generation and navigation tasks.

    This class represents the Unitree H1 humanoid robot configured for autonomous navigation and mobility generation scenarios. It inherits from PolicyMobilityGenRobot and uses the H1FlatTerrainPolicy for locomotion control. The robot is equipped with front-facing cameras, occupancy mapping capabilities, and supports multiple control modes including keyboard, gamepad, random actions, and path following.

    The H1Robot features a bipedal locomotion system with sophisticated terrain navigation capabilities. It includes configurable camera systems for visual navigation, occupancy mapping for obstacle avoidance, and various control parameters for different operational modes. The robot can operate in environments with varying terrain conditions and supports both manual and autonomous navigation modes.
    """

    physics_dt: float = 0.005
    """Physics simulation timestep in seconds."""

    z_offset: float = 1.05
    """Vertical offset applied to the robot's initial position."""

    chase_camera_base_path = "pelvis"
    """Base path for attaching the chase camera to the robot."""
    chase_camera_x_offset: float = -1.5
    """Horizontal distance of the chase camera behind the robot."""
    chase_camera_z_offset: float = 0.8
    """Vertical distance of the chase camera above the robot."""
    chase_camera_tilt_angle: float = 60.0
    """Downward tilt angle of the chase camera in degrees."""

    occupancy_map_radius: float = 1.0
    """Radius around the robot for occupancy map generation."""
    occupancy_map_z_min: float = 0.1
    """Minimum height for occupancy map collision detection."""
    occupancy_map_z_max: float = 2.0
    """Maximum height for occupancy map collision detection."""
    occupancy_map_cell_size: float = 0.05
    """Grid cell size for occupancy map discretization."""
    occupancy_map_collision_radius: float = 0.5
    """Collision radius used for occupancy map calculations."""

    front_camera_base_path = "d435_left_imager_link/front_camera/front"
    """Base path for attaching the front camera to the robot."""
    front_camera_rotation = (0.0, 250.0, 90.0)
    """Rotation angles (x, y, z) applied to the front camera."""
    front_camera_translation = (-0.06, 0.0, 0.0)
    """Translation offset (x, y, z) applied to the front camera."""
    front_camera_type = HawkCamera

    keyboard_linear_velocity_gain: float = 1.0
    """Multiplier for keyboard input linear velocity commands."""
    keyboard_angular_velocity_gain: float = 1.0
    """Multiplier for keyboard input angular velocity commands."""

    gamepad_linear_velocity_gain: float = 1.0
    """Multiplier for gamepad input linear velocity commands."""
    gamepad_angular_velocity_gain: float = 1.0
    """Multiplier for gamepad input angular velocity commands."""

    random_action_linear_velocity_range: Tuple[float, float] = (-0.3, 1.0)
    """Range (min, max) for random linear velocity actions."""
    random_action_angular_velocity_range: Tuple[float, float] = (-0.75, 0.75)
    """Range (min, max) for random angular velocity actions."""
    random_action_linear_acceleration_std: float = 5.0
    """Standard deviation for random linear acceleration noise."""
    random_action_angular_acceleration_std: float = 5.0
    """Standard deviation for random angular acceleration noise."""
    random_action_grid_pose_sampler_grid_size: float = 5.0
    """Grid size for random pose sampling."""

    path_following_speed: float = 1.0
    """Target linear velocity for path following behavior."""
    path_following_angular_gain: float = 1.0
    """Proportional gain for angular control during path following."""
    path_following_stop_distance_threshold: float = 0.5
    """Distance threshold for stopping near path targets."""
    path_following_forward_angle_threshold = math.pi / 4
    """Angle threshold for determining forward motion direction."""
    path_following_target_point_offset_meters: float = 1.0
    """Distance ahead on the path to target for following."""

    usd_url = get_assets_root_path() + "/Isaac/Robots/Unitree/H1/h1.usd"
    """File path to the robot's USD asset."""
    articulation_path = "pelvis"
    """Path to the articulation root within the robot hierarchy."""
    controller_z_offset: float = 1.05
    """Vertical offset applied to the robot controller's position."""

    @classmethod
    def build_policy(cls, prim_path: str):
        """Creates and configures a flat terrain policy for the H1 robot.

        Args:
            prim_path: USD prim path where the robot is located.

        Returns:
            Configured H1FlatTerrainPolicy instance for robot control.
        """
        return H1FlatTerrainPolicy(prim_path=prim_path, position=np.array([0.0, 0.0, cls.controller_z_offset]))


@ROBOTS.register()
class SpotRobot(PolicyMobilityGenRobot):
    """Boston Dynamics Spot quadruped robot for mobility generation in Isaac Sim.

    This class provides a complete implementation of the Spot robot with legged locomotion capabilities
    for use in robotic mobility scenarios. The robot uses a policy-based controller for navigation and
    features configurable parameters for physics simulation, camera systems, and motion control.

    The robot is equipped with a front-facing camera for perception, chase camera for visualization,
    and occupancy mapping capabilities for navigation. It supports multiple control modes including
    keyboard, gamepad, random actions, and path following behaviors.
    """

    physics_dt: float = 0.005
    """Physics simulation time step in seconds."""
    z_offset: float = 0.7
    """Vertical offset applied to the robot's position in meters."""

    chase_camera_base_path = "body"
    """Base prim path for positioning the chase camera relative to the robot."""
    chase_camera_x_offset: float = -1.5
    """Forward/backward offset of the chase camera from the base position in meters."""
    chase_camera_z_offset: float = 0.8
    """Vertical offset of the chase camera from the base position in meters."""
    chase_camera_tilt_angle: float = 60.0
    """Downward tilt angle of the chase camera in degrees."""

    occupancy_map_radius: float = 1.0
    """Radius around the robot for occupancy map generation in meters."""
    occupancy_map_z_min: float = 0.1
    """Minimum height for occupancy map obstacle detection in meters."""
    occupancy_map_z_max: float = 0.62
    """Maximum height for occupancy map obstacle detection in meters."""
    occupancy_map_cell_size: float = 0.05
    """Size of each cell in the occupancy map grid in meters."""
    occupancy_map_collision_radius: float = 0.5
    """Collision radius used for occupancy map calculations in meters."""

    front_camera_base_path = "body/front_camera"
    """Base prim path for positioning the front camera relative to the robot."""
    front_camera_rotation = (180, 180, 180)
    """Rotation angles (x, y, z) applied to the front camera in degrees."""
    front_camera_translation = (0.44, 0.075, 0.01)
    """Translation offset (x, y, z) applied to the front camera in meters."""
    front_camera_type = HawkCamera

    keyboard_linear_velocity_gain: float = 1.0
    """Scaling factor for linear velocity commands from keyboard input."""
    keyboard_angular_velocity_gain: float = 1.0
    """Scaling factor for angular velocity commands from keyboard input."""

    gamepad_linear_velocity_gain: float = 1.0
    """Scaling factor for linear velocity commands from gamepad input."""
    gamepad_angular_velocity_gain: float = 1.0
    """Scaling factor for angular velocity commands from gamepad input."""

    random_action_linear_velocity_range: Tuple[float, float] = (-0.3, 1.0)
    """Range (min, max) for random linear velocity commands in meters per second."""
    random_action_angular_velocity_range: Tuple[float, float] = (-0.75, 0.75)
    """Range (min, max) for random angular velocity commands in radians per second."""
    random_action_linear_acceleration_std: float = 5.0
    """Standard deviation for random linear acceleration noise."""
    random_action_angular_acceleration_std: float = 5.0
    """Standard deviation for random angular acceleration noise."""
    random_action_grid_pose_sampler_grid_size: float = 5.0
    """Grid size for random pose sampling in meters."""

    path_following_speed: float = 1.0
    """Target linear speed for path following mode in meters per second."""
    path_following_angular_gain: float = 1.0
    """Proportional gain for angular velocity control during path following."""
    path_following_stop_distance_threshold: float = 0.5
    """Distance threshold for stopping at path waypoints in meters."""
    path_following_forward_angle_threshold = math.pi / 4
    """Angular threshold for considering a target point as forward-facing in radians."""
    path_following_target_point_offset_meters: float = 1.0
    """Look-ahead distance for selecting target points along the path in meters."""

    usd_url = get_assets_root_path() + "/Isaac/Robots/BostonDynamics/spot/spot.usd"
    """File path to the USD asset for the Spot robot model."""
    articulation_path = "/"
    """Relative path to the articulation root within the robot's prim hierarchy."""
    controller_z_offset: float = 0.7
    """Vertical offset for the controller's reference position in meters."""

    @classmethod
    def build_policy(cls, prim_path: str) -> SpotFlatTerrainPolicy:
        """Creates and initializes a SpotFlatTerrainPolicy controller for the Spot robot.

        Args:
            prim_path: USD prim path where the Spot robot is located in the stage.

        Returns:
            Configured SpotFlatTerrainPolicy instance with the robot's position and z-offset.
        """
        return SpotFlatTerrainPolicy(prim_path=prim_path, position=np.array([0.0, 0.0, cls.controller_z_offset]))
