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


"""Provides pre-built scenario implementations for robot mobility generation and control."""


from typing import Tuple

import numpy as np
import PIL.Image
from isaacsim.replicator.mobility_gen.impl.common import Buffer, Module
from isaacsim.replicator.mobility_gen.impl.inputs import Gamepad, Keyboard
from isaacsim.replicator.mobility_gen.impl.occupancy_map import OccupancyMap
from isaacsim.replicator.mobility_gen.impl.path_planner import compress_path, generate_paths
from isaacsim.replicator.mobility_gen.impl.pose_samplers import GridPoseSampler, UniformPoseSampler
from isaacsim.replicator.mobility_gen.impl.robot import MobilityGenRobot

# isaacsim.replicator.mobility_gen.examples
from isaacsim.replicator.mobility_gen.impl.scenario import SCENARIOS, MobilityGenScenario
from isaacsim.replicator.mobility_gen.impl.utils.path_utils import PathHelper
from isaacsim.replicator.mobility_gen.impl.utils.registry import Registry
from PIL import Image, ImageDraw


@SCENARIOS.register()
class KeyboardTeleoperationScenario(MobilityGenScenario):
    """A teleoperation scenario that enables manual control of a robot using keyboard input.

    This scenario allows users to control a robot's movement through keyboard commands, where WASD keys control
    the robot's linear and angular velocities. The robot can be moved forward/backward and rotated left/right
    based on user input. The scenario handles real-time keyboard input processing and applies the corresponding
    velocity commands to the robot.

    The robot is initially positioned at a random valid location within the occupancy map's free space. During
    each simulation step, the scenario reads keyboard input and converts it to velocity commands that are applied
    to the robot.

    Args:
        robot: The robot instance to be controlled via keyboard input.
        occupancy_map: The occupancy map defining the navigable environment and obstacles.
    """

    def __init__(self, robot: MobilityGenRobot, occupancy_map: OccupancyMap):
        super().__init__(robot, occupancy_map)
        self.keyboard = Keyboard()
        self.pose_sampler = UniformPoseSampler()

    def reset(self):
        """Resets the scenario by placing the robot at a random pose and updating the state.

        Samples a new pose using the uniform pose sampler and sets the robot to that position.
        """
        pose = self.pose_sampler.sample(self.buffered_occupancy_map)
        self.robot.set_pose_2d(pose)
        self.update_state()

    def step(self, step_size):
        """Executes one step of keyboard teleoperation control.

        Reads keyboard input for WASD keys to control robot movement, calculates linear and angular velocities based on the input, and applies the action to the robot.

        Args:
            step_size: Time step for the simulation step.

        Returns:
            Always returns True indicating the scenario continues running.
        """

        self.update_state()

        buttons = self.keyboard.buttons.get_value()

        w_val = float(buttons[0])
        a_val = float(buttons[1])
        s_val = float(buttons[2])
        d_val = float(buttons[3])

        linear_velocity = (w_val - s_val) * self.robot.keyboard_linear_velocity_gain
        angular_velocity = (a_val - d_val) * self.robot.keyboard_angular_velocity_gain

        self.robot.action.set_value(np.array([linear_velocity, angular_velocity]))

        self.robot.write_action(step_size)

        self.update_state()

        return True


@SCENARIOS.register()
class GamepadTeleoperationScenario(MobilityGenScenario):
    """A mobility generation scenario that enables gamepad-based teleoperation of a robot.

    This scenario allows direct control of a robot using gamepad input, where the left analog stick controls
    linear velocity and the right analog stick controls angular velocity. The robot is initially positioned
    at a random valid location within the occupancy map and can be controlled in real-time through gamepad
    input.

    The scenario continuously reads gamepad axes values and translates them into robot movement commands.
    The linear velocity is controlled by the first axis (typically left stick vertical) multiplied by the
    robot's gamepad linear velocity gain, while angular velocity is controlled by the fourth axis (typically
    right stick horizontal) multiplied by the robot's gamepad angular velocity gain.

    This scenario is useful for manual robot control, data collection, testing robot behavior in specific
    situations, or providing human-in-the-loop control capabilities.

    Args:
        robot: The robot instance to be controlled via gamepad input.
        occupancy_map: The occupancy map defining the environment boundaries and obstacles.
    """

    def __init__(self, robot: MobilityGenRobot, occupancy_map: OccupancyMap):
        super().__init__(robot, occupancy_map)
        self.gamepad = Gamepad()
        self.pose_sampler = UniformPoseSampler()

    def reset(self):
        """Resets the scenario by randomly placing the robot and updating its state.

        Samples a random pose from free space and sets the robot to that position.
        """
        pose = self.pose_sampler.sample(self.buffered_occupancy_map)
        self.robot.set_pose_2d(pose)
        self.update_state()

    def step(self, step_size: float) -> bool:
        """Executes one simulation step using gamepad input to control the robot.

        Reads gamepad axes values, converts them to linear and angular velocities based on
        configured gains, and applies the resulting action to the robot.

        Args:
            step_size: Time step duration for the simulation.

        Returns:
            Always returns True indicating the scenario continues running.
        """

        self.gamepad.update_state()

        axes = self.gamepad.axes.get_value()
        linear_velocity = axes[0] * self.robot.gamepad_linear_velocity_gain
        angular_velocity = axes[3] * self.robot.gamepad_angular_velocity_gain

        self.robot.action.set_value(np.array([linear_velocity, angular_velocity]))
        self.robot.write_action(step_size)

        self.update_state()

        return True


@SCENARIOS.register()
class RandomAccelerationScenario(MobilityGenScenario):
    """A mobility scenario that generates random robot motion using acceleration-based control.

    This scenario drives the robot using random accelerations applied to both linear and angular velocities.
    The robot starts at a random pose and continuously applies random acceleration changes to its current
    velocities, creating unpredictable movement patterns. The scenario terminates when the robot goes out
    of bounds or collides with obstacles in the occupancy map.

    The random accelerations are sampled from normal distributions with configurable standard deviations,
    and the resulting velocities are clamped to specified ranges to ensure realistic robot behavior.
    This scenario is useful for generating diverse training data or testing robot behavior under
    unpredictable motion conditions.

    Args:
        robot: The mobility generation robot to control.
        occupancy_map: The occupancy map defining the environment boundaries and obstacles.
    """

    def __init__(self, robot: MobilityGenRobot, occupancy_map: OccupancyMap):
        super().__init__(robot, occupancy_map)
        self.pose_sampler = GridPoseSampler(robot.random_action_grid_pose_sampler_grid_size)
        self.is_alive = True
        self.collision_occupancy_map = occupancy_map.buffered(robot.occupancy_map_collision_radius)

    def reset(self):
        """Resets the robot to a new random pose and initializes the scenario state.

        Sets the robot's action to zero velocity, samples a new pose from the grid pose sampler,
        positions the robot at that pose, and marks the robot as alive.
        """
        self.robot.action.set_value(np.zeros(2))
        pose = self.pose_sampler.sample(self.buffered_occupancy_map)
        self.robot.set_pose_2d(pose)
        self.is_alive = True
        self.update_state()

    def step(self, step_size: float) -> bool:
        """Executes one simulation step with random acceleration applied to the robot.

        Updates the robot's velocity by adding random acceleration noise to the current action,
        clamps the velocities within configured ranges, and checks for collisions or out-of-bounds conditions.

        Args:
            step_size: Time step duration for the simulation step.

        Returns:
            True if the robot is still alive (no collision or out-of-bounds), False otherwise.
        """

        self.update_state()

        current_action = self.robot.action.get_value()

        linear_velocity = (
            current_action[0] + step_size * np.random.randn(1) * self.robot.random_action_linear_acceleration_std
        )
        angular_velocity = (
            current_action[1] + step_size * np.random.randn(1) * self.robot.random_action_angular_acceleration_std
        )

        linear_velocity = np.clip(linear_velocity, *self.robot.random_action_linear_velocity_range)[0]
        angular_velocity = np.clip(angular_velocity, *self.robot.random_action_angular_velocity_range)[0]

        self.robot.action.set_value(np.array([linear_velocity, angular_velocity]))
        self.robot.write_action(step_size)

        self.update_state()

        # Check out of bounds or collision
        pose = self.robot.get_pose_2d()
        if not self.collision_occupancy_map.check_world_point_in_bounds(pose):
            self.is_alive = False
        elif not self.collision_occupancy_map.check_world_point_in_freespace(pose):
            self.is_alive = False

        return self.is_alive


@SCENARIOS.register()
class RandomPathFollowingScenario(MobilityGenScenario):
    """A scenario that generates random paths and controls a robot to follow them using path-following algorithms.

    This scenario creates a complete autonomous navigation system where the robot navigates to randomly generated
    targets within the environment. The system handles path planning, obstacle avoidance, and robot control to
    ensure safe and efficient navigation.

    The scenario operates by sampling random target locations within the free space of the occupancy map,
    generating collision-free paths using path planning algorithms, and controlling the robot to follow these
    paths using proportional control for steering and speed regulation.

    Path following uses a look-ahead controller that selects target points ahead of the robot's current position
    along the planned path. The robot adjusts its linear and angular velocities based on the angle deviation
    between its current heading and the direction to the target point. When the angle deviation exceeds a
    threshold, the robot stops moving forward and focuses on rotating to align with the path.

    The scenario automatically terminates when the robot reaches the target destination within a specified
    distance threshold or encounters collision conditions. Safety is maintained through continuous collision
    checking against the buffered occupancy map that accounts for the robot's physical dimensions.

    Visualization capabilities include rendering the planned path on the occupancy map image, showing the
    intended route in green overlay for debugging and monitoring purposes.

    Args:
        robot: The mobility generation robot that will follow the generated paths. Must be configured with
            path-following parameters including target point offset, speed settings, angular gains, and collision
            detection radius.
        occupancy_map: The occupancy map representing the environment's free space and obstacles. Used for path
            planning, collision detection, and pose sampling within navigable areas.
    """

    def __init__(self, robot: MobilityGenRobot, occupancy_map: OccupancyMap):
        super().__init__(robot, occupancy_map)
        self.pose_sampler = UniformPoseSampler()
        self.is_alive = True
        self.target_path = Buffer()
        self.collision_occupancy_map = occupancy_map.buffered(robot.occupancy_map_collision_radius)

    def _vector_angle(self, w: np.ndarray, v: np.ndarray) -> float:
        """Calculates the angle between two 2D vectors.

        Args:
            w: First 2D vector.
            v: Second 2D vector.

        Returns:
            The angle between the vectors in radians.
        """
        delta_angle = np.arctan2(w[1] * v[0] - w[0] * v[1], w[0] * v[0] + w[1] * v[1])
        return delta_angle

    def set_random_target_path(self):
        """Generates a random target path from the robot's current position to a random freespace endpoint."""
        current_pose = self.robot.get_pose_2d()

        start_px = self.occupancy_map.world_to_pixel_numpy(np.array([[current_pose.x, current_pose.y]]))
        freespace = self.buffered_occupancy_map.freespace_mask()

        start = (start_px[0, 1], start_px[0, 0])

        output = generate_paths(start, freespace)
        end = output.sample_random_end_point()
        path = output.unroll_path(end)
        path, _ = compress_path(path)  # remove redundant points
        path = path[:, ::-1]  # y,x -> x,y coordinates
        path = self.occupancy_map.pixel_to_world_numpy(path)
        self.target_path.set_value(path)
        self.target_path_helper = PathHelper(path)

    def reset(self):
        """Resets the scenario by placing the robot at a random pose and generating a new target path."""
        self.robot.action.set_value(np.zeros(2))
        pose = self.pose_sampler.sample(self.buffered_occupancy_map)
        self.robot.set_pose_2d(pose)
        self.set_random_target_path()
        self.is_alive = True
        self.update_state()

    def step(self, step_size: float) -> bool:
        """Executes one simulation step by updating robot controls to follow the target path.

        The robot follows the path using proportional control for steering and stops when reaching
        the target or encountering collisions.

        Args:
            step_size: Time step size for the simulation.

        Returns:
            True if the robot is still alive (no collision, in bounds, not at target), False otherwise.
        """

        self.update_state()
        target_path = self.target_path.get_value()
        current_pose = self.robot.get_pose_2d()

        if not self.collision_occupancy_map.check_world_point_in_bounds(current_pose):
            self.is_alive = False
            return self.is_alive
        elif not self.collision_occupancy_map.check_world_point_in_freespace(current_pose):
            self.is_alive = False
            return self.is_alive

        pt_robot = np.array([current_pose.x, current_pose.y])
        pt_path, pt_path_length, _, _ = self.target_path_helper.find_nearest(pt_robot)
        pt_target = self.target_path_helper.get_point_by_distance(
            distance=pt_path_length + self.robot.path_following_target_point_offset_meters
        )

        path_end = target_path[-1]
        dist_to_target = np.sqrt(np.sum((pt_robot - path_end) ** 2))

        if dist_to_target < self.robot.path_following_stop_distance_threshold:
            self.is_alive = False
        else:
            vec_robot_unit = np.array([np.cos(current_pose.theta), np.sin(current_pose.theta)])
            vec_target = pt_target - pt_robot
            vec_target_unit = vec_target / np.sqrt(np.sum(vec_target**2))
            d_theta = self._vector_angle(vec_robot_unit, vec_target_unit)

            if abs(d_theta) > self.robot.path_following_forward_angle_threshold:
                linear_velocity = 0.0
            else:
                linear_velocity = self.robot.path_following_speed

            angular_gain: float = self.robot.path_following_angular_gain
            angular_velocity = -angular_gain * d_theta
            self.robot.action.set_value(np.array([linear_velocity, angular_velocity]))

        self.robot.write_action(step_size=step_size)

        return self.is_alive

    def get_visualization_image(self) -> PIL.Image.Image:
        """Creates a visualization image showing the occupancy map with the target path overlaid.

        Returns:
            RGBA image with the occupancy map and target path drawn in green.
        """
        image = self.occupancy_map.ros_image().copy().convert("RGBA")
        draw = ImageDraw.Draw(image)
        path = self.target_path.get_value()
        if path is not None:
            line_coordinates = []
            path_pixels = self.occupancy_map.world_to_pixel_numpy(path)
            for i in range(len(path_pixels)):
                line_coordinates.append(int(path_pixels[i, 0]))
                line_coordinates.append(int(path_pixels[i, 1]))
            width_pixels = self.robot.occupancy_map_radius / self.occupancy_map.resolution
            draw.line(line_coordinates, fill="green", width=int(width_pixels / 2), joint="curve")

        return image
