# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from __future__ import annotations

import pathlib

import cumotion
import isaacsim.robot_motion.experimental.motion_generation as mg
import numpy as np
import warp as wp

from .configuration_loader import CumotionRobot
from .cumotion_world_interface import CumotionWorldInterface
from .utils import isaac_sim_to_cumotion_rotation, isaac_sim_to_cumotion_translation


class RmpFlowController(mg.BaseController):
    """Reactive motion policy controller using cuMotion's RMPflow algorithm.

    RMPflow (Riemannian Motion Policies) is a reactive control algorithm that generates
    smooth, collision-free motions by combining multiple motion policies in task and
    configuration space. This controller continuously updates joint commands based on
    the desired targets.

    Args:
        cumotion_robot: Robot containing kinematics and joint information.
        cumotion_world_interface: World interface providing collision geometry.
        robot_joint_space: The full ordered joint-space of the controlled robot.
        robot_site_space: The full ordered site-space of the controlled robot (used for validation of tool-frame name only).
        rmp_flow_configuration_filename: Path to the RMPflow YAML configuration file.
            If a relative path is provided, it is resolved relative to
            cumotion_robot.directory. If an absolute path is provided,
            it is used as-is. Defaults to "rmp_flow.yaml".
        tool_frame: Name of the tool frame for end-effector control. Defaults to None,
            which uses the first tool frame defined in the robot description.
    Example:

        .. code-block:: python

            controller = RmpFlowController(
                cumotion_robot=robot,
                cumotion_world_interface=world_interface,
                rmp_flow_configuration_filename="rmp_flow.yaml"
            )
            controller.reset(estimated_state, setpoint_state, t=0.0)
            desired_state = controller.forward(estimated_state, setpoint_state, t=0.1)
    """

    def __init__(
        self,
        cumotion_robot: CumotionRobot,
        cumotion_world_interface: CumotionWorldInterface,
        robot_joint_space: list[str],
        robot_site_space: list[str],
        rmp_flow_configuration_filename: pathlib.Path | str = "rmp_flow.yaml",
        tool_frame: str | None = None,
    ):

        if not set(cumotion_robot.controlled_joint_names).issubset(set(robot_joint_space)):
            raise ValueError(
                f"Cumotion controlled joints {cumotion_robot.controlled_joint_names} are not a subset of the robot_joint_space {robot_joint_space}."
            )

        self._robot_joint_space = robot_joint_space

        self._tool_frame = tool_frame

        if self._tool_frame is None:
            tool_frame_names = cumotion_robot.robot_description.tool_frame_names()
            if not tool_frame_names:
                raise RuntimeError("No tool frames available in robot description and no tool_frame was provided.")
            self._tool_frame = tool_frame_names[0]

        if self._tool_frame not in robot_site_space:
            raise ValueError(
                f"The specified tool name {self._tool_frame} is not in the list of controlled sites {robot_site_space}"
            )

        self._cumotion_world_interface = cumotion_world_interface

        # there is no "RmpFlow" algorithm until we initialize.
        self._rmp_flow = None
        self._cumotion_robot = cumotion_robot

        self._output_position = None
        self._output_velocity = None
        self._previous_run_time = None

        # create the rmp-flow configuration. Do not initialize, which will let the user modify
        # the configuration should they so choose.
        rmp_flow_path = pathlib.Path(rmp_flow_configuration_filename)
        if rmp_flow_path.is_absolute():
            full_rmp_flow_path = rmp_flow_path
        else:
            full_rmp_flow_path = cumotion_robot.directory / rmp_flow_path

        self._rmp_flow_config = cumotion.create_rmpflow_config_from_file(
            rmpflow_config_file=full_rmp_flow_path,
            robot_description=cumotion_robot.robot_description,
            end_effector_frame=self._tool_frame,
            world_view=cumotion_world_interface.world_view,
        )

    def get_rmp_flow_config(self) -> cumotion.RmpFlowConfig:
        """Get the RMPflow configuration object.

        Returns the underlying cuMotion configuration object, allowing users to modify
        controller parameters before initialization.

        Returns:
            The cuMotion RMPflow configuration object.

        Example:

            .. code-block:: python

                config = controller.get_rmp_flow_config()
                config.set_param("cspace_target_rmp/damping_gain", 0.9)
        """
        return self._rmp_flow_config

    def forward(
        self, estimated_state: mg.RobotState, setpoint_state: mg.RobotState | None, t: float, **kwargs
    ) -> mg.RobotState | None:
        """Compute the desired joint action for the next time-step.

        Evaluates the RMPflow controller to compute desired joint positions and velocities
        based on the current desired robot state and target attractors. The controller integrates
        the computed accelerations over time.

        Args:
            estimated_state: Current estimated state of the robot.
            setpoint_state: Desired setpoint state containing target attractors. Can include
                joint positions, end-effector positions, and/or orientations. End-effector inputs
                must match the tool frame defined at initialization for this controller.
            t: Current clock time (simulation or real).
            **kwargs: Additional arguments for the controller.

        Returns:
            Robot state containing desired joint positions and velocities, or None if
            the controller is not initialized.

        Example:

            .. code-block:: python

                desired_state = controller.forward(current_state, target_state, t=1.0)
                if desired_state is not None:
                    robot.set_dof_position_targets(
                        desired_state.joints.positions,
                        indices=[0],
                        dof_indices=robot.get_joint_indices(desired_state.joints.names)
                    )
        """
        # if we haven't initialized our rmp flow algorithm, return:
        if self._rmp_flow is None:
            return None

        # update the worldview:
        self._cumotion_world_interface.world_view.update()

        # update the attractor according to the setpoint_state, if they exist:
        if setpoint_state is not None:
            # check if the joint states are defined, and matches all of the desired joints:
            joint_attractor = self._joint_position_from_robot_state(setpoint_state)
            if joint_attractor is not None:
                self._rmp_flow.set_cspace_attractor(joint_attractor)

            tool_position = self._tool_position_from_robot_state(setpoint_state)
            if tool_position is not None:
                self._rmp_flow.set_end_effector_position_attractor(tool_position)

            tool_orientation = self._tool_orientation_from_robot_state(setpoint_state)
            if tool_orientation is not None:
                self._rmp_flow.set_end_effector_orientation_attractor(tool_orientation)

        # TODO:
        # SUB-STEPPING.
        # BOOLEAN TO EVALUATE FROM ESTIMATED STATE.
        joint_acceleration = np.zeros_like(self._output_position)
        self._rmp_flow.eval_accel(self._output_position, self._output_velocity, joint_acceleration)

        self._output_velocity += joint_acceleration * (t - self._previous_run_time)
        self._output_position += self._output_velocity * (t - self._previous_run_time)
        self._previous_run_time = t

        # output the current desired joints & reference frames:
        return mg.RobotState(
            joints=mg.JointState.from_name(
                robot_joint_space=self._robot_joint_space,
                positions=(
                    self._cumotion_robot.controlled_joint_names,
                    wp.from_numpy(self._output_position),
                ),
                velocities=(
                    self._cumotion_robot.controlled_joint_names,
                    wp.from_numpy(self._output_velocity),
                ),
                efforts=None,
            )
        )

    def reset(self, estimated_state: mg.RobotState, setpoint_state: mg.RobotState | None, t: float, **kwargs) -> bool:
        """Reset the controller to a safe initial state.

        Initializes the RMPflow controller and sets the internal state to match the
        current robot configuration. This should be called before running the controller
        for the first time.

        Args:
            estimated_state: Current estimated state of the robot.
            setpoint_state: Initial setpoint state (currently unused).
            t: Current clock time (simulation or real).
            **kwargs: Additional arguments for the controller.

        Returns:
            True if reset was successful, False otherwise.

        Example:

            .. code-block:: python

                success = controller.reset(current_state, None, t=0.0)
                if success:
                    print("Controller initialized successfully")
        """

        # Get all of the joint relevant joint-states from the estimated
        current_joint_positions = self._joint_position_from_robot_state(estimated_state)
        if current_joint_positions is None:
            self._output_position = None
            self._output_velocity = None

            # We don't have the required joint-positions to reset RmpFlow.
            return False

        self._rmp_flow = cumotion.create_rmpflow(self._rmp_flow_config)

        # Check if initialization was successful:
        if self._rmp_flow is None:
            # this is not a successful reset, we are not initialized!!
            self._output_position = None
            self._output_velocity = None
            return False

        # otherwise, reset the attractors:
        self._rmp_flow.clear_end_effector_orientation_attractor()
        self._rmp_flow.clear_end_effector_position_attractor()

        # Set the attractor to our current position.
        self._rmp_flow.set_cspace_attractor(current_joint_positions)

        # set the internal integrator to be stationary at the current joint position.
        self._output_position = current_joint_positions
        self._output_velocity = np.zeros_like(current_joint_positions)

        self._previous_run_time = t

        # successful reset.
        return True

    def _joint_position_from_robot_state(self, state: mg.RobotState) -> np.ndarray | None:
        if state.joints is None:
            return None

        # Check whether all of the controller joint names are available in the input state:
        if not set(self._cumotion_robot.controlled_joint_names).issubset(set(state.joints.position_names)):
            return None

        # Get the joint values in the correct order:
        output = np.zeros(shape=[len(self._cumotion_robot.controlled_joint_names)])
        input_joint_positions = state.joints.positions.numpy()

        # map the cumotion indices to those used by isaac sim:
        cumotion_index_to_isaac_sim_index = [
            state.joints.position_names.index(cumotion_name)
            for cumotion_name in self._cumotion_robot.controlled_joint_names
        ]

        for i_cumotion in range(len(cumotion_index_to_isaac_sim_index)):
            output[i_cumotion] = input_joint_positions[cumotion_index_to_isaac_sim_index[i_cumotion]]

        return output

    def _tool_position_from_robot_state(self, state: mg.RobotState) -> np.ndarray | None:
        if state.sites is None:
            return None

        # Check if the tool-frame we are controlling is in the input state:
        if not self._tool_frame in state.sites.position_names:
            return None

        # return the position based on the correct index:
        position_index = state.sites.position_names.index(self._tool_frame)
        position_world = np.array(state.sites.positions.numpy()[position_index])

        world_to_robot_base_position, world_to_robot_base_quaternion = (
            self._cumotion_world_interface.get_world_to_robot_base_transform()
        )

        return isaac_sim_to_cumotion_translation(
            position_world_to_target=position_world,
            position_world_to_base=world_to_robot_base_position,
            orientation_world_to_base=world_to_robot_base_quaternion,
        )

    def _tool_orientation_from_robot_state(self, state: mg.RobotState) -> np.ndarray | None:
        if state.sites is None:
            return None

        # Check if the tool-frame we are controlling is in the input state:
        if not self._tool_frame in state.sites.orientation_names:
            return None

        # return the orientations based on the correct index:
        orientation_index = state.sites.orientation_names.index(self._tool_frame)
        rotation_world_target = state.sites.orientations.numpy()[orientation_index]

        _, world_to_robot_base_quaternion = self._cumotion_world_interface.get_world_to_robot_base_transform()

        return isaac_sim_to_cumotion_rotation(
            orientation_world_to_target=rotation_world_target,
            orientation_world_to_base=world_to_robot_base_quaternion,
        )
