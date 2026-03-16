# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Contains the base PolicyController class for loading and executing robot control policies."""


import io
from abc import ABC
from typing import Literal

import carb
import omni
from isaacsim.core.deprecation_manager import import_module
from isaacsim.core.experimental.prims import Articulation
from isaacsim.core.experimental.utils.prim import get_prim_at_path
from isaacsim.core.experimental.utils.stage import define_prim
from omni.physics.core import get_physics_simulation_interface

from .config_loader import get_articulation_props, get_physics_properties, get_robot_joint_properties, parse_env_config

torch = import_module("torch")


class PolicyController(ABC):
    """A controller that loads and executes a policy from a file.

    Args:
        prim_path: The path to the prim in the stage
        root_path: The path to the articulation root of the robot
        usd_path: The path to the USD file
        position: The initial position of the robot
        orientation: The initial orientation of the robot
    """

    def __init__(
        self,
        prim_path: str,
        root_path: str | None = None,
        usd_path: str | None = None,
        position: list[float] | None = None,
        orientation: list[float] | None = None,
    ):
        prim = get_prim_at_path(prim_path)

        if not prim.IsValid():
            prim = define_prim(prim_path, "Xform")
            if usd_path:
                prim.GetReferences().AddReference(usd_path)
            else:
                carb.log_error("unable to add robot usd, usd_path not provided")

        if root_path == None:
            self.robot = Articulation(paths=prim_path, positions=position, orientations=orientation)
        else:
            self.robot = Articulation(paths=root_path, positions=position, orientations=orientation)

    def load_policy(self, policy_file_path, policy_env_path):
        """Loads a policy from a file.

        Args:
            policy_file_path: The path to the policy file
            policy_env_path: The path to the environment configuration file
        """
        file_content = omni.client.read_file(policy_file_path)[2]
        file = io.BytesIO(memoryview(file_content).tobytes())
        self.policy = torch.jit.load(file).to(torch.device(str(self.robot._device)))
        self.policy_env_params = parse_env_config(policy_env_path)

        self._decimation, self._dt, self.render_interval = get_physics_properties(self.policy_env_params)

    def initialize(
        self,
        physics_sim_view: omni.physics.tensors.SimulationView | None = None,
        effort_modes: Literal["force", "acceleration"] = "force",
        control_mode: Literal["position", "velocity", "effort"] = "position",
        set_gains: bool = True,
        set_limits: bool = True,
        set_articulation_props: bool = True,
    ):
        """Initializes the robot and sets up the controller.

        Args:
            physics_sim_view: The physics simulation view
            effort_modes: The effort modes ("force" or "acceleration")
            control_mode: The control mode ("position", "velocity", or "effort")
            set_gains: Whether to set the joint gains
            set_limits: Whether to set the limits
            set_articulation_props: Whether to set the articulation properties
        """
        self.robot.set_dof_drive_types(effort_modes)

        get_physics_simulation_interface().flush_changes()

        self.robot.switch_dof_control_mode(control_mode)
        max_effort, max_vel, stiffness, damping, default_pos, default_vel = get_robot_joint_properties(
            self.policy_env_params, self.robot.dof_names
        )
        self.robot.set_dof_positions(default_pos)
        self.robot.set_dof_velocities(default_vel)

        self.robot.set_default_state(
            dof_positions=default_pos,
            dof_velocities=default_vel,
        )

        self.default_pos = torch.tensor(default_pos, device=torch.device(str(self.robot._device)))
        self.default_vel = torch.tensor(default_vel, device=torch.device(str(self.robot._device)))

        if set_gains:
            self.robot.set_dof_gains(stiffness, damping)
        if set_limits:
            self.robot.set_dof_max_efforts(max_effort)

            get_physics_simulation_interface().flush_changes()

            self.robot.set_dof_max_velocities(max_vel)
        if set_articulation_props:
            self._set_articulation_props()

    def _set_articulation_props(self):
        """Sets the articulation root properties from the policy environment parameters."""
        articulation_prop = get_articulation_props(self.policy_env_params)

        solver_position_iteration_count = articulation_prop.get("solver_position_iteration_count")
        solver_velocity_iteration_count = articulation_prop.get("solver_velocity_iteration_count")
        stabilization_threshold = articulation_prop.get("stabilization_threshold")
        enabled_self_collisions = articulation_prop.get("enabled_self_collisions")
        sleep_threshold = articulation_prop.get("sleep_threshold")

        if solver_position_iteration_count not in [None, float("inf")] and solver_velocity_iteration_count not in [
            None,
            float("inf"),
        ]:
            self.robot.set_solver_iteration_counts(
                position_counts=[solver_position_iteration_count],
                velocity_counts=[solver_velocity_iteration_count],
            )
        if stabilization_threshold not in [None, float("inf")]:
            self.robot.set_stabilization_thresholds([stabilization_threshold])
        if isinstance(enabled_self_collisions, bool):
            self.robot.set_enabled_self_collisions([enabled_self_collisions])
        if sleep_threshold not in [None, float("inf")]:
            self.robot.set_sleep_thresholds([sleep_threshold])

    def _compute_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute the action from the observation using the loaded policy.

        This method runs the policy network in inference mode to convert
        the current observation into an action command.

        Args:
            obs: The observation tensor matching the format specified in env.yaml

        Returns:
            The action tensor matching the format expected by the robot controller
        """
        with torch.no_grad():
            action = self.policy(obs).detach().view(-1)
        return action

    def _compute_observation(self) -> NotImplementedError:
        """Compute the current observation vector for the policy.

        This method must be implemented by derived classes to construct
        the observation vector in the format specified by env.yaml.
        The observation typically includes robot state like joint positions,
        velocities, base pose, etc.

        Raises:
            NotImplementedError: This base method must be overridden
        """
        raise NotImplementedError(
            "Compute observation needs to be implemented, expects observation tensor in the structure specified by env.yaml"
        )

    def forward(self) -> NotImplementedError:
        r"""Execute one step of the policy controller.

        This method must be implemented by derived classes to\:
        1. Compute the current observation
        2. Run the policy to get an action
        3. Apply the action to the robot

        The specific implementation depends on the robot and control mode
        (position, velocity, torque, etc.).

        Raises:
            NotImplementedError: This base method must be overridden
        """
        raise NotImplementedError(
            "Forward needs to be implemented to compute and apply robot control from observations"
        )

    def post_reset(self):
        """Called after the controller is reset."""
        self.robot.reset_to_default_state()
