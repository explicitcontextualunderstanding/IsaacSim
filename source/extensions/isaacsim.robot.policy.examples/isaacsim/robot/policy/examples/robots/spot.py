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

"""Spot quadruped robot policy controller for flat terrain locomotion."""


import isaacsim.core.experimental.utils.transform as transform_utils
import warp as wp
from isaacsim.core.deprecation_manager import import_module
from isaacsim.robot.policy.examples.controllers import PolicyController
from isaacsim.storage.native import get_assets_root_path

torch = import_module("torch")


class SpotFlatTerrainPolicy(PolicyController):
    """Policy controller for the Spot quadruped robot executing flat terrain locomotion.

    This controller implements a learned policy for stable walking on flat terrain,
    handling velocity commands for forward/backward motion, lateral motion, and turning.

    Args:
        prim_path: The prim path of the robot on the stage.
        root_path: The path to the articulation root of the robot.
        usd_path: The robot usd filepath in the directory.
        position: The position of the robot.
        orientation: The orientation of the robot.
    """

    def __init__(
        self,
        prim_path: str,
        root_path: str | None = None,
        usd_path: str | None = None,
        position: list[float] | None = None,
        orientation: list[float] | None = None,
    ):
        """
        Initialize robot and load RL policy.

        Args:
            prim_path: The prim path of the robot on the stage
            root_path: The path to the articulation root of the robot
            usd_path: The robot usd filepath in the directory
            position: The position of the robot
            orientation: The orientation of the robot
        """
        assets_root_path = get_assets_root_path()
        if usd_path == None:
            usd_path = assets_root_path + "/Isaac/Robots/BostonDynamics/spot/spot.usd"

        super().__init__(prim_path, root_path, usd_path, position, orientation)

        self.load_policy(
            assets_root_path + "/Isaac/Samples/Policies/Spot_Policies/spot_policy.pt",
            assets_root_path + "/Isaac/Samples/Policies/Spot_Policies/spot_env.yaml",
        )
        self._action_scale = 0.2
        self._previous_action = torch.zeros(12)
        self._policy_counter = 0

    def _compute_observation(self, command):
        """Compute the observation vector for the policy.

        The observation includes base linear/angular velocities, gravity direction,
        command velocities, joint positions/velocities, and previous actions.

        Args:
            command: The robot command velocities (v_x, v_y, w_z) in m/s and rad/s

        Returns:
            A 48-dimensional observation vector containing:
            - [0:3]: Base linear velocity in body frame
            - [3:6]: Base angular velocity in body frame
            - [6:9]: Gravity direction in body frame
            - [9:12]: Command velocities
            - [12:24]: Joint position error from default
            - [24:36]: Joint velocities
            - [36:48]: Previous action
        """
        lin_vel_I, ang_vel_I = self.robot.get_velocities()
        pos_IB, q_IB = self.robot.get_world_poses()

        R_IB = wp.to_torch(transform_utils.quaternion_to_rotation_matrix(q_IB))
        R_BI = R_IB.squeeze().t()
        lin_vel_b = R_BI @ wp.to_torch(lin_vel_I).t()
        ang_vel_b = R_BI @ wp.to_torch(ang_vel_I).t()
        gravity_b = R_BI @ torch.tensor([0.0, 0.0, -1.0], device=torch.device(str(self.robot._device)))

        obs = torch.zeros(48, device=torch.device(str(self.robot._device)))
        # Base lin vel
        obs[:3] = lin_vel_b.squeeze()
        # Base ang vel
        obs[3:6] = ang_vel_b.squeeze()
        # Gravity
        obs[6:9] = gravity_b.squeeze()
        # Command
        obs[9:12] = command
        # Joint states
        current_joint_pos = wp.to_torch(self.robot.get_dof_positions())
        current_joint_vel = wp.to_torch(self.robot.get_dof_velocities())
        obs[12:24] = current_joint_pos - self.default_pos
        obs[24:36] = current_joint_vel - self.default_vel
        # Previous Action
        obs[36:48] = self._previous_action
        return obs

    def forward(self, dt, command):
        """Compute the desired joint positions and apply them to the articulation.

        Uses the policy to compute joint position targets that achieve the desired
        command velocities, with position control at the joint level.

        Args:
            dt: Timestep update in the world in seconds
            command: The robot command velocities (v_x, v_y, w_z) in m/s and rad/s
        """
        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation(command)
            self.action = self._compute_action(obs)
            self._previous_action = self.action.clone()

        self.robot.set_dof_position_targets(
            positions=wp.from_torch(self.default_pos + (self.action * self._action_scale))
        )
        self._policy_counter += 1
