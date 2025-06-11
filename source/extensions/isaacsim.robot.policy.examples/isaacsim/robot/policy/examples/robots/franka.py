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

from typing import Optional

import numpy as np
import omni
import omni.kit.commands
import torch
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.policy.examples.controllers import PolicyController
from isaacsim.storage.native import get_assets_root_path
from pxr import UsdGeom


class FrankaOpenDrawerPolicy(PolicyController):
    """The Franka Policy"""

    def __init__(
        self,
        prim_path: str,
        cabinet: SingleArticulation,
        root_path: Optional[str] = None,
        name: str = "franka",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize franka robot and import flat terrain policy.

        Args:
            prim_path (str) -- prim path of the robot on the stage
            root_path (Optional[str]): The path to the articulation root of the robot
            name (str) -- name of the quadruped
            usd_path (str) -- robot usd filepath in the directory
            position (np.ndarray) -- position of the robot
            orientation (np.ndarray) -- orientation of the robot

        """
        assets_root_path = get_assets_root_path()

        policy_path = assets_root_path + "/Isaac/Samples/Policies/Franka_Policies/Open_Drawer_Policy/"
        if usd_path == None:
            usd_path = assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"

        super().__init__(name, prim_path, root_path, usd_path, position, orientation)

        self.load_policy(
            policy_path + "policy.pt",
            policy_path + "env.yaml",
        )

        self._action_scale = 0.5
        self._previous_action = np.zeros(9)
        self._policy_counter = 0

        self.cabinet = cabinet
        self.robot_grasp_rot = np.zeros(4)
        self.robot_grasp_pos = np.zeros(3)
        self.drawer_grasp_rot = np.zeros(4)
        self.drawer_grasp_pos = np.zeros(3)

    def _compute_observation(self):
        """
        Compute the observation vector for the policy.

        Returns:
        np.ndarray -- The observation vector.

        """

        def get_env_local_pose(xformable):
            """Compute pose in env-local coordinates"""
            env_pos = torch.Tensor([0, 0, 0])
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz])

        stage = get_current_stage()
        # retrieve the hand's world coordinates
        hand_pose_w = get_env_local_pose(UsdGeom.Xformable(stage.GetPrimAtPath("/World/franka/panda_link7")))
        self.hand_pose = hand_pose_w[0:3]
        self.hand_rot = hand_pose_w[3:7]

        # retrieve the panda fingers' world coordinates
        lfinger_pose = get_env_local_pose(UsdGeom.Xformable(stage.GetPrimAtPath("/World/franka/panda_leftfinger")))
        rfinger_pose = get_env_local_pose(UsdGeom.Xformable(stage.GetPrimAtPath("/World/franka/panda_rightfinger")))
        finger_pose = torch.zeros(7)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        self.finger_mid = finger_pose[0:3]

        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(self.hand_rot, self.hand_pose)
        robot_local_grasp_pose_rot, robot_local_pose_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], self.finger_mid
        )

        self.robot_local_grasp_pos = robot_local_pose_pos
        self.robot_local_grasp_rot = robot_local_grasp_pose_rot

        # retrieve the drawer world coordinate
        drawer_handle_pos_w = get_env_local_pose(
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/cabinet/drawer_handle_top"))
        )
        self.drawer_pos = drawer_handle_pos_w[0:3]
        self.drawer_rot = drawer_handle_pos_w[3:7]

        finger_pose[3:7] = lfinger_pose[3:7]

        # from RL training result env.yaml, we can see that drawer_handle_top frame is caulcuated in the common reference frame
        # cabinet frame with offset below, so we need directly set offset in code.
        # ---------------
        # target_frames:
        # - prim_path: /World/envs/env_.*/Cabinet/drawer_handle_top
        # name: drawer_handle_top
        # offset:
        #    pos: !!python/tuple
        #    - 0.305
        #    - 0.0
        #    - 0.01
        #    rot: !!python/tuple
        #    - 0.5
        #    - 0.5
        #    - -0.5
        #    - -0.5
        drawer_local_grasp_pose = torch.tensor([0.305, 0.0, 0.01, 0.5, 0.5, -0.5, -0.5])
        self.drawer_local_grasp_pos = drawer_local_grasp_pose[0:3]
        self.drawer_local_grasp_rot = drawer_local_grasp_pose[3:7]

        self._compute_intermediate_values()

        obs = np.zeros(31)
        # Base lin pos
        obs[:9] = self.robot.get_joint_positions() - self.default_pos

        # Base ang vel
        obs[9:18] = self.robot.get_joint_velocities() - self.default_vel

        # Joint states
        obs[18:19] = self.cabinet.get_joint_positions()[self.drawer_link_idx]
        obs[19:20] = self.cabinet.get_joint_velocities()[self.drawer_link_idx]

        obs[20:23] = self.drawer_grasp_pos - self.robot_grasp_pos

        # Previous Action
        obs[23:31] = self._previous_action[0:8]

        return obs

    def _compute_grasp_transforms(
        self,
        hand_rot,
        hand_pos,
        franka_local_grasp_rot,
        franka_local_grasp_pos,
        drawer_rot,
        drawer_pos,
        drawer_local_grasp_rot,
        drawer_local_grasp_pos,
    ):
        global_franka_rot, global_franka_pos = tf_combine(
            hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos
        )
        global_drawer_rot, global_drawer_pos = tf_combine(
            drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos
        )

        return global_franka_rot, global_franka_pos, global_drawer_rot, global_drawer_pos

    def _compute_intermediate_values(self, env_ids=None):

        if env_ids is None:
            env_ids = 0

        (
            self.robot_grasp_rot,
            self.robot_grasp_pos,
            self.drawer_grasp_rot,
            self.drawer_grasp_pos,
        ) = self._compute_grasp_transforms(
            self.hand_rot,
            self.hand_pose,
            self.robot_local_grasp_rot,
            self.robot_local_grasp_pos,
            self.drawer_rot,
            self.drawer_pos,
            self.drawer_local_grasp_rot,
            self.drawer_local_grasp_pos,
        )

    def forward(self, dt):
        """
        Compute the desired articulation action and apply them to the robot articulation.

        Argument:
        dt (float) -- Timestep update in the world.

        """
        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation()
            self.action = self._compute_action(obs)
            self._previous_action = self.action.copy()

        # articulation space
        # copy last item for two fingers in order to increase action size from 8 to 9
        action_input = np.append(self.action, self.action[-1])
        action = ArticulationAction(joint_positions=self.default_pos + (action_input * self._action_scale))

        # here action is size 9
        self.robot.apply_action(action)

        self._policy_counter += 1

    def initialize(self, physics_sim_view=None) -> None:
        """
        Initialize the articulation interface, set up drive mode
        """
        super().initialize(physics_sim_view=physics_sim_view, control_mode="effort")

        self.cabinet.initialize(physics_sim_view=physics_sim_view)

        self.hand_link_idx = self.robot.get_dof_index("panda_joint7")
        self.left_finger_link_idx = self.robot.get_dof_index("panda_finger_joint1")
        self.right_finger_link_idx = self.robot.get_dof_index("panda_finger_joint2")
        self.drawer_link_idx = self.cabinet.get_dof_index("drawer_top_joint")
