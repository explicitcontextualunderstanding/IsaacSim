# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import numpy as np
from isaacsim.core.experimental.prims import Articulation
from isaacsim.core.nodes import BaseResetNode
from isaacsim.core.nodes.ogn.OgnIsaacArticulationControllerDatabase import OgnIsaacArticulationControllerDatabase


class OgnIsaacArticulationControllerInternalState(BaseResetNode):
    """
    nodes for moving an articulated robot with joint commands
    """

    def __init__(self):
        self.prim_path = None
        self.articulation = None
        self.joint_names = None
        self.joint_indices = None
        self.joint_picked = False
        self.node = None
        super().__init__(initialize=False)

    def initialize_controller(self):
        self.articulation = Articulation(self.prim_path)
        self.initialized = True

    def joint_indicator(self):
        if self.joint_names:
            self.joint_indices = self.articulation.get_dof_indices(self.joint_names).numpy().flatten()
        elif np.size(self.joint_indices) > 0:
            self.joint_indices = self.joint_indices
        else:
            # when indices is none (not []), it defaults too all DOFs
            self.joint_indices = None
        self.joint_picked = True

    def apply_action(self, joint_positions, joint_velocities, joint_efforts):
        if self.initialized:
            if np.size(joint_positions) > 0:
                if np.isnan(joint_positions).any():
                    target = self.articulation.get_dof_position_targets(dof_indices=self.joint_indices).numpy()[0]
                    joint_positions = np.where(np.isnan(joint_positions), target, joint_positions)
                self.articulation.set_dof_position_targets(joint_positions, dof_indices=self.joint_indices)
            if np.size(joint_velocities) > 0:
                if np.isnan(joint_velocities).any():
                    target = self.articulation.get_dof_velocity_targets(dof_indices=self.joint_indices).numpy()[0]
                    joint_velocities = np.where(np.isnan(joint_velocities), target, joint_velocities)
                self.articulation.set_dof_velocity_targets(joint_velocities, dof_indices=self.joint_indices)
            if np.size(joint_efforts) > 0:
                if np.isnan(joint_efforts).any():
                    target = self.articulation.get_dof_efforts(dof_indices=self.joint_indices).numpy()[0]
                    joint_efforts = np.where(np.isnan(joint_efforts), target, joint_efforts)
                self.articulation.set_dof_efforts(joint_efforts, dof_indices=self.joint_indices)

    def custom_reset(self):
        self.articulation = None
        if self.initialized:
            self.node.get_attribute("inputs:positionCommand").set(np.empty(shape=(0, 0), dtype=np.double))
            self.node.get_attribute("inputs:velocityCommand").set(np.empty(shape=(0, 0), dtype=np.double))
            self.node.get_attribute("inputs:effortCommand").set(np.empty(shape=(0, 0), dtype=np.double))


class OgnIsaacArticulationController:
    """
    nodes for moving an articulated robot with joint commands
    """

    @staticmethod
    def init_instance(node, graph_instance_id):
        state = OgnIsaacArticulationControllerDatabase.get_internal_state(node, graph_instance_id)
        state.node = node

    @staticmethod
    def internal_state():
        return OgnIsaacArticulationControllerInternalState()

    @staticmethod
    def compute(db) -> bool:
        state = db.per_instance_state
        try:
            if not state.initialized:
                if len(db.inputs.robotPath) != 0:
                    state.prim_path = db.inputs.robotPath
                else:
                    if len(db.inputs.targetPrim) == 0:
                        db.log_error("No robot prim found for the articulation controller")
                        return False
                    else:
                        state.prim_path = db.inputs.targetPrim[0].GetString()

                # initialize the controller handle for the robot
                state.initialize_controller()

            # pick the joints that are being commanded, this can be different at every step
            joint_names = db.inputs.jointNames
            if joint_names and np.asarray([joint_names != state.joint_names]).flatten().any():
                state.joint_names = joint_names
                state.joint_picked = False

            joint_indices = db.inputs.jointIndices
            if np.asarray(joint_indices).any() and not np.array_equal(joint_indices, state.joint_indices):
                state.joint_indices = np.array(joint_indices)
                state.joint_picked = False

            if not state.joint_picked:
                state.joint_indicator()

            state.apply_action(db.inputs.positionCommand, db.inputs.velocityCommand, db.inputs.effortCommand)

        except Exception as error:
            db.log_warn(str(error))
            return False

        return True

    @staticmethod
    def release_instance(node, graph_instance_id):
        try:
            state = OgnIsaacArticulationControllerDatabase.get_internal_state(node, graph_instance_id)
        except Exception:
            state = None
            pass

        if state is not None:
            state.reset()
            state.initialized = False
