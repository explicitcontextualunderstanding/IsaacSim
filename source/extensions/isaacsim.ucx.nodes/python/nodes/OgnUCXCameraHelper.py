# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import traceback

import carb
import omni
import omni.graph.core as og
import omni.replicator.core as rep
import omni.syntheticdata
import omni.syntheticdata._syntheticdata as sd
from isaacsim.core.nodes import BaseWriterNode
from pxr import Usd


class OgnUCXCameraHelperInternalState(BaseWriterNode):
    def __init__(self):
        self.rv = ""
        self.resetSimulationTimeOnStop = True
        self.publishStepSize = 1

        super().__init__(initialize=False)

    def post_attach(self, writer, render_product):
        try:
            if self.rv != "":
                omni.syntheticdata.SyntheticData.Get().set_node_attributes(
                    self.rv + "IsaacSimulationGate", {"inputs:step": self.publishStepSize}, render_product
                )

            omni.syntheticdata.SyntheticData.Get().set_node_attributes(
                "IsaacReadSimulationTime", {"inputs:resetOnStop": self.resetSimulationTimeOnStop}, render_product
            )

        except Exception:
            pass


class OgnUCXCameraHelper:
    @staticmethod
    def internal_state():
        return OgnUCXCameraHelperInternalState()

    @staticmethod
    def compute(db) -> bool:
        if db.per_instance_state.initialized is False:
            db.per_instance_state.initialized = True
            stage = omni.usd.get_context().get_stage()
            with Usd.EditContext(stage, stage.GetSessionLayer()):
                render_product_path = db.inputs.renderProductPath
                if not render_product_path:
                    carb.log_warn("Render product not valid")
                    db.per_instance_state.initialized = False
                    return False
                if stage.GetPrimAtPath(render_product_path) is None:
                    carb.log_warn("Render product no created yet, retrying on next call")
                    db.per_instance_state.initialized = False
                    return False
                db.per_instance_state.resetSimulationTimeOnStop = db.inputs.resetSimulationTimeOnStop

                db.per_instance_state.publishStepSize = db.inputs.frameSkipCount + 1

                writer = None

                time_type = ""
                if db.inputs.useSystemTime:
                    time_type = "SystemTime"
                    if db.inputs.resetSimulationTimeOnStop:
                        carb.log_warn("System timestamp is being used. Ignoring resetSimulationTimeOnStop input")

                db.per_instance_state.rv = ""

                try:
                    db.per_instance_state.rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
                        sd.SensorType.Rgb.name
                    )
                    writer = rep.writers.get(db.per_instance_state.rv + f"UCX{time_type}PublishImage")
                    writer.initialize(
                        port=db.inputs.port,
                        tag=db.inputs.tag,
                    )

                    if writer is not None:
                        db.per_instance_state.append_writer(writer)

                    db.per_instance_state.attach_writers(render_product_path)
                except Exception:
                    print(traceback.format_exc())
                    return False

        db.outputs.execOut = og.ExecutionAttributeState.ENABLED
        return True

    @staticmethod
    def release_instance(node, graph_instance_id):
        try:
            state = OgnUCXCameraHelperInternalState.per_instance_internal_state(node)
        except Exception:
            state = None
            pass

        if state is not None:
            state.reset()
