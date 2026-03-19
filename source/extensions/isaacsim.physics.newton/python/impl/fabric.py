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
"""Fabric synchronization for Newton physics simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import carb
import usdrt
import warp as wp

if TYPE_CHECKING:
    import newton


@wp.kernel(enable_backward=False)
def set_fabric_transforms(
    fabric_transforms: wp.fabricarray(dtype=wp.mat44d),
    newton_indices: wp.fabricarray(dtype=wp.uint32),
    newton_body_q: wp.array(ndim=1, dtype=wp.transformf),
):
    """Write Newton body transforms to Fabric world matrices.

    Args:
        fabric_transforms: Output Fabric world matrices.
        newton_indices: Newton body indices to read from.
        newton_body_q: Newton body transforms.
    """
    i = int(wp.tid())
    idx = int(newton_indices[i])
    transform = newton_body_q[idx]
    fabric_transforms[i] = wp.transpose(wp.mat44d(wp.math.transform_to_matrix(transform)))


class FabricManager:
    """Manager for syncing Newton simulation state to USD Fabric.

    Args:
        usdrt_stage: The USDRT stage instance to synchronize with.
    """

    def __init__(self, usdrt_stage: usdrt.Usd.Stage):
        self.stage = usdrt_stage
        self.newton_index_attr = "newton:index"
        self._first_update_done = False
        self._no_prims_warning_logged = False

    def update_fabric(self, model: "newton.Model", state: "newton.State", scene_scale: float, device: str):
        """Sync Newton body transforms to Fabric.

        Args:
            model: Newton Model object.
            state: Newton State object containing body_q transforms.
            scene_scale: Scene scale factor (unused but kept for API compatibility).
            device: Device string.
        """
        # Select all prims that have both the world matrix and newton index
        selection = self.stage.SelectPrims(
            require_attrs=[
                (usdrt.Sdf.ValueTypeNames.Matrix4d, "omni:fabric:worldMatrix", usdrt.Usd.Access.ReadWrite),
                (usdrt.Sdf.ValueTypeNames.UInt, self.newton_index_attr, usdrt.Usd.Access.Read),
            ],
            device=str(device),
        )

        if selection.GetCount() == 0:
            if not self._no_prims_warning_logged:
                carb.log_warn(
                    "[isaacsim.physics.newton] No prims found with newton:index attribute. Fabric transforms will not update!"
                )
                self._no_prims_warning_logged = True
            return

        # Diagnostic: Check if selection count matches expected body count
        expected_bodies = len(model.body_label) if hasattr(model, "body_label") else model.body_count
        if selection.GetCount() != expected_bodies and not self._first_update_done:
            carb.log_warn(
                f"[isaacsim.physics.newton] Fabric selection mismatch: selected {selection.GetCount()} prims, "
                f"but model has {expected_bodies} bodies. Some bodies may not update!"
            )
        # Guard against obviously invalid mapping size
        try:
            body_count = int(model.body_count)
        except Exception:
            body_count = int(state.body_q.shape[0]) if state and state.body_q is not None else 0
        if selection.GetCount() > body_count and body_count > 0:
            carb.log_error(
                f"[isaacsim.physics.newton] Fabric selection count ({selection.GetCount()}) exceeds model bodies ({body_count}). Skipping update."
            )
            return

        try:
            fabric_transforms = wp.fabricarray(selection, "omni:fabric:worldMatrix")
            newton_indices = wp.fabricarray(selection, self.newton_index_attr)
            self._first_update_done = True
        except Exception as e:
            carb.log_error(
                f"[isaacsim.physics.newton] Failed to build fabric arrays: {e}. count={selection.GetCount()}, device={device}"
            )
            return

        # Early guards on shapes
        if newton_indices.shape[0] == 0 or state is None or state.body_q is None:
            return

        try:
            wp.launch(
                set_fabric_transforms,
                dim=newton_indices.shape[0],
                inputs=[fabric_transforms, newton_indices, state.body_q],
                device=device,
            )
            wp.synchronize_device(device)
        except Exception as e:
            carb.log_error(
                f"[isaacsim.physics.newton] Fabric update failed: {e}. count={newton_indices.shape[0]}, bodies={state.body_q.shape[0]}"
            )
