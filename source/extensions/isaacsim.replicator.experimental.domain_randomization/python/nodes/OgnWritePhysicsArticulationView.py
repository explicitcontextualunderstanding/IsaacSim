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

import numpy as np
import omni.graph.core as og
from isaacsim.core.experimental.utils.transform import euler_angles_to_quaternion
from isaacsim.replicator.experimental.domain_randomization import ARTICULATION_ATTRIBUTES, TENDON_ATTRIBUTES
from isaacsim.replicator.experimental.domain_randomization import physics_view as physics

OPERATION_TYPES = ["direct", "additive", "scaling"]


def apply_randomization_operation(view_name, operation, attribute_name, samples, indices, on_reset):
    """Apply randomization operation for indexed values."""
    if on_reset:
        return physics._articulation_views_reset_values[view_name][attribute_name][indices]
    if operation == "additive":
        return physics._articulation_views_reset_values[view_name][attribute_name][indices] + samples
    elif operation == "scaling":
        return physics._articulation_views_reset_values[view_name][attribute_name][indices] * samples
    else:
        return samples


def apply_randomization_operation_full_tensor(view_name, operation, attribute_name, samples, indices, on_reset):
    """Apply randomization operation for full tensor values."""
    if on_reset:
        return physics._articulation_views_reset_values[view_name][attribute_name]
    initial_values = np.copy(physics._articulation_views_reset_values[view_name][attribute_name])
    if operation == "additive":
        initial_values[indices] += samples
    elif operation == "scaling":
        initial_values[indices] *= samples
    else:
        initial_values[indices] = samples
    return initial_values


def modify_initial_values(view_name, operation, attribute_name, samples, indices):
    """Modify initial values based on operation type."""
    if operation == "additive":
        physics._articulation_views_reset_values[view_name][attribute_name][indices] = (
            physics._articulation_views_initial_values[view_name][attribute_name][indices] + samples
        )
    elif operation == "scaling":
        physics._articulation_views_reset_values[view_name][attribute_name][indices] = (
            physics._articulation_views_initial_values[view_name][attribute_name][indices] * samples
        )
    else:
        physics._articulation_views_reset_values[view_name][attribute_name][indices] = samples


def get_bucketed_values(view_name, attribute_name, samples, distribution, dist_param_1, dist_param_2, num_buckets):
    """Get bucketed values for material properties randomization."""
    new_samples = samples.copy()

    if distribution == "gaussian":
        lo = dist_param_1 - 2 * np.sqrt(dist_param_2)
        hi = dist_param_1 + 2 * np.sqrt(dist_param_2)
    elif distribution in ("uniform", "loguniform"):
        lo = dist_param_1
        hi = dist_param_2
    else:
        raise ValueError(f"Unsupported distribution for bucketing: {distribution!r}")

    dim = samples.shape[-1]
    lo = lo.reshape(-1, dim)[0]
    hi = hi.reshape(-1, dim)[0]
    for d in range(dim):
        buckets = np.array([(hi[d] - lo[d]) * i / num_buckets + lo[d] for i in range(num_buckets)])
        idx = np.clip(np.searchsorted(buckets, new_samples[..., d], side="right") - 1, 0, num_buckets - 1)
        new_samples[..., d] = buckets[idx]

    return new_samples


class OgnWritePhysicsArticulationView:
    """OmniGraph node that writes physics attributes to Articulation views."""

    @staticmethod
    def compute(db) -> bool:
        view_name = db.inputs.prims
        attribute_name = db.inputs.attribute
        operation = db.inputs.operation
        values = db.inputs.values

        distribution = db.inputs.distribution
        dist_param_1 = db.inputs.dist_param_1
        dist_param_2 = db.inputs.dist_param_2
        num_buckets = db.inputs.num_buckets

        if db.inputs.indices is None or len(db.inputs.indices) == 0:
            db.outputs.execOut = og.ExecutionAttributeState.ENABLED
            return False
        indices = np.array(db.inputs.indices)
        on_reset = db.inputs.on_reset

        try:
            view = physics._articulation_views.get(view_name)
            if view is None:
                raise ValueError(f"Expected a registered articulation_view, but instead received {view_name}")
            if attribute_name not in ARTICULATION_ATTRIBUTES:
                raise ValueError(
                    f"Expected an attribute in {ARTICULATION_ATTRIBUTES}, but instead received {attribute_name}"
                )
            if operation not in OPERATION_TYPES:
                raise ValueError(f"Expected an operation type in {OPERATION_TYPES}, but instead received {operation}")

            samples = np.array(values).reshape(len(indices), -1)
            physics_view = view._physics_articulation_view
        except Exception as error:
            db.log_error(f"WritePhysics Error: {error}")
            db.outputs.execOut = og.ExecutionAttributeState.DISABLED
            return False

        if on_reset:
            modify_initial_values(view_name, operation, attribute_name, samples, indices)

        if attribute_name == "stiffness":
            stiffnesses = apply_randomization_operation_full_tensor(
                view_name, operation, attribute_name, samples, indices, on_reset
            )
            physics_view.set_dof_stiffnesses(stiffnesses, indices)
        elif attribute_name == "damping":
            dampings = apply_randomization_operation_full_tensor(
                view_name, operation, attribute_name, samples, indices, on_reset
            )
            physics_view.set_dof_dampings(dampings, indices)
        elif attribute_name == "joint_friction":
            frictions = apply_randomization_operation_full_tensor(
                view_name, operation, attribute_name, samples, indices, on_reset
            )
            physics_view.set_dof_friction_coefficients(frictions, indices)
        elif attribute_name == "position":
            positions = apply_randomization_operation(view_name, operation, attribute_name, samples, indices, on_reset)
            view.set_world_poses(positions=positions, indices=indices)
        elif attribute_name == "orientation":
            rpys = apply_randomization_operation(view_name, operation, attribute_name, samples, indices, on_reset)
            # Convert euler angles to quaternions using experimental utils (returns wp.array, convert to numpy)
            orientations = euler_angles_to_quaternion(rpys, degrees=False, extrinsic=True).numpy()
            view.set_world_poses(orientations=orientations, indices=indices)
        elif attribute_name == "linear_velocity":
            linear_velocities = apply_randomization_operation(
                view_name, operation, attribute_name, samples, indices, on_reset
            )
            view.set_velocities(linear_velocities=linear_velocities, indices=indices)
        elif attribute_name == "angular_velocity":
            angular_velocities = apply_randomization_operation(
                view_name, operation, attribute_name, samples, indices, on_reset
            )
            view.set_velocities(angular_velocities=angular_velocities, indices=indices)
        elif attribute_name == "velocity":
            velocities = apply_randomization_operation(view_name, operation, attribute_name, samples, indices, on_reset)
            linear_vel = velocities[:, :3]
            angular_vel = velocities[:, 3:]
            view.set_velocities(linear_velocities=linear_vel, angular_velocities=angular_vel, indices=indices)
        elif attribute_name == "joint_positions":
            joint_positions = apply_randomization_operation(
                view_name, operation, attribute_name, samples, indices, on_reset
            )
            view.set_dof_positions(positions=joint_positions, indices=indices)
        elif attribute_name == "joint_velocities":
            joint_velocities = apply_randomization_operation(
                view_name, operation, attribute_name, samples, indices, on_reset
            )
            view.set_dof_velocities(velocities=joint_velocities, indices=indices)
        elif attribute_name == "lower_dof_limits":
            dof_limits = np.asarray(view.get_dof_limits())
            dof_limits[..., 0] = apply_randomization_operation_full_tensor(
                view_name, operation, attribute_name, samples, indices, on_reset
            )
            physics_view.set_dof_limits(dof_limits, indices)
        elif attribute_name == "upper_dof_limits":
            dof_limits = np.asarray(view.get_dof_limits())
            dof_limits[..., 1] = apply_randomization_operation_full_tensor(
                view_name, operation, attribute_name, samples, indices, on_reset
            )
            physics_view.set_dof_limits(dof_limits, indices)
        elif attribute_name == "max_efforts":
            max_efforts = apply_randomization_operation(
                view_name, operation, attribute_name, samples, indices, on_reset
            )
            view.set_max_efforts(efforts=max_efforts, indices=indices)
        elif attribute_name == "joint_armatures":
            joint_armatures = apply_randomization_operation_full_tensor(
                view_name, operation, attribute_name, samples, indices, on_reset
            )
            physics_view.set_dof_armatures(joint_armatures, indices)
        elif attribute_name == "joint_max_velocities":
            joint_max_velocities = apply_randomization_operation_full_tensor(
                view_name, operation, attribute_name, samples, indices, on_reset
            )
            physics_view.set_dof_max_velocities(joint_max_velocities, indices)
        elif attribute_name == "joint_efforts":
            view.set_dof_efforts(efforts=samples, indices=indices)
        elif attribute_name == "body_masses":
            body_masses = apply_randomization_operation_full_tensor(
                view_name, operation, attribute_name, samples, indices, on_reset
            )
            physics_view.set_masses(body_masses, indices)
        elif attribute_name == "body_inertias":
            diagonal_inertias = apply_randomization_operation_full_tensor(
                view_name, operation, attribute_name, samples, indices, on_reset
            )
            inertia_matrices = np.zeros((len(view), physics_view.max_links, 9), dtype=np.float32)
            inertia_matrices[:, :, [0, 4, 8]] = diagonal_inertias.reshape(len(view), physics_view.max_links, 3)
            physics_view.set_inertias(inertia_matrices, indices)
        elif attribute_name == "material_properties":
            material_properties = apply_randomization_operation_full_tensor(
                view_name, operation, attribute_name, samples, indices, on_reset
            ).reshape(len(view), physics_view.max_shapes, 3)
            if num_buckets is not None and num_buckets > 0:
                material_properties = get_bucketed_values(
                    view_name,
                    attribute_name,
                    material_properties,
                    distribution,
                    dist_param_1,
                    dist_param_2,
                    num_buckets,
                )
            physics_view.set_material_properties(material_properties, indices)
        elif attribute_name == "contact_offset":
            contact_offsets = apply_randomization_operation_full_tensor(
                view_name, operation, attribute_name, samples, indices, on_reset
            )
            physics_view.set_contact_offsets(contact_offsets, indices)
        elif attribute_name == "rest_offset":
            rest_offsets = apply_randomization_operation_full_tensor(
                view_name, operation, attribute_name, samples, indices, on_reset
            )
            physics_view.set_rest_offsets(rest_offsets, indices)
        elif attribute_name == "tendon_stiffnesses":
            tendon_stiffnesses = apply_randomization_operation(
                view_name, operation, attribute_name, samples, indices, on_reset
            )
            physics._current_tendon_properties["tendon_stiffnesses"][indices] = tendon_stiffnesses
        elif attribute_name == "tendon_dampings":
            tendon_dampings = apply_randomization_operation(
                view_name, operation, attribute_name, samples, indices, on_reset
            )
            physics._current_tendon_properties["tendon_dampings"][indices] = tendon_dampings
        elif attribute_name == "tendon_limit_stiffnesses":
            tendon_limit_stiffnesses = apply_randomization_operation(
                view_name, operation, attribute_name, samples, indices, on_reset
            )
            physics._current_tendon_properties["tendon_limit_stiffnesses"][indices] = tendon_limit_stiffnesses
        elif attribute_name == "tendon_lower_limits":
            tendon_lower_limits = apply_randomization_operation(
                view_name, operation, attribute_name, samples, indices, on_reset
            )
            physics._current_tendon_properties["tendon_lower_limits"][indices] = tendon_lower_limits
        elif attribute_name == "tendon_upper_limits":
            tendon_upper_limits = apply_randomization_operation(
                view_name, operation, attribute_name, samples, indices, on_reset
            )
            physics._current_tendon_properties["tendon_upper_limits"][indices] = tendon_upper_limits
        elif attribute_name == "tendon_rest_lengths":
            tendon_rest_lengths = apply_randomization_operation(
                view_name, operation, attribute_name, samples, indices, on_reset
            )
            physics._current_tendon_properties["tendon_rest_lengths"][indices] = tendon_rest_lengths
        elif attribute_name == "tendon_offsets":
            tendon_offsets = apply_randomization_operation(
                view_name, operation, attribute_name, samples, indices, on_reset
            )
            physics._current_tendon_properties["tendon_offsets"][indices] = tendon_offsets

        if attribute_name in TENDON_ATTRIBUTES:
            current_tendon_limits = np.stack(
                (
                    physics._current_tendon_properties["tendon_lower_limits"],
                    physics._current_tendon_properties["tendon_upper_limits"],
                ),
                axis=-1,
            )
            physics_view.set_fixed_tendon_properties(
                physics._current_tendon_properties["tendon_stiffnesses"],
                physics._current_tendon_properties["tendon_dampings"],
                physics._current_tendon_properties["tendon_limit_stiffnesses"],
                current_tendon_limits,
                physics._current_tendon_properties["tendon_rest_lengths"],
                physics._current_tendon_properties["tendon_offsets"],
                indices,
            )

        db.outputs.execOut = og.ExecutionAttributeState.ENABLED
        return True
