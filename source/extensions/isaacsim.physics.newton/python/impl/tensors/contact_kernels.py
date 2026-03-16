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
"""Additional Warp kernels for Newton contact sensor operations.

These kernels extend Newton's ContactSensor capabilities to match PhysX tensor API.

IMPORTANT: These kernels use pre-computed contact forces from Newton's solver, NOT
recomputed forces. The contact forces are populated by solvers (e.g., MuJoCo solver)
which add `contacts.force` (scalar magnitude) and `contacts.normal` (vec3) fields
dynamically. This ensures consistency with what the solver actually applied.

See:
- newton/_src/sensors/contact_sensor.py - ContactSensor using pre-computed forces
- newton/_src/solvers/mujoco/solver_mujoco.py - Where force/normal fields are added
"""

import warp as wp


@wp.kernel
def count_contacts_per_pair_kernel(
    # inputs - from Newton's contacts
    contact_count_total: wp.array(dtype=int),
    contact_pair: wp.array(dtype=wp.vec2i),  # MuJoCo uses pair instead of separate arrays
    # model data
    shape_body: wp.array(dtype=int),
    # sensor mapping
    body_sensor_map: wp.array(dtype=int),
    body_filter_map: wp.array2d(dtype=int),
    # outputs
    contact_counts: wp.array2d(dtype=wp.uint32),  # per sensor-filter pair (uint32 to match PhysX API)
):
    """Count how many contacts exist for each sensor-filter pair (for prefix scan).

    Args:
        contact_count_total: Total number of contacts.
        contact_pair: Contact shape pairs (vec2i).
        shape_body: Shape to body mapping.
        body_sensor_map: Body to sensor mapping.
        body_filter_map: Body filter mapping (2D).
        contact_counts: Output contact counts per sensor-filter pair.
    """
    tid = wp.tid()
    count = contact_count_total[0]
    if tid >= count:
        return

    pair = contact_pair[tid]
    shape_a = pair[0]
    shape_b = pair[1]

    if shape_a == shape_b or shape_a < 0 or shape_b < 0:
        return

    # Get bodies for both shapes
    body_a = shape_body[shape_a]
    body_b = shape_body[shape_b]

    # Check if either body is a sensor (with bounds checking)
    sensor_a = -1
    if body_a >= 0 and body_a < body_sensor_map.shape[0]:
        sensor_a = body_sensor_map[body_a]

    sensor_b = -1
    if body_b >= 0 and body_b < body_sensor_map.shape[0]:
        sensor_b = body_sensor_map[body_b]

    if sensor_a < 0 and sensor_b < 0:
        return

    # Increment count for appropriate sensor-filter pairs
    if sensor_a >= 0 and body_b >= 0 and body_b < body_filter_map.shape[1]:
        filter_idx = body_filter_map[sensor_a, body_b]
        if filter_idx >= 0:
            wp.atomic_add(contact_counts, sensor_a, filter_idx, wp.uint32(1))

    if sensor_b >= 0 and body_a >= 0 and body_a < body_filter_map.shape[1]:
        filter_idx = body_filter_map[sensor_b, body_a]
        if filter_idx >= 0:
            wp.atomic_add(contact_counts, sensor_b, filter_idx, wp.uint32(1))


@wp.kernel
def net_contact_forces_kernel(
    # inputs - from Newton's contacts (pre-computed by solver)
    contact_count: wp.array(dtype=int),
    contact_pair: wp.array(dtype=wp.vec2i),  # MuJoCo uses pair instead of separate arrays
    contact_normal: wp.array(dtype=wp.vec3),
    contact_force: wp.array(dtype=wp.float32),  # Pre-computed force magnitude from solver
    # model data
    shape_body: wp.array(dtype=int),
    # sensor mapping
    body_sensor_map: wp.array(dtype=int),  # maps body index -> sensor index
    # output
    net_forces: wp.array2d(dtype=wp.float32),  # shape: (sensor_count, 3)
):
    """Compute net contact forces per sensor using pre-computed forces from Newton's solver.

    This is the correct implementation that matches Newton's ContactSensor.select_aggregate_net_force,
    using forces that were actually applied by the solver rather than recomputing them.

    Args:
        contact_count: Total number of contacts.
        contact_pair: Contact shape pairs (vec2i).
        contact_normal: Contact normals (vec3).
        contact_force: Pre-computed force magnitudes from solver.
        shape_body: Shape to body mapping.
        body_sensor_map: Body to sensor mapping.
        net_forces: Output net forces per sensor (sensor_count, 3).
    """
    tid = wp.tid()
    count = contact_count[0]
    if tid >= count:
        return

    pair = contact_pair[tid]
    shape_a = pair[0]
    shape_b = pair[1]

    if shape_a == shape_b or shape_a < 0 or shape_b < 0:
        return

    # Get bodies for both shapes
    body_a = shape_body[shape_a]
    body_b = shape_body[shape_b]

    # Check if either body is a sensor (with bounds checking)
    sensor_a = -1
    if body_a >= 0 and body_a < body_sensor_map.shape[0]:
        sensor_a = body_sensor_map[body_a]

    sensor_b = -1
    if body_b >= 0 and body_b < body_sensor_map.shape[0]:
        sensor_b = body_sensor_map[body_b]

    if sensor_a < 0 and sensor_b < 0:
        return

    # Use pre-computed force from Newton's solver (same as ContactSensor)
    n = contact_normal[tid]
    force_magnitude = contact_force[tid]
    f_total = force_magnitude * n

    # Add force to appropriate sensor
    if sensor_a >= 0:
        wp.atomic_add(net_forces, sensor_a, 0, f_total[0])
        wp.atomic_add(net_forces, sensor_a, 1, f_total[1])
        wp.atomic_add(net_forces, sensor_a, 2, f_total[2])

    if sensor_b >= 0:
        wp.atomic_sub(net_forces, sensor_b, 0, f_total[0])
        wp.atomic_sub(net_forces, sensor_b, 1, f_total[1])
        wp.atomic_sub(net_forces, sensor_b, 2, f_total[2])


@wp.kernel
def contact_force_matrix_kernel(
    # inputs - from Newton's contacts (pre-computed by solver)
    contact_count: wp.array(dtype=int),
    contact_pair: wp.array(dtype=wp.vec2i),  # MuJoCo uses pair instead of separate arrays
    contact_normal: wp.array(dtype=wp.vec3),
    contact_force: wp.array(dtype=wp.float32),  # Pre-computed force magnitude from solver
    # model data
    shape_body: wp.array(dtype=int),
    # sensor mapping
    body_sensor_map: wp.array(dtype=int),  # maps body index -> sensor index
    body_filter_map: wp.array2d(dtype=int),  # maps (sensor_idx, body_idx) -> filter_idx (-1 if no filter)
    filter_count: int,
    # output
    force_matrix: wp.array3d(dtype=wp.float32),  # shape: (sensor_count, filter_count, 3)
):
    """Compute per-filter contact force matrix using pre-computed forces from Newton's solver.

    Uses the same approach as Newton's ContactSensor.select_aggregate_net_force kernel,
    but organizes forces into a sensor × filter matrix.

    Args:
        contact_count: Total number of contacts.
        contact_pair: Contact shape pairs (vec2i).
        contact_normal: Contact normals (vec3).
        contact_force: Pre-computed force magnitudes from solver.
        shape_body: Shape to body mapping.
        body_sensor_map: Body to sensor mapping.
        body_filter_map: Sensor-body to filter index mapping.
        filter_count: Number of filters.
        force_matrix: Output force matrix (sensor_count, filter_count, 3).
    """
    tid = wp.tid()
    count = contact_count[0]
    if tid >= count:
        return

    pair = contact_pair[tid]
    shape_a = pair[0]
    shape_b = pair[1]

    if shape_a == shape_b or shape_a < 0 or shape_b < 0:
        return

    # Get bodies for both shapes
    body_a = shape_body[shape_a]
    body_b = shape_body[shape_b]

    # Check if either body is a sensor (with bounds checking)
    sensor_a = -1
    if body_a >= 0 and body_a < body_sensor_map.shape[0]:
        sensor_a = body_sensor_map[body_a]

    sensor_b = -1
    if body_b >= 0 and body_b < body_sensor_map.shape[0]:
        sensor_b = body_sensor_map[body_b]

    if sensor_a < 0 and sensor_b < 0:
        return

    # Use pre-computed force from Newton's solver (same as ContactSensor)
    n = contact_normal[tid]
    force_magnitude = contact_force[tid]
    f_total = force_magnitude * n

    # Add force to appropriate sensor-filter pair (with bounds checking)
    if sensor_a >= 0 and body_b >= 0 and body_b < body_filter_map.shape[1]:
        filter_idx = body_filter_map[sensor_a, body_b]
        if filter_idx >= 0:
            wp.atomic_add(force_matrix, sensor_a, filter_idx, 0, f_total[0])
            wp.atomic_add(force_matrix, sensor_a, filter_idx, 1, f_total[1])
            wp.atomic_add(force_matrix, sensor_a, filter_idx, 2, f_total[2])

    if sensor_b >= 0 and body_a >= 0 and body_a < body_filter_map.shape[1]:
        filter_idx = body_filter_map[sensor_b, body_a]
        if filter_idx >= 0:
            wp.atomic_sub(force_matrix, sensor_b, filter_idx, 0, f_total[0])
            wp.atomic_sub(force_matrix, sensor_b, filter_idx, 1, f_total[1])
            wp.atomic_sub(force_matrix, sensor_b, filter_idx, 2, f_total[2])


@wp.kernel
def contact_data_kernel(
    # inputs - from Newton's contacts (pre-computed by solver)
    contact_count: wp.array(dtype=int),
    contact_pair: wp.array(dtype=wp.vec2i),  # MuJoCo uses pair instead of separate arrays
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_force: wp.array(dtype=wp.float32),  # Pre-computed force magnitude from solver
    contact_thickness0: wp.array(dtype=wp.float32),
    contact_thickness1: wp.array(dtype=wp.float32),
    # model data
    shape_body: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    # sensor mapping
    body_sensor_map: wp.array(dtype=int),
    body_filter_map: wp.array2d(dtype=int),
    max_contact_data_count: int,
    # outputs
    contact_forces: wp.array2d(dtype=wp.float32),  # force magnitudes (N, 1)
    contact_points: wp.array2d(dtype=wp.float32),  # contact points (N, 3)
    contact_normals: wp.array2d(dtype=wp.float32),  # contact normals (N, 3)
    contact_separations: wp.array2d(dtype=wp.float32),  # penetration depth (N, 1)
    contact_counts: wp.array2d(dtype=wp.uint32),  # per sensor-filter pair (uint32 to match PhysX API)
    contact_start_indices: wp.array2d(
        dtype=wp.uint32
    ),  # start index for each sensor-filter pair (uint32 to match PhysX API)
):
    """Gather detailed contact data per sensor-filter pair using pre-computed forces.

    Uses Newton's solver-computed forces, but gathers detailed geometric information
    (contact points, normals, separations) per sensor-filter pair.

    Args:
        contact_count: Total number of contacts.
        contact_pair: Contact shape pairs (vec2i).
        contact_point0: Contact points on first shape (vec3).
        contact_point1: Contact points on second shape (vec3).
        contact_normal: Contact normals (vec3).
        contact_force: Pre-computed force magnitudes from solver.
        contact_thickness0: Thickness of first shape at contact.
        contact_thickness1: Thickness of second shape at contact.
        shape_body: Shape to body mapping.
        body_q: Body transforms.
        body_sensor_map: Body to sensor mapping.
        body_filter_map: Sensor-body to filter index mapping.
        max_contact_data_count: Maximum contacts to store.
        contact_forces: Output force magnitudes (N, 1).
        contact_points: Output contact points (N, 3).
        contact_normals: Output contact normals (N, 3).
        contact_separations: Output penetration depths (N, 1).
        contact_counts: Output contact counts per pair.
        contact_start_indices: Output start indices per pair.
    """
    tid = wp.tid()
    count = contact_count[0]
    if tid >= count:
        return

    pair = contact_pair[tid]
    shape_a = pair[0]
    shape_b = pair[1]

    if shape_a == shape_b or shape_a < 0 or shape_b < 0:
        return

    body_a = shape_body[shape_a]
    body_b = shape_body[shape_b]

    # Check if either body is a sensor (with bounds checking)
    sensor_a = -1
    if body_a >= 0 and body_a < body_sensor_map.shape[0]:
        sensor_a = body_sensor_map[body_a]

    sensor_b = -1
    if body_b >= 0 and body_b < body_sensor_map.shape[0]:
        sensor_b = body_sensor_map[body_b]

    if sensor_a < 0 and sensor_b < 0:
        return

    # Get pre-computed force from Newton's solver
    force_magnitude = contact_force[tid]

    # Get contact geometry
    n = contact_normal[tid]
    bx_a = contact_point0[tid]
    bx_b = contact_point1[tid]
    thickness_a = contact_thickness0[tid]
    thickness_b = contact_thickness1[tid]

    # Transform contact points to world space
    X_wb_a = body_q[body_a]
    X_wb_b = body_q[body_b]

    bx_a_world = wp.transform_point(X_wb_a, bx_a) - thickness_a * n
    bx_b_world = wp.transform_point(X_wb_b, bx_b) + thickness_b * n

    # Penetration depth (negative = penetrating)
    d = wp.dot(n, bx_a_world - bx_b_world)

    # Contact point (average of two points)
    contact_point = (bx_a_world + bx_b_world) * 0.5

    # Store data for sensor A (with bounds checking)
    if sensor_a >= 0 and body_b >= 0 and body_b < body_filter_map.shape[1]:
        filter_idx = body_filter_map[sensor_a, body_b]
        if filter_idx >= 0:
            # Atomically increment count and get index
            data_idx = wp.atomic_add(contact_counts, sensor_a, filter_idx, wp.uint32(1))
            start_idx = int(contact_start_indices[sensor_a, filter_idx])
            write_idx = start_idx + int(data_idx)

            if write_idx < max_contact_data_count:
                # Write to 2D arrays matching PhysX format
                contact_forces[write_idx, 0] = force_magnitude
                contact_points[write_idx, 0] = contact_point[0]
                contact_points[write_idx, 1] = contact_point[1]
                contact_points[write_idx, 2] = contact_point[2]
                contact_normals[write_idx, 0] = n[0]
                contact_normals[write_idx, 1] = n[1]
                contact_normals[write_idx, 2] = n[2]
                contact_separations[write_idx, 0] = d

    # Store data for sensor B (with negated force and normal, with bounds checking)
    if sensor_b >= 0 and body_a >= 0 and body_a < body_filter_map.shape[1]:
        filter_idx = body_filter_map[sensor_b, body_a]
        if filter_idx >= 0:
            data_idx = wp.atomic_add(contact_counts, sensor_b, filter_idx, wp.uint32(1))
            start_idx = int(contact_start_indices[sensor_b, filter_idx])
            write_idx = start_idx + int(data_idx)

            if write_idx < max_contact_data_count:
                # Write to 2D arrays matching PhysX format (negated for sensor B)
                contact_forces[write_idx, 0] = -force_magnitude
                contact_points[write_idx, 0] = contact_point[0]
                contact_points[write_idx, 1] = contact_point[1]
                contact_points[write_idx, 2] = contact_point[2]
                contact_normals[write_idx, 0] = -n[0]
                contact_normals[write_idx, 1] = -n[1]
                contact_normals[write_idx, 2] = -n[2]
                contact_separations[write_idx, 0] = -d
