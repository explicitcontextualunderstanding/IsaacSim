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

"""Physics view registration and randomization.

This module provides functions to register and randomize physics views using
SimulationManager for reinforcement learning and sim-to-real applications.
"""

import copy
from typing import Optional, Type, Union

import numpy as np
from isaacsim.core.experimental.prims import Articulation, RigidPrim
from isaacsim.core.experimental.utils.transform import quaternion_to_euler_angles
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.replicator.experimental.domain_randomization.scripts import context
from omni.replicator.core import distribution
from omni.replicator.core.utils import ReplicatorItem, ReplicatorWrapper, utils
from pxr import Gf, UsdPhysics

from .attributes import TENDON_ATTRIBUTES
from .context import trigger_randomization

_simulation_context = None
_physics_sim_view = None
_rigid_prim_views = dict()
_articulation_views = dict()

_simulation_context_initial_values = dict()
_rigid_prim_views_initial_values = dict()
_articulation_views_initial_values = dict()
_current_tendon_properties = dict()

_simulation_context_reset_values = dict()
_rigid_prim_views_reset_values = dict()
_articulation_views_reset_values = dict()


def register_simulation_context(simulation_context: Optional[Type[SimulationManager]] = None):
    """Register SimulationManager for domain randomization.

    Note: Only SimulationManager is supported. Custom subclasses are not supported as
    physics views and scenes are always retrieved from SimulationManager directly.

    Args:
        simulation_context: Optional, defaults to SimulationManager. Pass None to use default.
    """
    global _simulation_context
    global _physics_sim_view

    if simulation_context is None:
        simulation_context = SimulationManager

    _simulation_context = simulation_context
    _physics_sim_view = SimulationManager._physics_sim_view__warp

    # Get gravity from the physics scene
    physics_scenes = SimulationManager.get_physics_scenes()
    if physics_scenes:
        physics_scene_wrapper = physics_scenes[0]
        physics_scene = physics_scene_wrapper.physics_scene
        direction = physics_scene.GetGravityDirectionAttr().Get()
        magnitude = physics_scene.GetGravityMagnitudeAttr().Get()

        # Initialize gravity if invalid. SimulationManager tracks physics scenes without
        # setting defaults, unlike World which calls PhysicsContext(set_defaults=True).
        if direction is None or np.linalg.norm(direction) == 0:
            direction = Gf.Vec3f(0.0, 0.0, -1.0)
            physics_scene.GetGravityDirectionAttr().Set(direction)

        if magnitude is None or not np.isfinite(magnitude) or magnitude <= 0:
            magnitude = 9.81
            physics_scene.GetGravityMagnitudeAttr().Set(magnitude)

        gravity_vector = np.array([direction[0], direction[1], direction[2]]) * magnitude
    else:
        gravity_vector = np.array([0.0, 0.0, -9.81])

    _simulation_context_initial_values["gravity"] = gravity_vector
    _simulation_context_reset_values["gravity"] = copy.deepcopy(gravity_vector)


def register_rigid_prim_view(rigid_prim_view: RigidPrim, name: str):
    """Register a RigidPrim view for domain randomization.

    Args:
        rigid_prim_view: The RigidPrim view to register.
        name: The name to register this view under. Used when calling randomize functions.
    """
    physics_view = rigid_prim_view._physics_rigid_body_view
    count = len(rigid_prim_view)

    _rigid_prim_views[name] = rigid_prim_view
    initial_values = dict()

    pos, quats = rigid_prim_view.get_world_poses()
    initial_values["position"] = np.asarray(pos)
    # Convert quaternions (w, x, y, z) to euler angles
    initial_values["orientation"] = quaternion_to_euler_angles(np.asarray(quats), extrinsic=True).numpy()

    lin_vel, ang_vel = rigid_prim_view.get_velocities()
    initial_values["linear_velocity"] = np.asarray(lin_vel)
    initial_values["angular_velocity"] = np.asarray(ang_vel)
    initial_values["velocity"] = np.concatenate(
        [initial_values["linear_velocity"], initial_values["angular_velocity"]], axis=-1
    )
    initial_values["force"] = np.zeros((count, 3), dtype=np.float32)
    initial_values["mass"] = np.asarray(physics_view.get_masses())
    # Extract the diagonal elements from inertia matrices
    inertias = np.asarray(physics_view.get_inertias())
    initial_values["inertia"] = inertias[:, [0, 4, 8]]
    material_props = np.asarray(physics_view.get_material_properties())
    initial_values["material_properties"] = material_props.reshape(count, physics_view.max_shapes * 3)
    initial_values["contact_offset"] = np.asarray(physics_view.get_contact_offsets())
    initial_values["rest_offset"] = np.asarray(physics_view.get_rest_offsets())

    _rigid_prim_views_initial_values[name] = initial_values
    _rigid_prim_views_reset_values[name] = copy.deepcopy(initial_values)


def register_articulation_view(articulation_view: Articulation, name: str):
    """Register an Articulation view for domain randomization.

    Args:
        articulation_view: The Articulation view to register.
        name: The name to register this view under. Used when calling randomize functions.
    """
    physics_view = articulation_view._physics_articulation_view
    count = len(articulation_view)

    _articulation_views[name] = articulation_view
    initial_values = dict()

    initial_values["stiffness"] = np.asarray(physics_view.get_dof_stiffnesses())
    initial_values["damping"] = np.asarray(physics_view.get_dof_dampings())
    # Use the new friction properties API (returns static, dynamic, viscous)
    static_friction, dynamic_friction, viscous_friction = articulation_view.get_dof_friction_properties()
    initial_values["joint_friction"] = np.asarray(static_friction)

    pos, quats = articulation_view.get_world_poses()
    initial_values["position"] = np.asarray(pos)
    # Convert quaternions (w, x, y, z) to euler angles
    initial_values["orientation"] = quaternion_to_euler_angles(np.asarray(quats), extrinsic=True).numpy()

    lin_vel, ang_vel = articulation_view.get_velocities()
    initial_values["linear_velocity"] = np.asarray(lin_vel)
    initial_values["angular_velocity"] = np.asarray(ang_vel)
    initial_values["velocity"] = np.concatenate(
        [initial_values["linear_velocity"], initial_values["angular_velocity"]], axis=-1
    )
    initial_values["joint_positions"] = np.asarray(articulation_view.get_dof_positions())
    initial_values["joint_velocities"] = np.asarray(articulation_view.get_dof_velocities())

    dof_limits = np.asarray(articulation_view.get_dof_limits())
    initial_values["lower_dof_limits"] = dof_limits[..., 0]
    initial_values["upper_dof_limits"] = dof_limits[..., 1]

    initial_values["max_efforts"] = np.asarray(articulation_view.get_dof_max_efforts())
    initial_values["joint_armatures"] = np.asarray(physics_view.get_dof_armatures())
    initial_values["joint_max_velocities"] = np.asarray(physics_view.get_dof_max_velocities())
    initial_values["joint_efforts"] = np.zeros(
        (initial_values["stiffness"].shape[0], initial_values["stiffness"].shape[1]), dtype=np.float32
    )
    initial_values["body_masses"] = np.asarray(physics_view.get_masses())
    # Extract the diagonal elements from inertia matrices
    inertias = np.asarray(physics_view.get_inertias())
    initial_values["body_inertias"] = inertias[:, :, [0, 4, 8]].reshape(count, physics_view.max_links * 3)

    material_props = np.asarray(physics_view.get_material_properties())
    initial_values["material_properties"] = material_props.reshape(count, physics_view.max_shapes * 3)
    initial_values["contact_offset"] = np.asarray(physics_view.get_contact_offsets())
    initial_values["rest_offset"] = np.asarray(physics_view.get_rest_offsets())

    if physics_view.max_fixed_tendons > 0:
        initial_values["tendon_stiffnesses"] = np.asarray(physics_view.get_fixed_tendon_stiffnesses())
        initial_values["tendon_dampings"] = np.asarray(physics_view.get_fixed_tendon_dampings())
        initial_values["tendon_limit_stiffnesses"] = np.asarray(physics_view.get_fixed_tendon_limit_stiffnesses())
        tendon_limits = np.asarray(physics_view.get_fixed_tendon_limits()).reshape(
            count, physics_view.max_fixed_tendons, 2
        )
        initial_values["tendon_lower_limits"] = tendon_limits[..., 0]
        initial_values["tendon_upper_limits"] = tendon_limits[..., 1]
        initial_values["tendon_rest_lengths"] = np.asarray(physics_view.get_fixed_tendon_rest_lengths())
        initial_values["tendon_offsets"] = np.asarray(physics_view.get_fixed_tendon_offsets())

        for attribute in TENDON_ATTRIBUTES:
            _current_tendon_properties[attribute] = initial_values[attribute].copy()

    _articulation_views_initial_values[name] = initial_values
    _articulation_views_reset_values[name] = copy.deepcopy(initial_values)


def step_randomization(reset_inds: Optional[Union[list, np.ndarray]] = None):
    """Step the randomization with the given reset indices.

    Args:
        reset_inds: The indices corresponding to the prims to be reset in the views.
    """
    if reset_inds is None:
        reset_inds = []
    trigger_randomization(np.asarray(reset_inds))


@ReplicatorWrapper
def _write_physics_view_node(view, attribute, values, operation, node_type, num_buckets=None):
    """Create and configure an OmniGraph node for physics view randomization.

    Args:
        view: The physics view to randomize.
        attribute: The attribute to randomize on the view.
        values: The values or distribution for randomization.
        operation: The operation type for randomization.
        node_type: The OmniGraph node type to create.
        num_buckets: Number of buckets for bucketed randomization.

    Returns:
        The created and configured OmniGraph node.
    """
    node = utils.create_node(node_type)
    node.get_attribute("inputs:attribute").set(attribute)
    node.get_attribute("inputs:prims").set(view)
    node.get_attribute("inputs:operation").set(operation)

    if not isinstance(values, ReplicatorItem):
        values = distribution.uniform(values, values)

    if num_buckets is not None:
        node.get_attribute("inputs:num_buckets").set(num_buckets)
        if values.node.get_node_type().get_node_type() == "omni.replicator.core.OgnSampleUniform":
            node.get_attribute("inputs:distribution").set("uniform")
            values.node.get_attribute("inputs:lower").connect(node.get_attribute("inputs:dist_param_1"), True)
            values.node.get_attribute("inputs:upper").connect(node.get_attribute("inputs:dist_param_2"), True)
        elif values.node.get_node_type().get_node_type() == "omni.replicator.core.OgnSampleNormal":
            node.get_attribute("inputs:distribution").set("gaussian")
            values.node.get_attribute("inputs:mean").connect(node.get_attribute("inputs:dist_param_1"), True)
            values.node.get_attribute("inputs:std").connect(node.get_attribute("inputs:dist_param_2"), True)
        elif values.node.get_node_type().get_node_type() == "omni.replicator.core.OgnSampleLogUniform":
            node.get_attribute("inputs:distribution").set("loguniform")
            values.node.get_attribute("inputs:lower").connect(node.get_attribute("inputs:dist_param_1"), True)
            values.node.get_attribute("inputs:upper").connect(node.get_attribute("inputs:dist_param_2"), True)

    counter = ReplicatorItem(utils.create_node, "isaacsim.replicator.experimental.domain_randomization.OgnCountIndices")

    upstream_node = ReplicatorItem._get_context()
    upstream_node.get_attribute("outputs:indices").connect(counter.node.get_attribute("inputs:indices"), True)
    counter.node.get_attribute("outputs:count").connect(values.node.get_attribute("inputs:numSamples"), True)
    upstream_node.get_attribute("outputs:indices").connect(node.get_attribute("inputs:indices"), True)
    upstream_node.get_attribute("outputs:on_reset").connect(node.get_attribute("inputs:on_reset"), True)

    utils.auto_connect(values.node, node)
    return node


@ReplicatorWrapper
def randomize_rigid_prim_view(
    view_name: str,
    operation: str = "direct",
    num_buckets: int = None,
    position: ReplicatorItem = None,
    orientation: ReplicatorItem = None,
    linear_velocity: ReplicatorItem = None,
    angular_velocity: ReplicatorItem = None,
    velocity: ReplicatorItem = None,
    force: ReplicatorItem = None,
    mass: ReplicatorItem = None,
    inertia: ReplicatorItem = None,
    material_properties: ReplicatorItem = None,
    contact_offset: ReplicatorItem = None,
    rest_offset: ReplicatorItem = None,
):
    """Randomize properties of a registered RigidPrim view.

    Args:
        view_name: The name of a registered RigidPrim view.
        operation: Can be "direct", "additive", or "scaling".
        num_buckets: Number of buckets for material_properties randomization.
        position: Randomizes the position of the prims.
        orientation: Randomizes the orientation using euler angles (rad).
        linear_velocity: Randomizes the linear velocity.
        angular_velocity: Randomizes the angular velocity.
        velocity: Randomizes both linear and angular velocity.
        force: Applies a random force to the prims.
        mass: Randomizes the mass (CPU pipeline only).
        inertia: Randomizes the diagonal inertia elements (CPU pipeline only).
        material_properties: Randomizes static friction, dynamic friction, restitution.
        contact_offset: Randomizes the contact offset.
        rest_offset: Randomizes the rest offset.
    """
    upstream_node_name = ReplicatorItem._get_context().get_node_type().get_node_type()
    if upstream_node_name != "isaacsim.replicator.experimental.domain_randomization.OgnIntervalFiltering":
        raise ValueError(
            "randomize_rigid_prim_view() must be called within the "
            "isaacsim.replicator.experimental.domain_randomization.gate.on_interval "
            "or isaacsim.replicator.experimental.domain_randomization.gate.on_env_reset context managers."
        )

    node_type = "isaacsim.replicator.experimental.domain_randomization.OgnWritePhysicsRigidPrimView"

    if _rigid_prim_views.get(view_name) is None:
        raise ValueError(f"Expected a registered rigid prim view, but instead received {view_name}")

    if position is not None:
        _write_physics_view_node(view_name, "position", position, operation, node_type)
    if orientation is not None:
        _write_physics_view_node(view_name, "orientation", orientation, operation, node_type)
    if linear_velocity is not None:
        _write_physics_view_node(view_name, "linear_velocity", linear_velocity, operation, node_type)
    if angular_velocity is not None:
        _write_physics_view_node(view_name, "angular_velocity", angular_velocity, operation, node_type)
    if velocity is not None:
        _write_physics_view_node(view_name, "velocity", velocity, operation, node_type)
    if force is not None:
        _write_physics_view_node(view_name, "force", force, operation, node_type)
    if mass is not None:
        _write_physics_view_node(view_name, "mass", mass, operation, node_type)
    if inertia is not None:
        _write_physics_view_node(view_name, "inertia", inertia, operation, node_type)
    if material_properties is not None:
        _write_physics_view_node(
            view_name, "material_properties", material_properties, operation, node_type, num_buckets
        )
    if contact_offset is not None:
        _write_physics_view_node(view_name, "contact_offset", contact_offset, operation, node_type)
    if rest_offset is not None:
        _write_physics_view_node(view_name, "rest_offset", rest_offset, operation, node_type)


@ReplicatorWrapper
def randomize_articulation_view(
    view_name: str,
    operation: str = "direct",
    num_buckets: int = None,
    stiffness: ReplicatorItem = None,
    damping: ReplicatorItem = None,
    joint_friction: ReplicatorItem = None,
    position: ReplicatorItem = None,
    orientation: ReplicatorItem = None,
    linear_velocity: ReplicatorItem = None,
    angular_velocity: ReplicatorItem = None,
    velocity: ReplicatorItem = None,
    joint_positions: ReplicatorItem = None,
    joint_velocities: ReplicatorItem = None,
    lower_dof_limits: ReplicatorItem = None,
    upper_dof_limits: ReplicatorItem = None,
    max_efforts: ReplicatorItem = None,
    joint_armatures: ReplicatorItem = None,
    joint_max_velocities: ReplicatorItem = None,
    joint_efforts: ReplicatorItem = None,
    body_masses: ReplicatorItem = None,
    body_inertias: ReplicatorItem = None,
    material_properties: ReplicatorItem = None,
    tendon_stiffnesses: ReplicatorItem = None,
    tendon_dampings: ReplicatorItem = None,
    tendon_limit_stiffnesses: ReplicatorItem = None,
    tendon_lower_limits: ReplicatorItem = None,
    tendon_upper_limits: ReplicatorItem = None,
    tendon_rest_lengths: ReplicatorItem = None,
    tendon_offsets: ReplicatorItem = None,
):
    """Randomize properties of a registered Articulation view.

    Args:
        view_name: The name of a registered Articulation view.
        operation: Can be "direct", "additive", or "scaling".
        num_buckets: Number of buckets for material_properties randomization.
        stiffness: Randomizes the joint stiffness.
        damping: Randomizes the joint damping.
        joint_friction: Randomizes the joint friction.
        position: Randomizes the root position.
        orientation: Randomizes the root orientation using euler angles (rad).
        linear_velocity: Randomizes the root linear velocity.
        angular_velocity: Randomizes the root angular velocity.
        velocity: Randomizes the root linear and angular velocity.
        joint_positions: Randomizes the joint positions.
        joint_velocities: Randomizes the joint velocities.
        lower_dof_limits: Randomizes the lower joint limits.
        upper_dof_limits: Randomizes the upper joint limits.
        max_efforts: Randomizes the maximum joint efforts.
        joint_armatures: Randomizes the joint armatures.
        joint_max_velocities: Randomizes the maximum joint velocities.
        joint_efforts: Randomizes the joint efforts.
        body_masses: Randomizes the body masses.
        body_inertias: Randomizes the body inertias.
        material_properties: Randomizes the material properties.
        tendon_stiffnesses: Randomizes fixed tendon stiffnesses.
        tendon_dampings: Randomizes fixed tendon dampings.
        tendon_limit_stiffnesses: Randomizes fixed tendon limit stiffnesses.
        tendon_lower_limits: Randomizes fixed tendon lower limits.
        tendon_upper_limits: Randomizes fixed tendon upper limits.
        tendon_rest_lengths: Randomizes fixed tendon rest lengths.
        tendon_offsets: Randomizes fixed tendon offsets.
    """
    upstream_node_name = ReplicatorItem._get_context().get_node_type().get_node_type()
    if upstream_node_name != "isaacsim.replicator.experimental.domain_randomization.OgnIntervalFiltering":
        raise ValueError(
            "randomize_articulation_view() must be called within the "
            "isaacsim.replicator.experimental.domain_randomization.gate.on_interval "
            "or isaacsim.replicator.experimental.domain_randomization.gate.on_env_reset context managers."
        )

    node_type = "isaacsim.replicator.experimental.domain_randomization.OgnWritePhysicsArticulationView"

    if _articulation_views.get(view_name) is None:
        raise ValueError(f"Expected a registered articulation view, but instead received {view_name}")

    tendon_nodes = list()

    if stiffness is not None:
        _write_physics_view_node(view_name, "stiffness", stiffness, operation, node_type)
    if damping is not None:
        _write_physics_view_node(view_name, "damping", damping, operation, node_type)
    if joint_friction is not None:
        _write_physics_view_node(view_name, "joint_friction", joint_friction, operation, node_type)
    if position is not None:
        _write_physics_view_node(view_name, "position", position, operation, node_type)
    if orientation is not None:
        _write_physics_view_node(view_name, "orientation", orientation, operation, node_type)
    if linear_velocity is not None:
        _write_physics_view_node(view_name, "linear_velocity", linear_velocity, operation, node_type)
    if angular_velocity is not None:
        _write_physics_view_node(view_name, "angular_velocity", angular_velocity, operation, node_type)
    if velocity is not None:
        _write_physics_view_node(view_name, "velocity", velocity, operation, node_type)
    if joint_positions is not None:
        _write_physics_view_node(view_name, "joint_positions", joint_positions, operation, node_type)
    if joint_velocities is not None:
        _write_physics_view_node(view_name, "joint_velocities", joint_velocities, operation, node_type)
    if lower_dof_limits is not None:
        _write_physics_view_node(view_name, "lower_dof_limits", lower_dof_limits, operation, node_type)
    if upper_dof_limits is not None:
        _write_physics_view_node(view_name, "upper_dof_limits", upper_dof_limits, operation, node_type)
    if max_efforts is not None:
        _write_physics_view_node(view_name, "max_efforts", max_efforts, operation, node_type)
    if joint_armatures is not None:
        _write_physics_view_node(view_name, "joint_armatures", joint_armatures, operation, node_type)
    if joint_max_velocities is not None:
        _write_physics_view_node(view_name, "joint_max_velocities", joint_max_velocities, operation, node_type)
    if joint_efforts is not None:
        _write_physics_view_node(view_name, "joint_efforts", joint_efforts, operation, node_type)
    if body_masses is not None:
        _write_physics_view_node(view_name, "body_masses", body_masses, operation, node_type)
    if body_inertias is not None:
        _write_physics_view_node(view_name, "body_inertias", body_inertias, operation, node_type)
    if material_properties is not None:
        _write_physics_view_node(
            view_name, "material_properties", material_properties, operation, node_type, num_buckets
        )
    if tendon_stiffnesses is not None:
        tendon_nodes.append(
            _write_physics_view_node(view_name, "tendon_stiffnesses", tendon_stiffnesses, operation, node_type).node
        )
    if tendon_dampings is not None:
        tendon_nodes.append(
            _write_physics_view_node(view_name, "tendon_dampings", tendon_dampings, operation, node_type).node
        )
    if tendon_limit_stiffnesses is not None:
        tendon_nodes.append(
            _write_physics_view_node(
                view_name, "tendon_limit_stiffnesses", tendon_limit_stiffnesses, operation, node_type
            ).node
        )
    if tendon_lower_limits is not None:
        tendon_nodes.append(
            _write_physics_view_node(view_name, "tendon_lower_limits", tendon_lower_limits, operation, node_type).node
        )
    if tendon_upper_limits is not None:
        tendon_nodes.append(
            _write_physics_view_node(view_name, "tendon_upper_limits", tendon_upper_limits, operation, node_type).node
        )
    if tendon_rest_lengths is not None:
        tendon_nodes.append(
            _write_physics_view_node(view_name, "tendon_rest_lengths", tendon_rest_lengths, operation, node_type).node
        )
    if tendon_offsets is not None:
        tendon_nodes.append(
            _write_physics_view_node(view_name, "tendon_offsets", tendon_offsets, operation, node_type).node
        )

    # Convert tendon nodes to sequential execution
    if len(tendon_nodes) > 0:
        for node in tendon_nodes:
            upstream_tendon_node = context._context.get_tendon_exec_context()
            context._context.add_tendon_exec_context(node)
            if upstream_tendon_node is not None:
                utils._disconnect(node.get_attribute("inputs:execIn"))
                upstream_tendon_node.get_attribute("outputs:execOut").connect(node.get_attribute("inputs:execIn"), True)


@ReplicatorWrapper
def randomize_simulation_context(operation: str = "direct", gravity: ReplicatorItem = None):
    """Randomize properties of the registered SimulationContext.

    Args:
        operation: Can be "direct", "additive", or "scaling".
        gravity: Randomizes the gravity vector.
    """
    upstream_node_name = ReplicatorItem._get_context().get_node_type().get_node_type()
    if upstream_node_name != "isaacsim.replicator.experimental.domain_randomization.OgnIntervalFiltering":
        raise ValueError(
            "randomize_simulation_context() must be called within the "
            "isaacsim.replicator.experimental.domain_randomization.gate.on_interval "
            "or isaacsim.replicator.experimental.domain_randomization.gate.on_env_reset context managers."
        )

    node_type = "isaacsim.replicator.experimental.domain_randomization.OgnWritePhysicsSimulationContext"

    global _simulation_context
    if _simulation_context is None:
        raise ValueError("Expected a registered simulation context")

    if gravity is not None:
        _write_physics_view_node("simulation_context", "gravity", gravity, operation, node_type)
