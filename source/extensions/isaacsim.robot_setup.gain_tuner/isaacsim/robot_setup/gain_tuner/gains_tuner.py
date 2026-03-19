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

"""Module for tuning joint gains on articulated robots through test signal generation and inertia analysis."""


import asyncio
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Set, Tuple

import carb
import numpy as np
import omni
import omni.timeline as timeline
import pxr
import usd.schema.isaac.robot_schema as robot_schema
import usd.schema.isaac.robot_schema.utils as rs_utils
from isaacsim.core.experimental.prims import Articulation
from pxr import Gf, Sdf, Usd, UsdPhysics


class GainsTestMode(IntEnum):
    """Enumeration of available test signal modes for gain tuning."""

    SINUSOIDAL = 0
    """Test mode that generates sinusoidal signals for gain tuning analysis."""
    STEP = 1
    """Test mode that generates step response signals for gain tuning analysis."""
    USER_PROVIDED = 2
    """Test mode that allows custom user-defined signals for gain tuning analysis."""


class JointMode(IntEnum):
    """Enumeration of joint control modes."""

    POSITION = 0
    """Position control mode where joints are commanded to specific position targets."""
    VELOCITY = 1
    """Velocity control mode where joints are commanded to specific velocity targets."""
    NONE = 2
    """No control mode active for the joint."""


@dataclass
class JointListEntry:
    """Data structure representing a joint entry in the gain tuner.

    Args:
        joint: The USD prim representing the joint.
        display_name: Human-readable name for UI display.
        dof_index: Index of this degree of freedom in the articulation.
        drive_axis: Axis token for D6 joints, None for single-axis joints.
    """

    joint: pxr.Usd.Prim
    display_name: str
    dof_index: int
    drive_axis: str


_D6_AXIS_TOKENS = [
    pxr.UsdPhysics.Tokens.transX,
    pxr.UsdPhysics.Tokens.transY,
    pxr.UsdPhysics.Tokens.transZ,
    pxr.UsdPhysics.Tokens.rotX,
    pxr.UsdPhysics.Tokens.rotY,
    pxr.UsdPhysics.Tokens.rotZ,
]

_AXIS_TOKEN_TO_VECTOR = {
    "X": Gf.Vec3f(1, 0, 0),
    "Y": Gf.Vec3f(0, 1, 0),
    "Z": Gf.Vec3f(0, 0, 1),
}


def _extract_d6_axis_token(dof_name: str):
    """Extract the D6 axis token from a DOF name if present.

    Args:
        dof_name: Name of the degree of freedom.

    Returns:
        The matching axis token, or None if no match found.
    """
    lower_name = dof_name.lower()
    for token in _D6_AXIS_TOKENS:
        if token.lower() in lower_name:
            return token
    return None


def _assign_d6_axis_token(joint_identifier: str, dof_name: str, usage_map: Dict[str, Set]) -> str:
    """Assign an unused D6 axis token for a joint.

    Args:
        joint_identifier: Unique identifier for the joint.
        dof_name: Name of the degree of freedom.
        usage_map: Map tracking which axes have been used per joint.

    Returns:
        The assigned axis token, or None if all axes are used.
    """
    token = _extract_d6_axis_token(dof_name)
    used_axes = usage_map.setdefault(joint_identifier, set())
    if token and token not in used_axes:
        used_axes.add(token)
        return token
    for candidate in _D6_AXIS_TOKENS:
        if candidate not in used_axes:
            used_axes.add(candidate)
            return candidate
    return None


def _d6_axis_has_unlocked_limit(joint_prim: pxr.Usd.Prim, axis_token: str) -> bool:
    """Check if a D6 joint axis has unlocked (non-fixed) limits.

    Args:
        joint_prim: The USD prim representing the joint.
        axis_token: The axis token to check.

    Returns:
        True if the axis has a valid range (lower < upper), False otherwise.
    """
    limit_api = pxr.UsdPhysics.LimitAPI.Get(joint_prim, axis_token)
    if not limit_api:
        return False
    lower_attr = limit_api.GetLowAttr()
    upper_attr = limit_api.GetHighAttr()
    if not lower_attr or not upper_attr:
        return False
    lower = lower_attr.Get()
    upper = upper_attr.Get()
    if lower is None or upper is None:
        return False
    try:
        return float(lower) < float(upper)
    except (TypeError, ValueError):
        return False


def _format_d6_display_name(joint_prim: pxr.Usd.Prim, dof_name: str, axis_token: str) -> str:
    """Format a display name for a D6 joint axis.

    Args:
        joint_prim: The USD prim representing the joint.
        dof_name: Name of the degree of freedom.
        axis_token: The axis token for this DOF.

    Returns:
        Formatted display name in the form "joint_name:axis".
    """
    suffix = axis_token if dof_name == joint_prim.GetName() else dof_name
    return f"{joint_prim.GetName()}:{suffix}"


def get_original_spec_for_drive_API(
    stage: pxr.Usd.Stage, joint_drive_path: str, drive_type: str
) -> pxr.Sdf.PropertySpec:
    """Get the original property spec where a drive attribute was authored.

    Args:
        stage: The USD stage.
        joint_drive_path: Path to the joint with the drive API.
        drive_type: The drive type (axis token).

    Returns:
        The original property spec, or None if not found.
    """
    drive_prim = stage.GetPrimAtPath(joint_drive_path)
    attr = pxr.UsdPhysics.DriveAPI(drive_prim, drive_type).GetStiffnessAttr()
    if attr:
        composition_stack = attr.GetPropertyStack()
        if composition_stack:
            return composition_stack[0]
    return None


def _gf_matrix3f_to_numpy(matrix: Gf.Matrix3f) -> np.ndarray:
    """Convert a Gf.Matrix3f to a numpy array.

    Args:
        matrix: The 3x3 Gf matrix.

    Returns:
        A 3x3 numpy array.
    """
    return np.array([[matrix[i][j] for j in range(3)] for i in range(3)])


def matrix_norm(matrix: Gf.Matrix3f) -> float:
    """Compute the Frobenius norm of a 3x3 matrix.

    Args:
        matrix: The 3x3 inertia tensor.

    Returns:
        The Frobenius norm of the matrix.
    """
    return np.linalg.norm(_gf_matrix3f_to_numpy(matrix))


def project_inertia_onto_axis(inertia_matrix: Gf.Matrix3f, axis: Gf.Vec3f) -> float:
    """Project an inertia tensor onto a specific axis to get the scalar moment of inertia.

    For a rotation about axis n, the effective moment of inertia is I_n = n^T * I * n.

    Args:
        inertia_matrix: The 3x3 inertia tensor.
        axis: Unit vector representing the rotation axis direction.

    Returns:
        Scalar moment of inertia about the specified axis.
    """
    axis_np = np.array([axis[0], axis[1], axis[2]])
    axis_norm = np.linalg.norm(axis_np)
    if axis_norm < 1e-9:
        return 0.0
    axis_normalized = axis_np / axis_norm
    inertia_np = _gf_matrix3f_to_numpy(inertia_matrix)
    return float(axis_normalized @ inertia_np @ axis_normalized)


def _get_axis_attr_from_joint(joint_prim: pxr.Usd.Prim) -> pxr.Usd.Attribute:
    """Get the axis attribute from a revolute or prismatic joint.

    Args:
        joint_prim: The USD prim representing the joint.

    Returns:
        The axis attribute, or None if not a supported joint type.
    """
    if joint_prim.IsA(pxr.UsdPhysics.RevoluteJoint):
        return pxr.UsdPhysics.RevoluteJoint(joint_prim).GetAxisAttr()
    elif joint_prim.IsA(pxr.UsdPhysics.PrismaticJoint):
        return pxr.UsdPhysics.PrismaticJoint(joint_prim).GetAxisAttr()
    return None


def get_joint_axis_world_direction(joint_prim: pxr.Usd.Prim, joint_pose: Gf.Matrix4d) -> Gf.Vec3f:
    """Get the world direction of the joint rotation/translation axis.

    Args:
        joint_prim: The USD prim representing the joint.
        joint_pose: The joint's pose matrix in robot coordinates.

    Returns:
        Unit vector representing the joint axis direction in world coordinates.
    """
    local_axis = Gf.Vec3f(1, 0, 0)  # Default X axis

    axis_attr = _get_axis_attr_from_joint(joint_prim)
    if axis_attr:
        axis_token = axis_attr.Get()
        local_axis = _AXIS_TOKEN_TO_VECTOR.get(axis_token, local_axis)

    # Transform local axis to world coordinates using the rotation part of joint pose
    local_vec = Gf.Vec3d(local_axis[0], local_axis[1], local_axis[2])
    origin = Gf.Vec3d(0, 0, 0)
    transformed_origin = joint_pose.Transform(origin)
    transformed_point = joint_pose.Transform(local_vec)
    world_axis = transformed_point - transformed_origin
    return Gf.Vec3f(world_axis[0], world_axis[1], world_axis[2]).GetNormalized()


def _compute_outer_product_matrix(displacement: Gf.Vec3f) -> Gf.Matrix3f:
    """Compute the outer product matrix d * d^T for parallel axis theorem.

    Args:
        displacement: The displacement vector.

    Returns:
        The 3x3 outer product matrix.
    """
    d = displacement
    return Gf.Matrix3f(
        d[0] * d[0],
        d[0] * d[1],
        d[0] * d[2],
        d[1] * d[0],
        d[1] * d[1],
        d[1] * d[2],
        d[2] * d[0],
        d[2] * d[1],
        d[2] * d[2],
    )


def _compute_displacement_inertia_term(displacement: Gf.Vec3f, mass: float) -> Gf.Matrix3f:
    """Compute the displacement term for parallel axis theorem: m * (||d||^2 * I - d*d^T).

    Args:
        displacement: Vector from center of mass to new origin.
        mass: Mass of the object.

    Returns:
        The displacement contribution to the inertia tensor.
    """
    d_squared = displacement[0] ** 2 + displacement[1] ** 2 + displacement[2] ** 2
    identity = Gf.Matrix3f(1.0)
    outer_product = _compute_outer_product_matrix(displacement)

    displacement_term = Gf.Matrix3f(identity)
    displacement_term *= d_squared
    displacement_term -= outer_product
    displacement_term *= mass
    return displacement_term


def compute_parallel_axis_inertia(
    center_of_mass_inertia: Gf.Matrix3f, mass: float, displacement: Gf.Vec3f
) -> Gf.Matrix3f:
    """Apply the parallel axis theorem to translate an inertia tensor.

    Args:
        center_of_mass_inertia: Inertia tensor about center of mass.
        mass: Mass of the object.
        displacement: Vector from center of mass to new origin.

    Returns:
        Inertia tensor about the new origin.
    """
    displacement_term = _compute_displacement_inertia_term(displacement, mass)
    result = Gf.Matrix3f(center_of_mass_inertia)
    result += displacement_term
    return result


def find_articulation_root(stage: pxr.Usd.Stage, robot_path: str) -> str:
    """Find the articulation root prim for a robot.

    Args:
        stage: The USD stage.
        robot_path: Path to the robot prim.

    Returns:
        Path to the articulation root, or None if not found.
    """
    articulations = [
        prim for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_path)) if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    if articulations:
        return str(articulations[0].GetPath())
    if robot_path == "/":
        return None
    return find_articulation_root(stage, pxr.Sdf.Path(robot_path).GetParentPath())


class GainTuner:
    """Controller for tuning joint gains on articulated robots.

    This class provides functionality to:
    - Set up and configure robots for gain tuning.
    - Run sinusoidal or step response tests.
    - Compute inertia properties for gain estimation.
    - Record and analyze joint state data.
    """

    def __init__(self):
        """Initialize the gain tuner with default values."""
        self._timeline = timeline.get_timeline_interface()
        self._test_duration = 5.0
        self.reset()

    def reset(self):
        """Reset all internal state to initial values."""
        self._initialized = False
        self._articulation = None
        self._articulation_root = None
        self._robot_prim_path = None
        self._robot = None
        self._joint_position_commands = []
        self._joint_velocity_commands = []
        self._observed_joint_positions = []
        self._observed_joint_velocities = []
        self._command_times = []
        self._joint_indices = None
        self._data_ready = False
        self._test_timestep = 0
        self._gains_test_generator = None
        self._joint_accumulated_inertia = {}
        self._joint_entries = []
        self.step = 0

    def stop_test(self):
        """Stop the current test and reset the articulation to default state."""
        self._articulation.reset_to_default_state()

    def on_reset(self):
        """Handle simulation reset event."""
        self._initialized = False
        if self._robot_prim_path:
            robot_path = self._robot_prim_path
            self.reset()
            self.setup(robot_path)
        else:
            self.setup(None)
        self.step = 0

    def setup(self, robot_path: str):
        """Configure the gain tuner for a specific robot.

        Args:
            robot_path: USD path to the robot prim.
        """
        if robot_path == self._robot_prim_path or robot_path is None:
            return

        stage = omni.usd.get_context().get_stage()
        self._robot_prim_path = robot_path
        self._robot = stage.GetPrimAtPath(robot_path)

        self._setup_robot_links(stage)
        self._setup_articulation(stage)
        self._setup_fixed_links(stage)
        self._setup_joint_entries(stage)

        self._joint_names = {i: self._joint_entries[i].joint.GetName() for i in self._joints.keys()}
        self._joint_map = {self._joint_names[i]: self._joints[i] for i in self._joints.keys()}
        self._all_joint_indices = self._articulation.get_dof_indices(self._joint_names.values()).list()

        self.initialize()

    def _setup_robot_links(self, stage: pxr.Usd.Stage):
        """Set up robot link mass properties.

        Args:
            stage: The USD stage.
        """
        self._robot_links = [
            link for link in pxr.Usd.PrimRange(self._robot) if link.HasAPI(robot_schema.Classes.LINK_API.value)
        ]

        async def update_link_mass():
            was_playing = self._timeline.is_playing()
            if not was_playing:
                self._timeline.play()
            await omni.kit.app.get_app().next_update_async()
            self.compute_joints_accumulated_inertia()
            if not was_playing:
                self._timeline.stop()

        asyncio.ensure_future(update_link_mass())
        self._robot_tree = rs_utils.GenerateRobotLinkTree(stage, self._robot)

    def _setup_articulation(self, stage: pxr.Usd.Stage):
        """Set up the articulation wrapper.

        Args:
            stage: The USD stage.
        """
        self._articulation_root = find_articulation_root(stage, self._robot_prim_path)
        self._articulation = Articulation(self._articulation_root)

    def _setup_fixed_links(self, stage: pxr.Usd.Stage):
        """Identify links that are fixed to the world.

        Args:
            stage: The USD stage.
        """
        robot_joints = rs_utils.GetAllRobotJoints(stage, self._robot, True)

        # Find fixed joints with one body connected to world (None)
        fixed_joint_candidates = [
            joint
            for joint in robot_joints
            if pxr.UsdPhysics.FixedJoint(joint)
            and (
                rs_utils.GetJointBodyRelationship(joint, 0) is None
                or rs_utils.GetJointBodyRelationship(joint, 1) is None
            )
        ]

        # Filter to only enabled fixed joints
        fixed_joints = []
        for joint in fixed_joint_candidates:
            enabled_attr = pxr.UsdPhysics.Joint(joint).GetJointEnabledAttr()
            if enabled_attr and enabled_attr.Get():
                fixed_joints.append(joint)

        # Collect all links connected to fixed joints
        fixed_links = set()
        for joint in fixed_joints:
            for body_index in (0, 1):
                link = rs_utils.GetJointBodyRelationship(joint, body_index)
                if link is not None:
                    fixed_links.add(link)

        self._fixed_links = fixed_links
        self._robot_joints = robot_joints

    def _setup_joint_entries(self, stage: pxr.Usd.Stage):
        """Build the list of joint entries for the UI.

        Args:
            stage: The USD stage.
        """
        joints = [stage.GetPrimAtPath(self._articulation.dof_paths[0][i]) for i in range(self._articulation.num_dofs)]

        self._joints = {}
        self._joint_entries = []
        dof_offset = 0
        dof_index = 0

        for joint in joints:
            if joint not in self._robot_joints:
                continue

            display_name = self._articulation.dof_names[dof_index]

            if joint.IsA(pxr.UsdPhysics.PrismaticJoint) or joint.IsA(pxr.UsdPhysics.RevoluteJoint):
                self._add_single_axis_joint_entry(joint, display_name, dof_index + dof_offset)
                dof_index += 1
                continue

            # Handle multi-axis (D6) joints
            dof_offset = self._add_multi_axis_joint_entries(joint, dof_index, dof_offset)
            dof_index += 1

    def _add_single_axis_joint_entry(self, joint: pxr.Usd.Prim, display_name: str, dof_index: int):
        """Add a single-axis joint to the entries list.

        Args:
            joint: The joint prim.
            display_name: Display name for the joint.
            dof_index: The DOF index.
        """
        self._joints[dof_index] = joint
        self._joint_entries.append(
            JointListEntry(joint=joint, display_name=display_name, dof_index=dof_index, drive_axis=None)
        )

    def _add_multi_axis_joint_entries(self, joint: pxr.Usd.Prim, base_dof_index: int, dof_offset: int) -> int:
        """Add multi-axis (D6) joint entries.

        Args:
            joint: The joint prim.
            base_dof_index: Base DOF index for this joint.
            dof_offset: Current offset for multi-axis joints.

        Returns:
            Updated dof_offset.
        """
        for axis_index, drive_axis in enumerate(_D6_AXIS_TOKENS):
            display_name = self._articulation.dof_names[base_dof_index]

            if joint.IsA(pxr.UsdPhysics.Joint) and not joint.IsA(pxr.UsdPhysics.FixedJoint):
                carb.log_warn(
                    f"Joint: {joint.GetName()} is a multi-axis joint and Articulation does not "
                    "support it. The gains tuner will allow editing the gains but Running tests "
                    "and Charts may have unpredictable results"
                )
                if not _d6_axis_has_unlocked_limit(joint, drive_axis):
                    continue
                display_name = _format_d6_display_name(joint, display_name, drive_axis)

            current_dof_index = base_dof_index + dof_offset
            self._joints[current_dof_index] = joint
            self._joint_entries.append(
                JointListEntry(
                    joint=joint, display_name=display_name, dof_index=current_dof_index, drive_axis=drive_axis
                )
            )
            dof_offset += 1

        return dof_offset

    def get_dof_type(self, dof_index: int) -> int:
        """Get the type of a degree of freedom.

        Args:
            dof_index: Index of the DOF.

        Returns:
            The DOF type enumeration value.
        """
        return self._articulation.dof_types[dof_index]

    def __del__(self):
        """Clean up resources."""
        self._articulation = None

    def initialize(self):
        """Initialize the gain tuner with the current articulation state."""
        if self._articulation and self._timeline.is_playing():
            positions, orientations = self._articulation.get_world_poses()
            linear_velocities, angular_velocities = self._articulation.get_velocities()
            dof_positions = self._articulation.get_dof_positions()
            dof_velocities = self._articulation.get_dof_velocities()
            dof_efforts = self._articulation.get_dof_efforts()
            self._articulation.set_default_state(
                positions=positions,
                orientations=orientations,
                linear_velocities=linear_velocities,
                angular_velocities=angular_velocities,
                dof_positions=dof_positions,
                dof_velocities=dof_velocities,
                dof_efforts=dof_efforts,
            )
            self._articulation.reset_to_default_state()
            self._initialized = True

    @property
    def initialized(self) -> bool:
        """Whether the gain tuner has been initialized.

        Returns:
            True if initialized, False otherwise.
        """
        return self._initialized

    @initialized.setter
    def initialized(self, value: bool):
        """Set the initialized state.

        Args:
            value: True if initialized, False otherwise.
        """
        self._initialized = value

    @property
    def joint_range_maximum(self) -> float:
        """Joint range maximum.

        Returns:
            Joint range maximum.
        """
        return self._joint_range_maximum

    @joint_range_maximum.setter
    def joint_range_maximum(self, value: float):
        """Set the joint range maximum.

        Args:
            value: Joint range maximum.
        """
        self._joint_range_maximum = value

    @property
    def position_impulse(self) -> float:
        """Position impulse.

        Returns:
            Position impulse.
        """
        return self._position_impulse

    @position_impulse.setter
    def position_impulse(self, value: float):
        """Set the position impulse.

        Args:
            value: Position impulse.
        """
        self._position_impulse = value

    @property
    def velocity_impulse(self) -> float:
        """Velocity impulse.

        Returns:
            Velocity impulse.
        """
        return self._velocity_impulse

    @velocity_impulse.setter
    def velocity_impulse(self, value: float):
        """Set the velocity impulse.

        Args:
            value: Velocity impulse.
        """
        self._velocity_impulse = value

    @property
    def robot(self) -> pxr.Usd.Prim:
        """Robot prim.

        Returns:
            The robot prim.
        """
        return self._robot

    def _accumulate_link_inertia(
        self,
        links: List[pxr.Usd.Prim],
        joint_pose: Gf.Matrix4d,
        robot_transform: Gf.Matrix4d,
        is_prismatic: bool,
        link_path_to_index: Dict[Sdf.Path, int],
        link_masses: np.ndarray,
        link_com_positions: np.ndarray,
        link_inertias: np.ndarray,
    ) -> Tuple[float, Gf.Matrix3f, bool]:
        """Accumulate mass and inertia for a set of links.

        Args:
            links: List of link prims to accumulate.
            joint_pose: The joint's pose matrix.
            robot_transform: The robot's world transform.
            is_prismatic: True if the joint is prismatic.
            link_path_to_index: Mapping from link prim path to articulation link index.
            link_masses: Mass for each link, shape ``(L,)``.
            link_com_positions: Local-frame COM positions for each link, shape ``(L, 3)``.
            link_inertias: Full 3x3 inertia tensors (row-major) for each link, shape ``(L, 9)``.

        Returns:
            Tuple of (total_mass, accumulated_inertia, is_fixed).
            Returns (0, zero_matrix, True) if a fixed link is encountered.
        """
        total_mass = 0.0
        accumulated_inertia = pxr.Gf.Matrix3f().SetZero()

        for link in links:
            link_path = link.GetPath()

            if link_path in self._fixed_links:
                return (0.0, pxr.Gf.Matrix3f().SetZero(), True)

            link_index = link_path_to_index.get(link_path)
            if link_index is None:
                continue

            mass = float(link_masses[link_index])
            total_mass += mass

            if not is_prismatic:
                link_pose = omni.usd.get_world_transform_matrix(link)
                com_local = link_com_positions[link_index].astype(float)
                world_com = pxr.Gf.Vec3d(
                    (
                        link_pose
                        * pxr.Gf.Matrix4d().SetTranslate(pxr.Gf.Vec3d(*com_local))
                        * robot_transform.GetInverse()
                    ).ExtractTranslation()
                )
                displacement = world_com - joint_pose.ExtractTranslation()

                inertia_flat = link_inertias[link_index].astype(float)
                body_frame_inertia = pxr.Gf.Matrix3f(
                    inertia_flat[0],
                    inertia_flat[1],
                    inertia_flat[2],
                    inertia_flat[3],
                    inertia_flat[4],
                    inertia_flat[5],
                    inertia_flat[6],
                    inertia_flat[7],
                    inertia_flat[8],
                )
                transformed_inertia = compute_parallel_axis_inertia(body_frame_inertia, mass, displacement)
                accumulated_inertia += transformed_inertia

        return (total_mass, accumulated_inertia, False)

    def compute_joints_accumulated_inertia(self):
        """Compute the effective inertia for each joint about its motion axis.

        For revolute joints (rotational spring-damper):
            I * theta'' + D * theta' + K * theta = tau
            Effective inertia is the moment of inertia projected onto the rotation axis.

        For prismatic joints (linear spring-damper):
            m * x'' + D * x' + K * x = F
            Effective inertia is simply the accumulated mass (F = ma).

        The equivalent inertia for series connection of both sides:
            - If one side is fixed: I_eq = I_moving
            - If both sides move: I_eq = (I1 * I2) / (I1 + I2)

        Note:
            This computation is dependent on the initial position of the robot.
            Natural frequency will be heavily biased by the initial position.
        """
        if not self._robot or not self._articulation:
            return

        # Build link path -> link index mapping
        link_paths = self._articulation.link_paths[0]
        link_path_to_index = {Sdf.Path(path): i for i, path in enumerate(link_paths)}

        # Query all link mass properties from the Articulation tensor API
        link_masses = self._articulation.get_link_masses().numpy()[0]  # (L,)
        com_positions, _ = self._articulation.get_link_coms()
        link_com_positions = com_positions.numpy()[0]  # (L, 3)
        link_inertias = self._articulation.get_link_inertias().numpy()[0]  # (L, 9)

        robot_transform = omni.usd.get_world_transform_matrix(self._robot)
        joint_inertia = {}

        for joint in self._joints.values():
            backward_links, forward_links = rs_utils.GetLinksFromJoint(self._robot_tree, joint)
            joint_pose = rs_utils.GetJointPose(self._robot, joint)
            is_prismatic = joint.IsA(pxr.UsdPhysics.PrismaticJoint)
            joint_axis = get_joint_axis_world_direction(joint, joint_pose)

            # Accumulate inertia for backward links
            backward_result = self._accumulate_link_inertia(
                backward_links,
                joint_pose,
                robot_transform,
                is_prismatic,
                link_path_to_index,
                link_masses,
                link_com_positions,
                link_inertias,
            )
            backward_mass, backward_inertia, backward_fixed = backward_result

            # Accumulate inertia for forward links
            forward_result = self._accumulate_link_inertia(
                forward_links,
                joint_pose,
                robot_transform,
                is_prismatic,
                link_path_to_index,
                link_masses,
                link_com_positions,
                link_inertias,
            )
            forward_mass, forward_inertia, forward_fixed = forward_result

            # Compute effective inertia based on joint type
            if is_prismatic:
                inertia_forward = forward_mass
                inertia_backward = backward_mass
            else:
                inertia_forward = project_inertia_onto_axis(forward_inertia, joint_axis)
                inertia_backward = project_inertia_onto_axis(backward_inertia, joint_axis)

            # Compute equivalent inertia for spring-damper system
            equivalent_inertia = self._compute_equivalent_inertia(
                inertia_forward, inertia_backward, forward_fixed, backward_fixed
            )
            joint_inertia[joint] = equivalent_inertia

        self._joint_accumulated_inertia = joint_inertia

    def _compute_equivalent_inertia(
        self,
        inertia_forward: float,
        inertia_backward: float,
        forward_fixed: bool,
        backward_fixed: bool,
    ) -> float:
        """Compute the equivalent inertia for a spring-damper system.

        For series connection: I_eq = (I1 * I2) / (I1 + I2).
        If one side is fixed, only the moving side contributes.

        Args:
            inertia_forward: Inertia of the forward link chain.
            inertia_backward: Inertia of the backward link chain.
            forward_fixed: True if forward chain is fixed to world.
            backward_fixed: True if backward chain is fixed to world.

        Returns:
            The equivalent inertia value.
        """
        epsilon = 1e-9

        if backward_fixed or inertia_backward < epsilon:
            return inertia_forward
        if forward_fixed or inertia_forward < epsilon:
            return inertia_backward

        return (inertia_forward * inertia_backward) / (inertia_forward + inertia_backward)

    def get_articulation(self) -> Articulation:
        """Get the articulation wrapper.

        Returns:
            The articulation object.
        """
        return self._articulation

    def get_all_joint_indices(self) -> List[int]:
        """Get all joint DOF indices.

        Returns:
            List of all joint indices.
        """
        return self._all_joint_indices

    def get_joint_entries(self) -> List[JointListEntry]:
        """Get the list of joint entries.

        Returns:
            Copy of the joint entries list.
        """
        return list(self._joint_entries)

    def get_permanent_fixed_joint_indices(self) -> List[int]:
        """Get indices of permanently fixed joints.

        Returns:
            List of fixed joint indices.
        """
        return self._permanent_fixed_joint_indices

    def get_test_duration(self) -> float:
        """Get the test duration.

        Returns:
            Test duration in seconds.
        """
        return self._test_duration

    def is_data_ready(self) -> bool:
        """Check if test data is ready for analysis.

        Returns:
            True if data is ready, False otherwise.
        """
        return self._data_ready

    def set_test_duration(self, duration: float):
        """Set the test duration.

        Args:
            duration: Test duration in seconds.
        """
        self._test_duration = duration

    # ======================== Run Gains Test ========================

    def _partition_joints_by_mode(self, sequence_index: int) -> Tuple[List[int], List[int], List[int], List[int]]:
        """Partition joints into position and velocity control groups.

        Args:
            sequence_index: Index of the current test sequence.

        Returns:
            Tuple of (position_dof_indices, velocity_dof_indices,
                     position_param_indices, velocity_param_indices).
        """
        joint_indices = self.test_params["sequence"][sequence_index]["joint_indices"]

        position_dof_indices = [i for i in joint_indices if self.joint_modes[i] == JointMode.POSITION]
        velocity_dof_indices = [i for i in joint_indices if self.joint_modes[i] == JointMode.VELOCITY]
        position_param_indices = [i for i, j in enumerate(joint_indices) if self.joint_modes[j] == JointMode.POSITION]
        velocity_param_indices = [i for i, j in enumerate(joint_indices) if self.joint_modes[j] == JointMode.VELOCITY]

        return position_dof_indices, velocity_dof_indices, position_param_indices, velocity_param_indices

    def _get_sequence_params(self, sequence_index: int, param_indices: List[int], param_name: str) -> np.ndarray:
        """Extract parameter values for specific joint indices from a sequence.

        Args:
            sequence_index: Index of the test sequence.
            param_indices: Indices to extract.
            param_name: Name of the parameter in the sequence config.

        Returns:
            Numpy array of parameter values.
        """
        sequence = self.test_params["sequence"][sequence_index]
        return np.array([sequence[param_name][i] for i in param_indices])

    def sinusoidal_step(
        self, timestep: float, sequence_index: int
    ) -> Tuple[List[int], np.ndarray, List[int], np.ndarray]:
        """Generate sinusoidal position and velocity commands.

        Args:
            timestep: Current simulation time.
            sequence_index: Index of the current test sequence.

        Returns:
            Tuple of (position_indices, position_commands,
                     velocity_indices, velocity_commands).
        """
        pos_dof_idx, vel_dof_idx, pos_param_idx, vel_param_idx = self._partition_joints_by_mode(sequence_index)

        # Position control - sinusoidal motion within joint limits
        lower_limits, upper_limits = [
            np.array(lim.list()) for lim in self._articulation.get_dof_limits(dof_indices=pos_dof_idx)
        ]
        amplitudes = self._get_sequence_params(sequence_index, pos_param_idx, "joint_amplitudes")
        offsets = self._get_sequence_params(sequence_index, pos_param_idx, "joint_offsets")
        periods = self._get_sequence_params(sequence_index, pos_param_idx, "joint_periods")
        phases = self._get_sequence_params(sequence_index, pos_param_idx, "joint_phases")

        joint_range = upper_limits - lower_limits
        center = (upper_limits + lower_limits) / 2
        position_command = np.clip(
            joint_range * amplitudes * np.sin(2 * np.pi * timestep / periods + phases) + center + offsets,
            lower_limits,
            upper_limits,
        )

        # Velocity control - sinusoidal velocity
        vel_periods = self._get_sequence_params(sequence_index, vel_param_idx, "joint_periods")
        vel_phases = self._get_sequence_params(sequence_index, vel_param_idx, "joint_phases")
        max_velocities = self._articulation.get_dof_max_velocities(dof_indices=vel_dof_idx).numpy()
        velocity_command = (
            max_velocities * 2 * np.pi / vel_periods * np.sin(2 * np.pi * timestep / vel_periods + vel_phases)
        )

        return pos_dof_idx, position_command, vel_dof_idx, velocity_command

    def step_step(self, timestep: float, sequence_index: int) -> Tuple[List[int], np.ndarray, List[int], np.ndarray]:
        """Generate square wave position and velocity commands.

        Args:
            timestep: Current simulation time.
            sequence_index: Index of the current test sequence.

        Returns:
            Tuple of (position_indices, position_commands,
                     velocity_indices, velocity_commands).
        """
        pos_dof_idx, vel_dof_idx, pos_param_idx, vel_param_idx = self._partition_joints_by_mode(sequence_index)

        # Position control - square wave between step_min and step_max
        lower_limits, upper_limits = [
            np.array(lim.list()) for lim in self._articulation.get_dof_limits(dof_indices=pos_dof_idx)
        ]
        step_max = self._get_sequence_params(sequence_index, pos_param_idx, "joint_step_max")
        step_min = self._get_sequence_params(sequence_index, pos_param_idx, "joint_step_min")
        periods = self._get_sequence_params(sequence_index, pos_param_idx, "joint_periods")
        phases = self._get_sequence_params(sequence_index, pos_param_idx, "joint_phases")

        wave_phase = np.sin(2 * np.pi * timestep / periods + phases)
        position_command = np.where(wave_phase >= 0, step_max, step_min)
        position_command = np.clip(position_command, lower_limits, upper_limits)

        # Velocity control - square wave between -v_max and +v_max
        vel_periods = self._get_sequence_params(sequence_index, vel_param_idx, "joint_periods")
        vel_phases = self._get_sequence_params(sequence_index, vel_param_idx, "joint_phases")
        max_velocities = self._articulation.get_dof_max_velocities(dof_indices=vel_dof_idx).numpy()

        vel_wave_phase = np.sin(2 * np.pi * timestep / vel_periods + vel_phases)
        velocity_command = np.where(vel_wave_phase >= 0, max_velocities, -max_velocities)

        return pos_dof_idx, position_command, vel_dof_idx, velocity_command

    def initialize_gains_test(self, test_params: dict):
        """Initialize a gains test with the given parameters.

        Args:
            test_params: Dictionary containing test configuration including
                test_mode, joint_indices, test_duration, and sequence data.
        """
        self.test_params = test_params
        self._test_duration = test_params["test_duration"]
        indices = self.test_params["joint_indices"]
        stiffnesses, dampings = [gains.list() for gains in self._articulation.get_dof_gains()]

        self.joint_modes = {}
        for index in range(len(indices)):
            dof_idx = indices[index]
            if stiffnesses[dof_idx] != 0:
                mode = JointMode.POSITION
            elif dampings[dof_idx] != 0:
                mode = JointMode.VELOCITY
            else:
                mode = JointMode.NONE
            self.joint_modes[dof_idx] = mode

        self._test_timestep = 0
        self._data_ready = False
        self._gains_test_generator = self._gains_test_generator_fn()

    def _compute_gains_test_dof_error_terms(self, joint_index: int) -> Tuple[float, float]:
        """Compute RMSE error terms for a single DOF.

        Args:
            joint_index: Index of the joint DOF.

        Returns:
            Tuple of (position_rmse, velocity_rmse).
        """
        if joint_index in self._joint_indices:
            remapped_index = np.argmax(self._joint_indices == joint_index)
            pos_rmse = np.sqrt(
                np.mean(
                    np.square(
                        self._joint_position_commands[:, remapped_index]
                        - self._observed_joint_positions[:, joint_index]
                    )
                )
            )
            vel_rmse = np.sqrt(
                np.mean(
                    np.square(
                        self._joint_velocity_commands[:, remapped_index]
                        - self._observed_joint_velocities[:, joint_index]
                    )
                )
            )
        else:
            remapped_index = np.argmax(self._fixed_joint_indices == joint_index)
            pos_rmse = np.sqrt(
                np.mean(
                    np.square(self._fixed_positions[remapped_index] - self._observed_joint_positions[:, joint_index])
                )
            )
            vel_rmse = np.sqrt(np.mean(np.square(self._observed_joint_velocities[:, joint_index])))
        return pos_rmse, vel_rmse

    def compute_gains_test_error_terms(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute RMSE error terms for all DOFs.

        Returns:
            Tuple of (position_rmse_array, velocity_rmse_array).
        """
        pos_rmse_list = []
        vel_rmse_list = []
        for dof_index in range(self._articulation.num_dofs):
            pos_rmse, vel_rmse = self._compute_gains_test_dof_error_terms(dof_index)
            pos_rmse_list.append(pos_rmse)
            vel_rmse_list.append(vel_rmse)
        return np.array(pos_rmse_list), np.array(vel_rmse_list)

    def update_gains_test(self, step: float) -> bool:
        """Advance the gains test by one step.

        Args:
            step: Time step size.

        Returns:
            True if test is complete, False otherwise.
        """
        try:
            self.step = step
            next(self._gains_test_generator)
            self._test_timestep += step
            return False
        except StopIteration:
            self._v_max = None
            self._T = None
            return True

    def _gains_test_generator_fn(self):
        """Generator function that runs the gains test sequence.

        Yields control back to the simulation loop after each physics step.
        """
        if self._articulation is None:
            return
        if self.test_params is None:
            carb.log_error("Attempted to run gains test without first calling initialize_test()")
            return

        test_mode = self.test_params["test_mode"]
        step_fn = self.sinusoidal_step if test_mode == GainsTestMode.SINUSOIDAL else self.step_step

        # Initialize data storage
        self._joint_position_commands = []
        self._joint_velocity_commands = []
        self._observed_joint_positions = []
        self._observed_joint_velocities = []
        self._command_times = []

        num_sequences = len(self.test_params["sequence"])

        for sequence_index in range(num_sequences):
            yield from self._run_test_sequence(sequence_index, step_fn)

        self._articulation.reset_to_default_state()
        self._finalize_test_data()

    def _run_test_sequence(self, sequence_index: int, step_fn):
        """Run a single test sequence.

        Args:
            sequence_index: Index of the sequence to run.
            step_fn: Function to generate step commands.

        Yields:
            Empty tuple to yield control to simulation.
        """
        sequence_time = 0
        self._articulation.reset_to_default_state()

        pos_idx, pos_cmd, vel_idx, vel_cmd = step_fn(sequence_time, sequence_index)
        self._articulation.set_dof_position_targets(pos_cmd, dof_indices=pos_idx)
        self._articulation.set_dof_velocity_targets(vel_cmd, dof_indices=vel_idx)

        position_targets = np.copy(self._articulation.get_dof_position_targets().numpy()[0])
        velocity_targets = np.copy(self._articulation.get_dof_velocity_targets().numpy()[0])

        self._record_test_sample(position_targets, velocity_targets)

        yield ()

        while sequence_time < self._test_duration:
            sequence_time += self.step

            pos_idx, pos_cmd, vel_idx, vel_cmd = step_fn(sequence_time, sequence_index)
            self._articulation.set_dof_position_targets(pos_cmd, dof_indices=pos_idx)
            self._articulation.set_dof_velocity_targets(vel_cmd, dof_indices=vel_idx)

            position_targets[pos_idx] = pos_cmd
            velocity_targets[vel_idx] = vel_cmd

            self._joint_position_commands.append(np.copy(position_targets))
            self._joint_velocity_commands.append(np.copy(velocity_targets))
            self._command_times.append(self._test_timestep)

            yield ()

            self._observed_joint_positions.append(self._articulation.get_dof_positions().numpy()[0])
            self._observed_joint_velocities.append(self._articulation.get_dof_velocities().numpy()[0])

    def _record_test_sample(self, position_targets: np.ndarray, velocity_targets: np.ndarray):
        """Record a single test sample.

        Args:
            position_targets: Current position targets.
            velocity_targets: Current velocity targets.
        """
        self._joint_position_commands.append(np.copy(position_targets))
        self._joint_velocity_commands.append(np.copy(velocity_targets))
        self._observed_joint_positions.append(self._articulation.get_dof_positions().numpy()[0])
        self._observed_joint_velocities.append(self._articulation.get_dof_velocities().numpy()[0])
        self._command_times.append(self._test_timestep)

    def _finalize_test_data(self):
        """Convert test data lists to numpy arrays and mark data as ready."""
        self._joint_position_commands = np.array(self._joint_position_commands)
        self._joint_velocity_commands = np.array(self._joint_velocity_commands)
        self._observed_joint_positions = np.array(self._observed_joint_positions)
        self._observed_joint_velocities = np.array(self._observed_joint_velocities)
        self._command_times = np.array(self._command_times)
        self._data_ready = True

    # ======================== For Plotting ========================

    def get_joint_states_from_gains_test(
        self, joint_index: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get recorded joint states from the last gains test.

        Args:
            joint_index: Index of the joint to retrieve data for.

        Returns:
            Tuple of (position_commands, velocity_commands, observed_positions,
                     observed_velocities, timestamps). All values are None if
                     no data is available for the specified joint.
        """
        empty_result = (None, None, None, None, None)

        if len(self._observed_joint_positions) == 0:
            return empty_result

        if joint_index >= self._joint_position_commands.shape[1]:
            return empty_result

        return (
            self._joint_position_commands[:, joint_index],
            self._joint_velocity_commands[:, joint_index],
            self._observed_joint_positions[:, joint_index],
            self._observed_joint_velocities[:, joint_index],
            self._command_times,
        )
