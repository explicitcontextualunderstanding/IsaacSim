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

"""Unit tests for Gain Tuner functionality."""

import asyncio
import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
import omni.kit.test
import omni.usd
import usd.schema.isaac.robot_schema as robot_schema
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.robot_setup.gain_tuner.gains_tuner import GainTuner
from pxr import Gf, PhysicsSchemaTools, Sdf, UsdGeom, UsdPhysics


class JointModality(Enum):
    """Joint type for the test articulation."""

    PRISMATIC = "prismatic"
    REVOLUTE = "revolute"


class DriveSubmodality(Enum):
    """Drive type for the joint."""

    FORCE = "force"
    ACCELERATION = "acceleration"


# @dataclass
# class OscillationAnalysis:
#     """Results from analyzing damped oscillation data."""

#     log_decrement_avg: float
#     damped_period: float
#     damped_freq: float
#     damping_ratio: float
#     natural_freq: float
#     peak_times: List[float]
#     peak_values: List[float]
#     num_samples: int = 0
#     value_min: float = 0.0
#     value_max: float = 0.0


def _compute_stiffness_damping_prismatic(
    mass: float, natural_freq_hz: float, damping_ratio: float
) -> Tuple[float, float]:
    """Compute stiffness and damping for prismatic joint from natural freq and damping ratio.

    For m*x'' + D*x' + K*x = 0:
    w_n = sqrt(K/m) => K = m * w_n^2
    zeta = D / (2*sqrt(m*K)) => D = 2*zeta*sqrt(m*K) = 2*zeta*m*w_n
    """
    w_n = 2.0 * math.pi * natural_freq_hz
    stiffness = mass * (w_n**2)
    damping = 2.0 * damping_ratio * mass * w_n
    return stiffness, damping


def _compute_stiffness_damping_revolute(
    inertia: float, natural_freq_hz: float, damping_ratio: float
) -> Tuple[float, float]:
    """Compute stiffness and damping for revolute joint from natural freq and damping ratio.

    For I*theta'' + D*theta' + K*theta = 0:
    w_n = sqrt(K/I) => K = I * w_n^2
    zeta = D / (2*sqrt(I*K)) => D = 2*zeta*I*w_n
    """
    w_n = 2.0 * math.pi * natural_freq_hz
    stiffness = inertia * (w_n**2)
    damping = 2.0 * damping_ratio * inertia * w_n
    return stiffness, damping


def _compute_natural_freq_damping_revolute(stiffness: float, damping: float, inertia: float) -> Tuple[float, float]:
    """Compute natural frequency (Hz) and damping ratio from revolute drive gains.

    Inverse of _compute_stiffness_damping_revolute: w_n = sqrt(K/I), zeta = D/(2*sqrt(I*K)).
    """
    if inertia <= 0 or stiffness <= 0:
        return 0.0, 0.0
    w_n = math.sqrt(stiffness / inertia)
    natural_freq_hz = w_n / (2.0 * math.pi)
    zeta = damping / (2.0 * math.sqrt(inertia * stiffness))
    return natural_freq_hz, zeta


def _compute_natural_freq_damping_prismatic(stiffness: float, damping: float, mass: float) -> Tuple[float, float]:
    """Compute natural frequency (Hz) and damping ratio from prismatic drive gains.

    Inverse of _compute_stiffness_damping_prismatic: w_n = sqrt(K/m), zeta = D/(2*sqrt(m*K)).
    """
    if mass <= 0 or stiffness <= 0:
        return 0.0, 0.0
    w_n = math.sqrt(stiffness / mass)
    natural_freq_hz = w_n / (2.0 * math.pi)
    zeta = damping / (2.0 * math.sqrt(mass * stiffness))
    return natural_freq_hz, zeta


# def _extract_peaks(times: np.ndarray, values: np.ndarray) -> Tuple[List[float], List[float]]:
#     """Extract local maxima (peaks) from time-series data.

#     Includes boundary indices when they are local maxima (e.g. initial condition
#     at release). If no interior peaks are found, falls back to peaks of the
#     absolute value (envelope) so both sides of a damped oscillation are counted.
#     """
#     if len(values) < 2:
#         return [], []
#     n = len(values)
#     # Interior peaks: strict local maxima
#     peak_indices = []
#     if n >= 3:
#         is_peak = (values[1:-1] > values[:-2]) & (values[1:-1] > values[2:])
#         peak_indices = (np.where(is_peak)[0] + 1).tolist()
#     # First point is a peak if it is >= neighbor (e.g. start at initial displacement)
#     if values[0] >= values[1]:
#         peak_indices.insert(0, 0)
#     # Last point is a peak if it is >= neighbor
#     if n >= 2 and values[-1] >= values[-2]:
#         peak_indices.append(n - 1)
#     if peak_indices:
#         peak_indices = sorted(set(peak_indices))
#         peak_times = times[peak_indices].tolist()
#         peak_values = values[peak_indices].tolist()
#         return peak_times, peak_values
#     # Fallback: peaks of |values| (envelope) so we catch both positive and negative humps
#     abs_vals = np.abs(values)
#     if n >= 3:
#         is_peak_abs = (abs_vals[1:-1] > abs_vals[:-2]) & (abs_vals[1:-1] > abs_vals[2:])
#         peak_indices = (np.where(is_peak_abs)[0] + 1).tolist()
#     if not peak_indices:
#         return [], []
#     peak_times = times[peak_indices].tolist()
#     # Store signed values for analysis (log decrement uses abs(peak_values))
#     peak_values = values[peak_indices].tolist()
#     return peak_times, peak_values


# def _peaks_fail_msg(result: OscillationAnalysis, prefix: str = "") -> str:
#     """Build assertion message when peak count is too low, including data diagnostics."""
#     return (
#         f"{prefix}Need at least 3 peaks for analysis (got {len(result.peak_values)}). "
#         f"Position data: n={result.num_samples} min={result.value_min:.6f} max={result.value_max:.6f}"
#     )


# def _analyze_oscillation(times: np.ndarray, values: np.ndarray) -> OscillationAnalysis:
#     """Extract natural frequency and damping ratio from damped oscillation data.

#     Uses:
#     - Logarithmic decrement: s_i = ln(A_i / A_{i+1}), average over peaks
#     - Damped period T_d from average time between successive peaks
#     - w_d = 2*pi / T_d
#     - zeta = s / sqrt(4*pi^2 + s^2)
#     - w_n = w_d / sqrt(1 - zeta^2)
#     """
#     num_samples = len(values)
#     value_min = float(np.min(values)) if num_samples else 0.0
#     value_max = float(np.max(values)) if num_samples else 0.0
#     peak_times, peak_values = _extract_peaks(times, values)
#     if len(peak_values) < 2:
#         return OscillationAnalysis(
#             log_decrement_avg=0.0,
#             damped_period=0.0,
#             damped_freq=0.0,
#             damping_ratio=0.0,
#             natural_freq=0.0,
#             peak_times=peak_times,
#             peak_values=peak_values,
#             num_samples=num_samples,
#             value_min=value_min,
#             value_max=value_max,
#         )

#     log_decrements = []
#     for i in range(len(peak_values) - 1):
#         a_i = abs(peak_values[i])
#         a_next = abs(peak_values[i + 1])
#         if a_next > 1e-12:
#             s_i = math.log(a_i / a_next + 1e-12)
#             log_decrements.append(s_i)
#     s_avg = np.mean(log_decrements) if log_decrements else 0.0

#     period_diffs = [peak_times[i + 1] - peak_times[i] for i in range(len(peak_times) - 1)]
#     T_d = np.mean(period_diffs) if period_diffs else 0.0
#     w_d = (2.0 * math.pi / T_d) if T_d > 1e-9 else 0.0

#     zeta = s_avg / math.sqrt(4.0 * math.pi**2 + s_avg**2) if (4 * math.pi**2 + s_avg**2) > 0 else 0.0
#     denom = 1.0 - zeta**2
#     w_n = (w_d / math.sqrt(denom)) if denom > 1e-9 else 0.0

#     return OscillationAnalysis(
#         log_decrement_avg=s_avg,
#         damped_period=T_d,
#         damped_freq=w_d,
#         damping_ratio=zeta,
#         natural_freq=w_n / (2.0 * math.pi),
#         peak_times=peak_times,
#         peak_values=peak_values,
#         num_samples=num_samples,
#         value_min=value_min,
#         value_max=value_max,
#     )


class TestGainTuner(omni.kit.test.AsyncTestCase):
    """Unit tests for Gain Tuner functionality."""

    async def setUp(self):
        self._physics_fps = 60
        self._physics_dt = 1.0 / self._physics_fps
        self._timeline = omni.timeline.get_timeline_interface()
        self._gain_tuner = GainTuner()
        await stage_utils.create_new_stage_async()
        self._stage = omni.usd.get_context().get_stage()
        self._stage.DefinePrim(Sdf.Path("/World"), "Xform")
        PhysicsSchemaTools.addGroundPlane(
            self._stage, "/World/groundPlane", "Z", 100, Gf.Vec3f(0, 0, -0.5), Gf.Vec3f(1.0)
        )
        SimulationManager.set_physics_dt(self._physics_dt)
        await app_utils.update_app_async()

    async def tearDown(self):
        self._timeline.stop()
        self._gain_tuner.reset()
        await app_utils.update_app_async()

        while omni.usd.get_context().get_stage_loading_status()[2] > 0:
            print("tearDown, assets still loading, waiting to finish...")
            await asyncio.sleep(1.0)

        stage_utils.close_stage()

    def _create_articulation(
        self,
        joint_modalities: List[JointModality],
        drive_submodality: DriveSubmodality,
        distance: float,
        mass: float,
        inertia_diag: float,
        natural_freq_hz: float,
        damping_ratio: float,
        *,
        chain: bool = False,
        fixed_base: bool = True,
        joint_axes: Optional[List[str]] = None,
        link_positions: Optional[List[Tuple[float, float, float]]] = None,
        base_mass: float = 1000.0,
        base_inertia: float = 1.0,
    ) -> str:
        """Create an articulation: fixed or floating base + one or more links connected by joints.

        Args:
            joint_modalities: Type of each joint (revolute or prismatic).
            drive_submodality: Force or acceleration drive.
            distance: Spacing used for default link positions (link i at (i+1)*distance, 0, 0.5).
            mass: Mass of each link.
            inertia_diag: Diagonal inertia of each link (and of base when fixed_base=False).
            natural_freq_hz: Target natural frequency for drive gains.
            damping_ratio: Target damping ratio for drive gains.
            chain: If True, joint i connects (base if i==0 else link_{i-1}) to link_i. If False, every joint connects base to link_i.
            fixed_base: If True, add a fixed joint to world and ArticulationRootAPI on it. If False, no fixed joint and ArticulationRootAPI on base.
            joint_axes: Per-joint axis (e.g. ["Z", "Y"] for revolute). Default: revolute "Z", prismatic "X".
            link_positions: Optional (x, y, z) for each link; default (i+1)*distance along X at z=0.5.
            base_mass: Mass of base link (used when fixed_base=False for floating base).
            base_inertia: Diagonal inertia of base (used when fixed_base=False).

        Returns:
            Robot prim path.
        """
        robot_path = "/World/robot"
        base_path = f"{robot_path}/base"
        fixed_joint_path = f"{robot_path}/root_joint"
        n_joints = len(joint_modalities)
        axes = (
            joint_axes
            if joint_axes is not None
            else (["Z" if m == JointModality.REVOLUTE else "X" for m in joint_modalities])
        )

        # Base
        base_geom = UsdGeom.Cube.Define(self._stage, base_path)
        base_geom.CreateSizeAttr(0.1)
        base_geom.AddTranslateOp().Set(Gf.Vec3f(0, 0, 0.5))
        base_prim = self._stage.GetPrimAtPath(base_path)
        UsdPhysics.CollisionAPI.Apply(base_prim)
        UsdPhysics.RigidBodyAPI.Apply(base_prim)
        UsdPhysics.MassAPI.Apply(base_prim)
        UsdPhysics.MassAPI(base_prim).CreateMassAttr(base_mass)
        UsdPhysics.MassAPI(base_prim).CreateDiagonalInertiaAttr(Gf.Vec3f(base_inertia, base_inertia, base_inertia))

        if fixed_base:
            fixed_joint = UsdPhysics.FixedJoint.Define(self._stage, fixed_joint_path)
            fixed_joint.CreateBody1Rel().SetTargets([base_path])
            UsdPhysics.ArticulationRootAPI.Apply(fixed_joint.GetPrim())
        else:
            fixed_joint = None
            UsdPhysics.ArticulationRootAPI.Apply(base_prim)

        link_prims = []
        joint_prims = []

        base_pos = Gf.Vec3f(0, 0, 0.5)
        body0_positions = {}
        body0_positions[base_path] = base_pos

        for joint_index, joint_modality in enumerate(joint_modalities):
            link_path = f"{robot_path}/link_{joint_index}"
            if link_positions is not None and joint_index < len(link_positions):
                pos = Gf.Vec3f(*link_positions[joint_index])
            else:
                pos = Gf.Vec3f((joint_index + 1) * distance, 0, 0.5)
            link_geom = UsdGeom.Cube.Define(self._stage, link_path)
            link_geom.CreateSizeAttr(0.2)
            link_geom.AddTranslateOp().Set(pos)
            link_prim = self._stage.GetPrimAtPath(link_path)
            UsdPhysics.CollisionAPI.Apply(link_prim)
            UsdPhysics.RigidBodyAPI.Apply(link_prim)
            UsdPhysics.MassAPI.Apply(link_prim)
            UsdPhysics.MassAPI(link_prim).CreateMassAttr(mass)
            UsdPhysics.MassAPI(link_prim).CreateDiagonalInertiaAttr(Gf.Vec3f(inertia_diag, inertia_diag, inertia_diag))
            link_prims.append(link_prim)
            body0_positions[link_path] = pos

            body0 = base_path if (not chain or joint_index == 0) else f"{robot_path}/link_{joint_index - 1}"
            body0_pos = body0_positions[body0]
            axis = (
                axes[joint_index]
                if joint_index < len(axes)
                else ("Z" if joint_modality == JointModality.REVOLUTE else "X")
            )

            if joint_modality == JointModality.PRISMATIC:
                joint = UsdPhysics.PrismaticJoint.Define(self._stage, f"{robot_path}/joint_{joint_index}")
                joint.CreateAxisAttr(axis)
                joint.CreateBody0Rel().SetTargets([body0])
                joint.CreateBody1Rel().SetTargets([link_path])
                drive_type = "linear"
                stiffness, damping = _compute_stiffness_damping_prismatic(mass, natural_freq_hz, damping_ratio)
            else:
                joint = UsdPhysics.RevoluteJoint.Define(self._stage, f"{robot_path}/joint_{joint_index}")
                joint.CreateAxisAttr(axis)
                joint.CreateBody0Rel().SetTargets([body0])
                joint.CreateBody1Rel().SetTargets([link_path])
                drive_type = "angular"
                stiffness, damping = _compute_stiffness_damping_revolute(inertia_diag, natural_freq_hz, damping_ratio)

            local_pos1 = Gf.Vec3f(body0_pos[0] - pos[0], body0_pos[1] - pos[1], body0_pos[2] - pos[2])
            joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, 0))
            joint.CreateLocalPos1Attr().Set(local_pos1)

            drive_api = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), drive_type)
            drive_api.CreateTypeAttr(drive_submodality.value)
            drive_api.CreateStiffnessAttr(stiffness)
            drive_api.CreateDampingAttr(damping)
            joint_prims.append(joint.GetPrim())

        # Robot schema: all links and all joints (fixed first when present)
        robot_prim = self._stage.GetPrimAtPath(robot_path)
        robot_schema.ApplyRobotAPI(robot_prim)
        all_links = [base_prim.GetPath()] + [p.GetPath() for p in link_prims]
        all_joints = ([fixed_joint.GetPrim().GetPath()] if fixed_joint is not None else []) + [
            p.GetPath() for p in joint_prims
        ]
        robot_prim.GetRelationship(robot_schema.Relations.ROBOT_LINKS.name).SetTargets(all_links)
        robot_prim.GetRelationship(robot_schema.Relations.ROBOT_JOINTS.name).SetTargets(all_joints)
        for p in [base_prim] + link_prims:
            robot_schema.ApplyLinkAPI(p)
        for j in ([] if fixed_joint is None else [fixed_joint.GetPrim()]) + joint_prims:
            robot_schema.ApplyJointAPI(j)

        return robot_path

    async def _run_setup_and_compute_inertia(self, robot_path: str, num_physics_steps: int = 60):
        """Setup gain tuner, run physics so mass query completes, then compute joint inertias."""
        self._gain_tuner.setup(robot_path)
        for _ in range(2):
            await app_utils.update_app_async()
        self._timeline.play()
        for _ in range(num_physics_steps):
            await app_utils.update_app_async()
        self._gain_tuner.compute_joints_accumulated_inertia()
        self._timeline.stop()

    def _create_fixed_base_two_revolute_chain(
        self,
        distance: float,
        mass: float,
        inertia_diag: float,
        natural_freq_hz: float,
        damping_ratio: float,
        second_axis_z: bool = True,
    ) -> Tuple[str, List[float]]:
        """Create fixed base -> revolute0 -> link0 -> revolute1 -> link1. Same plane if second_axis_z True."""
        robot_path = self._create_articulation(
            [JointModality.REVOLUTE, JointModality.REVOLUTE],
            DriveSubmodality.FORCE,
            distance=distance,
            mass=mass,
            inertia_diag=inertia_diag,
            natural_freq_hz=natural_freq_hz,
            damping_ratio=damping_ratio,
            chain=True,
            joint_axes=["Z", "Z" if second_axis_z else "Y"],
        )
        # j0: pose at base; forward = link0+link1 about base -> I_fwd_j0 = (I_d+m*d^2)+(I_d+m*(2d)^2) = 3.25.
        I_fwd_j0 = (inertia_diag + mass * distance**2) + (inertia_diag + mass * (2 * distance) ** 2)
        # j1: backward_links = [link0, base]; base is fixed so _accumulate_link_inertia returns (0,0,True).
        # I_eq = forward only. Joint pose at body0 (link0); forward = link1 about j1 = I_d + m*d^2 = 1.25.
        I_eq_j1 = inertia_diag + mass * distance**2
        return robot_path, [I_fwd_j0, I_eq_j1]

    def _create_fixed_base_two_prismatic_chain(
        self,
        distance: float,
        mass: float,
        natural_freq_hz: float,
        damping_ratio: float,
        same_axis: bool = True,
    ) -> Tuple[str, List[float]]:
        """Create fixed base -> prism0 -> link0 -> prism1 -> link1. Same axis X if same_axis else second Y."""
        link_positions = None if same_axis else [(distance, 0, 0.5), (distance, distance, 0.5)]
        robot_path = self._create_articulation(
            [JointModality.PRISMATIC, JointModality.PRISMATIC],
            DriveSubmodality.FORCE,
            distance=distance,
            mass=mass,
            inertia_diag=1.0,
            natural_freq_hz=natural_freq_hz,
            damping_ratio=damping_ratio,
            chain=True,
            joint_axes=["X", "X" if same_axis else "Y"],
            link_positions=link_positions,
        )
        # j0: backward fixed, forward = m0 + m1 = 2*mass. j1: backward chain includes fixed base -> I_eq = forward mass only = mass.
        return robot_path, [2.0 * mass, mass]

    def _create_moving_base_single_joint(
        self,
        joint_revolute: bool,
        distance: float,
        base_mass: float,
        base_inertia: float,
        link_mass: float,
        link_inertia_diag: float,
        natural_freq_hz: float,
        damping_ratio: float,
    ) -> Tuple[str, List[float]]:
        """Create floating base -> single joint -> link. No fixed joint. ArticulationRootAPI on base."""
        modality = JointModality.REVOLUTE if joint_revolute else JointModality.PRISMATIC
        robot_path = self._create_articulation(
            [modality],
            DriveSubmodality.FORCE,
            distance=distance,
            mass=link_mass,
            inertia_diag=link_inertia_diag,
            natural_freq_hz=natural_freq_hz,
            damping_ratio=damping_ratio,
            fixed_base=False,
            base_mass=base_mass,
            base_inertia=base_inertia,
        )
        if joint_revolute:
            I_eq = (base_inertia * link_inertia_diag) / (base_inertia + link_inertia_diag)
        else:
            I_eq = (base_mass * link_mass) / (base_mass + link_mass)
        return robot_path, [I_eq]

    def _create_fixed_base_revolute_prismatic_chain(
        self,
        distance: float,
        mass: float,
        inertia_diag: float,
        natural_freq_hz: float,
        damping_ratio: float,
    ) -> Tuple[str, List[float]]:
        """Create fixed base -> revolute -> link0 -> prismatic -> link1."""
        robot_path = self._create_articulation(
            [JointModality.REVOLUTE, JointModality.PRISMATIC],
            DriveSubmodality.FORCE,
            distance=distance,
            mass=mass,
            inertia_diag=inertia_diag,
            natural_freq_hz=natural_freq_hz,
            damping_ratio=damping_ratio,
            chain=True,
            joint_axes=["Z", "X"],
        )
        # j0 at base: I_fwd = (I_d + m*d^2) + (I_d + m*(2d)^2) = 3.25. j1 prismatic: backward includes fixed base -> I_eq = forward mass = mass.
        I_eq_j0 = (inertia_diag + mass * distance**2) + (inertia_diag + mass * (2 * distance) ** 2)
        I_eq_j1 = mass
        return robot_path, [I_eq_j0, I_eq_j1]

    # ---- Unit tests for compute_joints_accumulated_inertia ----
    # Hand-computed expected equivalent inertia and optional stiffness/damping from natural frequency
    # are asserted against the implementation. Relative tolerance 5% to allow PhysX vs analytical differences.

    async def test_compute_joints_accumulated_inertia_fixed_base_single_revolute(self):
        """Fixed base + single revolute: I_eq = link inertia about joint axis (backward fixed)."""
        robot_path = self._create_articulation(
            [JointModality.REVOLUTE],
            DriveSubmodality.FORCE,
            distance=0.5,
            mass=1.0,
            inertia_diag=1.0,
            natural_freq_hz=10.0,
            damping_ratio=0.05,
        )
        await self._run_setup_and_compute_inertia(robot_path)
        # Joint is at base (0,0,0.5); link COM at (0.5,0,0.5). I_eq = I_cm + m*d^2 = 1 + 1*0.5^2 = 1.25
        expected_I_eq = 1.0 + 1.0 * (0.5**2)
        entries = self._gain_tuner.get_joint_entries()
        self.assertEqual(len(entries), 1)
        computed = self._gain_tuner._joint_accumulated_inertia.get(entries[0].joint)
        self.assertIsNotNone(computed, "Should have computed inertia for single revolute")
        self.assertAlmostEqual(computed, expected_I_eq, delta=0.05 * max(expected_I_eq, 1e-6))
        # Stiffness/damping from natural frequency: K = I*w_n^2, D = 2*zeta*I*w_n
        stiffness_attr = UsdPhysics.DriveAPI(entries[0].joint, "angular").GetStiffnessAttr()
        damping_attr = UsdPhysics.DriveAPI(entries[0].joint, "angular").GetDampingAttr()
        K, D = stiffness_attr.Get(), damping_attr.Get()
        nat_freq, zeta = _compute_natural_freq_damping_revolute(K, D, computed)
        # Gains were authored with inertia_diag=1.0 but I_eq=1.25; recovered nat_freq = 10*sqrt(1/1.25)
        expected_nat_freq = 10.0 * math.sqrt(1.0 / expected_I_eq)
        self.assertAlmostEqual(nat_freq, expected_nat_freq, delta=0.2)
        self.assertAlmostEqual(zeta, 0.05, delta=0.02)

    async def test_compute_joints_accumulated_inertia_fixed_base_single_prismatic(self):
        """Fixed base + single prismatic: I_eq = link mass (backward fixed)."""
        robot_path = self._create_articulation(
            [JointModality.PRISMATIC],
            DriveSubmodality.FORCE,
            distance=0.5,
            mass=2.0,
            inertia_diag=1.0,
            natural_freq_hz=5.0,
            damping_ratio=0.1,
        )
        await self._run_setup_and_compute_inertia(robot_path)
        expected_I_eq = 2.0  # mass
        entries = self._gain_tuner.get_joint_entries()
        self.assertEqual(len(entries), 1)
        computed = self._gain_tuner._joint_accumulated_inertia.get(entries[0].joint)
        self.assertIsNotNone(computed)
        self.assertAlmostEqual(computed, expected_I_eq, delta=0.05 * expected_I_eq)
        stiffness_attr = UsdPhysics.DriveAPI(entries[0].joint, "linear").GetStiffnessAttr()
        damping_attr = UsdPhysics.DriveAPI(entries[0].joint, "linear").GetDampingAttr()
        K, D = stiffness_attr.Get(), damping_attr.Get()
        nat_freq, zeta = _compute_natural_freq_damping_prismatic(K, D, computed)
        self.assertAlmostEqual(nat_freq, 5.0, delta=0.3)
        self.assertAlmostEqual(zeta, 0.1, delta=0.03)

    async def test_compute_joints_accumulated_inertia_fixed_base_two_revolute_same_plane(self):
        """Fixed base + two revolute joints in the same plane: hand-computed I_eq for each joint."""
        distance = 0.5
        mass, inertia_diag = 1.0, 1.0
        robot_path, expected_list = self._create_fixed_base_two_revolute_chain(
            distance=distance,
            mass=mass,
            inertia_diag=inertia_diag,
            natural_freq_hz=10.0,
            damping_ratio=0.05,
            second_axis_z=True,
        )
        await self._run_setup_and_compute_inertia(robot_path)
        entries = self._gain_tuner.get_joint_entries()
        self.assertEqual(len(entries), 2)
        for entry, expected in zip(entries, expected_list):
            computed = self._gain_tuner._joint_accumulated_inertia.get(entry.joint)
            self.assertIsNotNone(computed, f"Inertia for {entry.joint.GetPath()}")
            self.assertAlmostEqual(computed, expected, delta=0.05 * max(expected, 1e-6))

    async def test_compute_joints_accumulated_inertia_fixed_base_two_revolute_orthogonal(self):
        """Fixed base + two revolute joints in orthogonal planes."""
        distance = 0.5
        mass, inertia_diag = 1.0, 1.0
        robot_path, expected_list = self._create_fixed_base_two_revolute_chain(
            distance=distance,
            mass=mass,
            inertia_diag=inertia_diag,
            natural_freq_hz=10.0,
            damping_ratio=0.05,
            second_axis_z=False,
        )
        await self._run_setup_and_compute_inertia(robot_path)
        entries = self._gain_tuner.get_joint_entries()
        self.assertEqual(len(entries), 2)
        for entry, expected in zip(entries, expected_list):
            computed = self._gain_tuner._joint_accumulated_inertia.get(entry.joint)
            self.assertIsNotNone(computed)
            self.assertAlmostEqual(computed, expected, delta=0.05 * max(expected, 1e-6))

    async def test_compute_joints_accumulated_inertia_moving_base_single_revolute(self):
        """Moving base + single revolute: I_eq = (I_base * I_link_about_joint) / (I_base + I_link_about_joint).

        Implementation uses inertia about the joint: link contributes I_d + m*d^2.
        """
        distance = 0.5
        base_inertia, link_inertia_diag, link_mass = 1.0, 1.0, 1.0
        robot_path, _ = self._create_moving_base_single_joint(
            joint_revolute=True,
            distance=distance,
            base_mass=10.0,
            base_inertia=base_inertia,
            link_mass=link_mass,
            link_inertia_diag=link_inertia_diag,
            natural_freq_hz=10.0,
            damping_ratio=0.05,
        )
        await self._run_setup_and_compute_inertia(robot_path)
        # Link inertia about joint (joint on base): I_d + m*d^2
        I_link_about_joint = link_inertia_diag + link_mass * (distance**2)
        expected_I_eq = (base_inertia * I_link_about_joint) / (base_inertia + I_link_about_joint)  # 5/9
        entries = self._gain_tuner.get_joint_entries()
        self.assertEqual(len(entries), 1)
        computed = self._gain_tuner._joint_accumulated_inertia.get(entries[0].joint)
        self.assertIsNotNone(computed)
        self.assertAlmostEqual(computed, expected_I_eq, delta=0.05 * max(expected_I_eq, 1e-6))

    async def test_compute_joints_accumulated_inertia_fixed_base_two_prismatic_same_axis(self):
        """Fixed base + two prismatic joints on the same axis: j0 I_eq = m0+m1, j1 I_eq = m0*m1/(m0+m1)."""
        robot_path, expected_list = self._create_fixed_base_two_prismatic_chain(
            distance=0.5,
            mass=1.0,
            natural_freq_hz=5.0,
            damping_ratio=0.1,
            same_axis=True,
        )
        await self._run_setup_and_compute_inertia(robot_path)
        entries = self._gain_tuner.get_joint_entries()
        self.assertEqual(len(entries), 2)
        for entry, expected in zip(entries, expected_list):
            computed = self._gain_tuner._joint_accumulated_inertia.get(entry.joint)
            self.assertIsNotNone(computed)
            self.assertAlmostEqual(computed, expected, delta=0.05 * max(expected, 1e-6))

    async def test_compute_joints_accumulated_inertia_fixed_base_two_prismatic_orthogonal(self):
        """Fixed base + two prismatic joints on orthogonal axes."""
        robot_path, expected_list = self._create_fixed_base_two_prismatic_chain(
            distance=0.5,
            mass=1.0,
            natural_freq_hz=5.0,
            damping_ratio=0.1,
            same_axis=False,
        )
        await self._run_setup_and_compute_inertia(robot_path)
        entries = self._gain_tuner.get_joint_entries()
        self.assertEqual(len(entries), 2)
        for entry, expected in zip(entries, expected_list):
            computed = self._gain_tuner._joint_accumulated_inertia.get(entry.joint)
            self.assertIsNotNone(computed)
            self.assertAlmostEqual(computed, expected, delta=0.05 * max(expected, 1e-6))

    async def test_compute_joints_accumulated_inertia_moving_base_single_prismatic(self):
        """Moving base + single prismatic: I_eq = (m_base * m_link) / (m_base + m_link)."""
        robot_path, expected_list = self._create_moving_base_single_joint(
            joint_revolute=False,
            distance=0.5,
            base_mass=10.0,
            base_inertia=1.0,
            link_mass=1.0,
            link_inertia_diag=1.0,
            natural_freq_hz=5.0,
            damping_ratio=0.1,
        )
        await self._run_setup_and_compute_inertia(robot_path)
        expected_I_eq = (10.0 * 1.0) / (10.0 + 1.0)  # 10/11
        entries = self._gain_tuner.get_joint_entries()
        self.assertEqual(len(entries), 1)
        computed = self._gain_tuner._joint_accumulated_inertia.get(entries[0].joint)
        self.assertIsNotNone(computed)
        self.assertAlmostEqual(computed, expected_I_eq, delta=0.05 * max(expected_I_eq, 1e-6))

    async def test_compute_joints_accumulated_inertia_fixed_base_revolute_prismatic_chain(self):
        """Fixed base + revolute then prismatic: j0 I_eq = inertia of (link0+link1) about axis, j1 I_eq = m0*m1/(m0+m1)."""
        robot_path, expected_list = self._create_fixed_base_revolute_prismatic_chain(
            distance=0.5,
            mass=1.0,
            inertia_diag=1.0,
            natural_freq_hz=10.0,
            damping_ratio=0.05,
        )
        await self._run_setup_and_compute_inertia(robot_path)
        entries = self._gain_tuner.get_joint_entries()
        self.assertEqual(len(entries), 2)
        for entry, expected in zip(entries, expected_list):
            computed = self._gain_tuner._joint_accumulated_inertia.get(entry.joint)
            self.assertIsNotNone(computed)
            self.assertAlmostEqual(computed, expected, delta=0.05 * max(expected, 1e-6))
