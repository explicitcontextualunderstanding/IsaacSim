# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for converting MJCF actuator/joint data to PhysX schemas."""

import logging
import math
from collections.abc import Sequence

import usd.schema.newton
from pxr import PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

from .importer_utils import create_physx_mimic_joint

_logger = logging.getLogger(__name__)


def convert_mjc_to_physx(stage: Usd.Stage) -> None:
    """Convert all MJCF actuators to PhysX actuators.

    Args:
        stage: USD stage to update with PhysX actuators.
    """
    for prim in stage.Traverse():
        if prim.GetTypeName() == "MjcActuator":
            convert_mjc_actuator_to_physics(prim, stage)
        elif prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
            convert_mjc_joint_to_physx(prim, stage)
            create_physx_mimic_joint(prim)


def convert_mjc_actuator_to_physics(mjc_actuator: Usd.Prim, stage: Usd.Stage) -> None:
    """Convert a MJCF actuator to a PhysX actuator.

    Args:
        mjc_actuator: MJCF actuator prim.
        stage: USD stage containing the target joint prim.

    Raises:
        ValueError: If the actuator or its target joint prim is invalid.
    """
    if not mjc_actuator.IsValid():
        raise ValueError(f"MJCF actuator prim not found at path: {mjc_actuator.GetPath()}")
    joint_path = mjc_actuator.GetRelationship("mjc:target").GetTargets()[0]
    joint = stage.GetPrimAtPath(joint_path)
    if not joint.IsValid():
        raise ValueError(f"Joint prim not found at path: {joint_path.pathString}")

    # Determine joint type and apply to the appropriate drive instance
    if joint.IsA(UsdPhysics.RevoluteJoint):
        drive_instance = "angular"
    elif joint.IsA(UsdPhysics.PrismaticJoint):
        drive_instance = "linear"
    else:
        return

    if joint.HasAPI(UsdPhysics.DriveAPI, drive_instance):
        drive_api = UsdPhysics.DriveAPI(joint, drive_instance)
    else:
        drive_api = UsdPhysics.DriveAPI.Apply(joint, drive_instance)

    force_range_max = (
        mjc_actuator.GetAttribute("mjc:forceRange:max").Get()
        if mjc_actuator.GetAttribute("mjc:forceRange:max").IsValid()
        else None
    )
    force_range_min = (
        mjc_actuator.GetAttribute("mjc:forceRange:min").Get()
        if mjc_actuator.GetAttribute("mjc:forceRange:min").IsValid()
        else None
    )

    if force_range_max:
        drive_api.CreateMaxForceAttr().Set(force_range_max)
        if force_range_min:
            if math.fabs(force_range_min) != force_range_max:
                _logger.warning(
                    "Magnitude of force range min is not equal to force range max for actuator "
                    + f"{mjc_actuator.GetPath()} for joint {joint.GetPath()}: {abs(force_range_min)} != {force_range_max}"
                )

    # Retrieve gainPrm and biasPrm arrays from MJCF actuator
    gain_prm = (
        mjc_actuator.GetAttribute("mjc:gainPrm").Get() if mjc_actuator.GetAttribute("mjc:gainPrm").IsValid() else None
    )
    bias_prm = (
        mjc_actuator.GetAttribute("mjc:biasPrm").Get() if mjc_actuator.GetAttribute("mjc:biasPrm").IsValid() else None
    )

    # Retrieve gainType and biasType from MJCF actuator
    gain_type = (
        mjc_actuator.GetAttribute("mjc:gainType").Get() if mjc_actuator.GetAttribute("mjc:gainType").IsValid() else None
    )
    bias_type = (
        mjc_actuator.GetAttribute("mjc:biasType").Get() if mjc_actuator.GetAttribute("mjc:biasType").IsValid() else None
    )

    if not bias_prm or len(bias_prm) < 3 or not gain_prm or len(gain_prm) < 3:
        _logger.warning(
            "Gain and bias prm arrays are not available or supported for actuator "
            + f"{mjc_actuator.GetPath()} for joint {joint.GetPath()}, physics drive stiffness and damping will not be created"
        )
        return

    if not gain_type or gain_type != "fixed" or not bias_type or bias_type != "affine":
        _logger.warning(
            "Gain type or bias type not available or supported for actuator "
            + f"{mjc_actuator.GetPath()} for joint {joint.GetPath()}, physics drive stiffness and damping will not be created"
        )
        return

    # position control
    # "gainprm" = [kp, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # "biasprm" = [0, -kp, -kd, 0, 0, 0, 0, 0, 0, 0]
    # stiffness = kp
    # damping = kd

    if (
        gain_prm[0] > 0
        and gain_prm[1] == 0
        and gain_prm[2] == 0
        and bias_prm[0] == 0
        and bias_prm[1] < 0
        and bias_prm[2] < 0
        and gain_prm[0] == -bias_prm[1]
    ):
        actuator_stiffness = gain_prm[0]
        actuator_damping = -bias_prm[2]
        drive_api.CreateStiffnessAttr().Set(actuator_stiffness)
        drive_api.CreateDampingAttr().Set(actuator_damping)

    # velocity control
    # "gainprm" = [kd, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # "biasprm" = [0, 0, -kd, 0, 0, 0, 0, 0, 0, 0]
    # stiffness = 0
    # damping = kd

    elif (
        gain_prm[0] > 0
        and gain_prm[1] == 0
        and gain_prm[2] == 0
        and bias_prm[0] == 0
        and bias_prm[1] == 0
        and bias_prm[2] < 0
        and gain_prm[0] == -bias_prm[2]
    ):
        actuator_damping = -bias_prm[2]
        drive_api.CreateDampingAttr().Set(actuator_damping)
        drive_api.CreateStiffnessAttr().Set(0)

    else:
        _logger.warning(
            "Gain and bias prm arrays are not in the expected format for actuator "
            + f"{mjc_actuator.GetPath()} for joint {joint.GetPath()}, physics drive stiffness and damping will not be created"
        )


def convert_mjc_joint_to_physx(joint: Usd.Prim, stage: Usd.Stage) -> None:
    """Convert a MJCF joint to a PhysX joint.

    Args:
        joint: MJCF joint prim.
        stage: USD stage containing the joint prim.
    """
    # Set joint friction
    joint_friction = (
        joint.GetAttribute("mjc:frictionloss").Get() if joint.GetAttribute("mjc:frictionloss").IsValid() else None
    )
    if joint_friction:
        if joint.HasAPI(PhysxSchema.PhysxJointAPI):
            physx_joint_api = PhysxSchema.PhysxJointAPI(joint)
        else:
            physx_joint_api = PhysxSchema.PhysxJointAPI.Apply(joint)

        physx_joint_api.CreateJointFrictionAttr().Set(joint_friction)

    # Set armature
    joint_armature = joint.GetAttribute("mjc:armature").Get() if joint.GetAttribute("mjc:armature").IsValid() else None
    if joint_armature:
        if joint.HasAPI(PhysxSchema.PhysxJointAPI):
            physx_joint_api = PhysxSchema.PhysxJointAPI(joint)
        else:
            physx_joint_api = PhysxSchema.PhysxJointAPI.Apply(joint)
        physx_joint_api.CreateArmatureAttr().Set(joint_armature)

    # Set target_position
    joint_target_position = joint.GetAttribute("mjc:ref").Get() if joint.GetAttribute("mjc:ref").IsValid() else None
    if joint_target_position:
        if joint.IsA(UsdPhysics.RevoluteJoint):
            joint_type = "angular"
        elif joint.IsA(UsdPhysics.PrismaticJoint):
            joint_type = "linear"
        else:
            return

        if joint.HasAPI(UsdPhysics.DriveAPI, joint_type):
            drive_api = UsdPhysics.DriveAPI(joint, joint_type)
        else:
            drive_api = UsdPhysics.DriveAPI.Apply(joint, joint_type)

        drive_api.CreateTargetPositionAttr().Set(joint_target_position)
