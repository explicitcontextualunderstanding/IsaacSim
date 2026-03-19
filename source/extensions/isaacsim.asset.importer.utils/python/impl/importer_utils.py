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

"""Utility helpers for asset importer preprocessing steps."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence

import usd.schema.newton
from isaacsim.asset.transformer import AssetTransformerManager, RuleProfile
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

_logger = logging.getLogger(__name__)

# USD Geometry types that should be considered as collision shapes.
USD_GEOMETRY_TYPES = {"Mesh", "Cube", "Sphere", "Capsule", "Cylinder", "Cone"}

# Mapping from UI-friendly labels to mesh collision approximation tokens.
MESH_APPROXIMATION_MAP = {
    "Convex Hull": UsdPhysics.Tokens.convexHull,
    "Convex Decomposition": UsdPhysics.Tokens.convexDecomposition,
    "Bounding Sphere": UsdPhysics.Tokens.boundingSphere,
    "Bounding Cube": UsdPhysics.Tokens.boundingCube,
}

PHYSICS_AXIS_MAP = {
    "X": UsdPhysics.Tokens.rotX,
    "Y": UsdPhysics.Tokens.rotY,
    "Z": UsdPhysics.Tokens.rotZ,
}


def collision_from_visuals(stage: Usd.Stage, collision_type: str) -> int:
    """Apply collisions from visual geometry and remove guide colliders.

    Args:
        stage: USD stage for authoring collision APIs.
        collision_type: Collision approximation label. Defaults to convex hull when unknown.

    Returns:
        Number of visual geometry prims processed.

    Example:

    .. code-block:: python

        >>> from pxr import Usd
        >>> import isaacsim.core.experimental.utils.stage as stage_utils
        >>> from isaacsim.asset.importer.utils import collision_from_visuals
        >>>
        >>> stage = Usd.Stage.CreateInMemory()
        >>> stage_utils.use_stage(stage)
        >>> collision_from_visuals(stage, "Convex Hull")  # doctest: +SKIP
    """
    removed_colliders: list[Sdf.Path] = []
    removed_count = 0
    processed_count = 0

    for prim in stage.Traverse():
        prim_type = prim.GetTypeName()
        prim_path = prim.GetPath().pathString
        try:
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                if prim_type in USD_GEOMETRY_TYPES:
                    imageable = UsdGeom.Imageable(prim)
                    purpose = imageable.GetPurposeAttr().Get()
                    if purpose == UsdGeom.Tokens.guide:
                        removed_colliders.append(prim.GetPath())
                        continue

            if prim_type not in USD_GEOMETRY_TYPES:
                continue

            imageable = UsdGeom.Imageable(prim)
            purpose = imageable.GetPurposeAttr().Get()
            if purpose not in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render):
                continue

            if prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_api = UsdPhysics.CollisionAPI(prim)
                collision_enabled_attr = collision_api.GetCollisionEnabledAttr()
                if not collision_enabled_attr:
                    collision_enabled_attr = collision_api.CreateCollisionEnabledAttr()
                collision_enabled_attr.Set(True)
            else:
                UsdPhysics.CollisionAPI.Apply(prim)

            if prim_type == "Mesh":
                if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
                else:
                    mesh_collision_api = UsdPhysics.MeshCollisionAPI(prim)

                approx_type = MESH_APPROXIMATION_MAP.get(collision_type, UsdPhysics.Tokens.convexHull)
                mesh_collision_api.GetApproximationAttr().Set(approx_type)

            processed_count += 1
        except Exception as exc:
            _logger.error(f"Error processing prim {prim_path}: {exc}")
            continue

    for prim_path in removed_colliders:
        stage.RemovePrim(prim_path)
        removed_count += 1

    _logger.info(f"Removed {removed_count} guide collision geometries")
    _logger.info(f"Processed collision for {processed_count} visual geometries")
    return processed_count


def enable_self_collision(usd_stage: Usd.Stage, enabled: bool = True) -> int:
    """Enable self-collisions on articulation roots.

    Args:
        usd_stage: USD stage for authoring PhysX articulation attributes.
        enabled: Whether to enable self collisions on articulation roots.

    Returns:
        Number of articulation roots updated.
    """
    articulation_roots = [
        prim
        for prim in usd_stage.Traverse()
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        or prim.HasAPI("PhysicsArticulationRootAPI")
        or prim.HasAPI("NewtonArticulationRootAPI")
    ]

    if len(articulation_roots) == 0:
        # If no articulation root found, get default prim and create articulation root schema on it.
        default_prim = usd_stage.GetDefaultPrim()
        if not default_prim:
            return 0
        default_prim.ApplyAPI("PhysicsArticulationRootAPI")
        PhysxSchema.PhysxArticulationAPI.Apply(default_prim)
        physx_api = PhysxSchema.PhysxArticulationAPI(default_prim)
        attr = physx_api.GetEnabledSelfCollisionsAttr()
        if not attr:
            attr = physx_api.CreateEnabledSelfCollisionsAttr()
        attr.Set(enabled)

        default_prim.ApplyAPI("NewtonArticulationRootAPI")
        attr = default_prim.GetAttribute("newton:selfCollisionEnabled")
        if not attr:
            attr = default_prim.CreateAttribute("newton:selfCollisionEnabled", Sdf.ValueTypeNames.Bool)
        attr.Set(enabled)
        return 1

    for articulation_root in articulation_roots:
        if not articulation_root or not articulation_root.IsValid():
            continue

        if not articulation_root.HasAPI("PhysicsArticulationRootAPI"):
            articulation_root.ApplyAPI("PhysicsArticulationRootAPI")

        if not articulation_root.HasAPI(PhysxSchema.PhysxArticulationAPI):
            PhysxSchema.PhysxArticulationAPI.Apply(articulation_root)

        if not articulation_root.HasAPI("NewtonArticulationRootAPI"):
            articulation_root.ApplyAPI("NewtonArticulationRootAPI")

        physx_api = PhysxSchema.PhysxArticulationAPI(articulation_root)
        attr = physx_api.GetEnabledSelfCollisionsAttr()
        if not attr:
            attr = physx_api.CreateEnabledSelfCollisionsAttr()
        attr.Set(enabled)

        attr = articulation_root.GetAttribute("newton:selfCollisionEnabled")
        if not attr:
            attr = articulation_root.CreateAttribute("newton:selfCollisionEnabled", Sdf.ValueTypeNames.Bool)
        attr.Set(enabled)

    return len(articulation_roots)


def run_asset_transformer_profile(
    input_stage_path: str,
    output_package_root: str,
    profile_json_path: str,
    *,
    log_path: str | None = None,
) -> None:
    """Run an asset structure profile against an input stage.

    Args:
        input_stage_path: Path to the input USD stage.
        output_package_root: Destination folder for output assets.
        profile_json_path: Path to the profile JSON file.
        log_path: Optional JSON report output path.

    Example:

    .. code-block:: python

        >>> from isaacsim.asset.importer.utils import run_asset_transformer_profile
        >>>
        >>> run_asset_transformer_profile(
        ...     input_stage_path="/tmp/input.usda",
        ...     output_package_root="/tmp/output",
        ...     profile_json_path="/tmp/profile.json",
        ... )  # doctest: +SKIP
    """
    with open(profile_json_path, encoding="utf-8") as handle:
        profile = RuleProfile.from_json(handle.read())

    manager = AssetTransformerManager()
    report = manager.run(
        input_stage_path=input_stage_path,
        profile=profile,
        package_root=output_package_root,
    )

    if log_path:
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as log_file:
            json.dump(json.loads(report.to_json()), log_file, indent=2)


def delete_scope(stage: Usd.Stage, prim_path: str) -> None:
    """Delete a scope prim from the stage, reparenting its children to the parent prim.

    Args:
        stage: USD stage containing the prim.
        prim_path: Path to the prim to delete.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid() or prim.IsPseudoRoot():
        _logger.warning(f"Scope does not exist at path: {prim_path}")
        return
    if not prim.IsA(UsdGeom.Scope):
        _logger.warning(f"Prim is not a scope at path: {prim_path}")
        return

    parent_prim = prim.GetParent()
    if not parent_prim:
        _logger.warning(f"Scope has no parent at path: {prim_path}")
        parent_prim = stage.GetPseudoRoot()

    children = list(prim.GetChildren())
    namespace_editor = Usd.NamespaceEditor(stage)
    for child in children:
        namespace_editor.ReparentPrim(child, parent_prim)

    namespace_editor.ApplyEdits()

    stage.RemovePrim(Sdf.Path(prim_path))


def add_joint_schemas(stage: Usd.Stage) -> None:
    """Apply joint-related physics schemas to all joint prims.

    Args:
        stage: USD stage to update with joint schemas.

    """
    for prim in stage.Traverse():
        if not (prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint)):
            continue

        if not prim.HasAPI(PhysxSchema.PhysxJointAPI):
            PhysxSchema.PhysxJointAPI.Apply(prim)
        instance_name = "angular" if prim.IsA(UsdPhysics.RevoluteJoint) else "linear"

        if not prim.HasAPI(UsdPhysics.DriveAPI, instance_name):
            UsdPhysics.DriveAPI.Apply(prim, instance_name)

        if not prim.HasAPI(PhysxSchema.JointStateAPI, instance_name):
            PhysxSchema.JointStateAPI.Apply(prim, instance_name)


def add_rigid_body_schemas(stage: Usd.Stage) -> None:
    """Apply rigid body-related physics schemas to all rigid body prims.

    Args:
        stage: USD stage to update with rigid body schemas.
    """
    for prim in stage.Traverse():
        if not (prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI("PhysicsRigidBodyAPI")):
            continue
        if not prim.HasAPI(UsdPhysics.MassAPI):
            UsdPhysics.MassAPI.Apply(prim)


def remove_custom_scopes(stage: Usd.Stage) -> None:
    """Remove custom scopes from the stage.

    Args:
        stage: USD stage to update with custom scopes.
    """
    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        return

    default_prim_path = default_prim.GetPath()
    scope_paths = default_prim_path.pathString + "/Geometry/custom"
    scope = stage.GetPrimAtPath(scope_paths)
    if scope and scope.IsA(UsdGeom.Scope):
        stage.RemovePrim(Sdf.Path(scope_paths))
    return


def create_physx_mimic_joint(prim: Usd.Prim) -> None:
    """Create a mimic joint for a joint.

    Args:
        prim: prim to create the mimic joint for.
    """
    if prim.HasAPI("NewtonMimicAPI"):
        # Get the mimic relation as a Usd.Rel
        mimic_rel = prim.GetRelationship("newton:mimicJoint")
        if mimic_rel is None or not mimic_rel.IsValid():
            _logger.warning(f"newton:mimicJoint not found or invalid for prim {prim.GetPath()}")
            return

        target = mimic_rel.GetTargets()[0]
        if not target:
            _logger.warning(f"newton:mimicJoint relationship has no target for prim {prim.GetPath()}")
            return

        # Read newton:mimicCoef1 and newton:mimicCoef0
        mimic_coef1_attr = prim.GetAttribute("newton:mimicCoef1")
        mimic_coef0_attr = prim.GetAttribute("newton:mimicCoef0")
        mimic_coef1 = mimic_coef1_attr.Get() if mimic_coef1_attr and mimic_coef1_attr.IsValid() else None
        mimic_coef0 = mimic_coef0_attr.Get() if mimic_coef0_attr and mimic_coef0_attr.IsValid() else None

        target_prim = prim.GetStage().GetPrimAtPath(target)
        if not target_prim:
            _logger.warning(f"target prim not found for prim {prim.GetPath()}")
            return
        if not target_prim.HasAttribute("physics:axis"):
            _logger.warning(f"target prim does not have physics:axis attribute for prim {prim.GetPath()}")
            return
        target_axis = target_prim.GetAttribute("physics:axis").Get().upper()

        axis = prim.GetAttribute("physics:axis").Get().upper() if prim.HasAttribute("physics:axis") else None
        if axis is None:
            _logger.warning(f"prim does not have physics:axis attribute for prim {prim.GetPath()}")
            return

        target_joint = UsdPhysics.Joint(target_prim)
        # Check if the target joint is flipped, and if so, invert the gearing (mimic_coef1)
        local_rot1 = target_joint.GetLocalRot1Attr().Get()
        if local_rot1.GetReal() == -1 or any(v == -1 for v in local_rot1.GetImaginary()):
            mimic_coef1 = -mimic_coef1
            _logger.info(f"Inverted gearing for prim {prim.GetPath()} because target joint is flipped")

        # Create the PhysX mimic joint attribute on the current prim
        physx_mimic_api = PhysxSchema.PhysxMimicJointAPI.Apply(prim, PHYSICS_AXIS_MAP[axis])
        mimic_rel_physx = physx_mimic_api.CreateReferenceJointRel()
        mimic_rel_physx.SetTargets([target])
        if mimic_coef1 is not None:
            physx_mimic_api.CreateGearingAttr().Set(mimic_coef1)
        else:
            _logger.warning(f"newton:mimicCoef1 not found or invalid for prim {prim.GetPath()}")
        if mimic_coef0 is not None:
            physx_mimic_api.CreateOffsetAttr().Set(mimic_coef0)
        else:
            _logger.warning(f"newton:mimicCoef0 not found or invalid for prim {prim.GetPath()}")
        physx_mimic_api.CreateReferenceJointAxisAttr().Set(PHYSICS_AXIS_MAP[target_axis])
