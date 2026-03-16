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

"""Mesh cleanup and merge operations for asset importers."""

from __future__ import annotations

import json
import logging

from pxr import Usd, UsdGeom, UsdPhysics

from . import importer_utils, stage_utils

_logger = logging.getLogger(__name__)


def clean_mesh_operation(stage: Usd.Stage) -> None:
    """Clean mesh prims.

    Args:
        stage: USD stage for executing clean operations.
    """
    try:
        import omni.kit.commands
        import omni.scene.optimizer.core

        clean_config = {
            "operation": "meshCleanup",
            "paths": [],
            "mergeVertices": True,
            "tolerance": 0.0,
            "mergeBoundaries": True,
            "mergeNeighbors": True,
            "contractDegenerateEdges": True,
            "removeDegenerateFaces": True,
            "removeIsolatedVertices": True,
            "removeDuplicateFaces": True,
            "makeManifold": True,
        }

        args = {"jsonFile": json.dumps([clean_config])}
        context = omni.scene.optimizer.core.ExecutionContext()
        context.usdStageId = stage_utils.get_stage_id(stage)
        context.generateReport = 0
        context.captureStats = 0

        omni.kit.commands.execute("SceneOptimizerJsonParser", context=context, args=args)

    except ImportError:
        _logger.error("omni.kit.commands and omni.scene.optimizer.core are not installed")
        return


def generate_mesh_uv_normals_operation(stage: Usd.Stage) -> None:
    """Generate mesh UV normals.

    Args:
        stage: USD stage for executing generate operations.
    """
    try:
        import omni.kit.commands
        import omni.scene.optimizer.core

        generate_normal_config = {
            "operation": "generateNormals",
            "paths": [],  # default to all meshes
            "binding": 0,
            "replaceExisting": True,
            "weightMode": 0,
            "sharpnessAngle": 60.0,
            "gpuThreshold": 500000,
        }

        generate_uv_config = {
            "operation": "generateProjectionUVs",
            "paths": [],  # default to all meshes
            "projectionType": 4,  # cube projection (default)
            "useWorldSpaceScales": True,
            "scaleFactor": 0.01,  # scale factor meters
            "overwriteExisting": True,
        }

        generate_normal_args = {"jsonFile": json.dumps([generate_normal_config])}
        generate_uv_args = {"jsonFile": json.dumps([generate_uv_config])}
        context = omni.scene.optimizer.core.ExecutionContext()
        context.usdStageId = stage_utils.get_stage_id(stage)
        context.generateReport = 0
        context.captureStats = 0

        omni.kit.commands.execute(
            "SceneOptimizerJsonParser", context=context, args=generate_normal_args
        )  # crashes in 109
        omni.kit.commands.execute("SceneOptimizerJsonParser", context=context, args=generate_uv_args)

    except ImportError:
        _logger.error("omni.kit.commands and omni.scene.optimizer.core are not installed")
        return


def merge_meshes_operation(stage: Usd.Stage) -> int:
    """Merge mesh prims grouped under rigid bodies.

    Args:
        stage: USD stage for executing merge operations.

    Returns:
        Number of mesh groups merged.

    Example:

    .. code-block:: python

        >>> from pxr import Usd
        >>> import isaacsim.core.experimental.utils.stage as stage_utils
        >>> from isaacsim.asset.importer.utils import merge_meshes_operation
        >>>
        >>> stage = Usd.Stage.CreateInMemory()
        >>> stage_utils.use_stage(stage)
        >>> merge_meshes_operation(stage)  # doctest: +SKIP
    """
    # generate_mesh_uv_normals_operation(stage)

    rigid_bodies = [
        prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI("PhysicsRigidBodyAPI")
    ]
    mesh_groups = []

    for rigid_body in rigid_bodies:
        if not rigid_body or not rigid_body.IsValid():
            continue

        group = []
        prim_stack = list(rigid_body.GetChildren())
        while prim_stack:
            child = prim_stack.pop()
            if child.HasAPI(UsdPhysics.RigidBodyAPI) or child.HasAPI("PhysicsRigidBodyAPI"):
                continue

            if child.GetTypeName() in importer_utils.USD_GEOMETRY_TYPES:
                imageable = UsdGeom.Imageable(child)
                purpose = imageable.GetPurposeAttr().Get()
                if purpose in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render):
                    group.append(child.GetPath().pathString)
            prim_stack.extend(child.GetChildren())

        if group:
            mesh_groups.append(group)

    for meshes in mesh_groups:
        merge_mesh(stage, meshes)

    return len(mesh_groups)


def merge_mesh(stage: Usd.Stage, meshes: list[str]) -> None:
    """Merge a list of mesh prim paths using Scene Optimizer.

    Args:
        stage: USD stage containing the mesh prims.
        meshes: List of mesh prim paths to merge.

    Example:

    .. code-block:: python

        >>> from pxr import Usd
        >>> import isaacsim.core.experimental.utils.stage as stage_utils
        >>> from isaacsim.asset.importer.utils import merge_mesh
        >>>
        >>> stage = Usd.Stage.CreateInMemory()
        >>> stage_utils.use_stage(stage)
        >>> merge_mesh(stage, ["/World/meshA", "/World/meshB"])  # doctest: +SKIP
    """
    try:
        import omni.kit.commands
        import omni.scene.optimizer.core

        if not meshes:
            return

        merge_config = {
            "operation": "merge",
            "meshPrimPaths": list(meshes),
            "considerMaterials": False,
            "materialAlbedoAsVertexColors": False,
            "originalGeomOption": 1,
            "mergePoint": 0,
            "rootPath": meshes[0],
            "considerAllAttributes": True,
            "allowSingleMeshes": False,
            "spatialMode": 0,
            "spatialThreshold": 10.0,
            "spatialMaxSize": 0.0,
            "spatialVertexCount": 10000,
            "spatialDebug": True,
        }

        args = {"jsonFile": json.dumps([merge_config])}
        context = omni.scene.optimizer.core.ExecutionContext()
        context.usdStageId = stage_utils.get_stage_id(stage)
        context.generateReport = 0
        context.captureStats = 0

        omni.kit.commands.execute("SceneOptimizerJsonParser", context=context, args=args)

    except ImportError:
        _logger.error("omni.kit.commands and omni.scene.optimizer.core are not installed")
        return
