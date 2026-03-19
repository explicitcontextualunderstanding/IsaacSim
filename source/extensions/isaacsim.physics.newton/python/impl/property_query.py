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
"""Newton property query interface compatible with PhysX property query API.

This module provides a Newton-compatible implementation of the PhysX property query
interface, allowing the same code to work with both backends.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import carb
from pxr import Sdf, Usd, UsdPhysics

if TYPE_CHECKING:
    from .tensors.articulation_view import NewtonArticulationView


class NewtonPropertyQueryArticulationLink:
    """Articulation link information compatible with PhysX response.

    Matches the interface of omni.physx.bindings._physx.PhysxPropertyQueryArticulationLink.

    Args:
        rigid_body: Encoded USD path of the rigid body.
        joint: Encoded USD path of the joint.
        joint_dof: Number of DOFs for this joint.
    """

    def __init__(self, rigid_body: int = 0, joint: int = 0, joint_dof: int = 0):
        self._rigid_body = rigid_body
        self._joint = joint
        self._joint_dof = joint_dof

    @property
    def rigid_body(self) -> int:
        """Encoded USD path of the rigid body.

        Returns:
            Encoded USD path of the rigid body.
        """
        return self._rigid_body

    @property
    def rigid_body_name(self) -> str:
        """USD path of the rigid body as string.

        Returns:
            USD path of the rigid body as string.
        """
        if self._rigid_body:
            from pxr import PhysicsSchemaTools

            return PhysicsSchemaTools.intToSdfPath(self._rigid_body).pathString
        return ""

    @property
    def joint(self) -> int:
        """Encoded USD path of the joint.

        Returns:
            Encoded USD path of the joint.
        """
        return self._joint

    @property
    def joint_name(self) -> str:
        """USD path of the joint as string.

        Returns:
            USD path of the joint as string.
        """
        if self._joint:
            from pxr import PhysicsSchemaTools

            return PhysicsSchemaTools.intToSdfPath(self._joint).pathString
        return ""

    @property
    def joint_dof(self) -> int:
        """Number of DOFs for this joint.

        Returns:
            Number of DOFs for this joint.
        """
        return self._joint_dof


class NewtonPropertyQueryArticulationResponse:
    """Articulation query response compatible with PhysX response.

    Matches the interface of omni.physx.bindings._physx.PhysxPropertyQueryArticulationResponse.

    Args:
        result: Query result code.
        stage_id: USD stage ID.
        path_id: Encoded USD prim path.
        links: List of articulation links.
    """

    def __init__(
        self,
        result: int = 0,
        stage_id: int = 0,
        path_id: int = 0,
        links: list[NewtonPropertyQueryArticulationLink] | None = None,
    ):
        self.result = result
        self.stage_id = stage_id
        self.path_id = path_id
        self.links = links if links is not None else []


class NewtonPropertyQueryInterface:
    """Newton implementation of property query interface."""

    _instance: NewtonPropertyQueryInterface | None = None
    """Singleton instance of the Newton property query interface."""

    def __new__(cls) -> NewtonPropertyQueryInterface:
        """Return singleton instance.

        Returns:
            The singleton instance of NewtonPropertyQueryInterface.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def query_prim(
        self,
        stage_id: int,
        query_mode: int,
        prim_id: int,
        timeout_ms: int = -1,
        finished_fn: Callable | None = None,
        rigid_body_fn: Callable | None = None,
        collider_fn: Callable | None = None,
        articulation_fn: Callable | None = None,
    ) -> None:
        """Query articulation properties from Newton backend.

        Args:
            stage_id: USD stage ID.
            query_mode: Query mode (1 = QUERY_ARTICULATION).
            prim_id: Encoded USD prim path.
            timeout_ms: Timeout in milliseconds (-1 for no timeout).
            finished_fn: Callback when query is finished.
            rigid_body_fn: Callback for rigid body results.
            collider_fn: Callback for collider results.
            articulation_fn: Callback for articulation results.
        """
        # Only handle articulation queries
        if query_mode != 1:  # QUERY_ARTICULATION
            if finished_fn:
                finished_fn()
            return

        if articulation_fn is None:
            if finished_fn:
                finished_fn()
            return

        # Get the stage and prim path
        from pxr import PhysicsSchemaTools, UsdUtils

        stage = UsdUtils.StageCache.Get().Find(Usd.StageCache.Id.FromLongInt(stage_id))
        if not stage:
            response = NewtonPropertyQueryArticulationResponse(result=3)  # ERROR_INVALID_USD_STAGE
            articulation_fn(response)
            if finished_fn:
                finished_fn()
            return

        prim_path = PhysicsSchemaTools.intToSdfPath(prim_id)
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            response = NewtonPropertyQueryArticulationResponse(result=4)  # ERROR_INVALID_USD_PRIM
            articulation_fn(response)
            if finished_fn:
                finished_fn()
            return

        # Get Newton simulation view
        from isaacsim.core.simulation_manager import SimulationManager

        physics_sim_view = SimulationManager._physics_sim_view__warp
        if physics_sim_view is None or not physics_sim_view.is_valid:
            # For USD backend, query directly from USD prims instead of physics simulation
            try:
                links = self._build_articulation_links_from_usd(stage, prim)
                response = NewtonPropertyQueryArticulationResponse(
                    result=0,  # VALID
                    stage_id=stage_id,
                    path_id=prim_id,
                    links=links,
                )
                articulation_fn(response)
            except Exception as e:
                carb.log_error(f"Failed to query articulation from USD: {e}")
                response = NewtonPropertyQueryArticulationResponse(result=6)  # ERROR_RUNTIME
                articulation_fn(response)

            if finished_fn:
                finished_fn()
            return

        # Create articulation view
        articulation_view = physics_sim_view.create_articulation_view(prim_path.pathString)

        if articulation_view is None or articulation_view.count == 0:
            response = NewtonPropertyQueryArticulationResponse(result=6)  # ERROR_RUNTIME
            articulation_fn(response)
            if finished_fn:
                finished_fn()
            return

        # Build the response from Newton's articulation metadata
        try:
            links = self._build_articulation_links(stage, articulation_view)

            response = NewtonPropertyQueryArticulationResponse(
                result=0,  # VALID
                stage_id=stage_id,
                path_id=prim_id,
                links=links,
            )
            articulation_fn(response)

        except Exception as e:
            carb.log_error(f"Newton property query failed: {e}")
            import traceback

            traceback.print_exc()
            response = NewtonPropertyQueryArticulationResponse(result=6)  # ERROR_RUNTIME
            articulation_fn(response)

        if finished_fn:
            finished_fn()

    def _build_articulation_links(
        self, stage: Usd.Stage, articulation_view: "NewtonArticulationView"
    ) -> list[NewtonPropertyQueryArticulationLink]:
        """Build articulation link list matching PhysX format.

        PhysX returns one entry per link, where each link may have an associated
        inbound joint. The mapping is: link[i] is connected by joint[i].

        Args:
            stage: USD stage.
            articulation_view: Newton articulation view.

        Returns:
            List of articulation links in PhysX-compatible format.
        """
        from pxr import PhysicsSchemaTools

        links = []
        # Get paths for the first articulation (homogeneous view, all have same structure)
        all_link_paths = articulation_view.link_paths
        all_joint_paths = articulation_view.joint_paths
        all_dof_paths = articulation_view.dof_paths

        if not all_link_paths or len(all_link_paths) == 0:
            return links

        link_paths = all_link_paths[0]
        joint_paths = all_joint_paths[0] if all_joint_paths else []
        dof_paths = all_dof_paths[0] if all_dof_paths else []

        # Create a mapping from joint paths to DOF count by counting DOFs
        # In Newton, each DOF path corresponds to exactly one joint
        joint_dof_count_map = {}
        for dof_path in dof_paths:
            # Count this DOF for its joint
            joint_dof_count_map[dof_path] = joint_dof_count_map.get(dof_path, 0) + 1

        # Build link entries
        # In Newton's structure, links and joints are aligned (each joint connects to a child link)
        for i, link_path in enumerate(link_paths):
            link = NewtonPropertyQueryArticulationLink()

            # Encode the link path
            link_sdf_path = Sdf.Path(link_path)
            link._rigid_body = PhysicsSchemaTools.sdfPathToInt(link_sdf_path)

            # Map link to its inbound joint
            # joint[i] connects to link[i]
            if i < len(joint_paths):
                joint_path = joint_paths[i]
                if joint_path:
                    joint_sdf_path = Sdf.Path(joint_path)
                    link._joint = PhysicsSchemaTools.sdfPathToInt(joint_sdf_path)
                    # Get DOF count from our mapping
                    link._joint_dof = joint_dof_count_map.get(joint_path, 0)

            links.append(link)

        return links

    def _build_articulation_links_from_usd(
        self, stage: Usd.Stage, articulation_prim: Usd.Prim
    ) -> list[NewtonPropertyQueryArticulationLink]:
        """Build articulation link list by querying USD directly.

        This method is used when the physics simulation view is not available.

        Args:
            stage: USD stage.
            articulation_prim: USD prim for the articulation root.

        Returns:
            List of articulation links in PhysX-compatible format.
        """
        from pxr import PhysicsSchemaTools

        links = []
        link_prim_to_joint_prim: dict[Sdf.Path, Usd.Prim | None] = {}

        def _collect_prims(prim: Usd.Prim) -> None:
            """Recursively collect links and their associated joints.

            Args:
                prim: USD prim to process.
            """
            for child in prim.GetChildren():
                # Check if this is a rigid body (link)
                if child.HasAPI(UsdPhysics.RigidBodyAPI):
                    # Look for joints in the same prim's children or as siblings
                    inbound_joint = None

                    # Check children for joints (common pattern)
                    for joint_child in child.GetChildren():
                        if joint_child.IsA(UsdPhysics.Joint):
                            # Verify this joint connects to this link
                            joint_api = UsdPhysics.Joint(joint_child)
                            body1_rel = joint_api.GetBody1Rel()
                            targets = body1_rel.GetTargets()
                            # If body1 points to this link, this is an inbound joint
                            if targets and targets[0] == child.GetPath():
                                inbound_joint = joint_child
                                break

                    link_prim_to_joint_prim[child.GetPath()] = inbound_joint

                # Recurse into children
                _collect_prims(child)

        _collect_prims(articulation_prim)

        # Second pass: build link entries in order
        for link_path, joint_prim in link_prim_to_joint_prim.items():
            link = NewtonPropertyQueryArticulationLink()
            link._rigid_body = PhysicsSchemaTools.sdfPathToInt(Sdf.Path(link_path))

            # Set joint information if present
            if joint_prim:
                joint_path = Sdf.Path(joint_prim.GetPath())
                link._joint = PhysicsSchemaTools.sdfPathToInt(joint_path)

                # Determine DOF count based on joint type
                if joint_prim.IsA(UsdPhysics.RevoluteJoint) or joint_prim.IsA(UsdPhysics.PrismaticJoint):
                    link._joint_dof = 1
                elif joint_prim.IsA(UsdPhysics.SphericalJoint):
                    link._joint_dof = 3
                elif joint_prim.IsA(UsdPhysics.FixedJoint):
                    link._joint_dof = 0
                else:
                    # Unknown joint type, assume 0 DOFs
                    link._joint_dof = 0

            links.append(link)

        return links

    def _count_joint_dofs(self, stage: Usd.Stage, joint_path: Sdf.Path, dof_map: dict) -> int:
        """Count the number of DOFs for a joint.

        Args:
            stage: USD stage.
            joint_path: Path to the joint prim.
            dof_map: Mapping from joint paths to DOF counts from Newton.

        Returns:
            Number of DOFs for the joint.
        """
        # First check if Newton already counted DOFs for this joint
        path_str = joint_path.pathString
        if path_str in dof_map:
            return dof_map[path_str]

        prim = stage.GetPrimAtPath(joint_path)
        if not prim.IsValid():
            return 0

        # Check joint type from USD schema
        if prim.IsA(UsdPhysics.RevoluteJoint):
            return 1
        if prim.IsA(UsdPhysics.PrismaticJoint):
            return 1
        if prim.IsA(UsdPhysics.FixedJoint):
            return 0
        if prim.IsA(UsdPhysics.SphericalJoint):
            return 3
        if prim.IsA(UsdPhysics.Joint):
            # Generic joint - check for drive attributes to determine DOFs
            dof_count = 0
            for attr in prim.GetAttributes():
                attr_name = attr.GetName()
                if attr_name.startswith("drive:"):
                    dof_count = 1
                    break
            return dof_count if dof_count > 0 else 1

        return 0


def get_newton_property_query_interface() -> NewtonPropertyQueryInterface:
    """Get the Newton property query interface singleton.

    Returns:
        Newton property query interface instance.
    """
    return NewtonPropertyQueryInterface()
