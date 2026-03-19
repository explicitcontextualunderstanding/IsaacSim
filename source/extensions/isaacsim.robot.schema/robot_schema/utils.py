# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Utilities for Isaac robot schema traversal and updates."""
from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Any

import carb
import omni
import pxr
from usd.schema.isaac.robot_schema import *

_DEPRECATED_DOF_ATTRS: tuple[tuple[str, str, pxr.TfToken], ...] = (
    ("isaac:physics:Tr_X:DoFOffset", "TransX", pxr.UsdPhysics.Tokens.transX),
    ("isaac:physics:Tr_Y:DoFOffset", "TransY", pxr.UsdPhysics.Tokens.transY),
    ("isaac:physics:Tr_Z:DoFOffset", "TransZ", pxr.UsdPhysics.Tokens.transZ),
    ("isaac:physics:Rot_X:DoFOffset", "RotX", pxr.UsdPhysics.Tokens.rotX),
    ("isaac:physics:Rot_Y:DoFOffset", "RotY", pxr.UsdPhysics.Tokens.rotY),
    ("isaac:physics:Rot_Z:DoFOffset", "RotZ", pxr.UsdPhysics.Tokens.rotZ),
)
_DOF_OFFSET_ATTR = "isaac:physics:DofOffsetOpOrder"
_TOKEN_FALLBACK_ORDER = {name: index for index, (_, name, _) in enumerate(_DEPRECATED_DOF_ATTRS)}

# Robot paths that have already emitted "missing from schema" warnings (one-time per path).
_warned_missing_schema_joints: set[str] = set()
_warned_missing_schema_links: set[str] = set()


def _path_key(path: pxr.Sdf.Path | None) -> str | None:
    """Normalize a Sdf.Path to a plain string key.

    Args:
        path: The path to normalize.

    Returns:
        The path as a string, or None if not provided.
    """
    if not path:
        return None
    return str(path)


def _find_articulation_root(prim: pxr.Usd.Prim) -> pxr.Usd.Prim | None:
    """Return prim itself if it carries ArticulationRootAPI, else search its subtree.

    Args:
        prim: Starting prim to search from.

    Returns:
        The prim carrying ArticulationRootAPI, or None if not found.
    """
    if prim.HasAPI(pxr.UsdPhysics.ArticulationRootAPI):
        return prim
    for child in pxr.Usd.PrimRange(prim):
        if child.HasAPI(pxr.UsdPhysics.ArticulationRootAPI):
            return child
    return None


def _collect_deprecated_dof_values(joint_prim: pxr.Usd.Prim) -> dict[str, int]:
    """Collect authored deprecated DOF attributes and their order values.

    Args:
        joint_prim: The joint prim to inspect for deprecated DoFOffset attributes.

    Returns:
        Dict mapping token name (e.g. "RotX") to its order index value.
    """
    deprecated_values: dict[str, int] = {}
    for attr_name, token_name, _ in _DEPRECATED_DOF_ATTRS:
        attr = joint_prim.GetAttribute(attr_name)
        if not attr or not attr.HasAuthoredValueOpinion():
            continue
        value = attr.Get()
        if value is None:
            continue
        try:
            deprecated_values[token_name] = int(value)
        except (TypeError, ValueError):
            continue
    return deprecated_values


def _remove_deprecated_dof_attrs(joint_prim: pxr.Usd.Prim, deprecated_tokens: set[str]):
    """Remove deprecated DOF attributes from the edit layer.

    Args:
        joint_prim: The joint prim containing deprecated attributes.
        deprecated_tokens: Set of token names (e.g. "RotX") whose attributes to remove.
    """
    if not deprecated_tokens:
        return
    stage = joint_prim.GetStage()
    edit_layer = stage.GetEditTarget().GetLayer()
    prim_spec = edit_layer.GetPrimAtPath(joint_prim.GetPath())
    if not prim_spec:
        return
    for attr_name, token_name, _ in _DEPRECATED_DOF_ATTRS:
        if token_name in deprecated_tokens and attr_name in prim_spec.attributes:
            prim_spec.RemoveProperty(prim_spec.attributes[attr_name])


def _collect_robot_prims(
    stage: pxr.Usd.Stage,
    prim: pxr.Usd.Prim,
    *,
    target_api: str,
    relation: Relations,
    include_predicate: Callable[[pxr.Usd.Prim], bool] | None = None,
    descend_predicate: Callable[[pxr.Usd.Prim], bool] | None = None,
    _visited: set[str] | None = None,
) -> list[pxr.Usd.Prim]:
    """Collect robot-related prims by walking schema relationships.

    Args:
        stage: The USD stage containing the prims.
        prim: The root prim to traverse.
        target_api: The API schema name to collect.
        relation: The robot relationship to traverse.
        include_predicate: Predicate to include a prim in results.
        descend_predicate: Predicate to descend into a prim's relationships.
        _visited: Internal set for cycle detection.

    Returns:
        List of collected USD prims matching the target API.
    """
    if not stage or not prim:
        return []

    # Cycle detection to prevent infinite recursion from self-referencing relationships
    if _visited is None:
        _visited = set()
    prim_path_str = str(prim.GetPath())
    if prim_path_str in _visited:
        return []
    _visited.add(prim_path_str)

    include_predicate = include_predicate or (lambda node: node.HasAPI(target_api))
    descend_predicate = descend_predicate or (lambda _node: False)

    if include_predicate(prim):
        return [prim]
    if not prim.HasAPI(Classes.ROBOT_API.value):
        return []

    relationship = prim.GetRelationship(relation.name)
    if not relationship:
        return []

    collected: list[pxr.Usd.Prim] = []
    for target in relationship.GetTargets():
        child = stage.GetPrimAtPath(target)
        if not child:
            continue
        if include_predicate(child):
            collected.append(child)
            continue
        if descend_predicate(child):
            collected.extend(
                _collect_robot_prims(
                    stage,
                    child,
                    target_api=target_api,
                    relation=relation,
                    include_predicate=include_predicate,
                    descend_predicate=descend_predicate,
                    _visited=_visited,
                )
            )
    return collected


def _is_single_dof_joint(joint_prim: pxr.Usd.Prim) -> bool:
    """Check if joint is a single-DOF or zero-DOF type (Revolute, Prismatic, Fixed).

    Args:
        joint_prim: The joint prim to inspect.

    Returns:
        True if the joint is Revolute, Prismatic, or Fixed.
    """
    return (
        joint_prim.IsA(pxr.UsdPhysics.RevoluteJoint)
        or joint_prim.IsA(pxr.UsdPhysics.PrismaticJoint)
        or joint_prim.IsA(pxr.UsdPhysics.FixedJoint)
    )


def UpdateDeprecatedJointDofOrder(joint_prim: pxr.Usd.Prim) -> bool:
    """Update isaac:physics:DofOffsetOpOrder using legacy attributes when applicable.

    Migrates deprecated per-axis DoFOffset attributes to the new DofOffsetOpOrder
    token array format. If the joint already has a DofOffsetOpOrder authored, the
    tokens are re-sorted according to the deprecated order values. After migration,
    removes the deprecated attributes from the current edit layer.

    For single-DOF joints (Revolute, Prismatic) and zero-DOF joints (Fixed), the
    DofOffsetOpOrder is not authored since ordering is not meaningful. Deprecated
    attributes are still removed.

    Args:
        joint_prim: The joint prim to update.

    Returns:
        True if migration was performed, False otherwise.
    """
    if not joint_prim:
        return False
    if not joint_prim.IsA(pxr.UsdPhysics.Joint):
        return False

    # Step 1: Collect deprecated attribute values
    deprecated_values = _collect_deprecated_dof_values(joint_prim)
    if not deprecated_values:
        return False

    # Step 2: For single-DOF joints, just remove deprecated attrs without authoring OP order
    if _is_single_dof_joint(joint_prim):
        _remove_deprecated_dof_attrs(joint_prim, set(deprecated_values.keys()))
        return True

    # Step 3: Get existing DofOffsetOpOrder or build from deprecated values
    dof_attr = joint_prim.GetAttribute(_DOF_OFFSET_ATTR)
    existing_tokens: list[str] = []
    if dof_attr and dof_attr.HasAuthoredValueOpinion():
        value = dof_attr.Get()
        if value:
            existing_tokens = list(value)

    # Step 4: Build the final token list
    if existing_tokens:
        # Re-sort existing tokens using deprecated order values where they overlap
        def sort_key(token: str) -> tuple[int, int]:
            if token in deprecated_values:
                return (deprecated_values[token], _TOKEN_FALLBACK_ORDER.get(token, 0))
            # Tokens not in deprecated values keep their relative order at the end
            return (len(deprecated_values) + existing_tokens.index(token), 0)

        ordered_tokens = sorted(existing_tokens, key=sort_key)
    else:
        # Build from deprecated values only
        entries = [(order_idx, token) for token, order_idx in deprecated_values.items()]
        entries.sort(key=lambda item: (item[0], _TOKEN_FALLBACK_ORDER.get(item[1], item[0])))
        ordered_tokens = [token for _, token in entries]

    # Step 5: Author the new attribute
    if not dof_attr:
        dof_attr = joint_prim.CreateAttribute(_DOF_OFFSET_ATTR, pxr.Sdf.ValueTypeNames.TokenArray, False)
    dof_attr.Set(pxr.Vt.TokenArray(ordered_tokens))

    # Step 6: Remove deprecated attributes
    _remove_deprecated_dof_attrs(joint_prim, set(deprecated_values.keys()))

    return True


def _discover_articulation_prims(
    stage: pxr.Usd.Stage, robot_prim: pxr.Usd.Prim
) -> tuple[list[pxr.Usd.Prim], list[pxr.Usd.Prim]]:
    """Discover all links and joints by traversing the articulation graph.

    Uses the same logic as PopulateRobotSchemaFromArticulation to find all
    connected rigid bodies and joints.

    Args:
        stage: The USD stage containing the robot.
        robot_prim: The USD prim representing the robot.

    Returns:
        Tuple of (discovered_links, discovered_joints).
    """
    if not stage or not robot_prim:
        return [], []

    articulation_root = _find_articulation_root(robot_prim)
    if articulation_root is None:
        return [], []

    root_link = articulation_root
    articulation_joints = [prim for prim in pxr.Usd.PrimRange(robot_prim) if prim.IsA(pxr.UsdPhysics.Joint)]

    root_joint = None
    if articulation_root.IsA(pxr.UsdPhysics.Joint):
        root_joint = articulation_root
        candidate_path = GetJointBodyRelationship(root_joint, 0) or GetJointBodyRelationship(root_joint, 1)
        if candidate_path:
            root_link = stage.GetPrimAtPath(candidate_path)
    if not root_link:
        return [], [root_joint] if root_joint else []

    root_link_key = str(root_link.GetPath())
    body_to_joints: dict[str, list[tuple[pxr.Usd.Prim, int]]] = {}
    for joint_prim in articulation_joints:
        for body_index in (0, 1):
            body_path = GetJointBodyRelationship(joint_prim, body_index)
            key = _path_key(body_path)
            if not key:
                continue
            body_to_joints.setdefault(key, []).append((joint_prim, body_index))
            if not root_joint and root_link_key and key == root_link_key:
                root_joint = joint_prim

    queue: deque[pxr.Usd.Prim] = deque()
    visited_links: set[str] = set()
    visited_joints: set[str] = set()
    ordered_links: list[pxr.Usd.Prim] = []
    ordered_joints: list[pxr.Usd.Prim] = []

    if root_joint:
        ordered_joints.append(root_joint)
        visited_joints.add(str(root_joint.GetPath()))

    if root_link:
        queue.append(root_link)

    while queue:
        link_prim = queue.popleft()
        if not link_prim:
            continue
        link_key = str(link_prim.GetPath())
        if link_key in visited_links:
            continue
        visited_links.add(link_key)
        if link_prim.HasAPI(pxr.UsdPhysics.RigidBodyAPI):
            ordered_links.append(link_prim)

        for joint_prim, body_index in body_to_joints.get(link_key, []):
            joint_key = str(joint_prim.GetPath())
            if joint_key not in visited_joints:
                ordered_joints.append(joint_prim)
                visited_joints.add(joint_key)

            other_index = 1 - body_index
            other_path = GetJointBodyRelationship(joint_prim, other_index)
            if not other_path:
                continue
            other_key = str(other_path)
            if other_key in visited_links:
                continue
            other_prim = stage.GetPrimAtPath(other_path)
            if other_prim:
                queue.append(other_prim)

    return ordered_links, ordered_joints


def GetAllRobotJoints(
    stage: pxr.Usd.Stage, robot_link_prim: pxr.Usd.Prim, parse_nested_robots: bool = True
) -> list[pxr.Usd.Prim]:
    """Return all the joints of the robot.

    Retrieves joints from the robot schema relationships and supplements with
    any missing joints discovered through articulation traversal. Missing joints
    are appended to the end of the list and a warning is issued.

    Args:
        stage: The USD stage containing the robot.
        robot_link_prim: The USD prim representing the robot link.
        parse_nested_robots: Whether to parse nested robots for joints.

    Returns:
        A list of prims representing the joints of the robot.

    Example:

    .. code-block:: python

        joints = GetAllRobotJoints(stage, robot_root_prim)
    """

    def _descend(child: pxr.Usd.Prim) -> bool:
        """Decide whether to descend into a child prim.

        Args:
            child: The child prim to evaluate.

        Returns:
            True if nested robots should be parsed for this prim.
        """
        return parse_nested_robots and child.HasAPI(Classes.ROBOT_API.value)

    schema_joints = _collect_robot_prims(
        stage,
        robot_link_prim,
        target_api=Classes.JOINT_API.value,
        relation=Relations.ROBOT_JOINTS,
        descend_predicate=_descend,
    )

    _, discovered_joints = _discover_articulation_prims(stage, robot_link_prim)

    schema_joint_paths = {str(j.GetPath()) for j in schema_joints}
    missing_joints = [j for j in discovered_joints if str(j.GetPath()) not in schema_joint_paths]

    if missing_joints:
        robot_path = str(robot_link_prim.GetPath())
        if robot_path not in _warned_missing_schema_joints:
            _warned_missing_schema_joints.add(robot_path)
            missing_paths = [str(j.GetPath()) for j in missing_joints]
            carb.log_warn(
                f"Robot at {robot_link_prim.GetPath()} has joints missing from schema relationship: {missing_paths}"
            )
        schema_joints.extend(missing_joints)

    return schema_joints


def GetAllRobotLinks(
    stage: pxr.Usd.Stage, robot_link_prim: pxr.Usd.Prim, include_reference_points: bool = False
) -> list[pxr.Usd.Prim]:
    """Return all the links of the robot.

    Retrieves links from the robot schema relationships and supplements with
    any missing links discovered through articulation traversal. Missing links
    are appended to the end of the list and a warning is issued.

    Args:
        stage: The USD stage containing the robot.
        robot_link_prim: The USD prim representing the robot link.
        include_reference_points: Whether to include reference points as links.

    Returns:
        A list of prims representing the links of the robot.

    Example:

    .. code-block:: python

        links = GetAllRobotLinks(stage, robot_root_prim)
    """

    def _include(prim: pxr.Usd.Prim) -> bool:
        """Decide whether a prim should be included as a link.

        Args:
            prim: The prim to evaluate.

        Returns:
            True if the prim should be treated as a link.
        """
        if prim.HasAPI(Classes.LINK_API.value):
            return True
        if include_reference_points and (
            prim.HasAPI(Classes.REFERENCE_POINT_API.value) or prim.HasAPI(Classes.SITE_API.value)
        ):
            if prim.HasAPI(Classes.REFERENCE_POINT_API.value):
                carb.log_warn(f"{prim.GetPath()} has ReferencePointAPI which is deprecated. Use SiteAPI instead.")
            return True
        return False

    schema_links = _collect_robot_prims(
        stage,
        robot_link_prim,
        target_api=Classes.LINK_API.value,
        relation=Relations.ROBOT_LINKS,
        include_predicate=_include,
        descend_predicate=lambda prim: prim.HasAPI(Classes.ROBOT_API.value),
    )

    discovered_links, _ = _discover_articulation_prims(stage, robot_link_prim)

    schema_link_paths = {str(lnk.GetPath()) for lnk in schema_links}
    missing_links = [lnk for lnk in discovered_links if str(lnk.GetPath()) not in schema_link_paths]

    if missing_links:
        robot_path = str(robot_link_prim.GetPath())
        if robot_path not in _warned_missing_schema_links:
            _warned_missing_schema_links.add(robot_path)
            missing_paths = [str(lnk.GetPath()) for lnk in missing_links]
            carb.log_warn(
                f"Robot at {robot_link_prim.GetPath()} has links missing from schema relationship: {missing_paths}"
            )
        schema_links.extend(missing_links)

    return schema_links


def GetAllNamedPoses(stage: pxr.Usd.Stage, robot_prim: pxr.Usd.Prim) -> list[pxr.Usd.Prim]:
    """Return all named pose prims for the robot from the schema relationship.

    Args:
        stage: The USD stage containing the robot.
        robot_prim: The USD prim representing the robot (must have Robot API).

    Returns:
        A list of prims representing the robot's named poses, in relationship order.

    Example:

    .. code-block:: python

        named_poses = GetAllNamedPoses(stage, robot_prim)
    """
    if not stage or not robot_prim or not robot_prim.HasAPI(Classes.ROBOT_API.value):
        return []
    rel = robot_prim.GetRelationship(Relations.NAMED_POSES.name)
    if not rel:
        return []
    prims = []
    for target in rel.GetTargets():
        prim = stage.GetPrimAtPath(target)
        if prim and prim.IsValid():
            prims.append(prim)
    return prims


def GetNamedPoseStartLink(named_pose_prim: pxr.Usd.Prim) -> pxr.Sdf.Path | None:
    """Get the start link path from a named pose prim.

    Args:
        named_pose_prim: The named pose prim.

    Returns:
        The target path of the start link relationship, or None.
    """
    if not named_pose_prim:
        return None
    rel = named_pose_prim.GetRelationship(Relations.POSE_START_LINK.name)
    if not rel:
        return None
    targets = rel.GetTargets()
    return targets[0] if targets else None


def GetNamedPoseEndLink(named_pose_prim: pxr.Usd.Prim) -> pxr.Sdf.Path | None:
    """Get the end link path from a named pose prim.

    Args:
        named_pose_prim: The named pose prim.

    Returns:
        The target path of the end link relationship, or None.
    """
    if not named_pose_prim:
        return None
    rel = named_pose_prim.GetRelationship(Relations.POSE_END_LINK.name)
    if not rel:
        return None
    targets = rel.GetTargets()
    return targets[0] if targets else None


def GetNamedPoseJoints(named_pose_prim: pxr.Usd.Prim) -> list[pxr.Sdf.Path]:
    """Get the joint paths from a named pose prim.

    Args:
        named_pose_prim: The named pose prim.

    Returns:
        List of joint target paths, in order.
    """
    if not named_pose_prim:
        return []
    rel = named_pose_prim.GetRelationship(Relations.POSE_JOINTS.name)
    if not rel:
        return []
    return list(rel.GetTargets())


def GetNamedPoseJointValues(named_pose_prim: pxr.Usd.Prim) -> list[float] | None:
    """Get the joint values from a named pose prim.

    Args:
        named_pose_prim: The named pose prim.

    Returns:
        The joint values array, or None if not authored.
    """
    if not named_pose_prim:
        return None
    attr = named_pose_prim.GetAttribute(Attributes.POSE_JOINT_VALUES.name)
    if not attr:
        return None
    return attr.Get()


def GetNamedPoseJointFixed(named_pose_prim: pxr.Usd.Prim) -> list[bool] | None:
    """Get the joint fixed mask from a named pose prim.

    Args:
        named_pose_prim: The named pose prim.

    Returns:
        The joint fixed array, or None if not authored.
    """
    if not named_pose_prim:
        return None
    attr = named_pose_prim.GetAttribute(Attributes.POSE_JOINT_FIXED.name)
    if not attr:
        return None
    return attr.Get()


def GetNamedPoseValid(named_pose_prim: pxr.Usd.Prim) -> bool | None:
    """Get the valid flag from a named pose prim.

    Args:
        named_pose_prim: The named pose prim.

    Returns:
        The valid value, or None if not authored.
    """
    if not named_pose_prim:
        return None
    attr = named_pose_prim.GetAttribute(Attributes.POSE_VALID.name)
    if not attr:
        return None
    return attr.Get()


class RobotLinkNode:
    """Node in the robot's kinematic tree structure.

    Stores the link prim, parent relationship, and joint connection
    used when building a hierarchical robot representation.

    Args:
        prim: The USD prim representing this link.
        parentLink: The parent link node, if any.
        joint: The joint prim connecting this link to its parent.

    Example:

    .. code-block:: python

        root_node = RobotLinkNode(root_prim)
    """

    def __init__(self, prim: pxr.Usd.Prim, parentLink: RobotLinkNode = None, joint: pxr.Usd.Prim = None):
        self.prim = prim
        if prim:
            self.path = prim.GetPath()
            self.name = prim.GetName()
        else:
            self.path = None
            self.name = None
        self._parent = parentLink
        self._children = []
        self._joints = []
        self._joint_to_parent = joint

    def add_child(self, child: RobotLinkNode):
        """Add a child link node to this node.

        Args:
            child: The child node to attach.

        Example:

        .. code-block:: python

            node.add_child(child_node)
        """
        self.children.append(child)

    @property
    def children(self):
        """List of child nodes.

        Returns:
            List of child link nodes.

        Example:

        .. code-block:: python

            children = node.children
        """
        return self._children

    @property
    def parent(self):
        """Parent node.

        Returns:
            The parent link node, or None if this is the root.

        Example:

        .. code-block:: python

            parent = node.parent
        """
        return self._parent


def GetJointBodyRelationship(joint_prim: pxr.Usd.Prim, bodyIndex: int):
    """Get the relationship target for a joint's body connection.

    Args:
        joint_prim: The USD prim representing the joint.
        bodyIndex: Index of the body (0 or 1) to get the relationship for.

    Returns:
        The target path of the body relationship, or None if not found or excluded from articulation.

    Example:

    .. code-block:: python

        body_path = GetJointBodyRelationship(joint_prim, 0)
    """
    joint = pxr.UsdPhysics.Joint(joint_prim)
    if joint:
        # Ignoring joints with ExcludeFromArticulation to avoid graph instances
        exclude_from_articulation = joint.GetExcludeFromArticulationAttr()
        if exclude_from_articulation and exclude_from_articulation.Get():
            return None
        rel = None
        if bodyIndex == 0:
            rel = joint.GetBody0Rel()
        elif bodyIndex == 1:
            rel = joint.GetBody1Rel()
        if rel:
            targets = rel.GetTargets()
            if targets:
                return targets[0]
    return None


def PrintRobotTree(root: RobotLinkNode, indent=0):
    """Print a visual representation of the robot link tree structure.

    Args:
        root: The root node of the robot link tree.
        indent: Number of spaces to indent each level.

    Example:

    .. code-block:: python

        PrintRobotTree(root_node)
    """
    print(" " * indent + root.name)
    for child in root.children:
        PrintRobotTree(child, indent + 2)


def _get_joint_local_transform(joint: pxr.UsdPhysics.Joint, body_index: int) -> pxr.Gf.Matrix4f | None:
    """Get the local transform of a joint body connection.

    Returns ``Gf.Matrix4f`` (single precision) because the USD joint
    attributes ``localPos`` and ``localRot`` are stored as ``Vec3f`` /
    ``Quatf``.  Callers that need double precision should promote the
    result via ``Gf.Matrix4d(local_transform)``.

    Args:
        joint: The USD physics joint.
        body_index: Body index (0 or 1) to read the local transform from.

    Returns:
        The local transform matrix (single precision), or None if attributes are missing.
    """
    translate_attr = joint.GetLocalPos0Attr() if body_index == 0 else joint.GetLocalPos1Attr()
    rotate_attr = joint.GetLocalRot0Attr() if body_index == 0 else joint.GetLocalRot1Attr()
    if not translate_attr or not rotate_attr:
        return None
    translate = translate_attr.Get()
    rotate = rotate_attr.Get()
    if translate is None or rotate is None:
        return None
    local_transform = pxr.Gf.Matrix4f()
    local_transform.SetTranslate(translate)
    local_transform.SetRotateOnly(rotate)
    return local_transform


def GetJointPose(robot_prim: pxr.Usd.Prim, joint_prim: pxr.Usd.Prim) -> pxr.Gf.Matrix4d:
    """Get the pose of a joint in the robot's coordinate system.

    Args:
        robot_prim: The USD prim representing the robot.
        joint_prim: The USD prim representing the joint.

    Returns:
        The joint's pose matrix in robot coordinates, or None if pose cannot be computed.

    Example:

    .. code-block:: python

        joint_pose = GetJointPose(robot_prim, joint_prim)
    """
    robot_transform = pxr.Gf.Matrix4d(omni.usd.get_world_transform_matrix(robot_prim))
    joint = pxr.UsdPhysics.Joint(joint_prim)
    if not joint:
        return None

    stage = joint_prim.GetStage()
    inverse_robot = robot_transform.GetInverse()
    compose_order = {0: lambda local, body_pose: local * body_pose, 1: lambda local, body_pose: body_pose * local}

    for body_index in (0, 1):
        body_path = GetJointBodyRelationship(joint_prim, body_index)
        if not body_path:
            continue
        body_prim = stage.GetPrimAtPath(body_path)
        if not body_prim:
            continue
        local_transform = _get_joint_local_transform(joint, body_index)
        if not local_transform:
            continue
        local_d = pxr.Gf.Matrix4d(local_transform)
        body_pose = pxr.Gf.Matrix4d(omni.usd.get_world_transform_matrix(body_prim))
        joint_pose = compose_order[body_index](local_d, body_pose)
        return joint_pose * inverse_robot

    return None


def GetLinksFromJoint(root: RobotLinkNode, joint_prim: pxr.Usd.Prim) -> tuple[list[pxr.Usd.Prim], list[pxr.Usd.Prim]]:
    """Return links before and after a joint in the kinematic chain.

    Args:
        root: The root node of the robot link tree.
        joint_prim: The joint prim to find in the tree.

    Returns:
        Lists of links (before_joint, after_joint).

    Example:

    .. code-block:: python

        before_links, after_links = GetLinksFromJoint(root_node, joint_prim)
    """

    def find_node_with_joint(node: RobotLinkNode, target_joint: pxr.Usd.Prim) -> RobotLinkNode:
        """Find the node that references the target joint as its parent joint.

        Args:
            node: The node to search.
            target_joint: The joint prim to match.

        Returns:
            The matching node, or None if not found.
        """
        if node._joint_to_parent == target_joint:
            return node
        for child in node.children:
            result = find_node_with_joint(child, target_joint)
            if result:
                return result
        return None

    def collect_forward_links(node: RobotLinkNode) -> list[pxr.Usd.Prim]:
        """Collect links in the forward direction from a node.

        Args:
            node: The node to collect links from.

        Returns:
            List of link prims in the forward direction.
        """
        links = [node.prim]
        for child in node.children:
            links.extend(collect_forward_links(child))
        return links

    def collect_backward_links(node: RobotLinkNode) -> list[pxr.Usd.Prim]:
        """Collect links in the backward direction from a node.

        Args:
            node: The node to collect links from.

        Returns:
            List of link prims in the backward direction.
        """
        links = [node.prim]
        current = node.parent
        old_current = node
        while current:
            links.append(current.prim)
            if current:
                for child in current.children:
                    if child != old_current and child.prim not in links:
                        links += collect_forward_links(child)
            current = current.parent
        return links

    # Find the node that has our target joint as _joint_to_parent
    node_after_joint = find_node_with_joint(root, joint_prim)
    if not node_after_joint:
        return [], []

    # Get links after the joint (including the link after the joint)
    forward_links = collect_forward_links(node_after_joint)

    # Get links before the joint (including the link before the joint)
    backward_links = collect_backward_links(node_after_joint.parent)

    return backward_links, forward_links


def GenerateRobotLinkTree(stage: pxr.Usd.Stage, robot_link_prim: pxr.Usd.Prim = None) -> RobotLinkNode:
    """Generate a tree structure of robot links using an iterative approach.

    Args:
        stage: The USD stage containing the robot.
        robot_link_prim: The root prim of the robot.

    Returns:
        The root node of the generated tree, or None if no links are found.

    Example:

    .. code-block:: python

        tree_root = GenerateRobotLinkTree(stage, robot_root_prim)
    """
    # Get all links and joints
    all_links = GetAllRobotLinks(stage, robot_link_prim)
    all_joints = GetAllRobotJoints(stage, robot_link_prim)

    # Get root link and initialize mappings
    links = GetAllRobotLinks(stage, robot_link_prim)
    if not links:
        return None
    root = RobotLinkNode(links[0])
    joints_per_body = [{link.GetPath(): [] for link in all_links} for _ in range(2)]

    # Build joint mappings
    for joint in all_joints:
        for body_index in (0, 1):
            body = GetJointBodyRelationship(joint, body_index)
            if body and body in joints_per_body[body_index]:
                joints_per_body[body_index][body].append(joint)

    # Use a stack for iterative traversal
    stack = [root]
    processed_joints = []
    unprocessed_joints = [a for a in all_joints]
    while stack:
        current = stack.pop()
        current_path = current.prim.GetPath()
        for i in range(2):
            joints = joints_per_body[i].get(current_path, [])
            for joint in joints:
                if joint not in processed_joints:
                    processed_joints.append(joint)
                    if joint in unprocessed_joints:
                        unprocessed_joints.remove(joint)
                    # Note: the original correct logic for body1 assignment is likely using 1 - i for proper traversal (as in one parent, one child).
                    body1 = GetJointBodyRelationship(joint, 1 - i)
                    # Check if it's not connecting to its parent
                    if body1 and (current.parent is None or current.parent.path != body1):
                        child = RobotLinkNode(stage.GetPrimAtPath(body1), current, joint)
                        current._joints.append(joint)
                        current.add_child(child)
                        stack.append(child)

    return root


def _detect_sites_for_link(link_prim: pxr.Usd.Prim) -> list[pxr.Usd.Prim]:
    """Detect child Xforms with no children under a link prim that qualify as sites.

    Args:
        link_prim: The link prim to scan for sites.

    Returns:
        List of child prims that qualify as sites.
    """
    if not link_prim or not link_prim.IsValid():
        return []

    sites: list[pxr.Usd.Prim] = []
    instance_proxy_predicate = pxr.Usd.TraverseInstanceProxies()
    for child in link_prim.GetFilteredChildren(instance_proxy_predicate):
        # Check if it's an Xform with no children (including instance proxies)
        if not child.IsA(pxr.UsdGeom.Xform):
            continue
        if child.GetFilteredChildren(instance_proxy_predicate):
            continue
        # Skip if already has SiteAPI or ReferencePointAPI
        if child.HasAPI(Classes.SITE_API.value):
            continue
        if child.HasAPI(Classes.REFERENCE_POINT_API.value):
            continue
        # Skip if it has other meaningful APIs (rigid body, joint, etc.)
        if child.HasAPI(pxr.UsdPhysics.RigidBodyAPI):
            continue
        if child.IsA(pxr.UsdPhysics.Joint):
            continue
        sites.append(child)
    return sites


def PopulateRobotSchemaFromArticulation(
    stage: pxr.Usd.Stage,
    robot_prim: pxr.Usd.Prim,
    articulation_prim: pxr.Usd.Prim | None = None,
    *,
    detect_sites: bool = False,
    sites_last: bool = False,
) -> tuple[pxr.Usd.Prim | None, pxr.Usd.Prim | None]:
    """Populate robot schema relationships using a PhysicsArticulation traversal.

    Discovers the articulation root link and root joint, walks all connected rigid bodies through
    their joints, applies the LinkAPI and JointAPI schemas, and updates the robot relationships
    with the ordered lists of discovered prims.

    Args:
        stage: Stage containing the robot articulation.
        robot_prim: Prim that already has the RobotAPI applied.
        articulation_prim: Optional prim that owns the PhysicsArticulationRootAPI. Defaults
            to ``robot_prim`` when omitted.
        detect_sites: If True, detect and apply SiteAPI to child Xforms with no children
            under each link as it's processed.
        sites_last: If False (default), sites are added immediately after their parent link
            in the robotLinks list. If True, all sites are added at the end of the list
            in their order of appearance.

    Returns:
        The detected root link prim and root joint prim as a two-element tuple.

    Raises:
        ValueError: If the stage or robot prim is invalid.

    Example:

    .. code-block:: python

        >>> stage = omni.usd.get_context().get_stage()
        >>> robot = stage.GetPrimAtPath("/World/MyRobot")
        >>> root_link, root_joint = PopulateRobotSchemaFromArticulation(stage, robot, robot)
    """
    if not stage:
        raise ValueError("Stage is invalid.")
    if not robot_prim:
        raise ValueError("Robot prim is invalid.")

    articulation_root = _find_articulation_root(articulation_prim or robot_prim)
    if articulation_root is None:
        return None, None

    root_link = articulation_root
    articulation_joints = [prim for prim in pxr.Usd.PrimRange(robot_prim) if prim.IsA(pxr.UsdPhysics.Joint)]

    root_joint = None
    if articulation_root.IsA(pxr.UsdPhysics.Joint):
        root_joint = articulation_root
        candidate_path = GetJointBodyRelationship(root_joint, 0) or GetJointBodyRelationship(root_joint, 1)
        if candidate_path:
            root_link = stage.GetPrimAtPath(candidate_path)
    if not root_link:
        return None, root_joint
    root_link_key = str(root_link.GetPath())
    body_to_joints: dict[str, list[tuple[pxr.Usd.Prim, int]]] = {}
    for joint_prim in articulation_joints:
        for body_index in (0, 1):
            body_path = GetJointBodyRelationship(joint_prim, body_index)
            key = _path_key(body_path)
            if not key:
                continue
            body_to_joints.setdefault(key, []).append((joint_prim, body_index))
            if not root_joint and root_link_key and key == root_link_key:
                root_joint = joint_prim

    queue: deque[pxr.Usd.Prim] = deque()
    visited_links: set[str] = set()
    visited_joints: set[str] = set()
    ordered_links: list[pxr.Usd.Prim] = []
    ordered_joints: list[pxr.Usd.Prim] = []
    deferred_sites: list[pxr.Usd.Prim] = []  # For sites_last mode

    if root_joint:
        ordered_joints.append(root_joint)
        visited_joints.add(str(root_joint.GetPath()))
        ApplyJointAPI(root_joint)

    if root_link:
        queue.append(root_link)

    while queue:
        link_prim = queue.popleft()
        if not link_prim:
            continue
        link_key = str(link_prim.GetPath())
        if link_key in visited_links:
            continue
        visited_links.add(link_key)
        ApplyLinkAPI(link_prim)
        ordered_links.append(link_prim)

        # Detect and apply sites for this link
        if detect_sites:
            link_sites = _detect_sites_for_link(link_prim)
            for site in link_sites:
                ApplySiteAPI(site)
                if sites_last:
                    deferred_sites.append(site)
                else:
                    # Add site immediately after its parent link
                    ordered_links.append(site)

        for joint_prim, body_index in body_to_joints.get(link_key, []):
            joint_key = str(joint_prim.GetPath())
            if joint_key not in visited_joints:
                ApplyJointAPI(joint_prim)
                ordered_joints.append(joint_prim)
                visited_joints.add(joint_key)

            other_index = 1 - body_index
            other_path = GetJointBodyRelationship(joint_prim, other_index)
            if not other_path:
                continue
            other_key = str(other_path)
            if other_key in visited_links:
                continue
            other_prim = stage.GetPrimAtPath(other_path)
            if other_prim:
                queue.append(other_prim)

    # Add deferred sites at the end if sites_last mode
    if detect_sites and sites_last:
        ordered_links.extend(deferred_sites)

    # Build the robotLinks relationship using prepend list
    # Reverse the list since FrontOfPrependList adds to front, which would reverse order
    robot_links_rel = robot_prim.CreateRelationship(Relations.ROBOT_LINKS.name, custom=True)
    robot_links_rel.ClearTargets(removeSpec=False)
    for prim in reversed(ordered_links):
        if prim.HasAPI(Classes.LINK_API.value) or prim.HasAPI(Classes.SITE_API.value):
            robot_links_rel.AddTarget(prim.GetPath(), position=pxr.Usd.ListPositionFrontOfPrependList)

    # Build the robotJoints relationship using prepend list
    robot_joints_rel = robot_prim.CreateRelationship(Relations.ROBOT_JOINTS.name, custom=True)
    robot_joints_rel.ClearTargets(removeSpec=False)
    for prim in reversed(ordered_joints):
        if prim.HasAPI(Classes.JOINT_API.value):
            robot_joints_rel.AddTarget(prim.GetPath(), position=pxr.Usd.ListPositionFrontOfPrependList)

    return root_link, root_joint


def UpdateDeprecatedSchemas(robot_prim: pxr.Usd.Prim):
    """Update deprecated schemas under a robot prim to their replacements.

    Traverses all prims under ``robot_prim``, replaces ``IsaacReferencePointAPI``
    with ``IsaacSiteAPI`` in the apiSchemas metadata, and migrates deprecated
    per-axis DoFOffset attributes on joints that have ``IsaacJointAPI``.

    Args:
        robot_prim: The robot prim whose subtree to scan and update.
    """
    if not robot_prim or not robot_prim.IsValid():
        return

    stage = robot_prim.GetStage()
    edit_layer = stage.GetEditTarget().GetLayer()
    deprecated_schema = Classes.REFERENCE_POINT_API.value
    replacement_schema = Classes.SITE_API.value
    joint_schema = Classes.JOINT_API.value

    for prim in pxr.Usd.PrimRange(robot_prim):
        prim_spec = edit_layer.GetPrimAtPath(prim.GetPath())
        if not prim_spec:
            continue

        api_schemas = prim_spec.GetInfo("apiSchemas")
        if not api_schemas:
            continue

        prepend_items = list(api_schemas.prependedItems) if api_schemas.prependedItems else []
        append_items = list(api_schemas.appendedItems) if api_schemas.appendedItems else []
        explicit_items = list(api_schemas.explicitItems) if api_schemas.explicitItems else []

        updated = False

        # Check and replace deprecated schema in prepended items
        if deprecated_schema in prepend_items:
            prepend_items.remove(deprecated_schema)
            if replacement_schema not in prepend_items:
                prepend_items.append(replacement_schema)
            updated = True

        # Check and replace deprecated schema in appended items
        if deprecated_schema in append_items:
            append_items.remove(deprecated_schema)
            if replacement_schema not in append_items:
                append_items.append(replacement_schema)
            updated = True

        # Check and replace deprecated schema in explicit items
        if deprecated_schema in explicit_items:
            explicit_items.remove(deprecated_schema)
            if replacement_schema not in explicit_items:
                explicit_items.append(replacement_schema)
            updated = True

        if updated:
            new_api_schemas = pxr.Sdf.TokenListOp()
            if prepend_items:
                new_api_schemas.prependedItems = prepend_items
            if append_items:
                new_api_schemas.appendedItems = append_items
            if explicit_items:
                new_api_schemas.explicitItems = explicit_items
            prim_spec.SetInfo("apiSchemas", new_api_schemas)

        # Check for joint API in spec lists and update dof order
        has_joint_api = joint_schema in prepend_items or joint_schema in append_items or joint_schema in explicit_items
        if has_joint_api:
            UpdateDeprecatedJointDofOrder(prim)


def DetectAndApplySites(
    stage: pxr.Usd.Stage,
    robot_prim: pxr.Usd.Prim,
    *,
    sites_last: bool = False,
) -> tuple[list[pxr.Usd.Prim], dict[str, list[pxr.Usd.Prim]]]:
    """Detect child Xforms with no children under Link prims and apply SiteAPI.

    For each prim with LinkAPI, scans for child Xforms that have no children
    and applies the SiteAPI to them.

    Args:
        stage: The USD stage containing the robot.
        robot_prim: The USD prim representing the robot.
        sites_last: If False (default), sites should be added after their parent link.
            If True, all sites should be added at the end of the links list.

    Returns:
        Tuple of (all_sites, sites_by_parent_path). ``all_sites`` is a list of all
        site prims in order of appearance. ``sites_by_parent_path`` is a dict mapping
        parent link path strings to lists of their site prims.
    """
    if not stage or not robot_prim or not robot_prim.IsValid():
        return [], {}

    from . import ApplySiteAPI

    all_sites: list[pxr.Usd.Prim] = []
    sites_by_parent: dict[str, list[pxr.Usd.Prim]] = {}

    for prim in pxr.Usd.PrimRange(robot_prim):
        if not prim.HasAPI(Classes.LINK_API.value):
            continue

        parent_path = str(prim.GetPath())
        link_sites = _detect_sites_for_link(prim)

        for site in link_sites:
            ApplySiteAPI(site)
            all_sites.append(site)
            sites_by_parent.setdefault(parent_path, []).append(site)

    return all_sites, sites_by_parent


def AddSitesToRobotLinks(
    robot_prim: pxr.Usd.Prim,
    sites: list[pxr.Usd.Prim] | None = None,
    sites_by_parent: dict[str, list[pxr.Usd.Prim]] | None = None,
    *,
    sites_last: bool = False,
):
    """Add sites to the robot's robotLinks relationship.

    Args:
        robot_prim: The USD prim representing the robot with RobotAPI.
        sites: List of all site prims (used when sites_last=True).
        sites_by_parent: Dict mapping parent link path to list of its sites
            (used when sites_last=False to insert sites after their parent).
        sites_last: If False (default), sites are inserted after their parent link.
            If True, all sites are appended at the end.
    """
    if not robot_prim or not robot_prim.IsValid():
        return
    if not robot_prim.HasAPI(Classes.ROBOT_API.value):
        return

    robot_links_rel = robot_prim.GetRelationship(Relations.ROBOT_LINKS.name)
    if not robot_links_rel:
        robot_links_rel = robot_prim.CreateRelationship(Relations.ROBOT_LINKS.name, custom=True)

    # Get current targets
    current_targets = list(robot_links_rel.GetTargets())

    if sites_last:
        # Append all sites at the end
        if sites:
            for site in sites:
                site_path = site.GetPath()
                if site_path not in current_targets and site.HasAPI(Classes.SITE_API.value):
                    current_targets.append(site_path)
    else:
        # Insert sites after their parent links
        if sites_by_parent:
            new_targets: list[pxr.Sdf.Path] = []
            for target in current_targets:
                new_targets.append(target)
                target_str = str(target)
                if target_str in sites_by_parent:
                    for site in sites_by_parent[target_str]:
                        if site.HasAPI(Classes.SITE_API.value):
                            new_targets.append(site.GetPath())
            current_targets = new_targets

    # Rebuild the relationship using prepend list
    # Reverse the list since FrontOfPrependList adds to front, which would reverse order
    robot_links_rel.ClearTargets(removeSpec=False)
    for target in reversed(current_targets):
        robot_links_rel.AddTarget(target, position=pxr.Usd.ListPositionFrontOfPrependList)


def ValidateRobotSchemaRelationships(
    robot_prim: pxr.Usd.Prim,
) -> tuple[list[pxr.Sdf.Path], list[pxr.Sdf.Path], list[pxr.Sdf.Path], list[pxr.Sdf.Path]]:
    """Validate existing robot schema relationships and identify invalid entries.

    Args:
        robot_prim: The robot prim with RobotAPI.

    Returns:
        Tuple of (valid_links, invalid_links, valid_joints, invalid_joints).
    """
    valid_links: list[pxr.Sdf.Path] = []
    invalid_links: list[pxr.Sdf.Path] = []
    valid_joints: list[pxr.Sdf.Path] = []
    invalid_joints: list[pxr.Sdf.Path] = []

    if not robot_prim or not robot_prim.IsValid():
        return valid_links, invalid_links, valid_joints, invalid_joints

    stage = robot_prim.GetStage()

    links_rel = robot_prim.GetRelationship(Relations.ROBOT_LINKS.name)
    if links_rel:
        for target in links_rel.GetTargets():
            prim = stage.GetPrimAtPath(target)
            if prim and prim.IsValid():
                valid_links.append(target)
            else:
                invalid_links.append(target)

    joints_rel = robot_prim.GetRelationship(Relations.ROBOT_JOINTS.name)
    if joints_rel:
        for target in joints_rel.GetTargets():
            prim = stage.GetPrimAtPath(target)
            if prim and prim.IsValid():
                valid_joints.append(target)
            else:
                invalid_joints.append(target)

    return valid_links, invalid_links, valid_joints, invalid_joints


def RebuildRelationshipAsPrepend(
    prim: pxr.Usd.Prim,
    rel_name: str,
    targets: list[pxr.Sdf.Path],
):
    """Rebuild a relationship using prepend list operations.

    Args:
        prim: The prim containing the relationship.
        rel_name: Name of the relationship.
        targets: List of target paths to set.
    """
    if not prim or not prim.IsValid():
        return

    rel = prim.GetRelationship(rel_name)
    if not rel:
        rel = prim.CreateRelationship(rel_name, custom=True)

    # Clear existing targets and set new ones via prepend
    # Reverse the list since FrontOfPrependList adds to front, which would reverse order
    rel.ClearTargets(removeSpec=False)
    for target in reversed(targets):
        rel.AddTarget(target, position=pxr.Usd.ListPositionFrontOfPrependList)


def EnsurePrependListForRobotRelationships(robot_prim: pxr.Usd.Prim):
    """Ensure that robot links and joints relationships use prepend list.

    Args:
        robot_prim: The robot prim with RobotAPI.
    """
    if not robot_prim or not robot_prim.IsValid():
        return

    for rel_name in [Relations.ROBOT_LINKS.name, Relations.ROBOT_JOINTS.name]:
        rel = robot_prim.GetRelationship(rel_name)
        if not rel:
            continue

        targets = rel.GetTargets()
        if not targets:
            continue

        # Rebuild using prepend list to ensure proper layering behavior
        RebuildRelationshipAsPrepend(robot_prim, rel_name, list(targets))


def RecalculateRobotSchema(
    stage: pxr.Usd.Stage,
    robot_prim: pxr.Usd.Prim,
    articulation_prim: pxr.Usd.Prim | None = None,
    *,
    detect_sites: bool = False,
    sites_last: bool = False,
) -> tuple[pxr.Usd.Prim | None, pxr.Usd.Prim | None]:
    """Recalculate robot schema relationships while preserving existing order.

    Unlike PopulateRobotSchemaFromArticulation which rebuilds lists from scratch,
    this function preserves the order of existing valid items and appends new
    items at the end. Invalid items are removed.

    Args:
        stage: Stage containing the robot articulation.
        robot_prim: Prim that already has the RobotAPI applied.
        articulation_prim: Prim that owns the PhysicsArticulationRootAPI.
        detect_sites: If True, detect and apply SiteAPI to child Xforms with no
            children under each link.
        sites_last: If False, new sites are added at the end after all existing
            items. If True, same behavior (sites at end).

    Returns:
        The detected root link prim and root joint prim.

    Raises:
        ValueError: If the stage or robot prim is invalid.

    Example:

    .. code-block:: python

        >>> stage = omni.usd.get_context().get_stage()
        >>> robot = stage.GetPrimAtPath("/World/MyRobot")
        >>> root_link, root_joint = RecalculateRobotSchema(stage, robot, robot)
    """
    if not stage:
        raise ValueError("Stage is invalid.")
    if not robot_prim:
        raise ValueError("Robot prim is invalid.")

    # Get existing relationships to preserve order
    existing_link_paths: list[pxr.Sdf.Path] = []
    existing_joint_paths: list[pxr.Sdf.Path] = []

    links_rel = robot_prim.GetRelationship(Relations.ROBOT_LINKS.name)
    if links_rel:
        existing_link_paths = list(links_rel.GetTargets())

    joints_rel = robot_prim.GetRelationship(Relations.ROBOT_JOINTS.name)
    if joints_rel:
        existing_joint_paths = list(joints_rel.GetTargets())

    articulation_root = _find_articulation_root(articulation_prim or robot_prim)
    if articulation_root is None:
        return None, None

    root_link = articulation_root
    articulation_joints = [prim for prim in pxr.Usd.PrimRange(robot_prim) if prim.IsA(pxr.UsdPhysics.Joint)]

    root_joint = None
    if articulation_root.IsA(pxr.UsdPhysics.Joint):
        root_joint = articulation_root
        candidate_path = GetJointBodyRelationship(root_joint, 0) or GetJointBodyRelationship(root_joint, 1)
        if candidate_path:
            root_link = stage.GetPrimAtPath(candidate_path)
    if not root_link:
        return None, root_joint
    root_link_key = str(root_link.GetPath())
    body_to_joints: dict[str, list[tuple[pxr.Usd.Prim, int]]] = {}
    for joint_prim in articulation_joints:
        for body_index in (0, 1):
            body_path = GetJointBodyRelationship(joint_prim, body_index)
            key = _path_key(body_path)
            if not key:
                continue
            body_to_joints.setdefault(key, []).append((joint_prim, body_index))
            if not root_joint and root_link_key and key == root_link_key:
                root_joint = joint_prim

    # Discover all current links and joints via BFS (same as PopulateRobotSchemaFromArticulation)
    queue: deque[pxr.Usd.Prim] = deque()
    visited_links: set[str] = set()
    visited_joints: set[str] = set()
    discovered_links: list[pxr.Usd.Prim] = []
    discovered_joints: list[pxr.Usd.Prim] = []
    discovered_sites: list[pxr.Usd.Prim] = []

    if root_joint:
        discovered_joints.append(root_joint)
        visited_joints.add(str(root_joint.GetPath()))
        ApplyJointAPI(root_joint)

    if root_link:
        queue.append(root_link)

    while queue:
        link_prim = queue.popleft()
        if not link_prim:
            continue
        link_key = str(link_prim.GetPath())
        if link_key in visited_links:
            continue
        visited_links.add(link_key)
        ApplyLinkAPI(link_prim)
        discovered_links.append(link_prim)

        # Detect and apply sites for this link
        if detect_sites:
            link_sites = _detect_sites_for_link(link_prim)
            for site in link_sites:
                ApplySiteAPI(site)
                discovered_sites.append(site)

        for joint_prim, body_index in body_to_joints.get(link_key, []):
            joint_key = str(joint_prim.GetPath())
            if joint_key not in visited_joints:
                ApplyJointAPI(joint_prim)
                discovered_joints.append(joint_prim)
                visited_joints.add(joint_key)

            other_index = 1 - body_index
            other_path = GetJointBodyRelationship(joint_prim, other_index)
            if not other_path:
                continue
            other_key = str(other_path)
            if other_key in visited_links:
                continue
            other_prim = stage.GetPrimAtPath(other_path)
            if other_prim:
                queue.append(other_prim)

    # Build sets of discovered paths for quick lookup
    discovered_link_paths = {str(p.GetPath()) for p in discovered_links}
    discovered_joint_paths = {str(p.GetPath()) for p in discovered_joints}
    discovered_site_paths = {str(p.GetPath()) for p in discovered_sites}

    # Build a set of all valid link paths (links + sites)
    all_valid_link_paths = discovered_link_paths | discovered_site_paths

    # Also include sites that already have SiteAPI in the valid set
    for prim in pxr.Usd.PrimRange(robot_prim):
        if prim.HasAPI(Classes.SITE_API.value) or prim.HasAPI(Classes.REFERENCE_POINT_API.value):
            all_valid_link_paths.add(str(prim.GetPath()))

    # Build final ordered list:
    # 1. Existing valid items (links AND sites) in their EXACT original order
    # 2. New links (discovered but not in existing) in discovery order
    # 3. New sites (discovered but not in existing) in discovery order
    #
    # This preserves interwoven links and sites in their original positions.

    final_links: list[pxr.Sdf.Path] = []
    existing_link_strs: set[str] = set()

    # 1. Add existing valid items in their exact original order
    for path in existing_link_paths:
        path_str = str(path)
        if path_str in all_valid_link_paths:
            final_links.append(path)
            existing_link_strs.add(path_str)

    # 2. Append new links (discovered but not in existing)
    for link_prim in discovered_links:
        path_str = str(link_prim.GetPath())
        if path_str not in existing_link_strs:
            final_links.append(link_prim.GetPath())
            existing_link_strs.add(path_str)

    # 3. Append new sites (discovered but not in existing)
    for site_prim in discovered_sites:
        path_str = str(site_prim.GetPath())
        if path_str not in existing_link_strs:
            final_links.append(site_prim.GetPath())
            existing_link_strs.add(path_str)

    # Process joints
    # The root_joint is special - if it's new, it should be prepended (first) since it's the root
    # Other new joints should be appended at the end
    final_joints: list[pxr.Sdf.Path] = []
    existing_joint_strs: set[str] = set()

    # Build set of existing joint paths for lookup
    existing_joint_path_strs = {str(p) for p in existing_joint_paths}

    # Check if root_joint is new and should be prepended
    root_joint_path_str = str(root_joint.GetPath()) if root_joint else None
    if root_joint and root_joint_path_str not in existing_joint_path_strs:
        final_joints.append(root_joint.GetPath())
        existing_joint_strs.add(root_joint_path_str)

    # Add existing valid joints in their original order
    for path in existing_joint_paths:
        path_str = str(path)
        if path_str in discovered_joint_paths:
            final_joints.append(path)
            existing_joint_strs.add(path_str)

    # Append other new joints (discovered but not in existing, excluding root_joint already added)
    for joint_prim in discovered_joints:
        path_str = str(joint_prim.GetPath())
        if path_str not in existing_joint_strs:
            final_joints.append(joint_prim.GetPath())
            existing_joint_strs.add(path_str)

    # Rebuild relationships with preserved order
    RebuildRelationshipAsPrepend(robot_prim, Relations.ROBOT_LINKS.name, final_links)
    RebuildRelationshipAsPrepend(robot_prim, Relations.ROBOT_JOINTS.name, final_joints)

    return root_link, root_joint


# ---------------------------------------------------------------------------
# Kinematic-tree and zero-config utilities (used by FK / IK chain building)
# ---------------------------------------------------------------------------


def _find_tree_node(root: Any, target_path: str) -> Any:
    """Depth-first search for a RobotLinkNode whose prim path matches target_path.

    Args:
        root: Root of the robot link tree.
        target_path: Prim path to find.

    Returns:
        The matching node, or None if not found.
    """
    if str(root.path) == target_path:
        return root
    for child in root.children:
        hit = _find_tree_node(child, target_path)
        if hit is not None:
            return hit
    return None


def _ancestors(node: Any) -> list[Any]:
    """Return [node, parent, grandparent, ..., root].

    Args:
        node: Tree node to start from.

    Returns:
        List of nodes from node to root.
    """
    chain: list[Any] = []
    cur = node
    while cur is not None:
        chain.append(cur)
        cur = cur.parent
    return chain


def _collect_chain_joints(start_node: Any, end_node: Any) -> list[tuple[Any, bool]]:
    """Return [(joint_prim, is_forward), ...] along the unique tree path.

    is_forward is True when the joint is traversed parent-to-child
    (natural FK direction) and False for child-to-parent.

    Args:
        start_node: Start tree node.
        end_node: End tree node.

    Returns:
        List of (joint_prim, is_forward) along the path.

    Raises:
        ValueError: When start and end nodes are not connected in the tree.
    """
    start_anc = _ancestors(start_node)
    end_anc = _ancestors(end_node)
    start_ids = {id(n) for n in start_anc}

    lca = None
    for n in end_anc:
        if id(n) in start_ids:
            lca = n
            break
    if lca is None:
        raise ValueError("start and end nodes are not connected in the kinematic tree")

    # start → LCA (child-to-parent, backward)
    backward: list[tuple] = []
    cur = start_node
    while cur is not lca:
        if cur._joint_to_parent is not None:
            backward.append((cur._joint_to_parent, False))
        cur = cur.parent

    # LCA → end (parent-to-child, forward)
    forward: list[tuple] = []
    cur = end_node
    while cur is not lca:
        if cur._joint_to_parent is not None:
            forward.append((cur._joint_to_parent, True))
        cur = cur.parent
    forward.reverse()

    return backward + forward


def _compute_zero_config_poses(tree_root: Any) -> dict[str, Any]:
    """Compute body world transforms at joint-zero configuration.

    Propagates from the tree root using only static joint local frames
    (``localPos0/localRot0/localPos1/localRot1``), ignoring any current
    joint state.  The root body's current world pose is the seed (it is
    not affected by any joints).

    The FK formula in USD row-vector convention is::

        body1_zero = local1^-1 * local0 * body0_zero

    Args:
        tree_root: Root node of the robot link tree (from GenerateRobotLinkTree).

    Returns:
        Dict mapping prim-path string to pxr.Gf.Matrix4d (zero-config world transform).
    """
    import omni.usd

    zero_world: dict[str, Any] = {}

    root_world = pxr.Gf.Matrix4d(omni.usd.get_world_transform_matrix(tree_root.prim))
    zero_world[str(tree_root.prim.GetPath())] = root_world

    stack = [tree_root]
    while stack:
        node = stack.pop()
        parent_zero = zero_world[str(node.prim.GetPath())]
        for child in node.children:
            child_path = str(child.prim.GetPath())
            joint_prim = child._joint_to_parent

            if joint_prim is not None:
                joint_usd = pxr.UsdPhysics.Joint(joint_prim)
                local0 = _get_joint_local_transform(joint_usd, 0)
                local1 = _get_joint_local_transform(joint_usd, 1)

                if local0 is not None and local1 is not None:
                    child_zero = pxr.Gf.Matrix4d(local1).GetInverse() * pxr.Gf.Matrix4d(local0) * parent_zero
                    zero_world[child_path] = child_zero
                else:
                    zero_world[child_path] = parent_zero
            else:
                zero_world[child_path] = parent_zero

            stack.append(child)

    return zero_world
