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

"""Tests for Robot Poser named-pose CRUD, import/export, and validation."""

import json
import os
import tempfile

import numpy as np
import omni.kit.app
import omni.kit.test
import omni.usd
from isaacsim.robot.poser.robot_poser import (
    NAMED_POSES_SCOPE,
    PoseResult,
    _sanitize_name,
    apply_pose_by_name,
    delete_named_pose,
    export_poses,
    get_named_pose,
    import_poses,
    list_named_poses,
    store_named_pose,
    validate_robot_schema,
)
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
from usd.schema.isaac import robot_schema
from usd.schema.isaac.robot_schema import utils as robot_utils


def _create_test_robot(stage: Usd.Stage) -> tuple[Usd.Prim, Usd.Prim, Usd.Prim]:
    """Create a two-link robot with a revolute joint and RobotAPI.

    Shared fixture used by multiple test classes.

    Args:
        stage: USD stage to define prims on.

    Returns:
        Tuple of (robot_prim, link1_prim, joint_prim).
    """
    robot_xform = UsdGeom.Xform.Define(stage, "/World/Robot")
    UsdPhysics.RigidBodyAPI.Apply(robot_xform.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(robot_xform.GetPrim())

    link1 = UsdGeom.Xform.Define(stage, "/World/Robot/Link1")
    UsdPhysics.RigidBodyAPI.Apply(link1.GetPrim())

    joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/Robot/joint1")
    joint.CreateBody0Rel().SetTargets([robot_xform.GetPrim().GetPath()])
    joint.CreateBody1Rel().SetTargets([link1.GetPrim().GetPath()])
    joint.CreateAxisAttr("X")
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(1.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    robot_schema.ApplyRobotAPI(robot_xform.GetPrim())
    return robot_xform.GetPrim(), link1.GetPrim(), joint.GetPrim()


class TestNamedPoseCRUD(omni.kit.test.AsyncTestCase):
    """Async tests for named-pose create, read, update, and delete."""

    async def setUp(self) -> None:
        """Create a fresh USD stage before each test."""
        await omni.usd.get_context().new_stage_async()
        await omni.kit.app.get_app().next_update_async()
        self._stage = omni.usd.get_context().get_stage()

    def _create_robot(self) -> tuple[Usd.Prim, Usd.Prim, Usd.Prim]:
        """Create the test robot and return (robot_prim, link_prim, joint_prim).

        Returns:
            Tuple of (robot_prim, link_prim, joint_prim).
        """
        return _create_test_robot(self._stage)

    def _make_pose_result(self, start: str = "/World/Robot", end: str = "/World/Robot/Link1") -> PoseResult:
        """Return a valid PoseResult with one revolute joint value (in radians).

        Args:
            start: Path of the start link.
            end: Path of the end link.

        Returns:
            A PoseResult with one joint at 45 degrees.
        """
        return PoseResult(
            success=True,
            joints={"/World/Robot/joint1": float(np.radians(45))},
            joint_fixed={"/World/Robot/joint1": False},
            start_link=start,
            end_link=end,
            target_position=[1.0, 0.0, 0.0],
            target_orientation=[1.0, 0.0, 0.0, 0.0],
        )

    # -- validate_robot_schema ---------------------------------------------

    async def test_validate_robot_schema_true(self) -> None:
        """Verify that validate_robot_schema returns True for a prim with RobotAPI."""
        robot_prim, _, _ = self._create_robot()
        self.assertTrue(validate_robot_schema(robot_prim))

    async def test_validate_robot_schema_false(self) -> None:
        """Verify that validate_robot_schema returns False for a prim without RobotAPI."""
        prim = UsdGeom.Xform.Define(self._stage, "/World/Plain").GetPrim()
        self.assertFalse(validate_robot_schema(prim))

    # -- _sanitize_name ----------------------------------------------------

    async def test_sanitize_name(self) -> None:
        """Verify that _sanitize_name replaces spaces and disallowed chars with underscores."""
        self.assertEqual(_sanitize_name("Home Pose"), "Home_Pose")
        self.assertEqual(_sanitize_name("a/b.c"), "a_b_c")

    # -- store_named_pose --------------------------------------------------

    async def test_store_named_pose_creates_prim(self) -> None:
        """Verify that store_named_pose creates an IsaacNamedPose prim under Named_Poses scope."""
        robot_prim, _, _ = self._create_robot()
        result = self._make_pose_result()
        ok = store_named_pose(self._stage, robot_prim, "Home", result)
        self.assertTrue(ok)

        scope = self._stage.GetPrimAtPath(robot_prim.GetPath().AppendChild(NAMED_POSES_SCOPE))
        self.assertTrue(scope.IsValid())

        pose_prim = self._stage.GetPrimAtPath(scope.GetPath().AppendChild("Home"))
        self.assertTrue(pose_prim.IsValid())
        self.assertEqual(pose_prim.GetTypeName(), robot_schema.Classes.NAMED_POSE.value)

    async def test_store_named_pose_sets_attributes(self) -> None:
        """Verify that store_named_pose writes joint values and valid to the pose prim."""
        robot_prim, _, _ = self._create_robot()
        result = self._make_pose_result()
        store_named_pose(self._stage, robot_prim, "Home", result)

        pose_path = robot_prim.GetPath().AppendChild(NAMED_POSES_SCOPE).AppendChild("Home")
        pose_prim = self._stage.GetPrimAtPath(pose_path)

        self.assertTrue(robot_utils.GetNamedPoseValid(pose_prim))
        values = robot_utils.GetNamedPoseJointValues(pose_prim)
        self.assertIsNotNone(values)
        self.assertAlmostEqual(float(values[0]), 45.0, places=4)

    async def test_store_named_pose_sets_relationships(self) -> None:
        """Verify that store_named_pose sets start_link, end_link, and joints relationships."""
        robot_prim, _, _ = self._create_robot()
        result = self._make_pose_result()
        store_named_pose(self._stage, robot_prim, "Home", result)

        pose_path = robot_prim.GetPath().AppendChild(NAMED_POSES_SCOPE).AppendChild("Home")
        pose_prim = self._stage.GetPrimAtPath(pose_path)

        self.assertEqual(
            robot_utils.GetNamedPoseStartLink(pose_prim),
            Sdf.Path("/World/Robot"),
        )
        self.assertEqual(
            robot_utils.GetNamedPoseEndLink(pose_prim),
            Sdf.Path("/World/Robot/Link1"),
        )
        joints = robot_utils.GetNamedPoseJoints(pose_prim)
        self.assertEqual(len(joints), 1)

    async def test_store_named_pose_registers_in_robot_relationship(self) -> None:
        """Verify that store_named_pose adds the pose to the robot's namedPoses relationship."""
        robot_prim, _, _ = self._create_robot()
        result = self._make_pose_result()
        store_named_pose(self._stage, robot_prim, "Home", result)

        poses = robot_utils.GetAllNamedPoses(self._stage, robot_prim)
        self.assertEqual(len(poses), 1)

    async def test_store_named_pose_rejects_failed_result(self) -> None:
        """Verify that store_named_pose returns False when PoseResult.success is False."""
        robot_prim, _, _ = self._create_robot()
        result = PoseResult(success=False)
        ok = store_named_pose(self._stage, robot_prim, "Bad", result)
        self.assertFalse(ok)

    async def test_store_named_pose_xform_ops(self) -> None:
        """Verify that stored pose has translate and orient xform ops matching target."""
        robot_prim, _, _ = self._create_robot()
        result = self._make_pose_result()
        store_named_pose(self._stage, robot_prim, "Home", result)

        pose_path = robot_prim.GetPath().AppendChild(NAMED_POSES_SCOPE).AppendChild("Home")
        pose_prim = self._stage.GetPrimAtPath(pose_path)
        xformable = UsdGeom.Xformable(pose_prim)

        has_translate = False
        has_orient = False
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                v = op.Get()
                self.assertAlmostEqual(float(v[0]), 1.0)
                has_translate = True
            elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                has_orient = True
        self.assertTrue(has_translate)
        self.assertTrue(has_orient)

    # -- get_named_pose ----------------------------------------------------

    async def test_get_named_pose_roundtrip(self) -> None:
        """Verify that get_named_pose returns stored joint values and links after store_named_pose."""
        robot_prim, _, _ = self._create_robot()
        original = self._make_pose_result()
        store_named_pose(self._stage, robot_prim, "Home", original)

        retrieved = get_named_pose(self._stage, robot_prim, "Home")
        self.assertIsNotNone(retrieved)
        self.assertTrue(retrieved.success)
        self.assertEqual(retrieved.start_link, "/World/Robot")
        self.assertEqual(retrieved.end_link, "/World/Robot/Link1")
        self.assertIn("/World/Robot/joint1", retrieved.joints)
        self.assertAlmostEqual(
            retrieved.joints["/World/Robot/joint1"],
            original.joints["/World/Robot/joint1"],
            places=4,
        )

    async def test_get_named_pose_nonexistent(self) -> None:
        """Verify that get_named_pose returns None when the pose name does not exist."""
        robot_prim, _, _ = self._create_robot()
        self.assertIsNone(get_named_pose(self._stage, robot_prim, "DoesNotExist"))

    async def test_get_named_pose_no_scope(self) -> None:
        """Verify that get_named_pose returns None when Named_Poses scope does not exist."""
        robot_prim, _, _ = self._create_robot()
        self.assertIsNone(get_named_pose(self._stage, robot_prim, "Any"))

    # -- list_named_poses --------------------------------------------------

    async def test_list_named_poses(self) -> None:
        """Verify that list_named_poses returns names of all stored poses for the robot."""
        robot_prim, _, _ = self._create_robot()
        store_named_pose(self._stage, robot_prim, "A", self._make_pose_result())
        store_named_pose(self._stage, robot_prim, "B", self._make_pose_result())

        names = list_named_poses(self._stage, robot_prim)
        self.assertIn("A", names)
        self.assertIn("B", names)
        self.assertEqual(len(names), 2)

    async def test_list_named_poses_empty(self) -> None:
        """Verify that list_named_poses returns empty list when no poses are stored."""
        robot_prim, _, _ = self._create_robot()
        self.assertEqual(list_named_poses(self._stage, robot_prim), [])

    # -- delete_named_pose -------------------------------------------------

    async def test_delete_named_pose(self) -> None:
        """Verify that delete_named_pose removes the pose prim and from robot relationship."""
        robot_prim, _, _ = self._create_robot()
        store_named_pose(self._stage, robot_prim, "Del", self._make_pose_result())

        self.assertTrue(delete_named_pose(self._stage, robot_prim, "Del"))
        self.assertEqual(list_named_poses(self._stage, robot_prim), [])
        self.assertIsNone(get_named_pose(self._stage, robot_prim, "Del"))

    async def test_delete_named_pose_nonexistent(self) -> None:
        """Verify that delete_named_pose returns False when the pose name does not exist."""
        robot_prim, _, _ = self._create_robot()
        self.assertFalse(delete_named_pose(self._stage, robot_prim, "Ghost"))

    async def test_delete_named_pose_removes_from_relationship(self) -> None:
        """Verify that delete_named_pose removes only the target from namedPoses relationship."""
        robot_prim, _, _ = self._create_robot()
        store_named_pose(self._stage, robot_prim, "A", self._make_pose_result())
        store_named_pose(self._stage, robot_prim, "B", self._make_pose_result())

        delete_named_pose(self._stage, robot_prim, "A")
        remaining = list_named_poses(self._stage, robot_prim)
        self.assertEqual(remaining, ["B"])


class TestNamedPoseImportExport(omni.kit.test.AsyncTestCase):
    """Tests for export_poses / import_poses JSON round-trip."""

    async def setUp(self) -> None:
        """Create a fresh USD stage before each test."""
        await omni.usd.get_context().new_stage_async()
        await omni.kit.app.get_app().next_update_async()
        self._stage = omni.usd.get_context().get_stage()

    def _create_robot(self) -> Usd.Prim:
        """Create the test robot and return the robot prim.

        Returns:
            The robot root prim.
        """
        robot_prim, _, _ = _create_test_robot(self._stage)
        return robot_prim

    async def test_export_import_roundtrip(self) -> None:
        """Verify that export to JSON and import restores named poses on another robot."""
        robot_prim = self._create_robot()
        pose = PoseResult(
            success=True,
            joints={"/World/Robot/joint1": float(np.radians(30))},
            joint_fixed={"/World/Robot/joint1": True},
            start_link="/World/Robot",
            end_link="/World/Robot/Link1",
            target_position=[0.5, 0.0, 0.0],
            target_orientation=[1.0, 0.0, 0.0, 0.0],
        )
        store_named_pose(self._stage, robot_prim, "Export_Test", pose)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
            path = fh.name

        try:
            ok = export_poses(self._stage, robot_prim, path)
            self.assertTrue(ok)

            with open(path) as fh:
                data = json.load(fh)
            self.assertIn("_meta", data)
            self.assertEqual(data["_meta"]["units"], "radians")
            self.assertIn("note", data["_meta"])
            self.assertIn("Export_Test", data["poses"])
            self.assertTrue(data["poses"]["Export_Test"]["valid"])

            # Import into a fresh robot on the same stage
            robot2 = UsdGeom.Xform.Define(self._stage, "/World/Robot2")
            UsdPhysics.RigidBodyAPI.Apply(robot2.GetPrim())
            UsdPhysics.ArticulationRootAPI.Apply(robot2.GetPrim())
            link2 = UsdGeom.Xform.Define(self._stage, "/World/Robot2/Link1")
            UsdPhysics.RigidBodyAPI.Apply(link2.GetPrim())
            joint2 = UsdPhysics.RevoluteJoint.Define(self._stage, "/World/Robot2/joint1")
            joint2.CreateBody0Rel().SetTargets([robot2.GetPrim().GetPath()])
            joint2.CreateBody1Rel().SetTargets([link2.GetPrim().GetPath()])
            joint2.CreateAxisAttr("X")
            joint2.CreateLocalPos0Attr().Set(Gf.Vec3f(1.0, 0.0, 0.0))
            joint2.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            joint2.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            joint2.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            robot_schema.ApplyRobotAPI(robot2.GetPrim())

            count = import_poses(self._stage, robot2.GetPrim(), path)
            self.assertEqual(count, 1)

            names = list_named_poses(self._stage, robot2.GetPrim())
            self.assertIn("Export_Test", names)
        finally:
            os.unlink(path)

    async def test_export_empty_poses(self) -> None:
        """Verify that export_poses writes empty dict when robot has no named poses."""
        robot_prim = self._create_robot()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
            path = fh.name
        try:
            ok = export_poses(self._stage, robot_prim, path)
            self.assertTrue(ok)
            with open(path) as fh:
                data = json.load(fh)
            self.assertIn("_meta", data)
            self.assertIn("note", data["_meta"])
            self.assertEqual(data["poses"], {})
        finally:
            os.unlink(path)

    async def test_import_multiple_poses(self) -> None:
        """Verify that import_poses loads multiple poses from a JSON file."""
        robot_prim = self._create_robot()
        data = {
            "_meta": {
                "units": "radians",
                "note": "Revolute joint values are stored in radians. USD natively uses degrees.",
            },
            "poses": {
                "Pose_1": {
                    "joints": {"/World/Robot/joint1": 0.5},
                    "joint_fixed": {"/World/Robot/joint1": False},
                    "start_link": "/World/Robot",
                    "end_link": "/World/Robot/Link1",
                    "target_position": [1.0, 0.0, 0.0],
                    "target_orientation": [1.0, 0.0, 0.0, 0.0],
                    "valid": True,
                },
                "Pose_2": {
                    "joints": {"/World/Robot/joint1": 1.0},
                    "joint_fixed": {"/World/Robot/joint1": True},
                    "start_link": "/World/Robot",
                    "end_link": "/World/Robot/Link1",
                    "target_position": [0.5, 0.5, 0.0],
                    "target_orientation": [1.0, 0.0, 0.0, 0.0],
                    "valid": True,
                },
            },
        }
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as fh:
            json.dump(data, fh)
            path = fh.name
        try:
            count = import_poses(self._stage, robot_prim, path)
            self.assertEqual(count, 2)
            names = list_named_poses(self._stage, robot_prim)
            self.assertEqual(sorted(names), ["Pose_1", "Pose_2"])
        finally:
            os.unlink(path)
