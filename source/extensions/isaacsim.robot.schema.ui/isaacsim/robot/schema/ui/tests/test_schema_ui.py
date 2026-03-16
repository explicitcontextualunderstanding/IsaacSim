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
"""Tests for the robot schema UI extension."""

from unittest import mock

import carb
import carb.settings
import omni.kit.app
import omni.kit.test
import omni.kit.ui_test as ui_test
import omni.ui as ui
import omni.usd
from isaacsim.robot.schema.ui import masking_state as ms
from isaacsim.robot.schema.ui import utils as ui_utils
from omni.ui.tests.test_base import OmniUiTest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
from usd.schema.isaac import robot_schema


class TestSchemaUI(OmniUiTest):
    """UI tests covering Robot Inspector visualization."""

    async def setUp(self) -> None:
        """Set up a fresh USD stage for each test."""
        await omni.usd.get_context().new_stage_async()
        await omni.kit.app.get_app().next_update_async()
        self._stage = omni.usd.get_context().get_stage()

    async def tearDown(self) -> None:
        """Wait for stage loads to finish and clean up."""
        while omni.usd.get_context().get_stage_loading_status()[2] > 0:
            carb.log_info("tearDown, assets still loading, waiting to finish...")
            await omni.kit.app.get_app().next_update_async()
        await omni.kit.app.get_app().next_update_async()

    def _build_simple_robot(self) -> tuple[Usd.Prim, Usd.Prim, Usd.Prim]:
        """Build a minimal articulated robot for hierarchy tests.

        Returns:
            Tuple of (robot_prim, link_prim, joint_prim).
        """
        robot_prim = UsdGeom.Xform.Define(self._stage, "/World/Robot").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(robot_prim)
        UsdPhysics.ArticulationRootAPI.Apply(robot_prim)

        link_prim = UsdGeom.Xform.Define(self._stage, "/World/Robot/Link1").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(link_prim)

        joint = UsdPhysics.Joint.Define(self._stage, "/World/Robot/joint1")
        joint.CreateBody0Rel().SetTargets([robot_prim.GetPath()])
        joint.CreateBody1Rel().SetTargets([link_prim.GetPath()])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.1, 0.0, 0.0))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

        robot_schema.ApplyRobotAPI(robot_prim)
        return robot_prim, link_prim, joint.GetPrim()

    async def test_robot_hierarchy_creation(self) -> None:
        """Generate hierarchy stage and verify mappings."""
        robot_prim, link_prim, joint_prim = self._build_simple_robot()
        await omni.kit.app.get_app().next_update_async()

        hierarchy_stage, path_map, joint_connections = ui_utils.generate_robot_hierarchy_stage()

        self.assertIsNotNone(hierarchy_stage)
        self.assertIsNotNone(path_map.get_hierarchy_path(robot_prim.GetPath()))
        self.assertIsNotNone(path_map.get_hierarchy_path(link_prim.GetPath()))
        self.assertEqual(len(joint_connections), 1)
        self.assertEqual(joint_connections[0].joint_prim_path, joint_prim.GetPath())


class TestSchemaUiUtils(omni.kit.test.AsyncTestCase):
    """Unit tests for robot schema UI utilities."""

    async def test_path_map_round_trip(self) -> None:
        """Verify PathMap insert, lookup, and clear round-trip."""
        path_map = ui_utils.PathMap()
        original = Sdf.Path("/World/Robot")
        hierarchy = Sdf.Path("/Hierarchy/Robot")
        path_map.insert(original, hierarchy)

        self.assertEqual(path_map.get_hierarchy_path(original), hierarchy)
        self.assertEqual(path_map.get_original_path(hierarchy), original)

        path_map.clear()
        self.assertIsNone(path_map.get_hierarchy_path(original))

    async def test_vector_helpers(self) -> None:
        """Test to_vec3d and to_float_list conversion helpers."""
        self.assertIsNone(ui_utils.to_vec3d(None))
        self.assertEqual(ui_utils.to_vec3d((1, 2, 3)), Gf.Vec3d(1, 2, 3))
        self.assertEqual(ui_utils.to_float_list(Gf.Vec3d(1, 2, 3)), [1.0, 2.0, 3.0])

    async def test_world_to_screen_position_fallback(self) -> None:
        """Fall back to identity when no active viewport is available."""
        with mock.patch("isaacsim.robot.schema.ui.utils.get_active_viewport", return_value=None):
            self.assertEqual(ui_utils.world_to_screen_position((1, 2, 3)), (1.0, 2.0, 3.0))

    async def test_joint_has_both_bodies(self) -> None:
        """Verify joint body detection with zero and two targets."""
        stage = Usd.Stage.CreateInMemory()
        joint = UsdPhysics.RevoluteJoint.Define(stage, "/Joint")
        joint_prim = joint.GetPrim()
        self.assertFalse(ui_utils.joint_has_both_bodies(joint_prim))

        joint.CreateBody0Rel().SetTargets(["/Body0"])
        joint.CreateBody1Rel().SetTargets(["/Body1"])
        self.assertTrue(ui_utils.joint_has_both_bodies(joint_prim))

    async def test_is_in_front_of_camera(self) -> None:
        """Test front-of-camera check with forward and backward points."""
        camera_position = Gf.Vec3d(0.0, 0.0, 0.0)
        camera_forward = Gf.Vec3d(0.0, 0.0, 1.0)
        self.assertTrue(ui_utils.is_in_front_of_camera((0.0, 0.0, 1.0), camera_position, camera_forward))
        self.assertFalse(ui_utils.is_in_front_of_camera((0.0, 0.0, -1.0), camera_position, camera_forward))


class TestSchemaUiUtilsExtended(omni.kit.test.AsyncTestCase):
    """Additional unit tests for viewport and position utilities."""

    async def test_world_to_screen_position_with_viewport(self) -> None:
        """Return (x, y) when the viewport transform succeeds."""
        mock_vp = mock.Mock()
        mock_vp.world_to_ndc.Transform.return_value = (0.5, 0.5, 0.5)
        mock_vp.map_ndc_to_texture.return_value = ((100.0, 200.0), mock.Mock())
        result = ui_utils.world_to_screen_position((1.0, 2.0, 3.0), viewport_api=mock_vp)
        self.assertEqual(result, (100.0, 200.0))

    async def test_world_to_screen_position_viewport_raises(self) -> None:
        """Fall back to world coords when the viewport transform raises."""
        mock_vp = mock.Mock()
        mock_vp.world_to_ndc.Transform.side_effect = RuntimeError("viewport error")
        result = ui_utils.world_to_screen_position((1.0, 2.0, 3.0), viewport_api=mock_vp)
        self.assertEqual(result, (1.0, 2.0, 3.0))

    async def test_world_to_screen_position_none_input(self) -> None:
        """Return None for a None position."""
        self.assertIsNone(ui_utils.world_to_screen_position(None))

    async def test_is_position_in_viewport_no_viewport(self) -> None:
        """Return False when no active viewport is available."""
        with mock.patch("isaacsim.robot.schema.ui.utils.get_active_viewport", return_value=None):
            self.assertFalse(ui_utils.is_position_in_viewport((1.0, 2.0, 3.0)))

    async def test_is_position_in_viewport_visible(self) -> None:
        """Return True when map_ndc_to_texture reports an in-viewport position."""
        mock_vp = mock.Mock()
        mock_vp.world_to_ndc.Transform.return_value = (0.5, 0.5, 0.5)
        mock_vp.map_ndc_to_texture.return_value = ((100.0, 200.0), mock.Mock())
        self.assertTrue(ui_utils.is_position_in_viewport((1.0, 2.0, 3.0), viewport_api=mock_vp))

    async def test_is_position_in_viewport_not_visible(self) -> None:
        """Return False when map_ndc_to_texture returns a None viewport."""
        mock_vp = mock.Mock()
        mock_vp.world_to_ndc.Transform.return_value = (2.0, 2.0, 0.5)
        mock_vp.map_ndc_to_texture.return_value = (None, None)
        self.assertFalse(ui_utils.is_position_in_viewport((100.0, 100.0, 100.0), viewport_api=mock_vp))


class TestStagePrimUtils(omni.kit.test.AsyncTestCase):
    """Tests for get_stage_safe and get_prim_safe."""

    async def setUp(self) -> None:
        """Set up a fresh USD stage for each test."""
        await omni.usd.get_context().new_stage_async()
        await omni.kit.app.get_app().next_update_async()
        self._stage = omni.usd.get_context().get_stage()

    async def test_get_stage_safe_returns_stage(self) -> None:
        """Return the current stage for the default context."""
        stage = ui_utils.get_stage_safe()
        self.assertIsNotNone(stage)

    async def test_get_stage_safe_no_context_returns_none(self) -> None:
        """Return None when the context itself is None."""
        with mock.patch("isaacsim.robot.schema.ui.utils.omni.usd.get_context", return_value=None):
            self.assertIsNone(ui_utils.get_stage_safe("ctx"))

    async def test_get_prim_safe_valid_string_path(self) -> None:
        """Return prim for a valid string path."""
        UsdGeom.Xform.Define(self._stage, "/World/TestPrim")
        prim = ui_utils.get_prim_safe("/World/TestPrim")
        self.assertIsNotNone(prim)
        self.assertEqual(str(prim.GetPath()), "/World/TestPrim")

    async def test_get_prim_safe_valid_sdf_path(self) -> None:
        """Return prim for a valid Sdf.Path."""
        UsdGeom.Xform.Define(self._stage, "/World/SdfPrim")
        prim = ui_utils.get_prim_safe(Sdf.Path("/World/SdfPrim"))
        self.assertIsNotNone(prim)

    async def test_get_prim_safe_nonexistent_path(self) -> None:
        """Return None for a path with no prim."""
        self.assertIsNone(ui_utils.get_prim_safe("/World/NonExistent"))

    async def test_get_prim_safe_none_path(self) -> None:
        """Return None when path argument is None."""
        self.assertIsNone(ui_utils.get_prim_safe(None))


class TestLinkJointPosition(omni.kit.test.AsyncTestCase):
    """Tests for get_link_position and get_joint_position."""

    async def setUp(self) -> None:
        """Set up a fresh USD stage for each test."""
        await omni.usd.get_context().new_stage_async()
        await omni.kit.app.get_app().next_update_async()
        self._stage = omni.usd.get_context().get_stage()

    async def test_get_link_position_none(self) -> None:
        """Return None for a None link argument."""
        self.assertIsNone(ui_utils.get_link_position(None))

    async def test_get_link_position_invalid_string_path(self) -> None:
        """Return None for a path with no prim."""
        self.assertIsNone(ui_utils.get_link_position("/World/NonExistent"))

    async def test_get_link_position_valid_prim(self) -> None:
        """Return a Vec3d for a valid prim passed directly."""
        prim = UsdGeom.Xform.Define(self._stage, "/World/Link1").GetPrim()
        position = ui_utils.get_link_position(prim)
        self.assertIsNotNone(position)
        self.assertIsInstance(position, Gf.Vec3d)

    async def test_get_link_position_valid_string_path(self) -> None:
        """Return a Vec3d when passing a path string for an existing prim."""
        UsdGeom.Xform.Define(self._stage, "/World/Link2")
        position = ui_utils.get_link_position("/World/Link2")
        self.assertIsNotNone(position)
        self.assertIsInstance(position, Gf.Vec3d)

    async def test_get_joint_position_invalid_paths(self) -> None:
        """Return None when robot or joint prim does not exist."""
        self.assertIsNone(ui_utils.get_joint_position("/World/NoRobot", "/World/NoJoint"))

    async def test_get_joint_position_valid(self) -> None:
        """Return a Vec3d for a fully configured robot joint."""
        robot_prim = UsdGeom.Xform.Define(self._stage, "/World/Robot").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(robot_prim)
        UsdPhysics.ArticulationRootAPI.Apply(robot_prim)
        link_prim = UsdGeom.Xform.Define(self._stage, "/World/Robot/Link1").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(link_prim)
        joint = UsdPhysics.Joint.Define(self._stage, "/World/Robot/joint1")
        joint.CreateBody0Rel().SetTargets([robot_prim.GetPath()])
        joint.CreateBody1Rel().SetTargets([link_prim.GetPath()])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(1.0, 0.0, 0.0))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        robot_schema.ApplyRobotAPI(robot_prim)
        await omni.kit.app.get_app().next_update_async()
        position = ui_utils.get_joint_position("/World/Robot", "/World/Robot/joint1")
        self.assertIsNotNone(position)
        self.assertIsInstance(position, Gf.Vec3d)


class TestHierarchyModes(OmniUiTest):
    """Tests for generate_robot_hierarchy_stage with each HierarchyMode."""

    async def setUp(self) -> None:
        """Set up a fresh USD stage for each test."""
        await omni.usd.get_context().new_stage_async()
        await omni.kit.app.get_app().next_update_async()
        self._stage = omni.usd.get_context().get_stage()

    async def tearDown(self) -> None:
        """Wait for pending asset loads before tearing down."""
        while omni.usd.get_context().get_stage_loading_status()[2] > 0:
            await omni.kit.app.get_app().next_update_async()
        await omni.kit.app.get_app().next_update_async()

    def _build_simple_robot(self):
        """Build a minimal articulated robot for hierarchy mode tests.

        Returns:
            Tuple of (robot_prim, link_prim, joint_prim).
        """
        robot_prim = UsdGeom.Xform.Define(self._stage, "/World/Robot").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(robot_prim)
        UsdPhysics.ArticulationRootAPI.Apply(robot_prim)
        link_prim = UsdGeom.Xform.Define(self._stage, "/World/Robot/Link1").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(link_prim)
        joint = UsdPhysics.Joint.Define(self._stage, "/World/Robot/joint1")
        joint.CreateBody0Rel().SetTargets([robot_prim.GetPath()])
        joint.CreateBody1Rel().SetTargets([link_prim.GetPath()])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.1, 0.0, 0.0))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        robot_schema.ApplyRobotAPI(robot_prim)
        return robot_prim, link_prim, joint.GetPrim()

    async def test_no_robot_returns_none_stage(self) -> None:
        """Empty stage yields (None, empty_map, [])."""
        hierarchy_stage, path_map, joint_connections = ui_utils.generate_robot_hierarchy_stage()
        self.assertIsNone(hierarchy_stage)
        self.assertEqual(len(joint_connections), 0)

    async def test_flat_hierarchy_mode(self) -> None:
        """FLAT mode creates Links and Joints scopes under the robot root."""
        robot_prim, link_prim, joint_prim = self._build_simple_robot()
        await omni.kit.app.get_app().next_update_async()

        hierarchy_stage, path_map, joint_connections = ui_utils.generate_robot_hierarchy_stage(
            ui_utils.HierarchyMode.FLAT
        )

        self.assertIsNotNone(hierarchy_stage)
        # In FLAT mode the hierarchy root keeps the same path as the original robot root,
        # so we can look up the scope prims directly without going through the path_map.
        # (The path_map entry for the robot root is overwritten by _generate_flat_hierarchy
        # when the root prim also appears in the robotLinks relationship as a link.)
        robot_hier_root = robot_prim.GetPath()
        links_scope = hierarchy_stage.GetPrimAtPath(robot_hier_root.AppendChild("Links"))
        joints_scope = hierarchy_stage.GetPrimAtPath(robot_hier_root.AppendChild("Joints"))
        self.assertTrue(links_scope.IsValid())
        self.assertTrue(joints_scope.IsValid())

        # Verify the link and joint prims were placed inside the scopes.
        hier_link_path = path_map.get_hierarchy_path(link_prim.GetPath())
        hier_joint_path = path_map.get_hierarchy_path(joint_prim.GetPath())
        self.assertIsNotNone(hier_link_path)
        self.assertIsNotNone(hier_joint_path)
        # Both paths must be children of their respective scope.
        self.assertTrue(hier_link_path.HasPrefix(links_scope.GetPath()))
        self.assertTrue(hier_joint_path.HasPrefix(joints_scope.GetPath()))
        # The prims themselves must exist in the hierarchy stage.
        self.assertTrue(hierarchy_stage.GetPrimAtPath(hier_link_path).IsValid())
        self.assertTrue(hierarchy_stage.GetPrimAtPath(hier_joint_path).IsValid())
        self.assertEqual(len(joint_connections), 1)

    async def test_linked_hierarchy_mode(self) -> None:
        """LINKED mode nests each link under its parent joint in the hierarchy."""
        robot_prim, link_prim, joint_prim = self._build_simple_robot()
        await omni.kit.app.get_app().next_update_async()

        hierarchy_stage, path_map, joint_connections = ui_utils.generate_robot_hierarchy_stage(
            ui_utils.HierarchyMode.LINKED
        )

        self.assertIsNotNone(hierarchy_stage)
        hier_robot_path = path_map.get_hierarchy_path(robot_prim.GetPath())
        hier_joint_path = path_map.get_hierarchy_path(joint_prim.GetPath())
        hier_link_path = path_map.get_hierarchy_path(link_prim.GetPath())
        self.assertIsNotNone(hier_robot_path)
        self.assertIsNotNone(hier_joint_path)
        self.assertIsNotNone(hier_link_path)
        # LINKED layout: robot_root → joint → Link1
        # joint is placed as a direct child of the robot root link.
        self.assertTrue(hier_joint_path.HasPrefix(hier_robot_path))
        # Link1 is placed as a child of the joint.
        self.assertTrue(hier_link_path.HasPrefix(hier_joint_path))
        # All three prims must exist in the hierarchy stage.
        self.assertTrue(hierarchy_stage.GetPrimAtPath(hier_robot_path).IsValid())
        self.assertTrue(hierarchy_stage.GetPrimAtPath(hier_joint_path).IsValid())
        self.assertTrue(hierarchy_stage.GetPrimAtPath(hier_link_path).IsValid())
        self.assertEqual(len(joint_connections), 1)

    async def test_mujoco_hierarchy_mode(self) -> None:
        """MUJOCO mode nests links under their parent link; joint is the last child of its link."""
        robot_prim, link_prim, joint_prim = self._build_simple_robot()
        await omni.kit.app.get_app().next_update_async()

        hierarchy_stage, path_map, joint_connections = ui_utils.generate_robot_hierarchy_stage(
            ui_utils.HierarchyMode.MUJOCO
        )

        self.assertIsNotNone(hierarchy_stage)
        hier_robot_path = path_map.get_hierarchy_path(robot_prim.GetPath())
        hier_link_path = path_map.get_hierarchy_path(link_prim.GetPath())
        hier_joint_path = path_map.get_hierarchy_path(joint_prim.GetPath())
        self.assertIsNotNone(hier_robot_path)
        self.assertIsNotNone(hier_link_path)
        self.assertIsNotNone(hier_joint_path)
        # MUJOCO layout: robot_root → Link1 → joint1
        # Link1 is a direct child of the robot root link.
        self.assertTrue(hier_link_path.HasPrefix(hier_robot_path))
        # joint1 is appended as the last child of Link1 (the link it connects to as body1).
        self.assertTrue(hier_joint_path.HasPrefix(hier_link_path))
        # All three prims must exist in the hierarchy stage.
        self.assertTrue(hierarchy_stage.GetPrimAtPath(hier_robot_path).IsValid())
        self.assertTrue(hierarchy_stage.GetPrimAtPath(hier_link_path).IsValid())
        self.assertTrue(hierarchy_stage.GetPrimAtPath(hier_joint_path).IsValid())
        self.assertEqual(len(joint_connections), 1)


class TestMaskingStateFunctions(omni.kit.test.AsyncTestCase):
    """Tests for masking_state module-level prim classifier functions."""

    async def setUp(self) -> None:
        """Set up a fresh USD stage for each test."""
        await omni.usd.get_context().new_stage_async()
        await omni.kit.app.get_app().next_update_async()
        self._stage = omni.usd.get_context().get_stage()

    def _prim_with_schemas(self, path: str, schemas: list) -> Usd.Prim:
        """Create an Xform prim and apply the given schema names.

        Args:
            path: USD prim path.
            schemas: List of applied schema name strings.

        Returns:
            The created prim with schemas applied.
        """
        prim = UsdGeom.Xform.Define(self._stage, path).GetPrim()
        for schema in schemas:
            prim.AddAppliedSchema(schema)
        return prim

    async def test_is_maskable_type_joint(self) -> None:
        """Prim with IsaacJointAPI is maskable."""
        prim = self._prim_with_schemas("/World/j1", ["IsaacJointAPI"])
        self.assertTrue(ms.is_maskable_type(prim))

    async def test_is_maskable_type_link(self) -> None:
        """Prim with IsaacLinkAPI is maskable."""
        prim = self._prim_with_schemas("/World/l1", ["IsaacLinkAPI"])
        self.assertTrue(ms.is_maskable_type(prim))

    async def test_is_maskable_type_plain_xform(self) -> None:
        """Plain Xform without robot schemas is not maskable."""
        prim = UsdGeom.Xform.Define(self._stage, "/World/Other").GetPrim()
        self.assertFalse(ms.is_maskable_type(prim))

    async def test_is_joint_type(self) -> None:
        """is_joint_type is True only for prims with IsaacJointAPI."""
        joint_prim = self._prim_with_schemas("/World/j2", ["IsaacJointAPI"])
        link_prim = self._prim_with_schemas("/World/l2", ["IsaacLinkAPI"])
        self.assertTrue(ms.is_joint_type(joint_prim))
        self.assertFalse(ms.is_joint_type(link_prim))

    async def test_is_link_type(self) -> None:
        """is_link_type is True only for prims with IsaacLinkAPI."""
        link_prim = self._prim_with_schemas("/World/l3", ["IsaacLinkAPI"])
        joint_prim = self._prim_with_schemas("/World/j3", ["IsaacJointAPI"])
        self.assertTrue(ms.is_link_type(link_prim))
        self.assertFalse(ms.is_link_type(joint_prim))

    async def test_is_anchorable_link_with_rigid_body(self) -> None:
        """Link with RigidBodyAPI is anchorable."""
        prim = self._prim_with_schemas("/World/al1", ["IsaacLinkAPI"])
        UsdPhysics.RigidBodyAPI.Apply(prim)
        self.assertTrue(ms.is_anchorable_link(prim))

    async def test_is_anchorable_link_without_rigid_body(self) -> None:
        """Link without RigidBodyAPI is not anchorable."""
        prim = self._prim_with_schemas("/World/al2", ["IsaacLinkAPI"])
        self.assertFalse(ms.is_anchorable_link(prim))


class TestMaskingState(omni.kit.test.AsyncTestCase):
    """Tests for MaskingState singleton state tracking."""

    async def setUp(self) -> None:
        """Reset the MaskingState singleton before each test."""
        ms.MaskingState._instance = None

    async def test_singleton_identity(self) -> None:
        """get_instance always returns the same object."""
        a = ms.MaskingState.get_instance()
        b = ms.MaskingState.get_instance()
        self.assertIs(a, b)

    async def test_toggle_deactivated(self) -> None:
        """Toggle deactivated state on and off."""
        state = ms.MaskingState.get_instance()
        path = "/World/Robot/Link1"
        self.assertFalse(state.is_deactivated(path))
        self.assertTrue(state.toggle_deactivated(path))
        self.assertTrue(state.is_deactivated(path))
        self.assertFalse(state.toggle_deactivated(path))
        self.assertFalse(state.is_deactivated(path))

    async def test_toggle_bypassed(self) -> None:
        """Bypassed path is also deactivated; toggling off clears both."""
        state = ms.MaskingState.get_instance()
        path = "/World/Robot/Link1"
        self.assertTrue(state.toggle_bypassed(path))
        self.assertTrue(state.is_bypassed(path))
        self.assertTrue(state.is_deactivated(path))
        self.assertFalse(state.toggle_bypassed(path))
        self.assertFalse(state.is_bypassed(path))
        self.assertFalse(state.is_deactivated(path))

    async def test_toggle_anchored(self) -> None:
        """Toggle anchor state on and off."""
        state = ms.MaskingState.get_instance()
        path = "/World/Robot/Link1"
        self.assertTrue(state.toggle_anchored(path))
        self.assertTrue(state.is_anchored(path))
        self.assertFalse(state.toggle_anchored(path))
        self.assertFalse(state.is_anchored(path))

    async def test_set_deactivated(self) -> None:
        """set_deactivated activates and deactivates a single path."""
        state = ms.MaskingState.get_instance()
        path = "/World/Robot/joint1"
        state.set_deactivated(path, True)
        self.assertTrue(state.is_deactivated(path))
        state.set_deactivated(path, False)
        self.assertFalse(state.is_deactivated(path))

    async def test_set_deactivated_batch(self) -> None:
        """set_deactivated_batch operates on multiple paths atomically."""
        state = ms.MaskingState.get_instance()
        paths = ["/World/L1", "/World/L2", "/World/j1"]
        state.set_deactivated_batch(paths, True)
        for p in paths:
            self.assertTrue(state.is_deactivated(p))
        state.set_deactivated_batch(paths, False)
        for p in paths:
            self.assertFalse(state.is_deactivated(p))

    async def test_set_bypassed(self) -> None:
        """set_bypassed marks a path as both bypassed and deactivated."""
        state = ms.MaskingState.get_instance()
        path = "/World/Robot/Link1"
        state.set_bypassed(path, True)
        self.assertTrue(state.is_bypassed(path))
        self.assertTrue(state.is_deactivated(path))
        state.set_bypassed(path, False)
        self.assertFalse(state.is_bypassed(path))
        self.assertFalse(state.is_deactivated(path))

    async def test_set_bypassed_batch(self) -> None:
        """set_bypassed_batch operates on multiple paths atomically."""
        state = ms.MaskingState.get_instance()
        paths = ["/World/L1", "/World/L2"]
        state.set_bypassed_batch(paths, True)
        for p in paths:
            self.assertTrue(state.is_bypassed(p))
        state.set_bypassed_batch(paths, False)
        for p in paths:
            self.assertFalse(state.is_bypassed(p))

    async def test_set_anchored(self) -> None:
        """set_anchored marks and unmarks a single path."""
        state = ms.MaskingState.get_instance()
        path = "/World/Robot/Link1"
        state.set_anchored(path, True)
        self.assertTrue(state.is_anchored(path))
        state.set_anchored(path, False)
        self.assertFalse(state.is_anchored(path))

    async def test_set_anchored_batch(self) -> None:
        """set_anchored_batch operates on multiple paths atomically."""
        state = ms.MaskingState.get_instance()
        paths = ["/World/L1", "/World/L2"]
        state.set_anchored_batch(paths, True)
        for p in paths:
            self.assertTrue(state.is_anchored(p))
        state.set_anchored_batch(paths, False)
        for p in paths:
            self.assertFalse(state.is_anchored(p))

    async def test_get_deactivated_paths_returns_copy(self) -> None:
        """get_deactivated_paths returns a mutable copy, not the internal set."""
        state = ms.MaskingState.get_instance()
        state.set_deactivated("/World/L1", True)
        copy = state.get_deactivated_paths()
        self.assertIn("/World/L1", copy)
        copy.add("/World/Phantom")
        self.assertNotIn("/World/Phantom", state.get_deactivated_paths())

    async def test_get_bypassed_paths_returns_copy(self) -> None:
        """get_bypassed_paths returns a mutable copy."""
        state = ms.MaskingState.get_instance()
        state.set_bypassed("/World/L1", True)
        copy = state.get_bypassed_paths()
        self.assertIn("/World/L1", copy)

    async def test_get_anchored_paths_returns_copy(self) -> None:
        """get_anchored_paths returns a mutable copy."""
        state = ms.MaskingState.get_instance()
        state.set_anchored("/World/L1", True)
        copy = state.get_anchored_paths()
        self.assertIn("/World/L1", copy)

    async def test_clear(self) -> None:
        """Clear resets all deactivated, bypassed, and anchored sets."""
        state = ms.MaskingState.get_instance()
        state.set_deactivated("/World/L1", True)
        state.set_bypassed("/World/L2", True)
        state.set_anchored("/World/L3", True)
        state.clear()
        self.assertEqual(len(state.get_deactivated_paths()), 0)
        self.assertEqual(len(state.get_bypassed_paths()), 0)
        self.assertEqual(len(state.get_anchored_paths()), 0)

    async def test_subscribe_and_unsubscribe_changed(self) -> None:
        """Callback fires on state changes; stops after unsubscribe."""
        state = ms.MaskingState.get_instance()
        call_count = [0]

        def cb():
            call_count[0] += 1

        state.subscribe_changed(cb)
        state.toggle_deactivated("/World/L1")
        self.assertEqual(call_count[0], 1)
        state.unsubscribe_changed(cb)
        state.toggle_deactivated("/World/L2")
        self.assertEqual(call_count[0], 1)

    async def test_bypass_controlled_joint_blocks_toggle(self) -> None:
        """A joint locked by an active link bypass cannot be independently toggled."""
        state = ms.MaskingState.get_instance()
        link_path = "/World/Robot/Link1"
        joint_path = "/World/Robot/backward_joint"

        mock_ops = mock.Mock()
        mock_ops.bypass_prim.return_value = (joint_path, "enabled")
        mock_ops.unbypass_prim.return_value = None
        state.operations = mock_ops

        state.toggle_bypassed(link_path)
        self.assertTrue(state.is_bypass_controlled(joint_path))
        self.assertTrue(state.is_deactivated(joint_path))

        # Controlled joint toggle is a no-op
        self.assertFalse(state.toggle_deactivated(joint_path))

        # Unbypassing the link releases the joint
        state.toggle_bypassed(link_path)
        self.assertFalse(state.is_bypass_controlled(joint_path))
        self.assertFalse(state.is_deactivated(joint_path))

    async def test_get_masking_layer_id_no_operations(self) -> None:
        """Returns None when no operations backend is set."""
        state = ms.MaskingState.get_instance()
        self.assertIsNone(state.get_masking_layer_id())

    async def test_get_masking_layer_id_with_operations(self) -> None:
        """Delegates to the operations backend when one is set."""
        state = ms.MaskingState.get_instance()
        mock_ops = mock.Mock()
        mock_ops.get_masking_layer_id.return_value = "layer-abc-123"
        state.operations = mock_ops
        self.assertEqual(state.get_masking_layer_id(), "layer-abc-123")


class TestConnectionItem(OmniUiTest):
    """Tests for ConnectionItem property accessors and is_valid."""

    async def setUp(self) -> None:
        """Set up a fresh USD stage for each test."""
        await omni.usd.get_context().new_stage_async()
        await omni.kit.app.get_app().next_update_async()
        self._stage = omni.usd.get_context().get_stage()

    async def tearDown(self) -> None:
        """Wait for pending asset loads before tearing down."""
        while omni.usd.get_context().get_stage_loading_status()[2] > 0:
            await omni.kit.app.get_app().next_update_async()
        await omni.kit.app.get_app().next_update_async()

    def _make_item(self, with_parent_joint: bool = False):
        """Create a `ConnectionItem` with a simple robot joint on stage.

        Args:
            with_parent_joint: Whether to set the parent joint prim.

        Returns:
            The constructed `ConnectionItem`.
        """
        from isaacsim.robot.schema.ui.model import ConnectionItem

        robot_prim = UsdGeom.Xform.Define(self._stage, "/World/Robot").GetPrim()
        link_prim = UsdGeom.Xform.Define(self._stage, "/World/Robot/Link1").GetPrim()
        joint = UsdPhysics.Joint.Define(self._stage, "/World/Robot/joint1")
        joint.CreateBody0Rel().SetTargets([robot_prim.GetPath()])
        joint.CreateBody1Rel().SetTargets([link_prim.GetPath()])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        parent_joint_prim = joint.GetPrim() if with_parent_joint else None
        return ConnectionItem(
            joint_prim=joint.GetPrim(),
            parent_joint_prim=parent_joint_prim,
            parent_link_prim=robot_prim,
            robot_root_path=robot_prim.GetPath(),
            joint_pos=Gf.Vec3d(1.0, 0.0, 0.0),
            parent_joint_pos=Gf.Vec3d(0.0, 0.0, 0.0),
        )

    async def test_needs_position_refresh_getter_and_setter(self) -> None:
        """needs_position_refresh returns False by default; setter updates it."""
        item = self._make_item()
        self.assertFalse(item.needs_position_refresh)
        item.needs_position_refresh = True
        self.assertTrue(item.needs_position_refresh)

    async def test_joint_prim_path(self) -> None:
        """joint_prim_path returns the path used at construction."""
        item = self._make_item()
        self.assertEqual(item.joint_prim_path, Sdf.Path("/World/Robot/joint1"))

    async def test_robot_root_path(self) -> None:
        """robot_root_path matches the path passed at construction."""
        item = self._make_item()
        self.assertEqual(item.robot_root_path, Sdf.Path("/World/Robot"))

    async def test_parent_joint_path_none(self) -> None:
        """parent_joint_path is None when no parent joint is provided."""
        item = self._make_item(with_parent_joint=False)
        self.assertIsNone(item.parent_joint_path)

    async def test_parent_joint_path_not_none(self) -> None:
        """parent_joint_path is set when a parent joint is provided."""
        item = self._make_item(with_parent_joint=True)
        self.assertIsNotNone(item.parent_joint_path)

    async def test_visible_default(self) -> None:
        """Visible returns True by default."""
        item = self._make_item()
        self.assertTrue(item.visible)

    async def test_overlay_paths_default_empty(self) -> None:
        """overlay_paths is an empty list by default."""
        item = self._make_item()
        self.assertEqual(item.overlay_paths, [])

    async def test_overlay_names_default_empty(self) -> None:
        """overlay_names is an empty list by default."""
        item = self._make_item()
        self.assertEqual(item.overlay_names, [])

    async def test_overlay_prims_setter_with_prim_list(self) -> None:
        """overlay_prims setter accepts a list of prims and stores their paths."""
        item = self._make_item()
        link_prim = self._stage.GetPrimAtPath("/World/Robot/Link1")
        item.overlay_prims = [link_prim]
        prims = item.overlay_prims
        self.assertEqual(len(prims), 1)

    async def test_overlay_names_setter(self) -> None:
        """overlay_names setter stores the provided list."""
        item = self._make_item()
        item.overlay_names = ["jointA", "jointB"]
        self.assertEqual(item.overlay_names, ["jointA", "jointB"])

    async def test_overlay_paths_setter(self) -> None:
        """overlay_paths setter stores the provided path list directly."""
        item = self._make_item()
        paths = [Sdf.Path("/World/Robot/joint1")]
        item.overlay_paths = paths
        self.assertEqual(item.overlay_paths, paths)

    async def test_joint_prim_valid(self) -> None:
        """joint_prim returns a valid prim when the joint exists on stage."""
        item = self._make_item()
        prim = item.joint_prim
        self.assertIsNotNone(prim)
        self.assertTrue(prim.IsValid())

    async def test_robot_root_prim_valid(self) -> None:
        """robot_root_prim returns a valid prim when the robot exists on stage."""
        item = self._make_item()
        prim = item.robot_root_prim
        self.assertIsNotNone(prim)

    async def test_parent_joint_prim_none(self) -> None:
        """parent_joint_prim is None when no parent joint path was stored."""
        item = self._make_item(with_parent_joint=False)
        self.assertIsNone(item.parent_joint_prim)

    async def test_is_valid_true(self) -> None:
        """is_valid returns True for a connection referencing existing prims."""
        item = self._make_item()
        self.assertTrue(item.is_valid())


class TestSelectionWatch(omni.kit.test.AsyncTestCase):
    """Tests for SelectionWatch simple methods that do not require a tree view."""

    async def setUp(self) -> None:
        """Set up a fresh USD stage for each test."""
        await omni.usd.get_context().new_stage_async()
        await omni.kit.app.get_app().next_update_async()

    async def test_create_and_destroy(self) -> None:
        """SelectionWatch creates and destroys without raising."""
        from isaacsim.robot.schema.ui.selection_watch import SelectionWatch

        watch = SelectionWatch()
        watch.destroy()

    async def test_update_path_map(self) -> None:
        """update_path_map accepts PathMap and None without raising."""
        from isaacsim.robot.schema.ui.selection_watch import SelectionWatch

        watch = SelectionWatch()
        pm = ui_utils.PathMap()
        watch.update_path_map(pm)
        watch.update_path_map(None)
        watch.destroy()

    async def test_set_filtering_string_lowercased(self) -> None:
        """set_filtering stores a lowercase version of the input string."""
        from isaacsim.robot.schema.ui.selection_watch import SelectionWatch

        watch = SelectionWatch()
        watch.set_filtering("ARM")
        self.assertEqual(watch._filter_string, "arm")
        watch.set_filtering(None)
        self.assertIsNone(watch._filter_string)
        watch.destroy()

    async def test_enable_filtering_checking(self) -> None:
        """enable_filtering_checking toggles the internal flag."""
        from isaacsim.robot.schema.ui.selection_watch import SelectionWatch

        watch = SelectionWatch()
        watch.enable_filtering_checking(True)
        self.assertTrue(watch._is_filter_checking_enabled)
        watch.enable_filtering_checking(False)
        self.assertFalse(watch._is_filter_checking_enabled)
        watch.destroy()

    async def test_on_stage_selection_changed_event_guard(self) -> None:
        """Guard returns early when _is_setting_usd_selection is True."""
        from isaacsim.robot.schema.ui.selection_watch import SelectionWatch

        watch = SelectionWatch()
        watch._is_setting_usd_selection = True
        with mock.patch.object(watch, "_on_selection_changed") as patched:
            watch._on_stage_selection_changed_event(mock.Mock())
            patched.assert_not_called()
        self.assertTrue(watch._is_setting_usd_selection)
        watch.destroy()

    async def test_on_selection_changed_returns_early_no_map(self) -> None:
        """_on_selection_changed returns early when path_map is None."""
        from isaacsim.robot.schema.ui.selection_watch import SelectionWatch

        watch = SelectionWatch()
        watch._on_selection_changed()
        watch.destroy()

    async def test_resolve_selected_items_empty_without_map(self) -> None:
        """_resolve_selected_items returns an empty set when path_map is None."""
        from isaacsim.robot.schema.ui.selection_watch import SelectionWatch

        watch = SelectionWatch()
        result = watch._resolve_selected_items(["/World/Robot"])
        self.assertEqual(result, set())
        watch.destroy()

    async def test_translate_to_original_paths_empty_without_map(self) -> None:
        """_translate_to_original_paths returns [] when path_map is None."""
        from isaacsim.robot.schema.ui.selection_watch import SelectionWatch

        watch = SelectionWatch()
        result = watch._translate_to_original_paths([])
        self.assertEqual(result, [])
        watch.destroy()

    async def test_sync_from_stage_no_map(self) -> None:
        """sync_from_stage returns without raising when no path_map is set."""
        from isaacsim.robot.schema.ui.selection_watch import SelectionWatch

        watch = SelectionWatch()
        watch.sync_from_stage()
        watch.destroy()


class TestMaskingLayer(omni.kit.test.AsyncTestCase):
    """Tests for MaskingLayer acquire, release, get, and is_empty."""

    async def setUp(self) -> None:
        """Set up a fresh USD stage for each test."""
        await omni.usd.get_context().new_stage_async()
        await omni.kit.app.get_app().next_update_async()

    async def test_initial_get_returns_none(self) -> None:
        """get() returns None before any layer has been acquired."""
        from isaacsim.robot.schema.ui.masking_ops import MaskingLayer

        ml = MaskingLayer()
        self.assertIsNone(ml.get())

    async def test_initial_is_empty_true(self) -> None:
        """is_empty() returns True before acquire."""
        from isaacsim.robot.schema.ui.masking_ops import MaskingLayer

        ml = MaskingLayer()
        self.assertTrue(ml.is_empty())

    async def test_acquire_creates_layer(self) -> None:
        """acquire() creates and returns a valid Sdf.Layer."""
        from isaacsim.robot.schema.ui.masking_ops import MaskingLayer

        ml = MaskingLayer()
        layer = ml.acquire()
        self.assertIsNotNone(layer)
        self.assertIsNotNone(ml.get())
        self.assertTrue(ml.is_empty())

    async def test_acquire_returns_same_layer_on_repeat(self) -> None:
        """acquire() returns the same layer on repeated calls."""
        from isaacsim.robot.schema.ui.masking_ops import MaskingLayer

        ml = MaskingLayer()
        first = ml.acquire()
        second = ml.acquire()
        self.assertIs(first, second)

    async def test_release_clears_layer(self) -> None:
        """release() removes the layer and clears the internal reference."""
        from isaacsim.robot.schema.ui.masking_ops import MaskingLayer

        ml = MaskingLayer()
        ml.acquire()
        ml.release()
        self.assertIsNone(ml.get())

    async def test_destroy_is_equivalent_to_release(self) -> None:
        """destroy() removes the layer, same as release()."""
        from isaacsim.robot.schema.ui.masking_ops import MaskingLayer

        ml = MaskingLayer()
        ml.acquire()
        ml.destroy()
        self.assertIsNone(ml.get())


class TestMaskingOperationsHelpers(omni.kit.test.AsyncTestCase):
    """Tests for MaskingOperations static helpers and simple public methods."""

    async def setUp(self) -> None:
        """Set up a fresh USD stage for each test."""
        await omni.usd.get_context().new_stage_async()
        await omni.kit.app.get_app().next_update_async()

    async def test_build_local_frame_identity(self) -> None:
        """_build_local_frame returns a valid matrix for None inputs."""
        from isaacsim.robot.schema.ui.masking_ops import MaskingOperations

        frame = MaskingOperations._build_local_frame(None, None)
        self.assertAlmostEqual(frame[3][3], 1.0)

    async def test_build_local_frame_with_values(self) -> None:
        """_build_local_frame applies translation and identity rotation."""
        from isaacsim.robot.schema.ui.masking_ops import MaskingOperations

        pos = Gf.Vec3f(1.0, 2.0, 3.0)
        rot = Gf.Quatf(1.0, 0.0, 0.0, 0.0)
        frame = MaskingOperations._build_local_frame(pos, rot)
        translation = frame.ExtractTranslation()
        self.assertAlmostEqual(translation[0], 1.0)
        self.assertAlmostEqual(translation[1], 2.0)
        self.assertAlmostEqual(translation[2], 3.0)

    async def test_decompose_local_frame_round_trip(self) -> None:
        """_decompose_local_frame round-trips through _build_local_frame."""
        from isaacsim.robot.schema.ui.masking_ops import MaskingOperations

        pos = Gf.Vec3f(1.0, 2.0, 3.0)
        rot = Gf.Quatf(1.0, 0.0, 0.0, 0.0)
        frame = MaskingOperations._build_local_frame(pos, rot)
        out_pos, out_rot = MaskingOperations._decompose_local_frame(frame)
        self.assertAlmostEqual(out_pos[0], pos[0], places=5)
        self.assertAlmostEqual(out_pos[1], pos[1], places=5)

    async def test_get_masking_layer_id_none_before_acquire(self) -> None:
        """get_masking_layer_id returns None when no masking layer has been acquired."""
        from isaacsim.robot.schema.ui.masking_ops import MaskingOperations

        ops = MaskingOperations()
        self.assertIsNone(ops.get_masking_layer_id())

    async def test_clear_all_safe_without_layer(self) -> None:
        """clear_all does not raise when no masking layer exists."""
        from isaacsim.robot.schema.ui.masking_ops import MaskingOperations

        ops = MaskingOperations()
        ops.clear_all()


class TestMaskingOperationsMask(omni.kit.test.AsyncTestCase):
    """Tests for mask_prim and unmask_prim on a simple articulated robot."""

    async def setUp(self) -> None:
        """Set up a fresh USD stage for each test."""
        await omni.usd.get_context().new_stage_async()
        await omni.kit.app.get_app().next_update_async()
        self._stage = omni.usd.get_context().get_stage()

    def _build_robot(self) -> tuple[Usd.Prim, Usd.Prim, Usd.Prim]:
        """Build a minimal articulated robot for masking operation tests.

        Returns:
            Tuple of (robot_prim, link_prim, joint_prim).
        """
        from usd.schema.isaac import robot_schema as rs

        robot_prim = UsdGeom.Xform.Define(self._stage, "/World/Robot").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(robot_prim)
        UsdPhysics.ArticulationRootAPI.Apply(robot_prim)
        link_prim = UsdGeom.Xform.Define(self._stage, "/World/Robot/Link1").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(link_prim)
        joint = UsdPhysics.Joint.Define(self._stage, "/World/Robot/joint1")
        joint.CreateBody0Rel().SetTargets([robot_prim.GetPath()])
        joint.CreateBody1Rel().SetTargets([link_prim.GetPath()])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        rs.ApplyRobotAPI(robot_prim)
        return robot_prim, link_prim, joint.GetPrim()

    async def test_mask_prim_invalid_path_returns_false(self) -> None:
        """mask_prim returns False for a non-existent path."""
        from isaacsim.robot.schema.ui.masking_ops import MaskingOperations

        ops = MaskingOperations()
        self.assertFalse(ops.mask_prim("/World/DoesNotExist"))

    async def test_mask_prim_joint_marks_masked(self) -> None:
        """mask_prim returns True and sets MASKED_KEY on the joint."""
        from isaacsim.robot.schema.ui.masking_ops import MASKED_KEY, MaskingOperations

        robot_prim, link_prim, joint_prim = self._build_robot()
        await omni.kit.app.get_app().next_update_async()

        ops = MaskingOperations()
        result = ops.mask_prim(str(joint_prim.GetPath()))
        self.assertTrue(result)
        self.assertTrue(joint_prim.GetCustomDataByKey(MASKED_KEY))

    async def test_unmask_prim_joint(self) -> None:
        """unmask_prim returns True after a joint was masked."""
        from isaacsim.robot.schema.ui.masking_ops import MaskingOperations

        robot_prim, link_prim, joint_prim = self._build_robot()
        await omni.kit.app.get_app().next_update_async()

        ops = MaskingOperations()
        ops.mask_prim(str(joint_prim.GetPath()))
        result = ops.unmask_prim(str(joint_prim.GetPath()))
        self.assertTrue(result)

    async def test_unmask_prim_no_layer_returns_true(self) -> None:
        """unmask_prim returns True when no masking layer exists (no-op)."""
        from isaacsim.robot.schema.ui.masking_ops import MaskingOperations

        robot_prim, link_prim, joint_prim = self._build_robot()
        ops = MaskingOperations()
        result = ops.unmask_prim(str(joint_prim.GetPath()))
        self.assertTrue(result)

    async def test_mask_prim_link_marks_masked(self) -> None:
        """mask_prim returns True and sets MASKED_KEY on the link."""
        from isaacsim.robot.schema.ui.masking_ops import MASKED_KEY, MaskingOperations

        robot_prim, link_prim, joint_prim = self._build_robot()
        await omni.kit.app.get_app().next_update_async()

        ops = MaskingOperations()
        result = ops.mask_prim(str(link_prim.GetPath()))
        self.assertTrue(result)
        self.assertTrue(link_prim.GetCustomDataByKey(MASKED_KEY))

    async def test_get_masking_layer_id_after_mask(self) -> None:
        """get_masking_layer_id returns a non-None string after masking a prim."""
        from isaacsim.robot.schema.ui.masking_ops import MaskingOperations

        robot_prim, link_prim, joint_prim = self._build_robot()
        await omni.kit.app.get_app().next_update_async()

        ops = MaskingOperations()
        ops.mask_prim(str(joint_prim.GetPath()))
        layer_id = ops.get_masking_layer_id()
        self.assertIsNotNone(layer_id)

    async def test_clear_all_removes_layer(self) -> None:
        """clear_all removes the masking layer after masking."""
        from isaacsim.robot.schema.ui.masking_ops import MaskingOperations

        robot_prim, link_prim, joint_prim = self._build_robot()
        await omni.kit.app.get_app().next_update_async()

        ops = MaskingOperations()
        ops.mask_prim(str(joint_prim.GetPath()))
        ops.clear_all()
        self.assertIsNone(ops.get_masking_layer_id())


class TestRobotInspectorUI(OmniUiTest):
    """UI tests verifying Robot Inspector widget identifiers and click interactions."""

    WINDOW_NAME = "Robot Inspector"
    SETTING_KEY = "/persistent/exts/isaacsim.robot.schema.ui/hierarchyMode"

    async def setUp(self) -> None:
        """Create a fresh stage and ensure the Robot Inspector window is visible."""
        await omni.usd.get_context().new_stage_async()
        await omni.kit.app.get_app().next_update_async()
        self._stage = omni.usd.get_context().get_stage()
        ui.Workspace.show_window(self.WINDOW_NAME)
        for _ in range(5):
            await omni.kit.app.get_app().next_update_async()

    async def tearDown(self) -> None:
        """Wait for pending asset loads."""
        while omni.usd.get_context().get_stage_loading_status()[2] > 0:
            await omni.kit.app.get_app().next_update_async()
        await omni.kit.app.get_app().next_update_async()

    def _build_simple_robot(self) -> tuple[Usd.Prim, Usd.Prim, Usd.Prim]:
        """Build a minimal robot for inspector tests.

        Returns:
            Tuple of (robot_prim, link_prim, joint_prim).
        """
        robot_prim = UsdGeom.Xform.Define(self._stage, "/World/Robot").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(robot_prim)
        UsdPhysics.ArticulationRootAPI.Apply(robot_prim)
        link_prim = UsdGeom.Xform.Define(self._stage, "/World/Robot/Link1").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(link_prim)
        joint = UsdPhysics.Joint.Define(self._stage, "/World/Robot/joint1")
        joint.CreateBody0Rel().SetTargets([robot_prim.GetPath()])
        joint.CreateBody1Rel().SetTargets([link_prim.GetPath()])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.1, 0.0, 0.0))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        robot_schema.ApplyRobotAPI(robot_prim)
        return robot_prim, link_prim, joint.GetPrim()

    async def test_view_mode_toolbar_identifiers(self) -> None:
        """Toolbar label and mode frames are discoverable by identifier."""
        self._build_simple_robot()
        await omni.kit.app.get_app().next_update_async()

        label = ui_test.find(f"{self.WINDOW_NAME}//Frame/**/Label[*].identifier=='robot_inspector_view_label'")
        self.assertIsNotNone(label, "View label not found by identifier")

        for mode_id in (
            "robot_inspector_mode_flat",
            "robot_inspector_mode_tree",
            "robot_inspector_mode_mujoco",
        ):
            frame = ui_test.find(f"{self.WINDOW_NAME}//Frame/**/Frame[*].identifier=='{mode_id}'")
            self.assertIsNotNone(frame, f"Frame '{mode_id}' not found")

    async def test_view_mode_label_identifiers(self) -> None:
        """Mode option labels (Flat, Tree, MuJoCo) are findable and show correct text."""
        self._build_simple_robot()
        await omni.kit.app.get_app().next_update_async()

        expected = {
            "robot_inspector_mode_flat_label": "Flat",
            "robot_inspector_mode_tree_label": "Tree",
            "robot_inspector_mode_mujoco_label": "MuJoCo",
        }
        for label_id, expected_text in expected.items():
            label = ui_test.find(f"{self.WINDOW_NAME}//Frame/**/Label[*].identifier=='{label_id}'")
            self.assertIsNotNone(label, f"Label '{label_id}' not found")
            self.assertEqual(label.widget.text, expected_text)

    async def test_view_mode_switching_via_click(self) -> None:
        """Clicking each mode frame updates the persisted hierarchy mode setting."""
        self._build_simple_robot()
        await omni.kit.app.get_app().next_update_async()

        settings = carb.settings.get_settings()
        mode_map = {
            "robot_inspector_mode_flat": "FLAT",
            "robot_inspector_mode_tree": "LINKED",
            "robot_inspector_mode_mujoco": "MUJOCO",
        }
        for mode_id, expected_mode in mode_map.items():
            frame = ui_test.find(f"{self.WINDOW_NAME}//Frame/**/Frame[*].identifier=='{mode_id}'")
            self.assertIsNotNone(frame, f"Frame '{mode_id}' not found")
            await frame.click()
            await ui_test.human_delay()
            stored = settings.get(self.SETTING_KEY)
            self.assertEqual(
                stored,
                expected_mode,
                f"After clicking {mode_id}: expected {expected_mode}, got {stored}",
            )
