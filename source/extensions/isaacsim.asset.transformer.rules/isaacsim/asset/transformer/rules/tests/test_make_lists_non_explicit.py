# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for MakeListsNonExplicitRule."""

import os
import shutil
import tempfile

import omni.kit.test
from isaacsim.asset.transformer.rules.isaac_sim.make_lists_non_explicit import MakeListsNonExplicitRule
from isaacsim.asset.transformer.rules.utils import compile_patterns, matches_any_pattern
from pxr import Sdf, Usd

from .common import _TEST_ADVANCED_USD


def _matches_any(name: str, patterns: list[str]) -> bool:
    """Check if a name fully matches any regex pattern.

    Args:
        name: Name to check.
        patterns: List of regex pattern strings.

    Returns:
        True if name matches any pattern, False otherwise.
    """
    return matches_any_pattern(name, compile_patterns(patterns))


class TestMakeListsNonExplicitRule(omni.kit.test.AsyncTestCase):
    """Async tests for MakeListsNonExplicitRule."""

    async def setUp(self):
        """Create a temporary directory for test output."""
        self._tmpdir = tempfile.mkdtemp()
        self._success = False

    async def tearDown(self):
        """Remove temporary directory after successful tests."""
        if self._success:
            shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _copy_test_asset(self) -> str:
        """Copy the test asset to a writable temporary path.

        Returns:
            Absolute path to the copied test asset.
        """
        temp_asset = os.path.join(self._tmpdir, "test_advanced.usda")
        shutil.copy(_TEST_ADVANCED_USD, temp_asset)
        return temp_asset

    def _find_explicit_metadata_list_op(
        self,
        stage: Usd.Stage,
        metadata_patterns: list[str],
    ) -> tuple[str, str, list[object]] | None:
        """Find a prim with explicit metadata list op matching patterns.

        Args:
            stage: Stage to inspect.
            metadata_patterns: Metadata name patterns to match.

        Returns:
            Tuple of (prim_path, metadata_key, explicit_items) or None if not found.
        """
        layer = stage.GetRootLayer()
        for prim in Usd.PrimRange(stage.GetPseudoRoot()):
            prim_spec = layer.GetPrimAtPath(prim.GetPath())
            if not prim_spec:
                continue
            for key in prim_spec.ListInfoKeys():
                if not _matches_any(key, metadata_patterns):
                    continue
                list_op = prim_spec.GetInfo(key)
                if not hasattr(list_op, "isExplicit") or not list_op.isExplicit:
                    continue
                explicit_items = list(list_op.explicitItems or [])
                if explicit_items:
                    return prim_spec.path.pathString, key, explicit_items
        return None

    def _find_explicit_relationship_list_op(
        self,
        stage: Usd.Stage,
        property_patterns: list[str],
    ) -> tuple[str, str, list[Sdf.Path]] | None:
        """Find a prim relationship with explicit target list op matching patterns.

        Args:
            stage: Stage to inspect.
            property_patterns: Property name patterns to match.

        Returns:
            Tuple of (prim_path, property_name, explicit_items) or None if not found.
        """
        layer = stage.GetRootLayer()
        for prim in Usd.PrimRange(stage.GetPseudoRoot()):
            prim_spec = layer.GetPrimAtPath(prim.GetPath())
            if not prim_spec:
                continue
            for prop_name, prop_spec in prim_spec.properties.items():
                if not _matches_any(prop_name, property_patterns):
                    continue
                if not isinstance(prop_spec, Sdf.RelationshipSpec):
                    continue
                list_op = prop_spec.targetPathList
                if not hasattr(list_op, "isExplicit") or not list_op.isExplicit:
                    continue
                explicit_items = list(list_op.explicitItems or [])
                if explicit_items:
                    return prim_spec.path.pathString, prop_name, explicit_items
        return None

    async def test_get_configuration_parameters(self):
        """Verify configuration parameters are exposed."""
        stage = Usd.Stage.Open(_TEST_ADVANCED_USD)
        rule = MakeListsNonExplicitRule(
            source_stage=stage,
            package_root=self._tmpdir,
            destination_path="",
            args={},
        )

        params = rule.get_configuration_parameters()
        param_names = [p.name for p in params]
        self.assertIn("metadata_names", param_names)
        self.assertIn("property_names", param_names)
        self.assertIn("list_op_type", param_names)
        self._success = True

    async def test_process_rule_converts_explicit_lists_prepend(self):
        """Verify explicit list ops are converted to prepended list ops."""
        temp_asset = self._copy_test_asset()
        stage = Usd.Stage.Open(temp_asset)

        metadata_patterns = ["apiSchemas"]
        property_patterns = ["material:binding"]

        metadata_match = self._find_explicit_metadata_list_op(stage, metadata_patterns)
        rel_match = self._find_explicit_relationship_list_op(stage, property_patterns)
        self.assertIsNotNone(metadata_match)
        self.assertIsNotNone(rel_match)

        rule = MakeListsNonExplicitRule(
            source_stage=stage,
            package_root=self._tmpdir,
            destination_path="",
            args={
                "params": {
                    "metadata_names": metadata_patterns,
                    "property_names": property_patterns,
                    "list_op_type": "prepend",
                }
            },
        )

        rule.process_rule()

        layer = Sdf.Layer.FindOrOpen(temp_asset)
        self.assertIsNotNone(layer)

        prim_path, metadata_key, metadata_items = metadata_match
        prim_spec = layer.GetPrimAtPath(prim_path)
        self.assertIsNotNone(prim_spec)
        updated_metadata = prim_spec.GetInfo(metadata_key)
        self.assertFalse(updated_metadata.isExplicit)
        self.assertEqual(list(updated_metadata.prependedItems), metadata_items)

        rel_prim_path, rel_name, rel_items = rel_match
        rel_prim_spec = layer.GetPrimAtPath(rel_prim_path)
        self.assertIsNotNone(rel_prim_spec)
        rel_spec = rel_prim_spec.relationships.get(rel_name)
        self.assertIsNotNone(rel_spec)
        rel_list_op = rel_spec.targetPathList
        self.assertFalse(rel_list_op.isExplicit)
        self.assertEqual(list(rel_list_op.prependedItems), rel_items)

        self._success = True

    async def test_process_rule_converts_explicit_lists_append(self):
        """Verify explicit list ops are converted to appended list ops."""
        temp_asset = self._copy_test_asset()
        stage = Usd.Stage.Open(temp_asset)

        metadata_patterns = ["apiSchemas"]

        metadata_match = self._find_explicit_metadata_list_op(stage, metadata_patterns)
        self.assertIsNotNone(metadata_match)

        rule = MakeListsNonExplicitRule(
            source_stage=stage,
            package_root=self._tmpdir,
            destination_path="",
            args={
                "params": {
                    "metadata_names": metadata_patterns,
                    "property_names": [],
                    "list_op_type": "append",
                }
            },
        )

        rule.process_rule()

        layer = Sdf.Layer.FindOrOpen(temp_asset)
        self.assertIsNotNone(layer)

        prim_path, metadata_key, metadata_items = metadata_match
        prim_spec = layer.GetPrimAtPath(prim_path)
        self.assertIsNotNone(prim_spec)
        updated_metadata = prim_spec.GetInfo(metadata_key)
        self.assertFalse(updated_metadata.isExplicit)
        self.assertEqual(list(updated_metadata.appendedItems), metadata_items)

        self._success = True

    async def test_process_rule_converts_explicit_schemas_prepend(self):
        """Verify explicit apiSchemas are converted to prepended list ops."""
        temp_asset = self._copy_test_asset()
        stage = Usd.Stage.Open(temp_asset)

        metadata_patterns = ["apiSchemas"]
        metadata_match = self._find_explicit_metadata_list_op(stage, metadata_patterns)
        self.assertIsNotNone(metadata_match)

        rule = MakeListsNonExplicitRule(
            source_stage=stage,
            package_root=self._tmpdir,
            destination_path="",
            args={
                "params": {
                    "metadata_names": metadata_patterns,
                    "property_names": [],
                    "list_op_type": "prepend",
                }
            },
        )

        rule.process_rule()

        layer = Sdf.Layer.FindOrOpen(temp_asset)
        self.assertIsNotNone(layer)

        prim_path, metadata_key, metadata_items = metadata_match
        prim_spec = layer.GetPrimAtPath(prim_path)
        self.assertIsNotNone(prim_spec)
        updated_metadata = prim_spec.GetInfo(metadata_key)
        self.assertFalse(updated_metadata.isExplicit)
        self.assertEqual(list(updated_metadata.prependedItems), metadata_items)

        self._success = True
