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

"""Tests for asset transformer rules utility functions.

Covers the public API of :mod:`isaacsim.asset.transformer.rules.utils`
including prim-name sanitization, path classification helpers, layer and
prim-spec manipulation, stage-level queries, metadata copy, file I/O
utilities, ``TokenListOp`` merging, and asset-path resolution.

Each test method exercises multiple related inputs and collects all failures
into a single summary assertion so that every check runs even when earlier
ones fail.
"""

import os
import shutil
import tempfile

import omni.kit.test
from isaacsim.asset.transformer.rules import utils
from pxr import Sdf, Usd

from .common import _UR10E_USD


class TestPureStringUtilities(omni.kit.test.AsyncTestCase):
    """Stateless string-based utility functions."""

    async def test_sanitize_prim_name(self):
        """Sanitization across normal, special-char, digit-prefixed, empty, and unicode inputs."""
        failures = []
        cases = [
            ("valid_name", {}, "valid_name"),
            ("name-with.special!chars", {}, "name_with_special_chars"),
            ("123name", {}, "prim_123name"),
            ("456test", {"prefix": "custom_"}, "custom_456test"),
            ("", {}, "prim_"),
        ]
        for name, kwargs, expected in cases:
            result = utils.sanitize_prim_name(name, **kwargs)
            if result != expected:
                failures.append(f"sanitize({name!r}, {kwargs}): {result!r} != {expected!r}")

        # Python 3 str.isalnum() returns True for accented unicode letters,
        # so sanitize_prim_name preserves them.  Verify non-alnum unicode is
        # replaced while accented letters survive.
        unicode_result = utils.sanitize_prim_name("tëst nàmé")
        if " " in unicode_result:
            failures.append(f"unicode: space not replaced in {unicode_result!r}")
        # Accented alphanumeric chars should be kept
        for ch in ("ë", "à", "é"):
            if ch not in unicode_result:
                failures.append(f"unicode: {ch!r} unexpectedly stripped from {unicode_result!r}")

        self.assertEqual(failures, [], "\n".join(failures))

    async def test_is_builtin_mdl_and_path_classifiers(self):
        """is_builtin_mdl, is_remote_path, is_usd_file across true/false cases."""
        failures = []

        # is_builtin_mdl: positive
        for p in ("OmniPBR.mdl", "OMNIPBR.MDL", "/some/dir/OmniPBR.mdl", "omniglass.mdl"):
            if not utils.is_builtin_mdl(p):
                failures.append(f"is_builtin_mdl({p!r}) should be True")
        # is_builtin_mdl: negative
        for p in ("CustomMaterial.mdl", "OmniPBR.usda", ""):
            if utils.is_builtin_mdl(p):
                failures.append(f"is_builtin_mdl({p!r}) should be False")

        # is_remote_path: positive
        for p in ("omniverse://server/a.usd", "http://x.com/a.usd", "https://x.com/a.usd"):
            if not utils.is_remote_path(p):
                failures.append(f"is_remote_path({p!r}) should be True")
        # is_remote_path: negative
        for p in ("/local/path/a.usd", "relative/a.usd", ""):
            if utils.is_remote_path(p):
                failures.append(f"is_remote_path({p!r}) should be False")

        # is_usd_file: positive
        for p in ("a.usd", "a.usda", "a.usdc", "a.usdz", "A.USD"):
            if not utils.is_usd_file(p):
                failures.append(f"is_usd_file({p!r}) should be True")
        # is_usd_file: negative
        for p in ("a.obj", "a.fbx", ""):
            if utils.is_usd_file(p):
                failures.append(f"is_usd_file({p!r}) should be False")

        self.assertEqual(failures, [], "\n".join(failures))

    async def test_norm_path_and_matches_prim_filter(self):
        """norm_path dot-collapse; matches_prim_filter include/exclude combinations."""
        failures = []

        # norm_path
        result = utils.norm_path("a/./b/../c")
        expected = os.path.normcase(os.path.normpath("a/c"))
        if result != expected:
            failures.append(f"norm_path: {result!r} != {expected!r}")

        # matches_prim_filter
        filter_cases = [
            # (prim_name, includes, excludes, expected)
            ("Body", ["Body.*"], None, True),
            ("Arm", ["Body.*"], None, False),
            ("BodyIgnored", ["Body.*"], [".*Ignored"], False),
            ("anything", [".*"], None, True),
            ("Body", [], None, False),
            ("Body", ["Body.*"], [], True),
        ]
        for prim_name, includes, excludes, expected in filter_cases:
            result = utils.matches_prim_filter(prim_name, includes, excludes)
            if result != expected:
                failures.append(f"matches_prim_filter({prim_name!r}, {includes}, {excludes}): {result} != {expected}")

        self.assertEqual(failures, [], "\n".join(failures))

    async def test_get_path_string(self):
        """Path extraction from str, None, and Sdf.AssetPath."""
        failures = []
        if utils.get_path_string("/some/path") != "/some/path":
            failures.append("str input failed")
        if utils.get_path_string(None) != "":
            failures.append("None should return ''")
        result = utils.get_path_string(Sdf.AssetPath("/test/asset.usd"))
        if not isinstance(result, str):
            failures.append(f"AssetPath returned {type(result)}, expected str")
        self.assertEqual(failures, [], "\n".join(failures))


class TestPrimSpecOperations(omni.kit.test.AsyncTestCase):
    """Prim spec creation, composition arc clearing, instanceable clearing."""

    async def setUp(self):
        """Create temporary directory."""
        self._tmpdir = tempfile.mkdtemp()

    async def tearDown(self):
        """Remove temporary directory."""
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    async def test_create_prim_spec_variants(self):
        """Default, typed, instanceable, and Over specifier in one layer."""
        layer = Sdf.Layer.CreateNew(os.path.join(self._tmpdir, "test.usda"))
        failures = []

        ps = utils.create_prim_spec(layer, "/A")
        if ps is None:
            failures.append("/A: returned None")
        elif ps.specifier != Sdf.SpecifierDef or ps.path.pathString != "/A":
            failures.append(f"/A: specifier={ps.specifier}, path={ps.path.pathString}")

        ps2 = utils.create_prim_spec(layer, "/B", type_name="Xform")
        if ps2 is None or ps2.typeName != "Xform":
            failures.append(f"/B: typeName={getattr(ps2, 'typeName', None)}")

        ps3 = utils.create_prim_spec(layer, "/C", instanceable=True)
        if ps3 is None or not ps3.instanceable:
            failures.append("/C: not instanceable")

        ps4 = utils.create_prim_spec(layer, "/D", specifier=Sdf.SpecifierOver)
        if ps4 is None or ps4.specifier != Sdf.SpecifierOver:
            failures.append(f"/D: specifier={getattr(ps4, 'specifier', None)}")

        self.assertEqual(failures, [], "\n".join(failures))

    async def test_clear_arcs_and_instanceable(self):
        """clear_composition_arcs (refs + payloads) and clear_instanceable_recursive (parent+children)."""
        layer = Sdf.Layer.CreateNew(os.path.join(self._tmpdir, "arcs.usda"))
        failures = []

        # References
        ps1 = utils.create_prim_spec(layer, "/Ref")
        ps1.referenceList.Append(Sdf.Reference("other.usda"))
        utils.clear_composition_arcs(ps1)
        if ps1.hasReferences:
            failures.append("References not cleared")

        # Payloads
        ps2 = utils.create_prim_spec(layer, "/Pay")
        ps2.payloadList.Append(Sdf.Payload("other.usda"))
        utils.clear_composition_arcs(ps2)
        if ps2.hasPayloads:
            failures.append("Payloads not cleared")

        # Recursive instanceable
        parent = utils.create_prim_spec(layer, "/P", instanceable=True)
        utils.create_prim_spec(layer, "/P/C", instanceable=True)
        utils.create_prim_spec(layer, "/P/C/G", instanceable=True)
        utils.clear_instanceable_recursive(parent)
        for path in ("/P", "/P/C", "/P/C/G"):
            spec = layer.GetPrimAtPath(path)
            if spec and spec.instanceable:
                failures.append(f"{path} still instanceable")

        self.assertEqual(failures, [], "\n".join(failures))


class TestLayerOperations(omni.kit.test.AsyncTestCase):
    """Layer-level utilities: relative paths, hierarchy, ensure spec."""

    async def setUp(self):
        """Create temporary directory."""
        self._tmpdir = tempfile.mkdtemp()

    async def tearDown(self):
        """Remove temporary directory."""
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    async def test_relative_path_and_hierarchy(self):
        """get_relative_layer_path (same/sub/parent dir) and ensure_prim_hierarchy (shallow+deep)."""
        failures = []

        # -- get_relative_layer_path --
        base_layer = Sdf.Layer.CreateNew(os.path.join(self._tmpdir, "base.usda"))

        rel = utils.get_relative_layer_path(base_layer, os.path.join(self._tmpdir, "target.usda"))
        if os.path.normpath(rel) != os.path.normpath("./target.usda"):
            failures.append(f"same-dir: {rel!r}")

        rel = utils.get_relative_layer_path(base_layer, os.path.join(self._tmpdir, "sub", "target.usda"))
        if os.path.normpath(rel) != os.path.normpath("./sub/target.usda"):
            failures.append(f"subdir: {rel!r}")

        subdir = os.path.join(self._tmpdir, "sub")
        os.makedirs(subdir, exist_ok=True)
        sub_layer = Sdf.Layer.CreateNew(os.path.join(subdir, "sub.usda"))
        rel = utils.get_relative_layer_path(sub_layer, os.path.join(self._tmpdir, "target.usda"))
        if os.path.normpath(rel) != os.path.normpath("../target.usda"):
            failures.append(f"parent-dir: {rel!r}")

        # -- ensure_prim_hierarchy --
        layer = Sdf.Layer.CreateNew(os.path.join(self._tmpdir, "hier.usda"))

        utils.ensure_prim_hierarchy(layer, "/R/C/G")
        for p in ("/R", "/R/C"):
            if not layer.GetPrimAtPath(p):
                failures.append(f"missing {p}")

        utils.ensure_prim_hierarchy(layer, "/A/B/C/D/E")
        for p in ("/A", "/A/B", "/A/B/C", "/A/B/C/D"):
            if not layer.GetPrimAtPath(p):
                failures.append(f"missing deep {p}")

        # -- ensure_prim_spec_in_layer --
        layer2 = Sdf.Layer.CreateNew(os.path.join(self._tmpdir, "ensure.usda"))
        ps = utils.ensure_prim_spec_in_layer(layer2, Sdf.Path("/New"))
        if ps is None:
            failures.append("ensure_prim_spec returned None")
        elif ps.specifier != Sdf.SpecifierOver:
            failures.append(f"ensure_prim_spec specifier: {ps.specifier}")
        ps2 = utils.ensure_prim_spec_in_layer(layer2, Sdf.Path("/New"))
        if ps2 is None or ps2.path != ps.path:
            failures.append("ensure_prim_spec did not return existing")

        self.assertEqual(failures, [], "\n".join(failures))


class TestStageQueries(omni.kit.test.AsyncTestCase):
    """Stage query utilities using UR10e test asset."""

    async def test_scope_root_default_prim_and_ancestor(self):
        """get_scope_root variants, get_default_prim_path, find_ancestor_matching."""
        stage = Usd.Stage.Open(_UR10E_USD)
        failures = []

        # Scope root: "/" and ""
        for scope in ("/", ""):
            if utils.get_scope_root(stage, scope) != stage.GetPseudoRoot():
                failures.append(f"get_scope_root({scope!r}) != pseudo root")

        # Valid scope
        valid = utils.get_scope_root(stage, "/ur10e")
        if valid is None or not valid.IsValid():
            failures.append("get_scope_root('/ur10e') invalid")

        # Invalid with fallback
        if utils.get_scope_root(stage, "/nope", fallback_to_pseudo_root=True) != stage.GetPseudoRoot():
            failures.append("Invalid scope with fallback != pseudo root")

        # Invalid without fallback
        if utils.get_scope_root(stage, "/nope", fallback_to_pseudo_root=False) is not None:
            failures.append("Invalid scope without fallback should be None")

        # Default prim path
        dp = utils.get_default_prim_path(stage)
        if dp != "/ur10e":
            failures.append(f"get_default_prim_path: {dp!r} != '/ur10e'")

        # Default prim path fallback (empty stage)
        tmpdir = tempfile.mkdtemp()
        try:
            empty_layer = Sdf.Layer.CreateNew(os.path.join(tmpdir, "empty.usda"))
            empty_stage = Usd.Stage.Open(empty_layer)
            if utils.get_default_prim_path(empty_stage, fallback="/Root") != "/Root":
                failures.append("Fallback default prim path failed")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        # find_ancestor_matching: found
        default_prim = stage.GetDefaultPrim()
        children = list(default_prim.GetChildren())
        if children:
            anc = utils.find_ancestor_matching(children[0], lambda p: p.GetTypeName() == "Xform")
            if anc is None:
                failures.append("Expected Xform ancestor for first child")

        # find_ancestor_matching: not found
        if utils.find_ancestor_matching(default_prim, lambda p: p.GetTypeName() == "Camera") is not None:
            failures.append("Camera ancestor should be None on default prim")

        self.assertEqual(failures, [], "\n".join(failures))


class TestStageMetadataCopy(omni.kit.test.AsyncTestCase):
    """Stage metadata copy from stage and layer."""

    async def setUp(self):
        """Create temporary directory."""
        self._tmpdir = tempfile.mkdtemp()

    async def tearDown(self):
        """Remove temporary directory."""
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    async def test_copy_metadata_from_stage_and_layer(self):
        """copy_stage_metadata copies metersPerUnit/upAxis; copy_stage_metadata_from_layer skips defaultPrim."""
        failures = []

        # From stage
        src_stage = Usd.Stage.Open(_UR10E_USD)
        dst1 = Sdf.Layer.CreateNew(os.path.join(self._tmpdir, "dst1.usda"))
        utils.copy_stage_metadata(src_stage, dst1)
        for key in ("metersPerUnit", "upAxis"):
            if dst1.pseudoRoot.GetInfo(key) is None:
                failures.append(f"copy_stage_metadata: {key} not copied")

        # From layer - defaultPrim should be skipped
        src_layer = src_stage.GetRootLayer()
        dst2 = Sdf.Layer.CreateNew(os.path.join(self._tmpdir, "dst2.usda"))
        utils.copy_stage_metadata_from_layer(src_layer, dst2)
        if dst2.defaultPrim:
            failures.append("copy_stage_metadata_from_layer: defaultPrim should not be copied")
        # But other metadata should be present
        if dst2.pseudoRoot.GetInfo("metersPerUnit") is None:
            failures.append("copy_stage_metadata_from_layer: metersPerUnit not copied")

        self.assertEqual(failures, [], "\n".join(failures))


class TestFileOperations(omni.kit.test.AsyncTestCase):
    """File comparison and copy utilities."""

    async def setUp(self):
        """Create temporary directory."""
        self._tmpdir = tempfile.mkdtemp()

    async def tearDown(self):
        """Remove temporary directory."""
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    async def test_files_are_identical_and_copy_file_to_directory(self):
        """files_are_identical (same/diff content/size); copy_file_to_directory (basic/missing/cached/conflict)."""
        failures = []

        # -- files_are_identical --
        f1 = os.path.join(self._tmpdir, "f1.txt")
        f2 = os.path.join(self._tmpdir, "f2.txt")
        f3 = os.path.join(self._tmpdir, "f3.txt")
        for path, content in [(f1, b"same"), (f2, b"same"), (f3, b"different content here")]:
            with open(path, "wb") as f:
                f.write(content)

        if not utils.files_are_identical(f1, f2):
            failures.append("Identical files not detected")
        if utils.files_are_identical(f1, f3):
            failures.append("Different content not detected")
        if utils.files_are_identical(f2, f3):
            failures.append("Different size not detected")

        # -- copy_file_to_directory --
        src_dir = os.path.join(self._tmpdir, "src")
        dst_dir = os.path.join(self._tmpdir, "dst")
        os.makedirs(src_dir)
        os.makedirs(dst_dir)

        # Basic copy
        src = os.path.join(src_dir, "file.txt")
        with open(src, "w") as f:
            f.write("content")
        result = utils.copy_file_to_directory(src, dst_dir)
        if result is None or not os.path.exists(result):
            failures.append("Basic copy failed")

        # Nonexistent source
        if utils.copy_file_to_directory("/nonexistent/x.txt", dst_dir) is not None:
            failures.append("Nonexistent source should return None")

        # Already collected
        normed = utils.norm_path(src)
        cached = {normed: "/already/collected.txt"}
        if utils.copy_file_to_directory(src, dst_dir, cached) != "/already/collected.txt":
            failures.append("Already-collected cache miss")

        # Conflict rename
        conflict = os.path.join(dst_dir, "conflict.txt")
        with open(conflict, "w") as f:
            f.write("existing")
        src2 = os.path.join(src_dir, "conflict.txt")
        with open(src2, "w") as f:
            f.write("different content")
        result2 = utils.copy_file_to_directory(src2, dst_dir)
        if result2 is None:
            failures.append("Conflict copy returned None")
        elif os.path.basename(result2) == "conflict.txt":
            failures.append(f"Conflict not renamed: {result2}")

        self.assertEqual(failures, [], "\n".join(failures))


class TestMergeTokenListOp(omni.kit.test.AsyncTestCase):
    """TokenListOp merging: None, explicit, prepended dedup."""

    async def test_merge_scenarios(self):
        """Merge into None, into existing explicit, and prepend deduplication."""
        failures = []

        # Into None -> explicit
        r1 = utils.merge_token_list_op(None, ["PhysicsAPI"])
        if not r1.isExplicit or "PhysicsAPI" not in list(r1.explicitItems):
            failures.append("Merge into None: missing PhysicsAPI or not explicit")

        # Into existing explicit
        existing = Sdf.TokenListOp.CreateExplicit(["ExistingAPI"])
        r2 = utils.merge_token_list_op(existing, ["NewAPI"])
        items2 = list(r2.explicitItems)
        for token in ("NewAPI", "ExistingAPI"):
            if token not in items2:
                failures.append(f"Merge into explicit: {token} missing")

        # Prepend dedup
        prep = Sdf.TokenListOp()
        prep.prependedItems = ["SharedAPI"]
        r3 = utils.merge_token_list_op(prep, ["SharedAPI", "NewAPI"])
        prepended = list(r3.prependedItems)
        if prepended.count("SharedAPI") != 1:
            failures.append(f"SharedAPI duplicated: count={prepended.count('SharedAPI')}")
        if "NewAPI" not in prepended:
            failures.append("NewAPI missing from prepended")

        self.assertEqual(failures, [], "\n".join(failures))


class TestAssetPathResolution(omni.kit.test.AsyncTestCase):
    """Asset path resolution, remapping, and arc priority."""

    async def setUp(self):
        """Create temporary directory."""
        self._tmpdir = tempfile.mkdtemp()

    async def tearDown(self):
        """Remove temporary directory."""
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    async def test_resolve_and_find_first_arc(self):
        """resolve_asset_path (empty/abs/missing/fallback); find_first_resolvable_arc (priority/empty)."""
        failures = []

        # resolve_asset_path
        if utils.resolve_asset_path("") != "":
            failures.append("Empty path should resolve to ''")

        existing = os.path.join(self._tmpdir, "test.usda")
        Sdf.Layer.CreateNew(existing)
        if utils.resolve_asset_path(existing) != existing:
            failures.append("Absolute existing not resolved to itself")

        if utils.resolve_asset_path("/nonexistent/path.usd") != "":
            failures.append("Nonexistent absolute should resolve to ''")

        fb_file = os.path.join(self._tmpdir, "asset.usda")
        Sdf.Layer.CreateNew(fb_file)
        fb_result = utils.resolve_asset_path("asset.usda", fallback_dirs=[self._tmpdir])
        if fb_result != fb_file:
            failures.append(f"Fallback resolve: {fb_result!r} != {fb_file!r}")

        # find_first_resolvable_arc
        p_file = os.path.join(self._tmpdir, "payload.usda")
        r_file = os.path.join(self._tmpdir, "ref.usda")
        Sdf.Layer.CreateNew(p_file)
        Sdf.Layer.CreateNew(r_file)

        def resolve(path):
            full = os.path.join(self._tmpdir, path)
            return full if os.path.isfile(full) else ""

        # Payload takes priority over reference
        arc1 = utils.find_first_resolvable_arc([Sdf.Payload("payload.usda")], [Sdf.Reference("ref.usda")], resolve)
        if arc1 != p_file:
            failures.append(f"Payload priority: {arc1!r}")

        # Empty payloads falls through to reference
        arc2 = utils.find_first_resolvable_arc([], [Sdf.Reference("ref.usda")], resolve)
        if arc2 != r_file:
            failures.append(f"Fallthrough to ref: {arc2!r}")

        # Both empty
        arc3 = utils.find_first_resolvable_arc([], [], resolve)
        if arc3 is not None:
            failures.append(f"Both empty should be None, got {arc3!r}")

        self.assertEqual(failures, [], "\n".join(failures))
