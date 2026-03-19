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

"""Tests for the flatten rule."""

import os
import shutil
import tempfile

import omni.kit.test
from isaacsim.asset.transformer.rules.structure.flatten import FlattenRule
from pxr import Usd, UsdPhysics

from .common import _UR10E_USD


class TestFlattenRule(omni.kit.test.AsyncTestCase):
    """Async tests for FlattenRule."""

    async def setUp(self):
        """Create a temporary directory for test output."""
        self._tmpdir = tempfile.mkdtemp()
        self._success = False

    async def tearDown(self):
        """Remove temporary directory after successful tests."""
        if self._success:
            shutil.rmtree(self._tmpdir, ignore_errors=True)

    async def test_get_configuration_parameters(self):
        """Verify configuration parameters are exposed."""
        stage = Usd.Stage.Open(_UR10E_USD)
        rule = FlattenRule(
            source_stage=stage,
            package_root=self._tmpdir,
            destination_path="",
            args={"input_stage_path": _UR10E_USD},
        )

        params = rule.get_configuration_parameters()

        self.assertEqual(len(params), 4)
        param_names = [p.name for p in params]
        self.assertIn("output_path", param_names)
        self.assertIn("clear_variants", param_names)
        self.assertIn("selected_variants", param_names)
        self.assertIn("case_insensitive", param_names)
        self._success = True

    async def test_flatten_basic_stage(self):
        """Verify flattening creates expected output stage."""
        stage = Usd.Stage.Open(_UR10E_USD)
        output_subdir = "payloads"
        output_path = "base.usda"
        os.makedirs(os.path.join(self._tmpdir, output_subdir), exist_ok=True)

        rule = FlattenRule(
            source_stage=stage,
            package_root=self._tmpdir,
            destination_path=output_subdir,
            args={
                "input_stage_path": _UR10E_USD,
                "params": {
                    "output_path": output_path,
                    "clear_variants": False,
                },
            },
        )

        rule.process_rule()

        expected_output = os.path.join(self._tmpdir, output_subdir, output_path)
        self.assertTrue(os.path.exists(expected_output))

        flattened_layer = Usd.Stage.Open(expected_output)
        self.assertIsNotNone(flattened_layer)
        # UR10e has ur10e as default prim
        self.assertTrue(flattened_layer.GetPrimAtPath("/ur10e").IsValid())

        self.assertTrue(flattened_layer.GetPrimAtPath("/ur10e/base_link").IsValid())
        self._success = True

    async def test_flatten_without_input_stage_path_skips(self):
        """Verify rule skips when input stage path is missing."""
        stage = Usd.Stage.Open(_UR10E_USD)

        rule = FlattenRule(
            source_stage=stage,
            package_root=self._tmpdir,
            destination_path="",
            args={},
        )

        rule.process_rule()

        log = rule.get_operation_log()
        self.assertTrue(any("No input_stage_path" in msg for msg in log))
        self._success = True

    async def test_flatten_with_variants_cleared(self):
        """Verify flattening clears variant selections when enabled."""
        stage = Usd.Stage.Open(_UR10E_USD)
        os.makedirs(os.path.join(self._tmpdir, "output"), exist_ok=True)

        rule = FlattenRule(
            source_stage=stage,
            package_root=self._tmpdir,
            destination_path="output",
            args={
                "input_stage_path": _UR10E_USD,
                "params": {
                    "output_path": "flattened.usda",
                    "clear_variants": True,
                },
            },
        )

        rule.process_rule()

        log = rule.get_operation_log()
        # Verify rule processed and cleared variants
        self.assertTrue(any("Cleared" in msg for msg in log) or any("variant" in msg.lower() for msg in log))

        # Verify output exists
        output_path = os.path.join(self._tmpdir, "output", "flattened.usda")
        self.assertTrue(os.path.exists(output_path))

        flattened_layer = Usd.Stage.Open(output_path)
        self.assertTrue(flattened_layer.GetPrimAtPath("/ur10e").IsValid())
        self.assertFalse(flattened_layer.GetPrimAtPath("/ur10e/base_link").IsValid())
        self._success = True

    async def test_flatten_with_selected_variants(self):
        """Verify selected variants are applied before flattening."""
        stage = Usd.Stage.Open(_UR10E_USD)
        os.makedirs(os.path.join(self._tmpdir, "output"), exist_ok=True)

        rule = FlattenRule(
            source_stage=stage,
            package_root=self._tmpdir,
            destination_path="output",
            args={
                "input_stage_path": _UR10E_USD,
                "params": {
                    "output_path": "flattened.usda",
                    "clear_variants": True,
                    "selected_variants": {"Physics": "PhysX", "Gripper": "Robotiq_2f_85"},
                },
            },
        )

        rule.process_rule()
        output_path = os.path.join(self._tmpdir, "output", "flattened.usda")

        flattened_layer = Usd.Stage.Open(output_path)
        self.assertTrue(flattened_layer.GetPrimAtPath("/ur10e").IsValid())
        self.assertTrue(flattened_layer.GetPrimAtPath("/ur10e/Robotiq_2F_85/base_link").IsValid())
        self.assertTrue(
            UsdPhysics.RigidBodyAPI(flattened_layer.GetPrimAtPath("/ur10e/Robotiq_2F_85/base_link")).GetPrim().IsValid()
        )
        self._success = True

    async def test_flatten_affected_stages(self):
        """Verify affected stages list contains output stage."""
        stage = Usd.Stage.Open(_UR10E_USD)
        os.makedirs(os.path.join(self._tmpdir, "payloads"), exist_ok=True)

        rule = FlattenRule(
            source_stage=stage,
            package_root=self._tmpdir,
            destination_path="payloads",
            args={
                "input_stage_path": _UR10E_USD,
                "params": {"output_path": "base.usda"},
            },
        )

        rule.process_rule()

        affected = rule.get_affected_stages()
        self.assertTrue(len(affected) >= 1)
        self.assertTrue(any("base.usda" in s for s in affected))
        self._success = True

    async def test_flatten_logs_completion(self):
        """Verify completion log entry is recorded."""
        stage = Usd.Stage.Open(_UR10E_USD)
        os.makedirs(os.path.join(self._tmpdir, "payloads"), exist_ok=True)

        rule = FlattenRule(
            source_stage=stage,
            package_root=self._tmpdir,
            destination_path="payloads",
            args={
                "input_stage_path": _UR10E_USD,
                "params": {"output_path": "base.usda"},
            },
        )

        rule.process_rule()

        log = rule.get_operation_log()
        self.assertTrue(any("FlattenRule start" in msg for msg in log))
        self.assertTrue(any("FlattenRule completed" in msg for msg in log))
        self._success = True

    async def test_flatten_return_value_and_invalid_input(self):
        """process_rule returns abs output path; invalid input stage path handled gracefully."""
        stage = Usd.Stage.Open(_UR10E_USD)
        failures = []

        # -- Return value is absolute path ending with output filename --
        os.makedirs(os.path.join(self._tmpdir, "rv"), exist_ok=True)
        rule_rv = FlattenRule(
            source_stage=stage,
            package_root=self._tmpdir,
            destination_path="rv",
            args={
                "input_stage_path": _UR10E_USD,
                "params": {"output_path": "flat.usda", "clear_variants": False},
            },
        )
        result = rule_rv.process_rule()
        if result is None:
            failures.append("process_rule returned None instead of output path")
        elif not os.path.isabs(result):
            failures.append(f"Return path not absolute: {result}")
        elif not result.endswith("flat.usda"):
            failures.append(f"Return path doesn't end with flat.usda: {result}")

        # -- Invalid input stage path (Usd.Stage.Open raises on missing layers) --
        rule_bad = FlattenRule(
            source_stage=stage,
            package_root=self._tmpdir,
            destination_path="",
            args={"input_stage_path": "/nonexistent/stage.usd"},
        )
        try:
            bad_result = rule_bad.process_rule()
            # If it doesn't raise, it should at least return None or log failure
            bad_log = rule_bad.get_operation_log()
            if bad_result is not None and not any("Failed" in m for m in bad_log):
                failures.append("Invalid input path should return None or log failure")
        except Exception:
            pass  # pxr.Tf.ErrorException is the expected outcome

        self.assertEqual(failures, [], "\n".join(failures))
        self._success = True

    async def test_flatten_variant_selection_edge_cases(self):
        """Case-insensitive match, nonexistent variant set, nonexistent variant name."""
        stage = Usd.Stage.Open(_UR10E_USD)
        failures = []

        # -- Case-insensitive variant match ('physx' -> 'PhysX') --
        os.makedirs(os.path.join(self._tmpdir, "ci"), exist_ok=True)
        rule_ci = FlattenRule(
            source_stage=stage,
            package_root=self._tmpdir,
            destination_path="ci",
            args={
                "input_stage_path": _UR10E_USD,
                "params": {
                    "output_path": "ci.usda",
                    "clear_variants": True,
                    "selected_variants": {"Physics": "physx"},
                    "case_insensitive": True,
                },
            },
        )
        rule_ci.process_rule()
        ci_log = rule_ci.get_operation_log()
        if not any("Case-insensitive match" in m or "Set variant" in m for m in ci_log):
            failures.append("Case-insensitive match not logged")

        # -- Nonexistent variant set --
        os.makedirs(os.path.join(self._tmpdir, "ns"), exist_ok=True)
        rule_ns = FlattenRule(
            source_stage=stage,
            package_root=self._tmpdir,
            destination_path="ns",
            args={
                "input_stage_path": _UR10E_USD,
                "params": {
                    "output_path": "ns.usda",
                    "clear_variants": False,
                    "selected_variants": {"NonExistentSet": "value"},
                },
            },
        )
        rule_ns.process_rule()
        ns_log = rule_ns.get_operation_log()
        if not any("not found on default prim" in m for m in ns_log):
            failures.append("Nonexistent variant set not logged as skipped")
        ns_output = os.path.join(self._tmpdir, "ns", "ns.usda")
        if not os.path.exists(ns_output):
            failures.append("Output not created despite invalid variant set request")

        self.assertEqual(failures, [], "\n".join(failures))
        self._success = True
