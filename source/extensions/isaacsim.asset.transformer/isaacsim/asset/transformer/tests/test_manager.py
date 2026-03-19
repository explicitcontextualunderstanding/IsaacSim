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

"""Tests for the asset transformer manager and registry."""

import types
from unittest.mock import patch

import omni.kit.test
from isaacsim.asset.transformer.manager import AssetTransformerManager, RuleRegistry
from isaacsim.asset.transformer.models import RuleConfigurationParam, RuleProfile, RuleSpec
from isaacsim.asset.transformer.rule_interface import RuleInterface


class _DummyRule(RuleInterface):
    """Minimal rule implementation for manager tests."""

    def process_rule(self) -> None:
        """Record a dummy log entry and affected stage."""
        self.log_operation("dummy-run")
        self.add_affected_stage(self.destination_path or "memory")
        return

    def get_configuration_parameters(self) -> list[RuleConfigurationParam]:
        """Return an empty configuration parameter list.

        Returns:
            Empty list of configuration parameters.
        """
        return []


class _FakeLayer:
    """Minimal layer mock supporting Export, Save, dirty, and realPath."""

    def __init__(self) -> None:
        self.realPath = ""
        self.dirty = False

    def Export(self, path: str) -> bool:  # noqa: N802
        self.realPath = path
        return True

    def Save(self) -> None:  # noqa: N802
        pass


class _FakeStage:
    """Minimal stage mock supporting GetRootLayer and Flatten."""

    def __init__(self) -> None:
        self._layer = _FakeLayer()

    def GetRootLayer(self):  # noqa: N802
        return self._layer

    def Flatten(self):  # noqa: N802
        return _FakeLayer()


def _fake_usd(open_returns: object | None) -> types.SimpleNamespace:
    """Create a fake Usd module with a controllable Open result.

    Args:
        open_returns: Value returned by the fake Usd.Stage.Open call.

    Returns:
        Simple namespace mimicking the Usd module.
    """
    fake_stage_mod = types.SimpleNamespace()
    fake_stage_mod.Open = lambda path: open_returns
    fake_stage_mod.CreateInMemory = lambda: object()
    return types.SimpleNamespace(Stage=fake_stage_mod)


def _fake_sdf() -> types.SimpleNamespace:
    """Create a fake Sdf module with a controllable Layer.FindOrOpen result.

    Returns:
        Simple namespace mimicking the Sdf module.
    """
    fake_layer_mod = types.SimpleNamespace()
    fake_layer_mod.FindOrOpen = lambda path: None
    return types.SimpleNamespace(Layer=fake_layer_mod)


class TestManager(omni.kit.test.AsyncTestCase):
    """Async tests for AssetTransformerManager and RuleRegistry."""

    async def asyncSetUp(self):
        """Clear the rule registry before each test."""
        RuleRegistry().clear()

    async def asyncTearDown(self):
        """Clear the rule registry after each test."""
        RuleRegistry().clear()

    async def test_rule_registry_register_get_clear(self):
        """Verify registry register, get, and clear behaviors."""
        reg = RuleRegistry()
        reg.register(_DummyRule)
        fqcn = f"{_DummyRule.__module__}.{_DummyRule.__qualname__}"
        cls = reg.get(fqcn)
        self.assertIs(cls, _DummyRule)
        reg.clear()
        self.assertIsNone(reg.get(fqcn))

        class NotARule:  # noqa: D401
            """Not a RuleInterface subclass."""

        with self.assertRaises(TypeError):
            reg.register(NotARule)  # type: ignore[arg-type]

    async def test_manager_run_happy_path(self):
        """Verify manager run succeeds with a registered rule."""
        fake_stage = _FakeStage()
        fake_usd = _fake_usd(open_returns=fake_stage)
        fake_sdf = _fake_sdf()
        with (
            patch("isaacsim.asset.transformer.manager.Usd", fake_usd, create=True),
            patch("isaacsim.asset.transformer.manager.Sdf", fake_sdf, create=True),
            patch("isaacsim.asset.transformer.manager.os.makedirs"),
            patch("isaacsim.asset.transformer.rule_interface.Usd", fake_usd, create=True),
        ):
            reg = RuleRegistry()
            reg.clear()
            reg.register(_DummyRule)
            fqcn = f"{_DummyRule.__module__}.{_DummyRule.__qualname__}"
            profile = RuleProfile(
                profile_name="p",
                rules=[
                    RuleSpec(name="r1", type=fqcn, destination="out.usda"),
                    RuleSpec(name="disabled", type=fqcn, enabled=False),
                ],
                interface_asset_name="iface",
                output_package_root="/pkg",
            )
            mgr = AssetTransformerManager()
            report = mgr.run(input_stage_path="in.usda", profile=profile, package_root=None)
            self.assertEqual(len(report.results), 1)
            result = report.results[0]
            self.assertTrue(result.success)
            self.assertIsNone(result.error)
            self.assertTrue(result.log and result.log[-1]["message"] == "dummy-run")
            self.assertIn("out.usda", result.affected_stages)
            self.assertIsInstance(result.finished_at, str)
            self.assertIsInstance(report.finished_at, str)

    async def test_manager_run_missing_implementation_sets_error(self):
        """Verify missing rule implementations set error status."""
        fake_stage = _FakeStage()
        fake_usd = _fake_usd(open_returns=fake_stage)
        fake_sdf = _fake_sdf()
        with (
            patch("isaacsim.asset.transformer.manager.Usd", fake_usd, create=True),
            patch("isaacsim.asset.transformer.manager.Sdf", fake_sdf, create=True),
            patch("isaacsim.asset.transformer.manager.os.makedirs"),
            patch("isaacsim.asset.transformer.rule_interface.Usd", fake_usd, create=True),
        ):
            RuleRegistry().clear()
            unknown_type = "non.existent.Rule"
            profile = RuleProfile(profile_name="p", rules=[RuleSpec(name="r", type=unknown_type)])
            mgr = AssetTransformerManager()
            report = mgr.run(input_stage_path="in.usda", profile=profile)
            self.assertEqual(len(report.results), 1)
            res = report.results[0]
            self.assertFalse(res.success)
            self.assertIsInstance(res.error, str)
            self.assertIn("No rule implementation registered", res.error)
            self.assertIsInstance(res.finished_at, str)

    async def test_manager_run_source_open_failure_raises(self):
        """Verify source stage open failures raise errors."""
        fake_usd = _fake_usd(open_returns=None)
        with patch("isaacsim.asset.transformer.manager.Usd", fake_usd, create=True):
            mgr = AssetTransformerManager()
            profile = RuleProfile(profile_name="p", rules=[])
            with self.assertRaises(RuntimeError) as excinfo:
                mgr.run(input_stage_path="missing.usda", profile=profile)
            self.assertIn("Failed to open source stage", str(excinfo.exception))
