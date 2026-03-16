# SPDX-FileCopyrightText: Copyright (c) 2018-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


"""Tests for URDF importer command implementations."""

import asyncio
import gc
import os
import shutil

import carb
import isaacsim.core.experimental.utils.stage as stage_utils

# NOTE:
#   omni.kit.test - std python's unittest module with additional wrapping to add suport for async/await tests
#   For most things refer to unittest docs: https://docs.python.org/3/library/unittest.html
import omni.kit.commands
import omni.kit.test
from isaacsim.asset.importer.urdf import URDFImporterConfig
from pxr import UsdPhysics


class TestURDFCommands(omni.kit.test.AsyncTestCase):
    """Test URDF importer command implementations.

    Example:

    .. code-block:: python

        >>> import omni.kit.test
        >>> class Example(omni.kit.test.AsyncTestCase):
        ...     pass
        ...
    """

    async def setUp(self) -> None:
        """Prepare shared test fixtures.

        Example:

        .. code-block:: python

            >>> import omni.usd
            >>> omni.usd.get_context()  # doctest: +SKIP
        """
        ext_manager = omni.kit.app.get_app().get_extension_manager()
        urdf_ext_id = ext_manager.get_enabled_extension_id("isaacsim.asset.importer.urdf")
        self._urdf_extension_path = ext_manager.get_extension_path(urdf_ext_id)
        self._urdf_path = os.path.join(
            self._urdf_extension_path, "data", "urdf", "robots", "carter", "urdf", "carter.urdf"
        )
        await omni.usd.get_context().new_stage_async()
        await omni.kit.app.get_app().next_update_async()
        self._stage = omni.usd.get_context().get_stage()

    async def tearDown(self) -> None:
        """Wait for stage loading to complete after tests.

        Example:

        .. code-block:: python

            >>> import asyncio
            >>> asyncio.sleep(0)  # doctest: +SKIP
        """
        while omni.usd.get_context().get_stage_loading_status()[2] > 0:
            carb.log_info("tearDown, assets still loading, waiting to finish...")
            await asyncio.sleep(1.0)
        await omni.kit.app.get_app().next_update_async()

    def test_urdf_create_import_config(self) -> None:
        """Test URDFCreateImportConfig command execution.

        Example:

        .. code-block:: python

            >>> import omni.kit.commands
            >>> from isaacsim.asset.importer.urdf import URDFImporterConfig

            >>> status, config = omni.kit.commands.execute("URDFCreateImportConfig")
            >>> isinstance(config, URDFImporterConfig)
            True
        """
        status, config = omni.kit.commands.execute("URDFCreateImportConfig")

        self.assertTrue(status)
        self.assertIsInstance(config, URDFImporterConfig)
        self.assertIsNone(config.urdf_path)
        self.assertIsNone(config.usd_path)
        self.assertFalse(config.merge_mesh)
        self.assertFalse(config.debug_mode)
        self.assertFalse(config.collision_from_visuals)
        self.assertEqual(config.collision_type, "Convex Hull")
        self.assertFalse(config.allow_self_collision)
        self.assertEqual(config.ros_package_paths, [])

    async def test_urdf_import_robot(self) -> None:
        """Test URDFImportRobot command execution.

        Example:

        .. code-block:: python

            >>> import omni.kit.commands
            >>> from isaacsim.asset.importer.urdf import URDFImporterConfig

            >>> config = URDFImporterConfig(urdf_path="/path/to/robot.urdf")
            >>> status, path = omni.kit.commands.execute(
            ...     "URDFImportRobot",
            ...     urdf_path="/path/to/robot.urdf",
            ...     import_config=config
            ... )  # doctest: +SKIP
            >>> isinstance(path, str)
            True
        """
        config = URDFImporterConfig(urdf_path=self._urdf_path)
        status, output_path = omni.kit.commands.execute(
            "URDFImportRobot",
            urdf_path=self._urdf_path,
            import_config=config,
            dest_path="",
        )
        print(f"status: {status}")
        print(f"output_path: {output_path}")
        self.assertTrue(status)
        self.assertIsInstance(output_path, str)
        self.assertTrue(os.path.exists(output_path))
        self.assertTrue(output_path.endswith(".usda"))

        # Verify stage was opened
        status, self._stage = stage_utils.open_stage(output_path)
        self.assertTrue(status)
        self.assertIsNotNone(self._stage)

        prim = self._stage.GetDefaultPrim()
        self.assertIsNotNone(prim)
        self.assertEqual(prim.GetName(), "carter")

        # Clean up
        self._stage = None
        gc.collect()
        try:
            shutil.rmtree(os.path.dirname(output_path))
        except OSError as e:
            carb.log_warn(f"Warning: {os.path.dirname(output_path)}: {e.strerror}")
