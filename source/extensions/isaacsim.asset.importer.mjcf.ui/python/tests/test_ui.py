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


"""UI tests for the MJCF importer extension."""

import asyncio
import gc
import os
import shutil

import carb

# NOTE:
#   omni.kit.test - std python's unittest module with additional wrapping to add suport for async/await tests
#   For most things refer to unittest docs: https://docs.python.org/3/library/unittest.html
import omni.kit.test
import omni.kit.ui_test as ui_test
from isaacsim.asset.importer.mjcf import MJCFImporter
from isaacsim.asset.importer.mjcf.ui.impl import extension as mjcf_ui_extension
from isaacsim.asset.importer.utils import test_utils
from isaacsim.core.experimental.utils import stage as stage_utils
from isaacsim.test.utils import MenuUITestCase
from pxr import Sdf


# Having a test class dervived from omni.kit.test.AsyncTestCase declared on the root of module will make it auto-discoverable by omni.kit.test
class TestImporterUI(MenuUITestCase):
    """Test MJCF importer UI interactions.

    Example:

    .. code-block:: python

        >>> import omni.kit.test
        >>> class Example(omni.kit.test.AsyncTestCase):
        ...     pass
        ...
    """

    # Before running each test
    async def setUp(self) -> None:
        """Prepare the UI test fixture.

        Example:

        .. code-block:: python

            >>> import omni.usd
            >>> omni.usd.get_context()  # doctest: +SKIP
        """
        self._timeline = omni.timeline.get_timeline_interface()

        ext_manager = omni.kit.app.get_app().get_extension_manager()
        ext_id = ext_manager.get_enabled_extension_id("isaacsim.asset.importer.mjcf.ui")
        mjcf_ext_id = ext_manager.get_enabled_extension_id("isaacsim.asset.importer.mjcf")
        self._extension_path = ext_manager.get_extension_path(ext_id)
        self._mjcf_extension_path = ext_manager.get_extension_path(mjcf_ext_id)
        self._mjcf_path = os.path.normpath(os.path.join(self._mjcf_extension_path, "data", "mjcf", "nv_ant.xml"))
        await omni.usd.get_context().new_stage_async()
        await omni.kit.app.get_app().next_update_async()

    # After running each test
    async def tearDown(self) -> None:
        """Wait for stage loading to complete after the test.

        Example:

        .. code-block:: python

            >>> import asyncio
            >>> asyncio.sleep(0)  # doctest: +SKIP
        """
        while omni.usd.get_context().get_stage_loading_status()[2] > 0:
            carb.log_info("tearDown, assets still loading, waiting to finish...")
            await asyncio.sleep(1.0)
        await omni.kit.app.get_app().next_update_async()
        # await omni.usd.get_context().new_stage_async()

    def find_content_file(self, window_name, filename):

        carb.log_info(f"Finding file {filename} in window {window_name}")
        for widget in ui_test.find_all(f"{window_name}//Frame/**/TreeView[*]"):
            for file_widget in widget.find_all("**/Label[*]"):
                if file_widget.widget.text == filename:
                    carb.log_info(f"Found file {filename} in window {window_name}")
                    return file_widget
        return None

    async def test_import_ant_from_ui(self, delete_output_on_success=True) -> None:
        """Import the ant asset via the UI and validate output.

        Example:

        .. code-block:: python

            >>> from isaacsim.asset.importer.mjcf import MJCFImporterConfig
            >>> MJCFImporterConfig()
            <...>
        """
        await self.menu_click_with_retry("File/Import")

        dir_widget = await self.find_widget_with_retry(
            "Select File//Frame/**/StringField[*].identifier=='filepicker_directory_path'"
        )

        await dir_widget.input(self._mjcf_path)
        await ui_test.human_delay()
        await omni.kit.app.get_app().next_update_async()

        grid_view = await self.find_widget_with_retry("Select File//Frame/**/VGrid[*].identifier=='None_grid_view'")
        file = await self.find_widget_with_retry("**/Label[*].text=='nv_ant.xml'", parent=grid_view)
        await file.click()
        await ui_test.human_delay()

        import_button = await self.find_widget_with_retry("Select File//Frame/VStack[0]/HStack[2]/Button[0]")
        await import_button.click()
        await ui_test.human_delay()

        for i in range(10):
            await omni.kit.app.get_app().next_update_async()

        stage = stage_utils.get_current_stage()

        # check if object is there
        prim = stage.GetPrimAtPath("/ant")
        prim.GetVariantSet("Physics").SetVariantSelection("mujoco")
        self.assertNotEqual(prim.GetPath(), Sdf.Path.emptyPath)

        # make sure the joints and links exist
        front_left_leg_joint = stage.GetPrimAtPath("/ant/Geometry/torso/front_left_leg/hip_1")
        self.assertNotEqual(front_left_leg_joint.GetPath(), Sdf.Path.emptyPath)
        self.assertEqual(front_left_leg_joint.GetTypeName(), "PhysicsRevoluteJoint")
        self.assertAlmostEqual(front_left_leg_joint.GetAttribute("physics:upperLimit").Get(), 40)
        self.assertAlmostEqual(front_left_leg_joint.GetAttribute("physics:lowerLimit").Get(), -40)

        front_left_leg = stage.GetPrimAtPath("/ant/Geometry/torso/front_left_leg")
        self.assertAlmostEqual(front_left_leg.GetAttribute("physics:diagonalInertia").Get()[0], 0.0)
        self.assertAlmostEqual(front_left_leg.GetAttribute("physics:mass").Get(), 0.0)

        actuator_1 = stage.GetPrimAtPath("/ant/Physics/Actuator_1")
        self.assertNotEqual(actuator_1.GetPath(), Sdf.Path.emptyPath)
        self.assertEqual(actuator_1.GetTypeName(), "MjcActuator")
        self.assertAlmostEqual(actuator_1.GetAttribute("mjc:gear").Get(), [15, 0, 0, 0, 0, 0])

        if delete_output_on_success:
            output_path = os.path.normpath(os.path.join(os.path.dirname(self._mjcf_path), "nv_ant"))
            stage = None
            gc.collect()
            try:
                shutil.rmtree(output_path)
            except OSError as e:
                carb.log_warn(f"Warning: unable to delete {output_path} : {e.strerror}")

    async def test_mjcf_ui_selections(self) -> None:
        """Update UI settings and validate importer config values.

        Example:

        .. code-block:: python

            >>> from isaacsim.asset.importer.mjcf import MJCFImporterConfig
            >>> MJCFImporterConfig()
            <...>
        """
        await self.menu_click_with_retry("File/Import")

        dir_widget = await self.find_widget_with_retry(
            "Select File//Frame/**/StringField[*].identifier=='filepicker_directory_path'"
        )

        await dir_widget.input(self._mjcf_path)
        await ui_test.human_delay()
        await omni.kit.app.get_app().next_update_async()

        grid_view = await self.find_widget_with_retry("Select File//Frame/**/VGrid[*].identifier=='None_grid_view'")
        file = await self.find_widget_with_retry("**/Label[*].text=='nv_ant.xml'", parent=grid_view)
        await file.click()
        await ui_test.human_delay()

        output_path_field = await self.find_widget_with_retry(
            "Select File//Frame/**/StringField[*].identifier=='mjcf_output_path'"
        )
        output_path_field.model.set_value(os.path.join(self._extension_path, "data", "mjcf", "temp/"))

        collision_from_visuals_btn = await self.find_widget_with_retry(
            "Select File//Frame/**/CheckBox[*].identifier=='mjcf_collision_from_visuals'"
        )
        await collision_from_visuals_btn.click()
        await ui_test.human_delay()

        collision_type_dropdown = await self.find_widget_with_retry(
            "Select File//Frame/**/ComboBox[*].identifier=='mjcf_collision_type'"
        )
        collision_type_dropdown.model.get_item_value_model(None, 0).set_value(2)
        await ui_test.human_delay()

        allow_self_collision_btn = await self.find_widget_with_retry(
            "Select File//Frame/**/CheckBox[*].identifier=='mjcf_allow_self_collision'"
        )
        await allow_self_collision_btn.click()
        await ui_test.human_delay()

        import_scene_btn = await self.find_widget_with_retry(
            "Select File//Frame/**/CheckBox[*].identifier=='mjcf_import_scene'"
        )
        await import_scene_btn.click()
        await ui_test.human_delay()

        merge_mesh_btn = await self.find_widget_with_retry(
            "Select File//Frame/**/CheckBox[*].identifier=='mjcf_merge_mesh'"
        )
        await merge_mesh_btn.click()
        await ui_test.human_delay()

        debug_mode_btn = await self.find_widget_with_retry(
            "Select File//Frame/**/CheckBox[*].identifier=='mjcf_debug_mode'"
        )
        await debug_mode_btn.click()
        await ui_test.human_delay()

        import_button = await self.find_widget_with_retry("Select File//Frame/VStack[0]/HStack[2]/Button[0]")
        await import_button.click()
        await ui_test.human_delay()

        for i in range(10):
            await omni.kit.app.get_app().next_update_async()

        extension = mjcf_ui_extension.get_instance()
        config = extension._get_config()
        self.assertEqual(
            os.path.normpath(config.mjcf_path.lower()),
            os.path.normpath(self._mjcf_path.lower()),
        )
        self.assertEqual(
            os.path.normpath(config.usd_path.lower()),
            os.path.normpath(os.path.join(self._extension_path, "data", "mjcf", "temp")).lower(),
        )
        self.assertEqual(config.import_scene, False)
        self.assertEqual(config.merge_mesh, True)
        self.assertEqual(config.debug_mode, True)
        self.assertEqual(config.collision_from_visuals, True)
        self.assertEqual(config.collision_type, "Bounding Sphere")
        self.assertEqual(config.allow_self_collision, True)

        await omni.kit.app.get_app().next_update_async()
        output_path = os.path.normpath(os.path.join(self._extension_path, "data", "mjcf", "temp"))
        gc.collect()
        try:
            shutil.rmtree(output_path)
        except OSError as e:
            carb.log_warn(f"Warning: unable to delete {output_path} : {e.strerror}")

    async def test_mjcf_ui_match_mjcf_importer(self) -> None:

        # UI workflow
        await self.test_import_ant_from_ui(delete_output_on_success=False)

        ui_mjcf_path = os.path.normpath(os.path.join(os.path.dirname(self._mjcf_path), "nv_ant", "nv_ant.usda"))
        # MJCF importer workflow
        config = mjcf_ui_extension.get_instance()._get_config()
        config.mjcf_path = self._mjcf_path
        config.usd_path = os.path.normpath(os.path.join(os.path.dirname(self._mjcf_path), "temp"))
        mjcf_importer = MJCFImporter(config)
        mjcf_importer_output_path = os.path.normpath(mjcf_importer.import_mjcf())

        carb.log_info(f"ui_mjcf_path: {ui_mjcf_path}")
        carb.log_info(f"mjcf_importer_output_path: {mjcf_importer_output_path}")

        result = await test_utils.compare_usd_files([ui_mjcf_path, mjcf_importer_output_path])
        self.assertTrue(result, "USD comparison failed")

        try:
            shutil.rmtree(os.path.normpath(os.path.dirname(ui_mjcf_path)))
            shutil.rmtree(os.path.normpath(os.path.dirname(mjcf_importer_output_path)))
        except OSError as e:
            carb.log_warn(f"Warning: unable to delete {os.path.normpath(os.path.dirname(ui_mjcf_path))} : {e.strerror}")
            carb.log_warn(
                f"Warning: unable to delete {os.path.normpath(os.path.dirname(mjcf_importer_output_path))} : {e.strerror}"
            )
