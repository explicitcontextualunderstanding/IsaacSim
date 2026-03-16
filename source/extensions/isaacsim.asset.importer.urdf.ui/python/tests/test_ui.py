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


"""UI tests for the URDF importer extension."""

import asyncio
import gc
import os
import shutil

import carb
import omni.kit.commands

# NOTE:
#   omni.kit.test - std python's unittest module with additional wrapping to add suport for async/await tests
#   For most things refer to unittest docs: https://docs.python.org/3/library/unittest.html
import omni.kit.test
import omni.kit.ui_test as ui_test
from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
from isaacsim.asset.importer.urdf.ui.impl import extension as urdf_ui_extension
from isaacsim.asset.importer.utils import test_utils
from isaacsim.core.experimental.utils import stage as stage_utils
from isaacsim.test.utils import MenuUITestCase
from pxr import Sdf


# Having a test class dervived from omni.kit.test.AsyncTestCase declared on the root of module will make it auto-discoverable by omni.kit.test
class TestImporterUI(MenuUITestCase):
    """Test URDF importer UI interactions.

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
        ext_id = ext_manager.get_enabled_extension_id("isaacsim.asset.importer.urdf.ui")
        urdf_ext_id = ext_manager.get_enabled_extension_id("isaacsim.asset.importer.urdf")
        self._extension_path = ext_manager.get_extension_path(ext_id)
        self._urdf_extension_path = ext_manager.get_extension_path(urdf_ext_id)
        self._urdf_path = os.path.normpath(
            os.path.join(self._urdf_extension_path, "data", "urdf", "robots", "ur10", "urdf", "ur10.urdf")
        )

        await omni.usd.get_context().new_stage_async()
        await omni.kit.app.get_app().next_update_async()
        self._stage = omni.usd.get_context().get_stage()

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

    async def test_import_ur10_from_ui(self, delete_output_on_success=True) -> None:
        """Import the ant asset via the UI and validate output.

        Example:

        .. code-block:: python

            >>> from isaacsim.asset.importer.urdf import URDFImporterConfig
            >>> URDFImporterConfig()
            <...>
        """
        await self.menu_click_with_retry("File/Import")

        dir_widget = await self.find_widget_with_retry(
            "Select File//Frame/**/StringField[*].identifier=='filepicker_directory_path'"
        )

        await dir_widget.input(self._urdf_path)
        await ui_test.human_delay()
        await omni.kit.app.get_app().next_update_async()

        grid_view = await self.find_widget_with_retry("Select File//Frame/**/VGrid[*].identifier=='None_grid_view'")
        file = await self.find_widget_with_retry("**/Label[*].text=='ur10.urdf'", parent=grid_view)
        await file.click()
        await ui_test.human_delay()

        import_button = await self.find_widget_with_retry("Select File//Frame/VStack[0]/HStack[2]/Button[0]")
        await import_button.click()
        await ui_test.human_delay()

        for i in range(10):
            await omni.kit.app.get_app().next_update_async()

        self._stage = stage_utils.get_current_stage()

        # check if object is there
        prim = self._stage.GetPrimAtPath("/ur10")
        prim.GetVariantSet("Physics").SetVariantSelection("physx")
        self.assertNotEqual(prim.GetPath(), Sdf.Path.emptyPath)

        # make sure the joints and links exist
        shoulder_pan_joint = self._stage.GetPrimAtPath("/ur10/Physics/shoulder_pan_joint")
        self.assertNotEqual(shoulder_pan_joint.GetPath(), Sdf.Path.emptyPath)
        self.assertEqual(shoulder_pan_joint.GetTypeName(), "PhysicsRevoluteJoint")
        self.assertAlmostEqual(shoulder_pan_joint.GetAttribute("physics:upperLimit").Get(), 360.0, delta=1e-2)
        self.assertAlmostEqual(shoulder_pan_joint.GetAttribute("physics:lowerLimit").Get(), -360.0, delta=1e-2)

        shoulder_link = self._stage.GetPrimAtPath("/ur10/Geometry/base_link/shoulder_link")
        self.assertAlmostEqual(
            shoulder_link.GetAttribute("physics:diagonalInertia").Get()[0], 0.03147431257693659, delta=1e-2
        )
        self.assertAlmostEqual(shoulder_link.GetAttribute("physics:mass").Get(), 7.778, delta=1e-2)

        await omni.kit.app.get_app().next_update_async()
        if delete_output_on_success:
            output_path = os.path.normpath(os.path.join(os.path.dirname(self._urdf_path), "ur10"))
            self._stage = None
            gc.collect()
            try:
                shutil.rmtree(output_path)
            except OSError as e:
                carb.log_warn(f"Warning: {output_path} : {e.strerror}")

    async def test_urdf_ui_selections(self) -> None:
        """Update UI settings and validate importer config values.

        Example:

        .. code-block:: python

            >>> from isaacsim.asset.importer.urdf import URDFImporterConfig
            >>> URDFImporterConfig()
            <...>
        """
        await self.menu_click_with_retry("File/Import")

        dir_widget = await self.find_widget_with_retry(
            "Select File//Frame/**/StringField[*].identifier=='filepicker_directory_path'"
        )

        await dir_widget.input(self._urdf_path)
        await ui_test.human_delay()
        await omni.kit.app.get_app().next_update_async()

        grid_view = await self.find_widget_with_retry("Select File//Frame/**/VGrid[*].identifier=='None_grid_view'")
        file = await self.find_widget_with_retry("**/Label[*].text=='ur10.urdf'", parent=grid_view)
        await file.click()
        await ui_test.human_delay()

        output_path_field = await self.find_widget_with_retry(
            "Select File//Frame/**/StringField[*].identifier=='urdf_output_path'"
        )
        output_path_field.model.set_value(os.path.join(self._extension_path, "data", "urdf", "temp/"))

        ros_package_table_name_field = await self.find_widget_with_retry(
            "Select File//Frame/**/StringField[*].identifier=='ros_package_table_name_field_0'"
        )
        ros_package_table_name_field.model.set_value("test_package")
        await ui_test.human_delay()

        ros_package_table_path_field = await self.find_widget_with_retry(
            "Select File//Frame/**/StringField[*].identifier=='ros_package_table_path_field_0'"
        )
        ros_package_table_path_field.model.set_value("test_path")
        await ui_test.human_delay()

        add_ros_package_row_btn = await self.find_widget_with_retry(
            "Select File//Frame/**/Button[*].identifier=='urdf_add_ros_package_row'"
        )
        await add_ros_package_row_btn.click()
        await ui_test.human_delay()

        ros_package_table_name_field = await self.find_widget_with_retry(
            "Select File//Frame/**/StringField[*].identifier=='ros_package_table_name_field_1'"
        )
        ros_package_table_name_field.model.set_value("test_package_row_1")
        await ui_test.human_delay()

        ros_package_table_path_field = await self.find_widget_with_retry(
            "Select File//Frame/**/StringField[*].identifier=='ros_package_table_path_field_1'"
        )
        ros_package_table_path_field.model.set_value("test_path_row_1")
        await ui_test.human_delay()

        collision_from_visuals_btn = await self.find_widget_with_retry(
            "Select File//Frame/**/CheckBox[*].identifier=='urdf_collision_from_visuals'"
        )
        await collision_from_visuals_btn.click()
        await ui_test.human_delay()

        collision_type_dropdown = await self.find_widget_with_retry(
            "Select File//Frame/**/ComboBox[*].identifier=='urdf_collision_type'"
        )
        collision_type_dropdown.model.get_item_value_model(None, 0).set_value(2)
        await ui_test.human_delay()

        allow_self_collision_btn = await self.find_widget_with_retry(
            "Select File//Frame/**/CheckBox[*].identifier=='urdf_allow_self_collision'"
        )
        await allow_self_collision_btn.click()
        await ui_test.human_delay()

        merge_mesh_btn = await self.find_widget_with_retry(
            "Select File//Frame/**/CheckBox[*].identifier=='urdf_merge_mesh'"
        )
        await merge_mesh_btn.click()
        await ui_test.human_delay()

        debug_mode_btn = await self.find_widget_with_retry(
            "Select File//Frame/**/CheckBox[*].identifier=='urdf_debug_mode'"
        )
        await debug_mode_btn.click()
        await ui_test.human_delay()

        import_button = await self.find_widget_with_retry("Select File//Frame/VStack[0]/HStack[2]/Button[0]")
        await import_button.click()
        await ui_test.human_delay()

        for i in range(10):
            await omni.kit.app.get_app().next_update_async()

        extension = urdf_ui_extension.get_instance()
        config = extension._get_config()
        self.assertEqual(os.path.normpath(config.urdf_path.lower()), os.path.normpath(self._urdf_path.lower()))
        self.assertEqual(
            os.path.normpath(config.usd_path.lower()),
            os.path.normpath(os.path.join(self._extension_path, "data", "urdf", "temp")).lower(),
        )
        self.assertEqual(config.merge_mesh, True)
        self.assertEqual(config.debug_mode, True)
        self.assertEqual(config.collision_from_visuals, True)
        self.assertEqual(config.collision_type, "Bounding Sphere")
        self.assertEqual(config.allow_self_collision, True)
        self.assertEqual(
            config.ros_package_paths,
            [{"name": "test_package", "path": "test_path"}, {"name": "test_package_row_1", "path": "test_path_row_1"}],
        )

        await omni.kit.app.get_app().next_update_async()
        output_path = os.path.normpath(os.path.join(self._extension_path, "data", "urdf", "temp"))
        self._stage = None
        gc.collect()
        try:
            shutil.rmtree(output_path)
        except OSError as e:
            carb.log_warn(f"Warning: {output_path} : {e.strerror}")

    async def test_urdf_ui_match_urdf_importer(self) -> None:

        # UI workflow
        await self.test_import_ur10_from_ui(delete_output_on_success=False)

        ui_ur10_path = os.path.normpath(os.path.join(os.path.dirname(self._urdf_path), "ur10", "ur10.usda"))
        # URDF importer workflow
        config = URDFImporterConfig()
        config.urdf_path = self._urdf_path
        config.usd_path = os.path.normpath(os.path.join(self._extension_path, "data", "urdf", "temp"))
        urdf_importer = URDFImporter(config)
        urdf_importer_output_path = os.path.normpath(urdf_importer.import_urdf())
        print(f"urdf_importer_output_path: {urdf_importer_output_path}")
        print(f"ui_ur10_path: {ui_ur10_path}")
        result = await test_utils.compare_usd_files([ui_ur10_path, urdf_importer_output_path])
        self.assertTrue(result, "USD comparison failed")

        await omni.kit.app.get_app().next_update_async()
        output_path = os.path.normpath(os.path.join(self._extension_path, "data", "urdf", "temp"))
        self._stage = None
        gc.collect()
        try:
            shutil.rmtree(output_path)
            shutil.rmtree(os.path.normpath(os.path.dirname(urdf_importer_output_path)))
        except OSError as e:
            carb.log_warn(f"Warning: {output_path} : {e.strerror}")
            carb.log_warn(f"Warning: {os.path.normpath(os.path.dirname(urdf_importer_output_path))} : {e.strerror}")
