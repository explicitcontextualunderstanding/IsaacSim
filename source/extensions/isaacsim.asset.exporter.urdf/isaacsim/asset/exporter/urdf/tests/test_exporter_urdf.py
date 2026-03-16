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

# Standard library imports
import asyncio
import os
import tempfile
import unittest
from typing import Any

import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
import omni.kit.actions.core
import omni.kit.commands
import omni.kit.test
from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
from isaacsim.asset.importer.utils import test_utils
from isaacsim.storage.native import get_assets_root_path
from nvidia.srl.from_usd.to_urdf import UsdToUrdf


# Having a test class derived from omni.kit.test.AsyncTestCase declared on the root of module will make it auto-discoverable by omni.kit.test
class TestUrdfExporter(omni.kit.test.AsyncTestCase):
    # Before running each test
    async def setUp(self):
        self._timeline = omni.timeline.get_timeline_interface()
        await omni.usd.get_context().new_stage_async()
        await omni.kit.app.get_app().next_update_async()
        pass

    # After running each test
    async def tearDown(self):
        # Wait for any pending stage loading operations to complete
        while omni.usd.get_context().get_stage_loading_status()[2] > 0:
            print("tearDown, assets still loading, waiting to finish...")
            await asyncio.sleep(1.0)

        # Stop timeline if running
        if self._timeline and self._timeline.is_playing():
            self._timeline.stop()

        # Ensure app updates are processed
        # Create a new stage to release any previously open files
        await omni.kit.app.get_app().next_update_async()
        await omni.usd.get_context().new_stage_async()
        await omni.kit.app.get_app().next_update_async()

        pass

    @staticmethod
    def check(
        prims,
        member,
        member_args: list[tuple] | None = None,
        member_kwargs: list[dict] | None = None,
        msg: list[str] | None = None,
    ) -> bool:
        import itertools

        def show_mismatch(member_name: str, *, i: int, x: Any, j: int, y: Any) -> None:
            if msg is None:
                print(f"\n'{member_name}' mismatch")
            else:
                print(f"\n'{member_name}' mismatch for {msg[i]} and {msg[j]}")
            print(f"  - {x}")
            print(f"  - {y}")

        def check_values(x: Any, y: Any) -> tuple[bool, Any, Any]:
            # tuple
            if isinstance(x, tuple):
                status = True
                new_x, new_y = [], []
                for xx, yy in zip(x, y):
                    result, xx, yy = check_values(xx, yy)
                    new_x.append(xx)
                    new_y.append(yy)
                    if not result:
                        status = False
                        break
                return status, tuple(new_x), tuple(new_y)
            # Warp arrays
            elif hasattr(x, "numpy"):  # Check if it has numpy method (wp.array, torch.Tensor, etc.)
                try:
                    x_np = x.numpy()
                    y_np = y.numpy()
                    return np.allclose(x_np, y_np, rtol=1e-03, atol=1e-05), x_np, y_np
                except:
                    pass
            # generic Python types
            elif x != y:
                return False, x, y
            return True, x, y

        status = True

        # Handle both properties and methods
        if isinstance(member, property):
            member_name = member.fget.__name__

            results = [member.fget(prim) for prim in prims]
        else:
            member_name = member.__name__
            member_args = [()] * len(prims) if member_args is None else member_args
            member_kwargs = [{}] * len(prims) if member_kwargs is None else member_kwargs
            results = [member(prim, *member_args[i], **member_kwargs[i]) for i, prim in enumerate(prims)]
        for (i, x), (j, y) in itertools.combinations(enumerate(results), 2):
            result, x, y = check_values(x, y)
            if not result:
                # check if only base name is different
                if member_name == "link_names":
                    status = True
                    for i in range(1, len(x)):
                        if x[i] != y[i]:
                            status = False
                            show_mismatch(member_name, i=i, x=x, j=j, y=y)
                            break
                else:
                    status = False
                    show_mismatch(member_name, i=i, x=x, j=j, y=y)
        return status

    async def test_exporter_ur10e(self):
        """Test exporting the UR10e robot from USD to URDF and validate the exported URDF"""
        assets_root_path = get_assets_root_path()[: len(get_assets_root_path())] + "/"
        robot_path = "Isaac/Robots/UniversalRobots/ur10e/ur10e.usd"
        robot_path = os.path.join(assets_root_path, robot_path)
        await stage_utils.open_stage_async(robot_path)
        stage = stage_utils.get_current_stage()

        # Create a temporary directory for test output
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:

            usd_to_urdf_kwargs = {
                "node_names_to_remove": None,
                "edge_names_to_remove": None,
                "root": "ur10e",
                "parent_link_is_body_1": None,
                "log_level": "INFO",
            }

            # Create UsdToUrdf object
            with self.assertRaises(Exception, msg="Expected failure: joint transforms inconsistent"):
                usd_to_urdf = UsdToUrdf(stage, **usd_to_urdf_kwargs)

            # Ensure stage is cleared before temp directory cleanup
            await stage_utils.create_new_stage_async()
            await omni.kit.app.get_app().next_update_async()

    async def test_exporter_2f_140_base(self):
        """Test exporting the 2F-140 base robot from USD to URDF and validate the exported URDF"""
        assets_root_path = get_assets_root_path()[: len(get_assets_root_path())] + "/"
        robot_path = "Isaac/Robots/Robotiq/2F-140/Robotiq_2F_140_base.usd"
        robot_path = os.path.join(assets_root_path, robot_path)
        await stage_utils.open_stage_async(robot_path)
        stage = stage_utils.get_current_stage()

        # Create a temporary directory for test output
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:

            usd_to_urdf_kwargs = {
                "node_names_to_remove": None,
                "edge_names_to_remove": None,
                "root": "Robotiq_2F_140",
                "parent_link_is_body_1": None,
                "log_level": "INFO",
            }

            # Create UsdToUrdf object
            with self.assertRaises(Exception, msg="Expected failure: kinematic loops found"):
                usd_to_urdf = UsdToUrdf(stage, **usd_to_urdf_kwargs)

            # Ensure stage is cleared before temp directory cleanup
            await stage_utils.create_new_stage_async()
            await omni.kit.app.get_app().next_update_async()

    @unittest.skip("urdf converter bug")
    async def test_exporter_nova_carter(self):
        """Test exporting the NovaCarter robot from USD to URDF and validate the exported URDF"""

        assets_root_path = get_assets_root_path()[: len(get_assets_root_path())] + "/"
        robot_path = "Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd"
        robot_path = os.path.join(assets_root_path, robot_path)
        await stage_utils.open_stage_async(robot_path)
        stage = stage_utils.get_current_stage()

        # Create a temporary directory for test output
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            temp_dir = "/home/stevfeng/Desktop"
            # Test exporting to URDF
            usd_to_check = os.path.normpath(os.path.join(temp_dir, f"nova_carter_original.usd"))

            usd_to_urdf_kwargs = {
                "node_names_to_remove": None,
                "edge_names_to_remove": None,
                "root": "nova_carter",
                "parent_link_is_body_1": None,
                "log_level": "INFO",
            }

            # Create UsdToUrdf object
            usd_to_urdf = UsdToUrdf(stage, **usd_to_urdf_kwargs)

            # Export to URDF
            output_urdf_path = os.path.normpath(os.path.join(temp_dir, f"nova_carter_exported.urdf"))
            output_path = usd_to_urdf.save_to_file(
                urdf_output_path=output_urdf_path,
                visualize_collision_meshes=False,
                mesh_dir=temp_dir,
                mesh_path_prefix="./",
                use_uri_file_prefix=True,
            )
            stage_utils.save_stage(os.path.normpath(usd_to_check))
            self.assertTrue(os.path.exists(output_urdf_path), f"Exported URDF file not found for nova_carter")

            await stage_utils.create_new_stage_async()

            import_config = URDFImporterConfig()
            import_config.urdf_path = output_path
            importer = URDFImporter(import_config)
            urdf_to_check = os.path.normpath(importer.import_urdf())
            print(f"reimported usd path: {urdf_to_check}")
            print(f"original usd path: {usd_to_check}")
            await stage_utils.create_new_stage_async()

            comparison_result = await test_utils.compare_usd_files([usd_to_check, urdf_to_check])
            self.assertTrue(comparison_result, "USD comparison failed")

            # Ensure stage is cleared before temp directory cleanup
            await stage_utils.create_new_stage_async()
            await omni.kit.app.get_app().next_update_async()

        return

    async def test_exporter_tien_kung(self):
        """Test exporting the TienKung robot from USD to URDF and validate the exported URDF"""
        assets_root_path = get_assets_root_path()[: len(get_assets_root_path())] + "/"
        robot_path = "Isaac/Robots/XHumanoid/Tien Kung/tienkung.usd"
        robot_path = os.path.join(assets_root_path, robot_path)

        await stage_utils.open_stage_async(robot_path)
        stage = stage_utils.get_current_stage()
        # Create a temporary directory for test output
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:

            usd_to_urdf_kwargs = {
                "node_names_to_remove": None,
                "edge_names_to_remove": None,
                "root": "humanoid",
                "parent_link_is_body_1": None,
                "log_level": "INFO",
            }

            # Create UsdToUrdf object
            with self.assertRaises(Exception, msg="Expected failure: joint transforms inconsistent"):
                usd_to_urdf = UsdToUrdf(stage, **usd_to_urdf_kwargs)

            # Ensure stage is cleared before temp directory cleanup
            await stage_utils.create_new_stage_async()
            await omni.kit.app.get_app().next_update_async()

    @unittest.skip("urdf converter bug")
    async def test_exporter_unitree_go2(self):
        """Test exporting the Unitree Go2 robot from USD to URDF and validate the exported URDF"""
        assets_root_path = get_assets_root_path()[: len(get_assets_root_path())] + "/"
        robot_path = "Isaac/Robots/Unitree/Go2/go2.usd"
        robot_path = os.path.join(assets_root_path, robot_path)

        await stage_utils.open_stage_async(robot_path)
        stage = stage_utils.get_current_stage()

        # Create a temporary directory for test output
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            usd_to_check = os.path.join(temp_dir, f"unitree_go2_original.usd")

            usd_to_urdf_kwargs = {
                "node_names_to_remove": None,
                "edge_names_to_remove": None,
                "root": "go2_description",
                "parent_link_is_body_1": None,
                "log_level": "INFO",
            }

            # Create UsdToUrdf object
            usd_to_urdf = UsdToUrdf(stage, **usd_to_urdf_kwargs)

            # Export to URDF
            output_urdf_path = os.path.join(temp_dir, f"unitree_go2_exported.urdf")
            output_path = usd_to_urdf.save_to_file(
                urdf_output_path=output_urdf_path,
                visualize_collision_meshes=False,
                mesh_dir=temp_dir,
                mesh_path_prefix="./",
                use_uri_file_prefix=True,
            )
            stage_utils.save_stage(usd_to_check)
            self.assertTrue(os.path.exists(output_urdf_path), f"Exported URDF file not found for go2")

            await stage_utils.create_new_stage_async()

            import_config = URDFImporterConfig()
            import_config.urdf_path = output_path

            importer = URDFImporter(import_config)
            urdf_to_check = importer.import_urdf()

            await stage_utils.create_new_stage_async()

            comparison_result = await test_utils.compare_usd_files([usd_to_check, urdf_to_check])
            self.assertTrue(comparison_result, "USD comparison failed")

            # Ensure stage is cleared before temp directory cleanup
            await stage_utils.create_new_stage_async()
            await omni.kit.app.get_app().next_update_async()

        return
