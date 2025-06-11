# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import isaacsim.core.experimental.utils.backend as backend_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import omni.kit.stage_templates
import omni.kit.test
import omni.usd
import usdrt
from isaacsim.storage.native import get_assets_root_path_async
from pxr import Usd, UsdGeom, UsdLux, UsdPhysics, UsdUtils


class TestStage(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        """Method called to prepare the test fixture"""
        super().setUp()
        # ---------------
        # Warning: don't create stage in the setUp method since we test the stage creation
        # ---------------

    async def tearDown(self):
        """Method called immediately after the test method has been called"""
        # ------------------
        # Do custom tearDown
        # ------------------
        super().tearDown()

    # --------------------------------------------------------------------

    async def test_context_manager(self):
        await stage_utils.create_new_stage_async()
        stage_in_memory = Usd.Stage.CreateInMemory()
        default_stage = omni.usd.get_context().get_stage()
        default_stage_id = UsdUtils.StageCache.Get().GetId(default_stage).ToLongInt()
        self.assertIs(stage_utils.get_current_stage(), default_stage)
        self.assertFalse(stage_utils.is_stage_set())
        self.assertEqual(stage_utils.get_stage_id(default_stage), default_stage_id)
        # - USD stage
        with stage_utils.use_stage(stage_in_memory):
            self.assertIs(stage_utils.get_current_stage(), stage_in_memory)
            self.assertTrue(stage_utils.is_stage_set())
            self.assertIsNot(stage_utils.get_current_stage(), default_stage)
            self.assertNotEqual(stage_utils.get_stage_id(stage_utils.get_current_stage()), default_stage_id)
        self.assertIs(stage_utils.get_current_stage(), default_stage)
        self.assertFalse(stage_utils.is_stage_set())
        self.assertEqual(stage_utils.get_stage_id(stage_utils.get_current_stage()), default_stage_id)
        # - USDRT/Fabric stage
        # -- via function argument
        with stage_utils.use_stage(stage_in_memory):
            self.assertIsInstance(stage_utils.get_current_stage(backend="usdrt"), usdrt.Usd.Stage)
            self.assertIsInstance(stage_utils.get_current_stage(backend="fabric"), usdrt.Usd.Stage)
        self.assertIsInstance(stage_utils.get_current_stage(backend="usdrt"), usdrt.Usd.Stage)
        self.assertIsInstance(stage_utils.get_current_stage(backend="fabric"), usdrt.Usd.Stage)
        # -- via context manager
        with backend_utils.use_backend("usdrt"):
            with stage_utils.use_stage(stage_in_memory):
                self.assertIsInstance(stage_utils.get_current_stage(), usdrt.Usd.Stage)
            self.assertIsInstance(stage_utils.get_current_stage(), usdrt.Usd.Stage)
        with backend_utils.use_backend("fabric"):
            with stage_utils.use_stage(stage_in_memory):
                self.assertIsInstance(stage_utils.get_current_stage(), usdrt.Usd.Stage)
            self.assertIsInstance(stage_utils.get_current_stage(), usdrt.Usd.Stage)

    async def test_create_new_stage(self):
        templates = sorted([name for item in omni.kit.stage_templates.get_stage_template_list() for name in item])
        self.assertEqual(templates, ["default stage", "empty", "sunlight"], f"Available templates: {templates}")
        # test cases
        # - sync
        for template in [None] + templates:
            stage = stage_utils.create_new_stage(template=template)
            self.assertIsInstance(stage, Usd.Stage)
            self.assertIs(stage, stage_utils.get_current_stage())
            self.assertEqual(
                stage.GetPrimAtPath("/World").IsValid(),
                template is not None,
                f"Invalid stage content for the given template: {template}",
            )
        # - async
        for template in [None] + templates:
            stage = await stage_utils.create_new_stage_async(template=template)
            self.assertIsInstance(stage, Usd.Stage)
            self.assertIs(stage, stage_utils.get_current_stage())
            self.assertEqual(
                stage.GetPrimAtPath("/World").IsValid(),
                template is not None,
                f"Invalid stage content for the given template: {template}",
            )

    async def test_open_stage(self):
        assets_root_path = await get_assets_root_path_async()
        assert assets_root_path is not None, "Could not find Isaac Sim's root assets path"
        # test cases
        # - sync
        (result, stage) = stage_utils.open_stage(
            usd_path=assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
        )
        self.assertTrue(result, "Failed to open stage")
        prim = stage.GetPrimAtPath("/panda")
        self.assertIsInstance(prim, Usd.Prim)
        self.assertEqual(prim.GetPath(), "/panda")
        # - async
        await stage_utils.create_new_stage_async()
        (result, stage) = await stage_utils.open_stage_async(
            usd_path=assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
        )
        self.assertTrue(result, "Failed to open stage")
        prim = stage.GetPrimAtPath("/panda")
        self.assertIsInstance(prim, Usd.Prim)
        self.assertEqual(prim.GetPath(), "/panda")

    async def test_add_reference_to_stage(self):
        assets_root_path = await get_assets_root_path_async()
        assert assets_root_path is not None, "Could not find Isaac Sim's root assets path"

        await stage_utils.create_new_stage_async()
        prim = stage_utils.add_reference_to_stage(
            usd_path=assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
            path="/World/panda",
            variants=[("Gripper", "AlternateFinger"), ("Mesh", "Performance")],
        )
        self.assertIsInstance(prim, Usd.Prim)
        self.assertEqual(prim.GetPath(), "/World/panda")
        self.assertEqual(prim.GetVariantSet("Gripper").GetVariantSelection(), "AlternateFinger")
        self.assertEqual(prim.GetVariantSet("Mesh").GetVariantSelection(), "Performance")

    async def test_define_prim(self):
        await stage_utils.create_new_stage_async()
        specs = [
            # UsdGeomTokensType
            ("Camera", UsdGeom.Camera),
            ("Capsule", UsdGeom.Capsule),
            ("Cone", UsdGeom.Cone),
            ("Cube", UsdGeom.Cube),
            ("Cylinder", UsdGeom.Cylinder),
            ("Mesh", UsdGeom.Mesh),
            ("Plane", UsdGeom.Plane),
            ("Points", UsdGeom.Points),
            ("Sphere", UsdGeom.Sphere),
            ("Xform", UsdGeom.Xform),
            # UsdLuxTokensType
            ("CylinderLight", UsdLux.CylinderLight),
            ("DiskLight", UsdLux.DiskLight),
            ("DistantLight", UsdLux.DistantLight),
            ("DomeLight", UsdLux.DomeLight),
            ("RectLight", UsdLux.RectLight),
            ("SphereLight", UsdLux.SphereLight),
            # UsdPhysicsTokensType
            ("PhysicsScene", UsdPhysics.Scene),
        ]
        # USD prim
        for token, prim_type in specs:
            prim = stage_utils.define_prim(f"/{token}", type_name=token)
            self.assertTrue(prim.IsA(prim_type), f"Prim ({prim.GetPath()}) is not a {prim_type}")
        # USDRT prim
        with backend_utils.use_backend("usdrt"):
            for token, _ in specs:
                prim = stage_utils.define_prim(f"/{token}", type_name=token)
                self.assertIsInstance(prim, usdrt.Usd.Prim, f"Prim ({prim.GetPath()}) is not a USDRT prim")
        # exceptions
        # - non-absolute path
        self.assertRaises(ValueError, stage_utils.define_prim, f"World")
        # - non-valid path
        self.assertRaises(ValueError, stage_utils.define_prim, f"/World/")
        # - prim already exists with a different type
        self.assertRaises(RuntimeError, stage_utils.define_prim, f"/Sphere", type_name="Cube")
