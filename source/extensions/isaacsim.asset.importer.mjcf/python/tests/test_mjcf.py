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


"""Tests for MJCF importer conversion results."""

import asyncio
import gc
import os
import shutil

import carb

# NOTE:
#   omni.kit.test - std python's unittest module with additional wrapping to add suport for async/await tests
#   For most things refer to unittest docs: https://docs.python.org/3/library/unittest.html
import omni.kit.test
from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig
from isaacsim.asset.importer.utils.impl import stage_utils
from pxr import Sdf, Usd, UsdGeom, UsdPhysics


# Having a test class dervived from omni.kit.test.AsyncTestCase declared on the root of module will make it auto-discoverable by omni.kit.test
class TestMJCF(omni.kit.test.AsyncTestCase):
    """Test MJCF importer conversion workflows.

    Example:

    .. code-block:: python

        >>> import omni.kit.test
        >>> class Example(omni.kit.test.AsyncTestCase):
        ...     pass
        ...
    """

    # Before running each test
    async def setUp(self) -> None:
        """Prepare shared test fixtures.

        Example:

        .. code-block:: python

            >>> import omni.timeline
            >>> omni.timeline.get_timeline_interface()  # doctest: +SKIP
        """
        self._timeline = omni.timeline.get_timeline_interface()

        ext_manager = omni.kit.app.get_app().get_extension_manager()
        ext_id = ext_manager.get_enabled_extension_id("isaacsim.asset.importer.mjcf")
        self._extension_path = ext_manager.get_extension_path(ext_id)
        self.importer = MJCFImporter()
        await omni.usd.get_context().new_stage_async()
        await omni.kit.app.get_app().next_update_async()

    # After running each test
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
        # await omni.usd.get_context().new_stage_async()

    async def test_mjcf_ant(self) -> None:
        """Import the ant MJCF and validate key outputs.

        Example:

        .. code-block:: python

            >>> from isaacsim.asset.importer.mjcf import MJCFImporterConfig
            >>> MJCFImporterConfig()
            <...>
        """
        config = MJCFImporterConfig(
            mjcf_path=os.path.normpath(os.path.join(self._extension_path, "data", "mjcf", "nv_ant.xml"))
        )
        self.importer.config = config

        output_path = self.importer.import_mjcf()
        stage = stage_utils.open_stage(output_path)

        self.assertIsNotNone(stage, "Failed to open stage at path: {output_path}")
        await omni.kit.app.get_app().next_update_async()

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

        front_left_foot_joint = stage.GetPrimAtPath("/ant/Geometry/torso/front_left_leg/front_left_foot/ankle_1")
        self.assertNotEqual(front_left_foot_joint.GetPath(), Sdf.Path.emptyPath)
        self.assertEqual(front_left_foot_joint.GetTypeName(), "PhysicsRevoluteJoint")
        self.assertAlmostEqual(front_left_foot_joint.GetAttribute("physics:upperLimit").Get(), 100)
        self.assertAlmostEqual(front_left_foot_joint.GetAttribute("physics:lowerLimit").Get(), 30)

        front_left_foot = stage.GetPrimAtPath("/ant/Geometry/torso/front_left_leg/front_left_foot")
        self.assertAlmostEqual(front_left_foot.GetAttribute("physics:diagonalInertia").Get()[0], 0.0)
        self.assertAlmostEqual(front_left_foot.GetAttribute("physics:mass").Get(), 0.0)

        actuator_0 = stage.GetPrimAtPath("/ant/Physics/Actuator_0")
        self.assertNotEqual(actuator_0.GetPath(), Sdf.Path.emptyPath)
        self.assertEqual(actuator_0.GetTypeName(), "MjcActuator")
        self.assertAlmostEqual(actuator_0.GetAttribute("mjc:gear").Get(), [15, 0, 0, 0, 0, 0])

        actuator_1 = stage.GetPrimAtPath("/ant/Physics/Actuator_1")
        self.assertNotEqual(actuator_1.GetPath(), Sdf.Path.emptyPath)
        self.assertEqual(actuator_1.GetTypeName(), "MjcActuator")
        self.assertAlmostEqual(actuator_1.GetAttribute("mjc:gear").Get(), [15, 0, 0, 0, 0, 0])

        # Start Simulation and wait
        self._timeline.play()
        await omni.kit.app.get_app().next_update_async()
        await asyncio.sleep(1.0)
        # nothing crashes
        self._timeline.stop()
        self.assertAlmostEqual(UsdGeom.GetStageMetersPerUnit(stage), 1.0)

        stage = None
        gc.collect()
        try:
            shutil.rmtree(os.path.normpath(os.path.dirname(output_path)))
        except OSError as e:
            carb.log_error(f"Error: {os.path.normpath(os.path.dirname(output_path))} : {e.strerror}")

    async def test_mjcf_humanoid(self) -> None:
        """Import the humanoid MJCF and validate key outputs.

        Example:

        .. code-block:: python

            >>> from isaacsim.asset.importer.mjcf import MJCFImporterConfig
            >>> MJCFImporterConfig()
            <...>
        """
        config = MJCFImporterConfig(
            mjcf_path=os.path.normpath(os.path.join(self._extension_path, "data", "mjcf", "nv_humanoid.xml"))
        )
        self.importer.config = config

        output_path = self.importer.import_mjcf()
        stage = stage_utils.open_stage(output_path)

        self.assertIsNotNone(stage, "Failed to open stage at path: {output_path}")
        await omni.kit.app.get_app().next_update_async()

        # check if object is there
        prim = stage.GetPrimAtPath("/humanoid")
        prim.GetVariantSet("Physics").SetVariantSelection("mujoco")
        self.assertNotEqual(prim.GetPath(), Sdf.Path.emptyPath)

        # make sure the joints and link exist
        pelvis_joint = stage.GetPrimAtPath("/humanoid/Geometry/torso/lower_waist/pelvis/abdomen_x")
        self.assertNotEqual(pelvis_joint.GetPath(), Sdf.Path.emptyPath)
        self.assertEqual(pelvis_joint.GetTypeName(), "PhysicsRevoluteJoint")
        self.assertAlmostEqual(pelvis_joint.GetAttribute("physics:upperLimit").Get(), 35)
        self.assertAlmostEqual(pelvis_joint.GetAttribute("physics:lowerLimit").Get(), -35)

        abdomen_y = stage.GetPrimAtPath("/humanoid/Geometry/torso/lower_waist/abdomen_y")
        self.assertNotEqual(abdomen_y.GetPath(), Sdf.Path.emptyPath)
        self.assertEqual(abdomen_y.GetTypeName(), "PhysicsRevoluteJoint")
        self.assertAlmostEqual(abdomen_y.GetAttribute("physics:upperLimit").Get(), 30)
        self.assertAlmostEqual(abdomen_y.GetAttribute("physics:lowerLimit").Get(), -75)

        abdomen_z = stage.GetPrimAtPath("/humanoid/Geometry/torso/lower_waist/abdomen_z")
        self.assertNotEqual(abdomen_z.GetPath(), Sdf.Path.emptyPath)
        self.assertEqual(abdomen_z.GetTypeName(), "PhysicsRevoluteJoint")
        self.assertAlmostEqual(abdomen_z.GetAttribute("physics:upperLimit").Get(), 45)
        self.assertAlmostEqual(abdomen_z.GetAttribute("physics:lowerLimit").Get(), -45)

        abdomen_x_actuator = stage.GetPrimAtPath("/humanoid/Physics/abdomen_x")
        self.assertNotEqual(abdomen_x_actuator.GetPath(), Sdf.Path.emptyPath)
        self.assertEqual(abdomen_x_actuator.GetTypeName(), "MjcActuator")
        self.assertAlmostEqual(abdomen_x_actuator.GetAttribute("mjc:gear").Get(), [67.5, 0, 0, 0, 0, 0])

        abdomen_y_actuator = stage.GetPrimAtPath("/humanoid/Physics/abdomen_y")
        self.assertNotEqual(abdomen_y_actuator.GetPath(), Sdf.Path.emptyPath)
        self.assertEqual(abdomen_y_actuator.GetTypeName(), "MjcActuator")
        self.assertAlmostEqual(abdomen_y_actuator.GetAttribute("mjc:gear").Get(), [67.5, 0, 0, 0, 0, 0])

        abdomen_z_actuator = stage.GetPrimAtPath("/humanoid/Physics/abdomen_z")
        self.assertNotEqual(abdomen_z_actuator.GetPath(), Sdf.Path.emptyPath)
        self.assertEqual(abdomen_z_actuator.GetTypeName(), "MjcActuator")
        self.assertAlmostEqual(abdomen_z_actuator.GetAttribute("mjc:gear").Get(), [67.5, 0, 0, 0, 0, 0])

        left_foot = stage.GetPrimAtPath("/humanoid/Geometry/torso/lower_waist/pelvis/left_thigh/left_shin/left_foot")
        self.assertAlmostEqual(left_foot.GetAttribute("physics:diagonalInertia").Get()[0], 0.0)
        self.assertAlmostEqual(left_foot.GetAttribute("physics:mass").Get(), 0.0)

        # Start Simulation and wait
        self._timeline.play()
        await omni.kit.app.get_app().next_update_async()
        await asyncio.sleep(1.0)
        # nothing crashes
        self._timeline.stop()

        self.assertAlmostEqual(UsdGeom.GetStageMetersPerUnit(stage), 1.0)

        stage = None
        gc.collect()
        try:
            shutil.rmtree(os.path.normpath(os.path.dirname(output_path)))
        except OSError as e:
            carb.log_error(f"Error: {os.path.normpath(os.path.dirname(output_path))} : {e.strerror}")

    # This sample corresponds to the example in the docs, keep this and the version in the docs in sync
    async def test_doc_sample(self) -> None:
        """Validate the doc sample pipeline on an MJCF asset.

        Example:

        .. code-block:: python

            >>> from isaacsim.asset.importer.mjcf import MJCFImporterConfig
            >>> MJCFImporterConfig()
            <...>
        """
        from pxr import Gf, Sdf, UsdLux, UsdPhysics

        # Get path to extension data:
        ext_manager = omni.kit.app.get_app().get_extension_manager()
        ext_id = ext_manager.get_enabled_extension_id("isaacsim.asset.importer.mjcf")
        extension_path = ext_manager.get_extension_path(ext_id)

        # import MJCF
        config = MJCFImporterConfig(
            mjcf_path=os.path.normpath(os.path.join(extension_path, "data", "mjcf", "nv_ant.xml"))
        )
        self.importer.config = config
        output_path = self.importer.import_mjcf()
        stage = stage_utils.open_stage(output_path)
        self.assertIsNotNone(stage, "Failed to open stage at path: {output_path}")

        # enable physics
        scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))
        # set gravity
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(9.81)

        # add lighting
        distantLight = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
        distantLight.CreateIntensityAttr(500)

        stage = None
        gc.collect()
        try:
            shutil.rmtree(os.path.normpath(os.path.dirname(output_path)))
        except OSError as e:
            carb.log_error(f"Error: {os.path.normpath(os.path.dirname(output_path))} : {e.strerror}")

    async def test_mjcf_self_collision(self) -> None:
        """Import with self-collision enabled and validate schema output.

        Example:

        .. code-block:: python

            >>> from isaacsim.asset.importer.mjcf import MJCFImporterConfig
            >>> MJCFImporterConfig()
            <...>
        """
        config = MJCFImporterConfig(
            allow_self_collision=True,
            mjcf_path=os.path.normpath(os.path.join(self._extension_path, "data", "mjcf", "nv_ant.xml")),
        )
        self.importer.config = config

        output_path = self.importer.import_mjcf()
        stage = stage_utils.open_stage(output_path)

        self.assertIsNotNone(stage, "Failed to open stage at path: {output_path}")
        await omni.kit.app.get_app().next_update_async()

        # check if object is there
        prim = stage.GetPrimAtPath("/ant")
        prim.GetVariantSet("Physics").SetVariantSelection("physx")
        self.assertNotEqual(prim.GetPath(), Sdf.Path.emptyPath)

        prim = stage.GetPrimAtPath("/ant/Geometry/torso")
        self.assertNotEqual(prim.GetPath(), Sdf.Path.emptyPath)
        self.assertEqual(prim.GetAttribute("physxArticulation:enabledSelfCollisions").Get(), True)

        # Start Simulation and wait
        self._timeline.play()
        await omni.kit.app.get_app().next_update_async()
        await asyncio.sleep(1.0)
        # nothing crashes
        self._timeline.stop()

        stage = None
        gc.collect()
        try:
            shutil.rmtree(os.path.normpath(os.path.dirname(output_path)))
        except OSError as e:
            carb.log_error(f"Error: {os.path.normpath(os.path.dirname(output_path))} : {e.strerror}")

    async def test_mjcf_visualize_collision_geom(self) -> None:
        """Import with collision-from-visuals enabled and validate results.

        Example:

        .. code-block:: python

            >>> from isaacsim.asset.importer.mjcf import MJCFImporterConfig
            >>> MJCFImporterConfig()
            <...>
        """
        from isaacsim.asset.importer.utils.impl import importer_utils

        config = MJCFImporterConfig(
            collision_from_visuals=True,
            collision_type="Convex Decomposition",
            mjcf_path=os.path.normpath(
                os.path.join(self._extension_path, "data", "mjcf", "open_ai_assets", "hand", "manipulate_block.xml")
            ),
        )
        self.importer.config = config

        output_path = self.importer.import_mjcf()
        stage = stage_utils.open_stage(os.path.normpath(output_path))

        self.assertIsNotNone(stage, "Failed to open stage at path: {output_path}")

        await omni.kit.app.get_app().next_update_async()

        prim = stage.GetPrimAtPath("/tn__MuJoCoModel_nB/Geometry/tn__robot0handmount_vFjD")
        self.assertNotEqual(prim.GetPath(), Sdf.Path.emptyPath)
        # Use PrimRange with TraverseInstanceProxies to find meshes that are instance proxies
        root_prim = stage.GetPrimAtPath("/")
        for child_prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()):
            prim_type = child_prim.GetTypeName()
            if prim_type in importer_utils.USD_GEOMETRY_TYPES:
                imageable = UsdGeom.Imageable(child_prim)
                purpose = imageable.GetPurposeAttr().Get() or "default"
                if purpose not in {"default", "render"}:
                    continue

                self.assertTrue(child_prim.HasAPI(UsdPhysics.CollisionAPI))
                collision_api = UsdPhysics.CollisionAPI(child_prim)
                collision_enabled_attr = collision_api.GetCollisionEnabledAttr()
                self.assertTrue(collision_enabled_attr.Get())
                if child_prim.IsA(UsdGeom.Mesh):
                    mesh_collision_api = UsdPhysics.MeshCollisionAPI(child_prim)
                    collider_type_attr = mesh_collision_api.GetApproximationAttr()
                    collider_type = collider_type_attr.Get()
                    self.assertEqual(collider_type, UsdPhysics.Tokens.convexDecomposition)
                    print(f"Collider type: {collider_type} for {child_prim.GetPath()}")

        # Start Simulation and wait
        self._timeline.play()
        await omni.kit.app.get_app().next_update_async()
        # nothing crashes
        self._timeline.stop()
        stage = None
        gc.collect()
        try:
            shutil.rmtree(os.path.normpath(os.path.dirname(output_path)))
        except OSError as e:
            carb.log_error(f"Error: {os.path.normpath(os.path.dirname(output_path))} : {e.strerror}")

    async def test_debug_mode(self) -> None:
        """Import with debug mode enabled and validate results.

        Example:

        .. code-block:: python

            >>> from isaacsim.asset.importer.mjcf import MJCFImporterConfig
            >>> MJCFImporterConfig()
            <...>
        """
        config = MJCFImporterConfig(
            debug_mode=True,
            mjcf_path=os.path.normpath(os.path.join(self._extension_path, "data", "mjcf", "nv_ant.xml")),
        )
        self.importer.config = config

        output_path = self.importer.import_mjcf()
        # Check if expected intermediate files exist (debug mode should keep temporary outputs)
        temp_usd_path = os.path.normpath(os.path.join(os.path.dirname(output_path), "..", "temp", "nv_ant.usd"))
        usdex_usdc_path = os.path.normpath(os.path.join(os.path.dirname(output_path), "..", "usdex", "ant.usdc"))
        self.assertTrue(os.path.exists(temp_usd_path), f"Temp USD file not found: {temp_usd_path}")
        self.assertTrue(os.path.exists(usdex_usdc_path), f"USDEx USDC file not found: {usdex_usdc_path}")
        self.assertTrue(os.path.exists(output_path), f"Output path not found: {output_path}")

        try:
            shutil.rmtree(os.path.normpath(os.path.dirname(temp_usd_path)))
        except OSError as e:
            carb.log_error(f"Error: {os.path.normpath(os.path.dirname(temp_usd_path))} : {e.strerror}")
        try:
            shutil.rmtree(os.path.normpath(os.path.dirname(usdex_usdc_path)))
        except OSError as e:
            carb.log_error(f"Error: {os.path.normpath(os.path.dirname(usdex_usdc_path))} : {e.strerror}")
        try:
            shutil.rmtree(os.path.normpath(os.path.dirname(output_path)))
        except OSError as e:
            carb.log_error(f"Error: {os.path.normpath(os.path.dirname(output_path))} : {e.strerror}")
