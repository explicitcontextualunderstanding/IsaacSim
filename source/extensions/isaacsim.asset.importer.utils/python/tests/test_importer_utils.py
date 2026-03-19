# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import omni.kit.test
import omni.usd
from isaacsim.asset.importer.utils.impl import importer_utils, stage_utils
from pxr import PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics


class TestImporterUtils(omni.kit.test.AsyncTestCase):
    """Test helpers in :mod:`isaacsim.asset.importer.utils.impl.importer_utils`.

    Example:

    .. code-block:: python

        >>> import omni.kit.test
        >>> class Example(omni.kit.test.AsyncTestCase):
        ...     pass
        ...
    """

    async def setUp(self) -> None:
        """Prepare a new USD stage for each test.

        Example:

        .. code-block:: python

            >>> import omni.usd
            >>> omni.usd.get_context()
            <...>
        """
        await omni.usd.get_context().new_stage_async()
        await omni.kit.app.get_app().next_update_async()

    async def test_delete_scope_reparents_children(self) -> None:
        """Reparent scope children to the parent prim before deleting the scope.

        Example:

        .. code-block:: python

            >>> from pxr import Usd, UsdGeom
            >>> stage = Usd.Stage.CreateInMemory()
            >>> UsdGeom.Xform.Define(stage, "/World")
            UsdGeom.Xform(Usd.Prim(</World>))
        """
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, "/World")
        UsdGeom.Scope.Define(stage, "/World/Scope")
        UsdGeom.Xform.Define(stage, "/World/Scope/Child")

        importer_utils.delete_scope(stage, "/World/Scope")

        self.assertFalse(stage.GetPrimAtPath("/World/Scope").IsValid())
        self.assertTrue(stage.GetPrimAtPath("/World/Child").IsValid())

    async def test_add_joint_schemas(self) -> None:
        """Apply joint-related schemas to prismatic and revolute joints.

        Example:

        .. code-block:: python

            >>> from pxr import Usd, UsdPhysics
            >>> stage = Usd.Stage.CreateInMemory()
            >>> UsdPhysics.RevoluteJoint.Define(stage, "/World/Joint")
            UsdPhysics.RevoluteJoint(Usd.Prim(</World/Joint>))
        """
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, "/World")
        revolute = UsdPhysics.RevoluteJoint.Define(stage, "/World/Revolute").GetPrim()
        prismatic = UsdPhysics.PrismaticJoint.Define(stage, "/World/Prismatic").GetPrim()

        importer_utils.add_joint_schemas(stage)

        self.assertTrue(revolute.HasAPI(PhysxSchema.PhysxJointAPI))
        self.assertTrue(revolute.HasAPI(UsdPhysics.DriveAPI, "angular"))
        self.assertTrue(revolute.HasAPI(PhysxSchema.JointStateAPI, "angular"))

        self.assertTrue(prismatic.HasAPI(PhysxSchema.PhysxJointAPI))
        self.assertTrue(prismatic.HasAPI(UsdPhysics.DriveAPI, "linear"))
        self.assertTrue(prismatic.HasAPI(PhysxSchema.JointStateAPI, "linear"))

    async def test_add_rigid_body_schemas(self) -> None:
        """Apply MassAPI to rigid bodies discovered via USDRT.

        Example:

        .. code-block:: python

            >>> from pxr import Usd, UsdPhysics
            >>> stage = Usd.Stage.CreateInMemory()
            >>> UsdPhysics.RigidBodyAPI.Apply(stage.DefinePrim("/World/Body"))
            UsdPhysics.RigidBodyAPI(Usd.Prim(</World/Body>))
        """
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, "/World")
        body_prim = stage.DefinePrim("/World/Body", "Xform")
        UsdPhysics.RigidBodyAPI.Apply(body_prim)
        importer_utils.add_rigid_body_schemas(stage)

        self.assertTrue(body_prim.HasAPI(UsdPhysics.RigidBodyAPI))
        self.assertTrue(body_prim.HasAPI(UsdPhysics.MassAPI))

    async def test_enable_self_collision_applies_articulation_api(self) -> None:
        """Enable self-collision on the default prim when missing roots."""
        stage = Usd.Stage.CreateInMemory()
        default_prim = UsdGeom.Xform.Define(stage, "/World").GetPrim()
        stage.SetDefaultPrim(default_prim)

        updated = importer_utils.enable_self_collision(stage, enabled=True)

        self.assertEqual(updated, 1)
        self.assertTrue(default_prim.HasAPI("PhysicsArticulationRootAPI"))
        self.assertTrue(default_prim.HasAPI(PhysxSchema.PhysxArticulationAPI))
        self.assertTrue(default_prim.HasAPI("NewtonArticulationRootAPI"))

        physx_api = PhysxSchema.PhysxArticulationAPI(default_prim)
        self.assertTrue(physx_api.GetEnabledSelfCollisionsAttr().IsValid())
        self.assertTrue(physx_api.GetEnabledSelfCollisionsAttr().Get())

        newton_attr = default_prim.GetAttribute("newton:selfCollisionEnabled")
        self.assertTrue(newton_attr.IsValid())
        self.assertTrue(newton_attr.Get())
