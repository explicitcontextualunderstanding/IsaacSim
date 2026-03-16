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

import omni.kit.test
from isaacsim.asset.importer.utils.impl import mjc_to_physx_conversion_utils, urdf_to_mjc_physx_conversion_utils
from pxr import PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics


class TestMjcPhysxConversionUtils(omni.kit.test.AsyncTestCase):
    """Tests for MJCF/PhysX conversion utilities."""

    async def test_convert_urdf_to_physx_applies_limits(self) -> None:
        """Apply URDF joint limits to PhysX drive and joint APIs."""
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, "/World")
        joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/Joint").GetPrim()

        joint.CreateAttribute("urdf:limit:effort", Sdf.ValueTypeNames.Float).Set(120.0)
        joint.CreateAttribute("urdf:limit:velocity", Sdf.ValueTypeNames.Float).Set(6.0)
        joint.CreateAttribute("urdf:dynamics:damping", Sdf.ValueTypeNames.Float).Set(0.1)
        joint.CreateAttribute("urdf:dynamics:friction", Sdf.ValueTypeNames.Float).Set(0.2)
        joint.CreateAttribute("urdf:calibration:reference_position", Sdf.ValueTypeNames.Float).Set(10.0)
        urdf_to_mjc_physx_conversion_utils.convert_urdf_to_physx(joint)

        drive_api = UsdPhysics.DriveAPI(joint, "angular")
        self.assertTrue(drive_api.GetMaxForceAttr().IsValid())
        self.assertEqual(drive_api.GetMaxForceAttr().Get(), 120.0)

        self.assertTrue(drive_api.GetDampingAttr().IsValid())
        self.assertAlmostEqual(drive_api.GetDampingAttr().Get(), 0.1, delta=1e-2)

        self.assertTrue(drive_api.GetTargetPositionAttr().IsValid())
        self.assertAlmostEqual(drive_api.GetTargetPositionAttr().Get(), 10.0, delta=1e-2)

        physx_joint_api = PhysxSchema.PhysxJointAPI(joint)
        self.assertTrue(physx_joint_api.GetMaxJointVelocityAttr().IsValid())
        self.assertAlmostEqual(physx_joint_api.GetMaxJointVelocityAttr().Get(), 6.0 * 180 / 3.1415926, delta=1e-2)

        self.assertTrue(physx_joint_api.GetJointFrictionAttr().IsValid())
        self.assertAlmostEqual(physx_joint_api.GetJointFrictionAttr().Get(), 0.2, delta=1e-2)

    async def test_convert_physx_to_mjc_authors_mjc_attrs(self) -> None:
        """Author MJCF attributes from PhysX joint data."""
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, "/World")
        joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/Joint").GetPrim()

        drive_api = UsdPhysics.DriveAPI.Apply(joint, "angular")
        drive_api.CreateTargetPositionAttr().Set(1.25)

        physx_joint_api = PhysxSchema.PhysxJointAPI.Apply(joint)
        physx_joint_api.CreateJointFrictionAttr().Set(0.4)
        physx_joint_api.CreateArmatureAttr().Set(0.02)

        urdf_to_mjc_physx_conversion_utils.convert_physx_to_mjc(joint)

        self.assertTrue(joint.GetAttribute("mjc:ref").IsValid())
        self.assertAlmostEqual(joint.GetAttribute("mjc:ref").Get(), 1.25, delta=1e-2)

        self.assertTrue(joint.GetAttribute("mjc:frictionloss").IsValid())
        self.assertAlmostEqual(joint.GetAttribute("mjc:frictionloss").Get(), 0.4, delta=1e-2)

        self.assertTrue(joint.GetAttribute("mjc:armature").IsValid())
        self.assertAlmostEqual(joint.GetAttribute("mjc:armature").Get(), 0.02, delta=1e-2)

    async def test_convert_joints_attributes_creates_actuator(self) -> None:
        """Create MJCF actuators and transfer joint data."""
        stage = Usd.Stage.CreateInMemory()
        default_prim = UsdGeom.Xform.Define(stage, "/World").GetPrim()
        stage.SetDefaultPrim(default_prim)

        joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/Joint").GetPrim()
        drive_api = UsdPhysics.DriveAPI.Apply(joint, "angular")
        drive_api.CreateMaxForceAttr().Set(55.0)
        drive_api.CreateStiffnessAttr().Set(10.0)
        drive_api.CreateDampingAttr().Set(1.5)
        drive_api.CreateTargetPositionAttr().Set(0.5)

        physx_joint_api = PhysxSchema.PhysxJointAPI.Apply(joint)
        physx_joint_api.CreateJointFrictionAttr().Set(0.2)

        urdf_to_mjc_physx_conversion_utils.convert_joints_attributes(stage)

        actuator_path = "/World/Physics/Joint_actuator"
        actuator = stage.GetPrimAtPath(actuator_path)
        self.assertTrue(actuator.IsValid())
        self.assertEqual(actuator.GetTypeName(), "MjcActuator")

        targets = actuator.GetRelationship("mjc:target").GetTargets()
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0], joint.GetPath())

        self.assertTrue(actuator.GetAttribute("mjc:gainPrm").IsValid())
        self.assertTrue(actuator.GetAttribute("mjc:biasPrm").IsValid())

        self.assertTrue(joint.GetAttribute("mjc:ref").IsValid())
        self.assertAlmostEqual(joint.GetAttribute("mjc:ref").Get(), 0.5, delta=1e-2)

        self.assertTrue(joint.GetAttribute("mjc:frictionloss").IsValid())
        self.assertAlmostEqual(joint.GetAttribute("mjc:frictionloss").Get(), 0.2, delta=1e-2)

    async def test_convert_mjc_to_physx_updates_drive(self) -> None:
        """Convert MJCF actuator and joint attributes into PhysX APIs."""
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, "/World")
        joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/Joint").GetPrim()
        joint.CreateAttribute("mjc:frictionloss", Sdf.ValueTypeNames.Float).Set(0.3)
        joint.CreateAttribute("mjc:ref", Sdf.ValueTypeNames.Float).Set(0.75)

        actuator = stage.DefinePrim("/World/Actuator", "MjcActuator")
        actuator.CreateRelationship("mjc:target", custom=False).SetTargets([joint.GetPath()])
        actuator.CreateAttribute("mjc:gainType", Sdf.ValueTypeNames.String).Set("fixed")
        actuator.CreateAttribute("mjc:biasType", Sdf.ValueTypeNames.String).Set("affine")
        actuator.CreateAttribute("mjc:gainPrm", Sdf.ValueTypeNames.FloatArray).Set(
            [12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        actuator.CreateAttribute("mjc:biasPrm", Sdf.ValueTypeNames.FloatArray).Set(
            [0.0, -12.0, -3.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        actuator.CreateAttribute("mjc:forceRange:max", Sdf.ValueTypeNames.Float).Set(88.0)
        actuator.CreateAttribute("mjc:forceRange:min", Sdf.ValueTypeNames.Float).Set(-88.0)

        mjc_to_physx_conversion_utils.convert_mjc_to_physx(stage)

        drive_api = UsdPhysics.DriveAPI.Get(joint, "angular")
        self.assertTrue(drive_api.GetStiffnessAttr().IsValid())
        self.assertAlmostEqual(drive_api.GetStiffnessAttr().Get(), 12.0, delta=1e-2)
        self.assertTrue(drive_api.GetDampingAttr().IsValid())
        self.assertAlmostEqual(drive_api.GetDampingAttr().Get(), 3.5, delta=1e-2)
        self.assertTrue(drive_api.GetMaxForceAttr().IsValid())
        self.assertAlmostEqual(drive_api.GetMaxForceAttr().Get(), 88.0, delta=1e-2)
        self.assertTrue(drive_api.GetTargetPositionAttr().IsValid())
        self.assertAlmostEqual(drive_api.GetTargetPositionAttr().Get(), 0.75, delta=1e-2)

        physx_joint_api = PhysxSchema.PhysxJointAPI(joint)
        self.assertTrue(physx_joint_api.GetJointFrictionAttr().IsValid())
        self.assertAlmostEqual(physx_joint_api.GetJointFrictionAttr().Get(), 0.3, delta=1e-2)

    async def test_create_mjc_actuator_from_physics_copies_drive_limits(self) -> None:
        """Create MJCF actuator attributes from PhysX drive limits."""
        stage = Usd.Stage.CreateInMemory()
        default_prim = UsdGeom.Xform.Define(stage, "/World").GetPrim()
        stage.SetDefaultPrim(default_prim)

        joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/Joint").GetPrim()
        drive_api = UsdPhysics.DriveAPI.Apply(joint, "angular")
        drive_api.CreateMaxForceAttr().Set(42.0)
        drive_api.CreateStiffnessAttr().Set(2.5)
        drive_api.CreateDampingAttr().Set(0.4)
        urdf_to_mjc_physx_conversion_utils.create_mjc_actuator_from_physics(joint, stage, "/World/Physics")
        actuator = stage.GetPrimAtPath("/World/Physics/Joint_actuator")
        self.assertTrue(actuator.IsValid())
        self.assertEqual(actuator.GetTypeName(), "MjcActuator")

        self.assertTrue(actuator.GetAttribute("mjc:forceRange:max").IsValid())
        self.assertAlmostEqual(actuator.GetAttribute("mjc:forceRange:max").Get(), 42.0, delta=1e-2)
        self.assertTrue(actuator.GetAttribute("mjc:forceRange:min").IsValid())
        self.assertAlmostEqual(actuator.GetAttribute("mjc:forceRange:min").Get(), -42.0, delta=1e-2)

        self.assertTrue(actuator.GetAttribute("mjc:gainPrm").IsValid())
        self.assertTrue(actuator.GetAttribute("mjc:biasPrm").IsValid())
        gain_prm = actuator.GetAttribute("mjc:gainPrm").Get()
        bias_prm = actuator.GetAttribute("mjc:biasPrm").Get()
        self.assertAlmostEqual(gain_prm[0], 2.5, delta=1e-2)
        self.assertAlmostEqual(gain_prm[1], 0.0, delta=1e-2)
        self.assertAlmostEqual(gain_prm[2], 0.0, delta=1e-2)
        self.assertAlmostEqual(bias_prm[0], 0.0, delta=1e-2)
        self.assertAlmostEqual(bias_prm[1], -2.5, delta=1e-2)
        self.assertAlmostEqual(bias_prm[2], -0.4, delta=1e-2)

        self.assertTrue(actuator.GetAttribute("mjc:gainType").IsValid())
        self.assertEqual(actuator.GetAttribute("mjc:gainType").Get(), "fixed")
        self.assertTrue(actuator.GetAttribute("mjc:biasType").IsValid())
        self.assertEqual(actuator.GetAttribute("mjc:biasType").Get(), "affine")
