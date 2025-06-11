# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
Backend of "Save Robot"

- create a file with variants for _base and _physics layer
- add background (light and potential floor) and physicsscene outside of the defaultprim in the new usd




"""

import os

import omni.usd
from pxr import Sdf, Usd, UsdGeom, UsdPhysics

from ..utils.utils import apply_standard_stage_settings
from .robot_templates import RobotRegistry


def create_variant_usd(add_ground=False, add_lights=False, add_physics_scene=False):

    def _add_ground(stage):
        print("Adding ground plane")
        from isaacsim.core.api.objects.ground_plane import GroundPlane

        ground_plane = GroundPlane(prim_path="/Environment/groundPlane", z_position=0)

    def _add_light(stage):
        print("Adding light")
        from pxr import UsdLux

        light = UsdLux.DistantLight.Define(stage, "/Environment/defaultLight")
        light.CreateIntensityAttr().Set(1000.0)

    def _add_physics_scene(stage):
        print("Adding physics scene")
        from pxr import UsdPhysics

        physics_scene = UsdPhysics.Scene.Define(stage, "/Environment/physicsScene")

    robot = RobotRegistry().get()
    stage = Usd.Stage.CreateInMemory()
    apply_standard_stage_settings(stage)
    robot_xform = UsdGeom.Xform.Define(stage, Sdf.Path(f"/{robot.name}"))
    stage.SetDefaultPrim(robot_xform.GetPrim())

    # create a variant set for _base and _physics layer
    vs = robot_xform.GetPrim().GetVariantSets().AddVariantSet("Physics")
    for level in ("None", "PhysX"):
        vs.AddVariant(level)

    base_filepath = f"{robot.name}_base.usd"
    physics_filepath = f"{robot.name}_physics.usd"

    vs.SetVariantSelection("None")
    with vs.GetVariantEditContext():
        # define a prim to carry the payload
        # addPayload(assetPath, primPath) — primPath optional (defaults to defaultPrim)
        robot_xform.GetPrim().GetPayloads().AddPayload(
            assetPath=f"configurations/{base_filepath}",
        )

    vs.SetVariantSelection("PhysX")
    with vs.GetVariantEditContext():
        # define a prim to carry the payload
        # addPayload(assetPath, primPath) — primPath optional (defaults to defaultPrim)
        robot_xform.GetPrim().GetPayloads().AddPayload(
            assetPath=f"configurations/{physics_filepath}",
        )

    # 4) Export master layer

    root_dir = robot.robot_root_folder
    variant_usd_path = os.path.join(root_dir, f"{robot.name}.usd")

    stage.GetRootLayer().Export(variant_usd_path)

    # open the master layer as stage, and add the ground, light and physics scene as needed, save again
    omni.usd.get_context().open_stage(variant_usd_path)
    stage = omni.usd.get_context().get_stage()
    if add_ground:
        _add_ground(stage)
    if add_lights:
        _add_light(stage)
    if add_physics_scene:
        _add_physics_scene(stage)

    # save the stage
    stage.Save()


def apply_articulation_apis(robot_path, articulation_root_path):
    stage = omni.usd.get_context().get_stage()
    robot_prim = stage.GetPrimAtPath(robot_path)
    robot_name = robot_prim.GetName()
    # if articulation root not given, use the parent robot prim
    if articulation_root_path == "Pick from the Robot":
        articulation_prim = stage.GetPrimAtPath(f"/{robot_name}")
    # make sure there isn't already an articulation root on stage, if there is, delete it if it's not on the prim desired
    articulation_prim = stage.GetPrimAtPath(articulation_root_path)

    def remove_articulation_root_recursive(prim):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        for child in prim.GetChildren():
            remove_articulation_root_recursive(child)

    # delete any previous articulation root prim that might be on the robot
    remove_articulation_root_recursive(robot_prim)
    UsdPhysics.ArticulationRootAPI.Apply(articulation_prim)
