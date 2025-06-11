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

import carb
import omni.isaac.IsaacSensorSchema as IsaacSensorSchema
import omni.isaac.RangeSensorSchema as RangeSensorSchema
import omni.kit.commands
import omni.kit.utils
from isaacsim.core.utils.stage import get_next_free_path
from pxr import Gf, UsdGeom


def setup_base_prim(prim, enabled, draw_points, draw_lines, min_range, max_range):
    RangeSensorSchema.RangeSensor(prim).CreateEnabledAttr(enabled)
    RangeSensorSchema.RangeSensor(prim).CreateDrawPointsAttr(draw_points)
    RangeSensorSchema.RangeSensor(prim).CreateDrawLinesAttr(draw_lines)
    RangeSensorSchema.RangeSensor(prim).CreateMinRangeAttr(min_range)
    RangeSensorSchema.RangeSensor(prim).CreateMaxRangeAttr(max_range)


# this command is used to create each REB prim, it also handles undo so that each individual prim command doesn't have to
class RangeSensorCreatePrim(omni.kit.commands.Command):
    def __init__(
        self,
        path: str = "",
        parent: str = "",
        schema_type=RangeSensorSchema.RangeSensor,
        min_range: float = 0.4,
        max_range: float = 100.0,
        draw_points: bool = False,
        draw_lines: bool = False,
    ):
        # condensed way to copy all input arguments into self with an underscore prefix
        for name, value in vars().items():
            if name != "self":
                setattr(self, f"_{name}", value)
        self._prim_path = None
        pass

    def do(self):
        self._stage = omni.usd.get_context().get_stage()
        # make prim path unique
        self._prim_path = get_next_free_path(self._path, self._parent)
        self._prim = self._schema_type.Define(self._stage, self._prim_path)
        setup_base_prim(self._prim, True, self._draw_points, self._draw_lines, self._min_range, self._max_range)

        xform = UsdGeom.Xformable(self._prim)
        xform_trans = xform.AddXformOp(UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.PrecisionDouble, "")
        xform_rot = xform.AddXformOp(UsdGeom.XformOp.TypeRotateXYZ, UsdGeom.XformOp.PrecisionDouble, "")

        # rotate sensor to align correctly if stage is y up
        if UsdGeom.GetStageUpAxis(self._stage) == UsdGeom.Tokens.y:
            xform_rot.Set(Gf.Vec3d(270, 0, 0))
        return self._prim

    def undo(self):
        if self._prim_path is not None:
            return self._stage.RemovePrim(self._prim_path)


class RangeSensorCreateLidar(omni.kit.commands.Command):
    """Commands class to create a lidar sensor.

    Typical usage example:

    .. code-block:: python

        result, prim = omni.kit.commands.execute(
            "RangeSensorCreateLidar",
            path="/Lidar",
            parent=None,
            min_range=0.4,
            max_range=100.0,
            draw_points=False,
            draw_lines=False,
            horizontal_fov=360.0,
            vertical_fov=30.0,
            horizontal_resolution=0.4,
            vertical_resolution=4.0,
            rotation_rate=20.0,
            high_lod=False,
            yaw_offset=0.0,
            enable_semantics=False,
        )
    """

    def __init__(
        self,
        path: str = "/Lidar",
        parent=None,
        min_range: float = 0.4,
        max_range: float = 100.0,
        draw_points: bool = False,
        draw_lines: bool = False,
        horizontal_fov: float = 360.0,
        vertical_fov: float = 30.0,
        horizontal_resolution: float = 0.4,
        vertical_resolution: float = 4.0,
        rotation_rate: float = 20.0,
        high_lod: bool = False,
        yaw_offset: float = 0.0,
        enable_semantics: bool = False,
    ):
        # condensed way to copy all input arguments into self with an underscore prefix
        for name, value in vars().items():
            if name != "self":
                setattr(self, f"_{name}", value)
        self._prim = None
        pass

    def do(self):
        success, self._prim = omni.kit.commands.execute(
            "RangeSensorCreatePrim",
            path=self._path,
            parent=self._parent,
            schema_type=RangeSensorSchema.Lidar,
            draw_points=self._draw_points,
            draw_lines=self._draw_lines,
            min_range=self._min_range,
            max_range=self._max_range,
        )
        if success and self._prim:
            self._prim.CreateHorizontalFovAttr().Set(self._horizontal_fov)
            self._prim.CreateVerticalFovAttr().Set(self._vertical_fov)
            self._prim.CreateRotationRateAttr().Set(self._rotation_rate)
            self._prim.CreateHorizontalResolutionAttr().Set(self._horizontal_resolution)
            self._prim.CreateVerticalResolutionAttr().Set(self._vertical_resolution)
            self._prim.CreateHighLodAttr().Set(self._high_lod)
            self._prim.CreateYawOffsetAttr().Set(self._yaw_offset)
            self._prim.CreateEnableSemanticsAttr().Set(self._enable_semantics)
        else:
            carb.log.error("Could not create lidar prim")
        return self._prim

    def undo(self):
        # undo must be defined even if empty
        pass


class RangeSensorCreateGeneric(omni.kit.commands.Command):
    """Commands class to create a generic range sensor.

    Typical usage example:

    .. code-block:: python

        result, prim = omni.kit.commands.execute(
            "RangeSensorCreateGeneric",
            path="/GenericSensor",
            parent=None,
            min_range=0.4,
            max_range=100.0,
            draw_points=False,
            draw_lines=False,
            sampling_rate=60,
        )
    """

    def __init__(
        self,
        path: str = "/GenericSensor",
        parent=None,
        min_range: float = 0.4,
        max_range: float = 100.0,
        draw_points: bool = False,
        draw_lines: bool = False,
        sampling_rate: int = 60,
    ):
        # condensed way to copy all input arguments into self with an underscore prefix
        for name, value in vars().items():
            if name != "self":
                setattr(self, f"_{name}", value)
        self._prim = None
        pass

    def do(self):
        success, self._prim = omni.kit.commands.execute(
            "RangeSensorCreatePrim",
            path=self._path,
            parent=self._parent,
            schema_type=RangeSensorSchema.Generic,
            draw_points=self._draw_points,
            draw_lines=self._draw_lines,
            min_range=self._min_range,
            max_range=self._max_range,
        )
        if success and self._prim:
            self._prim.CreateSamplingRateAttr().Set(self._sampling_rate)
        else:
            carb.log.error("Could not create generic sensor prim")
        return self._prim

    def undo(self):
        if self._prim_path is not None:
            return self._stage.RemovePrim(self._prim_path)
        pass


class IsaacSensorCreateLightBeamSensor(omni.kit.commands.Command):
    def __init__(
        self,
        path: str = "/LightBeam_Sensor",
        parent: str = None,
        translation: Gf.Vec3d = Gf.Vec3d(0, 0, 0),
        orientation: Gf.Quatd = Gf.Quatd(1, 0, 0, 0),
        num_rays: int = 1,
        curtain_length: float = 0.0,
        forward_axis: Gf.Vec3d = Gf.Vec3d(1, 0, 0),  # default to x axis
        curtain_axis: Gf.Vec3d = Gf.Vec3d(0, 0, 1),  # default to z axis
        min_range: float = 0.4,
        max_range: float = 100.0,
        draw_points: bool = False,
        draw_lines: bool = False,
    ):
        # condensed way to copy all input arguments into self with an underscore prefix
        for name, value in vars().items():
            if name != "self":
                setattr(self, f"_{name}", value)
        self._prim = None
        pass

    def do(self):
        if self._num_rays > 1 and self._curtain_length == 0:
            carb.log_error("Must specify curtain length if num rays > 1")
        success, self._prim = omni.kit.commands.execute(
            "RangeSensorCreatePrim",
            path=self._path,
            parent=self._parent,
            schema_type=IsaacSensorSchema.IsaacLightBeamSensor,
            draw_points=self._draw_points,
            draw_lines=self._draw_lines,
            min_range=self._min_range,
            max_range=self._max_range,
        )

        if success and self._prim:
            self._prim.CreateNumRaysAttr().Set(self._num_rays)
            self._prim.CreateCurtainLengthAttr().Set(self._curtain_length)
            self._prim.CreateForwardAxisAttr().Set(self._forward_axis)
            self._prim.CreateCurtainAxisAttr().Set(self._curtain_axis)
            self._prim.CreateMinRangeAttr().Set(self._min_range)
            self._prim.CreateMaxRangeAttr().Set(self._max_range)

            return self._prim
        else:
            carb.log_error("Could not create light beam sensor prim")
            return None

    def undo(self):
        # undo must be defined even if empty
        pass


omni.kit.commands.register_all_commands_in_module(__name__)
