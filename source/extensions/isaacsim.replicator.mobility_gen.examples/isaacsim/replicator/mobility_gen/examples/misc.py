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

"""Provides stereo camera components for mobility generation, including the Leopard Imaging Hawk camera system."""


import os
from typing import Tuple

from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.replicator.mobility_gen.impl.camera import MobilityGenCamera
from isaacsim.replicator.mobility_gen.impl.common import Module
from isaacsim.replicator.mobility_gen.impl.utils.global_utils import join_sdf_paths
from isaacsim.storage.native import get_assets_root_path


class HawkCamera(Module):
    """A stereo camera module representing the Leopard Imaging Hawk camera system.

    This class provides a high-level interface for working with the Hawk stereo camera, which consists of left and
    right camera sensors. The Hawk camera has a resolution of 960x600 pixels and can be used for stereo vision
    applications in Isaac Sim.

    The class supports both building new Hawk camera instances from USD assets and attaching to existing camera
    prims in the stage. It manages the left and right camera components as MobilityGenCamera instances.

    Args:
        left: The left camera component of the stereo pair.
        right: The right camera component of the stereo pair.
    """

    usd_url: str = get_assets_root_path() + "/Isaac/Sensors/LeopardImaging/Hawk/hawk_v1.1_nominal.usd"
    """USD file path for the Hawk camera asset."""
    resolution: Tuple[int, int] = (960, 600)
    """Resolution of the left and right cameras as (width, height)."""
    left_camera_path: str = "left/camera_left"
    """Relative path to the left camera within the Hawk camera prim."""
    right_camera_path: str = "right/camera_right"
    """Relative path to the right camera within the Hawk camera prim."""

    def __init__(self, left: MobilityGenCamera, right: MobilityGenCamera):
        self.left = left
        self.right = right

    @classmethod
    def build(cls, prim_path: str) -> "HawkCamera":
        """Creates a new HawkCamera by adding the Hawk camera USD reference to the stage.

        Args:
            prim_path: USD prim path where the Hawk camera will be created.

        Returns:
            A new HawkCamera instance with left and right cameras configured.
        """

        stage = get_current_stage()

        add_reference_to_stage(usd_path=cls.usd_url, prim_path=prim_path)

        return cls.attach(prim_path)

    @classmethod
    def attach(cls, prim_path: str) -> "HawkCamera":
        """Attaches to an existing Hawk camera prim on the stage.

        Args:
            prim_path: USD prim path of the existing Hawk camera.

        Returns:
            A new HawkCamera instance connected to the existing prim's left and right cameras.
        """

        left_camera = MobilityGenCamera(join_sdf_paths(prim_path, cls.left_camera_path), cls.resolution)
        right_camera = MobilityGenCamera(join_sdf_paths(prim_path, cls.right_camera_path), cls.resolution)

        return HawkCamera(left_camera, right_camera)
