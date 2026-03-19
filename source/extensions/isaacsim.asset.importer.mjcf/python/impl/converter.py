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


"""MJCF to USD conversion pipeline."""

from __future__ import annotations

import gc
import importlib
import os
import shutil
from typing import Any

import omni
from isaacsim.asset.importer.utils.impl import (
    importer_utils,
    merge_mesh_utils,
    mjc_to_physx_conversion_utils,
    stage_utils,
)
from pxr import Sdf

from .config import MJCFImporterConfig


class MJCFImporter:
    """MuJoCo MJCF to USD importer.

    Uses mujoco-usd-converter to convert MJCF files to USD format.

    Args:
        config: Optional configuration for the import operation.

    Example:

    .. code-block:: python

        >>> from isaacsim.asset.importer.mjcf import MJCFImporter
        >>> MJCFImporter()
        <...>
    """

    def __init__(self, config: MJCFImporterConfig | None = None) -> None:
        self._config = config if config else MJCFImporterConfig()
        self.converter: Any = None

    @property
    def config(self) -> MJCFImporterConfig:
        """Get the importer configuration.

        Returns:
            Current importer configuration.

        Example:

        .. code-block:: python

            >>> from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig

            >>> importer = MJCFImporter()
            >>> importer.config  # doctest: +ELLIPSIS
            MJCFImporterConfig(...)
        """
        return self._config

    @config.setter
    def config(self, config: MJCFImporterConfig) -> None:
        self._config = config

    def import_mjcf(self, config: MJCFImporterConfig | None = None) -> str:
        """Import an MJCF file and convert it to USD.

        Args:
            config: Optional configuration for the import operation.
                If not provided, the stored importer configuration will be used.

        Returns:
            Path to the generated USD file.

        Raises:
            ValueError: If the MJCF path is not configured.
            FileNotFoundError: If the MJCF file does not exist at the given path.

        Example:

        .. code-block:: python

            >>> from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig

            >>> importer = MJCFImporter()
            >>> config = MJCFImporterConfig(mjcf_path="/tmp/robot.xml")
            >>> importer.config = config
            >>> # output_path = importer.import_mjcf()
        """
        if config is not None:
            self.config = config

        if not self.config.mjcf_path:
            raise ValueError("MJCF path is not set in the importer configuration.")

        mjcf_path = os.path.normpath(self.config.mjcf_path)
        if not os.path.exists(mjcf_path):
            raise FileNotFoundError(f"MJCF file does not exist at path: {mjcf_path}")

        robot_name = os.path.basename(mjcf_path).split(".")[0]

        if self.config.usd_path is None:
            self.config.usd_path = os.path.normpath(os.path.dirname(mjcf_path))

        usd_path = os.path.normpath(self.config.usd_path)

        usdex_path = os.path.normpath(os.path.join(usd_path, "usdex"))
        intermediate_path = os.path.normpath(os.path.join(usd_path, "temp", f"{robot_name}.usd"))

        if self.converter is None:
            mujoco_usd_converter = importlib.import_module("mujoco_usd_converter")
            self.converter = mujoco_usd_converter.Converter(layer_structure=False, scene=self.config.import_scene)

        asset: Sdf.AssetPath = self.converter.convert(mjcf_path, usdex_path)

        # Now open the usdex stage in the USD context
        self.stage = stage_utils.open_stage(asset.path)

        if not self.stage:
            raise ValueError(f"Failed to open usdex stage at path: {asset.path}")

        importer_utils.add_rigid_body_schemas(self.stage)
        importer_utils.add_joint_schemas(self.stage)

        if self.config.merge_mesh:
            merge_mesh_utils.clean_mesh_operation(self.stage)
            merge_mesh_utils.generate_mesh_uv_normals_operation(self.stage)
            merge_mesh_utils.merge_meshes_operation(self.stage)

        if self.config.collision_from_visuals:
            importer_utils.collision_from_visuals(self.stage, self.config.collision_type)

        importer_utils.enable_self_collision(self.stage, self.config.allow_self_collision)
        mjc_to_physx_conversion_utils.convert_mjc_to_physx(self.stage)
        stage_utils.save_stage(self.stage, intermediate_path)  # save the stage to the output path
        self.stage = None
        gc.collect()

        ext_manager = omni.kit.app.get_app().get_extension_manager()
        ext_id = ext_manager.get_enabled_extension_id("isaacsim.asset.transformer.rules")
        extension_path = ext_manager.get_extension_path(ext_id)

        asset_structure_profile_json_path = os.path.normpath(
            os.path.abspath(os.path.join(f"{extension_path}", "data", "isaacsim_structure.json"))
        )

        if self.config.debug_mode:
            log_path = os.path.normpath(
                os.path.join(os.path.dirname(intermediate_path), "isaacsim_structure_logs.json")
            )
        else:
            log_path = None
        importer_utils.run_asset_transformer_profile(
            input_stage_path=intermediate_path,
            output_package_root=os.path.normpath(os.path.join(usd_path, robot_name)),
            profile_json_path=asset_structure_profile_json_path,
            log_path=log_path,
        )

        if not self.config.debug_mode:
            # Clean up the temporary usdex directory
            if os.path.exists(usdex_path):
                shutil.rmtree(usdex_path)
            if os.path.exists(intermediate_path):
                shutil.rmtree(os.path.normpath(os.path.dirname(intermediate_path)))

        final_path = os.path.normpath(os.path.join(usd_path, robot_name, f"{robot_name}.usda"))

        return final_path
