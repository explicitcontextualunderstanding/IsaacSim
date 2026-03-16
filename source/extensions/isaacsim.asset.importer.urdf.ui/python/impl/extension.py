# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""URDF importer UI extension and delegate integration."""

import copy
import gc
import os
import typing
from collections import namedtuple
from pathlib import Path

import carb
import omni.ext
import omni.kit.app
import omni.kit.tool.asset_importer as ai
import omni.usd
from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
from isaacsim.core.experimental.utils import stage as stage_utils
from omni.kit.helper.file_utils import asset_types
from omni.kit.notification_manager import NotificationStatus, post_notification

from .option_widget import OptionWidget

_extension_instance = None


def is_urdf_file(path: str) -> bool:
    """Check whether a path points to a URDF file.

    Args:
        path: Path to check.

    Returns:
        True if the path has a URDF extension, otherwise False.
    """
    _, ext = os.path.splitext(path.lower())
    return ext in [".urdf", ".URDF"]


class Extension(omni.ext.IExt):
    """UI Extension for the URDF Importer.

    Provides the user interface for importing URDF files into USD.
    Depends on isaacsim.asset.importer.urdf for core import functionality.
    """

    def on_startup(self, ext_id: str) -> None:
        """Initialize the extension when it is loaded.

        Args:
            ext_id: Extension identifier provided by the extension manager.
        """
        self._usd_context = omni.usd.get_context()
        self._models: dict[str, typing.Any] = {}
        self._config = URDFImporterConfig()
        self._extension_path = omni.kit.app.get_app().get_extension_manager().get_extension_path(ext_id)
        self._importer = URDFImporter(self._config)

        self.reset_config()
        self._option_builder = OptionWidget(self._models, self._config)

        self._delegate = UrdfImporterDelegate(
            "Urdf Importer",
            [r"(.*\.urdf$)|(.*\.URDF$)"],
            ["Urdf Files (*.urdf, *.URDF)"],
        )

        self._delegate.set_importer(self)
        ai.register_importer(self._delegate)

        global _extension_instance
        _extension_instance = self

    def build_new_options(self) -> None:
        """Build the UI options and sync to the current stage."""
        self._option_builder.build_options()

    def reset_config(self) -> None:
        """Reset importer configuration to default values."""
        # Set defaults
        self._config.usd_path = None
        self._config.merge_mesh = False
        self._config.debug_mode = False
        self._config.collision_from_visuals = False
        self._config.collision_type = "Convex Hull"
        self._config.allow_self_collision = False
        self._config.ros_package_paths = []

    async def _start_import(self, path: str | None = None, **kargs) -> str | None:
        """Start the URDF import process.

        Args:
            path: Path to the URDF file to import.
            **kargs: Additional keyword arguments.

        Returns:
            Path to the imported prim in the USD stage, or None if import failed.
        """
        if not path:
            carb.log_error("URDF Importer: No path provided")
            return None

        export_folder = self._models["dst_path"].get_value_as_string() if self._models.get("dst_path") else ""
        if export_folder == "Same as Imported Model(Default)":
            export_folder = ""

        if export_folder:
            if not os.path.isdir(export_folder):
                export_folder = os.path.dirname(export_folder)
        else:
            export_folder = os.path.dirname(path)

        if hasattr(self._option_builder, "get_ros_package_map"):
            self._config.ros_package_paths = self._option_builder.get_ros_package_map()

        self._config.urdf_path = path
        self._config.usd_path = export_folder

        stage_utils.create_new_stage()
        self._importer.config = self._config

        output_path = self._importer.import_urdf()
        if not output_path:
            carb.log_error(f"Failed to import URDF at path: {path}")
            return None

        result, _ = stage_utils.open_stage(output_path)
        self._last_config = copy.deepcopy(self._config)
        self.reset_config()
        if not result:
            carb.log_error(f"Failed to open stage at path: {output_path}")
            return None

        return output_path

    def _get_config(self) -> URDFImporterConfig:
        """Get the current importer configuration.

        Returns:
            Current importer configuration instance.
        """
        return self._last_config

    def on_shutdown(self) -> None:
        """Clean up resources when the extension is unloaded."""
        if hasattr(self, "_delegate"):
            self._delegate.destroy()
            ai.remove_importer(self._delegate)

        global _extension_instance
        _extension_instance = None
        gc.collect()

    def _print_config(self):
        carb.log_info(f"config urdf path: {self._config.urdf_path}")
        carb.log_info(f"config usd path: {self._config.usd_path}")
        carb.log_info(f"config merge mesh: {self._config.merge_mesh}")
        carb.log_info(f"config debug mode: {self._config.debug_mode}")
        carb.log_info(f"config collision from visuals: {self._config.collision_from_visuals}")
        carb.log_info(f"config collision type: {self._config.collision_type}")
        carb.log_info(f"config allow self collision: {self._config.allow_self_collision}")
        carb.log_info(f"config ros package paths: {self._config.ros_package_paths}")


class UrdfImporterDelegate(ai.AbstractImporterDelegate):
    """Delegate implementation for the asset importer integration.

    Args:
        name: Display name for the importer.
        filters: Regex filters for supported file types.
        descriptions: Descriptions for supported file types.
    """

    def __init__(self, name: str, filters: list[str], descriptions: list[str]) -> None:
        super().__init__()
        self._name = name
        self._filters = filters
        self._descriptions = descriptions
        self._importer: Extension | None = None
        # register the urdf icon to asset types
        ext_path = omni.kit.app.get_app().get_extension_manager().get_extension_path_by_module(__name__)
        icon_path = Path(ext_path).joinpath("icons").absolute()
        AssetTypeDef = namedtuple("AssetTypeDef", "glyph thumbnail matching_exts")
        known_asset_types = asset_types.known_asset_types()
        known_asset_types["urdf"] = AssetTypeDef(
            f"{icon_path}/icoFileURDF.png",
            f"{icon_path}/icoFileURDF.png",
            [".urdf", ".URDF"],
        )

    def set_importer(self, importer: Extension) -> None:
        """Set the importer instance used for conversions.

        Args:
            importer: Extension instance that performs imports.
        """
        self._importer = importer

    def show_destination_frame(self) -> bool:
        """Return whether the destination frame should be shown.

        Returns:
            False to hide the destination frame.
        """
        return False

    def destroy(self) -> None:
        """Release references held by the delegate."""
        self._importer = None

    def _on_import_complete(self, file_paths: list[str]) -> None:
        """Handle import completion callbacks.

        Args:
            file_paths: List of imported file paths.
        """

    @property
    def name(self) -> str:
        """Get the display name for the importer."""
        return self._name

    @property
    def filter_regexes(self) -> list[str]:
        """Get the filter regex patterns supported by the importer."""
        return self._filters

    @property
    def filter_descriptions(self) -> list[str]:
        """Get the filter descriptions shown in the UI."""
        return self._descriptions

    def build_options(self, paths: list[str]) -> None:
        """Build options for the provided asset paths.

        Args:
            paths: Paths selected for import.
        """
        if self._importer is not None:
            self._importer.reset_config()
            self._importer.build_new_options()
        else:
            carb.log_error("URDF Importer: Importer not initialized, cannot build options")

    def supports_usd_stage_cache(self) -> bool:
        """Report whether USD stage cache is supported.

        Returns:
            False since the importer does not support stage caching.
        """
        return False

    async def convert_assets(self, paths: list[str], **kargs) -> dict | None:
        """Convert selected URDF assets to USD.

        Args:
            paths: Paths selected for import.
            **kargs: Additional keyword arguments for import.

        Returns:
            An empty result dictionary on success, or None on failure.
        """
        if not paths:
            post_notification(
                "No file selected",
                "Please select a file to import",
                NotificationStatus.ERROR,
            )
            return None
        if self._importer is None:
            carb.log_warn("URDF Importer: Importer not initialized, cannot import assets")
            return None
        for path in paths:
            await self._importer._start_import(path=path, **kargs)
        # Don't need to return dest path here, _load_robot do the insertion to stage
        return {}


def get_instance() -> Extension | None:
    """Return the active URDF UI extension instance.

    Returns:
        Extension instance if initialized, otherwise None.

    Example:

    .. code-block:: python

        >>> from isaacsim.asset.importer.urdf.ui.impl import extension
        >>> extension.get_instance()  # doctest: +SKIP
    """
    return _extension_instance
