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


"""MJCF importer UI extension entry point."""

import copy
import gc
import os
from collections import namedtuple
from pathlib import Path

import carb
import omni.ext
import omni.kit.tool.asset_importer as ai
import omni.ui as ui
from isaacsim.asset.importer.mjcf.impl import MJCFImporter, MJCFImporterConfig
from isaacsim.core.experimental.utils import stage as stage_utils
from omni.kit.helper.file_utils import asset_types
from omni.kit.notification_manager import NotificationStatus, post_notification

from .option_widget import OptionWidget

_extension_instance = None


def is_mjcf_file(path: str) -> bool:
    """Check whether the given path is an MJCF file.

    Args:
        path: File path to check.

    Returns:
        True if the file has an ``.xml`` extension.

    Example:

    .. code-block:: python

        >>> from isaacsim.asset.importer.mjcf.ui.impl import extension
        >>> extension.is_mjcf_file("/tmp/robot.xml")
        True
    """
    _, ext = os.path.splitext(path.lower())
    return ext == ".xml"


class Extension(omni.ext.IExt):
    """UI Extension for the MJCF Importer.

    Provides the user interface for importing MJCF files into USD.
    Depends on isaacsim.asset.importer.mjcf for core import functionality.
    """

    def on_startup(self, ext_id: str) -> None:
        """Initialize the MJCF importer UI extension.

        Args:
            ext_id: Extension identifier provided by Kit.

        Example:

        .. code-block:: python

            >>> import omni.ext
            >>> isinstance(omni.ext.IExt(), omni.ext.IExt)  # doctest: +SKIP
            True
        """
        self._usd_context = omni.usd.get_context()
        self._models: dict[str, ui.AbstractValueModel] = {}
        self._config: MJCFImporterConfig = MJCFImporterConfig()
        self._extension_path = omni.kit.app.get_app().get_extension_manager().get_extension_path(ext_id)
        self.reset_config()
        self._option_builder = OptionWidget(self._models, self._config)
        self._delegate = MjcfImporterDelegate(
            "MJCF Importer",
            ["(.*\\.xml$)|(.*\\.XML$)"],
            ["Mjcf Files (*.xml, *.XML)"],
        )
        self._collision_type_items = [
            "Convex Hull",
            "Convex Decomposition",
            "Bounding Sphere",
            "Bounding Cube",
        ]
        self._delegate.set_importer(self)
        self._importer = MJCFImporter(self._config)
        ai.register_importer(self._delegate)

        global _extension_instance
        _extension_instance = self

    def build_new_options(self) -> None:
        """Build a fresh options UI for the importer.

        Example:

        .. code-block:: python

            >>> from isaacsim.asset.importer.mjcf.ui.impl.extension import Extension
            >>> Extension().build_new_options()  # doctest: +SKIP
        """
        self._option_builder.build_options()

    def _get_config(self) -> MJCFImporterConfig:
        """Return the active importer configuration.

        Returns:
            Current importer configuration.

        Example:

        .. code-block:: python

            >>> from isaacsim.asset.importer.mjcf.ui.impl.extension import Extension
            >>> Extension()._get_config()  # doctest: +SKIP
            <...>
        """
        return self._last_config

    def reset_config(self) -> None:
        """Reset importer configuration to default values.

        Example:

        .. code-block:: python

            >>> from isaacsim.asset.importer.mjcf.ui.impl.extension import Extension
            >>> Extension().reset_config()  # doctest: +SKIP
        """
        self._config.usd_path = None
        self._config.import_scene = True
        self._config.merge_mesh = False
        self._config.debug_mode = False
        self._config.collision_from_visuals = False
        self._config.collision_type = "default"
        self._config.allow_self_collision = False

    def _start_import(self, path: str | None = None, **kargs: object) -> str | None:
        """Start the MJCF import process.

        Args:
            path: Path to the MJCF file to import.
            **kargs: Additional keyword arguments.

        Returns:
            Path to the imported prim in the USD stage, or None if import failed.

        Example:

        .. code-block:: python

            >>> from isaacsim.asset.importer.mjcf.ui.impl.extension import Extension
            >>> Extension()._start_import(None)  # doctest: +SKIP
        """
        if not path:
            carb.log_error("MJCF Importer: No path provided")
            return None

        # Get user's export folder preference
        export_folder = self._models["dst_path"].get_value_as_string() if self._models["dst_path"] else ""

        carb.log_info(f"export_folder: {export_folder}")

        # Treat "default" string as empty
        if export_folder == "Same as Imported Model(Default)":
            export_folder = ""

        # Determine USD output directory
        if export_folder:
            # User specified a custom output folder
            if not os.path.isdir(export_folder):
                export_folder = os.path.dirname(export_folder)
        else:
            # Use the directory containing the MJCF file
            export_folder = os.path.dirname(path)

        # Update configuration with paths and UI options, the other options are set in the OptionWidget class
        self._config.mjcf_path = path
        self._config.usd_path = export_folder

        # Print configuration for debugging
        stage_utils.create_new_stage()

        # Perform the import
        self._importer.config = self._config
        output_path = self._importer.import_mjcf()
        if not output_path:
            carb.log_error(f"Failed to import MJCF file at path: {path}")
            return None
        result, _ = stage_utils.open_stage(output_path)

        self._last_config = copy.deepcopy(self._config)  # for testing only
        self.reset_config()
        if not result:
            carb.log_error(f"Failed to open stage at path: {output_path}")
            return None
        return output_path

    def _print_config(self) -> None:
        """Log the current importer configuration."""
        carb.log_info(f"config mjcf path: {self._config.mjcf_path}")
        carb.log_info(f"config usd path: {self._config.usd_path}")
        carb.log_info(f"config import scene: {self._config.import_scene}")
        carb.log_info(f"config merge mesh: {self._config.merge_mesh}")
        carb.log_info(f"config debug mode: {self._config.debug_mode}")
        carb.log_info(f"config collision from visuals: {self._config.collision_from_visuals}")
        carb.log_info(f"config collision type: {self._config.collision_type}")
        carb.log_info(f"config allow self collision: {self._config.allow_self_collision}")

    def on_shutdown(self) -> None:
        """Clean up resources when the extension is shut down.

        Example:

        .. code-block:: python

            >>> from isaacsim.asset.importer.mjcf.ui.impl.extension import Extension
            >>> Extension().on_shutdown()  # doctest: +SKIP
        """
        if hasattr(self, "_delegate"):
            self._delegate.destroy()
            ai.remove_importer(self._delegate)

        global _extension_instance
        _extension_instance = None
        gc.collect()


class MjcfImporterDelegate(ai.AbstractImporterDelegate):
    """Delegate implementation for the MJCF importer UI.

    Args:
        name: Importer display name.
        filters: Regex filters for supported files.
        descriptions: Human-readable filter descriptions.

    Example:

    .. code-block:: python

        >>> from isaacsim.asset.importer.mjcf.ui.impl.extension import MjcfImporterDelegate
        >>> MjcfImporterDelegate("MJCF", [".*"], ["XML"])  # doctest: +SKIP
    """

    def __init__(self, name: str, filters: list[str], descriptions: list[str]) -> None:
        super().__init__()
        self._name = name
        self._filters = filters
        self._descriptions = descriptions
        self._importer: Extension | None = None
        # register the mjcf icon to asset types
        ext_path = omni.kit.app.get_app().get_extension_manager().get_extension_path_by_module(__name__)
        icon_path = Path(ext_path).joinpath("icons").absolute()
        AssetTypeDef = namedtuple("AssetTypeDef", "glyph thumbnail matching_exts")
        known_asset_types = asset_types.known_asset_types()
        known_asset_types["mjcf"] = AssetTypeDef(
            f"{icon_path}/icoFileMJCF.png",
            f"{icon_path}/icoFileMJCF.png",
            [".xml", ".XML"],
        )

    def set_importer(self, importer: Extension) -> None:
        """Assign the importer instance used by this delegate.

        Args:
            importer: MJCF UI extension instance.

        Example:

        .. code-block:: python

            >>> from isaacsim.asset.importer.mjcf.ui.impl.extension import Extension, MjcfImporterDelegate
            >>> delegate = MjcfImporterDelegate("MJCF", [".*"], ["XML"])
            >>> delegate.set_importer(Extension())  # doctest: +SKIP
        """
        self._importer = importer

    def show_destination_frame(self) -> bool:
        """Return whether the destination frame should be shown.

        Returns:
            False, because MJCF handles destination paths directly.

        Example:

        .. code-block:: python

            >>> from isaacsim.asset.importer.mjcf.ui.impl.extension import MjcfImporterDelegate
            >>> MjcfImporterDelegate("MJCF", [".*"], ["XML"]).show_destination_frame()
            False
        """
        return False

    def destroy(self) -> None:
        """Release references held by the delegate.

        Example:

        .. code-block:: python

            >>> from isaacsim.asset.importer.mjcf.ui.impl.extension import MjcfImporterDelegate
            >>> delegate = MjcfImporterDelegate("MJCF", [".*"], ["XML"])
            >>> delegate.destroy()
        """
        self._importer = None

    def _on_import_complete(self, file_paths: list[str]) -> None:
        """Handle import completion callbacks.

        Args:
            file_paths: List of imported file paths.
        """

    @property
    def name(self) -> str:
        """Return the importer name.

        Returns:
            Importer display name.

        Example:

        .. code-block:: python

            >>> from isaacsim.asset.importer.mjcf.ui.impl.extension import MjcfImporterDelegate
            >>> delegate = MjcfImporterDelegate("MJCF", [".*"], ["XML"])
            >>> delegate.name
            'MJCF'
        """
        return self._name

    @property
    def filter_regexes(self) -> list[str]:
        """Return the importer filter regexes.

        Returns:
            List of regex strings.

        Example:

        .. code-block:: python

            >>> from isaacsim.asset.importer.mjcf.ui.impl.extension import MjcfImporterDelegate
            >>> delegate = MjcfImporterDelegate("MJCF", [".*"], ["XML"])
            >>> delegate.filter_regexes
            ['.*']
        """
        return self._filters

    @property
    def filter_descriptions(self) -> list[str]:
        """Return the importer filter descriptions.

        Returns:
            List of filter descriptions.

        Example:

        .. code-block:: python

            >>> from isaacsim.asset.importer.mjcf.ui.impl.extension import MjcfImporterDelegate
            >>> delegate = MjcfImporterDelegate("MJCF", [".*"], ["XML"])
            >>> delegate.filter_descriptions
            ['XML']
        """
        return self._descriptions

    def build_options(self, paths: list[str]) -> None:
        """Build the options UI for the selected files.

        Args:
            paths: List of file paths selected for import.

        Example:

        .. code-block:: python

            >>> from isaacsim.asset.importer.mjcf.ui.impl.extension import MjcfImporterDelegate
            >>> delegate = MjcfImporterDelegate("MJCF", [".*"], ["XML"])
            >>> delegate.build_options([])  # doctest: +SKIP
        """
        if self._importer is not None:
            self._importer.build_new_options()
        else:
            carb.log_warn("MJCF Importer: Importer not initialized, cannot build options")

    async def convert_assets(self, paths: list[str], **kargs: object) -> dict:
        """Convert selected MJCF assets.

        Args:
            paths: List of MJCF file paths to convert.
            **kargs: Additional keyword arguments forwarded to the importer.

        Returns:
            Dictionary describing the conversion results.

        Example:

        .. code-block:: python

            >>> from isaacsim.asset.importer.mjcf.ui.impl.extension import MjcfImporterDelegate
            >>> delegate = MjcfImporterDelegate("MJCF", [".*"], ["XML"])
            >>> isinstance((await delegate.convert_assets([])), dict)  # doctest: +SKIP
            True
        """
        if not paths:
            post_notification(
                "No file selected",
                "Please select a file to import",
                NotificationStatus.ERROR,
            )
            return {}
        if self._importer is None:
            return {}
        for path in paths:
            self._importer._start_import(path, **kargs)
        return {}


def get_instance() -> Extension | None:
    """Return the active MJCF UI extension instance.

    Returns:
        Extension instance if initialized, otherwise None.

    Example:

    .. code-block:: python

        >>> from isaacsim.asset.importer.mjcf.ui.impl import extension
        >>> extension.get_instance()  # doctest: +SKIP
    """
    return _extension_instance
