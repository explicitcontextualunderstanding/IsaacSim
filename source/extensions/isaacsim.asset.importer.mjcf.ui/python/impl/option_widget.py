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
"""Option widget builders for the MJCF importer UI."""

from collections.abc import Callable

import omni.ui as ui
from isaacsim.asset.importer.mjcf.impl import MJCFImporterConfig

from .style import get_option_style
from .ui_utils import checkbox_builder, dropdown_builder, string_filed_builder


def option_header(collapsed: bool, title: str) -> None:
    """Build a collapsable frame header.

    Args:
        collapsed: Whether the frame is collapsed.
        title: Title text for the header.

    Example:

    .. code-block:: python

        >>> from isaacsim.asset.importer.mjcf.ui.impl import option_widget
        >>> option_widget.option_header(False, "Title")  # doctest: +SKIP
    """
    with ui.HStack(height=22):
        ui.Spacer(width=4)
        with ui.VStack(width=10):
            ui.Spacer()
            if collapsed:
                triangle = ui.Triangle(height=7, width=5)
                triangle.alignment = ui.Alignment.RIGHT_CENTER
            else:
                triangle = ui.Triangle(height=5, width=7)
                triangle.alignment = ui.Alignment.CENTER_BOTTOM
            ui.Spacer()
        ui.Spacer(width=4)
        ui.Label(title, name="collapsable_header", width=0)
        ui.Spacer(width=3)
        ui.Line()


def option_frame(
    title: str,
    build_content_fn: Callable[[], None],
    collapse_fn: Callable[[bool], None] | None = None,
) -> None:
    """Create a styled options frame and populate it.

    Args:
        title: Title text for the frame.
        build_content_fn: Callback that builds the inner UI content.
        collapse_fn: Optional callback invoked on collapse changes.

    Example:

    .. code-block:: python

        >>> from isaacsim.asset.importer.mjcf.ui.impl import option_widget
        >>> option_widget.option_frame("Options", lambda: None)  # doctest: +SKIP
    """
    with ui.CollapsableFrame(
        title, name="option", height=0, collapsed=False, build_header_fn=option_header, collapsed_changed_fn=collapse_fn
    ):
        with ui.HStack():
            ui.Spacer(width=2)
            build_content_fn()
            ui.Spacer(width=2)


class OptionWidget:
    """Build and manage importer option widgets.

    Args:
        models: Dictionary of UI models to populate.
        config: Importer configuration instance.

    Example:

    .. code-block:: python

        >>> from isaacsim.asset.importer.mjcf.ui.impl.option_widget import OptionWidget
        >>> OptionWidget({}, object())  # doctest: +SKIP
    """

    def __init__(self, models: dict[str, ui.AbstractValueModel], config: MJCFImporterConfig) -> None:
        self._models = models
        self._config = config

    @property
    def models(self) -> dict[str, ui.AbstractValueModel]:
        """Return the models dictionary.

        Returns:
            Dictionary of UI models keyed by option name.

        Example:

        .. code-block:: python

            >>> from isaacsim.asset.importer.mjcf.ui.impl.option_widget import OptionWidget
            >>> OptionWidget({}, object()).models  # doctest: +SKIP
            {}
        """
        return self._models

    @property
    def config(self):
        """Return the current configuration object.

        Returns:
            Configuration object used by the widget.

        Example:

        .. code-block:: python

            >>> from isaacsim.asset.importer.mjcf.ui.impl.option_widget import OptionWidget
            >>> OptionWidget({}, object()).config  # doctest: +SKIP
            <object object at ...>
        """
        return self._config

    def build_options(self) -> None:
        """Build all option frames for the importer UI.

        Example:

        .. code-block:: python

            >>> from isaacsim.asset.importer.mjcf.ui.impl.option_widget import OptionWidget
            >>> OptionWidget({}, object()).build_options()  # doctest: +SKIP
        """
        with ui.VStack(style=get_option_style()):
            self._build_model_frame()
            self._build_colliders_frame()
            self._build_options_frame()

    def _build_model_frame(self) -> None:
        """Build the model output frame."""

        def build_model_content() -> None:
            """Build the model output content."""
            with ui.VStack(spacing=4):
                ui.Label("USD Output")
                self._models["dst_path"] = string_filed_builder(
                    tooltip="USD file to store output usd in",
                    default_val="Same as Imported Model(Default)",
                    folder_dialog_title="Select Output File",
                    folder_button_title="Select File",
                    read_only=True,
                    identifier="mjcf_output_path",
                )

        option_frame("Output", build_model_content)

    def _build_colliders_frame(self) -> None:
        """Build the Colliders options frame.

        Creates UI elements for:
        - Collision from visuals checkbox
        - Collision type dropdown (shown only when collision from visuals is checked)
        - Allow self-collision checkbox
        """

        def build_colliders_content() -> None:
            """Build collider option content."""
            with ui.VStack(spacing=4):
                # Collision from visuals checkbox
                self._models["collision_from_visuals"] = checkbox_builder(
                    "Collision From Visuals",
                    tooltip="If True, collision geoms will be generated from visual geometries",
                    default_val=False,
                    on_clicked_fn=self._on_collision_from_visuals_changed,
                    identifier="mjcf_collision_from_visuals",
                )

                # Collision type dropdown (initially hidden, on separate line)
                self._collision_type_frame = ui.VStack(spacing=0, height=0)
                with self._collision_type_frame:
                    ui.Spacer(height=4)
                    with ui.HStack():
                        ui.Spacer(width=4)
                        with ui.VStack(width=ui.Fraction(1)):

                            def set_collision_type(value: str) -> None:
                                self._config.collision_type = value

                            self._models["collision_type"] = dropdown_builder(
                                "Collision Type",
                                tooltip="Type of collision geometry to use when generating from visuals",
                                default_val=0,
                                items=[
                                    "Convex Hull",
                                    "Convex Decomposition",
                                    "Bounding Sphere",
                                    "Bounding Cube",
                                ],
                                on_clicked_fn=set_collision_type,
                                identifier="mjcf_collision_type",
                            )
                        ui.Spacer(width=4)
                    ui.Spacer(height=4)

                # Initially hide collision type dropdown
                self._collision_type_frame.visible = False

                # Self-collision checkbox
                def set_allow_self_collision(value: bool) -> None:
                    self._config.allow_self_collision = value

                self._models["allow_self_collision"] = checkbox_builder(
                    "Allow Self-Collision",
                    tooltip="If true, allows self intersection between links in the robot, can cause instability if collision meshes between links are self intersecting",
                    default_val=False,
                    on_clicked_fn=set_allow_self_collision,
                    identifier="mjcf_allow_self_collision",
                )

        option_frame("Colliders", build_colliders_content)

    def _build_options_frame(self) -> None:
        """Build the Options frame.

        Creates UI elements for:
        - Layer Structure checkbox (default: True)
        - Merge Mesh checkbox (default: False)
        - Debug Mode checkbox (default: False)
        - Open Gains Tuner checkbox (default: False)
        """

        def build_options_content() -> None:
            """Build general option content."""
            with ui.VStack(spacing=4):

                # Import Scene checkbox
                def set_import_scene(value: bool) -> None:
                    self._config.import_scene = value

                self._models["import_scene"] = checkbox_builder(
                    "Import Scene",
                    tooltip="If True, imports the MJCF simulation settings along with the model",
                    default_val=True,
                    on_clicked_fn=set_import_scene,
                    identifier="mjcf_import_scene",
                )

                # Merge Mesh checkbox
                def set_merge_mesh(value: bool) -> None:
                    self._config.merge_mesh = value

                self._models["merge_mesh"] = checkbox_builder(
                    "Merge Mesh",
                    tooltip="If True, merges meshes where possible to optimize the model",
                    default_val=False,
                    on_clicked_fn=set_merge_mesh,
                    identifier="mjcf_merge_mesh",
                )

                # Debug Mode checkbox
                def set_debug_mode(value: bool) -> None:
                    self._config.debug_mode = value

                self._models["debug_mode"] = checkbox_builder(
                    "Debug Mode",
                    tooltip="If True, enables debug mode with additional logging and visualization",
                    default_val=False,
                    on_clicked_fn=set_debug_mode,
                    identifier="mjcf_debug_mode",
                )

                # Open Gains Tuner checkbox
                # self._models["open_gains_tuner"] = checkbox_builder(
                #     "Open Gains Tuner",
                #     tooltip="If True, opens the gains tuner after import for PID tuning",
                #     default_val=False,
                #     on_clicked_fn=lambda m, config=self._config: config.set_open_gains_tuner(m),
                # )

        option_frame("Options", build_options_content)

    def _on_collision_from_visuals_changed(self, value: bool) -> None:
        """Toggle visibility of collision type dropdown.

        Args:
            value: Boolean indicating if collision from visuals is enabled.

        Example:

        .. code-block:: python

            >>> from isaacsim.asset.importer.mjcf.ui.impl.option_widget import OptionWidget
            >>> OptionWidget({}, object())._on_collision_from_visuals_changed(True)  # doctest: +SKIP
        """
        self._config.collision_from_visuals = value
        self._collision_type_frame.visible = value
