# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Interactive bin filling demonstration extension that showcases robotic manipulation using a UR10 robot with realistic gripper physics in Isaac Sim."""


import asyncio
import os

import omni.ext
import omni.ui as ui
from isaacsim.examples.browser import get_instance as get_browser_instance
from isaacsim.examples.interactive.base_sample import BaseSampleUITemplate
from isaacsim.examples.interactive.bin_filling import BinFilling
from isaacsim.gui.components.ui_utils import btn_builder


class BinFillingExtension(omni.ext.IExt):
    """Extension for the Bin Filling interactive example.

    This extension provides an interactive demonstration of bin filling using a UR10 robot in Isaac Sim.
    It showcases a realistic surface gripper that can break under heavy bin loads, providing insights into
    manipulation challenges in industrial automation scenarios.

    The extension registers itself with the examples browser and creates a user interface that allows users
    to control the bin filling simulation. Users can start the bin filling process and observe how the robot
    handles different scenarios, including gripper failure conditions.

    The example is designed to teach manipulation concepts and demonstrate realistic physical interactions
    between robotic systems and objects in a bin filling context.
    """

    def on_startup(self, ext_id: str):
        """Initializes the Bin Filling extension when it starts up.

        Registers the bin filling example with the browser instance, setting up the UI components
        and documentation links for the manipulation tutorial.

        Args:
            ext_id: The extension identifier.
        """
        self.example_name = "Bin Filling"
        self.category = "Manipulation"

        ui_kwargs = {
            "ext_id": ext_id,
            "file_path": os.path.abspath(__file__),
            "title": "Bin Filling",
            "doc_link": "https://docs.isaacsim.omniverse.nvidia.com/latest/core_api_tutorials/tutorial_core_adding_manipulator.html",
            "overview": "This Example shows how to do bin filling using UR10 robot in Isaac Sim.\n It showcases a realistic surface gripper that breaks with heavy bin load.\nPress the 'Open in IDE' button to view the source code.",
            "sample": BinFilling(),
        }

        ui_handle = BinFillingUI(**ui_kwargs)

        get_browser_instance().register_example(
            name=self.example_name,
            execute_entrypoint=ui_handle.build_window,
            ui_hook=ui_handle.build_ui,
            category=self.category,
        )

        return

    def on_shutdown(self):
        """Cleans up the Bin Filling extension when it shuts down.

        Deregisters the bin filling example from the browser instance to ensure proper cleanup.
        """
        get_browser_instance().deregister_example(name=self.example_name, category=self.category)
        return


class BinFillingUI(BaseSampleUITemplate):
    """User interface for the bin filling example in Isaac Sim.

    Provides an interactive UI for controlling a UR10 robot performing bin filling tasks. The interface includes
    task controls with a "Start Bin Filling" button that triggers an asynchronous bin filling operation. The UI
    extends the base sample template to provide a consistent interface experience with additional frames for
    task-specific controls.

    The interface automatically manages button states, disabling the start button during active operations and
    re-enabling it after reset or load operations. It integrates with the BinFilling sample class to execute
    the actual bin filling logic.

    Args:
        *args: Variable length argument list passed to the parent BaseSampleUITemplate class.
        **kwargs: Additional keyword arguments passed to the parent class. Typically includes UI configuration
            parameters such as ext_id, file_path, title, doc_link, overview, and sample instance.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_extra_frames(self):
        """Builds the extra UI frames for the bin filling task control panel.

        Creates a collapsible frame containing task control UI elements with scrollbars
        for managing bin filling operations.
        """
        extra_stacks = self.get_extra_frames_handle()
        self.task_ui_elements = {}

        with extra_stacks:
            with ui.CollapsableFrame(
                title="Task Control",
                width=ui.Fraction(0.33),
                height=0,
                visible=True,
                collapsed=False,
                # style=get_style(),
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
            ):
                self.build_task_controls_ui()

    def _on_fill_bin_button_event(self):
        """Handles the fill bin button click event.

        Triggers the asynchronous bin filling operation and disables the start button
        to prevent multiple simultaneous operations.
        """
        asyncio.ensure_future(self.sample.on_fill_bin_event_async())
        self.task_ui_elements["Start Bin Filling"].enabled = False
        return

    def post_reset_button_event(self):
        """Handles post-reset operations for the UI.

        Re-enables the "Start Bin Filling" button after a reset operation.
        """
        self.task_ui_elements["Start Bin Filling"].enabled = True
        return

    def post_load_button_event(self):
        """Handles post-load operations for the UI.

        Re-enables the "Start Bin Filling" button after a load operation.
        """
        self.task_ui_elements["Start Bin Filling"].enabled = True
        return

    def post_clear_button_event(self):
        """Handles post-clear operations for the UI.

        Disables the "Start Bin Filling" button after a clear operation.
        """
        self.task_ui_elements["Start Bin Filling"].enabled = False
        return

    def build_task_controls_ui(self):
        """Builds the task control UI elements.

        Creates the "Start Bin Filling" button with appropriate tooltip and click handler.
        The button is initially disabled until the scene is properly loaded.
        """
        with ui.VStack(spacing=5):

            dict = {
                "label": "Start Bin Filling",
                "type": "button",
                "text": "Start Bin Filling",
                "tooltip": "Start Bin Filling",
                "on_clicked_fn": self._on_fill_bin_button_event,
            }

            self.task_ui_elements["Start Bin Filling"] = btn_builder(**dict)
            self.task_ui_elements["Start Bin Filling"].enabled = False
