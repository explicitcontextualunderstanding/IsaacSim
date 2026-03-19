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

"""Interactive Follow Target manipulation example extension for Isaac Sim demonstrating a Franka robot following a target cube."""


import asyncio
import os

import omni.ext
import omni.ui as ui
from isaacsim.examples.browser import get_instance as get_browser_instance
from isaacsim.examples.interactive.base_sample import BaseSampleUITemplate
from isaacsim.examples.interactive.follow_target import FollowTarget
from isaacsim.gui.components.ui_utils import btn_builder, get_style, setup_ui_headers, state_btn_builder, str_builder


class FollowTargetExtension(omni.ext.IExt):
    """Extension for the Follow Target manipulation example.

    Provides an interactive demonstration of a Franka robot following a target cube in Isaac Sim.
    Registers the example with the isaacsim.examples.browser for access through the Examples window.

    The extension creates a complete UI interface allowing users to:

    - Load the scene with a Franka robot and target cube
    - Start/stop the target following behavior
    - Add and remove obstacles dynamically during simulation
    - Control data logging for analysis
    - Save logged data to specified output directory

    Users can manipulate the target by selecting the 'Target Cube' in the Stage tree and dragging it around.
    The robot will continuously follow the target while avoiding any added obstacles. Multiple obstacles can
    be added and removed in reverse order.

    The extension integrates with the Isaac Sim examples browser system, categorizing itself under
    'Manipulation' examples. It provides both programmatic access through the FollowTarget sample class
    and an interactive UI through the FollowTargetUI template.
    """

    def on_startup(self, ext_id: str):
        """Called when the extension is enabled.

        Initializes the Follow Target example by setting up UI components and registering the example with the browser.

        Args:
            ext_id: The extension identifier.
        """
        self.example_name = "Follow Target"
        self.category = "Manipulation"

        ui_kwargs = {
            "ext_id": ext_id,
            "file_path": os.path.abspath(__file__),
            "title": "Follow Target Task",
            "doc_link": "https://docs.isaacsim.omniverse.nvidia.com/latest/introduction/examples.html",
            "overview": "This Example shows how to follow a target using Franka robot in Isaac Sim.Click 'Load' to load the scene, and 'Start' to start the following. \n\n To move the target, select 'Target Cube' on the Stage tree, then drag it around on stage. \n\nYou can add multiple obstacles. 'Removing' them will remove the obstacles in reverse order of when its added.",
            "sample": FollowTarget(),
        }

        ui_handle = FollowTargetUI(**ui_kwargs)

        get_browser_instance().register_example(
            name=self.example_name,
            execute_entrypoint=ui_handle.build_window,
            ui_hook=ui_handle.build_ui,
            category=self.category,
        )

        return

    def on_shutdown(self):
        """Called when the extension is disabled.

        Cleans up by deregistering the Follow Target example from the browser.
        """
        get_browser_instance().deregister_example(name=self.example_name, category=self.category)
        return


class FollowTargetUI(BaseSampleUITemplate):
    """User interface for the Follow Target manipulation example.

    Provides interactive controls for managing a Franka robot following a target cube in Isaac Sim.
    The interface includes task control buttons for starting/stopping the following behavior, adding and removing
    obstacles, and data logging capabilities with save functionality.

    The UI is organized into collapsible frames:
    - Task Control: Contains buttons to start/stop target following, add obstacles, and remove obstacles
    - Data Logging: Provides controls for starting/pausing data collection and saving logged data to a file

    Button states are automatically managed based on the current simulation state. For example, obstacle
    removal is only enabled when obstacles exist, and data saving is enabled after logging has been started.

    Args:
        *args: Variable length argument list passed to the parent BaseSampleUITemplate class.
        **kwargs: Additional keyword arguments passed to the parent BaseSampleUITemplate class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_extra_frames(self):
        """Creates additional UI frames for task controls and data logging.

        Builds the Task Control frame containing buttons for follow target, add obstacle, and remove obstacle operations.
        Also builds the Data Logging frame with controls for starting logging and saving data.
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
            with ui.CollapsableFrame(
                title="Data Logging",
                width=ui.Fraction(0.33),
                height=0,
                visible=True,
                collapsed=False,
                # style=get_style(),
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
            ):

                self.build_data_logging_ui()

        return

    def _on_follow_target_button_event(self, val):
        """Handles the follow target button click event.

        Args:
            val: Button state value indicating start or stop action.
        """
        asyncio.ensure_future(self.sample._on_follow_target_event_async(val))
        return

    def _on_add_obstacle_button_event(self):
        """Handles the add obstacle button click event.

        Adds a new obstacle to the scene and enables the Remove Obstacle button.
        """
        self.sample._on_add_obstacle_event()
        self.task_ui_elements["Remove Obstacle"].enabled = True
        return

    def _on_remove_obstacle_button_event(self):
        """Handles the remove obstacle button click event.

        Removes the most recently added obstacle from the scene and disables the Remove Obstacle button if no obstacles remain.
        """
        self.sample._on_remove_obstacle_event()
        world = self.sample.get_world()
        current_task = list(world.get_current_tasks().values())[0]
        if not current_task.obstacles_exist():
            self.task_ui_elements["Remove Obstacle"].enabled = False
        return

    def _on_logging_button_event(self, val):
        """Handles the logging button click event.

        Args:
            val: Button state value indicating start or pause logging action.
        """
        self.sample._on_logging_event(val)
        self.task_ui_elements["Save Data"].enabled = True
        return

    def _on_save_data_button_event(self):
        """Handles the save data button click event.

        Saves the logged data to the file path specified in the Output Directory field.
        """
        self.sample._on_save_data_event(self.task_ui_elements["Output Directory"].get_value_as_string())
        return

    def post_reset_button_event(self):
        """Updates UI control states after the reset button is pressed.

        Enables follow target, add obstacle, and start logging controls while resetting button states to their default values.
        """
        self.task_ui_elements["Follow Target"].enabled = True
        self.task_ui_elements["Remove Obstacle"].enabled = False
        self.task_ui_elements["Add Obstacle"].enabled = True
        self.task_ui_elements["Start Logging"].enabled = True
        self.task_ui_elements["Save Data"].enabled = False
        if self.task_ui_elements["Follow Target"].text == "STOP":
            self.task_ui_elements["Follow Target"].text = "START"
        return

    def post_load_button_event(self):
        """Updates UI control states after the load button is pressed.

        Enables follow target, add obstacle, and start logging controls to allow interaction with the loaded scene.
        """
        self.task_ui_elements["Follow Target"].enabled = True
        self.task_ui_elements["Add Obstacle"].enabled = True
        self.task_ui_elements["Start Logging"].enabled = True
        self.task_ui_elements["Save Data"].enabled = False
        return

    def post_clear_button_event(self):
        """Updates UI control states after the clear button is pressed.

        Disables all task control buttons and resets the follow target button text to START state.
        """
        self.task_ui_elements["Follow Target"].enabled = False
        self.task_ui_elements["Remove Obstacle"].enabled = False
        self.task_ui_elements["Add Obstacle"].enabled = False
        self.task_ui_elements["Start Logging"].enabled = False
        self.task_ui_elements["Save Data"].enabled = False
        if self.task_ui_elements["Follow Target"].text == "STOP":
            self.task_ui_elements["Follow Target"].text = "START"
        return

    def build_task_controls_ui(self):
        """Builds the task control UI elements.

        Creates Follow Target toggle button, Add Obstacle button, and Remove Obstacle button within a vertical stack layout.
        All buttons are initially disabled until a scene is loaded.
        """
        with ui.VStack(spacing=5):
            dict = {
                "label": "Follow Target",
                "type": "button",
                "a_text": "START",
                "b_text": "STOP",
                "tooltip": "Follow Target",
                "on_clicked_fn": self._on_follow_target_button_event,
            }
            self.task_ui_elements["Follow Target"] = state_btn_builder(**dict)
            self.task_ui_elements["Follow Target"].enabled = False

            dict = {
                "label": "Add Obstacle",
                "type": "button",
                "text": "ADD",
                "tooltip": "Add Obstacle",
                "on_clicked_fn": self._on_add_obstacle_button_event,
            }

            self.task_ui_elements["Add Obstacle"] = btn_builder(**dict)
            self.task_ui_elements["Add Obstacle"].enabled = False
            dict = {
                "label": "Remove Obstacle",
                "type": "button",
                "text": "REMOVE",
                "tooltip": "Remove Obstacle",
                "on_clicked_fn": self._on_remove_obstacle_button_event,
            }

            self.task_ui_elements["Remove Obstacle"] = btn_builder(**dict)
            self.task_ui_elements["Remove Obstacle"].enabled = False

    def build_data_logging_ui(self):
        """Builds the data logging UI controls.

        Creates UI elements for configuring output directory, starting/pausing data logging,
        and saving collected data to file.
        """
        with ui.VStack(spacing=5):
            dict = {
                "label": "Output Directory",
                "type": "stringfield",
                "default_val": os.path.join(os.getcwd(), "output_data.json"),
                "tooltip": "Output Directory",
                "on_clicked_fn": None,
                "use_folder_picker": False,
                "read_only": False,
            }
            self.task_ui_elements["Output Directory"] = str_builder(**dict)

            dict = {
                "label": "Start Logging",
                "type": "button",
                "a_text": "START",
                "b_text": "PAUSE",
                "tooltip": "Start Logging",
                "on_clicked_fn": self._on_logging_button_event,
            }
            self.task_ui_elements["Start Logging"] = state_btn_builder(**dict)
            self.task_ui_elements["Start Logging"].enabled = False

            dict = {
                "label": "Save Data",
                "type": "button",
                "text": "Save Data",
                "tooltip": "Save Data",
                "on_clicked_fn": self._on_save_data_button_event,
            }

            self.task_ui_elements["Save Data"] = btn_builder(**dict)
            self.task_ui_elements["Save Data"].enabled = False
        return
