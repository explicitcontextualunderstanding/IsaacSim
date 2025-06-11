# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import math
import traceback
from functools import partial

import carb
import numpy as np
import omni.timeline
import omni.ui as ui
import pxr
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.gui.components.element_wrappers import (
    Button,
    CheckBox,
    CollapsableFrame,
    DropDown,
    FloatField,
    StateButton,
    TextBlock,
    XYPlot,
)
from omni.usd import StageEventType
from pxr import Sdf, Usd
from usd.schema.isaac.robot_schema import Classes

from ..gains_tuner import GainsTestMode, GainTuner
from ..global_variables import EXTENSION_TITLE
from .color_table_widget import ColorJointWidget
from .dropdown_widget import CustomDropDown
from .frame_widget import CustomCollapsableFrame as CollapsableFrame
from .joint_table_widget import (
    JointSettingMode,
    JointWidget,
    get_damping_attr,
    get_joint_drive_type_attr,
    get_mimic_damping_ratio_attr,
    get_mimic_natural_frequency_attr,
    get_stiffness_attr,
    is_joint_mimic,
)
from .plot_widget import CustomXYPlot
from .style import get_style
from .test_table_widget import TestJointWidget


class UIBuilder:
    def __init__(self):
        # Frames are sub-windows that can contain multiple UI elements
        self.frames = []
        # UI elements created using a UIElementWrapper instance
        self.wrapped_ui_elements = []

        # Get access to the timeline to control stop/pause/play programmatically
        self._timeline = omni.timeline.get_timeline_interface()

        self._gains_tuner = GainTuner()

        self._test_mode = GainsTestMode.SINUSOIDAL
        self._test_running = False
        self._joint_setting_mode = JointSettingMode.STIFFNESS

        self._reset_ui_next_frame = False
        self._make_plot_on_next_frame = False

        self._gains_table_widget = None
        self._test_table_widget = None
        self._test_button = None
        self._color_joint_widget = None
        self._position_frame = None
        self._velocity_frame = None
        self._plotting_indices = []
        self._plotting_colors = []
        self._rebuild_gains_widget = False
        self.force_query_mass = True
        self._gains_tuner = GainTuner()

    ###################################################################################
    #           The Functions Below Are Called Automatically By extension.py
    ###################################################################################

    def on_menu_callback(self):
        """Callback for when the UI is opened from the toolbar.
        This is called directly after build_ui().
        """
        # Reset internal state when UI window is closed and reopened
        self._invalidate_articulation()

        # Handles the case where the user loads their Articulation and
        # presses play before opening this extension
        # self._articulation_menu.repopulate()
        # self._stop_text.visible = True

    def on_timeline_event(self, event):
        """Callback for Timeline events (Play, Pause, Stop)

        Args:
            event (omni.timeline.TimelineEventType): Event Type
        """
        if self._articulation_menu.get_items() == []:
            return
        if event.type == int(omni.timeline.TimelineEventType.PLAY):
            self._gains_tuner.initialize()
            self._gains_tuning_frame.collapsed = True
            self._test_gains_frame.collapsed = False
            if self._test_button:
                self._test_button.enabled = True
            if self._articulation_menu:
                self._articulation_menu.enabled = False
        if event.type == int(omni.timeline.TimelineEventType.STOP):
            self._gains_tuning_frame.collapsed = False
            self._test_gains_frame.collapsed = True
            if self._test_button:
                self._test_button.enabled = False
            if self._articulation_menu:
                self._articulation_menu.enabled = True

    def on_physics_step(self, step: float):
        """Callback for Physics Step.
        Physics steps only occur when the timeline is playing

        Args:
            step (float): Size of physics step
        """
        pass

    def on_render_step(self, e: carb.events.IEvent):
        """Render event set up to cancel physics subscriptions that run the gains test.

        Args:
            e (carb.events.IEvent): _description_
        """

        if self._articulation_menu.get_items() == []:
            return
        if self._reset_ui_next_frame:
            if self._make_plot_on_next_frame:
                # TODO: rebuild plot here
                self._charts_frame.enabled = True
                self._charts_frame.visible = True
                self._charts_frame.rebuild()

            self._gains_tuning_frame.enabled = True
            self._test_gains_frame.enabled = True
            self._reset_ui_next_frame = False

        if not self._gains_tuner._joint_acumulated_inertia:
            if self._gains_tuner.compute_joints_acumulated_inertia():
                self._rebuild_gains_widget = True
            else:
                self._timeline.play()
                self.force_query_mass = True
                self._rebuild_gains_widget = True
        elif self._rebuild_gains_widget:
            if self.force_query_mass:
                self._timeline.stop()
            self._gains_tuning_frame.rebuild()
            self._test_gains_frame.rebuild()
            self._rebuild_gains_widget = False

    def on_stage_event(self, event):
        """Callback for Stage Events

        Args:
            event (omni.usd.StageEventType): Event Type
        """
        if event.type == int(omni.usd.StageEventType.ASSETS_LOADED):  # Any asset added or removed
            self._articulation_menu.repopulate()
            # self._gains_tuning_frame.rebuild()
        elif event.type == int(omni.usd.StageEventType.SIMULATION_START_PLAY):  # Timeline played
            if self._articulation_menu.get_items() == []:
                return
            if self._gains_tuner._articulation_root is None:
                self._articulation_menu.repopulate()
            self._gains_tuner.initialize()
        elif event.type == int(omni.usd.StageEventType.SIMULATION_STOP_PLAY):  # Timeline stopped
            if self._articulation_menu.get_items() == []:
                self._articulation_menu.repopulate()
                self._test_gains_frame.rebuild()
                return
            # Ignore pause events
            if self._timeline.is_stopped():
                self._articulation_menu.repopulate()

    def cleanup(self):
        """
        Called when the stage is closed or the extension is hot reloaded.
        Perform any necessary cleanup such as removing active callback functions
        Buttons imported from isaacsim.gui.components.element_wrappers implement a cleanup function that should be called
        """
        for ui_elem in self.wrapped_ui_elements:
            ui_elem.cleanup()

    def _on_help_click(self, b):
        print("Help clicked")

    def _populate_robot_menu(self):
        items = []
        stage = omni.usd.get_context().get_stage()
        if stage:
            for prim in pxr.Usd.PrimRange(stage.GetPrimAtPath("/")):
                # Get prim type get_prim_object_type
                if prim.HasAPI(Classes.ROBOT_API.value):
                    path = str(prim.GetPath())
                    items.append(path)

        return items

    def build_ui(self):

        with ui.VStack(style=get_style(), spacing=5, height=0):
            with ui.HStack(height=34):
                ui.Label("Robot Selection", name="robot_header")
                ui.Image(
                    name="help",
                    height=28,
                    width=28,
                    mouse_pressed_fn=lambda x, y, b, a: self._on_help_click(b),
                )
            self._articulation_menu = CustomDropDown(
                "Select Robot:",
                tooltip="Select from Articulations found on the stage after the timeline has been played.",
                on_selection_fn=self._on_articulation_selection,
                keep_old_selections=True,
            )
            self._articulation_menu.set_populate_fn(self._populate_robot_menu, repopulate=False)

        self._gains_tuning_frame = CollapsableFrame(
            "Tune Gains",
            collapsed=self._articulation_menu.get_items() == [],
            enabled=True,
            build_fn=self._build_gains_tuning_frame,
            show_copy_button=True,
        )

        self._test_gains_frame = CollapsableFrame(
            "Test Gains Settings", collapsed=True, enabled=True, build_fn=self._build_test_gains_frame
        )

        self._charts_frame = CollapsableFrame("Charts", collapsed=True, enabled=True, build_fn=self._build_charts_frame)

        self._articulation_menu.repopulate()
        if self._articulation_menu.get_items() != []:
            self._on_articulation_selection(self._articulation_menu.get_items()[0])
        self._rebuild_gains_widget = True

    def _build_gains_tuning_frame(self):

        with self._gains_tuning_frame.frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                with ui.HStack():
                    ui.Spacer(width=10)
                    self._edit_mode_collection = ui.RadioCollection()
                    with ui.HStack(width=0):
                        with ui.HStack(width=0):
                            with ui.VStack(width=0):
                                ui.Spacer()
                                ui.RadioButton(width=20, height=20, radio_collection=self._edit_mode_collection)
                                ui.Spacer()
                            ui.Spacer(width=4)
                            ui.Label(
                                "Stiffness",
                                width=0,
                                mouse_pressed_fn=lambda x, y, m, w: self._edit_mode_collection.model.set_value(0),
                            )
                        ui.Spacer(width=10)
                        with ui.HStack(width=0):
                            with ui.VStack(width=0):
                                ui.Spacer()
                                ui.RadioButton(width=20, height=20, radio_collection=self._edit_mode_collection)
                                ui.Spacer()
                            ui.Spacer(width=4)
                            ui.Label(
                                "Natural Frequency",
                                mouse_pressed_fn=lambda x, y, m, w: self._edit_mode_collection.model.set_value(1),
                            )
                    self._edit_mode_collection.model.set_value(0)
                    self._edit_mode_collection.model.add_value_changed_fn(lambda m: self._switch_tuning_mode(m))

                if not self._gains_tuner._joint_acumulated_inertia:
                    if self._gains_tuner.compute_joints_acumulated_inertia():
                        self._rebuild_gains_widget = True
                else:
                    self._gains_table_widget = JointWidget(
                        self._gains_tuner._joints.values(), self._gains_tuner._joint_acumulated_inertia
                    )
                ui.Spacer(height=5)
                with ui.HStack(width=ui.Fraction(1)):
                    ui.Spacer(width=ui.Fraction(1))
                    ui.Button(
                        "Save Gains to Physics Layer".upper(),
                        height=30,
                        width=210,
                        clicked_fn=self._on_save_gains_to_physics_layer,
                    )
                    ui.Spacer(width=5)
                    ui.Spacer(width=10)
                    with ui.Frame(width=0):
                        with ui.VStack():
                            ui.Spacer()
                            with ui.Placer(offset_x=0, offset_y=5):
                                ui.Rectangle(height=5, width=5, alignment=ui.Alignment.CENTER)
                            ui.Spacer()
                    ui.Spacer(width=5)
                ui.Spacer(height=5)
                self._no_permision_frame = ui.Frame(width=ui.Fraction(1))
                with self._no_permision_frame:
                    with ui.HStack(width=ui.Fraction(1)):
                        ui.Spacer(width=ui.Fraction(1))
                        ui.Label(
                            "No permission to save gains to physics layer.\nYou can still save the gains to the current stage as an override to the original values.",
                            width=0,
                            alignment=ui.Alignment.RIGHT_CENTER,
                            style={"color": 0xFF6666FF},
                        )
                        ui.Spacer(width=5)
                        ui.Spacer(width=10)
                        with ui.Frame(width=0):
                            with ui.VStack():
                                ui.Spacer()
                                with ui.Placer(offset_x=0, offset_y=5):
                                    ui.Rectangle(height=5, width=5, alignment=ui.Alignment.CENTER)
                                ui.Spacer()
                        ui.Spacer(width=5)
                self._no_permision_frame.visible = False
                ui.Spacer(height=5)

    def _on_save_gains_to_physics_layer(self, *args):
        joint_gains = self._gains_table_widget.model.get_item_children()
        edits = {}
        self._no_permision_frame.visible = False
        for joint_gain in joint_gains:
            joint = joint_gain.joint
            attrs = []
            if is_joint_mimic(joint):
                attrs = [get_mimic_natural_frequency_attr(joint), get_mimic_damping_ratio_attr(joint)]
            else:
                attrs = [get_stiffness_attr(joint), get_damping_attr(joint), get_joint_drive_type_attr(joint)]
            for attr in attrs:
                current_value = attr.Get()
                spec_stack = attr.GetPropertyStack()
                if spec_stack:
                    defining_spec = spec_stack[-1]
                    if defining_spec.layer not in edits:
                        edits[defining_spec.layer] = []
                    edits[defining_spec.layer].append((defining_spec.path, current_value))

        # Now apply all changes within edit contexts
        for layer, edits in edits.items():
            if layer.permissionToEdit and layer.permissionToSave:
                edit_stage = Usd.Stage.Open(layer.identifier)
                for path, value in edits:
                    attr = edit_stage.GetAttributeAtPath(path)
                    attr.Set(value)
                edit_stage.Save()
            else:
                self._no_permision_frame.visible = True

    def _switch_tuning_mode(self, switch):
        if self._gains_table_widget:
            self._gains_table_widget.switch_mode(JointSettingMode(switch.get_value_as_int()))

    def _on_test_duration_changed(self, model):
        self._gains_tuner.test_duration = model.get_value_as_float()

    def _build_test_gains_frame(self):
        if not self._gains_tuner.initialized:
            if self._articulation_menu.get_items() != []:
                self._gains_tuner.initialize()
            else:
                self._test_table_widget = None
                ui.Label("Start Simulation to run Tests", name="gains_tuner_not_initialized")
                return
        with ui.VStack(style=get_style(), spacing=5, height=0):
            with ui.HStack():
                ui.Spacer(width=10)
                ui.Label("Test Duration (s)", width=0)
                ui.Spacer(width=10)
                self._test_duration_field = ui.FloatField(
                    value=5.0,
                    # width=ui.Fraction(1),
                    height=20,
                    step=0.1,
                    min_value=0.0,
                    max_value=10.0,
                )
                self._test_duration_field.model.set_value(5)
                self._gains_tuner.set_test_duration(5)
                self._test_duration_field.model.add_value_changed_fn(lambda m: self._on_test_duration_changed(m))
            with ui.HStack():
                ui.Spacer(width=10)
                self._test_mode_collection = ui.RadioCollection()
                with ui.HStack(width=0):
                    with ui.HStack(width=0):
                        with ui.VStack(width=0):
                            ui.Spacer()
                            ui.RadioButton(width=20, height=20, radio_collection=self._test_mode_collection)
                            ui.Spacer()
                        ui.Spacer(width=4)
                        ui.Label(
                            "Sinewave",
                            width=0,
                            mouse_pressed_fn=lambda x, y, m, w: self._test_mode_collection.model.set_value(0),
                        )
                    ui.Spacer(width=10)
                    with ui.HStack(width=0):
                        with ui.VStack(width=0):
                            ui.Spacer()
                            ui.RadioButton(width=20, height=20, radio_collection=self._test_mode_collection)
                            ui.Spacer()
                        ui.Spacer(width=4)
                        ui.Label(
                            "Step Function",
                            mouse_pressed_fn=lambda x, y, m, w: self._test_mode_collection.model.set_value(1),
                        )
                    ui.Spacer(width=10)
                    # with ui.HStack(width=0):
                    #     with ui.VStack(width=0):
                    #         ui.Spacer()
                    #         ui.RadioButton(width=20, height=20, radio_collection=self._test_mode_collection)
                    #         ui.Spacer()
                    #     ui.Spacer(width=4)
                    #     ui.Label(
                    #         "User Provided Telemetry",
                    #         mouse_pressed_fn=lambda x, y, m, w: self._test_mode_collection.model.set_value(1),
                    #     )
                self._test_mode_collection.model.set_value(0)
                self._test_mode = GainsTestMode.SINUSOIDAL
                self._test_mode_collection.model.add_value_changed_fn(lambda m: self._switch_test_mode(m))
            self._test_table_widget = TestJointWidget(self._gains_tuner)
            ui.Spacer(height=10)
            with ui.HStack(height=40):

                ui.Spacer(width=ui.Fraction(1))
                with ui.Frame(width=0):
                    self._test_button = StateButton(
                        "",
                        "Run Test",
                        "Cancel Test",
                        on_a_click_fn=partial(self._on_run_gains_test, self._test_mode),
                        on_b_click_fn=self._on_cancel_gains_test,
                        physics_callback_fn=self._update_gains_test,
                    )
                # ui.Spacer(width=10)
                self._test_button.enabled = self._timeline.is_playing()
                self._articulation_menu.enabled = not self._timeline.is_playing()

    def _on_cancel_gains_test(self):
        self._test_running = False
        if self._test_button:
            self._test_button.reset()
        self._gains_tuner.stop_test()

    def _switch_test_mode(self, switch):
        self._gains_tuner.test_mode = switch.get_value_as_int()
        if self._test_table_widget:
            self._test_table_widget.switch_mode(self._gains_tuner.test_mode)

    def _build_charts_frame(self):
        if not self._gains_tuner.initialized:
            return
        with ui.HStack(style=get_style(), spacing=5, height=0):
            self._color_joint_widget = ColorJointWidget(
                self._gains_tuner, selected_changed_fn=self._on_color_joint_selection
            )
            if not self._gains_tuner.is_data_ready():
                return
            with ui.VStack(style=get_style(), spacing=5, height=0):

                self._position_frame = CollapsableFrame("Position charts", collapsed=False, show_copy_button=True)
                self._position_frame.set_build_fn(self._build_position_plot)

                self._velocity_frame = CollapsableFrame("Velocity charts", collapsed=False, show_copy_button=True)
                self._velocity_frame.set_build_fn(self._build_velocity_plot)

    def _on_color_joint_selection(self, selection):
        self._plotting_indices = [item.joint_index for item in selection]
        group_colors = [item.colors for item in selection]
        self._plotting_colors = []
        for i in range(2):
            for colors in group_colors:
                self._plotting_colors.append(colors[i])
        if self._position_frame:
            self._position_frame.rebuild()
        if self._velocity_frame:
            self._velocity_frame.rebuild()

    def _build_position_plot(self):
        if len(self._plotting_indices) == 0:
            return
        joint_indices = self._plotting_indices
        cmd_pos_list = []
        obs_pos_list = []
        cmd_vel_list = []
        obs_vel_list = []
        cmd_times_list = []
        for joint_index in joint_indices:
            (cmd_pos, cmd_vel, obs_pos, obs_vel, cmd_times) = self._gains_tuner.get_joint_states_from_gains_test(
                joint_index
            )
            if cmd_pos is None:
                continue
            cmd_pos_list.append(cmd_pos)
            obs_pos_list.append(obs_pos)
            cmd_vel_list.append(cmd_vel)
            obs_vel_list.append(obs_vel)
            cmd_times_list.append(cmd_times)

        plot = CustomXYPlot(
            show_legend=True,
            x_data=cmd_times_list + cmd_times_list,
            # TODO: not sure how to get user  data here, use obs_pos_list for now
            y_data=cmd_pos_list + obs_pos_list,
            data_colors=self._plotting_colors,
            header_count=2,
        )
        plot.set_data_colors(self._plotting_colors)

    def _build_velocity_plot(self):
        if len(self._plotting_indices) == 0:
            return
        joint_indices = self._plotting_indices
        cmd_pos_list = []
        cmd_vel_list = []
        obs_pos_list = []
        obs_vel_list = []
        cmd_times_list = []
        for joint_index in joint_indices:
            (cmd_pos, cmd_vel, obs_pos, obs_vel, cmd_times) = self._gains_tuner.get_joint_states_from_gains_test(
                joint_index
            )
            if cmd_pos is None:
                continue
            cmd_pos_list.append(cmd_pos)
            cmd_vel_list.append(cmd_vel)
            obs_pos_list.append(obs_pos)
            obs_vel_list.append(obs_vel)
            cmd_times_list.append(cmd_times)
        plot = CustomXYPlot(
            show_legend=True,
            x_data=cmd_times_list + cmd_times_list,
            # TODO: not sure how to get user  data here, use obs_vel_list for now
            y_data=cmd_vel_list + obs_vel_list,
            data_colors=self._plotting_colors,
            header_count=2,
        )
        plot.set_data_colors(self._plotting_colors)

    def _invalidate_articulation(self):
        """
        This function handles the event that the existing articulation becomes invalid and there is
        not a new articulation to select.  It is called explicitly in the code when the timeline is
        stopped and when the DropDown menu finds no articulations on the stage.
        """
        self._gains_tuner.on_reset()

    def _on_articulation_selection(self, articulation_path):

        if articulation_path is None:
            self._invalidate_articulation()
            return

        if self._gains_tuner._robot_prim_path != articulation_path:
            self._gains_tuner.reset()
            self._gains_tuner.setup(articulation_path)
            self._gains_tuning_frame.rebuild()
            self._test_running = False
            self._test_mode = GainsTestMode.SINUSOIDAL
            if self._test_table_widget:
                self._test_table_widget.switch_mode(self._test_mode)
            # self._test_gains_frame.rebuild()
            self._reset_ui_next_frame = True

    def _update_gains_test(self, step: float):
        if not self._test_running:
            return
        done = self._gains_tuner.update_gains_test(step)
        if done:
            self._reset_ui_next_frame = True
            if self._gains_tuner.is_data_ready():
                self._make_plot_on_next_frame = True
                self._make_plot_on_next_frame = True
                self._test_running = False
                self._test_button.reset()

    def _on_run_gains_test(self, gains_test_mode):
        if not self._gains_tuner.initialized:
            self._gains_tuner.initialize()
        """Disable all buttons until the gains test has finished."""
        self._gains_tuning_frame.collapsed = True
        # self._gains_tuning_frame.enabled = False

        test_mode = self._test_table_widget.mode

        joint_params = [p for p in self._test_table_widget.model.get_item_children() if p.test]

        # First create the base dictionary
        test_params = {
            "test_mode": test_mode,
            "test_duration": self._test_duration_field.model.get_value_as_float(),
            "joint_indices": [param.joint_index for param in joint_params],
        }

        # Then add the sequence dictionary separately
        test_params["sequence"] = [
            {
                "joint_indices": np.array(
                    [param.joint_index for param in joint_params if param.sequence == i], dtype=np.int32
                ),
                "joint_amplitudes": np.array(
                    [param.amplitude * 0.005 for param in joint_params if param.sequence == i], dtype=np.float32
                ),
                "joint_offsets": np.array(
                    [param.offset for param in joint_params if param.sequence == i], dtype=np.float32
                ),
                "joint_periods": np.array(
                    [param.period for param in joint_params if param.sequence == i], dtype=np.float32
                ),
                "joint_phases": np.array(
                    [param.phase for param in joint_params if param.sequence == i], dtype=np.float32
                ),
                "joint_step_max": np.array(
                    [param.step_max for param in joint_params if param.sequence == i], dtype=np.float32
                ),
                "joint_step_min": np.array(
                    [param.step_min for param in joint_params if param.sequence == i], dtype=np.float32
                ),
                "joint_user_provided": [param.user_provided for param in joint_params if param.sequence == i],
            }
            for i in set([p.sequence for p in joint_params])
        ]

        self._gains_tuner.initialize_gains_test(test_params)
        self._test_running = True
