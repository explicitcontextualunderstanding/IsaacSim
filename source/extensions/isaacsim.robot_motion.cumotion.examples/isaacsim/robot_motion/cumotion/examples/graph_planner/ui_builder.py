# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import asyncio

import numpy as np
import omni.kit.app
import omni.timeline
import omni.ui as ui
from isaacsim.core.experimental.objects import Cube
from isaacsim.core.experimental.prims import Articulation, GeomPrim
from isaacsim.core.experimental.utils import stage as stage_utils
from isaacsim.core.experimental.utils.stage import add_reference_to_stage
from isaacsim.core.rendering_manager import RenderingManager, ViewportManager
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.gui.components.element_wrappers import Button, CollapsableFrame, StateButton
from isaacsim.gui.components.style import get_style
from isaacsim.robot_motion.cumotion import load_cumotion_supported_robot
from isaacsim.storage.native import get_assets_root_path
from pxr import UsdPhysics

from .scenario import FrankaGraphPlannerExample


class UIBuilder:
    def __init__(self):
        self.frames = []
        self.wrapped_ui_elements = []
        self._timeline = omni.timeline.get_timeline_interface()
        self._joint_slider_models = []
        self._on_init()

    def on_menu_callback(self):
        pass

    def on_timeline_event(self, event):
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = False

    def on_physics_step(self, step: float):
        pass

    def on_stage_event(self, event):
        self._reset_extension()

    def cleanup(self):
        for ui_elem in self.wrapped_ui_elements:
            ui_elem.cleanup()

    def build_ui(self):
        world_controls_frame = CollapsableFrame("World Controls", collapsed=False)

        with world_controls_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._load_btn = Button(
                    "Load Button",
                    "LOAD",
                    tooltip="Load the scene and initialize the scenario",
                    on_click_fn=self._on_load_btn,
                )
                self.wrapped_ui_elements.append(self._load_btn)

                # Joint angle sliders for C-space target
                self._joint_sliders_frame = CollapsableFrame("Configuration Target", collapsed=False)
                with self._joint_sliders_frame:
                    with ui.VStack(style=get_style(), spacing=5, height=0):
                        # Sliders will be created in _setup_scenario after robot is loaded
                        self._joint_sliders_container = ui.VStack(style=get_style(), spacing=3, height=0)

                self._to_cspace_btn = ui.Button(
                    "TO CSPACE TARGET",
                    clicked_fn=self._on_to_cspace_target_btn,
                    style=get_style(),
                )
                self._to_cspace_btn.enabled = False

                self._to_task_space_btn = ui.Button(
                    "TO TASK-SPACE TARGET",
                    clicked_fn=self._on_to_task_space_target_btn,
                    style=get_style(),
                )
                self._to_task_space_btn.enabled = False

        run_scenario_frame = CollapsableFrame("Run Scenario")

        with run_scenario_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._scenario_state_btn = StateButton(
                    "Run Scenario",
                    "RUN",
                    "STOP",
                    on_a_click_fn=self._on_run_scenario_a_text,
                    on_b_click_fn=self._on_run_scenario_b_text,
                    physics_callback_fn=self._update_scenario,
                )
                self._scenario_state_btn.enabled = False
                self.wrapped_ui_elements.append(self._scenario_state_btn)

    def _on_init(self):
        self._scenario = FrankaGraphPlannerExample()

    def _load_example_assets(self):
        """Load robot, target, and obstacle assets to the stage."""
        self._scenario._robot_prim_path = "/panda"
        path_to_robot_usd = get_assets_root_path() + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"

        add_reference_to_stage(path_to_robot_usd, self._scenario._robot_prim_path)
        self._scenario._articulation = Articulation(self._scenario._robot_prim_path)

        angle = np.pi / 2
        target_orientation = np.array([np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0])

        # Create target cube (non-collision, can be moved around)
        self._scenario._target = Cube(
            paths="/World/target", sizes=0.04, positions=[0.5, 0.0, 0.7], orientations=target_orientation
        )

        # Create fixed cube obstacle
        obstacle_path = "/World/obstacle"
        obstacle_size = 0.1
        obstacle_position = np.array([0.25, 0.0, 0.5])

        # Create cube geometry
        cube = Cube(obstacle_path, sizes=obstacle_size, positions=[obstacle_position])

        # Plan to a C-space target
        # Use default configuration as starting point (ensures valid configuration)
        self._scenario._cumotion_robot = load_cumotion_supported_robot("franka")
        self._scenario._controlled_dof_indices = (
            self._scenario._articulation.get_dof_indices(self._scenario._cumotion_robot.controlled_joint_names)
            .numpy()
            .flatten()
        )
        self._scenario._q_initial = self._scenario._cumotion_robot.robot_description.default_cspace_configuration()
        self._scenario._q_initial[0] = -np.pi / 2
        self._scenario._q_initial[1] = -np.pi / 8  # Modify joint 1 for a different starting pose
        self._scenario._first_trajectory = True

        # Apply collision APIs
        GeomPrim(obstacle_path, apply_collision_apis=True)

    def _on_load_btn(self):
        """Handle Load button click - loads scene and initializes scenario."""
        asyncio.ensure_future(self._load_scene_async())

    async def _load_scene_async(self):
        """Async function to load the scene without using World."""
        # Create new stage
        await stage_utils.create_new_stage_async(template="default stage")

        # Set up stage properties
        stage_utils.set_stage_up_axis("Z")
        stage_utils.set_stage_units(meters_per_unit=1.0)

        # Setup scene (load assets)
        self._setup_scene()

        # Set camera view
        ViewportManager.set_camera_view(camera="/OmniverseKit_Persp", eye=[2, 1.5, 2], target=[0, 0, 0])

        # Create physics scene if it doesn't exist
        stage = stage_utils.get_current_stage()
        physics_scene_path = "/World/PhysicsScene"
        if not stage.GetPrimAtPath(physics_scene_path).IsValid():
            UsdPhysics.Scene.Define(stage, physics_scene_path)
        await omni.kit.app.get_app().next_update_async()

        # Set physics and rendering timesteps
        SimulationManager.set_physics_dt(dt=1.0 / 60.0)
        RenderingManager.set_dt(dt=1.0 / 60.0)
        await omni.kit.app.get_app().next_update_async()

        # Initialize physics if needed
        if SimulationManager.get_physics_sim_view() is None:
            SimulationManager.initialize_physics()

        # Play timeline to initialize physics tensors
        self._timeline.play()
        # Wait for multiple updates to ensure physics runs and tensors are initialized
        for _ in range(5):
            await omni.kit.app.get_app().next_update_async()
        self._timeline.stop()
        await omni.kit.app.get_app().next_update_async()

        # Setup scenario (post-load callback)
        self._setup_scenario()

    def _setup_scene(self):
        """Load assets onto the stage."""
        # Load assets - prims are automatically added to the stage
        self._load_example_assets()

    def _setup_scenario(self):
        # Don't call setup() here - let user choose target type via buttons
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = False  # Enable only after planning
        self._to_cspace_btn.enabled = True
        self._to_task_space_btn.enabled = True

        # Create joint sliders after robot is loaded
        self._create_joint_sliders()

    def _get_joint_limits(self):
        """Get joint limits for all controlled joints.

        Returns:
            tuple: (lower_limits, upper_limits) as numpy arrays
        """
        if self._scenario._cumotion_robot is None:
            return None, None

        kinematics = self._scenario._cumotion_robot.kinematics
        num_joints = kinematics.num_cspace_coords()
        lower_limits = []
        upper_limits = []

        for i in range(num_joints):
            limits = kinematics.cspace_coord_limits(i)
            lower_limits.append(limits.lower)
            upper_limits.append(limits.upper)

        return np.array(lower_limits), np.array(upper_limits)

    def _get_joint_names(self):
        """Get the names of all controlled joints.

        Returns:
            list: List of joint names
        """
        if self._scenario._cumotion_robot is None:
            return []
        return self._scenario._cumotion_robot.controlled_joint_names

    def _create_joint_sliders(self):
        """Create sliders for each joint with limits from the robot description."""
        # Clear existing sliders
        self._joint_slider_models.clear()
        self._joint_sliders_container.clear()

        # Get joint limits and names
        lower_limits, upper_limits = self._get_joint_limits()
        joint_names = self._get_joint_names()

        if lower_limits is None or upper_limits is None or len(joint_names) == 0:
            return

        # Get default configuration for initial slider values
        cumotion_robot = self._scenario._cumotion_robot
        default_q = cumotion_robot.robot_description.default_cspace_configuration()
        # Set default target values (matching the original default target)
        default_target = default_q.copy()
        default_target[0] = np.pi / 2
        default_target[1] = -np.pi / 3

        # Create a slider for each joint inside the container
        with self._joint_sliders_container:
            for i, joint_name in enumerate(joint_names):
                with ui.HStack(style=get_style(), spacing=5):
                    ui.Label(
                        joint_name,
                        width=120,
                        alignment=ui.Alignment.LEFT_CENTER,
                    )
                    # Create FloatField model for the value
                    field_model = ui.FloatField(
                        name="Field",
                        width=80,
                        alignment=ui.Alignment.LEFT_CENTER,
                    ).model
                    field_model.set_value(float(default_target[i]))

                    # Create FloatSlider with min/max limits
                    slider = ui.FloatSlider(
                        width=ui.Fraction(1),
                        alignment=ui.Alignment.LEFT_CENTER,
                        min=float(lower_limits[i]),
                        max=float(upper_limits[i]),
                        step=0.01,
                        model=field_model,
                    )

                    self._joint_slider_models.append(field_model)

    def _show_error_dialog(self, message: str):
        """Show a modal error dialog with the given message.

        Args:
            message: Error message to display.
        """
        dialog = ui.Window(
            "Planning Failed",
            width=500,
            height=0,
            visible=True,
            flags=(ui.WINDOW_FLAGS_NO_SCROLLBAR | ui.WINDOW_FLAGS_MODAL | ui.WINDOW_FLAGS_NO_SAVED_SETTINGS),
        )

        def _close_dialog():
            """Hide immediately, destroy on the next frame."""
            dialog.visible = False

            async def _destroy_next_frame():
                await omni.kit.app.get_app().next_update_async()
                dialog.destroy()

            asyncio.ensure_future(_destroy_next_frame())

        with dialog.frame:
            with ui.VStack(spacing=10):
                ui.Spacer(height=10)
                ui.Label(message, word_wrap=True, alignment=ui.Alignment.LEFT_TOP)
                ui.Spacer(height=10)
                ui.Button("OK", clicked_fn=_close_dialog, alignment=ui.Alignment.CENTER)
                ui.Spacer(height=10)

    def _on_to_cspace_target_btn(self):
        """Handle click on 'to cspace_target' button."""
        # Pause timeline before planning
        self._timeline.pause()

        # Get target configuration from sliders
        q_target = [model.get_value_as_float() for model in self._joint_slider_models]

        error_message = self._scenario.plan_to_cspace_target(q_target=q_target)
        if error_message is not None:
            self._show_error_dialog(error_message)
            self._scenario_state_btn.reset()
            self._scenario_state_btn.enabled = False
        else:
            self._scenario_state_btn.reset()
            self._scenario_state_btn.enabled = True

    def _on_to_task_space_target_btn(self):
        """Handle click on 'to task-space target' button."""
        # Pause timeline before planning
        self._timeline.pause()
        error_message = self._scenario.plan_to_task_space_target()
        if error_message is not None:
            self._show_error_dialog(error_message)
            self._scenario_state_btn.reset()
            self._scenario_state_btn.enabled = False
        else:
            self._scenario_state_btn.reset()
            self._scenario_state_btn.enabled = True

    def _update_scenario(self, step: float, *args, **kwargs):
        # Check if physics tensors are valid before updating
        if self._scenario._articulation is not None:
            if not self._scenario._articulation.is_physics_tensor_entity_valid():
                return
        self._scenario.update(step)

    def _on_run_scenario_a_text(self):
        self._timeline.play()

    def _on_run_scenario_b_text(self):
        self._timeline.pause()

    def _reset_extension(self):
        self._on_init()
        self._reset_ui()

    def _reset_ui(self):
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = False
        self._to_cspace_btn.enabled = False
        self._to_task_space_btn.enabled = False
        self._joint_slider_models.clear()
        if hasattr(self, "_joint_sliders_container"):
            self._joint_sliders_container.clear()
