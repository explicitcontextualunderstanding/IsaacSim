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

import omni.kit.app
import omni.timeline
import omni.ui as ui
from isaacsim.core.experimental.utils import stage as stage_utils
from isaacsim.core.rendering_manager import RenderingManager, ViewportManager
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.gui.components.element_wrappers import Button, CollapsableFrame, StateButton
from isaacsim.gui.components.style import get_style
from pxr import UsdPhysics

from .scenario import UR10TrajectoryGeneratorExample


class UIBuilder:
    def __init__(self):
        self.frames = []
        self.wrapped_ui_elements = []
        self._timeline = omni.timeline.get_timeline_interface()
        self._on_init()

    def on_menu_callback(self):
        pass

    def on_timeline_event(self, event):
        self._cspace_trajectory_btn.reset()
        self._taskspace_trajectory_btn.reset()
        self._hybrid_trajectory_btn.reset()
        self._cspace_trajectory_btn.enabled = False
        self._taskspace_trajectory_btn.enabled = False
        self._hybrid_trajectory_btn.enabled = False

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

                self._reset_btn = Button(
                    "Reset Button", "RESET", tooltip="Reset the scenario", on_click_fn=self._on_reset_btn
                )
                self._reset_btn.enabled = False
                self.wrapped_ui_elements.append(self._reset_btn)

        run_scenario_frame = CollapsableFrame("Run Scenario", collapsed=False)

        with run_scenario_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._cspace_trajectory_btn = StateButton(
                    "Run CSpace Trajectory",
                    "CSPACE TRAJECTORY",
                    "STOP",
                    on_a_click_fn=self._on_run_cspace,
                    on_b_click_fn=self._on_stop,
                    physics_callback_fn=self._update_scenario,
                )
                self._cspace_trajectory_btn.enabled = False

                self._taskspace_trajectory_btn = StateButton(
                    "Run TaskSpace Trajectory",
                    "TASKSPACE TRAJECTORY",
                    "STOP",
                    on_a_click_fn=self._on_run_taskspace,
                    on_b_click_fn=self._on_stop,
                    physics_callback_fn=self._update_scenario,
                )
                self._taskspace_trajectory_btn.enabled = False

                self._hybrid_trajectory_btn = StateButton(
                    "Run Hybrid Trajectory",
                    "HYBRID TRAJECTORY",
                    "STOP",
                    on_a_click_fn=self._on_run_hybrid,
                    on_b_click_fn=self._on_stop,
                    physics_callback_fn=self._update_scenario,
                )
                self._hybrid_trajectory_btn.enabled = False

                self.wrapped_ui_elements.append(self._cspace_trajectory_btn)
                self.wrapped_ui_elements.append(self._taskspace_trajectory_btn)
                self.wrapped_ui_elements.append(self._hybrid_trajectory_btn)

    def _on_init(self):
        self._scenario = UR10TrajectoryGeneratorExample()

    def _on_load_btn(self):
        """Handle Load button click - loads scene and initializes scenario."""
        asyncio.ensure_future(self._load_scene_async())

    async def _load_scene_async(self):
        """Async function to load the scene without using World."""
        # Create new stage
        await stage_utils.create_new_stage_async(template="sunlight")

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
        self._scenario.load_example_assets()

    def _setup_scenario(self):
        """Initialize the scenario after assets are loaded."""
        self._scenario.setup()
        self._cspace_trajectory_btn.enabled = True
        self._taskspace_trajectory_btn.enabled = True
        self._hybrid_trajectory_btn.enabled = True
        self._reset_btn.enabled = True

    def _on_reset_btn(self):
        """Handle Reset button click - resets the scenario."""
        asyncio.ensure_future(self._reset_scene_async())

    async def _reset_scene_async(self):
        """Async function to reset the scene without using World."""
        # Stop timeline
        self._timeline.stop()
        await omni.kit.app.get_app().next_update_async()

        # Reset scenario
        self._scenario.reset()
        await omni.kit.app.get_app().next_update_async()

        # UI management
        self._cspace_trajectory_btn.reset()
        self._taskspace_trajectory_btn.reset()
        self._hybrid_trajectory_btn.reset()
        self._cspace_trajectory_btn.enabled = True
        self._taskspace_trajectory_btn.enabled = True
        self._hybrid_trajectory_btn.enabled = True

    def _update_scenario(self, step: float, *args, **kwargs):
        # Check if physics tensors are valid before updating
        if self._scenario._articulation is not None:
            if not self._scenario._articulation.is_physics_tensor_entity_valid():
                return
        self._scenario.update(step)

    def _on_run_cspace(self):
        self._timeline.play()
        self._scenario.reset()
        self._taskspace_trajectory_btn.reset()
        self._hybrid_trajectory_btn.reset()
        self._scenario.setup_cspace_trajectory()

    def _on_run_taskspace(self):
        self._timeline.play()
        self._scenario.reset()
        self._cspace_trajectory_btn.reset()
        self._hybrid_trajectory_btn.reset()
        self._scenario.setup_taskspace_trajectory()

    def _on_run_hybrid(self):
        self._timeline.play()
        self._scenario.reset()
        self._cspace_trajectory_btn.reset()
        self._taskspace_trajectory_btn.reset()
        self._scenario.setup_hybrid_trajectory()

    def _on_stop(self):
        self._timeline.pause()

    def _reset_extension(self):
        self._on_init()
        self._reset_ui()

    def _reset_ui(self):
        self._cspace_trajectory_btn.reset()
        self._taskspace_trajectory_btn.reset()
        self._hybrid_trajectory_btn.reset()
        self._cspace_trajectory_btn.enabled = False
        self._taskspace_trajectory_btn.enabled = False
        self._hybrid_trajectory_btn.enabled = False
        self._reset_btn.enabled = False
