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

"""Base class for creating Isaac Sim simulation samples with structured lifecycle management."""


import gc
from abc import abstractmethod

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import omni.kit.app
import omni.physics.core
from isaacsim.core.rendering_manager import RenderingManager, ViewportManager
from isaacsim.core.simulation_manager import SimulationManager


class BaseSample(object):
    def __init__(self) -> None:
        self._physics_sim_interface = omni.physics.core.get_physics_simulation_interface()
        self._world_settings = {"physics_dt": 1.0 / 60.0, "stage_units_in_meters": 1.0, "rendering_dt": 1.0 / 60.0}
        self._logging_info = ""

    def set_world_settings(self, physics_dt=None, stage_units_in_meters=None, rendering_dt=None):
        """Updates the world settings with the provided values.

        Args:
            physics_dt: Physics simulation timestep in seconds.
            stage_units_in_meters: Number of meters per stage unit.
            rendering_dt: Rendering timestep in seconds.
        """
        if physics_dt is not None:
            self._world_settings["physics_dt"] = physics_dt
        if stage_units_in_meters is not None:
            self._world_settings["stage_units_in_meters"] = stage_units_in_meters
        if rendering_dt is not None:
            self._world_settings["rendering_dt"] = rendering_dt

    async def load_world_async(self):
        """Function called when clicking load button."""
        await stage_utils.create_new_stage_async()

        # Set up stage properties
        stage_utils.set_stage_up_axis("Z")
        stage_utils.set_stage_units(meters_per_unit=self._world_settings["stage_units_in_meters"])
        self.setup_scene()
        ViewportManager.set_camera_view(eye=[1.5, 1.5, 1.5], target=[0.01, 0.01, 0.01], camera="/OmniverseKit_Persp")

        await omni.kit.app.get_app().next_update_async()

        SimulationManager.setup_simulation(dt=self._world_settings["physics_dt"])
        RenderingManager.set_dt(dt=self._world_settings["rendering_dt"])
        await omni.kit.app.get_app().next_update_async()

        app_utils.play()
        await omni.kit.app.get_app().next_update_async()

        await self.setup_post_load()

    async def reset_async(self):
        """Function called when clicking reset button."""
        await self.setup_pre_reset()

        # Stop and restart timeline to reset simulation
        app_utils.stop()
        await omni.kit.app.get_app().next_update_async()

        app_utils.play()
        await omni.kit.app.get_app().next_update_async()

        await self.setup_post_reset()

    @abstractmethod
    def setup_scene(self):
        """Used to setup anything in the world, adding tasks happen here for instance."""
        pass

    @abstractmethod
    async def setup_post_load(self):
        """Called after first reset of the world when pressing load,
        intializing private variables happen here.
        """
        pass

    @abstractmethod
    async def setup_pre_reset(self):
        """Called in reset button before resetting the world
        to remove a physics callback for instance or a controller reset.
        """
        pass

    @abstractmethod
    async def setup_post_reset(self):
        """Called in reset button after resetting the world which includes one step with rendering."""
        pass

    @abstractmethod
    async def setup_post_clear(self):
        """Called after clicking clear button
        or after creating a new stage and clearing the instance of the world with its callbacks.
        """
        pass

    def log_info(self, info):
        """Appends information to the logging string.

        Args:
            info: Information to log.
        """
        self._logging_info += str(info) + "\n"

    def _physics_cleanup(self):
        """Cleanup physics resources"""
        if app_utils.is_playing():
            app_utils.stop()
        self.physics_cleanup()

    def physics_cleanup(self):
        """Function called when extension shutdowns and starts again, (hot reloading feature)"""
        pass

    async def clear_async(self):
        """Function called when clicking clear button"""
        await stage_utils.create_new_stage_async()
        self._physics_cleanup()
        gc.collect()
        await self.setup_post_clear()
