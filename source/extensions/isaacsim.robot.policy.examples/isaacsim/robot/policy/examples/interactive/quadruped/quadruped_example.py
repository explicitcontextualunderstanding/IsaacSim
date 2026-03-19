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

"""Interactive quadruped robot control example using reinforcement learning policies and keyboard input."""


import carb
import isaacsim.core.experimental.utils.stage as stage_utils
import omni
import omni.appwindow  # Contains handle to keyboard
from isaacsim.core.deprecation_manager import import_module
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents
from isaacsim.examples.base.base_sample_experimental import BaseSample
from isaacsim.robot.policy.examples.robots import SpotFlatTerrainPolicy
from isaacsim.storage.native import get_assets_root_path

torch = import_module("torch")


class QuadrupedExample(BaseSample):
    """Interactive example demonstrating quadruped robot control using reinforcement learning policies.

    This example creates a Spot quadruped robot in a physics simulation environment and demonstrates
    how to control it using keyboard input. The robot uses a reinforcement learning policy trained for
    flat terrain locomotion and runs on GPU for high-performance physics simulation.

    The simulation is configured with 500 Hz physics updates and 50 Hz rendering for smooth real-time
    interaction. Users can control the robot's movement using keyboard inputs:

    - Arrow keys or Numpad 8/2/4/6: Forward/backward and left/right movement
    - N/M or Numpad 7/9: Yaw rotation commands

    The example sets up a complete simulation environment including ground plane, robot initialization,
    physics callbacks, and keyboard event handling. It demonstrates integration between Isaac Sim's
    physics simulation, reinforcement learning policies, and interactive control systems.

    Key features:
    - GPU-accelerated physics simulation using PyTorch backend
    - Real-time keyboard control interface
    - Automatic robot policy execution and state management
    - Proper simulation lifecycle management with setup and cleanup phases
    """

    def __init__(self):
        super().__init__()
        # Configure simulation settings for GPU dynamics with high-frequency physics
        self._world_settings["stage_units_in_meters"] = 1.0
        self._world_settings["physics_dt"] = 1.0 / 500.0  # 500 Hz physics
        self._world_settings["rendering_dt"] = 10.0 / 500.0  # 50 Hz rendering (10 physics steps per render)
        self._world_settings["device"] = "cuda"  # GPU dynamics
        self._world_settings["backend"] = "torch"  # PyTorch backend

        self._base_command = torch.tensor([0.0, 0.0, 0.0], device="cuda")
        self._physics_ready = False
        self.spot = None
        self._physics_callback_id = None
        self._event_timer_callback = None

        # Bindings for keyboard to command
        self._input_keyboard_mapping = {
            # forward command
            "NUMPAD_8": [2.0, 0.0, 0.0],
            "UP": [2.0, 0.0, 0.0],
            # back command
            "NUMPAD_2": [-2.0, 0.0, 0.0],
            "DOWN": [-2.0, 0.0, 0.0],
            # left command
            "NUMPAD_6": [0.0, -2.0, 0.0],
            "RIGHT": [0.0, -2.0, 0.0],
            # right command
            "NUMPAD_4": [0.0, 2.0, 0.0],
            "LEFT": [0.0, 2.0, 0.0],
            # yaw command (positive)
            "NUMPAD_7": [0.0, 0.0, 2.0],
            "N": [0.0, 0.0, 2.0],
            # yaw command (negative)
            "NUMPAD_9": [0.0, 0.0, -2.0],
            "M": [0.0, 0.0, -2.0],
        }

    def setup_scene(self):
        """Set up the scene with robot and environment."""
        # Set device and backend BEFORE creating robot so it uses GPU
        SimulationManager.set_backend(self._world_settings["backend"])
        SimulationManager.set_physics_sim_device(self._world_settings["device"])

        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")

        # Add ground plane environment for physics simulation
        ground_plane = stage_utils.add_reference_to_stage(
            usd_path=get_assets_root_path() + "/Isaac/Environments/Grid/default_environment.usd",
            path="/World/ground",
        )
        # Create Spot robot (will now use GPU device)
        self.spot = SpotFlatTerrainPolicy(
            prim_path="/World/Spot",
            position=[0, 0, 0.8],
        )
        print("Scene setup complete with Spot quadruped robot")

    async def setup_post_load(self):
        """Setup keyboard input and physics callback after initial load."""
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._sub_keyboard_event)

        self._physics_ready = False

        # Register physics callback using SimulationManager
        if self._physics_callback_id is None:
            self._physics_callback_id = SimulationManager.register_callback(
                self.on_physics_step, IsaacEvents.POST_PHYSICS_STEP
            )
        print("Spot quadruped scene loaded successfully")

    async def setup_pre_reset(self):
        """Called before world reset."""
        self._physics_ready = False

    async def setup_post_reset(self):
        """Called after world reset."""
        self._physics_ready = False

    async def setup_post_clear(self):
        """Called after clearing the scene."""
        # Deregister physics callback
        if self._physics_callback_id is not None:
            try:
                SimulationManager.deregister_callback(self._physics_callback_id)
            except Exception as e:
                print(f"Note: Could not deregister callback {self._physics_callback_id}: {e}")
            self._physics_callback_id = None

        self._event_timer_callback = None
        self._sub_keyboard = None
        self.spot = None
        self._physics_ready = False

    def on_physics_step(self, dt, context):
        """Physics step callback - initialize on first step, then run policy.

        Args:
            dt: Time delta for the physics step.
            context: Physics step context.
        """
        if not self.spot:
            return

        # Check if physics tensors are valid, if not, reinitialize
        if not self.spot.robot.is_physics_tensor_entity_valid():
            self._physics_ready = False

        if self._physics_ready:
            # Robot is initialized, run the policy
            self.spot.forward(dt, self._base_command)
        else:
            self._physics_ready = True
            self.spot.initialize()
            self.spot.post_reset()

    def _sub_keyboard_event(self, event, *args, **kwargs) -> bool:
        """Handle keyboard input for robot control.

        Args:
            event: Keyboard event containing input information.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            True to indicate the event was handled.
        """
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # On pressing, the command is incremented
            if event.input.name in self._input_keyboard_mapping:
                self._base_command += torch.tensor(
                    self._input_keyboard_mapping[event.input.name], device=self._base_command.device
                )
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            # On release, the command is decremented
            if event.input.name in self._input_keyboard_mapping:
                self._base_command -= torch.tensor(
                    self._input_keyboard_mapping[event.input.name], device=self._base_command.device
                )
        return True

    def physics_cleanup(self):
        """Clean up physics resources."""
        # Deregister physics callback
        if self._physics_callback_id is not None:
            try:
                SimulationManager.deregister_callback(self._physics_callback_id)
            except Exception as e:
                print(f"Note: Could not deregister callback {self._physics_callback_id}: {e}")
            self._physics_callback_id = None

        self._event_timer_callback = None
        self._sub_keyboard = None
        self.spot = None
        self._physics_ready = False
