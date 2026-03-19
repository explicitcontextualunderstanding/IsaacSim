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

import os
import unittest

import carb.settings
import omni.kit
import omni.usd
from isaacsim.test.utils.file_validation import get_folder_file_summary, validate_folder_contents


class TestAmrNavigation(omni.kit.test.AsyncTestCase):

    async def setUp(self):
        await omni.kit.app.get_app().next_update_async()
        await omni.usd.get_context().new_stage_async()
        await omni.kit.app.get_app().next_update_async()
        self.original_dlss_exec_mode = carb.settings.get_settings().get("rtx/post/dlss/execMode")

    async def tearDown(self):
        omni.usd.get_context().close_stage()
        await omni.kit.app.get_app().next_update_async()
        # In some cases the test will end before the asset is loaded, in this case wait for assets to load
        while omni.usd.get_context().get_stage_loading_status()[2] > 0:
            await omni.kit.app.get_app().next_update_async()
        carb.settings.get_settings().set("rtx/post/dlss/execMode", self.original_dlss_exec_mode)

    async def test_amr_navigation(self):
        import asyncio
        import builtins
        import os
        import random
        from itertools import cycle

        import carb.settings
        import omni.client
        import omni.kit.app
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        import omni.usd.commands
        from isaacsim.core.utils.stage import create_new_stage
        from isaacsim.storage.native import get_assets_root_path
        from pxr import Gf, UsdGeom

        ENV_URLS = [
            "/Isaac/Environments/Grid/default_environment.usd",
            "/Isaac/Environments/Simple_Warehouse/warehouse.usd",
            "/Isaac/Environments/Grid/gridroom_black.usd",
        ]
        NUM_FRAMES = 9
        ENV_INTERVAL = 3
        USE_TEMP_RP = True

        class NavSDGDemo:
            """Demonstration of synthetic data generation using an AMR navigating towards a target."""

            CARTER_URL = "/Isaac/Samples/Replicator/OmniGraph/nova_carter_nav_only.usd"
            DOLLY_URL = "/Isaac/Props/Dolly/dolly.usd"
            PROPS_URL = "/Isaac/Props/YCB/Axis_Aligned_Physics"
            LEFT_CAMERA_REL_PATH = "sensors/front_hawk/left/camera_left"
            RIGHT_CAMERA_REL_PATH = "sensors/front_hawk/right/camera_right"

            def __init__(self) -> None:
                """Initialize the navigation SDG demo with default values."""
                self._carter_chassis = None
                self._carter_nav_target = None
                self._dolly = None
                self._dolly_light = None
                self._props = []
                self._cycled_env_urls = None
                self._env_interval = 1
                self._timeline = None
                self._timeline_sub = None
                self._stage_event_sub = None
                self._stage = None
                self._trigger_distance = 2.0
                self._num_frames = 0
                self._frame_counter = 0
                self._writer = None
                self._out_dir = None
                self._render_products = []
                self._use_temp_rp = False
                self._in_running_state = False
                self._completion_event = None

            async def run_async(
                self,
                num_frames: int = 10,
                out_dir: str | None = None,
                env_urls: list[str] = [],
                env_interval: int = 3,
                use_temp_rp: bool = False,
                seed: int | None = None,
            ) -> None:
                """Run the SDG demo asynchronously and wait for completion."""
                self._completion_event = asyncio.Event()
                self.start(
                    num_frames=num_frames,
                    out_dir=out_dir,
                    env_urls=env_urls,
                    env_interval=env_interval,
                    use_temp_rp=use_temp_rp,
                    seed=seed,
                )
                await self._completion_event.wait()

            def start(
                self,
                num_frames: int = 10,
                out_dir: str | None = None,
                env_urls: list[str] = [],
                env_interval: int = 3,
                use_temp_rp: bool = False,
                seed: int | None = None,
            ) -> None:
                """Start the SDG demo with the given configuration."""
                print(f"[SDG] Starting")
                if seed is not None:
                    rep.set_global_seed(seed)
                    random.seed(seed)
                self._num_frames = num_frames
                self._out_dir = out_dir if out_dir is not None else os.path.join(os.getcwd(), "_out_nav_sdg_demo")
                self._cycled_env_urls = cycle(env_urls)
                self._env_interval = env_interval
                self._use_temp_rp = use_temp_rp
                self._frame_counter = 0
                self._trigger_distance = 2.0
                self._load_env()
                self._randomize_dolly_pose()
                self._randomize_dolly_light()
                self._randomize_prop_poses()
                self._setup_sdg()
                self._timeline = omni.timeline.get_timeline_interface()
                self._timeline.play()
                self._timeline_sub = self._timeline.get_timeline_event_stream().create_subscription_to_pop_by_type(
                    int(omni.timeline.TimelineEventType.CURRENT_TIME_TICKED), self._on_timeline_event
                )
                self._stage_event_sub = (
                    omni.usd.get_context()
                    .get_stage_event_stream()
                    .create_subscription_to_pop_by_type(
                        int(omni.usd.StageEventType.CLOSING), self._on_stage_closing_event
                    )
                )
                self._in_running_state = True

            def clear(self) -> None:
                """Reset all state variables and unsubscribe from events."""
                self._cycled_env_urls = None
                self._carter_chassis = None
                self._carter_nav_target = None
                self._dolly = None
                self._dolly_light = None
                self._timeline = None
                self._frame_counter = 0
                if self._stage_event_sub:
                    self._stage_event_sub.unsubscribe()
                self._stage_event_sub = None
                if self._timeline_sub:
                    self._timeline_sub.unsubscribe()
                self._timeline_sub = None
                self._clear_sdg_render_products()
                self._stage = None
                self._in_running_state = False
                # Signal completion for async waiters
                if self._completion_event:
                    self._completion_event.set()
                    self._completion_event = None

            def is_running(self) -> bool:
                """Return whether the SDG demo is currently running."""
                return self._in_running_state

            def _is_running_in_script_editor(self) -> bool:
                """Return whether the script is running in the Isaac Sim script editor."""
                return builtins.ISAAC_LAUNCHED_FROM_TERMINAL is True

            def _on_stage_closing_event(self, e: carb.events.IEvent) -> None:
                """Handle stage closing event by clearing state."""
                self.clear()

            def _load_env(self) -> None:
                """Create a new stage and load environment, robot, dolly, light, and props."""
                create_new_stage()
                self._stage = omni.usd.get_context().get_stage()
                rep.functional.physics.create_physics_scene(
                    "/PhysicsScene", enableCCD=True, broadphaseType="MBP", enableGPUDynamics=False
                )

                # Environment
                assets_root_path = get_assets_root_path()
                rep.functional.create.reference(
                    usd_path=assets_root_path + next(self._cycled_env_urls), name="Environment"
                )

                # Nova Carter
                rep.functional.create.scope(name="NavWorld")
                carter = rep.functional.create.reference(
                    position=(0, 0, 0),
                    rotation=(0, 0, 0),
                    usd_path=assets_root_path + self.CARTER_URL,
                    parent="/NavWorld",
                    name="CarterNav",
                )

                # Iterate children until targetXform (for navigation target) and chassis_link (for current location) are found
                for child in carter.GetChildren():
                    if child.GetName() == "targetXform":
                        self._carter_nav_target = child
                        break
                for child in carter.GetChildren():
                    if child.GetName() == "chassis_link":
                        self._carter_chassis = child
                        break

                # Dolly
                self._dolly = rep.functional.create.reference(
                    position=(0, 0, 0),
                    rotation=(0, 0, 0),
                    usd_path=assets_root_path + self.DOLLY_URL,
                    parent="/NavWorld",
                    name="Dolly",
                )

                # Add colliders to the dolly and its geometry primitives
                for desc_prim in self._dolly.GetChildren():
                    if desc_prim.IsA(UsdGeom.Gprim):
                        rep.functional.physics.apply_rigid_body(desc_prim)

                # Light
                self._dolly_light = rep.functional.create.sphere_light(
                    position=(0, 0, 0),
                    intensity=250000,
                    radius=0.3,
                    color=(1.0, 1.0, 1.0),
                    parent="/NavWorld",
                    name="DollyLight",
                )

                # Props
                props_urls = []
                props_folder_path = assets_root_path + self.PROPS_URL
                result, entries = omni.client.list(props_folder_path)
                if result != omni.client.Result.OK:
                    carb.log_error(f"Could not list assets in path: {props_folder_path}")
                    return
                for entry in entries:
                    _, ext = os.path.splitext(entry.relative_path)
                    if ext == ".usd":
                        props_urls.append(f"{props_folder_path}/{entry.relative_path}")

                cycled_props_url = cycle(props_urls)
                for i in range(15):
                    prop_url = next(cycled_props_url)
                    prop_name = os.path.splitext(os.path.basename(prop_url))[0]
                    path = f"/NavWorld/Props/Prop_{prop_name}_{i}"
                    prim = self._stage.DefinePrim(path, "Xform")
                    prim.GetReferences().AddReference(prop_url)
                    self._props.append(prim)

            def _randomize_dolly_pose(self) -> None:
                """Set random dolly position ensuring minimum distance from Carter."""
                min_dist_from_carter = 4
                carter_loc = self._carter_chassis.GetAttribute("xformOp:translate").Get()
                for _ in range(100):
                    x, y = random.uniform(-6, 6), random.uniform(-6, 6)
                    dist = (Gf.Vec2f(x, y) - Gf.Vec2f(carter_loc[0], carter_loc[1])).GetLength()
                    if dist > min_dist_from_carter:
                        self._dolly.GetAttribute("xformOp:translate").Set((x, y, 0))
                        self._carter_nav_target.GetAttribute("xformOp:translate").Set((x, y, 0))
                        break
                self._dolly.GetAttribute("xformOp:rotateXYZ").Set((0, 0, random.uniform(-180, 180)))

            def _randomize_dolly_light(self) -> None:
                """Position light above dolly with random color."""
                dolly_loc = self._dolly.GetAttribute("xformOp:translate").Get()
                self._dolly_light.GetAttribute("xformOp:translate").Set(dolly_loc + (0, 0, 3))
                self._dolly_light.GetAttribute("inputs:color").Set(
                    (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
                )

            def _randomize_prop_poses(self) -> None:
                """Stack props above the dolly with random horizontal offsets."""
                spawn_loc = self._dolly.GetAttribute("xformOp:translate").Get()
                spawn_loc[2] = spawn_loc[2] + 0.5
                for prop in self._props:
                    prop.GetAttribute("xformOp:translate").Set(
                        spawn_loc + (random.uniform(-1, 1), random.uniform(-1, 1), 0)
                    )
                    spawn_loc[2] = spawn_loc[2] + 0.2

            def _setup_sdg(self) -> None:
                """Configure SDG settings, camera parameters, writer, and render products."""
                rep.orchestrator.set_capture_on_play(False)

                # Set DLSS to Quality mode (2) for best SDG results , options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
                carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

                # Set camera sensors fStop to 0.0 to get well lit sharp images
                left_camera_path = self._carter_chassis.GetPath().AppendPath(self.LEFT_CAMERA_REL_PATH)
                left_camera_prim = self._stage.GetPrimAtPath(left_camera_path)
                left_camera_prim.GetAttribute("fStop").Set(0.0)
                right_camera_path = self._carter_chassis.GetPath().AppendPath(self.RIGHT_CAMERA_REL_PATH)
                right_camera_prim = self._stage.GetPrimAtPath(right_camera_path)
                right_camera_prim.GetAttribute("fStop").Set(0.0)

                backend = rep.backends.get("DiskBackend")
                backend.initialize(output_dir=self._out_dir)
                print(f"[SDG] Writing data to: {self._out_dir}")
                self._writer = rep.writers.get("BasicWriter")
                self._writer.initialize(backend=backend, rgb=True)
                self._setup_sdg_render_products()

            def _setup_sdg_render_products(self) -> None:
                """Create and attach render products for left and right cameras."""
                print(f"[SDG] Creating SDG render products")
                left_camera_path = self._carter_chassis.GetPath().AppendPath(self.LEFT_CAMERA_REL_PATH)
                rp_left = rep.create.render_product(
                    str(left_camera_path),
                    (1024, 1024),
                    name="left_sensor",
                    force_new=True,
                )
                right_camera_path = self._carter_chassis.GetPath().AppendPath(self.RIGHT_CAMERA_REL_PATH)
                rp_right = rep.create.render_product(
                    str(right_camera_path),
                    (1024, 1024),
                    name="right_sensor",
                    force_new=True,
                )
                self._render_products = [rp_left, rp_right]
                # For better performance the render products can be disabled when not in use, and re-enabled only during SDG
                if self._use_temp_rp:
                    self._disable_render_products()
                self._writer.attach(self._render_products)

            def _clear_sdg_render_products(self) -> None:
                """Detach writer and destroy all render products."""
                print(f"[SDG] Clearing SDG render products")
                if self._writer:
                    self._writer.detach()
                for rp in self._render_products:
                    rp.destroy()
                self._render_products.clear()
                if self._stage.GetPrimAtPath("/Replicator"):
                    omni.kit.commands.execute("DeletePrimsCommand", paths=["/Replicator"])

            def _enable_render_products(self) -> None:
                """Enable texture updates on all render products."""
                print(f"[SDG] Enabling render products for SDG..")
                for rp in self._render_products:
                    rp.hydra_texture.set_updates_enabled(True)

            def _disable_render_products(self) -> None:
                """Disable texture updates on all render products."""
                print(f"[SDG] Disabling render products (enabled only during SDG)..")
                for rp in self._render_products:
                    rp.hydra_texture.set_updates_enabled(False)

            def _run_sdg(self) -> None:
                """Execute one SDG capture step synchronously."""
                if self._use_temp_rp:
                    self._enable_render_products()
                rep.orchestrator.step(rt_subframes=16)
                if self._use_temp_rp:
                    self._disable_render_products()

            async def _run_sdg_async(self) -> None:
                """Execute one SDG capture step asynchronously."""
                if self._use_temp_rp:
                    self._enable_render_products()
                await rep.orchestrator.step_async(rt_subframes=16)
                if self._use_temp_rp:
                    self._disable_render_products()

            def _load_next_env(self) -> None:
                """Replace current environment with the next one from the cycle."""
                if self._stage.GetPrimAtPath("/Environment"):
                    omni.kit.commands.execute("DeletePrimsCommand", paths=["/Environment"])
                assets_root_path = get_assets_root_path()
                rep.functional.create.scope(name="Environment")
                rep.functional.create.reference(
                    usd_path=assets_root_path + next(self._cycled_env_urls), name="Environment"
                )

            def _on_sdg_done(self, task) -> None:
                """Callback invoked when async SDG step completes."""
                self._setup_next_frame()

            def _setup_next_frame(self) -> None:
                """Prepare scene for next frame or finish if all frames captured."""
                self._frame_counter += 1
                if self._frame_counter >= self._num_frames:
                    print(f"[SDG] Finished")
                    if self._is_running_in_script_editor():
                        task = asyncio.ensure_future(rep.orchestrator.wait_until_complete_async())
                        task.add_done_callback(lambda t: self.clear())
                    else:
                        rep.orchestrator.wait_until_complete()
                        self.clear()
                    return

                self._randomize_dolly_pose()
                self._randomize_dolly_light()
                self._randomize_prop_poses()
                if self._frame_counter % self._env_interval == 0:
                    self._load_next_env()
                # Set a new random distance from which to take capture the next frame
                self._trigger_distance = random.uniform(1.75, 2.5)
                self._timeline.play()
                self._timeline_sub = self._timeline.get_timeline_event_stream().create_subscription_to_pop_by_type(
                    int(omni.timeline.TimelineEventType.CURRENT_TIME_TICKED), self._on_timeline_event
                )

            def _on_timeline_event(self, e: carb.events.IEvent) -> None:
                """Check distance to dolly and trigger SDG capture when close enough."""
                carter_loc = self._carter_chassis.GetAttribute("xformOp:translate").Get()
                dolly_loc = self._dolly.GetAttribute("xformOp:translate").Get()
                dist = (Gf.Vec2f(dolly_loc[0], dolly_loc[1]) - Gf.Vec2f(carter_loc[0], carter_loc[1])).GetLength()
                if dist < self._trigger_distance:
                    print(f"[SDG] Starting SDG for frame no. {self._frame_counter}")
                    self._timeline.pause()
                    self._timeline_sub.unsubscribe()
                    if self._is_running_in_script_editor():
                        task = asyncio.ensure_future(self._run_sdg_async())
                        task.add_done_callback(self._on_sdg_done)
                    else:
                        self._run_sdg()
                        self._setup_next_frame()

        out_dir = os.path.join(os.getcwd(), "_out_nav_sdg_demo", "")
        nav_demo = NavSDGDemo()
        # asyncio.ensure_future(
        #     nav_demo.run_async(
        #         num_frames=NUM_FRAMES,
        #         out_dir=out_dir,
        #         env_urls=ENV_URLS,
        #         env_interval=ENV_INTERVAL,
        #         use_temp_rp=USE_TEMP_RP,
        #         seed=22,
        #     )
        # )

        # Test parameters
        test_num_frames = 2
        test_env_interval = 1
        test_env_urls = [
            "/Isaac/Environments/Grid/default_environment.usd",
            "/Isaac/Environments/Grid/gridroom_black.usd",
        ]
        await nav_demo.run_async(
            num_frames=test_num_frames,
            env_interval=test_env_interval,
            out_dir=out_dir,
            env_urls=test_env_urls,
            use_temp_rp=USE_TEMP_RP,
            seed=22,
        )

        # Validate that all expected files were written to disk
        num_cameras = 2
        # pngs: num_frames * num_cameras * (rgb)
        expected_pngs = test_num_frames * num_cameras * 1
        all_data_written = validate_folder_contents(out_dir, {"png": expected_pngs}, recursive=True)
        self.assertTrue(all_data_written, f"Not all files were written in to: {out_dir}")
