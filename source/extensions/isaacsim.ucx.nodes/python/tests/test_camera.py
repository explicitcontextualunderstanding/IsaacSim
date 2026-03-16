# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import time

# UCX imports
import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.semantics as semantics_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
import omni
import omni.graph.core as og
import omni.kit.commands
import omni.kit.viewport.utility
import ucxx._lib.libucxx as ucx_api
import usdrt.Sdf
from isaacsim.core.experimental.objects import Cube
from isaacsim.core.rendering_manager import ViewportManager
from isaacsim.ucx.nodes.tests.common import UCXTestCase, find_available_port, unpack_image_message
from pxr import Sdf
from ucxx._lib.arr import Array


class TestUCXCamera(UCXTestCase):
    """Test UCX Camera Helper node"""

    async def setUp(self):
        await super().setUp()

        # Get assets root path
        from isaacsim.storage.native import get_assets_root_path

        self._assets_root_path = get_assets_root_path()
        if self._assets_root_path is None:
            raise RuntimeError("Could not find Isaac Sim assets folder")

        omni.usd.get_context().new_stage()
        await omni.kit.app.get_app().next_update_async()

        # Acquire the viewport window
        viewport_api = omni.kit.viewport.utility.get_active_viewport()
        # Set viewport resolution, changes will occur on next frame
        viewport_api.set_texture_resolution((1280, 720))
        await omni.kit.app.get_app().next_update_async()

    async def setup_ucx_client_with_listener(self, port):
        """Setup UCX client.

        Args:
            port: Port number to connect to.
        """
        for _ in range(20):
            await omni.kit.app.get_app().next_update_async()

        self.create_ucx_client(port)

    async def receive_image_message(self, tag=10, timeout_frames=1000, retry_count=15):
        """Receive and unpack an image message.

        Args:
            tag: UCX tag to receive on.
            timeout_frames: Maximum number of frames to wait per retry.
            retry_count: Number of times to retry receiving if data is invalid (default 15).

        Returns:
            Tuple of (timestamp, width, height, encoding, step, image_data).
        """
        # Allocate buffer for image (1280x720 RGB = ~3MB to be safe)
        max_buffer_size = 3 * 1024 * 1024

        for retry in range(retry_count):
            # Initialize buffer to zeros to distinguish no-data from garbage
            buffer = np.zeros(max_buffer_size, dtype=np.uint8)

            # Give worker time to process any pending async operations
            # The sender uses async send, so we need to ensure it completes
            for _ in range(10):
                await omni.kit.app.get_app().next_update_async()

            request = self.client_endpoint.tag_recv(Array(buffer), tag=ucx_api.UCXXTag(tag))

            for _ in range(timeout_frames):
                if request.completed:
                    break
                time.sleep(0.001)
                await omni.kit.app.get_app().next_update_async()

            if not request.completed:
                if retry < retry_count - 1:
                    await app_utils.update_app_async(steps=60)
                    continue
                else:
                    self.fail("Did not receive image message after retries")

            request.check_error()

            try:
                return unpack_image_message(buffer)
            except ValueError:
                # Invalid data received, might be timing issue - retry after waiting
                if retry < retry_count - 1:
                    await app_utils.update_app_async(steps=60)
                    continue
                else:
                    raise

        self.fail("Failed to receive valid image message after all retries")

    async def test_camera_rgb(self):
        """Test RGB camera publishing from render product."""
        scene_path = "/Isaac/Environments/Grid/default_environment.usd"
        await stage_utils.open_stage_async(self._assets_root_path + scene_path)

        cube_1 = Cube("/cube_1", positions=[0, 0, 0], scales=[1.5, 1, 1])
        semantics_utils.add_labels(cube_1.prims[0], labels=["Cube0"], taxonomy="class")

        try:
            og.Controller.edit(
                {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("RGBPublish", "isaacsim.ucx.nodes.UCXCameraHelper"),
                        ("CreateRenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        ("CreateRenderProduct.inputs:cameraPrim", [usdrt.Sdf.Path("/OmniverseKit_Persp")]),
                        ("CreateRenderProduct.inputs:height", 600),
                        ("CreateRenderProduct.inputs:width", 800),
                        ("RGBPublish.inputs:port", self.port),
                        ("RGBPublish.inputs:tag", 10),
                        ("RGBPublish.inputs:resetSimulationTimeOnStop", True),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("OnPlaybackTick.outputs:tick", "CreateRenderProduct.inputs:execIn"),
                        ("CreateRenderProduct.outputs:execOut", "RGBPublish.inputs:execIn"),
                        ("CreateRenderProduct.outputs:renderProductPath", "RGBPublish.inputs:renderProductPath"),
                    ],
                },
            )
        except Exception as e:
            print(e)
            raise

        # Start timeline FIRST so the UCX node executes and creates its listener
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()

        # Now setup UCX client to connect to the listener
        await self.setup_ucx_client_with_listener(self.port)

        # Receive RGB image
        timestamp, width, height, encoding, step, image_data = await self.receive_image_message(tag=10)

        # Verify metadata
        self.assertIsNotNone(image_data)
        self.assertEqual(width, 800)
        self.assertEqual(height, 600)
        self.assertEqual(encoding, "rgb8")
        self.assertEqual(step, 800 * 3)  # RGB = 3 channels

        # Verify data size
        expected_size = height * step
        self.assertEqual(len(image_data), expected_size)

        # Verify timestamp is reasonable (simulation time)
        self.assertGreater(timestamp, 0.0)

    async def test_camera_system_time(self):
        """Test camera publishing with system time."""
        scene_path = "/Isaac/Environments/Grid/default_environment.usd"
        await stage_utils.open_stage_async(self._assets_root_path + scene_path)

        cube_1 = Cube("/cube_1", positions=[0, 0, 0], scales=[1.5, 1, 1])
        semantics_utils.add_labels(cube_1.prims[0], labels=["Cube0"], taxonomy="class")

        try:
            og.Controller.edit(
                {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("RGBPublish", "isaacsim.ucx.nodes.UCXCameraHelper"),
                        ("CreateRenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        ("CreateRenderProduct.inputs:cameraPrim", [usdrt.Sdf.Path("/OmniverseKit_Persp")]),
                        ("CreateRenderProduct.inputs:height", 600),
                        ("CreateRenderProduct.inputs:width", 800),
                        ("RGBPublish.inputs:port", self.port),
                        ("RGBPublish.inputs:tag", 10),
                        ("RGBPublish.inputs:useSystemTime", True),
                        ("RGBPublish.inputs:resetSimulationTimeOnStop", True),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("OnPlaybackTick.outputs:tick", "CreateRenderProduct.inputs:execIn"),
                        ("CreateRenderProduct.outputs:execOut", "RGBPublish.inputs:execIn"),
                        ("CreateRenderProduct.outputs:renderProductPath", "RGBPublish.inputs:renderProductPath"),
                    ],
                },
            )
        except Exception as e:
            print(e)
            raise

        await omni.kit.app.get_app().next_update_async()

        # Start timeline FIRST so the UCX node executes and creates its listener
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()

        # Now setup UCX client to connect to the listener
        await self.setup_ucx_client_with_listener(self.port)

        # Capture system time AFTER connection is established
        system_time = time.time()

        # Wait significantly longer for replicator pipeline to stabilize
        await app_utils.update_app_async(steps=180)

        # Receive RGB image
        timestamp, width, height, encoding, step, image_data = await self.receive_image_message(tag=10)

        # Verify metadata
        self.assertIsNotNone(image_data)
        self.assertEqual(width, 800)
        self.assertEqual(height, 600)
        self.assertEqual(encoding, "rgb8")

        # Verify timestamp is system time (within reasonable range)
        # The image may have been generated slightly before we captured system_time
        # so allow a few seconds of tolerance
        time_diff = abs(timestamp - system_time)
        self.assertLess(
            time_diff, 5.0, f"Timestamp {timestamp} too far from system time {system_time} (diff: {time_diff}s)"
        )

    async def test_camera_frame_skip(self):
        """Test camera publishing with frame skip."""
        scene_path = "/Isaac/Environments/Grid/default_environment.usd"
        await stage_utils.open_stage_async(self._assets_root_path + scene_path)

        Cube("/cube_1", positions=[0, 0, 0], scales=[1.5, 1, 1])

        try:
            og.Controller.edit(
                {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("RGBPublish", "isaacsim.ucx.nodes.UCXCameraHelper"),
                        ("CreateRenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        ("CreateRenderProduct.inputs:cameraPrim", [usdrt.Sdf.Path("/OmniverseKit_Persp")]),
                        ("CreateRenderProduct.inputs:height", 480),
                        ("CreateRenderProduct.inputs:width", 640),
                        ("RGBPublish.inputs:port", self.port),
                        ("RGBPublish.inputs:tag", 10),
                        ("RGBPublish.inputs:frameSkipCount", 5),  # Skip 5 frames, publish every 6th
                        ("RGBPublish.inputs:resetSimulationTimeOnStop", True),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("OnPlaybackTick.outputs:tick", "CreateRenderProduct.inputs:execIn"),
                        ("CreateRenderProduct.outputs:execOut", "RGBPublish.inputs:execIn"),
                        ("CreateRenderProduct.outputs:renderProductPath", "RGBPublish.inputs:renderProductPath"),
                    ],
                },
            )
        except Exception as e:
            print(e)
            raise

        # Start timeline FIRST so the UCX node executes and creates its listener
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()

        # Now setup UCX client to connect to the listener
        await self.setup_ucx_client_with_listener(self.port)

        # Receive RGB image (with longer timeout since publish is less frequent)
        timestamp, width, height, encoding, step, image_data = await self.receive_image_message(
            tag=10, timeout_frames=3000
        )

        # Verify metadata
        self.assertIsNotNone(image_data)
        self.assertEqual(width, 640)
        self.assertEqual(height, 480)
        self.assertEqual(encoding, "rgb8")

    async def test_camera_multiple_resolutions(self):
        """Test camera publishing with different resolutions."""
        scene_path = "/Isaac/Environments/Grid/default_environment.usd"
        await stage_utils.open_stage_async(self._assets_root_path + scene_path)

        Cube("/cube_1", positions=[0, 0, 0], scales=[1.5, 1, 1])
        ViewportManager.set_camera_view(camera="/OmniverseKit_Persp", eye=[0, -6, 0.5], target=[0, 0, 0.5])

        # Test with different resolutions
        resolutions = [(320, 240), (640, 480), (1280, 720)]

        for idx, (width, height) in enumerate(resolutions):
            port = find_available_port()
            tag = 10 + idx

            try:
                og.Controller.edit(
                    {"graph_path": f"/ActionGraph_{idx}", "evaluator_name": "execution"},
                    {
                        og.Controller.Keys.CREATE_NODES: [
                            ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                            ("RGBPublish", "isaacsim.ucx.nodes.UCXCameraHelper"),
                            ("CreateRenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                        ],
                        og.Controller.Keys.SET_VALUES: [
                            ("CreateRenderProduct.inputs:cameraPrim", [usdrt.Sdf.Path("/OmniverseKit_Persp")]),
                            ("CreateRenderProduct.inputs:height", height),
                            ("CreateRenderProduct.inputs:width", width),
                            ("RGBPublish.inputs:port", port),
                            ("RGBPublish.inputs:tag", tag),
                            ("RGBPublish.inputs:resetSimulationTimeOnStop", True),
                        ],
                        og.Controller.Keys.CONNECT: [
                            ("OnPlaybackTick.outputs:tick", "CreateRenderProduct.inputs:execIn"),
                            ("CreateRenderProduct.outputs:execOut", "RGBPublish.inputs:execIn"),
                            ("CreateRenderProduct.outputs:renderProductPath", "RGBPublish.inputs:renderProductPath"),
                        ],
                    },
                )
            except Exception as e:
                print(e)
                raise

            # Start timeline FIRST so the UCX node executes and creates its listener
            timeline = omni.timeline.get_timeline_interface()
            timeline.play()

            # Setup client for this resolution
            await self.setup_ucx_client_with_listener(port=port)

            # Receive RGB image
            timestamp, recv_width, recv_height, encoding, step, image_data = await self.receive_image_message(tag=tag)

            # Verify metadata
            self.assertEqual(recv_width, width, f"Width mismatch for resolution {width}x{height}")
            self.assertEqual(recv_height, height, f"Height mismatch for resolution {width}x{height}")
            self.assertEqual(encoding, "rgb8")
            self.assertEqual(step, width * 3)

            # Verify data size
            expected_size = height * step
            self.assertEqual(len(image_data), expected_size, f"Data size mismatch for resolution {width}x{height}")

            timeline.stop()
            await omni.kit.app.get_app().next_update_async()

            # Clean up UCX client for next iteration
            if self.client_worker:
                try:
                    self.client_worker.stop_progress_thread()
                except Exception:
                    pass
            self.client_endpoint = None
            self.client_worker = None
            self.client_context = None

            # Important: Clean up the graph to avoid interference with next resolution
            graph_path = f"/ActionGraph_{idx}"
            omni.kit.commands.execute("DeletePrims", paths=[graph_path])
