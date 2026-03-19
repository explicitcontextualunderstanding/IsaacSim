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

import asyncio
import struct
import time

import isaacsim.core.experimental.utils.app as app_utils
import numpy as np
import omni
import omni.graph.core as og
import ucxx._lib.libucxx as ucx_api
from isaacsim.ucx.nodes.tests.common import UCXTestCase, find_available_port
from ucxx._lib.arr import Array

# Test configuration constants
CONNECTION_WAIT_FRAMES = 60
CONNECTION_ESTABLISH_FRAMES = 20
SEND_WAIT_FRAMES = 50
RECEIVE_TIMEOUT_SECONDS = 1.0
RECEIVE_TIMEOUT_FRAMES = 1000
SIMULATION_ADVANCE_TIME = 2.0
SIMULATION_SHORT_ADVANCE_TIME = 0.5
SIMULATION_VERY_SHORT_ADVANCE_TIME = 0.1
DEFAULT_TEST_PORT = 13337
DEFAULT_TEST_TAG = 5
CLOCK_MESSAGE_SIZE_BYTES = 8


class TestUCXPublishClock(UCXTestCase):
    """Test UCX clock publishing"""

    async def setUp(self):
        await super().setUp()
        await omni.usd.get_context().new_stage_async()
        await omni.kit.app.get_app().next_update_async()

    async def setup_ucx_client_with_listener(self):
        """Setup UCX client to connect to the OmniGraph node's listener.

        The OmniGraph nodes create their own internal listeners automatically.
        We create a client endpoint to connect and receive messages from them.
        """
        # Give Isaac Sim a moment to start the node's listener
        for _ in range(CONNECTION_WAIT_FRAMES):
            await omni.kit.app.get_app().next_update_async()

        # Create client connection using the base class helper
        self.create_ucx_client(self.port)

        # Give a few more frames for the connection to establish
        for _ in range(CONNECTION_ESTABLISH_FRAMES):
            await omni.kit.app.get_app().next_update_async()

    async def receive_clock_message(self, tag=DEFAULT_TEST_TAG, timeout_frames=RECEIVE_TIMEOUT_FRAMES):
        """Receive and unpack a clock message from the client endpoint"""
        # Clock message format: double timestamp (8 bytes)
        buffer = np.zeros(CLOCK_MESSAGE_SIZE_BYTES, dtype=np.uint8)  # Initialize with zeros instead of empty

        # Receive using the endpoint
        request = self.client_endpoint.tag_recv(Array(buffer), tag=ucx_api.UCXXTag(tag))

        for _ in range(timeout_frames):
            if request.completed:
                break
            await omni.kit.app.get_app().next_update_async()

        # Check if completed
        self.assertTrue(request.completed, f"Did not receive clock message after {timeout_frames} frames")
        request.check_error()

        # Debug: print raw buffer
        print(f"DEBUG: Clock buffer (hex): {buffer.tobytes().hex()}")
        print(f"DEBUG: Clock buffer (bytes): {list(buffer.tobytes())}")

        # Unpack timestamp (double, native byte order to match the C++ memcpy)
        timestamp = struct.unpack("d", buffer.tobytes())[0]
        print(f"DEBUG: Unpacked timestamp: {timestamp}")
        return timestamp

    async def test_sim_clock(self):
        """Test clock publishing with simulation time"""

        # Create graph with clock publisher using manual trigger (like test_manual_clock)
        try:
            og.Controller.edit(
                {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("OnImpulse", "omni.graph.action.OnImpulseEvent"),
                        ("PublishClock", "isaacsim.ucx.nodes.UCXPublishClock"),
                        ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        ("PublishClock.inputs:port", self.port),
                        ("PublishClock.inputs:tag", DEFAULT_TEST_TAG),
                        ("PublishClock.inputs:timeoutMs", 5000),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("OnImpulse.outputs:execOut", "PublishClock.inputs:execIn"),
                        ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                    ],
                },
            )
        except Exception as e:
            print(f"Error creating graph: {e}")
            raise

        # Start timeline
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()

        # Trigger impulse to set up the node and the listener
        og.Controller.attribute("/ActionGraph/OnImpulse.state:enableImpulse").set(True)
        await omni.kit.app.get_app().next_update_async()

        # Setup client
        await self.setup_ucx_client_with_listener()

        # Post receive request
        buffer = np.zeros(CLOCK_MESSAGE_SIZE_BYTES, dtype=np.uint8)
        request = self.client_endpoint.tag_recv(Array(buffer), tag=ucx_api.UCXXTag(DEFAULT_TEST_TAG))

        # Trigger impulse again to publish the clock
        og.Controller.attribute("/ActionGraph/OnImpulse.state:enableImpulse").set(True)
        await omni.kit.app.get_app().next_update_async()

        # Wait for the message to be received
        start_time = time.time()
        while time.time() - start_time < RECEIVE_TIMEOUT_SECONDS:
            if request.completed:
                break
            await asyncio.sleep(0.001)

        self.assertTrue(request.completed, "Did not receive clock message")
        request.check_error()

        # Unpack timestamp
        timestamp = struct.unpack("d", buffer.tobytes())[0]

        # Verify timestamp is reasonable (simulation time should be positive)
        self.assertGreater(timestamp, 0.0, "Timestamp should be greater than 0.0 seconds")

    async def test_clock_progression(self):
        """Test that clock values increase over time"""

        # Create graph with manual trigger to control when messages are sent
        try:
            og.Controller.edit(
                {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("OnImpulse", "omni.graph.action.OnImpulseEvent"),
                        ("PublishClock", "isaacsim.ucx.nodes.UCXPublishClock"),
                        ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        ("PublishClock.inputs:port", self.port),
                        ("PublishClock.inputs:tag", DEFAULT_TEST_TAG),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("OnImpulse.outputs:execOut", "PublishClock.inputs:execIn"),
                        ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                    ],
                },
            )
        except Exception as e:
            print(f"Error creating graph: {e}")
            raise

        # Start timeline
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()

        # Trigger impulse to publish clock
        og.Controller.attribute("/ActionGraph/OnImpulse.state:enableImpulse").set(True)
        await omni.kit.app.get_app().next_update_async()

        # Setup client
        await self.setup_ucx_client_with_listener()

        timestamps = []

        # Receive multiple clock messages with simulation between each
        for i in range(3):
            # Post receive request BEFORE triggering send
            buffer = np.zeros(CLOCK_MESSAGE_SIZE_BYTES, dtype=np.uint8)
            request = self.client_endpoint.tag_recv(Array(buffer), tag=ucx_api.UCXXTag(DEFAULT_TEST_TAG))

            # Trigger impulse to publish clock
            og.Controller.attribute("/ActionGraph/OnImpulse.state:enableImpulse").set(True)
            await omni.kit.app.get_app().next_update_async()

            # Wait for the receive to complete
            start_time = time.time()
            while not request.completed and time.time() - start_time < RECEIVE_TIMEOUT_SECONDS:
                time.sleep(0.001)

            self.assertTrue(request.completed, f"Did not receive clock message for sample {i}")
            request.check_error()

            # Unpack and store timestamp
            timestamp = struct.unpack("d", buffer.tobytes())[0]
            timestamps.append(timestamp)
            print(f"Clock sample {i}: {timestamp}")

        # Verify timestamps are increasing
        for i in range(1, len(timestamps)):
            self.assertGreater(
                timestamps[i],
                timestamps[i - 1],
                f"Timestamp should increase (sample {i}: {timestamps[i]} <= sample {i-1}: {timestamps[i-1]})",
            )

    async def test_multiple_nodes_same_port(self):
        """Test that multiple nodes can share the same port (listener is reused)"""

        try:
            og.Controller.edit(
                {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("OnImpulse", "omni.graph.action.OnImpulseEvent"),
                        ("PublishClock1", "isaacsim.ucx.nodes.UCXPublishClock"),
                        ("PublishClock2", "isaacsim.ucx.nodes.UCXPublishClock"),
                        ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        ("PublishClock1.inputs:port", self.port),
                        ("PublishClock1.inputs:tag", DEFAULT_TEST_TAG),
                        ("PublishClock2.inputs:port", self.port),  # Same port
                        ("PublishClock2.inputs:tag", DEFAULT_TEST_TAG + 1),  # Different tag
                        ("PublishClock2.inputs:timeoutMs", 1000),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("OnImpulse.outputs:execOut", "PublishClock1.inputs:execIn"),
                        ("OnImpulse.outputs:execOut", "PublishClock2.inputs:execIn"),
                        ("ReadSimTime.outputs:simulationTime", "PublishClock1.inputs:timeStamp"),
                        ("ReadSimTime.outputs:simulationTime", "PublishClock2.inputs:timeStamp"),
                    ],
                },
            )
        except Exception as e:
            print(f"Error creating graph: {e}")
            raise

        # Start timeline
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        og.Controller.attribute("/ActionGraph/OnImpulse.state:enableImpulse").set(True)
        await omni.kit.app.get_app().next_update_async()

        # Setup client
        await self.setup_ucx_client_with_listener()

        # Receive messages with different tags
        og.Controller.attribute("/ActionGraph/OnImpulse.state:enableImpulse").set(True)
        _, timestamp1, timestamp2 = await asyncio.gather(
            omni.kit.app.get_app().next_update_async(),
            self.receive_clock_message(tag=DEFAULT_TEST_TAG),
            self.receive_clock_message(tag=DEFAULT_TEST_TAG + 1),
        )

        print(f"Received from node 1: {timestamp1} seconds")
        print(f"Received from node 2: {timestamp2} seconds")

        # Both should be valid and similar
        self.assertGreater(timestamp1, 0.0)
        self.assertGreater(timestamp2, 0.0)
        self.assertAlmostEqual(timestamp1, timestamp2, delta=0.1)

    async def test_no_connection(self):
        """Test node behavior when no client is connected"""

        another_port = find_available_port()
        try:
            og.Controller.edit(
                {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("PublishClock", "isaacsim.ucx.nodes.UCXPublishClock"),
                        ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        ("PublishClock.inputs:port", another_port),  # Different port
                        ("PublishClock.inputs:tag", DEFAULT_TEST_TAG),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                        ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                    ],
                },
            )
        except Exception as e:
            print(f"Error creating graph: {e}")
            raise

        # Start timeline
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()

        # Don't create a client - just trigger the node
        await omni.kit.app.get_app().next_update_async()

        # Node should handle no connection gracefully (no crash)
        # This is a success if we reach this point
