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

import struct
import time

import isaacsim.core.experimental.utils.app as app_utils
import numpy as np
import omni
import omni.graph.core as og
import ucxx._lib.libucxx as ucx_api
from isaacsim.ucx.nodes.tests.common import UCXTestCase
from ucxx._lib.arr import Array


def unpack_imu_message(buffer):
    """Unpack a UCX IMU message.

    Message format (from OgnUCXPublishImu.cpp):
    - timestamp (double, 8 bytes)
    - frameId_length (uint32_t, 4 bytes)
    - frameId (variable bytes, NO padding)
    - publish_flags (uint8_t, 1 byte): bit 0=orientation, bit 1=linear_accel, bit 2=angular_vel
    - orientation (4 doubles, 32 bytes) - quaternion (IJKR) if bit 0 set
    - linear_acceleration (3 doubles, 24 bytes) - if bit 1 set
    - angular_velocity (3 doubles, 24 bytes) - if bit 2 set

    Args:
        buffer: Buffer containing the packed IMU message.

    Returns:
        Tuple of (timestamp, frame_id, orientation, angular_velocity, linear_acceleration).
    """
    offset = 0
    timestamp = struct.unpack("<d", buffer[offset : offset + 8].tobytes())[0]
    offset += 8

    # Read frame ID
    frame_id_len = struct.unpack("<I", buffer[offset : offset + 4].tobytes())[0]
    offset += 4
    frame_id = buffer[offset : offset + frame_id_len].tobytes().decode("utf-8")
    offset += frame_id_len

    # Read publish flags
    publish_flags = struct.unpack("<B", buffer[offset : offset + 1].tobytes())[0]
    offset += 1

    has_orientation = (publish_flags & 0x01) != 0
    has_linear_accel = (publish_flags & 0x02) != 0
    has_angular_vel = (publish_flags & 0x04) != 0

    orientation = None
    angular_velocity = None
    linear_acceleration = None

    # Read orientation if enabled (IJKR format: x, y, z, w)
    if has_orientation:
        orientation = struct.unpack("<4d", buffer[offset : offset + 32].tobytes())
        offset += 32

    # Read linear acceleration if enabled
    if has_linear_accel:
        linear_acceleration = struct.unpack("<3d", buffer[offset : offset + 24].tobytes())
        offset += 24

    # Read angular velocity if enabled
    if has_angular_vel:
        angular_velocity = struct.unpack("<3d", buffer[offset : offset + 24].tobytes())
        offset += 24

    return timestamp, frame_id, orientation, angular_velocity, linear_acceleration


class TestUCXPublishImu(UCXTestCase):
    """Test UCX IMU publishing"""

    async def setUp(self):
        await super().setUp()
        await omni.usd.get_context().new_stage_async()
        await omni.kit.app.get_app().next_update_async()

    async def receive_imu_message(self, tag=2, timeout_frames=1000):
        """Receive and unpack an IMU message"""
        max_buffer_size = 512
        buffer = np.empty(max_buffer_size, dtype=np.uint8)

        request = self.client_endpoint.tag_recv(Array(buffer), tag=ucx_api.UCXXTag(tag))

        for _ in range(timeout_frames):
            if request.completed:
                break
            time.sleep(0.001)
            await omni.kit.app.get_app().next_update_async()

        self.assertTrue(request.completed, "Did not receive IMU message")
        request.check_error()

        return unpack_imu_message(buffer)

    async def test_imu_basic(self):
        """Test basic IMU publishing"""

        try:
            og.Controller.edit(
                {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("PublishImu", "isaacsim.ucx.nodes.UCXPublishImu"),
                        ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        ("PublishImu.inputs:port", self.port),
                        ("PublishImu.inputs:tag", 2),
                        ("PublishImu.inputs:frameId", "test_imu"),
                        ("PublishImu.inputs:orientation", [0.0, 0.0, 0.0, 1.0]),
                        ("PublishImu.inputs:angularVelocity", [0.1, 0.2, 0.3]),
                        ("PublishImu.inputs:linearAcceleration", [0.0, 0.0, 9.81]),
                        ("PublishImu.inputs:timeoutMs", 1000),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("OnPlaybackTick.outputs:tick", "PublishImu.inputs:execIn"),
                        ("ReadSimTime.outputs:simulationTime", "PublishImu.inputs:timeStamp"),
                    ],
                },
            )
        except Exception as e:
            print(f"Error creating graph: {e}")
            raise

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()

        await app_utils.update_app_async(steps=3)
        self.create_ucx_client(self.port)
        timestamp, frame_id, orientation, angular_vel, linear_accel = await self.receive_imu_message()

        print(f"Received IMU data:")
        print(f"  Timestamp: {timestamp}")
        print(f"  Frame ID: {frame_id}")
        print(f"  Orientation: {orientation}")
        print(f"  Angular velocity: {angular_vel}")
        print(f"  Linear acceleration: {linear_accel}")

        self.assertGreater(timestamp, 0.0)
        self.assertEqual(frame_id, "test_imu")
        self.assertAlmostEqual(linear_accel[2], 9.81, places=1)
