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

import os
import socket
import struct

try:
    from isaacsim.test.utils import TimedAsyncTestCase
except ImportError:
    raise ImportError(
        "isaacsim.test.utils is required to use UCXTestCase. " "This module should only be imported in test contexts."
    )

# UCX imports
import omni
import omni.timeline
import ucxx._lib.libucxx as ucx_api


def find_available_port() -> int:
    """Find an available port for a UCX listener.

    Creates a temporary socket bound to port 0, allowing the OS to assign
    an available ephemeral port, then closes the socket and returns the port.

    Returns:
        An available port number.

    Example:

    .. code-block:: python

        >>> from isaacsim.ucx.nodes.tests.common import find_available_port
        >>>
        >>> port = find_available_port()
        >>> isinstance(port, int) and 1024 <= port <= 65535
        True
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


class UCXTestCase(TimedAsyncTestCase):
    """Base class for UCX node tests"""

    async def setUp(self):
        await super().setUp()

        # Set UCX environment variables for TCP-only transport
        os.environ["UCX_TLS"] = "tcp,self"
        os.environ["UCX_NET_DEVICES"] = "all"

        self.port = find_available_port()

        # Initialize UCX client tracking
        self.client_context = None
        self.client_worker = None
        self.client_endpoint = None

    async def tearDown(self):

        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()

        await omni.kit.app.get_app().next_update_async()

        # Clean up UCX client resources
        if self.client_worker:
            try:
                self.client_worker.stop_progress_thread()
            except Exception:
                pass

        if self.client_endpoint:
            self.client_endpoint = None

        self.client_context = None
        self.client_worker = None

        await super().tearDown()

    def create_ucx_client(self, port: int):
        """Create a UCX client connection to the specified port.

        Args:
            port: Port number to connect to.

        Returns:
            Tuple of (context, worker, endpoint).
        """
        self.client_context = ucx_api.UCXContext()
        self.client_worker = ucx_api.UCXWorker(self.client_context)

        self.client_endpoint = ucx_api.UCXEndpoint.create(
            self.client_worker,
            "127.0.0.1",
            port,
            endpoint_error_handling=True,
        )

        # Start the progress thread to handle async communication
        self.client_worker.start_progress_thread()

        return self.client_context, self.client_worker, self.client_endpoint


def unpack_image_message(buffer):
    """Unpack a UCX image message.

    Message format:
    - timestamp (double, 8 bytes)
    - width (uint32_t, 4 bytes)
    - height (uint32_t, 4 bytes)
    - encoding length (uint32_t, 4 bytes)
    - encoding (variable bytes, padded to 4-byte boundary)
    - step (uint32_t, 4 bytes) - row length in bytes
    - image data (variable bytes)

    Args:
        buffer: Buffer containing the packed image message.

    Returns:
        Tuple of (timestamp, width, height, encoding, step, image_data).

    Raises:
        ValueError: If the buffer contains invalid data.
    """
    offset = 0

    # Ensure buffer has minimum size for header
    if len(buffer) < 24:
        raise ValueError(f"Buffer too small: {len(buffer)} bytes, need at least 24")

    timestamp = struct.unpack("<d", buffer[offset : offset + 8].tobytes())[0]
    offset += 8

    width = struct.unpack("<I", buffer[offset : offset + 4].tobytes())[0]
    offset += 4

    height = struct.unpack("<I", buffer[offset : offset + 4].tobytes())[0]
    offset += 4

    # Read encoding
    encoding_len = struct.unpack("<I", buffer[offset : offset + 4].tobytes())[0]
    offset += 4

    # Validate encoding length
    if encoding_len == 0 or encoding_len > 100:
        raise ValueError(f"Invalid encoding length: {encoding_len}")

    if offset + encoding_len > len(buffer):
        raise ValueError(f"Buffer too small for encoding: need {offset + encoding_len}, have {len(buffer)}")

    try:
        encoding = buffer[offset : offset + encoding_len].tobytes().decode("utf-8")
    except UnicodeDecodeError as e:
        # Print buffer content for debugging
        raw_bytes = buffer[offset : offset + encoding_len].tobytes()
        raise ValueError(f"Failed to decode encoding string (length {encoding_len}): {raw_bytes.hex()} - {e}")

    offset += encoding_len
    offset = (offset + 3) & ~3  # Align to 4-byte boundary

    if offset + 4 > len(buffer):
        raise ValueError(f"Buffer too small for step field: need {offset + 4}, have {len(buffer)}")

    step = struct.unpack("<I", buffer[offset : offset + 4].tobytes())[0]
    offset += 4

    # Validate dimensions
    if width == 0 or height == 0:
        raise ValueError(f"Invalid dimensions: width={width}, height={height}")

    # Read image data
    data_size = height * step
    if offset + data_size > len(buffer):
        raise ValueError(f"Buffer too small for image data: need {offset + data_size}, have {len(buffer)}")

    image_data = buffer[offset : offset + data_size]

    return timestamp, width, height, encoding, step, image_data
