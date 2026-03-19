# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""IMU sensor wrapper providing high-level sensor access.

This module provides the IMUSensor class which wraps the ImuSensorBackend
with a convenient object-oriented interface including frame-based data access.
"""
from __future__ import annotations

from typing import Any

import carb
import numpy as np
import omni.isaac.IsaacSensorSchema as IsaacSensorSchema
import omni.kit.commands
from isaacsim.core.experimental.prims import XformPrim
from isaacsim.core.experimental.utils import prim as prim_utils
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.sensors.experimental.physics.impl.imu_sensor_backend import ImuSensorBackend


class IMUSensor(XformPrim):
    """High-level IMU sensor wrapper with frame-based data access.

    Provides a convenient interface for IMU sensing including automatic
    sensor creation if the prim doesn't exist, configurable filter widths,
    and structured frame data output.

    Args:
        prim_path: USD path where the sensor should be located.
        name: Human-readable name for the sensor.
        translation: Local translation offset from parent. Cannot be used with position.
        position: World position. Cannot be used with translation.
        orientation: Sensor orientation as [w, x, y, z] quaternion.
        linear_acceleration_filter_size: Rolling average window for acceleration.
        angular_velocity_filter_size: Rolling average window for angular velocity.
        orientation_filter_size: Rolling average window for orientation.

    Raises:
        ValueError: If both position and translation are specified.
        RuntimeError: If sensor creation fails.

    Example:

        .. code-block:: python

            from isaacsim.sensors.experimental.physics import IMUSensor

            # Create sensor on existing prim
            sensor = IMUSensor("/World/Robot/body/imu")

            # Or create new sensor with custom parameters
            sensor = IMUSensor(
                "/World/Robot/body/imu",
                linear_acceleration_filter_size=5
            )

            # Get current IMU data
            frame = sensor.get_current_frame()
            print(f"Linear acceleration: {frame['linear_acceleration']}")
            print(f"Angular velocity: {frame['angular_velocity']}")
            print(f"Orientation: {frame['orientation']}")
    """

    def __init__(
        self,
        prim_path: str,
        name: str | None = "imu_sensor",
        translation: np.ndarray | None = None,
        position: np.ndarray | None = None,
        orientation: np.ndarray | None = None,
        linear_acceleration_filter_size: int | None = 1,
        angular_velocity_filter_size: int | None = 1,
        orientation_filter_size: int | None = 1,
    ):
        if position is not None and translation is not None:
            raise ValueError("Sensor position and translation can't be both specified")

        # Extract parent path
        self._body_prim_path = "/".join(prim_path.split("/")[:-1])
        self._sensor_name = prim_path.split("/")[-1]
        # Ensure filter sizes are at least 1
        if linear_acceleration_filter_size is None:
            linear_acceleration_filter_size = 1
        if angular_velocity_filter_size is None:
            angular_velocity_filter_size = 1
        if orientation_filter_size is None:
            orientation_filter_size = 1
        linear_acceleration_filter_size = max(linear_acceleration_filter_size, 1)
        angular_velocity_filter_size = max(angular_velocity_filter_size, 1)
        orientation_filter_size = max(orientation_filter_size, 1)

        prim = prim_utils.get_prim_at_path(prim_path)
        if prim.IsValid():
            # Use existing sensor prim
            self._isaac_sensor_prim = IsaacSensorSchema.IsaacImuSensor(prim)
            super().__init__(
                prim_path,
                positions=position,
                translations=translation,
                orientations=orientation,
                reset_xform_op_properties=True,
            )
        else:
            # Create new sensor prim
            carb.log_warn(f"Creating a new IMU prim at path {prim_path}")
            success, self._isaac_sensor_prim = omni.kit.commands.execute(
                "IsaacSensorExperimentalCreateImuSensor",
                path="/" + self._sensor_name,
                parent=self._body_prim_path,
                linear_acceleration_filter_size=linear_acceleration_filter_size,
                angular_velocity_filter_size=angular_velocity_filter_size,
                orientation_filter_size=orientation_filter_size,
            )
            if not success:
                raise RuntimeError("Failed to create IMU sensor prim")
            super().__init__(
                prim_path,
                positions=position,
                translations=translation,
                orientations=orientation,
                reset_xform_op_properties=True,
            )

        self._prim = self.prims[0]
        self._backend: ImuSensorBackend = ImuSensorBackend(prim_path)

        self._current_time = 0.0

        # Initialize frame data structure with default values
        orientation_array = np.zeros((4,), dtype=np.float32)
        orientation_array[0] = 1.0  # Identity quaternion [w, x, y, z]
        self._current_frame: dict[str, object] = {
            "time": 0.0,
            "physics_step": 0.0,
            "linear_acceleration": np.zeros((3,), dtype=np.float32),
            "angular_velocity": np.zeros((3,), dtype=np.float32),
            "orientation": orientation_array,
        }

    @property
    def prim_path(self) -> str:
        """Get the USD path of this sensor.

        Returns:
            USD path string.
        """
        return self.paths[0]

    def initialize(self, physics_sim_view: Any = None):
        """Initialize the sensor for simulation.

        This method is provided for API compatibility and currently performs
        no action as initialization happens automatically.

        Args:
            physics_sim_view: Unused. Provided for API compatibility.
        """

    def get_current_frame(self, read_gravity: bool = True) -> dict:
        """Get the current IMU sensor data as a structured frame.

        Args:
            read_gravity: If True, include gravity in acceleration readings.

        Returns:
            Frame data containing:
            - "linear_acceleration": Linear acceleration [x, y, z].
            - "angular_velocity": Angular velocity [x, y, z].
            - "orientation": Orientation as [w, x, y, z] quaternion.
            - "time": Simulation time of reading.
            - "physics_step": Physics step number.

        Example:

        .. code-block:: python

            >>> frame = sensor.get_current_frame()  # doctest: +NO_CHECK
            >>> frame["orientation"]  # doctest: +NO_CHECK
            array([1., 0., 0., 0.], dtype=float32)
        """
        reading = self._backend.get_sensor_reading(read_gravity=read_gravity)

        if reading.is_valid:
            linear_acceleration = self._current_frame["linear_acceleration"]
            linear_acceleration[0] = reading.linear_acceleration_x
            linear_acceleration[1] = reading.linear_acceleration_y
            linear_acceleration[2] = reading.linear_acceleration_z

            angular_velocity = self._current_frame["angular_velocity"]
            angular_velocity[0] = reading.angular_velocity_x
            angular_velocity[1] = reading.angular_velocity_y
            angular_velocity[2] = reading.angular_velocity_z

            orientation = self._current_frame["orientation"]
            orientation[0] = reading.orientation_w
            orientation[1] = reading.orientation_x
            orientation[2] = reading.orientation_y
            orientation[3] = reading.orientation_z

            self._current_frame["time"] = reading.time
            self._current_frame["physics_step"] = float(SimulationManager.get_num_physics_steps())

        return self._current_frame
