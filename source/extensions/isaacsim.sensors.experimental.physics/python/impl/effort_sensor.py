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
"""Effort sensor for measuring joint torques.

This module provides the EffortSensor class for reading joint effort
(torque/force) values from articulated physics bodies. All physics
computation is delegated to the C++ IEffortSensor plugin via
EffortSensorBackend.
"""
from __future__ import annotations

from collections import deque
from typing import Any

import carb
import carb.eventdispatcher
import omni.timeline
import omni.usd
from isaacsim.sensors.experimental.physics.impl.effort_sensor_backend import EffortSensorBackend


class EffortSensorReading:
    """Effort sensor reading data.

    Args:
        is_valid: Whether this reading contains valid data.
        time: Simulation time when the reading was taken.
        value: Effort (torque/force) value at the joint.
    """

    def __init__(self, is_valid: bool = False, time: float = 0, value: float = 0):
        self.is_valid = is_valid
        self.time = time
        self.value = value


class EffortSensor:
    """Sensor for measuring joint effort (torque/force).

    Reads effort values from an articulated body's joint via the C++
    IEffortSensor plugin. Maintains a Python-side circular buffer of
    readings for historical access.

    Args:
        prim_path: USD path to the joint, formatted as articulation_path/joint_name.
        enabled: Whether the sensor is initially enabled.

    Example:

        .. code-block:: python

            from isaacsim.sensors.experimental.physics import EffortSensor

            # Create sensor for a robot joint
            sensor = EffortSensor("/World/Robot/joint_1")

            # Read joint effort
            reading = sensor.get_sensor_reading()
            if reading.is_valid:
                print(f"Joint torque: {reading.value}")
    """

    def __init__(self, prim_path: str, enabled: bool = True):
        self.enabled = enabled
        self.prim_path = prim_path

        self.body_prim_path = "/".join(prim_path.split("/")[:-1])
        self.dof_name = prim_path.split("/")[-1]

        self.data_buffer_size = 10
        self.sensor_reading_buffer = deque(
            (EffortSensorReading() for _ in range(self.data_buffer_size)), maxlen=self.data_buffer_size
        )

        self.physics_num_steps = 0.0
        self.current_time = 0.0

        self._backend = EffortSensorBackend(prim_path)
        self._callback_ids: list[Any] = []

        self._initialize_callbacks()

    def _initialize_callbacks(self) -> None:
        """Register timeline event callbacks for lifecycle management."""
        self._stage_open_sub = carb.eventdispatcher.get_eventdispatcher().observe_event(
            event_name=omni.usd.get_context().stage_event_name(omni.usd.StageEventType.OPENED),
            on_event=self._stage_open_callback_fn,
            observer_name="isaacsim.sensors.experimental.physics.EffortSensor._stage_open_callback",
        )

        self._timeline_stop_sub = carb.eventdispatcher.get_eventdispatcher().observe_event(
            event_name=omni.timeline.GLOBAL_EVENT_STOP,
            on_event=self._timeline_stop_callback_fn,
            observer_name="isaacsim.sensors.experimental.physics.EffortSensor._timeline_stop_callback",
        )

    def _stage_open_callback_fn(self, event: Any = None):
        """Handle stage open by releasing all subscriptions.

        Args:
            event: Stage opened event payload.
        """
        self._stage_open_sub = None
        self._timeline_stop_sub = None

    def _timeline_stop_callback_fn(self, event: Any):
        """Handle timeline stop by resetting state.

        Args:
            event: Timeline stop event payload.
        """
        self.current_time = 0
        self.sensor_reading_buffer = deque(
            (EffortSensorReading() for _ in range(self.data_buffer_size)), maxlen=self.data_buffer_size
        )
        self.physics_num_steps = 0
        self._backend.on_timeline_stop()

    def get_sensor_reading(self) -> EffortSensorReading:
        """Get the current effort sensor reading.

        Returns:
            Reading with effort value and validity state.

        Example:

        .. code-block:: python

            >>> reading = sensor.get_sensor_reading()  # doctest: +NO_CHECK
            >>> reading.is_valid  # doctest: +NO_CHECK
            False
        """
        if not self.enabled:
            return EffortSensorReading()

        cpp_reading = self._backend.get_sensor_reading()
        if not cpp_reading.is_valid:
            return EffortSensorReading()

        reading = EffortSensorReading(
            is_valid=True,
            time=cpp_reading.time,
            value=cpp_reading.value,
        )

        self.sensor_reading_buffer.appendleft(reading)

        return reading

    def update_dof_name(self, dof_name: str):
        """Update the DOF (degree of freedom) name being measured.

        Creates a new backend targeting the updated joint path.

        Args:
            dof_name: Name of the joint DOF to monitor.

        Example:

        .. code-block:: python

            >>> sensor.update_dof_name("joint_1")
        """
        self.dof_name = dof_name
        new_path = self.body_prim_path + "/" + dof_name
        self._backend.reset()
        self._backend = EffortSensorBackend(new_path)

    def change_buffer_size(self, new_buffer_size: int):
        """Change the size of the sensor reading buffer.

        Args:
            new_buffer_size: New buffer size (number of readings to store).

        Example:

        .. code-block:: python

            >>> sensor.change_buffer_size(4)
        """
        old = list(self.sensor_reading_buffer)
        self.data_buffer_size = new_buffer_size
        self.sensor_reading_buffer = deque(maxlen=new_buffer_size)
        for item in old[:new_buffer_size]:
            self.sensor_reading_buffer.append(item)
        while len(self.sensor_reading_buffer) < new_buffer_size:
            self.sensor_reading_buffer.append(EffortSensorReading())
