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
"""IMU sensor backend -- thin Python wrapper around the C++ IImuSensor plugin.

This module provides API-compatible ImuSensorBackend that delegates all
physics computation to the C++ plugin. The C++ backend uses IPrimDataReader
for velocity data and Pose.h for world transforms, supporting both PhysX
and Newton engines transparently.
"""
from __future__ import annotations

from .common import (
    _PhysicsSensorBase,
    _SensorStepManager,
)


def _get_imu_interface() -> object | None:
    from .extension import get_imu_sensor_interface

    return get_imu_sensor_interface()


# Singleton invalid reading cached once to avoid per-call allocation.
_INVALID_IMU_READING = None


def _get_invalid_imu_reading() -> object:
    global _INVALID_IMU_READING
    if _INVALID_IMU_READING is None:
        from .. import _physics_sensors

        _INVALID_IMU_READING = _physics_sensors.ImuSensorReading()
    return _INVALID_IMU_READING


class ImuSensorBackend(_PhysicsSensorBase):
    """Backend implementation for IMU sensors, backed by a C++ plugin.

    Delegates all physics-based IMU computation to the C++ IImuSensor
    Carbonite interface. Returns the C++ pybind struct directly for
    minimal per-read overhead.

    Args:
        prim_path: USD path to the IsaacImuSensor prim.

    Example:

    .. code-block:: python

        >>> from isaacsim.sensors.experimental.physics.impl.imu_sensor_backend import ImuSensorBackend

        >>> backend = ImuSensorBackend("/World/ImuSensor")  # doctest: +NO_CHECK
    """

    def __init__(self, prim_path: str):
        self._prim_path = prim_path
        self._sensor_id: int = -1
        self._iface = None
        _SensorStepManager.instance().register(self)

    def _ensure_sensor(self) -> bool:
        """Ensure the IMU sensor is created and initialized.

        Returns:
            True if the sensor is created and initialized, False otherwise.
        """
        if self._iface is None:
            self._iface = _get_imu_interface()
        if self._iface is None:
            return False
        if self._sensor_id >= 0:
            return True
        self._sensor_id = self._iface.create_sensor(self._prim_path)
        return self._sensor_id >= 0

    def get_sensor_reading(self, read_gravity: bool = True) -> object:
        """Get the current IMU sensor reading.

        Returns the C++ ImuSensorReading struct directly. Access fields via
        ``reading.linear_acceleration_x``, ``reading.orientation_w``, etc.
        An ``orientation`` property returns ``(w, x, y, z)`` as a tuple.
        """
        if self._sensor_id < 0 and not self._ensure_sensor():
            return _get_invalid_imu_reading()

        reading = self._iface.get_sensor_reading(self._sensor_id, read_gravity)
        if not reading.is_valid:
            self._sensor_id = -1
            if not self._ensure_sensor():
                return _get_invalid_imu_reading()
            reading = self._iface.get_sensor_reading(self._sensor_id, read_gravity)
        return reading

    def on_physics_step(self, step_dt: float):
        """Handle physics step events.

        Args:
            step_dt: Physics step duration in seconds.
        """
        pass

    def on_timeline_stop(self):
        """Handle timeline stop events."""
        self._sensor_id = -1
        self._iface = None

    def reset(self):
        """Reset the IMU sensor.

        Removes the sensor from the simulation and resets the sensor ID.
        """
        if self._iface is not None and self._sensor_id >= 0:
            self._iface.remove_sensor(self._sensor_id)
        self._sensor_id = -1
