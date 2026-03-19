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
"""Contact sensor wrapper providing high-level sensor access.

This module provides the ContactSensor class which wraps the ContactSensorBackend
with a convenient object-oriented interface including frame-based data access.
"""
from __future__ import annotations

from typing import Any, cast

import carb
import numpy as np
import omni.isaac.IsaacSensorSchema as IsaacSensorSchema
import omni.kit.commands
from isaacsim.core.experimental.prims import XformPrim
from isaacsim.core.experimental.utils import prim as prim_utils
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.sensors.experimental.physics.impl.contact_sensor_backend import ContactSensorBackend
from pxr import Gf, PhysicsSchemaTools, UsdPhysics


class ContactSensor(XformPrim):
    """High-level contact sensor wrapper with frame-based data access.

    Provides a convenient interface for contact sensing including automatic
    sensor creation if the prim doesn't exist, configurable thresholds and
    radius, and structured frame data output.

    Args:
        prim_path: USD path where the sensor should be located.
        name: Human-readable name for the sensor.
        translation: Local translation offset from parent. Cannot be used with position.
        position: World position. Cannot be used with translation.
        min_threshold: Minimum force threshold in Newtons.
        max_threshold: Maximum force threshold in Newtons.
        radius: Contact detection radius. Negative means no radius filtering.

    Raises:
        ValueError: If both position and translation are specified.
        ValueError: If parent prim doesn't have CollisionAPI.
        RuntimeError: If sensor creation fails.

    Example:

    .. code-block:: python

        from isaacsim.sensors.experimental.physics import ContactSensor

        # Create sensor on existing prim
        sensor = ContactSensor("/World/Robot/foot/contact_sensor")

        # Or create new sensor with custom parameters
        sensor = ContactSensor(
            "/World/Robot/foot/contact_sensor",
            min_threshold=1.0,
            max_threshold=1000.0,
            radius=0.05
        )

        # Get current contact data
        frame = sensor.get_current_frame()
        if frame["in_contact"]:
            print(f"Contact force: {frame['force']}")
    """

    def __init__(
        self,
        prim_path: str,
        name: str | None = "contact_sensor",
        translation: np.ndarray | None = None,
        position: np.ndarray | None = None,
        min_threshold: float | None = None,
        max_threshold: float | None = None,
        radius: float | None = None,
    ):
        if position is not None and translation is not None:
            raise ValueError("Sensor position and translation can't be both specified")

        # Extract parent path (sensor must be under a collision-enabled prim)
        self._body_prim_path = "/".join(prim_path.split("/")[:-1])
        prim = prim_utils.get_prim_at_path(prim_path)

        # Validate parent has collision API
        if prim.IsValid() and not prim_utils.has_api(self._body_prim_path, UsdPhysics.CollisionAPI):
            raise ValueError("Contact Sensor needs to be created under another prim that has collision api enabled on.")

        self._sensor_name = prim_path.split("/")[-1]
        if prim.IsValid():
            # Use existing sensor prim
            self._isaac_sensor_prim = IsaacSensorSchema.IsaacContactSensor(prim)
            super().__init__(
                prim_path,
                positions=position,
                translations=translation,
                reset_xform_op_properties=True,
            )
            # Apply optional parameter overrides
            if min_threshold is not None:
                self.set_min_threshold(min_threshold)
            if max_threshold is not None:
                self.set_max_threshold(max_threshold)
            if radius is not None:
                self.set_radius(radius)
        else:
            # Create new sensor prim with defaults
            if min_threshold is None:
                min_threshold = 0
            if max_threshold is None:
                max_threshold = 100000
            if radius is None:
                radius = -1
            color_rgba = np.array([1.0, 1.0, 1.0, 1.0])

            carb.log_info(f"Creating a new contact sensor prim at path {prim_path}")
            success, self._isaac_sensor_prim = omni.kit.commands.execute(
                "IsaacSensorExperimentalCreateContactSensor",
                path="/" + self._sensor_name,
                parent=self._body_prim_path,
                min_threshold=min_threshold,
                max_threshold=max_threshold,
                color=Gf.Vec4f(*color_rgba.tolist()),
                radius=radius,
            )
            if not success:
                raise RuntimeError("Failed to create contact sensor prim")
            super().__init__(
                prim_path,
                positions=position,
                translations=translation,
                reset_xform_op_properties=True,
            )

        self._prim = self.prims[0]
        self._backend: ContactSensorBackend = ContactSensorBackend(prim_path)

        self._current_time = 0.0
        self._number_of_physics_steps = 0.0

        # Initialize frame data structure
        self._current_frame: dict[str, object] = {
            "time": 0.0,
            "physics_step": 0.0,
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

    def get_current_frame(self) -> dict:
        """Get the current contact sensor data as a structured frame.

        Returns:
            Frame data containing:
            - "in_contact": Whether contact is detected.
            - "force": Contact force magnitude.
            - "time": Simulation time of reading.
            - "physics_step": Physics step number.
            - "number_of_contacts": Number of contact points.
            - "contacts": Raw contact data if enabled via ``add_raw_contact_data_to_frame``.

        Example:

        .. code-block:: python

            >>> frame = sensor.get_current_frame()  # doctest: +NO_CHECK
            >>> frame["in_contact"]  # doctest: +NO_CHECK
            False
        """
        contact_sensor_reading = self._backend.get_sensor_reading()

        if contact_sensor_reading.is_valid:
            self._current_frame["in_contact"] = bool(contact_sensor_reading.in_contact)
            self._current_frame["force"] = float(contact_sensor_reading.value)
            self._current_frame["time"] = float(contact_sensor_reading.time)
            self._current_frame["physics_step"] = float(SimulationManager.get_num_physics_steps())

            contact_raw_data = self._backend.get_raw_data()
            self._current_frame["number_of_contacts"] = len(contact_raw_data)

            if isinstance(self._current_frame.get("contacts"), list):
                contacts: list[dict[str, object]] = []
                for i in range(len(contact_raw_data)):
                    contact_frame: dict[str, object] = {}
                    body0 = cast(int, contact_raw_data[i]["body0"])
                    body1 = cast(int, contact_raw_data[i]["body1"])
                    contact_frame["body0"] = str(PhysicsSchemaTools.intToSdfPath(int(body0)))
                    contact_frame["body1"] = str(PhysicsSchemaTools.intToSdfPath(int(body1)))
                    position_dict = cast(dict[str, float], contact_raw_data[i]["position"])
                    contact_frame["position"] = np.array(
                        [
                            position_dict["x"],
                            position_dict["y"],
                            position_dict["z"],
                        ],
                        dtype=np.float32,
                    )
                    normal_dict = cast(dict[str, float], contact_raw_data[i]["normal"])
                    contact_frame["normal"] = np.array(
                        [
                            normal_dict["x"],
                            normal_dict["y"],
                            normal_dict["z"],
                        ],
                        dtype=np.float32,
                    )
                    impulse_dict = cast(dict[str, float], contact_raw_data[i]["impulse"])
                    contact_frame["impulse"] = np.array(
                        [
                            impulse_dict["x"],
                            impulse_dict["y"],
                            impulse_dict["z"],
                        ],
                        dtype=np.float32,
                    )
                    contacts.append(contact_frame)
                self._current_frame["contacts"] = contacts

        return self._current_frame

    def add_raw_contact_data_to_frame(self):
        """Enable raw contact data in frame output.

        After calling this, get_current_frame() will include a "contacts" list
        with detailed per-contact information.

        Example:

        .. code-block:: python

            >>> sensor.add_raw_contact_data_to_frame()
        """
        contacts: list[dict[str, object]] = []
        self._current_frame["contacts"] = contacts

    def remove_raw_contact_data_from_frame(self):
        """Disable raw contact data in frame output.

        Removes the "contacts" key from frame output to reduce overhead.

        Example:

        .. code-block:: python

            >>> sensor.remove_raw_contact_data_from_frame()
        """
        del self._current_frame["contacts"]

    def get_radius(self) -> float:
        """Get the contact detection radius.

        Returns:
            Radius in stage units. Negative means no radius filtering.

        Example:

        .. code-block:: python

            >>> sensor.get_radius()  # doctest: +NO_CHECK
            -1.0
        """
        return self._prim.GetAttribute("radius").Get()

    def set_radius(self, value: float):
        """Set the contact detection radius.

        Args:
            value: Radius in stage units. Use negative to disable radius filtering.

        Example:

        .. code-block:: python

            >>> sensor.set_radius(0.1)
        """
        if self.get_radius() is None:
            self._isaac_sensor_prim.CreateRadiusAttr().Set(value)
        else:
            self._prim.GetAttribute("radius").Set(value)

    def get_min_threshold(self) -> float | None:
        """Minimum force threshold in Newtons.

        Returns:
            Minimum threshold in Newtons, or None if not set.

        Example:

        .. code-block:: python

            >>> sensor.get_min_threshold()  # doctest: +NO_CHECK
            0.0
        """
        threshold = self._prim.GetAttribute("threshold").Get()
        if threshold is not None:
            return threshold[0]
        else:
            return None

    def set_min_threshold(self, value: float):
        """Set the minimum force threshold.

        Contacts with force below this threshold are ignored.

        Args:
            value: Threshold in Newtons.

        Example:

        .. code-block:: python

            >>> sensor.set_min_threshold(0.5)
        """
        if self.get_min_threshold() is None:
            self._isaac_sensor_prim.CreateThresholdAttr().Set((value, 10000))
        else:
            self._prim.GetAttribute("threshold").Set((value, self.get_max_threshold()))

    def get_max_threshold(self) -> float | None:
        """Maximum force threshold in Newtons.

        Returns:
            Maximum threshold in Newtons, or None if not set.

        Example:

        .. code-block:: python

            >>> sensor.get_max_threshold()  # doctest: +NO_CHECK
            100000.0
        """
        threshold = self._prim.GetAttribute("threshold").Get()
        if threshold is not None:
            return threshold[1]
        else:
            return None

    def set_max_threshold(self, value: float):
        """Set the maximum force threshold.

        Contact forces are clamped to this maximum value.

        Args:
            value: Threshold in Newtons.

        Example:

        .. code-block:: python

            >>> sensor.set_max_threshold(2000.0)
        """
        if self.get_max_threshold() is None:
            self._isaac_sensor_prim.CreateThresholdAttr().Set((0, value))
        else:
            self._prim.GetAttribute("threshold").Set((self.get_min_threshold(), value))
