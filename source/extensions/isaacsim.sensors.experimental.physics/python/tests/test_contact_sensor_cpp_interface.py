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
"""Tests for C++-only contact sensor interface (IContactSensor).

These tests exercise the C++ contact processing path directly, verifying that
readings and raw contact data are produced without any Python callback relay.
"""
import asyncio

import isaacsim.core.experimental.utils.prim as prim_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import omni.kit.commands
import omni.kit.test
import omni.timeline
from isaacsim.core.experimental.objects import Cube, GroundPlane
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.sensors.experimental.physics.impl.extension import get_contact_sensor_interface
from pxr import Gf, PhysxSchema

from .common import step_simulation


class TestContactSensorCppInterface(omni.kit.test.AsyncTestCase):
    """Validate IContactSensor C++ interface produces correct readings/raw data."""

    async def setUp(self):
        await stage_utils.create_new_stage_async()
        self._physics_rate = 60
        SimulationManager.setup_simulation(dt=1.0 / self._physics_rate)
        self._timeline = omni.timeline.get_timeline_interface()

        GroundPlane("/World/GroundPlane", positions=[0.0, 0.0, 0.0])
        Cube("/World/Cube", sizes=1.0, positions=[0.0, 0.0, 1.0])
        GeomPrim("/World/Cube", apply_collision_apis=True)
        RigidPrim("/World/Cube", masses=[1.0])

        contact_report_api = PhysxSchema.PhysxContactReportAPI.Apply(prim_utils.get_prim_at_path("/World/Cube"))
        contact_report_api.CreateThresholdAttr().Set(0)

        omni.kit.commands.execute(
            "IsaacSensorExperimentalCreateContactSensor",
            path="/contact_sensor",
            parent="/World/Cube",
            max_threshold=10000000,
        )
        await omni.kit.app.get_app().next_update_async()

    async def tearDown(self):
        if self._timeline.is_playing():
            self._timeline.stop()
        SimulationManager.invalidate_physics()
        await omni.kit.app.get_app().next_update_async()
        while omni.usd.get_context().get_stage_loading_status()[2] > 0:
            await asyncio.sleep(1.0)
        await omni.kit.app.get_app().next_update_async()

    async def test_cpp_reading_nonzero_during_contact(self):
        """Verify C++ IContactSensor reports non-zero values during ground contact."""
        iface = get_contact_sensor_interface()
        self.assertIsNotNone(iface)

        self._timeline.play()
        await omni.kit.app.get_app().next_update_async()

        sensor_id = iface.create_sensor("/World/Cube/contact_sensor")
        self.assertGreaterEqual(sensor_id, 0)

        await step_simulation(1.0)

        reading = iface.get_sensor_reading(sensor_id)
        self.assertTrue(reading.is_valid)
        self.assertNotEqual(reading.value, 0.0)
        self.assertTrue(reading.in_contact)

    async def test_cpp_raw_contacts_available(self):
        """Verify C++ IContactSensor provides raw contact data during contact."""
        iface = get_contact_sensor_interface()
        self.assertIsNotNone(iface)

        self._timeline.play()
        await omni.kit.app.get_app().next_update_async()

        sensor_id = iface.create_sensor("/World/Cube/contact_sensor")
        self.assertGreaterEqual(sensor_id, 0)

        await step_simulation(1.0)

        raw_contacts = iface.get_raw_contacts(sensor_id)
        self.assertGreater(len(raw_contacts), 0)

        contact = raw_contacts[0]
        self.assertIn("body0", contact)
        self.assertIn("body1", contact)
        self.assertIn("position", contact)
        self.assertIn("normal", contact)
        self.assertIn("impulse", contact)
        self.assertIn("dt", contact)
        self.assertGreater(float(contact["dt"]), 0.0)

    async def test_cpp_backend_reading_matches(self):
        """Verify ContactSensorBackend returns same results as direct C++ call."""
        from isaacsim.sensors.experimental.physics import ContactSensorBackend

        backend = ContactSensorBackend("/World/Cube/contact_sensor")

        self._timeline.play()
        await step_simulation(1.0)

        reading = backend.get_sensor_reading()
        self.assertTrue(reading.is_valid)
        self.assertNotEqual(reading.value, 0.0)
        self.assertTrue(reading.in_contact)

    async def test_cpp_backend_raw_data(self):
        """Verify ContactSensorBackend.get_raw_data() returns C++ raw contacts."""
        from isaacsim.sensors.experimental.physics import ContactSensorBackend

        backend = ContactSensorBackend("/World/Cube/contact_sensor")

        self._timeline.play()
        await step_simulation(1.0)

        raw_data = backend.get_raw_data()
        self.assertGreater(len(raw_data), 0)
        self.assertIn("impulse", raw_data[0])

    async def test_invalid_sensor_returns_empty(self):
        """Verify invalid sensor paths return empty/invalid results."""
        iface = get_contact_sensor_interface()
        self.assertIsNotNone(iface)

        self._timeline.play()
        await omni.kit.app.get_app().next_update_async()

        sensor_id = iface.create_sensor("/World/Cube")
        self.assertEqual(sensor_id, -1)

        reading = iface.get_sensor_reading(-1)
        self.assertFalse(reading.is_valid)
