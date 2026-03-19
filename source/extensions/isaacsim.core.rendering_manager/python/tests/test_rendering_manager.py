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

import asyncio

import carb
import isaacsim.core.experimental.utils.stage as stage_utils
import omni.kit.test
from isaacsim.core.rendering_manager import RenderingEvent, RenderingManager, ViewportManager

_SETTING_RATE_LIMIT_ENABLED = "/app/runLoops/main/rateLimitEnabled"


class TestRenderingManager(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        """Method called to prepare the test fixture"""
        super().setUp()
        # ---------------
        self._callback_call = [0, 0]
        self._callback_call_stack = []
        self._rate_limit_enabled = carb.settings.get_settings().get_as_bool(_SETTING_RATE_LIMIT_ENABLED)
        await stage_utils.create_new_stage_async()
        # ---------------

    async def tearDown(self):
        """Method called immediately after the test method has been called"""
        # ------------------
        carb.settings.get_settings().set_bool(_SETTING_RATE_LIMIT_ENABLED, self._rate_limit_enabled)
        stage_utils.close_stage()
        # ------------------
        super().tearDown()

    # --------------------------------------------------------------------

    def _callback(self, *args, **kwargs):
        self._callback_call[0] += 1
        self._callback_call_stack.append(self._callback_call[:])

    # --------------------------------------------------------------------

    async def test_render(self):
        for _ in range(10):
            await RenderingManager.render_async()

    async def test_dt(self):
        # get default vale
        for enabled in [False, True]:
            carb.settings.get_settings().set_bool(_SETTING_RATE_LIMIT_ENABLED, enabled)
            expected_dt = 1 / 120 if enabled else 1 / 60
            current_dt = RenderingManager.get_dt()
            self.assertAlmostEqual(
                expected_dt,
                current_dt,
                msg=f"Expected {1 / expected_dt} Hz. Got {1 / current_dt} Hz (rateLimitEnabled: {enabled})",
            )
        # set custom value
        for i, enabled in enumerate([False, True]):
            custom_dt = 1 / (99 + 10 * i)
            RenderingManager.set_dt(custom_dt)
            current_dt = RenderingManager.get_dt()
            self.assertAlmostEqual(
                custom_dt,
                current_dt,
                msg=f"Expected {1 / custom_dt} Hz. Got {1 / current_dt} Hz (rateLimitEnabled: {enabled})",
            )

    async def test_callback(self):
        def callback(*args, **kwargs):
            self._callback_call[1] += 1
            self._callback_call_stack.append(self._callback_call[:])

        status, frames = await ViewportManager.wait_for_viewport_async()
        self.assertTrue(status, f"Viewport not ready after {frames} frames")
        self.assertListEqual(self._callback_call, [0, 0])
        await asyncio.sleep(0.5)  # let previous (unregistered) events pass, since we are not waiting idly
        # test cases
        # - register local function callback and step rendering
        local_function_callback_id = RenderingManager.register_callback(RenderingEvent.NEW_FRAME, callback=callback)
        await RenderingManager.render_async()
        self.assertEqual(local_function_callback_id, 0)
        # - register class method callback and step rendering
        class_method_callback_id = RenderingManager.register_callback(RenderingEvent.NEW_FRAME, callback=self._callback)
        await RenderingManager.render_async()
        self.assertEqual(class_method_callback_id, 1)
        # - step rendering
        await RenderingManager.render_async()
        # - deregister local function callback
        RenderingManager.deregister_callback(local_function_callback_id)
        await RenderingManager.render_async()
        # - deregister class method callback
        RenderingManager.deregister_callback(class_method_callback_id)
        await RenderingManager.render_async()
        # - register a new class method callback and step rendering
        class_method_callback_id = RenderingManager.register_callback(RenderingEvent.NEW_FRAME, callback=self._callback)
        await RenderingManager.render_async()
        self.assertEqual(class_method_callback_id, 2)
        # - deregister all callbacks
        RenderingManager.deregister_all_callbacks()
        await RenderingManager.render_async()
        # - deregister the already deregistered callback and check for warning
        RenderingManager.deregister_callback(local_function_callback_id)
        RenderingManager.deregister_callback(class_method_callback_id)
        await RenderingManager.render_async()
        # check the callback call stack
        await asyncio.sleep(0.5)  # wait for triggered events to occur, since we are not waiting idly
        self.assertListEqual(self._callback_call_stack, [[0, 1], [0, 2], [1, 2], [1, 3], [2, 3], [3, 3], [4, 3]])
