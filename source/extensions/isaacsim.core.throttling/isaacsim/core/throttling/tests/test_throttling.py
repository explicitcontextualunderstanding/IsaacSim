# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import carb
import carb.settings
import omni.ext
import omni.kit.test


class TestIsaacThrottling(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        self._timeline = omni.timeline.get_timeline_interface()
        self._timeline.set_start_time(0)
        self._timeline.set_end_time(1)
        pass

    async def tearDown(self):
        pass

    async def test_on_stop_play_callback(self):
        self._settings = carb.settings.get_settings()
        self._settings.set("/rtx/ecoMode/enabled", True)
        self._timeline.play()
        await omni.kit.app.get_app().next_update_async()
        self.assertEqual(self._settings.get("/rtx/ecoMode/enabled"), False)
        self._timeline.stop()
        await omni.kit.app.get_app().next_update_async()
        self.assertEqual(self._settings.get("/rtx/ecoMode/enabled"), True)
        self._timeline.play()
        await omni.kit.app.get_app().next_update_async()
        self.assertEqual(self._settings.get("/rtx/ecoMode/enabled"), False)
        self._timeline.stop()
        await omni.kit.app.get_app().next_update_async()
        self.assertEqual(self._settings.get("/rtx/ecoMode/enabled"), True)
        pass
