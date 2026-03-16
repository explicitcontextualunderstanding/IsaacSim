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

"""Tests for Robot Poser UI extension startup and UIBuilder construction."""

import omni.kit.test
from isaacsim.core.utils.stage import create_new_stage_async, update_stage_async


class TestRobotPoserStartup(omni.kit.test.AsyncTestCase):
    """Async tests for Robot Poser UI extension."""

    async def setUp(self) -> None:
        """Create a fresh USD stage for each test."""
        await create_new_stage_async()
        await update_stage_async()

    async def test_import_and_construct(self):
        """Verify the package imports and the UIBuilder can be constructed."""
        from isaacsim.robot.poser.ui.ui.ui_builder import UIBuilder

        builder = UIBuilder()
        self.assertIsNotNone(builder)
