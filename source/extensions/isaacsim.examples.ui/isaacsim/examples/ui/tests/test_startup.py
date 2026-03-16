# SPDX-FileCopyrightText: Copyright (c) 2018-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# NOTE:
#   omni.kit.test - std python's unittest module with additional wrapping to add suport for async/await tests
#   For most things refer to unittest docs: https://docs.python.org/3/library/unittest.html

"""Tests for the isaacsim.examples.ui extension startup and basic functionality."""


import omni.kit.test


# Having a test class dervived from omni.kit.test.AsyncTestCase declared on the root of module will make it auto-discoverable by omni.kit.test
class TestUITemplate(omni.kit.test.AsyncTestCase):
    """Test class for the isaacsim.examples.ui extension.

    This class provides automated testing functionality to verify the extension loads and operates correctly.
    It runs asynchronous tests to ensure the UI components function properly within the Isaac Sim environment.
    The test validates extension stability by running multiple update cycles and checking for errors.
    """

    # Before running each test
    async def setUp(self):
        """Sets up the test environment before each test case.

        Waits for the next application update to ensure proper initialization.
        """
        await omni.kit.app.get_app().next_update_async()
        pass

    # After running each test
    async def tearDown(self):
        """Cleans up the test environment after each test case.

        Waits for the next application update to ensure proper cleanup.
        """
        await omni.kit.app.get_app().next_update_async()
        pass

    # Run for 60 frames and make sure there were no errors loading
    async def test_template(self):
        """Tests the UI template by running for 60 frames.

        Verifies that no errors occur during loading and rendering over 60 application update cycles.
        """
        for frame in range(60):
            await omni.kit.app.get_app().next_update_async()
        pass
