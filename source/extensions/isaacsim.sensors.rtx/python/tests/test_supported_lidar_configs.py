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

"""Tests for validating all supported lidar configurations and variants in the isaacsim.sensors.rtx extension."""


import asyncio
import os
import tempfile
from pathlib import Path

import omni.kit.test
from isaacsim.core.utils.stage import create_new_stage_async, save_stage, update_stage_async
from isaacsim.sensors.rtx import SUPPORTED_LIDAR_CONFIGS
from isaacsim.sensors.rtx.sensor_checker import ModelInfo, SensorCheckerUtil
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf

from .common import create_sarcophagus


class TestSupportedLidarConfigs(omni.kit.test.AsyncTestCase):
    """Test class for validating all supported lidar configurations and variants in the isaacsim.sensors.rtx extension.

    This test class dynamically generates test methods for each supported lidar configuration and variant combination
    from SUPPORTED_LIDAR_CONFIGS. Each test method creates a lidar sensor prim using the IsaacSensorCreateRtxLidar
    command and validates its parameters using the SensorCheckerUtil with the LidarCore model.

    The class sets up a test environment with a new stage and a sarcophagus object for sensor validation. It uses
    the SensorCheckerUtil to verify that each lidar configuration produces valid sensor parameters that conform to
    the expected schema and model requirements.

    Test methods are automatically generated at module load time using the _create_lidar_parameters_test function,
    which creates individual test cases for each config-variant pair. Each generated test validates sensor creation,
    parameter validation, and ensures the sensor prim has the correct type and valid parameters.
    """

    async def setUp(self):
        """Sets up the test environment for lidar configuration testing.

        Creates a new USD stage, initializes a sarcophagus object with non-visual materials,
        sets up the timeline interface, and configures a sensor checker utility with a generic
        LidarCore model for parameter validation.
        """
        await create_new_stage_async()
        await update_stage_async()

        create_sarcophagus(enable_nonvisual_material=True)
        self._timeline = omni.timeline.get_timeline_interface()
        # Create a sensor checker
        self._checker = SensorCheckerUtil()

        # Initialize with LidarCore model
        model_info = ModelInfo()
        model_info.modelName = "LidarCore"
        model_info.modelVersion = "1.0"
        model_info.schemaVersion = "1.0"
        model_info.modelVendor = "NVIDIA"
        model_info.marketName = "GenericLidar"

        self._checker.init(model_info)

    async def tearDown(self):
        """Cleans up the test environment after lidar configuration testing.

        Removes the sensor checker, waits for any pending asset loading to complete,
        and updates the stage to ensure a clean state for subsequent tests.
        """
        del self._checker
        await omni.kit.app.get_app().next_update_async()
        while omni.usd.get_context().get_stage_loading_status()[2] > 0:
            print("tearDown, assets still loading, waiting to finish...")
            await asyncio.sleep(1.0)
        await update_stage_async()


# Iterate over all supported lidar configs and variants, creating a test for each as sensor prims
def _create_lidar_parameters_test(config_name, variant):
    """Creates a test function for validating lidar sensor parameters with specified configuration and variant.

    Generates an async test function that creates an RTX lidar sensor prim with the given configuration
    and variant, then validates the sensor parameters using the sensor checker utility.

    Args:
        config_name: Name of the lidar configuration to test.
        variant: Variant of the lidar configuration to test.

    Returns:
        An async test function that validates the lidar sensor parameters.
    """

    async def test_function(self):
        # Create sensor prim
        kwargs = {
            "path": "lidar",
            "parent": None,
            "translation": Gf.Vec3d(0.0, 0.0, 0.0),
            "orientation": Gf.Quatd(1.0, 0.0, 0.0, 0.0),
            "config": config_name,
            "variant": variant,
            "omni:sensor:Core:outputFrameOfReference": "WORLD",
        }

        _, self.sensor = omni.kit.commands.execute(f"IsaacSensorCreateRtxLidar", **kwargs)
        sensor_type = self.sensor.GetTypeName()
        self.assertEqual(
            sensor_type, "OmniLidar", f"Expected OmniLidar prim, got {sensor_type}. Was sensor prim created?"
        )

        error_string = self._checker.validateParams(self.sensor)
        self.assertIsNone(
            error_string,
            f"Sensor parameter validation failed for config {config_name} variant {variant}: {error_string}",
        )

        validated_params = self._checker.getValidatedParams()
        self.assertIsNotNone(validated_params, "Failed to retrieve validated parameters")
        self.assertGreater(
            validated_params.numParams,
            0,
            f"Expected validated parameters, got {validated_params.numParams} parameters",
        )

    return test_function


for config_path in SUPPORTED_LIDAR_CONFIGS:
    config_name = Path(config_path).stem
    for variant in SUPPORTED_LIDAR_CONFIGS[config_path] or [None]:
        test_name = f"test_lidar_parameters_{config_name}_{variant}"
        test_func = _create_lidar_parameters_test(config_name, variant)
        test_func.__name__ = test_name
        test_func.__doc__ = (
            f"Test supported lidar configs and variants, with config {config_name} and variant {variant}."
        )
        setattr(TestSupportedLidarConfigs, test_name, test_func)
