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

from typing import Literal

import carb
import omni.replicator.core as rep
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.sensors.camera.camera import Camera


class SingleViewDepthSensor(Camera):
    _render_product_prim = None

    def initialize(self, physics_sim_view=None, attach_rgb_annotator=True) -> None:
        """Initialize the depth camera.

        Calls the parent class's initialize method, then retrieves the render product prim
        and applies the appropriate schema to enable depth sensing functionality.

        Supports the following annotators:
        - DepthSensorDistance
        - DepthSensorPointCloudPosition
        - DepthSensorPointCloudColor
        - DepthSensorImager
        """
        super().initialize(physics_sim_view=physics_sim_view, attach_rgb_annotator=attach_rgb_annotator)
        # Retrieve the render product prim and apply the appropriate schema to it
        self._render_product_prim = get_prim_at_path(self._render_product_path)
        self._render_product_prim.ApplyAPI("OmniSensorDepthSensorSingleViewAPI")

        settings = carb.settings.get_settings()
        settings.set("/exts/omni.usd.schema.render_settings/rtx/renderSettings/apiSchemas/autoApply", None)
        settings.set("/exts/omni.usd.schema.render_settings/rtx/camera/apiSchemas/autoApply", None)
        settings.set("/exts/omni.usd.schema.render_settings/rtx/renderProduct/apiSchemas/autoApply", None)

        self.set_enabled(enabled=True)

    def set_baseline_mm(self, baseline_mm: float = 55) -> None:
        """Set the baseline distance in millimeters."""
        if self._render_product_prim:
            self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:baselineMM").Set(baseline_mm)

    def get_baseline_mm(self) -> float:
        """Get the baseline distance in millimeters."""
        if self._render_product_prim and self._render_product_prim.HasAttribute("omni:rtx:post:depthSensor:baselineMM"):
            return self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:baselineMM").Get()
        return None

    def set_confidence_threshold(self, confidence_threshold: float = 0.95) -> None:
        """Set the confidence threshold for the depth sensor."""
        if self._render_product_prim:
            self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:confidenceThreshold").Set(
                confidence_threshold
            )

    def get_confidence_threshold(self) -> float:
        """Get the confidence threshold for the depth sensor."""
        if self._render_product_prim and self._render_product_prim.HasAttribute(
            "omni:rtx:post:depthSensor:confidenceThreshold"
        ):
            return self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:confidenceThreshold").Get()
        return None

    def set_enabled(self, enabled: bool = False) -> None:
        """Enable or disable the depth sensor."""
        if self._render_product_prim:
            self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:enabled").Set(enabled)

    def get_enabled(self) -> bool:
        """Get whether the depth sensor is enabled."""
        if self._render_product_prim and self._render_product_prim.HasAttribute("omni:rtx:post:depthSensor:enabled"):
            return self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:enabled").Get()
        return None

    def set_focal_length_pixel(self, focal_length_pixel: float = 897) -> None:
        """Set the focal length in pixels."""
        if self._render_product_prim:
            self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:focalLengthPixel").Set(focal_length_pixel)

    def get_focal_length_pixel(self) -> float:
        """Get the focal length in pixels."""
        if self._render_product_prim and self._render_product_prim.HasAttribute(
            "omni:rtx:post:depthSensor:focalLengthPixel"
        ):
            return self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:focalLengthPixel").Get()
        return None

    def set_max_disparity_pixel(self, max_disparity_pixel: float = 110) -> None:
        """Set the maximum disparity in pixels."""
        if self._render_product_prim:
            self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:maxDisparityPixel").Set(
                max_disparity_pixel
            )

    def get_max_disparity_pixel(self) -> float:
        """Get the maximum disparity in pixels."""
        if self._render_product_prim and self._render_product_prim.HasAttribute(
            "omni:rtx:post:depthSensor:maxDisparityPixel"
        ):
            return self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:maxDisparityPixel").Get()
        return None

    def set_max_distance(self, max_distance: float = 10000000) -> None:
        """Set the maximum distance for disparity generation."""
        if self._render_product_prim:
            self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:maxDistance").Set(max_distance)

    def get_max_distance(self) -> float:
        """Get the maximum distance for disparity generation."""
        if self._render_product_prim and self._render_product_prim.HasAttribute(
            "omni:rtx:post:depthSensor:maxDistance"
        ):
            return self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:maxDistance").Get()
        return None

    def set_min_distance(self, min_distance: float = 0.5) -> None:
        """Set the minimum distance for disparity generation."""
        if self._render_product_prim:
            self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:minDistance").Set(min_distance)

    def get_min_distance(self) -> float:
        """Get the minimum distance for disparity generation."""
        if self._render_product_prim and self._render_product_prim.HasAttribute(
            "omni:rtx:post:depthSensor:minDistance"
        ):
            return self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:minDistance").Get()
        return None

    def set_noise_downscale_factor_pixel(self, noise_downscale_factor_pixel: float = 1) -> None:
        """Set the noise downscale factor in pixels."""
        if self._render_product_prim:
            self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:noiseDownscaleFactorPixel").Set(
                noise_downscale_factor_pixel
            )

    def get_noise_downscale_factor_pixel(self) -> float:
        """Get the noise downscale factor in pixels."""
        if self._render_product_prim and self._render_product_prim.HasAttribute(
            "omni:rtx:post:depthSensor:noiseDownscaleFactorPixel"
        ):
            return self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:noiseDownscaleFactorPixel").Get()
        return None

    def set_noise_mean(self, noise_mean: float = 0.25) -> None:
        """Set the noise mean."""
        if self._render_product_prim:
            self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:noiseMean").Set(noise_mean)

    def get_noise_mean(self) -> float:
        """Get the noise mean."""
        if self._render_product_prim and self._render_product_prim.HasAttribute("omni:rtx:post:depthSensor:noiseMean"):
            return self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:noiseMean").Get()
        return None

    def set_noise_sigma(self, noise_sigma: float = 0.25) -> None:
        """Set the noise sigma."""
        if self._render_product_prim:
            self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:noiseSigma").Set(noise_sigma)

    def get_noise_sigma(self) -> float:
        """Get the noise sigma."""
        if self._render_product_prim and self._render_product_prim.HasAttribute("omni:rtx:post:depthSensor:noiseSigma"):
            return self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:noiseSigma").Get()
        return None

    def set_outlier_removal_enabled(self, outlier_removal_enabled: int = 3) -> None:
        """Set the outlier removal enabled attribute. Samples separated by this range (in pixels) will be removed."""
        if self._render_product_prim:
            self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:outlierRemovalEnabled").Set(
                outlier_removal_enabled
            )

    def get_outlier_removal_enabled(self) -> int:
        """Get the outlier removal enabled attribute. Samples separated by this range (in pixels) will be removed."""
        if self._render_product_prim and self._render_product_prim.HasAttribute(
            "omni:rtx:post:depthSensor:outlierRemovalEnabled"
        ):
            return self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:outlierRemovalEnabled").Get()
        return None

    def set_rgb_depth_output_mode(self, rgb_depth_output_mode: int = 0) -> None:
        """Set the RGB depth output mode."""
        if self._render_product_prim:
            self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:rgbDepthOutputMode").Set(
                rgb_depth_output_mode
            )

    def get_rgb_depth_output_mode(self) -> int:
        """Get the RGB depth output mode."""
        if self._render_product_prim and self._render_product_prim.HasAttribute(
            "omni:rtx:post:depthSensor:rgbDepthOutputMode"
        ):
            return self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:rgbDepthOutputMode").Get()
        return None

    def set_sensor_size_pixel(self, sensor_size_pixel: int = 1280) -> None:
        """Set the sensor size in pixels."""
        if self._render_product_prim:
            self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:sensorSizePixel").Set(sensor_size_pixel)

    def get_sensor_size_pixel(self) -> int:
        """Get the sensor size in pixels."""
        if self._render_product_prim and self._render_product_prim.HasAttribute(
            "omni:rtx:post:depthSensor:sensorSizePixel"
        ):
            return self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:sensorSizePixel").Get()
        return None

    def set_show_distance(self, show_distance: bool = False) -> None:
        """Set whether to show the distance."""
        if self._render_product_prim:
            self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:showDistance").Set(show_distance)

    def get_show_distance(self) -> bool:
        """Get whether to show the distance."""
        if self._render_product_prim and self._render_product_prim.HasAttribute(
            "omni:rtx:post:depthSensor:showDistance"
        ):
            return self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:showDistance").Get()
        return None
