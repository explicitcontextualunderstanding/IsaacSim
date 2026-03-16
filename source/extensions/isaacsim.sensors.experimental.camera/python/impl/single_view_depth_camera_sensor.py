# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from __future__ import annotations

from typing import Literal, get_args

import carb
import isaacsim.core.experimental.utils.prim as prim_utils
from isaacsim.core.experimental.objects import Camera

from ._common import ANNOTATOR_SPEC
from .camera_sensor import CameraSensor

ANNOTATOR = Literal[
    "bounding_box_2d_loose",
    "bounding_box_2d_tight",
    "bounding_box_3d",
    "distance_to_camera",
    "distance_to_image_plane",
    "instance_id_segmentation",
    "instance_segmentation",
    "motion_vectors",
    "normals",
    "pointcloud",
    "semantic_segmentation",
    # single view depth sensor annotators
    "depth_sensor_distance",
    "depth_sensor_imager",
    "depth_sensor_point_cloud_color",
    "depth_sensor_point_cloud_position",
]


class SingleViewDepthCameraSensor(CameraSensor):
    """High level class for creating/wrapping and operating single view depth camera sensor.

    The sensor is modeled using a single camera view to simulate a stereo camera pair and compute disparity and depth.
    The sensor is implemented as a post-process operation in the renderer, where the `OmniSensorDepthSensorSingleViewAPI`
    schema is applied to the USD render product prim rather than to the camera prim.

    Args:
        path: ``Camera`` object or single path to existing or non-existing (one of both) USD Camera prim.
            Can include regular expression for matching a prim.
        resolution: Resolution of the sensor (following OpenCV/NumPy convention: ``(height, width)``).
        annotators: Annotator/sensor types to configure.

    Raises:
        ValueError: If no prim is found matching the specified path.
        ValueError: If the input argument refers to more than one camera prim.
        ValueError: If an unsupported annotator type is specified.

    Example:

    .. code-block:: python

        >>> import isaacsim.core.experimental.utils.app as app_utils
        >>> from isaacsim.sensors.experimental.camera import SingleViewDepthCameraSensor
        >>>
        >>> # given a USD stage with the Camera prim: /World/prim_0
        >>> resolution = (240, 320)  # following OpenCV/NumPy convention `(height, width)`
        >>> camera_sensor = SingleViewDepthCameraSensor(
        ...     "/World/prim_0",
        ...     resolution=resolution,
        ...     annotators="depth_sensor_distance",
        ... )  # doctest: +NO_CHECK
        >>>
        >>> # play the simulation so the sensor can fetch data
        >>> app_utils.play(commit=True)
    """

    def __init__(
        self,
        path: str | Camera,
        *,
        # CameraSensor
        resolution: tuple[int, int],
        annotators: ANNOTATOR | list[ANNOTATOR],
    ):
        # define properties
        self._annotators_spec = {annotator: ANNOTATOR_SPEC[annotator] for annotator in get_args(ANNOTATOR)}
        # initialize base class
        super().__init__(path, resolution=resolution, annotators=annotators)
        # initialize instance
        self._render_product_prim = prim_utils.get_prim_at_path(self.render_product)
        self._render_product_prim.ApplyAPI("OmniSensorDepthSensorSingleViewAPI")
        # - update render settings
        settings = carb.settings.get_settings()
        settings.set("/exts/omni.usd.schema.render_settings/rtx/renderSettings/apiSchemas/autoApply", None)
        settings.set("/exts/omni.usd.schema.render_settings/rtx/camera/apiSchemas/autoApply", None)
        settings.set("/exts/omni.usd.schema.render_settings/rtx/renderProduct/apiSchemas/autoApply", None)

    """
    Methods.
    """

    def set_sensor_baseline(self, baseline: float) -> None:
        """Set the distance between the simulated depth camera sensor, in millimeters.

        Larger positive/negative values will increase the unknown black/hole regions around objects
        where the camera sensor cannot see.

        Args:
            baseline: Sensor baseline in millimeters.

        Example:

        .. code-block:: python

            >>> camera_sensor.set_sensor_baseline(50.0)
        """
        self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:baselineMM").Set(baseline)

    def get_sensor_baseline(self) -> float:
        """Get the distance between the simulated depth camera sensor, in millimeters.

        Larger positive/negative values will increase the unknown black/hole regions around objects
        where the depth camera sensor cannot see.

        Returns:
            The distance between the simulated depth camera sensor, in millimeters.

        Example:

        .. code-block:: python

            >>> camera_sensor.get_sensor_baseline()
            55.0
        """
        return self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:baselineMM").Get()

    def set_sensor_disparity_confidence(self, confidence_threshold: float) -> None:
        """Set the confidence threshold for the depth sensor.

        Control how likely a depth sample is considered valid. Higher values make depth values vary wider across
        the quantization (noise mean) range.

        Args:
            confidence_threshold: Confidence threshold.

        Example:

        .. code-block:: python

            >>> camera_sensor.set_sensor_disparity_confidence(0.75)
        """
        self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:confidenceThreshold").Set(
            confidence_threshold
        )

    def get_sensor_disparity_confidence(self) -> float:
        """Get the confidence threshold for the depth sensor.

        Control how likely a depth sample is considered valid. Higher values make depth values vary wider across
        the quantization (noise mean) range.

        Returns:
            The confidence threshold for the depth sensor.

        Example:

        .. code-block:: python

            >>> camera_sensor.get_sensor_disparity_confidence()
             0.6999...
        """
        return self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:confidenceThreshold").Get()

    def set_sensor_maximum_disparity(self, maximum_disparity: float) -> None:
        """Set the maximum number of disparity pixels for the depth sensor.

        Higher values allow the sensor to resolve closer (more disparate) objects.
        Lower values reduce the depth sensing range.

        Args:
            maximum_disparity: Maximum disparity.

        Example:

        .. code-block:: python

            >>> camera_sensor.set_sensor_maximum_disparity(120.0)
        """
        self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:maxDisparityPixel").Set(maximum_disparity)

    def get_sensor_maximum_disparity(self) -> float:
        """Get the maximum number of disparity pixels for the depth sensor.

        Higher values allow the sensor to resolve closer (more disparate) objects.
        Lower values reduce the depth sensing range.

        Returns:
            The maximum number of disparity pixels for the depth sensor.

        Example:

        .. code-block:: python

            >>> camera_sensor.get_sensor_maximum_disparity()
            110.0
        """
        return self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:maxDisparityPixel").Get()

    def set_enabled_post_processing(self, enabled: bool) -> None:
        """Enable or disable the post-process operation for depth sensing in the renderer.

        Args:
            enabled: Boolean flag to enable/disable the depth sensor post-process.

        Example:

        .. code-block:: python

            >>> camera_sensor.set_enabled_post_processing(True)
        """
        self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:enabled").Set(enabled)

    def get_enabled_post_processing(self) -> bool:
        """Get the enabled state of the post-process operation for depth sensing in the renderer of the prims.

        Returns:
            Boolean flag indicating if the depth sensor post-process is enabled.

        Example:

        .. code-block:: python

            >>> camera_sensor.get_enabled_post_processing()
            True
        """
        return self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:enabled").Get()

    def set_sensor_focal_length(self, focal_length: float) -> None:
        """Set the simulated focal length of the depth sensor, in pixels.

        Combined with the sensor size, this sets the field of view for the disparity calculation.
        Since the actual FOV is controlled on the camera prim, this only adjusts the amount of
        left/right disparity. Lower focal length decreases disparity.

        Args:
            focal_length: Sensor focal length in pixels.

        Example:

        .. code-block:: python

            >>> camera_sensor.set_sensor_focal_length(800.0)
        """
        self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:focalLengthPixel").Set(focal_length)

    def get_sensor_focal_length(self) -> float:
        """Get the simulated focal length of the depth sensor, in pixels.

        Combined with the sensor size, this sets the field of view for the disparity calculation.
        Since the actual FOV is controlled on the camera prim, this only adjusts the amount of
        left/right disparity. Lower focal length decreases disparity.

        Returns:
            The sensor focal length, in pixels.

        Example:

        .. code-block:: python

            >>> camera_sensor.get_sensor_focal_length()
            897.0
        """
        return self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:focalLengthPixel").Get()

    def set_sensor_distance_cutoffs(self, minimum_distance: float = None, maximum_distance: float = None) -> None:
        """Set the cutoff range (minimum and maximum distance) of the depth sensor.

        Args:
            minimum_distance: Minimum cutoff distance.
            maximum_distance: Maximum cutoff distance.

        Raises:
            ValueError: If both ``minimum_distance`` and ``maximum_distance`` are not defined.

        Example:

        .. code-block:: python

            >>> camera_sensor.set_sensor_distance_cutoffs(minimum_distance=0.1, maximum_distance=1000000.0)
        """
        assert (
            minimum_distance is not None or maximum_distance is not None
        ), "Both 'minimum_distance' and 'maximum_distance' are not defined. Define at least one of them"
        if minimum_distance is not None:
            self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:minDistance").Set(minimum_distance)
        if maximum_distance is not None:
            self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:maxDistance").Set(maximum_distance)

    def get_sensor_distance_cutoffs(self) -> tuple[float, float]:
        """Get the cutoff range (minimum and maximum distance) of the depth sensor.

        Returns:
            The minimum cutoff distance.
            The maximum cutoff distance.

        Example:

        .. code-block:: python

            >>> camera_sensor.get_sensor_distance_cutoffs()
            (0.5, 10000000.0)
        """
        return (
            self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:minDistance").Get(),
            self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:maxDistance").Get(),
        )

    def set_sensor_disparity_noise_downscale(self, downscale: float) -> None:
        """Set the coarseness of the disparity noise, in pixels, of the depth sensor.

        Higher values reduce the spatial resolution of the noise.

        Args:
            downscale: Disparity noise downscale factor in pixels.

        Example:

        .. code-block:: python

            >>> camera_sensor.set_sensor_disparity_noise_downscale(1.5)
        """
        self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:noiseDownscaleFactorPixel").Set(downscale)

    def get_sensor_disparity_noise_downscale(self) -> float:
        """Get the coarseness of the disparity noise, in pixels, of the depth sensor.

        Higher values reduce the spatial resolution of the noise.

        Returns:
            The disparity noise downscale factor, in pixels.

        Example:

        .. code-block:: python

            >>> camera_sensor.get_sensor_disparity_noise_downscale()
            1.0
        """
        return self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:noiseDownscaleFactorPixel").Get()

    def set_sensor_noise_parameters(self, noise_mean: float = None, noise_sigma: float = None) -> None:
        """Set the quantization factor (mean and sigma) for the disparity noise, in pixels, of the depth sensor.

        Higher mean values reduce depth resolution. Higher sigma values make depth values vary wider
        across the quantization (noise mean) range.

        Args:
            noise_mean: Disparity noise mean value in pixels.
            noise_sigma: Disparity noise sigma value in pixels.

        Raises:
            AssertionError: If neither ``noise_mean`` nor ``noise_sigma`` are specified.

        Example:

        .. code-block:: python

            >>> camera_sensor.set_sensor_noise_parameters(noise_mean=0.5, noise_sigma=0.1)
        """
        assert (
            noise_mean is not None or noise_sigma is not None
        ), "Both 'noise_mean' and 'noise_sigma' are not defined. Define at least one of them"
        if noise_mean is not None:
            self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:noiseMean").Set(noise_mean)
        if noise_sigma is not None:
            self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:noiseSigma").Set(noise_sigma)

    def get_sensor_noise_parameters(self) -> tuple[float, float]:
        """Get the quantization factor (mean and sigma) for the disparity noise, in pixels, of the depth sensor.

        Returns:
            Two-elements tuple. 1) The disparity noise mean value, in pixels.
            2) The disparity noise sigma value, in pixels.

        Example:

        .. code-block:: python

            >>> camera_sensor.get_sensor_noise_parameters()
            (0.25, 0.25)
        """
        return (
            self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:noiseMean").Get(),
            self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:noiseSigma").Get(),
        )

    def set_enabled_outlier_removal(self, enabled: bool) -> None:
        """Enable or disable the outlier removal filter of the depth sensor.

        Filter out single pixel samples caused by antialiasing jitter and reprojection resolution.

        Args:
            enabled: Boolean flag to enable/disable outlier removal.

        Example:

        .. code-block:: python

            >>> camera_sensor.set_enabled_outlier_removal(True)
        """
        self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:outlierRemovalEnabled").Set(enabled)

    def get_enabled_outlier_removal(self) -> bool:
        """Get the enabled state of the outlier removal filter of the depth sensor.

        Filter out single pixel samples caused by antialiasing jitter and reprojection resolution.

        Returns:
            Boolean flag indicating if the outlier removal filter is enabled.

        Example:

        .. code-block:: python

            >>> camera_sensor.get_enabled_outlier_removal()
            True
        """
        return self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:outlierRemovalEnabled").Get()

    def set_sensor_output_mode(self, mode: int) -> None:
        """Set the output mode to override the LDRColor buffer with a debug visualization of the depth sensor.

        Supported modes:

        * ``0``: Pass through LDRColor.
        * ``1``: Repeated 1 meter grayscale gradient.
        * ``2``: Grayscale gradient over min/max distance.
        * ``3``: Rainbow gradient over min/max distance.
        * ``4``: Input Depth values in grayscale.
        * ``5``: Reprojected depth with confidence culling applied.
        * ``6``: Confidence Map with Disparity.
        * ``7``: Disparity values in grayscale.

        Args:
            mode: Output mode.

        Example:

        .. code-block:: python

            >>> camera_sensor.set_sensor_output_mode(0)  # pass through LDRColor
        """
        self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:rgbDepthOutputMode").Set(mode)

    def get_sensor_output_mode(self) -> int:
        """Get the output mode used to override the LDRColor buffer with a debug visualization of the depth sensor.

        Supported modes:

        * ``0``: Pass through LDRColor.
        * ``1``: Repeated 1 meter grayscale gradient.
        * ``2``: Grayscale gradient over min/max distance.
        * ``3``: Rainbow gradient over min/max distance.
        * ``4``: Input Depth values in grayscale.
        * ``5``: Reprojected depth with confidence culling applied.
        * ``6``: Confidence Map with Disparity.
        * ``7``: Disparity values in grayscale.

        Returns:
            The output mode.

        Example:

        .. code-block:: python

            >>> camera_sensor.get_sensor_output_mode()
            0
        """
        return self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:rgbDepthOutputMode").Get()

    def set_sensor_size(self, size: float) -> None:
        """Set the size of the sensor, in pixels, of the depth sensor.

        Combined with focal length, this affects the amount of disparity. Higher values decrease disparity.

        Args:
            size: Sensor size, in pixels.

        Example:

        .. code-block:: python

            >>> camera_sensor.set_sensor_size(1920.0)
        """
        self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:sensorSizePixel").Set(size)

    def get_sensor_size(self) -> float:
        """Get the size of the sensor, in pixels, of the depth sensor.

        Combined with focal length, this affects the amount of disparity. Higher values decrease disparity.

        Returns:
            The sensor size, in pixels.

        Example:

        .. code-block:: python

            >>> camera_sensor.get_sensor_size()
            1280.0
        """
        return self._render_product_prim.GetAttribute("omni:rtx:post:depthSensor:sensorSizePixel").Get()
