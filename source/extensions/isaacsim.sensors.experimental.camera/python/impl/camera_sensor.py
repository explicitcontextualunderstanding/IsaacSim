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

from typing import Any, Literal, get_args

import carb
import isaacsim.core.experimental.utils.prim as prim_utils
import omni.replicator.core as rep
import warp as wp
from isaacsim.core.experimental.objects import Camera
from pxr import UsdRender

from ._common import ANNOTATOR_SPEC

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
    "rgb",
    "rgba",
    "semantic_segmentation",
]


class CameraSensor:
    """High level class for creating/wrapping and operating single camera sensor.

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
        >>> from isaacsim.sensors.experimental.camera import CameraSensor
        >>>
        >>> # given a USD stage with the Camera prim: /World/prim_0
        >>> resolution = (240, 320)  # following OpenCV/NumPy convention `(height, width)`
        >>> camera_sensor = CameraSensor(
        ...     "/World/prim_0",
        ...     resolution=resolution,
        ...     annotators=["rgb", "distance_to_image_plane"],
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
        self._resolution = resolution
        self._hydra_texture = None
        self._annotators = {}
        if not hasattr(self, "_annotators_spec"):
            self._annotators_spec = {annotator: ANNOTATOR_SPEC[annotator] for annotator in get_args(ANNOTATOR)}
        # check for supported annotators
        self._validate_annotators(annotators)
        # get or create camera object
        self._camera = path if isinstance(path, Camera) else Camera(path)
        if len(self._camera) > 1:
            raise ValueError(
                "The sensor only supports one camera prim, ",
                f"but the provided argument refers to {len(self._camera)} camera prims: {self._camera.paths}",
            )
        self._camera.enforce_square_pixels(self._resolution, modes="horizontal")
        # initialize instance from arguments
        self._initialize_sensor(annotators)

    def __del__(self) -> None:
        """Clean up instance."""
        self._invalidate_sensor()

    """
    Properties.
    """

    @property
    def annotators(self) -> list[str]:
        """Annotators.

        Returns:
            Sorted list of registered annotators.

        Example:

        .. code-block:: python

            >>> camera_sensor.annotators
            ['distance_to_image_plane', 'rgb']
        """
        return sorted(list(self._annotators.keys()))

    @property
    def camera(self) -> Camera:
        """Camera object encapsulated by the sensor.

        Returns:
            Camera object encapsulated by the sensor.

        Example:

        .. code-block:: python

            >>> camera_sensor.camera
            <isaacsim.core.experimental.objects.impl.camera.Camera object at 0x...>
        """
        return self._camera

    @property
    def resolution(self) -> tuple[int, int]:
        """Resolution of the sensor.

        Returns:
            Resolution of sensor frames (following OpenCV/NumPy convention: ``(height, width)``).

        Example:

        .. code-block:: python

            >>> camera_sensor.resolution
            (240, 320)
        """
        return self._resolution

    @property
    def render_product(self) -> UsdRender.Product:
        """Render product.

        Returns:
            Render product of the camera sensor.

        Example:

        .. code-block:: python

            >>> camera_sensor.render_product
            UsdRender.Product(Usd.Prim(</Render/OmniverseKit/HydraTextures/camera_sensor_...>))
        """
        prim = prim_utils.get_prim_at_path(self._hydra_texture.path)
        if prim.IsValid() and prim.IsA(UsdRender.Product):
            return UsdRender.Product(prim)
        raise RuntimeError(f"Invalid render product at path '{self._hydra_texture.path}'")

    """
    Methods.
    """

    def attach_annotators(self, annotators: str | list[str]) -> None:
        """Attach annotators to the sensor.

        Args:
            annotators: Annotator/sensor types to attach.

        Raises:
            ValueError: If the specified annotator is not supported.

        Example:

        .. code-block:: python

            >>> camera_sensor.annotators
            ['distance_to_image_plane', 'rgb']
            >>> camera_sensor.attach_annotators("pointcloud")
            >>> camera_sensor.annotators
            ['distance_to_image_plane', 'pointcloud', 'rgb']
        """
        annotators = [annotators] if isinstance(annotators, str) else annotators
        self._validate_annotators(annotators)
        # define annotator instances
        for annotator in annotators:
            spec = self._get_annotator_spec(annotator)
            device = "cuda"
            if annotator in ["bounding_box_2d_tight", "bounding_box_2d_loose", "bounding_box_3d"]:
                device = "cpu"
            self._annotators[annotator] = rep.AnnotatorRegistry.get_annotator(
                spec["name"], device=device, do_array_copy=False
            )
        # attach annotator instances to the hydra texture
        for annotator in annotators:
            self._annotators[annotator].attach(self._hydra_texture.path)

    def detach_annotators(self, annotators: str | list[str]) -> None:
        """Detach annotators from the sensor.

        Args:
            annotators: Annotator/sensor types to detach. If the annotator is not attached,
                or it has already been detached, a warning is logged and the method does nothing.

        Raises:
            ValueError: If the specified annotator is not supported.

        Example:

        .. code-block:: python

            >>> camera_sensor.annotators
            ['distance_to_image_plane', 'pointcloud', 'rgb']
            >>> camera_sensor.detach_annotators(["distance_to_image_plane", "pointcloud"])
            >>> camera_sensor.annotators
            ['rgb']
        """
        annotators = [annotators] if isinstance(annotators, str) else annotators
        self._validate_annotators(annotators)
        # detach annotator instances from the hydra texture
        for annotator in annotators:
            if annotator not in self._annotators:
                carb.log_warn(f"Unable to detach annotator '{annotator}'. It might have been already detached")
                continue
            self._annotators[annotator].detach([self._hydra_texture.path])
            del self._annotators[annotator]

    def get_data(self, annotator: str, *, out: wp.array | None = None) -> tuple[wp.array | None, dict[str, Any]]:
        """Fetch the specified annotator/sensor data for the camera.

        Args:
            annotator: Annotator/sensor type from which fetch the data.
            out: Pre-allocated array to fill with the fetched data.

        Returns:
            Two-elements tuple. 1) Array containing the fetched data. If ``out`` is defined, such instance is returned
            filled with the data. If no data is available at the moment of calling the method, ``None`` is returned.
            2) Dictionary containing additional information according to the requested annotator/sensor.

        Raises:
            ValueError: If the specified annotator is not supported.
            ValueError: If the specified annotator is not configured.

        Example:

        .. code-block:: python

            >>> data, info = camera_sensor.get_data("rgb")  # doctest: +NO_CHECK
            >>> data.shape  # doctest: +SKIP
            (240, 320, 3)
            >>> info
            {}
        """
        self._validate_annotators(annotator)
        if annotator not in self._annotators:
            raise ValueError(f"The annotator '{annotator}' was not configured. Enable it when instantiating the class")
        # fetch data from annotator
        data = self._annotators[annotator].get_data(device=str(out.device) if out is not None else "cuda")
        if isinstance(data, dict):
            info = data["info"]
            data = data["data"]
        else:
            info = {}
        # - check if there is no data available
        if data is None or not data.shape[0]:
            return None, {}
        if annotator in [
            "bounding_box_2d_tight",
            "bounding_box_2d_loose",
            "bounding_box_3d",
            "pointcloud",
        ]:
            info["resolution"] = self._resolution
            return data, info
        # process data
        spec = self._get_annotator_spec(annotator)
        input_channels = spec["channels"]
        output_channels = spec.get("output_channels", input_channels)
        data = data.reshape((*self._resolution, input_channels))
        if out is None:
            out = data[:, :, :output_channels] if "output_channels" in spec else data
        else:
            wp.copy(out, data[:, :, :output_channels] if "output_channels" in spec else data)
        return out, info

    """
    Internal methods.
    """

    def _invalidate_sensor(self):
        """Invalidate sensor by detaching annotators and destroying the hydra texture."""
        # detach annotators and destroy the hydra texture
        if self._hydra_texture is not None:
            self.detach_annotators(list(self._annotators.keys()))
            self._hydra_texture.destroy()
        # reset properties
        self._annotators = {}
        self._hydra_texture = None

    def _initialize_sensor(self, annotators: str | list[str]):
        """Initialize sensor by creating the hydra texture and attaching annotators."""
        # create the hydra texture
        self._hydra_texture = rep.create.render_product(
            camera=self._camera.paths[0],
            resolution=(self._resolution[1], self._resolution[0]),  # (width, height)
            name=f"camera_sensor_{hash(self)}",
        )
        # attach annotators
        self.attach_annotators(annotators)

    def _get_annotator_spec(self, annotator: str) -> dict[str, Any]:
        """Get the specification of the given annotator."""
        try:
            return self._annotators_spec[annotator]
        except KeyError:
            raise ValueError(
                f"Unsupported annotator '{annotator}'. Supported annotator are {list(self._annotators_spec.keys())}"
            )

    def _validate_annotators(self, annotators: str | list[str]) -> None:
        """Validate the given annotators."""
        annotators = [annotators] if isinstance(annotators, str) else annotators
        for annotator in annotators:
            if annotator not in self._annotators_spec:
                raise ValueError(
                    f"Unsupported annotator '{annotator}'. Supported annotator are {list(self._annotators_spec.keys())}"
                )
