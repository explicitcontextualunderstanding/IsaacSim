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

"""Common utilities and constants for ROS 2 nodes extension."""

from typing import List

import numpy as np
import omni.replicator.core as rep
import omni.syntheticdata
import omni.usd
from isaacsim.core.nodes.scripts.utils import (
    register_annotator_from_node_with_telemetry,
    register_node_writer_with_telemetry,
)
from pxr import Usd

# Nodes extension constants
BRIDGE_NAME = "isaacsim.ros2.bridge"
BRIDGE_PREFIX = "ROS2"


class CompressedImageManager:
    """Manage per-camera H.264 compression annotators and writers.

    Each render product gets its own annotator instance (unique rendervar hash)
    so encoder pipelines and writers are fully independent across cameras.
    Annotator instances are cached so the hash stays stable across stop/play cycles.
    """

    _annotators: dict = {}
    #: Per-render-product annotator instances keyed by render product path.

    @classmethod
    def attach(cls, render_product_path: str) -> None:
        """Attach the H.264 encoder pipeline to a render product.

        Creates the annotator on first call for this render product, then attaches it.
        This activates the POST_RENDER encoder templates and the ON_DEMAND Ptr template.

        Args:
            render_product_path: Path to the render product.
        """

        stage = omni.usd.get_context().get_stage()
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            if render_product_path not in cls._annotators:
                cls._annotators[render_product_path] = rep.AnnotatorRegistry.get_annotator(
                    "LdrColor", init_params={"compression": "h264"}
                )
            cls._annotators[render_product_path].attach([render_product_path])

    @classmethod
    def detach(cls, render_product_path: str) -> None:
        """Detach the H.264 encoder pipeline from a specific render product.

        Only detaches from the specified render product — other cameras are not affected.

        Args:
            render_product_path: Path to the render product.
        """
        stage = omni.usd.get_context().get_stage()
        with Usd.EditContext(stage, stage.GetSessionLayer()):

            annotator = cls._annotators.get(render_product_path)
            if annotator is not None:
                try:
                    annotator.detach([render_product_path])
                except Exception:
                    pass

    @classmethod
    def get_writer(cls, render_product_path: str, use_system_time: bool = False) -> rep.Writer:
        """Get a compressed image writer for a specific render product.

        Registers the writer on first call (unique name per annotator hash).
        The writer's Ptr template hash matches the annotator's encoder pipeline.

        Args:
            render_product_path: Path to the render product.
            use_system_time: If True, use system time for timestamps.

        Returns:
            The replicator writer instance for this render product.
        """
        stage = omni.usd.get_context().get_stage()
        with Usd.EditContext(stage, stage.GetSessionLayer()):

            annotator = cls._annotators.get(render_product_path)
            if annotator is None:
                raise RuntimeError(
                    f"H.264 annotator not attached for render product '{render_product_path}'. "
                    "Call CompressedImageManager.attach() first."
                )

            rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar("Rgb")
            time_type = "SystemTime" if use_system_time else ""
            time_source = "systemTime" if use_system_time else "simulationTime"

            writer_name = f"{rv}{BRIDGE_PREFIX}{time_type}PublishCompressedImage_{annotator.template_name}"
            if writer_name not in rep.WriterRegistry.get_writers():
                register_node_writer_with_telemetry(
                    name=writer_name,
                    node_type_id=f"{BRIDGE_NAME}.{BRIDGE_PREFIX}PublishCompressedImage",
                    annotators=[
                        omni.syntheticdata.SyntheticData.NodeConnectionTemplate(
                            annotator.template_name,
                            attributes_mapping={
                                "outputs:dataPtr": "inputs:dataPtr",
                                "outputs:bufferSize": "inputs:bufferSize",
                            },
                        ),
                        omni.syntheticdata.SyntheticData.NodeConnectionTemplate(
                            f"IsaacRead{time_source[0].upper()}{time_source[1:]}",
                            attributes_mapping={f"outputs:{time_source}": "inputs:timeStamp"},
                        ),
                    ],
                    input_format="h264",
                    category=BRIDGE_NAME,
                )

            return rep.writers.get(writer_name)


def build_rtx_sensor_pointcloud_writer(
    metadata: List[str], enable_full_scan: bool = True, use_system_time: bool = False
) -> rep.Writer:
    """Build and register an RTX sensor point cloud writer with specified metadata.

    Dynamically creates and registers a custom annotator and writer for publishing RTX sensor
    point cloud data with the specified metadata fields. The annotator and writer are cached
    and reused if they already exist with the same configuration.

    Args:
        param metadata: List of metadata field names to include in the point cloud output.
            Valid options include "intensity", "timestamp", "emitterId", "channelId",
            "materialId", "tickId", "hitNormal", "velocity", "objectId", "echoId",
            "tickState", and "radialVelocityMS".
        param enable_full_scan: If True, enables full scan buffer mode. If False, enables
            per-frame output mode.
        param use_system_time: If True, uses system time for timestamps. If False, uses
            simulation time.

    Returns:
        The registered replicator writer instance configured for publishing RTX sensor
        point cloud data with the specified metadata and timing configuration.

    Example:

    .. code-block:: python

        >>> from isaacsim.ros2.nodes.python.impl.ros2_common import build_rtx_sensor_pointcloud_writer
        >>>
        >>> writer = build_rtx_sensor_pointcloud_writer(
        ...     metadata=["intensity", "objectId"],
        ...     enable_full_scan=True,
        ...     use_system_time=False,
        ... )
        >>>
        >>> writer.initialize(
        ...     frameId="lidar_frame",
        ...     topicName="point_cloud",
        ... )
    """

    # Dynamically name, build, and register a new Annotator based on the selected metadata if it doesn't exist yet
    annotator_name = "IsaacCreateRTXLidarScanBuffer"
    annotator_name += "PerFrame" if not enable_full_scan else ""
    annotator_name += "_".join(metadata)
    if annotator_name not in rep.AnnotatorRegistry.get_registered_annotators():
        init_params = {"enablePerFrameOutput": not enable_full_scan}
        for metadata_item in metadata:
            init_params[f"output{metadata_item[0].upper()}{metadata_item[1:]}"] = True
        register_annotator_from_node_with_telemetry(
            name=annotator_name,
            input_rendervars=[
                "GenericModelOutputPtr",
            ],
            node_type_id="isaacsim.sensors.rtx.IsaacCreateRTXLidarScanBuffer",
            init_params=init_params,
            output_data_type=np.float32,
            output_channels=3,
        )

    # Dynamically name, build, and register a new Writer based on the selected metadata if it doesn't exist yet
    time_type = "SystemTime" if use_system_time else ""
    time_node_type = "system" if use_system_time else "simulation"

    writer_name = "RtxLidar" + f"ROS2{time_type}PublishPointCloud"
    writer_name += "Buffer" if enable_full_scan else ""
    writer_name += "_".join(metadata)
    if writer_name not in rep.WriterRegistry.get_writers():
        register_node_writer_with_telemetry(
            name=writer_name,
            node_type_id="isaacsim.ros2.bridge.ROS2PublishPointCloud",
            annotators=[
                annotator_name,
                "PostProcessDispatchIsaacSimulationGate",
                omni.syntheticdata.SyntheticData.NodeConnectionTemplate(
                    f"IsaacRead{time_node_type.capitalize()}Time",
                    attributes_mapping={f"outputs:{time_node_type}Time": "inputs:timeStamp"},
                ),
            ],
            category="isaacsim.ros2.bridge",
        )

    writer = rep.writers.get(writer_name)
    return writer
