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

import asyncio
import math
import os
import shutil

import numpy as np
import omni.graph.core as og

# Import extension python module we are testing with absolute import path, as if we are external user (other extension)
import omni.kit.commands

# NOTE:
#   omni.kit.test - std python's unittest module with additional wrapping to add support for async/await tests
#   For most things refer to unittest docs: https://docs.python.org/3/library/unittest.html
import omni.kit.test
import omni.kit.usd
import omni.kit.viewport.utility
import omni.usd
import rclpy
import usdrt.Sdf
from isaacsim.core.api.objects import VisualCone, VisualCuboid, VisualCylinder, VisualSphere
from isaacsim.core.experimental.prims import RigidPrim, XformPrim
from isaacsim.core.experimental.utils import stage as stage_utils
from isaacsim.core.experimental.utils import transform as transform_utils
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.utils.semantics import add_labels
from isaacsim.core.utils.stage import open_stage_async
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path
from isaacsim.test.utils.image_comparison import compare_arrays_within_tolerances
from isaacsim.test.utils.image_io import read_image_as_array, save_rgb_image
from pxr import PhysxSchema, Sdf
from sensor_msgs.msg import Image

from .common import ROS2TestCase, get_qos_profile, ros2_image_to_buffer


def _camera_orientation_at_angle_deg(angle_deg: float):
    """Return quaternion (w,x,y,z) for camera at center looking at angle_deg in XY (0° = +X), up = world +Z.

    Camera local -Z is the view direction. Extrinsic ZYX Euler: Rz(angle-90) * Ry(0) * Rx(90)
    maps camera -Z to (cos(angle), sin(angle), 0) and camera +Y to world +Z.
    """
    quat = transform_utils.euler_angles_to_quaternion([angle_deg - 90.0, 0.0, 90.0], degrees=True, extrinsic=True)
    return quat.numpy().tolist()


def _view_angle_deg_from_quat_wxyz(quat_wxyz):
    """Angle in XY plane (degrees [0, 360)) that the camera is looking, from quat (w,x,y,z).

    Inverse of _camera_orientation_at_angle_deg: extract the extrinsic yaw and undo the -90° offset.
    """
    euler = transform_utils.quaternion_to_euler_angles(quat_wxyz, degrees=True, extrinsic=True)
    yaw = float(euler.numpy()[2])  # output order is [roll, pitch, yaw] = [X, Y, Z]
    return (yaw + 90.0 + 360.0) % 360.0


def _create_rgb_camera_graph(graph_path, camera_path, topic_name, width, height):
    """Create an OmniGraph that publishes RGB images from a camera via ROS2."""
    og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("CreateRenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                ("RGBPublish", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("CreateRenderProduct.inputs:cameraPrim", [usdrt.Sdf.Path(camera_path)]),
                ("CreateRenderProduct.inputs:height", height),
                ("CreateRenderProduct.inputs:width", width),
                ("RGBPublish.inputs:topicName", topic_name),
                ("RGBPublish.inputs:type", "rgb"),
                ("RGBPublish.inputs:resetSimulationTimeOnStop", True),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "CreateRenderProduct.inputs:execIn"),
                ("CreateRenderProduct.outputs:execOut", "RGBPublish.inputs:execIn"),
                ("CreateRenderProduct.outputs:renderProductPath", "RGBPublish.inputs:renderProductPath"),
            ],
        },
    )


def _match_buffered_images(image_buffer, sim_times, timestamp_tolerance, label=""):
    """Match buffered (timestamp, image) pairs to target sim_times by closest timestamp.

    Args:
        image_buffer: List of (timestamp, image_array) tuples.
        sim_times: Dict mapping target_angle -> sim_time to match against.
        timestamp_tolerance: Maximum allowed difference between image timestamp and sim_time.
        label: Optional prefix for log messages (e.g. "golden ").

    Returns:
        Dict mapping target_angle -> image_array for all matched targets.
    """
    matched = {}
    matched_ts = {}
    for target, target_sim_time in sim_times.items():
        best_match = None
        best_diff = float("inf")
        best_ts = None
        for ts, img in image_buffer:
            diff = abs(ts - target_sim_time)
            if diff < best_diff:
                best_diff = diff
                best_match = img
                best_ts = ts
        if best_diff <= timestamp_tolerance:
            matched[target] = best_match
            matched_ts[target] = best_ts
            print(f"Matched {label}{target}° - " f"best_diff={best_diff:.6f}s (tolerance={timestamp_tolerance:.6f}s)")
        else:
            print(
                f"WARNING: No image matched {label}{target}° "
                f"(sim_time={target_sim_time:.6f}s, best_diff={best_diff:.6f}s)"
            )
    return matched, matched_ts


class TestRos2Camera(ROS2TestCase):
    async def setUp(self):
        await super().setUp()

        # acquire the viewport window
        viewport_api = omni.kit.viewport.utility.get_active_viewport()
        # Set viewport resolution, changes will occur on next frame
        viewport_api.set_texture_resolution((1280, 720))
        await omni.kit.app.get_app().next_update_async()

    async def test_camera(self):
        scene_path = "/Isaac/Environments/Grid/default_environment.usd"
        await open_stage_async(self._assets_root_path + scene_path)

        cube_1 = VisualCuboid("/cube_1", position=[0, 0, 0], scale=[1.5, 1, 1])
        add_labels(cube_1.prim, labels=["Cube0"], instance_name="class")

        import rclpy
        import usdrt.Sdf

        try:
            og.Controller.edit(
                {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("RGBPublish", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                        ("DepthPublish", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                        ("DepthPclPublish", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                        ("InstancePublish", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                        ("SemanticPublish", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                        ("Bbox2dTightPublish", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                        ("Bbox2dLoosePublish", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                        ("Bbox3dPublish", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                        ("CreateRenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        ("CreateRenderProduct.inputs:cameraPrim", [usdrt.Sdf.Path("/OmniverseKit_Persp")]),
                        ("CreateRenderProduct.inputs:height", 600),
                        ("CreateRenderProduct.inputs:width", 800),
                        ("RGBPublish.inputs:topicName", "rgb"),
                        ("RGBPublish.inputs:type", "rgb"),
                        ("RGBPublish.inputs:resetSimulationTimeOnStop", True),
                        ("DepthPublish.inputs:topicName", "depth"),
                        ("DepthPublish.inputs:type", "depth"),
                        ("DepthPublish.inputs:resetSimulationTimeOnStop", True),
                        ("DepthPclPublish.inputs:topicName", "depth_pcl"),
                        ("DepthPclPublish.inputs:type", "depth_pcl"),
                        ("DepthPclPublish.inputs:resetSimulationTimeOnStop", True),
                        ("InstancePublish.inputs:topicName", "instance_segmentation"),
                        ("InstancePublish.inputs:type", "instance_segmentation"),
                        ("InstancePublish.inputs:resetSimulationTimeOnStop", True),
                        ("SemanticPublish.inputs:topicName", "semantic_segmentation"),
                        ("SemanticPublish.inputs:type", "semantic_segmentation"),
                        ("SemanticPublish.inputs:resetSimulationTimeOnStop", True),
                        ("Bbox2dTightPublish.inputs:topicName", "bbox_2d_tight"),
                        ("Bbox2dTightPublish.inputs:type", "bbox_2d_tight"),
                        ("Bbox2dTightPublish.inputs:resetSimulationTimeOnStop", True),
                        ("Bbox2dLoosePublish.inputs:topicName", "bbox_2d_loose"),
                        ("Bbox2dLoosePublish.inputs:type", "bbox_2d_loose"),
                        ("Bbox2dLoosePublish.inputs:resetSimulationTimeOnStop", True),
                        ("Bbox3dPublish.inputs:topicName", "bbox_3d"),
                        ("Bbox3dPublish.inputs:type", "bbox_3d"),
                        ("Bbox3dPublish.inputs:resetSimulationTimeOnStop", True),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("OnPlaybackTick.outputs:tick", "CreateRenderProduct.inputs:execIn"),
                        ("CreateRenderProduct.outputs:execOut", "RGBPublish.inputs:execIn"),
                        ("CreateRenderProduct.outputs:execOut", "DepthPublish.inputs:execIn"),
                        ("CreateRenderProduct.outputs:execOut", "DepthPclPublish.inputs:execIn"),
                        ("CreateRenderProduct.outputs:execOut", "InstancePublish.inputs:execIn"),
                        ("CreateRenderProduct.outputs:execOut", "SemanticPublish.inputs:execIn"),
                        ("CreateRenderProduct.outputs:execOut", "Bbox2dTightPublish.inputs:execIn"),
                        ("CreateRenderProduct.outputs:execOut", "Bbox2dLoosePublish.inputs:execIn"),
                        ("CreateRenderProduct.outputs:execOut", "Bbox3dPublish.inputs:execIn"),
                        ("CreateRenderProduct.outputs:renderProductPath", "RGBPublish.inputs:renderProductPath"),
                        ("CreateRenderProduct.outputs:renderProductPath", "DepthPublish.inputs:renderProductPath"),
                        ("CreateRenderProduct.outputs:renderProductPath", "DepthPclPublish.inputs:renderProductPath"),
                        ("CreateRenderProduct.outputs:renderProductPath", "InstancePublish.inputs:renderProductPath"),
                        ("CreateRenderProduct.outputs:renderProductPath", "SemanticPublish.inputs:renderProductPath"),
                        (
                            "CreateRenderProduct.outputs:renderProductPath",
                            "Bbox2dTightPublish.inputs:renderProductPath",
                        ),
                        (
                            "CreateRenderProduct.outputs:renderProductPath",
                            "Bbox2dLoosePublish.inputs:renderProductPath",
                        ),
                        ("CreateRenderProduct.outputs:renderProductPath", "Bbox3dPublish.inputs:renderProductPath"),
                    ],
                },
            )
        except Exception as e:
            print(e)
        await omni.kit.app.get_app().next_update_async()

        from sensor_msgs.msg import Image, PointCloud2
        from vision_msgs.msg import Detection2DArray, Detection3DArray

        self._rgb = None
        self._depth = None
        self._depth_pcl = None
        self._instance_segmentation = None
        self._semantic_segmentation = None
        self._bbox_2d_tight = None
        self._bbox_2d_loose = None
        self._bbox_3d = None

        def rgb_callback(data):
            self._rgb = data

        def depth_callback(data):
            self._depth = data

        def depth_pcl_callback(data):
            self._depth_pcl = data

        def instance_segmentation_callback(data):
            self._instance_segmentation = data

        def semantic_segmentation_callback(data):
            self._semantic_segmentation = data

        def bbox_2d_tight_callback(data):
            self._bbox_2d_tight = data

        def bbox_2d_loose_callback(data):
            self._bbox_2d_loose = data

        def bbox_3d_callback(data):
            self._bbox_3d = data

        node = self.create_node("camera_tester")
        rgb_sub = self.create_subscription(node, Image, "rgb", rgb_callback, get_qos_profile())
        depth_sub = self.create_subscription(node, Image, "depth", depth_callback, get_qos_profile())
        depth_pcl_sub = self.create_subscription(node, PointCloud2, "depth_pcl", depth_pcl_callback, get_qos_profile())
        instance_segmentation_sub = self.create_subscription(
            node, Image, "instance_segmentation", instance_segmentation_callback, get_qos_profile()
        )
        semantic_segmentation_sub = self.create_subscription(
            node, Image, "semantic_segmentation", semantic_segmentation_callback, get_qos_profile()
        )
        bbox_2d_tight_sub = self.create_subscription(
            node, Detection2DArray, "bbox_2d_tight", bbox_2d_tight_callback, get_qos_profile()
        )
        bbox_2d_loose_sub = self.create_subscription(
            node, Detection2DArray, "bbox_2d_loose", bbox_2d_loose_callback, get_qos_profile()
        )
        bbox_3d_sub = self.create_subscription(node, Detection3DArray, "bbox_3d", bbox_3d_callback, get_qos_profile())

        await omni.kit.app.get_app().next_update_async()
        omni.kit.commands.execute(
            "ChangeProperty", prop_path=Sdf.Path("/OmniverseKit_Persp.horizontalAperture"), value=6.0, prev=0
        )

        # square pixels, vertical apertures are computed by the horizontal aperture
        # omni.kit.commands.execute(
        #     "ChangeProperty", prop_path=Sdf.Path("/OmniverseKit_Persp.verticalAperture"), value=4.5, prev=0
        # )

        def spin():
            rclpy.spin_once(node, timeout_sec=0.1)

        import time

        # Turn on SystemTime for timestamp of all camera publishers
        og.Controller.attribute("/ActionGraph/RGBPublish" + ".inputs:useSystemTime").set(True)
        og.Controller.attribute("/ActionGraph/DepthPublish" + ".inputs:useSystemTime").set(True)
        og.Controller.attribute("/ActionGraph/DepthPclPublish" + ".inputs:useSystemTime").set(True)
        og.Controller.attribute("/ActionGraph/InstancePublish" + ".inputs:useSystemTime").set(True)
        og.Controller.attribute("/ActionGraph/SemanticPublish" + ".inputs:useSystemTime").set(True)
        og.Controller.attribute("/ActionGraph/Bbox2dTightPublish" + ".inputs:useSystemTime").set(True)
        og.Controller.attribute("/ActionGraph/Bbox2dLoosePublish" + ".inputs:useSystemTime").set(True)
        og.Controller.attribute("/ActionGraph/Bbox3dPublish" + ".inputs:useSystemTime").set(True)

        await omni.kit.app.get_app().next_update_async()

        system_time = time.time()

        self._timeline.play()
        await self.simulate_until_condition(
            lambda: (
                self._rgb is not None
                and self._instance_segmentation is not None
                and self._semantic_segmentation is not None
                and self._bbox_2d_tight is not None
                and self._bbox_2d_loose is not None
                and self._bbox_3d is not None
            ),
            max_frames=600,
            per_frame_callback=spin,
        )

        self.assertIsNotNone(self._rgb)
        self.assertIsNotNone(self._instance_segmentation)
        self.assertIsNotNone(self._semantic_segmentation)
        self.assertIsNotNone(self._bbox_2d_tight)
        self.assertIsNotNone(self._bbox_2d_loose)
        self.assertIsNotNone(self._bbox_3d)

        self.assertGreaterEqual(self._rgb.header.stamp.sec, system_time)
        self.assertGreaterEqual(self._depth.header.stamp.sec, system_time)
        self.assertGreaterEqual(self._depth_pcl.header.stamp.sec, system_time)
        self.assertGreaterEqual(self._instance_segmentation.header.stamp.sec, system_time)
        self.assertGreaterEqual(self._semantic_segmentation.header.stamp.sec, system_time)
        self.assertGreaterEqual(self._bbox_2d_tight.header.stamp.sec, system_time)
        self.assertGreaterEqual(self._bbox_2d_loose.header.stamp.sec, system_time)
        self.assertGreaterEqual(self._bbox_3d.header.stamp.sec, system_time)

    async def test_bbox(self):
        cube_1 = VisualCuboid("/cube_1", position=[2, 0, 0], scale=[1.5, 1, 1])
        cube_2 = VisualCuboid("/cube_2", position=[-1.5, 0, 0], scale=[1, 2, 1])
        cube_3 = VisualCuboid("/cube_3", position=[100, 0, 0], scale=[1, 1, 3])
        cube_4 = VisualCuboid("/cube_4", position=[0, 1, 0], scale=[1, 1, 3])
        add_labels(cube_1.prim, labels=["Cube0"], instance_name="class")
        add_labels(cube_2.prim, labels=["Cube1"], instance_name="class")
        add_labels(cube_3.prim, labels=["Cube2"], instance_name="class")
        add_labels(cube_4.prim, labels=["Cube3"], instance_name="class")
        set_camera_view(eye=[0, -6, 0.5], target=[0, 0, 0.5], camera_prim_path="/OmniverseKit_Persp")
        import json

        import rclpy

        viewport_api = omni.kit.viewport.utility.get_active_viewport()
        render_product_path = viewport_api.get_render_product_path()

        try:
            og.Controller.edit(
                {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("Bbox2dTightPublish", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                        ("Bbox2dLoosePublish", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                        ("Bbox3dPublish", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                        ("InstancePublish", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                        ("SemanticPublish", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        ("InstancePublish.inputs:renderProductPath", render_product_path),
                        ("InstancePublish.inputs:topicName", "instance_segmentation"),
                        ("InstancePublish.inputs:type", "instance_segmentation"),
                        ("InstancePublish.inputs:resetSimulationTimeOnStop", True),
                        ("SemanticPublish.inputs:renderProductPath", render_product_path),
                        ("SemanticPublish.inputs:topicName", "semantic_segmentation"),
                        ("SemanticPublish.inputs:type", "semantic_segmentation"),
                        ("SemanticPublish.inputs:resetSimulationTimeOnStop", True),
                        ("Bbox2dTightPublish.inputs:renderProductPath", render_product_path),
                        ("Bbox2dTightPublish.inputs:topicName", "bbox_2d_tight"),
                        ("Bbox2dTightPublish.inputs:type", "bbox_2d_tight"),
                        ("Bbox2dTightPublish.inputs:resetSimulationTimeOnStop", True),
                        ("Bbox2dLoosePublish.inputs:renderProductPath", render_product_path),
                        ("Bbox2dLoosePublish.inputs:topicName", "bbox_2d_loose"),
                        ("Bbox2dLoosePublish.inputs:type", "bbox_2d_loose"),
                        ("Bbox2dLoosePublish.inputs:resetSimulationTimeOnStop", True),
                        ("Bbox3dPublish.inputs:renderProductPath", render_product_path),
                        ("Bbox3dPublish.inputs:topicName", "bbox_3d"),
                        ("Bbox3dPublish.inputs:type", "bbox_3d"),
                        ("Bbox3dPublish.inputs:resetSimulationTimeOnStop", True),
                        # enable semantics
                        ("InstancePublish.inputs:enableSemanticLabels", True),
                        ("InstancePublish.inputs:semanticLabelsTopicName", "semantic_labels_instance"),
                        ("SemanticPublish.inputs:enableSemanticLabels", True),
                        ("SemanticPublish.inputs:semanticLabelsTopicName", "semantic_labels_semantic"),
                        ("Bbox2dTightPublish.inputs:enableSemanticLabels", True),
                        ("Bbox2dTightPublish.inputs:semanticLabelsTopicName", "semantic_labels_tight"),
                        ("Bbox2dLoosePublish.inputs:enableSemanticLabels", True),
                        ("Bbox2dLoosePublish.inputs:semanticLabelsTopicName", "semantic_labels_loose"),
                        ("Bbox3dPublish.inputs:enableSemanticLabels", True),
                        ("Bbox3dPublish.inputs:semanticLabelsTopicName", "semantic_labels_3d"),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("OnPlaybackTick.outputs:tick", "InstancePublish.inputs:execIn"),
                        ("OnPlaybackTick.outputs:tick", "SemanticPublish.inputs:execIn"),
                        ("OnPlaybackTick.outputs:tick", "Bbox2dTightPublish.inputs:execIn"),
                        ("OnPlaybackTick.outputs:tick", "Bbox2dLoosePublish.inputs:execIn"),
                        ("OnPlaybackTick.outputs:tick", "Bbox3dPublish.inputs:execIn"),
                    ],
                },
            )
        except Exception as e:
            print(e)

        # acquire the viewport window
        viewport_api = omni.kit.viewport.utility.get_active_viewport()
        # Set viewport resolution, changes will occur on next frame

        await omni.kit.app.get_app().next_update_async()

        from std_msgs.msg import String
        from vision_msgs.msg import Detection2DArray, Detection3DArray

        self._bbox_2d_tight = None
        self._bbox_2d_loose = None
        self._bbox_3d = None
        self._semantic_data_instance = None
        self._semantic_data_semantic = None
        self._semantic_data_3d = None
        self._semantic_data_tight = None
        self._semantic_data_loose = None

        def bbox_2d_tight_callback(data):
            self._bbox_2d_tight = data

        def bbox_2d_loose_callback(data):
            self._bbox_2d_loose = data

        def bbox_3d_callback(data):
            self._bbox_3d = data

        def semantic_callback_instance(data):
            self._semantic_data_instance = data

        def semantic_callback_semantic(data):
            self._semantic_data_semantic = data

        def semantic_callback_3d(data):
            self._semantic_data_3d = data

        def semantic_callback_tight(data):
            self._semantic_data_tight = data

        def semantic_callback_loose(data):
            self._semantic_data_loose = data

        node = self.create_node("bbox_tester")

        bbox_2d_tight_sub = self.create_subscription(
            node, Detection2DArray, "bbox_2d_tight", bbox_2d_tight_callback, get_qos_profile()
        )
        bbox_2d_loose_sub = self.create_subscription(
            node, Detection2DArray, "bbox_2d_loose", bbox_2d_loose_callback, get_qos_profile()
        )

        bbox_3d_sub = self.create_subscription(node, Detection3DArray, "bbox_3d", bbox_3d_callback, get_qos_profile())
        semantic_labels_instance_sub = self.create_subscription(
            node, String, "semantic_labels_instance", semantic_callback_instance, get_qos_profile()
        )
        semantic_labels_semantic_sub = self.create_subscription(
            node, String, "semantic_labels_semantic", semantic_callback_semantic, get_qos_profile()
        )
        semantic_labels_3d_sub = self.create_subscription(
            node, String, "semantic_labels_3d", semantic_callback_3d, get_qos_profile()
        )
        semantic_labels_tight_sub = self.create_subscription(
            node, String, "semantic_labels_tight", semantic_callback_tight, get_qos_profile()
        )
        semantic_labels_loose_sub = self.create_subscription(
            node, String, "semantic_labels_loose", semantic_callback_loose, get_qos_profile()
        )

        def spin():
            rclpy.spin_once(node, timeout_sec=0.01)

        await omni.kit.app.get_app().next_update_async()

        self._timeline.play()
        await omni.kit.app.get_app().next_update_async()
        await self.simulate_until_condition(
            lambda: (
                self._bbox_2d_tight is not None
                and self._bbox_2d_loose is not None
                and self._bbox_3d is not None
                and self._semantic_data_instance is not None
                and self._semantic_data_semantic is not None
                and self._semantic_data_3d is not None
                and self._semantic_data_tight is not None
                and self._semantic_data_loose is not None
            ),
            max_frames=600,
            per_frame_callback=spin,
        )

        self.assertIsNotNone(self._bbox_2d_tight)
        self.assertIsNotNone(self._bbox_2d_loose)
        self.assertIsNotNone(self._bbox_3d)
        self.assertIsNotNone(self._semantic_data_instance)
        self.assertIsNotNone(self._semantic_data_semantic)
        self.assertIsNotNone(self._semantic_data_3d)
        self.assertIsNotNone(self._semantic_data_tight)
        self.assertIsNotNone(self._semantic_data_loose)

        detections = self._bbox_3d.detections
        semantic_instance_dict = json.loads(self._semantic_data_instance.data)
        semantic_semantic_dict = json.loads(self._semantic_data_semantic.data)
        semantic_tight_dict = json.loads(self._semantic_data_tight.data)
        semantic_loose_dict = json.loads(self._semantic_data_loose.data)
        semantic_3d_dict = json.loads(self._semantic_data_3d.data)
        print(semantic_instance_dict)
        print(semantic_semantic_dict)
        print(semantic_tight_dict)
        print(semantic_loose_dict)
        print(semantic_3d_dict)
        self.assertEqual(semantic_instance_dict["0"], "BACKGROUND")
        self.assertEqual(semantic_instance_dict["1"], "UNLABELLED")
        self.assertEqual(semantic_instance_dict["2"], "/cube_1")
        self.assertEqual(semantic_instance_dict["3"], "/cube_2")
        self.assertEqual(semantic_instance_dict["5"], "/cube_4")

        self.assertEqual(semantic_semantic_dict["0"]["class"], "BACKGROUND")
        self.assertEqual(len(semantic_semantic_dict.keys()), 6)  # (background + unalbeled + 3 cubes + timestamp)

        # all times should match
        # TODO: Find a way to align timestamps for testing
        # self.assertDictEqual(semantic_3d_dict["time_stamp"], semantic_instance_dict["time_stamp"])
        # self.assertDictEqual(semantic_3d_dict["time_stamp"], semantic_semantic_dict["time_stamp"])

        # bbox semantics should match
        # TODO: Find a way to align timestamps for testing
        # self.assertEqual(self._semantic_data_3d.data, self._semantic_data_tight.data)
        # self.assertEqual(self._semantic_data_3d.data, self._semantic_data_loose.data)

        self.assertEqual(semantic_3d_dict["0"]["class"], "cube0")
        self.assertEqual(semantic_3d_dict["1"]["class"], "cube1")
        self.assertEqual(semantic_3d_dict["2"]["class"], "cube3")

        # there should be 3 bboxes
        self.assertEqual(len(detections), 3)
        self.assertEqual(detections[0].results[0].hypothesis.class_id, "0")
        self.assertEqual(detections[1].results[0].hypothesis.class_id, "1")
        self.assertEqual(detections[2].results[0].hypothesis.class_id, "2")

        self.assertEqual(detections[0].bbox.size.x, 1.5)
        self.assertEqual(detections[0].bbox.size.y, 1)
        self.assertEqual(detections[0].bbox.size.z, 1)

        self.assertEqual(detections[1].bbox.size.x, 1)
        self.assertEqual(detections[1].bbox.size.y, 2)
        self.assertEqual(detections[1].bbox.size.z, 1)

        self.assertEqual(detections[2].bbox.size.x, 1)
        self.assertEqual(detections[2].bbox.size.y, 1)
        self.assertEqual(detections[2].bbox.size.z, 3)

        self.assertEqual(detections[0].bbox.center.position.x, 2)
        self.assertEqual(detections[0].bbox.center.position.y, 0)
        self.assertEqual(detections[0].bbox.center.position.z, 0)

        self.assertEqual(detections[1].bbox.center.position.x, -1.5)
        self.assertEqual(detections[1].bbox.center.position.y, 0)
        self.assertEqual(detections[1].bbox.center.position.z, 0)

        self.assertEqual(detections[2].bbox.center.position.x, 0)
        self.assertEqual(detections[2].bbox.center.position.y, 1)
        self.assertEqual(detections[2].bbox.center.position.z, 0)

        detections = self._bbox_2d_tight.detections
        self.assertEqual(len(detections), 3)

        print(detections[0].results)
        print(detections[1].results)
        print(detections[2].results)

        self.assertEqual(detections[0].results[0].hypothesis.class_id, "0")
        self.assertEqual(detections[1].results[0].hypothesis.class_id, "1")
        self.assertEqual(detections[2].results[0].hypothesis.class_id, "2")

        self.assertEqual(detections[0].bbox.size_x, 340.0)
        self.assertEqual(detections[0].bbox.size_y, 201.0)

        self.assertEqual(detections[1].bbox.size_x, 284.0)
        self.assertEqual(detections[1].bbox.size_y, 221.0)

        self.assertEqual(detections[2].bbox.size_x, 169.0)
        self.assertEqual(detections[2].bbox.size_y, 511.0)
        self.assertEqual(detections[0].bbox.center.position.x, 1023.0)
        self.assertEqual(detections[0].bbox.center.position.y, 460.5)
        self.assertEqual(detections[0].bbox.center.theta, 0)

        self.assertEqual(detections[1].bbox.center.position.x, 339.0)
        self.assertEqual(detections[1].bbox.center.position.y, 470.5)
        self.assertEqual(detections[1].bbox.center.theta, 0)

        self.assertEqual(detections[2].bbox.center.position.x, 639.5)
        self.assertEqual(detections[2].bbox.center.position.y, 444.5)
        self.assertEqual(detections[2].bbox.center.theta, 0)

        detections = self._bbox_2d_loose.detections
        self.assertEqual(len(detections), 3)

        self.assertEqual(detections[0].results[0].hypothesis.class_id, "0")
        self.assertEqual(detections[1].results[0].hypothesis.class_id, "1")
        self.assertEqual(detections[2].results[0].hypothesis.class_id, "2")

        self.assertEqual(detections[0].bbox.size_x, 340.0)
        self.assertEqual(detections[0].bbox.size_y, 201.0)

        self.assertEqual(detections[1].bbox.size_x, 284.0)
        self.assertEqual(detections[1].bbox.size_y, 221.0)

        self.assertEqual(detections[2].bbox.size_x, 169.0)
        self.assertEqual(detections[2].bbox.size_y, 511.0)

        self.assertEqual(detections[0].bbox.center.position.x, 1023.0)
        self.assertEqual(detections[0].bbox.center.position.y, 460.5)
        self.assertEqual(detections[0].bbox.center.theta, 0)

        self.assertEqual(detections[1].bbox.center.position.x, 339.0)
        self.assertEqual(detections[1].bbox.center.position.y, 470.5)
        self.assertEqual(detections[1].bbox.center.theta, 0)

        self.assertEqual(detections[2].bbox.center.position.x, 639.5)
        self.assertEqual(detections[2].bbox.center.position.y, 444.5)
        self.assertEqual(detections[2].bbox.center.theta, 0)

    async def test_empty_semantics(self):
        cube_3 = VisualCuboid("/cube_3", position=[100, 0, 0], scale=[1, 1, 3])
        add_labels(cube_3.prim, labels=["Cube2"], instance_name="class")
        set_camera_view(eye=[0, -6, 0.5], target=[0, 0, 0.5], camera_prim_path="/OmniverseKit_Persp")
        import json

        import rclpy

        viewport_api = omni.kit.viewport.utility.get_active_viewport()
        render_product_path = viewport_api.get_render_product_path()

        try:
            og.Controller.edit(
                {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("Bbox3dPublish", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        ("Bbox3dPublish.inputs:renderProductPath", render_product_path),
                        ("Bbox3dPublish.inputs:topicName", "bbox_3d"),
                        ("Bbox3dPublish.inputs:type", "bbox_3d"),
                        ("Bbox3dPublish.inputs:resetSimulationTimeOnStop", True),
                        # enable semantics
                        ("Bbox3dPublish.inputs:enableSemanticLabels", True),
                        ("Bbox3dPublish.inputs:semanticLabelsTopicName", "semantic_labels"),
                    ],
                    og.Controller.Keys.CONNECT: [("OnPlaybackTick.outputs:tick", "Bbox3dPublish.inputs:execIn")],
                },
            )
        except Exception as e:
            print(e)

        # acquire the viewport window
        viewport_api = omni.kit.viewport.utility.get_active_viewport()
        # Set viewport resolution, changes will occur on next frame

        await omni.kit.app.get_app().next_update_async()

        from std_msgs.msg import String
        from vision_msgs.msg import Detection2DArray, Detection3DArray

        self._bbox_3d = None
        self._semantic_data = None

        def bbox_3d_callback(data):
            self._bbox_3d = data

        def semantic_callback(data):
            self._semantic_data = data

        node = self.create_node("bbox_tester")

        bbox_3d_sub = self.create_subscription(node, Detection3DArray, "bbox_3d", bbox_3d_callback, get_qos_profile())
        semantic_labels_sub = self.create_subscription(
            node, String, "semantic_labels", semantic_callback, get_qos_profile()
        )

        def spin():
            rclpy.spin_once(node, timeout_sec=0.01)

        await asyncio.sleep(2.0)

        self._timeline.play()
        await omni.kit.app.get_app().next_update_async()
        await self.simulate_until_condition(
            lambda: self._semantic_data is not None, max_frames=600, per_frame_callback=spin
        )

        self.assertIsNotNone(self._semantic_data)

        semantic_dict = json.loads(self._semantic_data.data)
        self.assertTrue("time_stamp" in semantic_dict)
        self.assertFalse("0" in semantic_dict)

    async def test_rgb_golden_image_comparison(self):
        """Subscribe to an RGB image topic and compare received buffer against a golden image."""

        # Retrieve golden image from data/tests/golden_img folder
        golden_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "golden")
        golden_img_path = os.path.join(golden_dir, "nova_carter_warehouse_front_stereo_left_rgb.png")

        # Open the nova carter warehouse scene (following simulation_control pattern)
        self._timeline.stop()
        await omni.kit.app.get_app().next_update_async()

        assets_root_path = get_assets_root_path()
        warehouse_scene = assets_root_path + "/Isaac/Samples/ROS2/Scenario/carter_warehouse_navigation.usd"
        (success, error) = await stage_utils.open_stage_async(warehouse_scene)
        self.assertTrue(success, f"Failed to open stage: {error}")

        await omni.kit.app.get_app().next_update_async()
        await omni.kit.app.get_app().next_update_async()

        # Setup ROS2 subscriber for the RGB image topic
        self._received_rgb_image = None

        def rgb_callback(data):
            self._received_rgb_image = data

        node = self.create_node("rgb_image_test_node")
        rgb_sub = self.create_subscription(
            node, Image, "/front_stereo_camera/left/image_raw", rgb_callback, get_qos_profile()
        )

        def spin():
            rclpy.spin_once(node, timeout_sec=0.1)

        await omni.kit.app.get_app().next_update_async()

        # Move /World/Nova_Carter_ROS to -6, -1, 0 and 180 degree rotation around z axis
        # Quaternion for 180 deg rotation around z: (w=0, x=0, y=0, z=1)
        nova_carter = XformPrim("/World/Nova_Carter_ROS", reset_xform_op_properties=True)
        nova_carter.set_world_poses(positions=[-6, -1, 0], orientations=[0, 0, 0, 1])

        await omni.kit.app.get_app().next_update_async()

        # Hit Play on scene and wait for image
        self._timeline.play()
        await self.simulate_until_condition(
            lambda: self._received_rgb_image is not None,
            max_frames=300,
            per_frame_callback=spin,
        )

        # Verify image was received
        self.assertIsNotNone(self._received_rgb_image, "Failed to receive RGB image from topic")

        # Retrieve image buffer from subscriber
        received_array = ros2_image_to_buffer(
            self._received_rgb_image,
            normalize_color_order=True,
            squeeze_singleton_channel=True,
            copy=True,
        )

        # Hit Stop on scene
        self._timeline.stop()
        await omni.kit.app.get_app().next_update_async()

        # Compare image with golden image
        golden_img_data = read_image_as_array(str(golden_img_path))

        # Handle channel mismatch between RGBA golden and RGB received
        if golden_img_data.ndim == 3 and golden_img_data.shape[2] == 4:
            golden_img_data = golden_img_data[:, :, :3]

        results = compare_arrays_within_tolerances(
            golden_img_data,
            received_array,
            allclose_rtol=None,
            allclose_atol=None,
            mean_tolerance=10,
            print_all_stats=True,
        )
        self.assertTrue(results["passed"], f"Image comparison failed: {results}")

    async def test_rgb_h264_compressed_golden_image_comparison(self):
        """Subscribe to a compressed RGB H264 image topic, decode with PyNvVideoCodec, and compare against golden image."""
        try:
            import PyNvVideoCodec as nvc
        except ImportError:
            self.skipTest("PyNvVideoCodec not available - skipping H264 decode test")

        import numpy as np
        from sensor_msgs.msg import CompressedImage

        # Ensure omni.replicator.nv extension is enabled (provides H264 hardware encoder)
        ext_manager = omni.kit.app.get_app().get_extension_manager()
        ext_manager.set_extension_enabled_immediate("omni.replicator.nv", True)
        await omni.kit.app.get_app().next_update_async()

        # Retrieve golden image from data/tests/golden_img folder
        golden_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "golden")
        golden_img_path = os.path.join(golden_dir, "nova_carter_warehouse_front_stereo_left_rgb.png")

        # Open the nova carter warehouse scene
        self._timeline.stop()
        await omni.kit.app.get_app().next_update_async()

        assets_root_path = get_assets_root_path()
        warehouse_scene = assets_root_path + "/Isaac/Samples/ROS2/Scenario/carter_warehouse_navigation.usd"
        (success, error) = await stage_utils.open_stage_async(warehouse_scene)
        self.assertTrue(success, f"Failed to open stage: {error}")

        await omni.kit.app.get_app().next_update_async()
        await omni.kit.app.get_app().next_update_async()

        # Modify the existing front stereo camera left RGB publisher to use H264 compression
        og.Controller.attribute("/World/Nova_Carter_ROS/front_hawk/left_camera_publish_image.inputs:type").set(
            "rgb_h264"
        )

        og.Controller.attribute("/World/Nova_Carter_ROS/front_hawk/left_camera_publish_image.inputs:topicName").set(
            "left/image_raw/compressed"
        )

        await omni.kit.app.get_app().next_update_async()

        # Setup ROS2 subscriber for the compressed image topic
        self._received_compressed_image = None

        def compressed_callback(data):
            self._received_compressed_image = data

        node = self.create_node("rgb_h264_test_node")
        compressed_sub = self.create_subscription(
            node,
            CompressedImage,
            "/front_stereo_camera/left/image_raw/compressed",
            compressed_callback,
            get_qos_profile(),
        )

        def spin():
            rclpy.spin_once(node, timeout_sec=0.1)

        await omni.kit.app.get_app().next_update_async()

        # Move /World/Nova_Carter_ROS to -6, -1, 0
        nova_carter = XformPrim("/World/Nova_Carter_ROS", reset_xform_op_properties=True)
        nova_carter.set_world_poses(positions=[-6, -1, 0], orientations=[0, 0, 0, 1])

        await omni.kit.app.get_app().next_update_async()

        # Play scene and wait for compressed image
        self._timeline.play()
        await self.simulate_until_condition(
            lambda: self._received_compressed_image is not None,
            max_frames=300,
            per_frame_callback=spin,
        )

        self.assertIsNotNone(self._received_compressed_image, "Failed to receive compressed image from topic")

        self._timeline.stop()
        await omni.kit.app.get_app().next_update_async()

        # Get the H264 bitstream from ROS CompressedImage message
        h264_bitstream = self._received_compressed_image.data.tobytes()

        # Decode H264 using PyNvVideoCodec (core Decoder + buffer demuxer)
        # Buffer feeder serves raw H264 elementary stream bytes to the demuxer
        class H264BufferFeeder:
            def __init__(self, data):
                self._buffer = bytearray(data)
                self._pos = 0
                self._remaining = len(self._buffer)

            def feed_chunk(self, demuxer_buffer):
                chunk = min(self._remaining, len(demuxer_buffer))
                if chunk == 0:
                    return 0
                demuxer_buffer[:chunk] = self._buffer[self._pos : self._pos + chunk]
                self._pos += chunk
                self._remaining -= chunk
                return chunk

        feeder = H264BufferFeeder(h264_bitstream)
        dmx = nvc.CreateDemuxer(feeder.feed_chunk)
        dec = nvc.CreateDecoder(
            gpuid=0,
            codec=dmx.GetNvCodecId(),
            usedevicememory=False,
        )

        frames = []
        for pkt in dmx:
            for frame in dec.Decode(pkt):
                frames.append(frame)

        self.assertTrue(len(frames) > 0, f"Failed to decode H264 frame ({len(h264_bitstream)} bytes)")

        # Convert last decoded frame to numpy array via DLPack
        # Core decoder outputs NV12 (native format); convert to RGB
        decoded_np = np.from_dlpack(frames[-1])
        if decoded_np.dtype != np.uint8:
            decoded_np = np.clip(decoded_np, 0, 255).astype(np.uint8)

        # NV12 frame has shape (H * 3/2, W) — convert to RGB (H, W, 3)
        import cv2

        received_array = cv2.cvtColor(decoded_np, cv2.COLOR_YUV2RGB_NV12)

        # Compare image with golden image
        golden_img_data = read_image_as_array(str(golden_img_path))

        # Handle channel mismatch between RGBA golden and RGB received
        if golden_img_data.ndim == 3 and golden_img_data.shape[2] == 4:
            golden_img_data = golden_img_data[:, :, :3]

        # H264 compression is lossy, so we need a higher tolerance
        results = compare_arrays_within_tolerances(
            golden_img_data,
            received_array,
            allclose_rtol=None,
            allclose_atol=None,
            mean_tolerance=15,
            print_all_stats=True,
        )
        self.assertTrue(results["passed"], f"H264 compressed image comparison failed: {results}")

    async def test_spinning_camera_golden_images(self):
        """Two cameras on one spinning rigid body: compare physics images to golden images.

        Camera 1 is a Camera prim that also carries the RigidBodyAPI and spins at 90 deg/s.
        After camera 1 completes one full rotation, camera 2 is added *live* (no pause) as a
        child prim of camera 1 -- facing 180 degrees opposite, offset vertically, and tilted
        down ~9.5 degrees so it sees a completely different perspective while sharing the same
        spin.  No second rigid body is created.

        Two golden image sets are captured and validated:
          1a. Camera 1 first rotation (camera 2 does not exist yet).
          1b. Camera 1 second rotation + camera 2 first rotation (same rig angles).
          2.  Golden images by teleporting the rig to recorded angles.
          3.  Comparison: camera 1 reuses its first-rotation goldens for the
              second rotation (same speed), camera 2 uses its own goldens.
        """
        update_golden_images = False
        save_debug_images = False
        keyframe_angles_deg = list(range(0, 360, 30))
        camera_height = 0.5
        rotation_speed_deg_per_sec = 90
        cam2_vertical_offset = 1.0  # metres above camera 1 (local +Y = world +Z)
        cam2_tilt_deg = 10.0  # degrees downward tilt

        golden_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "golden", "spinning_camera")

        # Open the pre-built scene USD (contains physics scene and scattered objects).
        scene_usd_path = os.path.join(golden_dir, "spinning_camera_scene.usd")
        await open_stage_async(scene_usd_path)
        await omni.kit.app.get_app().next_update_async()

        # Add the grid environment as a reference.
        stage_utils.add_reference_to_stage(
            usd_path=get_assets_root_path() + "/Isaac/Environments/Grid/default_environment.usd",
            path="/World/EnvGrid",
        )

        width, height = 640, 360

        # Single camera prim at center (Camera + rigid body, no CollisionAPI)
        camera_path = "/World/SpinningCamera"
        stage_utils.define_prim(camera_path, type_name="Camera")
        # RigidPrim automatically applies RigidBodyAPI, PhysxRigidBodyAPI, and MassAPI
        camera_rigid = RigidPrim(camera_path, masses=[0.1], reset_xform_op_properties=True)
        camera_rigid.set_enabled_gravities([False])
        # Zero damping so angular velocity is maintained exactly
        physx_api = PhysxSchema.PhysxRigidBodyAPI(camera_rigid.prims[0])
        physx_api.CreateAngularDampingAttr().Set(0.0)
        physx_api.CreateLinearDampingAttr().Set(0.0)
        camera_rigid.set_world_poses(
            positions=[[0.0, 0.0, camera_height]],
            orientations=[_camera_orientation_at_angle_deg(0.0)],
        )

        # Rotate camera around world Z (angular velocity in rad/s)
        camera_rigid.set_velocities(
            linear_velocities=[[0.0, 0.0, 0.0]],
            angular_velocities=[[0.0, 0.0, 0.0]],
        )
        await omni.kit.app.get_app().next_update_async()

        # ROS2 camera publisher
        _create_rgb_camera_graph("/ActionGraph", camera_path, "spinning_camera_rgb", width, height)
        await omni.kit.app.get_app().next_update_async()

        # Buffer all received ROS2 images with their timestamps
        image_buffer = []  # list of (timestamp, image_array)

        def rgb_callback(data):
            ts = data.header.stamp.sec + data.header.stamp.nanosec / 1e9
            img = ros2_image_to_buffer(data, normalize_color_order=True, squeeze_singleton_channel=True, copy=True)
            image_buffer.append((ts, img))

        node = self.create_node("spinning_camera_test_node")
        self.start_async_spinning(node)
        self.create_subscription(
            node,
            Image,
            "spinning_camera_rgb",
            rgb_callback,
            get_qos_profile(history="system_default"),
        )

        # ============================================================
        # STEP 1: Physics-based rotation - simulate and buffer images
        # ============================================================
        self._timeline.play()
        await omni.kit.app.get_app().next_update_async()
        await omni.kit.app.get_app().next_update_async()

        # Wait for the first ROS2 image to confirm the pipeline is running
        await self.simulate_until_condition(
            lambda: len(image_buffer) > 0,
        )

        stage_fps = self._timeline.get_time_codes_per_second()
        angle_tolerance_deg = 0.1
        # One full rotation plus extra frames for pipeline-delayed images to arrive
        rotation_frames = int((360.0 / rotation_speed_deg_per_sec) * stage_fps)
        pipeline_drain_frames = 30  # extra frames for delayed images to flush through
        total_frames = rotation_frames + pipeline_drain_frames

        # Record angle + sim_time at each target during the rotation
        recorded_sim_times = {}  # target_angle -> sim_time
        recorded_angles = {}  # target_angle -> actual_angle
        image_buffer.clear()

        camera_rigid.set_velocities(
            linear_velocities=[[0.0, 0.0, 0.0]],
            angular_velocities=[[0.0, 0.0, math.radians(rotation_speed_deg_per_sec)]],
        )

        print(f"Starting physics-based rotation capture ({rotation_speed_deg_per_sec} deg/s)...")
        for _ in range(total_frames):
            await omni.kit.app.get_app().next_update_async()

            # Record angle if near any unrecorded target
            if len(recorded_sim_times) < len(keyframe_angles_deg):
                sim_time = SimulationManager.get_simulation_time()
                _, orientations = camera_rigid.get_world_poses()
                ori = orientations.numpy()[0]
                actual_angle = _view_angle_deg_from_quat_wxyz([ori[0], ori[1], ori[2], ori[3]])
                for target in keyframe_angles_deg:
                    if target in recorded_sim_times:
                        continue
                    angle_diff = abs(actual_angle - target)
                    if angle_diff > 180:
                        angle_diff = 360 - angle_diff
                    if angle_diff <= angle_tolerance_deg:
                        recorded_sim_times[target] = sim_time
                        recorded_angles[target] = actual_angle
                        print(f"Angle {target}° at sim_time={sim_time:.6f}s (actual={actual_angle:.2f}°)")

        missing_angles = [a for a in keyframe_angles_deg if a not in recorded_sim_times]
        if missing_angles:
            self.fail(f"Did not observe angles during rotation: {missing_angles}")

        print(f"Buffered {len(image_buffer)} ROS2 images during rotation.")

        # Snapshot the first-rotation buffer; matching is deferred until after both
        # rotations finish so the rig isn't wasting simulation frames on processing.
        timestamp_tolerance = 1.5 / stage_fps
        image_buffer_r1 = list(image_buffer)

        # ===========================================================
        # ADD CAMERA 2 as child of camera 1 (while paused)
        # ============================================================
        # self._timeline.pause()
        # await omni.kit.app.get_app().next_update_async()

        print("[Live] Adding camera 2 as child of camera 1...")
        camera_path_2 = camera_path + "/Camera2"
        stage_utils.define_prim(camera_path_2, type_name="Camera")
        cam2_xform = XformPrim(camera_path_2, reset_xform_op_properties=True)
        cam2_local_quat = transform_utils.euler_angles_to_quaternion(
            [0.0, 180.0, -cam2_tilt_deg], degrees=True, extrinsic=True
        )
        cam2_xform.set_local_poses(
            translations=[[0.0, cam2_vertical_offset, 0.0]],
            orientations=[cam2_local_quat.numpy().tolist()],
        )
        await omni.kit.app.get_app().next_update_async()

        # ROS2 camera 2 publisher Graph
        _create_rgb_camera_graph("/ActionGraph2", camera_path_2, "spinning_camera_2_rgb", width, height)
        await omni.kit.app.get_app().next_update_async()

        image_buffer_2 = []

        def rgb_callback_2(data):
            ts = data.header.stamp.sec + data.header.stamp.nanosec / 1e9
            img = ros2_image_to_buffer(data, normalize_color_order=True, squeeze_singleton_channel=True, copy=True)
            image_buffer_2.append((ts, img))

        self.create_subscription(
            node,
            Image,
            "spinning_camera_2_rgb",
            rgb_callback_2,
            get_qos_profile(history="system_default"),
        )
        print("[Live] Camera 2 subscription added, background executor handles both cameras.")

        # ============================================================
        # STEP 1b: Both cameras rotate on the same rig.
        # Camera 1 continues its second rotation; camera 2 rides along for its first.
        # Only one set of keyframe times is needed (same rig orientation).
        # ============================================================
        image_buffer.clear()
        image_buffer_2.clear()

        recorded_sim_times_1b = {}
        recorded_angles_1b = {}
        angle_tolerance_1b_deg = 0.2

        print("[STEP 1b] Both cameras rotating (same rig)...")
        for _ in range(total_frames):
            await omni.kit.app.get_app().next_update_async()

            if len(recorded_sim_times_1b) < len(keyframe_angles_deg):
                sim_time = SimulationManager.get_simulation_time()
                _, orientations = camera_rigid.get_world_poses()
                ori = orientations.numpy()[0]
                actual_angle = _view_angle_deg_from_quat_wxyz([ori[0], ori[1], ori[2], ori[3]])
                for target in keyframe_angles_deg:
                    if target in recorded_sim_times_1b:
                        continue
                    angle_diff = abs(actual_angle - target)
                    if angle_diff > 180:
                        angle_diff = 360 - angle_diff
                    if angle_diff <= angle_tolerance_1b_deg:
                        recorded_sim_times_1b[target] = sim_time
                        recorded_angles_1b[target] = actual_angle
                        print(f"  rig {target}° at sim_time={sim_time:.6f}s (actual={actual_angle:.2f}°)")

        missing_angles_1b = [a for a in keyframe_angles_deg if a not in recorded_sim_times_1b]
        if missing_angles_1b:
            self.fail(f"[STEP 1b] Did not observe rig angles: {missing_angles_1b}")

        print(f"[STEP 1b] Buffered {len(image_buffer)} cam1 and {len(image_buffer_2)} cam2 images.")

        # Match camera 1 first rotation (deferred from STEP 1a to avoid wasting sim frames)
        physics_images, _ = _match_buffered_images(
            image_buffer_r1, recorded_sim_times, timestamp_tolerance, label="physics "
        )
        if save_debug_images:
            debug_dir = os.path.join(golden_dir, "debug_captured")
            os.makedirs(debug_dir, exist_ok=True)
            for target, img in physics_images.items():
                save_rgb_image(img, debug_dir, f"physics_angle_{target}.png")

        missing_images = [a for a in keyframe_angles_deg if a not in physics_images]
        if missing_images:
            self.fail(
                f"Could not match physics images for angles: {missing_images}. "
                f"Buffered {len(image_buffer_r1)} images, tolerance={timestamp_tolerance:.6f}s"
            )

        # Find timestamps present in both cam1 and cam2, then match to rig angles
        cam1_by_ts = {ts: img for ts, img in image_buffer}
        cam2_by_ts = {ts: img for ts, img in image_buffer_2}
        cam1_timestamps = sorted(cam1_by_ts.keys())
        cam2_timestamps = sorted(cam2_by_ts.keys())
        common_timestamps = sorted(set(cam1_timestamps) & set(cam2_timestamps))

        print(f"\n=== Timestamp dump (cam1: {len(cam1_timestamps)}, cam2: {len(cam2_timestamps)}) ===")
        print(f"cam1 timestamps: {[f'{t:.6f}' for t in cam1_timestamps]}")
        print(f"cam2 timestamps: {[f'{t:.6f}' for t in cam2_timestamps]}")
        print(f"Common timestamps ({len(common_timestamps)}): {[f'{t:.6f}' for t in common_timestamps]}")
        cam1_only = sorted(set(cam1_timestamps) - set(cam2_timestamps))
        cam2_only = sorted(set(cam2_timestamps) - set(cam1_timestamps))
        if cam1_only:
            print(f"cam1 only ({len(cam1_only)}): {[f'{t:.6f}' for t in cam1_only]}")
        if cam2_only:
            print(f"cam2 only ({len(cam2_only)}): {[f'{t:.6f}' for t in cam2_only]}")
        print(f"Recorded rig sim_times: { {a: f'{t:.6f}' for a, t in sorted(recorded_sim_times_1b.items())} }")
        print("=== End timestamp dump ===\n")

        physics_images_1b = {}
        physics_images_2 = {}
        for target_angle in keyframe_angles_deg:
            target_sim_time = recorded_sim_times_1b[target_angle]
            best_ts = None
            best_diff = float("inf")
            for ts in common_timestamps:
                diff = abs(ts - target_sim_time)
                if diff < best_diff:
                    best_diff = diff
                    best_ts = ts
            if best_ts is not None and best_diff <= timestamp_tolerance:
                physics_images_1b[target_angle] = cam1_by_ts[best_ts]
                physics_images_2[target_angle] = cam2_by_ts[best_ts]
                print(f"  {target_angle}° paired at ts={best_ts:.6f}s (diff={best_diff:.6f}s)")
            else:
                print(
                    f"WARNING: no common timestamp for {target_angle}° "
                    f"(sim_time={target_sim_time:.6f}s, best_diff={best_diff:.6f}s)"
                )

        if save_debug_images:
            debug_dir_1b = os.path.join(golden_dir, "debug_captured_camera_1_2nd")
            os.makedirs(debug_dir_1b, exist_ok=True)
            for target, img in physics_images_1b.items():
                save_rgb_image(img, debug_dir_1b, f"physics_angle_{target}.png")
            debug_dir_2 = os.path.join(golden_dir, "debug_captured_camera_2")
            os.makedirs(debug_dir_2, exist_ok=True)
            for target, img in physics_images_2.items():
                save_rgb_image(img, debug_dir_2, f"physics_angle_{target}.png")

        missing_1b = [a for a in keyframe_angles_deg if a not in physics_images_1b]
        if missing_1b:
            self.fail(
                f"[STEP 1b] Could not match cam1/cam2 pair for angles: {missing_1b}. "
                f"{len(common_timestamps)} common timestamps, tolerance={timestamp_tolerance:.6f}s"
            )

        # ============================================================
        # STEP 2: Golden images - generate or load from disk
        #   golden_images  = camera 1 (STEP 1a angles, reused for 2nd rotation)
        #   golden_images_2 = camera 2 (STEP 1b rig angles, same as cam1)
        # ============================================================
        golden_images = {}
        golden_images_2 = {}

        if update_golden_images:
            # Zero the rig velocity so teleport sticks (sim still running)
            camera_rigid.set_velocities(
                linear_velocities=[[0.0, 0.0, 0.0]],
                angular_velocities=[[0.0, 0.0, 0.0]],
            )
            image_buffer.clear()
            image_buffer_2.clear()

            for _ in range(15):
                await omni.kit.app.get_app().next_update_async()

            # Single pass: teleport rig once per angle, capture both cameras together.
            print("Generating goldens (both cameras, single pass)...")
            for target_angle in keyframe_angles_deg:
                actual_angle = recorded_angles_1b[target_angle]
                camera_rigid.set_world_poses(
                    positions=[[0.0, 0.0, camera_height]],
                    orientations=[_camera_orientation_at_angle_deg(actual_angle)],
                )
                cam1_pre = len(image_buffer)
                cam2_pre = len(image_buffer_2)
                for _ in range(15):
                    await omni.kit.app.get_app().next_update_async()
                self.assertGreater(
                    len(image_buffer),
                    cam1_pre,
                    f"No new cam1 image after teleporting to {target_angle}°",
                )
                self.assertGreater(
                    len(image_buffer_2),
                    cam2_pre,
                    f"No new cam2 image after teleporting to {target_angle}°",
                )
                golden_images[target_angle] = image_buffer[-1][1]
                golden_images_2[target_angle] = image_buffer_2[-1][1]
                print(f"  {target_angle}° captured (cam1 + cam2)")

            self.stop_async_spinning(node)
            self._timeline.stop()
            await omni.kit.app.get_app().next_update_async()

            for target_angle in keyframe_angles_deg:
                save_rgb_image(golden_images[target_angle], golden_dir, f"angle_{target_angle}_camera_1.png")
                save_rgb_image(golden_images_2[target_angle], golden_dir, f"angle_{target_angle}_camera_2.png")
            print("Golden image generation complete.")
        else:
            self.stop_async_spinning(node)
            self._timeline.stop()
            await omni.kit.app.get_app().next_update_async()
            print("Loading existing golden images from disk...")
            for target_angle in keyframe_angles_deg:
                golden_path_1 = os.path.join(golden_dir, f"angle_{target_angle}_camera_1.png")
                golden_path_2 = os.path.join(golden_dir, f"angle_{target_angle}_camera_2.png")
                for p in [golden_path_1, golden_path_2]:
                    self.assertTrue(
                        os.path.isfile(p),
                        f"Golden image not found: {p}. Set update_golden_images=True to generate.",
                    )
                golden_img_1 = read_image_as_array(golden_path_1)
                if golden_img_1.ndim == 3 and golden_img_1.shape[2] == 4:
                    golden_img_1 = golden_img_1[:, :, :3]
                golden_images[target_angle] = golden_img_1
                golden_img_2 = read_image_as_array(golden_path_2)
                if golden_img_2.ndim == 3 and golden_img_2.shape[2] == 4:
                    golden_img_2 = golden_img_2[:, :, :3]
                golden_images_2[target_angle] = golden_img_2
                print(f"Loaded goldens for {target_angle}°")

        # ============================================================
        # STEP 3: Compare physics-captured images to golden images
        # ============================================================
        print("Comparing camera 1 (1st rotation) physics vs golden...")
        for target_angle in keyframe_angles_deg:
            print(f"Comparing camera 1 (1st rot) at {target_angle}°")
            results = compare_arrays_within_tolerances(
                golden_images[target_angle],
                physics_images[target_angle],
                allclose_rtol=None,
                allclose_atol=None,
                mean_tolerance=10,
                print_all_stats=True,
            )
            self.assertTrue(
                results["passed"],
                f"Camera 1 (1st rotation) image comparison failed at {target_angle}°: {results}",
            )
        print("Comparing camera 1 (2nd rotation) physics vs golden...")
        for target_angle in keyframe_angles_deg:
            print(f"Comparing camera 1 (2nd rot) at {target_angle}°")
            results = compare_arrays_within_tolerances(
                golden_images[target_angle],
                physics_images_1b[target_angle],
                allclose_rtol=None,
                allclose_atol=None,
                mean_tolerance=10,
                print_all_stats=True,
            )
            self.assertTrue(
                results["passed"],
                f"Camera 1 (2nd rotation) image comparison failed at {target_angle}°: {results}",
            )
        print("Comparing camera 2 physics vs golden...")
        for target_angle in keyframe_angles_deg:
            print(f"Comparing camera 2 at {target_angle}°")
            results = compare_arrays_within_tolerances(
                golden_images_2[target_angle],
                physics_images_2[target_angle],
                allclose_rtol=None,
                allclose_atol=None,
                mean_tolerance=10,
                print_all_stats=True,
            )
            self.assertTrue(
                results["passed"],
                f"Camera 2 image comparison failed at {target_angle}°: {results}",
            )

    async def test_dual_camera_moving_cube(self):
        """Two co-located cameras must produce matching images of a laterally moving cube.

        Both cameras share the same position and orientation. A cube is placed
        in front of them and teleported 0.5 m laterally each frame for 30 frames.
        Images from both cameras are collected via ROS2, matched by their
        simulation-time timestamps, and compared to verify identical output.
        """
        save_debug_images = False
        num_frames = 10
        cube_travel_distance = 4.0
        cube_step_m = cube_travel_distance / num_frames
        width, height = 640, 360
        camera_height = 0.5
        camera_pos = [0.0, 0.0, camera_height]
        cube_distance = 10.0
        cube_start_y = -(cube_travel_distance / 2.0)

        debug_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "data", "dual_camera_moving_cube", "debug"
        )

        scene_path = "/Isaac/Environments/Grid/default_environment.usd"
        await open_stage_async(self._assets_root_path + scene_path)
        await omni.kit.app.get_app().next_update_async()

        # Two cameras at the exact same world pose
        camera_path_1 = "/World/Camera1"
        camera_path_2 = "/World/Camera2"
        stage_utils.define_prim(camera_path_1, type_name="Camera")
        stage_utils.define_prim(camera_path_2, type_name="Camera")

        cam1_xform = XformPrim(camera_path_1, reset_xform_op_properties=True)
        cam2_xform = XformPrim(camera_path_2, reset_xform_op_properties=True)

        cam_orientation = _camera_orientation_at_angle_deg(0.0)
        cam1_xform.set_world_poses(positions=[camera_pos], orientations=[cam_orientation])
        cam2_xform.set_world_poses(positions=[camera_pos], orientations=[cam_orientation])
        await omni.kit.app.get_app().next_update_async()

        cube = VisualCuboid("/World/MovingCube", position=[cube_distance, cube_start_y, camera_height])
        cube_xform = XformPrim("/World/MovingCube")
        await omni.kit.app.get_app().next_update_async()

        _create_rgb_camera_graph("/ActionGraph1", camera_path_1, "dual_cam_1_rgb", width, height)
        _create_rgb_camera_graph("/ActionGraph2", camera_path_2, "dual_cam_2_rgb", width, height)
        await omni.kit.app.get_app().next_update_async()

        image_buffer_1 = []
        image_buffer_2 = []

        def rgb_callback_1(data):
            ts = data.header.stamp.sec + data.header.stamp.nanosec / 1e9
            img = ros2_image_to_buffer(data, normalize_color_order=True, squeeze_singleton_channel=True, copy=True)
            image_buffer_1.append((ts, img))

        def rgb_callback_2(data):
            ts = data.header.stamp.sec + data.header.stamp.nanosec / 1e9
            img = ros2_image_to_buffer(data, normalize_color_order=True, squeeze_singleton_channel=True, copy=True)
            image_buffer_2.append((ts, img))

        node = self.create_node("dual_camera_test_node")
        self.start_async_spinning(node)
        self.create_subscription(node, Image, "dual_cam_1_rgb", rgb_callback_1, get_qos_profile(depth=num_frames + 40))
        self.create_subscription(node, Image, "dual_cam_2_rgb", rgb_callback_2, get_qos_profile(depth=num_frames + 40))

        self._timeline.play()
        await omni.kit.app.get_app().next_update_async()
        await omni.kit.app.get_app().next_update_async()

        # Wait for both render pipelines to start producing images
        await self.simulate_until_condition(
            lambda: len(image_buffer_1) > 0 and len(image_buffer_2) > 0,
        )

        image_buffer_1.clear()
        image_buffer_2.clear()
        capture_start_time = SimulationManager.get_simulation_time()
        print(f"Buffers cleared at sim_time={capture_start_time:.6f}s")

        # Move the cube 0.5 m laterally each frame
        print(f"Moving cube across {num_frames} frames ({cube_step_m} m/frame)...")
        for frame_idx in range(num_frames):
            cube_y = cube_start_y + frame_idx * cube_step_m
            cube_xform.set_world_poses(positions=[[cube_distance, cube_y, camera_height]])
            await omni.kit.app.get_app().next_update_async()

        # Extra frames so pipeline-delayed images flush through
        pipeline_drain_frames = 30
        for _ in range(pipeline_drain_frames):
            await omni.kit.app.get_app().next_update_async()

        self.stop_async_spinning(node)
        self._timeline.stop()
        await omni.kit.app.get_app().next_update_async()

        print(f"Buffered {len(image_buffer_1)} cam1 and {len(image_buffer_2)} cam2 images (raw).")

        # Keep only images with timestamps after the buffer-clear point
        image_buffer_1 = [(ts, img) for ts, img in image_buffer_1 if ts >= capture_start_time]
        image_buffer_2 = [(ts, img) for ts, img in image_buffer_2 if ts >= capture_start_time]
        print(
            f"After filtering (ts >= {capture_start_time:.6f}s): "
            f"{len(image_buffer_1)} cam1 and {len(image_buffer_2)} cam2 images."
        )

        self.assertGreater(len(image_buffer_1), 0, "No cam1 images after capture_start_time")
        self.assertGreater(len(image_buffer_2), 0, "No cam2 images after capture_start_time")

        if save_debug_images:
            if os.path.isdir(debug_dir):
                shutil.rmtree(debug_dir)
            os.makedirs(debug_dir)

        # Save all debug images first (both cameras, every frame)
        if save_debug_images:
            for idx, (ts, img) in enumerate(image_buffer_1):
                save_rgb_image(img, debug_dir, f"cam1_{idx:03d}_ts_{ts:.6f}.png")
            for idx, (ts, img) in enumerate(image_buffer_2):
                save_rgb_image(img, debug_dir, f"cam2_{idx:03d}_ts_{ts:.6f}.png")
            print(f"Saved {len(image_buffer_1)} cam1 + {len(image_buffer_2)} cam2 " f"debug images to {debug_dir}")

        # Index cam2 images by timestamp for exact matching
        cam2_by_ts = {ts: img for ts, img in image_buffer_2}

        matched_pairs = 0
        comparison_failures = []

        for ts1, img1 in image_buffer_1:
            self.assertIn(ts1, cam2_by_ts, f"cam1 timestamp {ts1:.6f}s has no exact match in cam2")
            # if ts1 not in cam2_by_ts:
            #     print(f"Skipping cam1 ts={ts1:.6f}s (no matching cam2 image)")
            #     continue
            img2 = cam2_by_ts[ts1]

            matched_pairs += 1
            print(f"Pair {matched_pairs}: ts={ts1:.6f}s")

            results = compare_arrays_within_tolerances(
                img1,
                img2,
                allclose_rtol=None,
                allclose_atol=None,
                mean_tolerance=10,
                print_all_stats=True,
            )
            if not results["passed"]:
                comparison_failures.append((ts1, results))

        print(f"Compared {matched_pairs} image pairs with identical timestamps")

        self.assertGreater(matched_pairs, 0, "No image pairs with matching timestamps")
        self.assertEqual(
            len(comparison_failures),
            0,
            f"{len(comparison_failures)} of {matched_pairs} pairs failed image comparison: "
            f"{comparison_failures[0][1] if comparison_failures else 'N/A'}",
        )
