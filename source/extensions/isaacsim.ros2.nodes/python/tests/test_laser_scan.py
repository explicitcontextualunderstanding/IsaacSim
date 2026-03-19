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

import copy

import carb
import omni.kit.commands
import omni.kit.test
import omni.kit.usd
import usdrt.Sdf
from isaacsim.core.utils.physics import simulate_async
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.ros2.core.impl.ros2_test_case import ROS2TestCase
from pxr import Gf, Sdf

from .common import get_qos_profile


class TestRos2LaserScan(ROS2TestCase):
    async def test_physx_lidar(self):
        # Test Lidar buffer with replicator activated
        import omni.graph.core as og
        import rclpy
        from sensor_msgs.msg import LaserScan

        stage = add_reference_to_stage(
            usd_path=self._assets_root_path + "/Isaac/Environments/Simple_Room/simple_room.usd", prim_path="/World"
        )

        await omni.kit.app.get_app().next_update_async()
        # Make sure the stage loaded
        stage = omni.usd.get_context().get_stage()
        await omni.kit.app.get_app().next_update_async()

        # Add lidar
        result, lidar = omni.kit.commands.execute(
            "RangeSensorCreateLidar",
            path="/World/Lidar",
            parent=None,
            min_range=0.4,
            max_range=100.0,
            draw_points=True,
            draw_lines=True,
            horizontal_fov=360.0,
            vertical_fov=30.0,
            horizontal_resolution=0.4,
            vertical_resolution=4.0,
            rotation_rate=0.0,
            high_lod=False,
            yaw_offset=0.0,
            enable_semantics=True,
        )
        await omni.kit.app.get_app().next_update_async()
        lidarPath = str(lidar.GetPath())
        lidar.GetPrim().GetAttribute("xformOp:translate").Set(Gf.Vec3d(0.0, -0.5, 0.5))

        # Setup a camera_helper to activate replicator to test PhysX Lidar buffer
        viewport_api = omni.kit.viewport.utility.get_active_viewport()
        render_product_path = viewport_api.get_render_product_path()
        # Set viewport resolution, changes will occur on next frame
        viewport_api.set_texture_resolution((1280, 720))
        await omni.kit.app.get_app().next_update_async()

        try:
            keys = og.Controller.Keys
            (graph, nodes, _, _) = og.Controller.edit(
                {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
                {
                    keys.CREATE_NODES: [
                        ("OnTick", "omni.graph.action.OnTick"),
                        ("LidarLaserScanNode", "isaacsim.sensors.physx.IsaacReadLidarBeams"),
                        ("LaserScanPublisher", "isaacsim.ros2.bridge.ROS2PublishLaserScan"),
                        ("CameraHelper", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        ("LidarLaserScanNode.inputs:lidarPrim", [usdrt.Sdf.Path(lidarPath)]),
                        ("CameraHelper.inputs:renderProductPath", render_product_path),
                        ("LaserScanPublisher.inputs:timeStamp", 1.0),  # Use a non zero timestamp for test
                    ],
                    keys.CONNECT: [
                        ("OnTick.outputs:tick", "LidarLaserScanNode.inputs:execIn"),
                        ("OnTick.outputs:tick", "CameraHelper.inputs:execIn"),
                        ("LidarLaserScanNode.outputs:execOut", "LaserScanPublisher.inputs:execIn"),
                        ("LidarLaserScanNode.outputs:azimuthRange", "LaserScanPublisher.inputs:azimuthRange"),
                        ("LidarLaserScanNode.outputs:depthRange", "LaserScanPublisher.inputs:depthRange"),
                        ("LidarLaserScanNode.outputs:horizontalFov", "LaserScanPublisher.inputs:horizontalFov"),
                        (
                            "LidarLaserScanNode.outputs:horizontalResolution",
                            "LaserScanPublisher.inputs:horizontalResolution",
                        ),
                        ("LidarLaserScanNode.outputs:intensitiesData", "LaserScanPublisher.inputs:intensitiesData"),
                        ("LidarLaserScanNode.outputs:linearDepthData", "LaserScanPublisher.inputs:linearDepthData"),
                        ("LidarLaserScanNode.outputs:numCols", "LaserScanPublisher.inputs:numCols"),
                        ("LidarLaserScanNode.outputs:numRows", "LaserScanPublisher.inputs:numRows"),
                        ("LidarLaserScanNode.outputs:rotationRate", "LaserScanPublisher.inputs:rotationRate"),
                    ],
                },
            )
        except Exception as e:
            print(e)

        self._lidar_data = None
        self._lidar_data_prev = None
        self._lidar_callback_count = 0

        def lidar_callback(data: LaserScan):
            self._lidar_callback_count += 1
            self._lidar_data = data
            self.assertGreater(data.angle_max, data.angle_min)
            self.assertEqual(len(data.intensities), 900)
            if self._lidar_data_prev is not None:
                self.assertEqual(data.intensities, self._lidar_data_prev.intensities)
                self.assertEqual(data.ranges, self._lidar_data_prev.ranges)
            self._lidar_data_prev = copy.deepcopy(data)

        node = self.create_node("lidar_tester")
        subscriber = self.create_subscription(node, LaserScan, "scan", lidar_callback, get_qos_profile())

        omni.kit.commands.execute(
            "ChangeProperty", prop_path=Sdf.Path(lidarPath + ".rotation_rate"), value=0.0, prev=None
        )

        def spin():
            rclpy.spin_once(node, timeout_sec=0.01)

        # 0.0 Hz Lidar rotation
        self._lidar_callback_count = 0
        self._timeline.play()
        await simulate_async(0.5, 60, spin)
        self.assertGreater(self._lidar_callback_count, 0)
        self.assertEqual(self._lidar_data.time_increment, 0)

        self._timeline.stop()
        await omni.kit.app.get_app().next_update_async()

        # 21.0 Hz Lidar rotation
        omni.kit.commands.execute(
            "ChangeProperty",
            prop_path=Sdf.Path(lidarPath + ".rotationRate"),
            value=21.0,
            prev=None,
        )

        # Clear previous messages
        await omni.kit.app.get_app().next_update_async()
        spin()
        self._lidar_callback_count = 0
        self._lidar_data = None
        self._lidar_data_prev = None
        await omni.kit.app.get_app().next_update_async()

        self._timeline.play()
        await simulate_async(0.5, 60, spin)
        self.assertGreater(self._lidar_callback_count, 0)
        self.assertGreater(self._lidar_data.time_increment, 0.0)

        self._timeline.stop()
        await omni.kit.app.get_app().next_update_async()

        prev_time_increment = self._lidar_data.time_increment

        # 201.0 Hz Lidar rotation
        omni.kit.commands.execute(
            "ChangeProperty",
            prop_path=Sdf.Path(lidarPath + ".rotationRate"),
            value=201.0,
            prev=None,
        )

        # Clear previous messages
        await omni.kit.app.get_app().next_update_async()
        spin()
        self._lidar_callback_count = 0
        self._lidar_data = None
        self._lidar_data_prev = None
        await omni.kit.app.get_app().next_update_async()

        self._timeline.play()
        await simulate_async(0.5, 60, spin)
        self.assertGreater(self._lidar_callback_count, 0)
        self.assertGreater(prev_time_increment, self._lidar_data.time_increment)

        self._timeline.stop()
        spin()

        pass
