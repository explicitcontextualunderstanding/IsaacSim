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

"""Tests for RTX sensor annotators including generic model output, lidar scan buffer, lidar flat scan, and radar point cloud functionality."""


import asyncio
import os

import carb
import isaacsim.sensors.rtx.generic_model_output as gmo_utils
import matplotlib.pyplot as plt
import numpy as np
import omni.kit.test
import omni.replicator.core as rep
from isaacsim.core.utils.stage import create_new_stage_async, update_stage_async
from isaacsim.sensors.rtx import (
    LidarRtx,
    get_gmo_data,
)
from pxr import Gf

from .common import create_sarcophagus

DEBUG_DRAW_PRINT = True

DEFAULT_CONFIG = "Example_Rotary"  # Default configuration for tests
DEFAULT_VARIANT = None  # Default variant for tests
MAX_TIMESTAMP_DIFF = 3500  # Maximum difference in fireTimeNs for DEFAULT_CONFIG configuration
NEAR_EDGE_THRESHOLD = 0.5  # Threshold for near edge returns in degrees


class TestGenericModelOutput(omni.kit.test.AsyncTestCase):
    """Test the Generic Model Output annotator"""

    async def setUp(self):
        """Setup test environment with a cube and lidar"""
        await create_new_stage_async()
        await update_stage_async()

        # Ordering octants in binary order, such that octant 0 is +++, octant 1 is ++-, etc. for XYZ.
        self._octant_dimensions = [
            (10, 10, 5),
            (10, 10, 7),
            (25, 25, 17),
            (25, 25, 19),
            (15, 15, 9),
            (15, 15, 11),
            (20, 20, 13),
            (20, 20, 15),
        ]
        self.cube_info = create_sarcophagus()

        self._timeline = omni.timeline.get_timeline_interface()
        self.hydra_texture = None
        self._annotator = rep.AnnotatorRegistry.get_annotator("GenericModelOutput")
        self._annotator_data = None

    async def tearDown(self):
        """Clean up test environment and stop timeline."""
        self._timeline.stop()
        self._annotator.detach()
        self.hydra_texture.destroy()
        self.hydra_texture = None
        await omni.kit.app.get_app().next_update_async()
        while omni.usd.get_context().get_stage_loading_status()[2] > 0:
            print("tearDown, assets still loading, waiting to finish...")
            await asyncio.sleep(1.0)
        await update_stage_async()

    async def _test_point_cloud(self):
        """Tests sensor returns stored in GMO buffer against expected range."""
        # NOTE: if an element of unit_vecs is 0, indicating the return vector is parallel to the plane, the result of np.divide will be inf
        # Suppress the error
        np.seterr(divide="ignore")
        # We have spherical coordinates, so convert to cartesian unit vectors and re-normalize
        unit_vecs = np.concatenate(
            [
                np.cos(np.radians(self.azimuth))[..., None],
                np.sin(np.radians(self.azimuth))[..., None],
                np.sin(np.radians(self.elevation))[..., None],
            ],
            axis=1,
        )
        unit_vecs = unit_vecs / np.linalg.norm(unit_vecs, axis=1, keepdims=True)
        # Get octant dimensions and indices
        octant = (unit_vecs[:, 0] < 0) * 4 + (unit_vecs[:, 1] < 0) * 2 + (unit_vecs[:, 2] < 0)
        dims = np.array([self._octant_dimensions[o] for o in octant])

        # Let alpha be the angle between the normal to the plane and the return vector
        # Let the distance from the origin of the return vector along the normal vector to the plane be l =  dims(idx)
        # Let expected range R be the distance along the return vector to the point of intersection with the plane
        # Then, cos(alpha) = l / R
        # Next, observe cos(alpha) = n-hat dot r-hat, where n-hat is the unit normal vector to the plane
        # and r-hat is the unit return vector
        # Therefore, R = l / (n-hat dot r-hat)
        # n-hat dot r-hat is simply the index of the unit return vector corresponding to the plane
        # This simplifies the computation of expected range to elementwise-division of the dimensions by the unit return vectors
        # The minimum of these values is the expected range to the first plane the return vector will intersect
        expected_range = np.min(np.divide(dims, np.abs(unit_vecs)), axis=1)
        # Compute index of the plane that the return vector intersects first for later use
        plane_idx = np.argmin(np.divide(dims, np.abs(unit_vecs)), axis=1)

        if DEBUG_DRAW_PRINT:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            x_plot = np.multiply(unit_vecs[:, 0], self.distance)
            y_plot = np.multiply(unit_vecs[:, 1], self.distance)
            z_plot = np.multiply(unit_vecs[:, 2], self.distance)
            x_expected = np.multiply(unit_vecs[:, 0], expected_range)
            y_expected = np.multiply(unit_vecs[:, 1], expected_range)
            z_expected = np.multiply(unit_vecs[:, 2], expected_range)
            ax.scatter(x_plot, y_plot, z_plot, c="b")
            ax.scatter(x_expected, y_expected, z_expected, c="r")
            plt.savefig(f"test_returns_cartesian.png")
            plt.close()

        # Compute percent differences
        percent_diffs = np.divide(np.abs(expected_range - self.distance), expected_range)

        # Exclude returns that are within 0.5deg of an octant edge or corner
        near_edge = np.full(self.azimuth.shape, False)
        for excl_az in np.arange(-180, 181, 45):
            near_edge = np.logical_or(near_edge, np.abs(self.azimuth - excl_az) < NEAR_EDGE_THRESHOLD)
        self._not_near_edge = np.logical_not(near_edge)

        # Compute the number of returns that exceed the threshold of 2%
        num_exceeding_threshold = np.sum(np.logical_and(percent_diffs > 2e-2, np.array(self._not_near_edge)))
        num_returns = np.size(self.azimuth)
        carb.log_warn(f"num_returns: {num_returns}")
        pct_exceeding_threshold = num_exceeding_threshold / num_returns * 100
        valid_threshold = 1.0 if num_returns >= 100 else 10.0
        self.assertLessEqual(
            pct_exceeding_threshold,
            valid_threshold,
            f"Expected fewer than 1% of returns to differ from expected range by more than 2%. {num_exceeding_threshold} of {num_returns} returns exceeded threshold.",
        )

        # Determine the cube which was struck by the return vector.
        # The first check is which octant the return vector is in. Note the sequence is based on the sequence of
        # octants as defined in the test setup.
        cube_idx = np.zeros(octant.shape, dtype=int)
        cube_idx[octant == 0] = 0
        cube_idx[octant == 1] = 0
        cube_idx[octant == 2] = 3
        cube_idx[octant == 3] = 3
        cube_idx[octant == 4] = 1
        cube_idx[octant == 5] = 1
        cube_idx[octant == 6] = 2
        cube_idx[octant == 7] = 2
        # Next, multiply the cube index by 4 to get which iteration of the test setup we're in, then add the plane index
        # to select if the return vector struck the x-normal face, y-normal face, or one of the z-normal faces.
        cube_idx = cube_idx * 4 + plane_idx
        # For odd octants (z < 0), the z-normal face is the bottom face, so add 1 to the cube index.
        cube_idx[np.bitwise_and(octant % 2 == 1, plane_idx == 2)] += 1
        # Finally, convert the cube index to the prim path of the cube.
        self.cube_prim_paths = [f"/World/cube_{int(i)}" for i in cube_idx]

    async def _test_intensity(self):
        """Tests that intensity values are non-negative."""
        self.assertTrue(np.all(self.intensity >= 0), "Intensities are not non-negative.")
        pass

    async def _test_timestamp(self):
        """Tests that timestamps are monotonically increasing and within expected range."""
        timestamp_diffs = np.diff(self.timestamp)
        self.assertTrue(np.all(timestamp_diffs >= 0), "Timestamps are not monotonically increasing.")
        max_timestamp_diff = np.max(timestamp_diffs)
        self.assertTrue(
            max_timestamp_diff <= MAX_TIMESTAMP_DIFF,
            f"Max difference in timestamps {max_timestamp_diff}ns > {MAX_TIMESTAMP_DIFF}ns, the maximum difference in fireTimeNs in the Example_Rotary configuration.",
        )

    async def _test_emitter_id(self):
        """Tests that emitter IDs are non-negative and within expected range."""
        self.assertTrue(np.all(self.emitterId >= 0), "Emitter IDs are not non-negative.")
        self.assertTrue(np.all(self.emitterId < 1024), "Emitter IDs are expected to be less than 1024.")

    async def _test_channel_id(self):
        """Tests that channel IDs are non-negative and within expected range."""
        self.assertTrue(np.all(self.channelId >= 0), "Channel IDs are not non-negative.")
        self.assertTrue(np.all(self.channelId < 1024), "Channel IDs are expected to be less than 1024.")

    async def _test_material_id(self):
        """Tests that material IDs match expected values for cube prim paths."""
        self.assertEqual(
            len(self.materialId),
            len(self.cube_prim_paths),
            "Expected same number of material ids as number of valid returns.",
        )

        failure_count = 0
        checked_count = 0
        for i, (material_id, prim_path) in enumerate(zip(self.materialId, self.cube_prim_paths)):
            if not self._not_near_edge[i]:
                continue
            checked_count += 1
            expected_material_id = self.cube_info[prim_path]["material_id"]
            if material_id != expected_material_id:
                failure_count += 1

        failure_pct = (failure_count / checked_count * 100) if checked_count > 0 else 0
        self.assertLess(
            failure_pct,
            1.0,
            f"Expected fewer than 1% of returns to fail material ID check. "
            f"{failure_count} of {checked_count} returns ({failure_pct:.2f}%) failed.",
        )

    async def _test_tick_id(self):
        """Tests tick ID values."""
        pass

    async def _test_hit_normal(self):
        """Tests hit normal values from GMO data."""
        # TODO: Implement hitNormal validation
        pass

    async def _test_velocity(self):
        """Validates that velocity values are near zero for stationary objects.

        Prints the maximum absolute velocity and asserts all velocities are close to zero with a tolerance of 5e-3.
        """
        print(np.max(np.abs(self.velocity)))
        self.assertTrue(np.allclose(self.velocity, 0, atol=5e-3), "Velocities are expected to be 0.")

    async def _test_object_id(self):
        """Validates object ID mapping and consistency with cube prim paths.

        Decodes the stable ID mapping, validates object IDs exist in the mapping, and checks that object IDs
        correctly correspond to the expected cube prim paths. Skips returns near octant edges and requires
        less than 1% failure rate.
        """
        stable_id_map = LidarRtx.decode_stable_id_mapping(self._annotator_stable_id_map_data.tobytes())
        self.assertGreater(len(stable_id_map), 0, "Expected non-empty stable id map.")
        object_ids = LidarRtx.get_object_ids(self.objectId)
        self.assertEqual(
            len(object_ids), len(self.cube_prim_paths), "Expected same number of object ids as number of valid returns."
        )

        unexpected_object_ids = set(object_ids) - stable_id_map.keys()
        self.assertFalse(
            len(unexpected_object_ids) > 0,
            f"Expected no unexpected object ids. Unexpected object ids: {unexpected_object_ids}",
        )

        failure_count = 0
        checked_count = 0
        for i, object_id in enumerate(object_ids):
            if not self._not_near_edge[i]:
                # Skip returns that are within 0.5deg of an octant edge or corner
                continue
            checked_count += 1
            stable_id = stable_id_map[object_id]
            if stable_id != self.cube_prim_paths[i]:
                failure_count += 1

        failure_pct = (failure_count / checked_count * 100) if checked_count > 0 else 0
        self.assertLess(
            failure_pct,
            1.0,
            f"Expected fewer than 1% of returns to fail object ID check. "
            f"{failure_count} of {checked_count} returns ({failure_pct:.2f}%) failed.",
        )

    async def _test_echo_id(self):
        """Validates that all echo IDs are zero as expected."""
        self.assertTrue(np.all(self.echoId == 0), "Echo IDs are expected to be 0.")
        pass

    async def _test_tick_state(self):
        """Validates that all tick states are zero as expected."""
        self.assertTrue(np.all(self.tickState == 0), "Tick states are expected to be 0.")
        pass

    async def _test_annotator_outputs(self, config: str = DEFAULT_CONFIG, variant: str = DEFAULT_VARIANT):
        """Creates RTX Lidar sensor and validates all Generic Model Output annotator data.

        Creates an OmniLidar prim with specified configuration, attaches GenericModelOutput and StableIdMap
        annotators, renders frames until valid data is obtained, then extracts and validates all GMO fields
        including point cloud accuracy, intensity, timestamps, and object/material IDs.

        Args:
            config: RTX Lidar configuration name.
            variant: RTX Lidar variant name.
        """
        # Create sensor prim
        kwargs = {
            "path": "lidar",
            "parent": None,
            "translation": Gf.Vec3d(0.0, 0.0, 0.0),
            "orientation": Gf.Quatd(1.0, 0.0, 0.0, 0.0),
            "config": config,
            "variant": variant,
            "omni:sensor:Core:outputFrameOfReference": "WORLD",
            "omni:sensor:Core:auxOutputType": "FULL",
            "omni:sensor:Core:skipDroppingInvalidPoints": False,
        }

        _, self.sensor = omni.kit.commands.execute(f"IsaacSensorCreateRtxLidar", **kwargs)
        sensor_type = self.sensor.GetTypeName()
        self.assertEqual(
            sensor_type, "OmniLidar", f"Expected OmniLidar prim, got {sensor_type}. Was sensor prim created?"
        )

        # Create render product and attach to sensor
        self.hydra_texture = rep.create.render_product(
            self.sensor.GetPath(),
            [32, 32],
            name="RtxSensorRenderProduct",
            render_vars=["GenericModelOutput", "RtxSensorMetadata"],
        )
        # Attach annotator to render product
        self._annotator.attach([self.hydra_texture.path])

        # Attach StableIdMap annotator to OmniLidar prim to pick up object IDs
        self._annotator_stable_id_map = rep.AnnotatorRegistry.get_annotator("StableIdMap")
        self._annotator_stable_id_map.attach([self.hydra_texture.path])

        # Render frames until we get valid data, or until we've rendered the maximum number of frames
        self._timeline.play()
        for _ in range(10):
            # Wait for a single frame
            await omni.kit.app.get_app().next_update_async()
            self._annotator_data = self._annotator.get_data()
            # Test that the annotator data is not empty
            if self._annotator_data is None or self._annotator_data.size == 0:
                continue

            gmo = get_gmo_data(self._annotator_data)
            self._annotator_stable_id_map_data = self._annotator_stable_id_map.get_data()
            # Test that the GMO magic number is correct
            if gmo.magicNumber != gmo_utils.getMagicNumberGMO():
                continue
        self._timeline.stop()

        # Test that all expected keys are present in the annotator data, then copy those values to the test class attributes
        self.azimuth = gmo.x.copy()
        self.elevation = gmo.y.copy()
        self.distance = gmo.z.copy()
        self.intensity = gmo.scalar.copy()
        self.flags = gmo.flags.copy()
        self.timestamp = gmo.timeOffsetNs.copy()
        self.emitterId = gmo.emitterId.copy()
        self.channelId = gmo.channelId.copy()
        self.materialId = gmo.matId.copy()
        self.tickId = gmo.tickId.copy()
        self.hitNormal = gmo.hitNormals.copy()
        self.velocity = gmo.velocities.copy()
        self.objectId = gmo.objId.copy()
        self.echoId = gmo.echoId.copy()
        self.tickState = gmo.tickStates.copy()

        # Test point cloud data shape
        num_points = self.azimuth.shape[0]
        self.assertGreater(num_points, 0, "Expected non-empty azimuth data.")

        # Select only valid points
        valid = np.bitwise_and(self.flags, 64) == 64
        self.azimuth = self.azimuth[valid]
        self.elevation = self.elevation[valid]
        self.distance = self.distance[valid]
        self.intensity = self.intensity[valid]
        self.timestamp = self.timestamp[valid]
        self.emitterId = self.emitterId[valid]
        self.channelId = self.channelId[valid]
        self.materialId = self.materialId[valid]
        self.tickId = self.tickId[valid]
        self.hitNormal = self.hitNormal[np.repeat(valid, 3)].reshape(-1, 3)
        self.velocity = self.velocity[np.repeat(valid, 3)].reshape(-1, 3)
        self.objectId = self.objectId[np.repeat(valid, 16)]
        self.echoId = self.echoId[valid]
        self.tickState = self.tickState[valid]

        await self._test_point_cloud()
        await self._test_intensity()
        await self._test_timestamp()
        await self._test_emitter_id()
        await self._test_channel_id()
        await self._test_material_id()
        await self._test_tick_id()
        await self._test_hit_normal()
        await self._test_velocity()
        await self._test_object_id()
        await self._test_echo_id()
        await self._test_tick_state()

    async def _soak_annotator(self, config: str = DEFAULT_CONFIG, variant: str = DEFAULT_VARIANT):
        """Performs sustained testing of the Generic Model Output annotator over multiple frames.

        Creates an RTX Lidar sensor and runs extended validation including frame data consistency,
        magic number verification, unique data per frame checks, and complete scan azimuth coverage testing.
        Validates that scan data has no gaps and covers the full -180 to +180 degree range.

        Args:
            config: RTX Lidar configuration name.
            variant: RTX Lidar variant name.
        """
        # Create sensor prim
        kwargs = {
            "path": "lidar",
            "parent": None,
            "translation": Gf.Vec3d(0.0, 0.0, 0.0),
            "orientation": Gf.Quatd(1.0, 0.0, 0.0, 0.0),
            "config": config,
            "variant": variant,
            "omni:sensor:Core:outputFrameOfReference": "WORLD",
            "omni:sensor:Core:auxOutputType": "FULL",
            "omni:sensor:Core:skipDroppingInvalidPoints": False,
        }

        _, self.sensor = omni.kit.commands.execute(f"IsaacSensorCreateRtxLidar", **kwargs)
        sensor_type = self.sensor.GetTypeName()
        self.assertEqual(
            sensor_type, "OmniLidar", f"Expected OmniLidar prim, got {sensor_type}. Was sensor prim created?"
        )
        scan_rate_base_hz = self.sensor.GetAttribute("omni:sensor:Core:scanRateBaseHz").Get()

        # Create render product and attach to sensor
        self.hydra_texture = rep.create.render_product(
            self.sensor.GetPath(),
            [32, 32],
            name="RtxSensorRenderProduct",
            render_vars=["GenericModelOutput", "RtxSensorMetadata"],
        )
        # Attach annotator to render product
        self._annotator.initialize()
        self._annotator.attach([self.hydra_texture.path])

        NUM_FRAMES = 100
        WARMUP_FRAMES = 3
        MAX_BAD_MAGIC_NUMBER_FRAME_COUNT = 1

        data_frame_count = 0
        empty_frame_count = 0
        bad_magic_number_frame_count = 0
        scan_azimuth = np.array([], dtype=np.float32)
        num_concatenations = 0
        last_gmo_buffer = None

        self._timeline.play()
        for _ in range(NUM_FRAMES):
            # Wait for a single frame
            await omni.kit.app.get_app().next_update_async()
            self._annotator_data = self._annotator.get_data()
            # Test that the annotator data is not empty
            if self._annotator_data is None or self._annotator_data.size == 0:
                empty_frame_count += 1
                self.assertLessEqual(
                    empty_frame_count, WARMUP_FRAMES, f"Expected at most {WARMUP_FRAMES} empty frames."
                )
                continue

            gmo = get_gmo_data(self._annotator_data)
            # Test that the GMO magic number is correct
            if gmo.magicNumber != gmo_utils.getMagicNumberGMO():
                bad_magic_number_frame_count += 1
                self.assertLessEqual(
                    bad_magic_number_frame_count,
                    MAX_BAD_MAGIC_NUMBER_FRAME_COUNT,
                    f"Expected at most {MAX_BAD_MAGIC_NUMBER_FRAME_COUNT} bad magic number frames.",
                )
                continue

            # Test that the GMO data is unique per frame
            if last_gmo_buffer is not None and last_gmo_buffer.shape == self._annotator_data.shape:
                self.assertFalse(
                    np.all(last_gmo_buffer == self._annotator_data),
                    "GMO data is the same as the last frame. Expected unique data per frame.",
                )
            last_gmo_buffer = self._annotator_data
            data_frame_count += 1
            # Concatenate the GMO x values to form a complete scan
            scan_azimuth = np.concatenate((scan_azimuth, gmo.x), axis=0)
            num_concatenations += 1
            if num_concatenations == 60 / scan_rate_base_hz:
                # Test that the complete scan has no gaps
                sorted_azimuth = np.sort(scan_azimuth)
                azimuth_diffs = np.diff(sorted_azimuth)
                self.assertAlmostEqual(np.min(sorted_azimuth), -180.0, delta=5e-3)
                self.assertAlmostEqual(np.max(sorted_azimuth), 180.0, delta=5e-3)
                self.assertAlmostEqual(np.min(azimuth_diffs), 0.0)
                # self.assertLessEqual(np.max(azimuth_diffs), 0.05)
                # Reset the scan azimuth buffer
                scan_azimuth = np.array([], dtype=np.float32)
                num_concatenations = 0
        self._timeline.stop()

        expected_valid_frames = NUM_FRAMES - WARMUP_FRAMES
        self.assertGreaterEqual(data_frame_count, expected_valid_frames)

    async def test_3d_lidar(self):
        """Tests Generic Model Output annotator with Example_Rotary 3D LiDAR configuration."""
        await self._test_annotator_outputs(config="Example_Rotary", variant=None)

    async def test_3d_lidar_soak(self):
        """Performs sustained testing of Generic Model Output annotator with Example_Rotary 3D LiDAR."""
        await self._soak_annotator(config="Example_Rotary", variant=None)

    async def _test_timestamp_alignment(self, sensor_type: str):
        """Test that GMO timestampNs aligns with timeline time and pausing prevents new data.

        Args:
            sensor_type: Either "lidar" or "radar".
        """
        # Create sensor prim based on type
        if sensor_type == "lidar":
            kwargs = {
                "path": "lidar",
                "parent": None,
                "translation": Gf.Vec3d(0.0, 0.0, 0.0),
                "orientation": Gf.Quatd(1.0, 0.0, 0.0, 0.0),
                "config": DEFAULT_CONFIG,
                "variant": DEFAULT_VARIANT,
                "omni:sensor:Core:outputFrameOfReference": "WORLD",
                "omni:sensor:Core:auxOutputType": "FULL",
                "omni:sensor:Core:skipDroppingInvalidPoints": False,
            }
            _, self.sensor = omni.kit.commands.execute("IsaacSensorCreateRtxLidar", **kwargs)
            expected_prim_type = "OmniLidar"
        else:  # radar
            kwargs = {
                "path": "radar",
                "parent": None,
                "translation": Gf.Vec3d(0.0, 0.0, 0.0),
                "orientation": Gf.Quatd(1.0, 0.0, 0.0, 0.0),
                "omni:sensor:WpmDmat:outputFrameOfReference": "WORLD",
                "omni:sensor:WpmDmat:auxOutputType": "FULL",
            }
            _, self.sensor = omni.kit.commands.execute("IsaacSensorCreateRtxRadar", **kwargs)
            expected_prim_type = "OmniRadar"

        self.assertEqual(
            self.sensor.GetTypeName(),
            expected_prim_type,
            f"Expected {expected_prim_type} prim, got {self.sensor.GetTypeName()}",
        )

        # Create render product and attach to sensor
        self.hydra_texture = rep.create.render_product(
            self.sensor.GetPath(),
            [32, 32],
            name="RtxSensorRenderProduct",
            render_vars=["GenericModelOutput", "RtxSensorMetadata"],
        )
        self._annotator.initialize()
        self._annotator.attach([self.hydra_texture.path])

        WARMUP_FRAMES = 10
        NUM_FRAMES = 10
        PAUSE_FRAMES = 10
        GRACE_FRAMES = 3
        # Expected per-frame advance when playing: 1/60s in nanoseconds
        EXPECTED_ADVANCE_NS = int(1.0 / 60.0 * 1e9)  # 16_666_666 ns
        # Tolerance for timestamp delta (2ms in nanoseconds)
        TOLERANCE_NS = 2_000_000

        prev_gmo_timestamp_ns = None

        # Phase 1: Start timeline and wait for warmup, then verify timestamp advances by 1/60s
        self._timeline.play()

        # Wait for warmup
        for i in range(WARMUP_FRAMES):
            await omni.kit.app.get_app().next_update_async()
            timeline_time_ns = int(self._timeline.get_current_time() * 1e9)
            self._annotator_data = self._annotator.get_data()
            gmo_timestamp_ns = None
            if self._annotator_data is not None and self._annotator_data.size > 0:
                gmo = get_gmo_data(self._annotator_data)
                if gmo.magicNumber == gmo_utils.getMagicNumberGMO():
                    gmo_timestamp_ns = int(gmo.timestampNs)
                    prev_gmo_timestamp_ns = gmo_timestamp_ns
            print(f"[WARMUP  ] step {i:3d}  timeline_ns={timeline_time_ns:>10d}  gmo_ns={str(gmo_timestamp_ns):>10s}")

        self.assertIsNotNone(prev_gmo_timestamp_ns, "Expected at least one valid GMO frame during warmup")

        # Collect frames and verify GMO timestamp advances by 1/60s relative to previous
        for i in range(NUM_FRAMES):
            await omni.kit.app.get_app().next_update_async()
            timeline_time_ns = int(self._timeline.get_current_time() * 1e9)
            self._annotator_data = self._annotator.get_data()

            if self._annotator_data is None or self._annotator_data.size == 0:
                print(f"[PLAYING ] step {i:3d}  timeline_ns={timeline_time_ns:>10d}  gmo_ns={'None':>10s}  (no data)")
                continue

            gmo = get_gmo_data(self._annotator_data)
            if gmo.magicNumber != gmo_utils.getMagicNumberGMO():
                print(f"[PLAYING ] step {i:3d}  timeline_ns={timeline_time_ns:>10d}  gmo_ns={'None':>10s}  (bad magic)")
                continue

            gmo_timestamp_ns = int(gmo.timestampNs)
            delta_ns = gmo_timestamp_ns - prev_gmo_timestamp_ns
            print(
                f"[PLAYING ] step {i:3d}  timeline_ns={timeline_time_ns:>10d}  gmo_ns={gmo_timestamp_ns:>10d}  delta_ns={delta_ns:>10d}  expected={EXPECTED_ADVANCE_NS}"
            )

            self.assertAlmostEqual(
                delta_ns,
                EXPECTED_ADVANCE_NS,
                delta=TOLERANCE_NS,
                msg=f"Playing: GMO timestamp delta ({delta_ns} ns) != expected ({EXPECTED_ADVANCE_NS} ns). "
                f"prev={prev_gmo_timestamp_ns}, curr={gmo_timestamp_ns}",
            )
            prev_gmo_timestamp_ns = gmo_timestamp_ns

        # Phase 2: Pause the timeline and verify GMO timestamp does not advance (delta == 0)
        self._timeline.pause()

        for i in range(PAUSE_FRAMES):
            await omni.kit.app.get_app().next_update_async()
            timeline_time_ns = int(self._timeline.get_current_time() * 1e9)
            self._annotator_data = self._annotator.get_data()

            if self._annotator_data is None or self._annotator_data.size == 0:
                print(f"[PAUSED  ] step {i:3d}  timeline_ns={timeline_time_ns:>10d}  gmo_ns={'None':>10s}  (no data)")
                continue

            gmo = get_gmo_data(self._annotator_data)
            if gmo.magicNumber != gmo_utils.getMagicNumberGMO():
                print(f"[PAUSED  ] step {i:3d}  timeline_ns={timeline_time_ns:>10d}  gmo_ns={'None':>10s}  (bad magic)")
                continue

            gmo_timestamp_ns = int(gmo.timestampNs)
            delta_ns = gmo_timestamp_ns - prev_gmo_timestamp_ns

            if i < GRACE_FRAMES:
                print(
                    f"[PAUSED  ] step {i:3d}  timeline_ns={timeline_time_ns:>10d}  gmo_ns={gmo_timestamp_ns:>10d}  delta_ns={delta_ns:>10d}  (grace)"
                )
                prev_gmo_timestamp_ns = gmo_timestamp_ns
                continue

            print(
                f"[PAUSED  ] step {i:3d}  timeline_ns={timeline_time_ns:>10d}  gmo_ns={gmo_timestamp_ns:>10d}  delta_ns={delta_ns:>10d}  expected=0"
            )

            self.assertEqual(
                delta_ns,
                0,
                f"Paused: GMO timestamp should not advance (delta={delta_ns} ns). "
                f"prev={prev_gmo_timestamp_ns}, curr={gmo_timestamp_ns}",
            )

        # Phase 3: Resume the timeline and verify timestamp advances by 1/60s again
        self._timeline.play()

        for i in range(GRACE_FRAMES):
            await omni.kit.app.get_app().next_update_async()
            timeline_time_ns = int(self._timeline.get_current_time() * 1e9)
            self._annotator_data = self._annotator.get_data()

            gmo_timestamp_ns_str = "None"
            delta_ns_str = ""
            if self._annotator_data is not None and self._annotator_data.size > 0:
                gmo = get_gmo_data(self._annotator_data)
                if gmo.magicNumber == gmo_utils.getMagicNumberGMO():
                    gmo_timestamp_ns = int(gmo.timestampNs)
                    delta_ns = gmo_timestamp_ns - prev_gmo_timestamp_ns
                    gmo_timestamp_ns_str = f"{gmo_timestamp_ns:>10d}"
                    delta_ns_str = f"  delta_ns={delta_ns:>10d}"
                    prev_gmo_timestamp_ns = gmo_timestamp_ns

            print(
                f"[RESUMED ] step {i:3d}  timeline_ns={timeline_time_ns:>10d}  gmo_ns={gmo_timestamp_ns_str}  prev_gmo_ns={prev_gmo_timestamp_ns:>10d}  {delta_ns_str}    (grace)"
            )

        resumed_frames_collected = 0
        found_new_data_after_resume = False

        for i in range(NUM_FRAMES * 2):  # Allow more frames to find new data
            await omni.kit.app.get_app().next_update_async()
            timeline_time_ns = int(self._timeline.get_current_time() * 1e9)
            self._annotator_data = self._annotator.get_data()

            if self._annotator_data is None or self._annotator_data.size == 0:
                print(f"[RESUMED ] step {i:3d}  timeline_ns={timeline_time_ns:>10d}  gmo_ns={'None':>10s}  (no data)")
                continue

            gmo = get_gmo_data(self._annotator_data)
            if gmo.magicNumber != gmo_utils.getMagicNumberGMO():
                print(f"[RESUMED ] step {i:3d}  timeline_ns={timeline_time_ns:>10d}  gmo_ns={'None':>10s}  (bad magic)")
                continue

            gmo_timestamp_ns = int(gmo.timestampNs)
            delta_ns = gmo_timestamp_ns - prev_gmo_timestamp_ns

            # Wait for timestamp to actually advance (skip stale/cached frames)
            if not found_new_data_after_resume:
                if gmo_timestamp_ns <= prev_gmo_timestamp_ns:
                    print(
                        f"[RESUMED ] step {i:3d}  timeline_ns={timeline_time_ns:>10d}  gmo_ns={gmo_timestamp_ns:>10d}  delta_ns={delta_ns:>10d}  (stale)"
                    )
                    continue
                found_new_data_after_resume = True

            delta_ns = gmo_timestamp_ns - prev_gmo_timestamp_ns
            print(
                f"[RESUMED ] step {i:3d}  timeline_ns={timeline_time_ns:>10d}  gmo_ns={gmo_timestamp_ns:>10d}  delta_ns={delta_ns:>10d}  expected={EXPECTED_ADVANCE_NS}"
            )

            self.assertAlmostEqual(
                delta_ns,
                EXPECTED_ADVANCE_NS,
                delta=TOLERANCE_NS,
                msg=f"Resumed: GMO timestamp delta ({delta_ns} ns) != expected ({EXPECTED_ADVANCE_NS} ns). "
                f"prev={prev_gmo_timestamp_ns}, curr={gmo_timestamp_ns}",
            )
            prev_gmo_timestamp_ns = gmo_timestamp_ns
            resumed_frames_collected += 1

    async def test_lidar_timestamp_alignment(self):
        """Test that RTX Lidar GMO timestamps align with timeline and pause/resume behavior."""
        await self._test_timestamp_alignment("lidar")

    async def test_radar_timestamp_alignment(self):
        """Test that RTX Radar GMO timestamps align with timeline and pause/resume behavior."""
        await self._test_timestamp_alignment("radar")


class TestIsaacCreateRTXLidarScanBuffer(omni.kit.test.AsyncTestCase):
    """Test the Isaac Create RTX Lidar Scan Buffer annotator"""

    async def setUp(self):
        """Setup test environment with a cube and lidar"""
        await create_new_stage_async()
        await update_stage_async()

        # Ordering octants in binary order, such that octant 0 is +++, octant 1 is ++-, etc. for XYZ.
        self._octant_dimensions = [
            (10, 10, 5),
            (10, 10, 7),
            (25, 25, 17),
            (25, 25, 19),
            (15, 15, 9),
            (15, 15, 11),
            (20, 20, 13),
            (20, 20, 15),
        ]
        self.cube_info = create_sarcophagus()

        self._timeline = omni.timeline.get_timeline_interface()
        self.hydra_texture = None
        self._annotator = rep.AnnotatorRegistry.get_annotator("IsaacCreateRTXLidarScanBuffer")
        self._annotator_data = None
        self._annotator_generic_model_output_data = None
        self._default_use_fixed_time_stepping = carb.settings.get_settings().get("/app/player/useFixedTimeStepping")

    async def tearDown(self):
        """Clean up test environment and restore default settings."""
        self._timeline.stop()
        self._annotator.detach()
        self.hydra_texture.destroy()
        self.hydra_texture = None
        await omni.kit.app.get_app().next_update_async()
        while omni.usd.get_context().get_stage_loading_status()[2] > 0:
            print("tearDown, assets still loading, waiting to finish...")
            await asyncio.sleep(1.0)
        await update_stage_async()
        carb.settings.get_settings().set_bool("/app/player/useFixedTimeStepping", self._default_use_fixed_time_stepping)

    async def _test_annotator_outputs(
        self, config: str = DEFAULT_CONFIG, variant: str = DEFAULT_VARIANT, enable_per_frame_output: bool = False
    ):
        """Test the IsaacCreateRTXLidarScanBuffer annotator output data.

        Args:
            config: Lidar configuration name.
            variant: Lidar variant name.
            enable_per_frame_output: Whether to enable per-frame output mode instead of full scan mode.
        """
        # Create sensor prim
        kwargs = {
            "path": "lidar",
            "parent": None,
            "translation": Gf.Vec3d(0.0, 0.0, 0.0),
            "orientation": Gf.Quatd(1.0, 0.0, 0.0, 0.0),
            "config": config,
            "variant": variant,
            "omni:sensor:Core:outputFrameOfReference": "WORLD",
            "omni:sensor:Core:auxOutputType": "FULL",
        }

        _, self.sensor = omni.kit.commands.execute(f"IsaacSensorCreateRtxLidar", **kwargs)
        sensor_type = self.sensor.GetTypeName()
        self.assertEqual(
            sensor_type, "OmniLidar", f"Expected OmniLidar prim, got {sensor_type}. Was sensor prim created?"
        )

        # Create render product and attach to sensor
        self.hydra_texture = rep.create.render_product(
            self.sensor.GetPath(),
            [32, 32],
            name="RtxSensorRenderProduct",
            render_vars=["GenericModelOutput", "RtxSensorMetadata"],
        )
        # Attach annotator to render product
        self._annotator.initialize(
            outputAzimuth=True,
            outputElevation=True,
            outputDistance=True,
            outputIntensity=True,
            outputTimestamp=True,
            outputEmitterId=True,
            outputChannelId=True,
            outputMaterialId=True,
            outputTickId=True,
            outputHitNormal=True,
            outputVelocity=True,
            outputObjectId=True,
            outputEchoId=True,
            outputTickState=True,
            enablePerFrameOutput=enable_per_frame_output,
        )
        self._annotator.attach([self.hydra_texture.path])

        # Attach GenericModelOutput annotator to OmniLidar prim to pick up object IDs
        annotator_generic_model_output = rep.AnnotatorRegistry.get_annotator("GenericModelOutput")
        annotator_generic_model_output.attach([self.hydra_texture.path])

        # Render frames until we get valid data, or until we've rendered the maximum number of frames
        self._timeline.play()
        gmo_buffer = {}
        for _ in range(20):
            # Wait for a single frame
            await omni.kit.app.get_app().next_update_async()
            self._annotator_data = self._annotator.get_data(do_array_copy=True)
            self._annotator_generic_model_output_data = annotator_generic_model_output.get_data(do_array_copy=True)
            gmo = get_gmo_data(self._annotator_generic_model_output_data)
            if gmo.magicNumber == gmo_utils.getMagicNumberGMO() and gmo.numElements > 0:
                key = gmo.timestampNs + gmo.timeOffsetNs[0].astype(np.uint64)
                num_elements = gmo.numElements
                valid_elements = np.bitwise_and(gmo.flags, 64) == 64
                # Need to copy data out of the GMO buffer here to avoid losing the original data when the GMO buffer is cleared
                gmo_buffer[key] = {
                    "azimuth": gmo.x[valid_elements],
                    "elevation": gmo.y[valid_elements],
                    "distance": gmo.z[valid_elements],
                    "intensity": gmo.scalar[valid_elements],
                    "timestamp": gmo.timeOffsetNs[valid_elements].astype(np.uint64) + gmo.timestampNs,
                    "emitterId": gmo.emitterId[valid_elements],
                    "channelId": gmo.channelId[valid_elements],
                    "materialId": gmo.matId[valid_elements],
                    "tickId": gmo.tickId[valid_elements],
                    "hitNormal": gmo.hitNormals[np.repeat(valid_elements, 3)].reshape(-1, 3),
                    "velocity": gmo.velocities[np.repeat(valid_elements, 3)].reshape(-1, 3),
                    "objectId": gmo.objId[np.repeat(valid_elements, 16)],
                    "echoId": gmo.echoId[valid_elements],
                    "tickState": gmo.tickStates[valid_elements],
                }
            if (
                not enable_per_frame_output
                and self._annotator_data
                and "data" in self._annotator_data
                and self._annotator_data["data"].size > 0
            ):
                break
        self._timeline.stop()
        annotator_generic_model_output.detach()

        # Test point cloud data shape and dtype
        self.assertIn("data", self._annotator_data)
        self.data = self._annotator_data["data"]
        self.assertGreater(self.data.shape[0], 0, "Expected non-empty data.")
        self.assertEqual(self.data.shape[1], 3)
        self.assertEqual(self.data.dtype, np.float32)
        num_points = self.data.shape[0]

        # Test that all expected keys are present in the annotator data with the correct shape and dtype
        self.assertIn("info", self._annotator_data)

        self.assertIn("transform", self._annotator_data["info"])
        self.transform = self._annotator_data["info"]["transform"]
        self.assertEqual(self.transform.shape[0], 16, "Expected non-empty transform.")
        self.assertEqual(self.transform.dtype, np.float64)
        self.assertTrue(
            np.allclose(self.transform, np.eye(4, dtype=np.float64).flatten()), "Expected identity transform."
        )

        self.assertIn("radialVelocityMS", self._annotator_data)
        self.assertEqual(self._annotator_data["radialVelocityMS"].size, 0, "Expected empty radial velocity data.")

        expected_keys = {
            "azimuth": (num_points, np.float32),
            "elevation": (num_points, np.float32),
            "distance": (num_points, np.float32),
            "intensity": (num_points, np.float32),
            "timestamp": (num_points, np.uint64),
            "emitterId": (num_points, np.uint32),
            "channelId": (num_points, np.uint32),
            "materialId": (num_points, np.uint32),
            "tickId": (num_points, np.uint32),
            "hitNormal": (num_points, np.float32),
            "velocity": (num_points, np.float32),
            "objectId": (num_points * 4, np.uint32),
            "echoId": (num_points, np.uint8),
            "tickState": (num_points, np.uint8),
        }

        for attribute in expected_keys:
            self.assertIn(attribute, self._annotator_data, f"Expected {attribute} in annotator.get_data().")
            self.assertIn(
                attribute, self._annotator_data["info"], f"Expected {attribute} in annotator.get_data()['info']."
            )
            self.assertTrue(
                np.allclose(self._annotator_data[attribute], self._annotator_data["info"][attribute]),
                f"Expected {attribute} in annotator.get_data() and annotator.get_data()['info'] to be the same.",
            )
            attribute_data = self._annotator_data[attribute]
            self.assertEqual(
                attribute_data.shape[0],
                expected_keys[attribute][0],
                f"Expected {attribute} to have {expected_keys[attribute][0]} points, got {attribute_data.shape[0]}.",
            )
            self.assertEqual(
                attribute_data.dtype,
                expected_keys[attribute][1],
                f"Expected {attribute} to have dtype {expected_keys[attribute][1]}, got {attribute_data.dtype}.",
            )

        self._compare_annotator_to_gmo(gmo_buffer, expected_keys, compare_data_values=enable_per_frame_output)

    def _compare_annotator_to_gmo(self, gmo_buffer: dict, expected_keys: dict, compare_data_values: bool = True):
        """Compare annotator output data against GMO buffer data.

        Args:
            gmo_buffer: Dictionary of GMO frame data keyed by timestamp.
            expected_keys: Dictionary of expected attribute keys with (shape, dtype) tuples.
            compare_data_values: If True, compare against single frame matching timestamp and verify data values.
                If False, compare against concatenated 6 frames and only verify shape and dtype.
        """
        if compare_data_values:
            # For per-frame output, use single GMO frame matching the annotator's timestamp
            self.assertIn(
                self._annotator_data["timestamp"][0],
                gmo_buffer.keys(),
                f"Expected timestamp {self._annotator_data['timestamp'][0]} to be in GMO buffer (keys: {gmo_buffer.keys()}).",
            )
            gmo_data_for_comparison = gmo_buffer[self._annotator_data["timestamp"][0]]
        else:
            # For full scan, compute expected shape from most recent GMO buffer scaled by 6x
            sorted_keys = sorted(gmo_buffer.keys())
            most_recent_key = sorted_keys[-1]
            most_recent_gmo = gmo_buffer[most_recent_key]
            gmo_data_for_comparison = {}
            for attribute in expected_keys:
                single_frame_data = most_recent_gmo[attribute]
                expected_size = single_frame_data.shape[0] * 6
                # Create placeholder array with the expected shape and dtype
                if len(single_frame_data.shape) == 1:
                    gmo_data_for_comparison[attribute] = np.empty((expected_size,), dtype=single_frame_data.dtype)
                else:
                    gmo_data_for_comparison[attribute] = np.empty(
                        (expected_size,) + single_frame_data.shape[1:], dtype=single_frame_data.dtype
                    )

        for attribute in expected_keys:
            attribute_data = self._annotator_data[attribute]
            comparison_data = gmo_data_for_comparison[attribute]
            expected_shape = comparison_data.shape
            expected_dtype = comparison_data.dtype
            if attribute == "objectId":
                expected_shape = (comparison_data.shape[0] // 4,)
                expected_dtype = np.uint32

            # Verify shape matches
            self.assertAlmostEqual(
                attribute_data.shape[0],
                expected_shape[0],
                delta=0.01 * expected_shape[0],
                msg=f"Expected {attribute} to have shape {expected_shape[0]}, got {attribute_data.shape[0]}.",
            )

            # Verify dtype matches
            self.assertEqual(
                attribute_data.dtype,
                expected_dtype,
                f"Expected {attribute} to have dtype {expected_dtype}, got {attribute_data.dtype}.",
            )

            # Optionally verify data values match
            if compare_data_values:
                if attribute == "objectId":
                    self.assertTrue(
                        np.all(attribute_data.view(np.uint8) == comparison_data.view(np.uint8)),
                        f"Expected objectId to match the corresponding data from the GMO buffer.",
                    )
                else:
                    self.assertTrue(
                        np.allclose(attribute_data, comparison_data),
                        f"Expected {attribute} to match the corresponding data from the GMO buffer.",
                    )

    async def _soak_annotator(
        self,
        config: str = DEFAULT_CONFIG,
        variant: str = DEFAULT_VARIANT,
        enable_full_scan: bool = False,
        enable_fixed_time_stepping: bool = False,
    ):
        """Test the annotator over multiple frames to verify stability and correct data production patterns.

        Args:
            config: Lidar configuration name.
            variant: Lidar variant name.
            enable_full_scan: Whether to enable full scan mode instead of per-frame output.
            enable_fixed_time_stepping: Whether to enable fixed time stepping for consistent frame timing.
        """
        # Create sensor prim
        kwargs = {
            "path": "lidar",
            "parent": None,
            "translation": Gf.Vec3d(0.0, 0.0, 0.0),
            "orientation": Gf.Quatd(1.0, 0.0, 0.0, 0.0),
            "config": config,
            "variant": variant,
            "omni:sensor:Core:outputFrameOfReference": "WORLD",
            "omni:sensor:Core:auxOutputType": "FULL",
        }

        if enable_fixed_time_stepping:
            carb.settings.get_settings().set_bool("/app/player/useFixedTimeStepping", enable_fixed_time_stepping)

        _, self.sensor = omni.kit.commands.execute("IsaacSensorCreateRtxLidar", **kwargs)
        sensor_type = self.sensor.GetTypeName()
        self.assertEqual(
            sensor_type, "OmniLidar", f"Expected OmniLidar prim, got {sensor_type}. Was sensor prim created?"
        )
        scan_rate_base_hz = self.sensor.GetAttribute("omni:sensor:Core:scanRateBaseHz").Get()

        # Create render product and attach to sensor
        self.hydra_texture = rep.create.render_product(
            self.sensor.GetPath(),
            [32, 32],
            name="RtxSensorRenderProduct",
            render_vars=["GenericModelOutput", "RtxSensorMetadata"],
        )
        # Attach annotator to render product
        self._annotator.initialize(enablePerFrameOutput=(not enable_full_scan), outputAzimuth=True)
        self._annotator.attach([self.hydra_texture.path])

        self._timeline.play()
        data_frame_count = 0
        NUM_FRAMES = 100
        WARMUP_FRAMES = 5

        # For full scan mode: track frames between data updates
        frames_per_scan = int(60.0 / scan_rate_base_hz)  # Expected frames between full scans
        last_nonzero_data = None  # Track last nonzero data to detect updates
        frames_since_last_update = 0  # Counter for frames since last data change

        for frame_idx in range(NUM_FRAMES):
            # Wait for a single frame
            await omni.kit.app.get_app().next_update_async()
            self._annotator_data = self._annotator.get_data(do_array_copy=True)

            has_data = self._annotator_data and "data" in self._annotator_data and self._annotator_data["data"].size > 0

            if not enable_full_scan:
                # Per-frame output mode: count frames with data
                if has_data:
                    data_frame_count += 1
                continue

            # Full scan mode: verify data update pattern
            current_data = self._annotator_data["data"] if has_data else None
            is_new_data = has_data and (
                last_nonzero_data is None or not np.array_equal(current_data, last_nonzero_data)
            )

            if is_new_data:
                data_frame_count += 1
                if last_nonzero_data is None:
                    # First nonzero data received - verify warmup requirement
                    self.assertGreaterEqual(
                        frame_idx,
                        WARMUP_FRAMES,
                        f"Expected first valid data after at least {WARMUP_FRAMES} frames, "
                        f"but got valid data at frame {frame_idx}",
                    )
                else:
                    # Data changed - verify interval and full scan properties
                    self.assertEqual(
                        frames_since_last_update,
                        frames_per_scan - 1,
                        f"Expected data update after exactly {frames_per_scan} frames (inclusive), "
                        f"but got update after {frames_since_last_update + 1} frames (frame {frame_idx})",
                    )
                    azimuth = np.sort(self._annotator_data["azimuth"])
                    azimuth_diff = np.diff(azimuth)
                    self.assertAlmostEqual(np.min(azimuth), -180.0, delta=5e-3)
                    self.assertAlmostEqual(np.max(azimuth), 180.0, delta=5e-3)
                    self.assertAlmostEqual(np.min(azimuth_diff), 0.0)
                    if enable_fixed_time_stepping:
                        self.assertLessEqual(np.max(azimuth_diff), 0.05)

                last_nonzero_data = current_data.copy()
                frames_since_last_update = 0
            elif last_nonzero_data is not None:
                # After first valid data, count frames until next update
                frames_since_last_update += 1

        self._timeline.stop()

        expected_valid_frames = NUM_FRAMES - WARMUP_FRAMES
        expected_data_frame_count = (
            int(expected_valid_frames * scan_rate_base_hz / 60.0) if enable_full_scan else expected_valid_frames
        )
        self.assertGreaterEqual(data_frame_count, expected_data_frame_count)

    async def test_3d_lidar(self):
        """Test the annotator with 3D lidar in per-frame output mode."""
        await self._test_annotator_outputs(config="Example_Rotary", variant=None, enable_per_frame_output=True)

    async def test_3d_lidar_full_scan(self):
        """Test the annotator with 3D lidar in full scan mode."""
        await self._test_annotator_outputs(config="Example_Rotary", variant=None, enable_per_frame_output=False)

    async def test_3d_lidar_soak(self):
        """Test the annotator stability with 3D lidar in per-frame output mode over multiple frames."""
        await self._soak_annotator(config="Example_Rotary", variant=None, enable_full_scan=False)

    async def test_3d_lidar_soak_full_scan(self):
        """Test the annotator stability with 3D lidar in full scan mode over multiple frames."""
        await self._soak_annotator(config="Example_Rotary", variant=None, enable_full_scan=True)

    async def test_3d_lidar_soak_full_scan_fixed_time_stepping(self):
        """Test the annotator stability with 3D lidar in full scan mode using fixed time stepping over multiple frames."""
        await self._soak_annotator(
            config="Example_Rotary", variant=None, enable_full_scan=True, enable_fixed_time_stepping=True
        )


class TestIsaacComputeRTXLidarFlatScan(omni.kit.test.AsyncTestCase):
    """Test class for IsaacComputeRTXLidarFlatScan annotator"""

    async def setUp(self):
        """Setup test environment with a cube and lidar."""
        await create_new_stage_async()
        await update_stage_async()

        # Ordering octants in binary order, such that octant 0 is +++, octant 1 is ++-, etc. for XYZ.
        self._octant_dimensions = [
            (10, 10, 5),
            (10, 10, 7),
            (25, 25, 17),
            (25, 25, 19),
            (15, 15, 9),
            (15, 15, 11),
            (20, 20, 13),
            (20, 20, 15),
        ]
        self.cube_info = create_sarcophagus()

        self._timeline = omni.timeline.get_timeline_interface()
        self._annotator_data = None
        self.hydra_texture = None

        self._lidar_scan_buffer_annotator = rep.AnnotatorRegistry.get_annotator(
            "IsaacCreateRTXLidarScanBufferForFlatScan"
        )
        self._lidar_flat_scan_annotator = rep.AnnotatorRegistry.get_annotator("IsaacComputeRTXLidarFlatScan")

    async def tearDown(self):
        """Clean up test environment by stopping timeline and detaching annotators."""
        self._timeline.stop()
        self._lidar_scan_buffer_annotator.detach()
        self._lidar_scan_buffer_annotator = None
        self._lidar_flat_scan_annotator.detach()
        self._lidar_flat_scan_annotator = None
        self.hydra_texture.destroy()
        self.hydra_texture = None

    async def _test_annotator_outputs(
        self, config: str = DEFAULT_CONFIG, variant: str = DEFAULT_VARIANT
    ):  # Create sensor prim
        """Test the IsaacComputeRTXLidarFlatScan annotator outputs against expected values.

        Creates RTX Lidar sensor, attaches scan buffer and flat scan annotators, then validates
        all output parameters against sensor configuration and point cloud data for both 3D and 2D lidars.

        Args:
            config: Lidar configuration to use for testing.
            variant: Lidar variant to use for testing.
        """
        kwargs = {
            "path": "lidar",
            "parent": None,
            "translation": Gf.Vec3d(0.0, 0.0, 0.0),
            "orientation": Gf.Quatd(1.0, 0.0, 0.0, 0.0),
            "config": config,
            "variant": variant,
            "omni:sensor:Core:outputFrameOfReference": "WORLD",
            "omni:sensor:Core:auxOutputType": "BASIC",
        }

        _, self.sensor = omni.kit.commands.execute(f"IsaacSensorCreateRtxLidar", **kwargs)
        sensor_type = self.sensor.GetTypeName()
        self.assertEqual(
            sensor_type, "OmniLidar", f"Expected OmniLidar prim, got {sensor_type}. Was sensor prim created?"
        )
        # Test attributes of the sensor prim
        azimuthDeg = self.sensor.GetAttribute("omni:sensor:Core:emitterState:s001:azimuthDeg").Get()
        elevationDeg = self.sensor.GetAttribute("omni:sensor:Core:emitterState:s001:elevationDeg").Get()
        scanRateBaseHz = self.sensor.GetAttribute("omni:sensor:Core:scanRateBaseHz").Get()
        patternFiringRateHz = self.sensor.GetAttribute("omni:sensor:Core:patternFiringRateHz").Get()
        nearRangeM = self.sensor.GetAttribute("omni:sensor:Core:nearRangeM").Get()
        farRangeM = self.sensor.GetAttribute("omni:sensor:Core:farRangeM").Get()
        is_solid_state = self.sensor.GetAttribute("omni:sensor:Core:scanType").Get() == "SOLID_STATE"
        self._is_2d_lidar = all([abs(i) < 1e-3 for i in list(elevationDeg)])
        carb.log_warn(f"is_2d_lidar: {self._is_2d_lidar}")

        # Create render product and attach to sensor
        self.hydra_texture = rep.create.render_product(
            self.sensor.GetPath(),
            [32, 32],
            name="RtxSensorRenderProduct",
            render_vars=["GenericModelOutput", "RtxSensorMetadata"],
        )

        self._lidar_scan_buffer_annotator.attach([self.hydra_texture.path])
        self._lidar_flat_scan_annotator.attach([self.hydra_texture.path])

        # Render frames until we get valid data, or until we've rendered the maximum number of frames
        self._timeline.play()
        for i in range(10):
            # Wait for a single frame
            await omni.kit.app.get_app().next_update_async()
            self._lidar_scan_buffer_annotator_data = self._lidar_scan_buffer_annotator.get_data()
            self._lidar_flat_scan_annotator_data = self._lidar_flat_scan_annotator.get_data()
            if (
                self._lidar_scan_buffer_annotator_data
                and "data" in self._lidar_scan_buffer_annotator_data
                and self._lidar_scan_buffer_annotator_data["data"].size > 0
                and self._lidar_flat_scan_annotator_data
                and "numCols" in self._lidar_flat_scan_annotator_data
                and self._lidar_flat_scan_annotator_data["numCols"] > 0
            ):
                break
        self._timeline.stop()

        # Test that all expected keys are present in the annotator data, then copy those values to the test class attributes
        for expected_key in [
            "azimuthRange",
            "depthRange",
            "horizontalFov",
            "horizontalResolution",
            "intensitiesData",
            "linearDepthData",
            "numCols",
            "numRows",
            "rotationRate",
        ]:
            self.assertIn(expected_key, self._lidar_flat_scan_annotator_data)
            setattr(self, expected_key, self._lidar_flat_scan_annotator_data[expected_key])

        # Confirm default values for 3D lidar
        if not self._is_2d_lidar:
            self.assertTrue(np.allclose(np.zeros([1, 2]), self.azimuthRange), f"azimuthRange: {self.azimuthRange}")
            self.assertTrue(np.allclose(np.zeros([1, 2]), self.depthRange), f"depthRange: {self.depthRange}")
            self.assertEqual(0.0, self.horizontalFov)
            self.assertEqual(0.0, self.horizontalResolution)
            self.assertEqual(0, self.intensitiesData.size)
            self.assertEqual(0, self.linearDepthData.size)
            self.assertEqual(0, self.numCols)
            self.assertEqual(1, self.numRows)
            self.assertEqual(0.0, self.rotationRate)
            return

        pcl_azimuth = self._lidar_scan_buffer_annotator_data["azimuth"]
        pcl_distance = self._lidar_scan_buffer_annotator_data["distance"]
        pcl_intensity = self._lidar_scan_buffer_annotator_data["intensity"]

        # Test IsaacComputeRTXLidarFlatScan annotator outputs against prim attributes or IsaacCreateRTXLidarScanBuffer outputs as appropriate
        if is_solid_state:
            expectedMinAzimuth = min(azimuthDeg)
            expectedMaxAzimuth = max(azimuthDeg)
            expectedHorizontalFov = expectedMaxAzimuth - expectedMinAzimuth
            expectedHorizontalResolution = expectedHorizontalFov / len(azimuthDeg)
            if expectedMaxAzimuth > 180.0:
                expectedMinAzimuth -= 180.0
                expectedMaxAzimuth -= 180.0
        else:
            expectedMinAzimuth = -180.0
            expectedMaxAzimuth = 180.0
            expectedHorizontalFov = 360.0
            expectedHorizontalResolution = 360.0 * patternFiringRateHz / scanRateBaseHz

        self.assertAlmostEqual(self.azimuthRange[0], expectedMinAzimuth)
        self.assertAlmostEqual(self.azimuthRange[1], expectedMaxAzimuth - expectedHorizontalResolution)
        self.assertAlmostEqual(self.horizontalResolution, expectedHorizontalResolution)
        self.assertAlmostEqual(self.horizontalFov, expectedHorizontalFov)
        self.assertAlmostEqual(self.rotationRate, scanRateBaseHz)
        self.assertAlmostEqual(self.depthRange[0], nearRangeM)
        self.assertAlmostEqual(self.depthRange[1], farRangeM)

        expectedNumCols = round(self.horizontalFov / self.horizontalResolution)
        self.assertEqual(self.numCols, expectedNumCols)
        self.assertEqual(self.numRows, 1)
        self.assertEqual(self.linearDepthData.size, expectedNumCols)
        self.assertEqual(self.intensitiesData.size, expectedNumCols)

        expectedLinearDepthData = np.ones(expectedNumCols, dtype=np.float32) * -1.0
        expectedIntensitiesData = np.zeros(expectedNumCols, dtype=np.uint8)
        for i in range(pcl_azimuth.shape[0]):
            azimuth = pcl_azimuth[i]
            distance = pcl_distance[i]
            intensity = int(pcl_intensity[i] * 255.0)
            index = int((azimuth - expectedMinAzimuth) / expectedHorizontalResolution)
            if index >= expectedNumCols:
                index = expectedNumCols - 1
            expectedLinearDepthData[index] = distance
            expectedIntensitiesData[index] = intensity

        self.assertTrue(np.allclose(self.linearDepthData, expectedLinearDepthData))
        self.assertTrue(np.allclose(self.intensitiesData, expectedIntensitiesData))

    async def test_3d_lidar(self):
        """Test 3D lidar using Example_Rotary configuration."""
        await self._test_annotator_outputs(config="Example_Rotary", variant=None)

    async def test_2d_lidar(self):
        """Test 2D lidar using SICK_picoScan150 configuration with Profile_1 variant."""
        await self._test_annotator_outputs(config="SICK_picoScan150", variant="Profile_1")


class TestIsaacCreateRTXRadarPointCloud(omni.kit.test.AsyncTestCase):
    """Test class for IsaacCreateRTXRadarPointCloud annotator."""

    async def setUp(self):
        """Setup test environment with a cube and radar sensor.

        Creates a new stage, initializes octant dimensions for geometric testing,
        and prepares the radar testing environment with cube geometries.
        """
        await create_new_stage_async()
        await update_stage_async()

        # Ordering octants in binary order, such that octant 0 is +++, octant 1 is ++-, etc. for XYZ.
        self._octant_dimensions = [
            (10, 10, 5),
            (10, 10, 7),
            (25, 25, 17),
            (25, 25, 19),
            (15, 15, 9),
            (15, 15, 11),
            (20, 20, 13),
            (20, 20, 15),
        ]
        self.cube_info = create_sarcophagus()

        self._timeline = omni.timeline.get_timeline_interface()
        self._annotator_data = None
        self.hydra_texture = None

    async def tearDown(self):
        """Clean up test environment.

        Stops the timeline, detaches the annotator, destroys the hydra texture,
        and waits for all assets to finish loading before updating the stage.
        """
        self._timeline.stop()
        if self._annotator:
            self._annotator.detach()
        self.hydra_texture.destroy()
        self.hydra_texture = None
        await omni.kit.app.get_app().next_update_async()
        while omni.usd.get_context().get_stage_loading_status()[2] > 0:
            print("tearDown, assets still loading, waiting to finish...")
            await asyncio.sleep(1.0)
        await update_stage_async()

    async def test_rtx_radar(self):
        """Test RTX radar point cloud generation and data validation.

        Creates an RTX radar sensor, attaches the IsaacCreateRTXRadarPointCloud annotator,
        and validates that the generated point cloud data contains proper intensity values
        and zero radial velocity measurements for stationary objects.
        """

        kwargs = {
            "path": "radar",
            "parent": None,
            "translation": Gf.Vec3d(0.0, 0.0, 0.0),
            "orientation": Gf.Quatd(1.0, 0.0, 0.0, 0.0),
            "omni:sensor:WpmDmat:outputFrameOfReference": "WORLD",
            "omni:sensor:WpmDmat:auxOutputType": "FULL",
        }

        _, self.sensor = omni.kit.commands.execute(f"IsaacSensorCreateRtxRadar", **kwargs)
        sensor_type = self.sensor.GetTypeName()
        self.assertEqual(
            sensor_type, "OmniRadar", f"Expected OmniRadar prim, got {sensor_type}. Was sensor prim created?"
        )

        # Create render product and attach to sensor
        self.hydra_texture = rep.create.render_product(
            self.sensor.GetPath(),
            [32, 32],
            name="RtxSensorRenderProduct",
            render_vars=["GenericModelOutput"],
        )
        # Attach annotator to render product
        self._annotator = rep.AnnotatorRegistry.get_annotator("IsaacCreateRTXRadarPointCloud")
        self._annotator.initialize(
            outputIntensity=True,
            outputRadialVelocityMS=True,
        )
        self._annotator.attach([self.hydra_texture.path])

        # Render frames until we get valid data, or until we've rendered the maximum number of frames
        self._timeline.play()
        for _ in range(10):
            # Wait for a single frame
            await omni.kit.app.get_app().next_update_async()
            self._annotator_data = self._annotator.get_data()
            if self._annotator_data and "data" in self._annotator_data and self._annotator_data["data"].size > 0:
                break
        self._timeline.stop()

        # Test that all expected keys are present in the annotator data, then copy those values to the test class attributes
        for expected_key in [
            "data",
            "intensity",
            "radialVelocityMS",
        ]:
            self.assertIn(expected_key, self._annotator_data)
            setattr(self, expected_key, self._annotator_data[expected_key])

        self.assertIn("info", self._annotator_data)
        for expected_key in [
            "intensity",
            "radialVelocityMS",
        ]:
            self.assertIn(expected_key, self._annotator_data["info"])
            setattr(self, expected_key, self._annotator_data["info"][expected_key])

        # Test point cloud data shape
        self.assertGreater(self.data.shape[0], 0, "Expected non-empty data.")
        self.assertEqual(self.data.shape[1], 3)

        self.assertTrue(np.all(self.intensity != 0))
        self.assertTrue(np.allclose(self.radialVelocityMS, 0))
