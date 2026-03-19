# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Unit tests for HeightmapImporter class."""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import omni.kit.test
from isaacsim.asset.importer.heightmap.importer import (
    OCCUPIED_PIXEL_THRESHOLD,
    HeightmapImporter,
)
from PIL import Image
from pxr import Gf


class TestHeightmapImporter(omni.kit.test.AsyncTestCase):
    """Test suite for HeightmapImporter class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_stage = MagicMock()
        self.importer = HeightmapImporter(self.mock_stage)

    def test_initialization_with_stage(self):
        """Test initialization with a provided stage."""
        importer = HeightmapImporter(self.mock_stage)
        self.assertEqual(importer._stage, self.mock_stage)

    def test_initialization_without_stage(self):
        """Test initialization without a provided stage."""
        importer = HeightmapImporter()
        self.assertIsNone(importer._stage)

    @patch("isaacsim.asset.importer.heightmap.importer.omni.usd.get_context")
    def test_create_heightmap_with_none_image(self, mock_get_context):
        """Test that create_heightmap raises ValueError when image is None."""
        with self.assertRaises(ValueError) as context:
            self.importer.create_heightmap(None, 0.05)
        self.assertIn("Image cannot be None", str(context.exception))

    def test_create_heightmap_with_invalid_cell_scale(self):
        """Test that create_heightmap raises ValueError with invalid cell scale."""
        # Create a simple test image
        test_image = Image.new("RGBA", (10, 10), color=(255, 255, 255, 255))

        with self.assertRaises(ValueError) as context:
            self.importer.create_heightmap(test_image, 0.0)
        self.assertIn("Cell scale must be positive", str(context.exception))

        with self.assertRaises(ValueError) as context:
            self.importer.create_heightmap(test_image, -0.5)
        self.assertIn("Cell scale must be positive", str(context.exception))

    @patch("isaacsim.asset.importer.heightmap.importer.omni.usd.get_context")
    def test_create_heightmap_no_stage_available(self, mock_get_context):
        """Test that create_heightmap raises RuntimeError when stage is not available."""
        # Setup importer without a stage
        importer = HeightmapImporter(None)
        mock_context = Mock()
        mock_context.get_stage.return_value = None
        mock_get_context.return_value = mock_context

        test_image = Image.new("RGBA", (10, 10), color=(255, 255, 255, 255))

        with self.assertRaises(RuntimeError) as context:
            importer.create_heightmap(test_image, 0.05)
        self.assertIn("No USD stage available", str(context.exception))

    @patch("isaacsim.asset.importer.heightmap.importer.UsdPhysics")
    @patch("isaacsim.asset.importer.heightmap.importer.stage_utils")
    @patch("isaacsim.asset.importer.heightmap.importer.UsdGeom")
    @patch("isaacsim.asset.importer.heightmap.importer.GroundPlane")
    @patch("isaacsim.asset.importer.heightmap.importer.UsdLux")
    def test_create_heightmap_success(
        self, mock_usd_lux, mock_ground_plane, mock_usd_geom, mock_stage_utils, mock_usd_physics
    ):
        """Test successful heightmap creation."""
        # Create a simple test image with some black pixels
        test_image = Image.new("RGBA", (5, 5), color=(255, 255, 255, 255))
        pixels = test_image.load()
        # Make a few pixels black (below threshold)
        pixels[0, 0] = (0, 0, 0, 255)
        pixels[1, 1] = (50, 50, 50, 255)
        pixels[2, 2] = (100, 100, 100, 255)

        # Mock USD methods
        mock_world_prim = MagicMock()
        mock_cube_prim = MagicMock()

        def get_prim_side_effect(path):
            if "occupiedCube" in str(path):
                return mock_cube_prim
            return mock_world_prim

        self.mock_stage.GetPrimAtPath.side_effect = get_prim_side_effect
        self.mock_stage.DefinePrim.return_value = MagicMock()
        self.mock_stage.SetDefaultPrim = MagicMock()

        # Mock UsdGeom methods
        mock_xform = MagicMock()
        mock_xform.AddTranslateOp.return_value.Set = MagicMock()
        mock_usd_geom.Xform.return_value = mock_xform

        mock_point_instancer = MagicMock()
        mock_point_instancer.AddScaleOp.return_value.Set = MagicMock()
        mock_point_instancer.AddTranslateOp.return_value.Set = MagicMock()
        mock_point_instancer.CreatePositionsAttr.return_value.Set = MagicMock()
        mock_point_instancer.CreatePrototypesRel.return_value.SetTargets = MagicMock()
        mock_point_instancer.CreateProtoIndicesAttr.return_value.Set = MagicMock()
        mock_usd_geom.PointInstancer.return_value = mock_point_instancer

        mock_cube = MagicMock()
        mock_cube.AddScaleOp.return_value.Set = MagicMock()
        mock_cube.CreateSizeAttr = MagicMock()
        mock_cube.CreateDisplayColorPrimvar.return_value.Set = MagicMock()
        mock_cube.GetPath.return_value = "/test/path"
        mock_usd_geom.Cube.return_value = mock_cube

        # Mock lighting
        mock_light = MagicMock()
        mock_light.CreateIntensityAttr = MagicMock()
        mock_usd_lux.DistantLight.Define.return_value = mock_light

        # Mock UsdPhysics.CollisionAPI.Apply
        mock_collision_api = MagicMock()
        mock_usd_physics.CollisionAPI.Apply.return_value = mock_collision_api

        # Run the test
        num_cells = self.importer.create_heightmap(
            test_image, cell_scale=0.05, create_ground_plane=True, create_lighting=True
        )

        # Verify that cells were created (3 pixels below threshold)
        self.assertEqual(num_cells, 3)

        # Verify that stage setup methods were called
        mock_usd_geom.SetStageMetersPerUnit.assert_called_once()
        mock_stage_utils.set_stage_up_axis.assert_called_once_with("Z")

        # Verify collision API was applied
        mock_usd_physics.CollisionAPI.Apply.assert_called_once()
        mock_ground_plane.assert_called_once()

    def test_generate_occupied_positions_with_simple_image(self):
        """Test position generation with a simple test image."""
        # Create a 3x3 image with specific pattern
        test_image = Image.new("RGBA", (3, 3), color=(255, 255, 255, 255))
        pixels = test_image.load()
        # Make corners black (below threshold)
        pixels[0, 0] = (0, 0, 0, 255)  # Top-left
        pixels[2, 0] = (0, 0, 0, 255)  # Top-right
        pixels[0, 2] = (0, 0, 0, 255)  # Bottom-left
        pixels[2, 2] = (0, 0, 0, 255)  # Bottom-right

        cell_scale = 1.0
        cell_offset = 0.5
        positions = self.importer._generate_occupied_positions(test_image, cell_scale, cell_offset)

        # Should have 4 positions
        self.assertEqual(len(positions), 4)

        # Verify positions are Gf.Vec3f
        for pos in positions:
            self.assertIsInstance(pos, Gf.Vec3f)

        # Verify positions are calculated correctly
        # Expected positions for the four corners:
        # (0, 0) -> x=0.5, y=-0.5
        # (2, 0) -> x=2.5, y=-0.5
        # (0, 2) -> x=0.5, y=-2.5
        # (2, 2) -> x=2.5, y=-2.5
        expected_positions = {
            (0.5, -0.5),
            (2.5, -0.5),
            (0.5, -2.5),
            (2.5, -2.5),
        }

        actual_positions = {(float(pos[0]), float(pos[1])) for pos in positions}

        # Compare sets (order doesn't matter)
        for exp_pos in expected_positions:
            # Find matching position within tolerance
            found = any(
                abs(exp_pos[0] - act_pos[0]) < 1e-5 and abs(exp_pos[1] - act_pos[1]) < 1e-5
                for act_pos in actual_positions
            )
            self.assertTrue(found, f"Expected position {exp_pos} not found in actual positions")

    def test_generate_occupied_positions_with_threshold(self):
        """Test that only pixels below threshold are converted to positions."""
        # Create a 2x2 image
        test_image = Image.new("RGBA", (2, 2), color=(255, 255, 255, 255))
        pixels = test_image.load()
        # Below threshold
        pixels[0, 0] = (OCCUPIED_PIXEL_THRESHOLD - 1, 0, 0, 255)
        # At threshold (should not be included)
        pixels[1, 0] = (OCCUPIED_PIXEL_THRESHOLD, 0, 0, 255)
        # Above threshold (should not be included)
        pixels[0, 1] = (OCCUPIED_PIXEL_THRESHOLD + 1, 0, 0, 255)
        # Well below threshold
        pixels[1, 1] = (0, 0, 0, 255)

        positions = self.importer._generate_occupied_positions(test_image, 1.0, 0.5)

        # Only 2 positions should be created (below threshold)
        self.assertEqual(len(positions), 2)

    def test_generate_occupied_positions_with_all_white_image(self):
        """Test that an all-white image produces no positions."""
        test_image = Image.new("RGBA", (5, 5), color=(255, 255, 255, 255))
        positions = self.importer._generate_occupied_positions(test_image, 1.0, 0.5)
        self.assertEqual(len(positions), 0)

    def test_generate_occupied_positions_with_all_black_image(self):
        """Test that an all-black image produces positions for all pixels."""
        test_image = Image.new("RGBA", (5, 5), color=(0, 0, 0, 255))
        positions = self.importer._generate_occupied_positions(test_image, 1.0, 0.5)
        # Should create 25 positions (5x5)
        self.assertEqual(len(positions), 25)

    def test_generate_occupied_positions_with_different_scales(self):
        """Test position generation with different cell scales."""
        test_image = Image.new("RGBA", (2, 2), color=(0, 0, 0, 255))

        # Test with small scale
        positions_small = self.importer._generate_occupied_positions(test_image, 0.1, 0.05)
        self.assertEqual(len(positions_small), 4)
        self.assertAlmostEqual(positions_small[0][0], 0.05, places=5)

        # Test with large scale
        positions_large = self.importer._generate_occupied_positions(test_image, 10.0, 5.0)
        self.assertEqual(len(positions_large), 4)
        self.assertAlmostEqual(positions_large[0][0], 5.0, places=5)

    def test_generate_occupied_positions_z_coordinate(self):
        """Test that all positions have z-coordinate of 0.0."""
        test_image = Image.new("RGBA", (3, 3), color=(0, 0, 0, 255))
        positions = self.importer._generate_occupied_positions(test_image, 1.0, 0.5)

        for pos in positions:
            self.assertEqual(pos[2], 0.0)

    @patch("isaacsim.asset.importer.heightmap.importer.carb")
    def test_generate_occupied_positions_with_invalid_image(self, mock_carb):
        """Test position generation with invalid image data."""
        # Create a grayscale image (2D array instead of 3D)
        test_image = Image.new("L", (5, 5), color=0)

        positions = self.importer._generate_occupied_positions(test_image, 1.0, 0.5)

        # Should return empty list and log error
        self.assertEqual(len(positions), 0)
        mock_carb.log_error.assert_called()


class TestHeightmapImporterIntegration(omni.kit.test.AsyncTestCase):
    """Integration tests for HeightmapImporter with realistic scenarios."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_stage = MagicMock()
        self.importer = HeightmapImporter(self.mock_stage)

    def test_checkerboard_pattern(self):
        """Test position generation with a checkerboard pattern."""
        # Create 4x4 checkerboard
        test_image = Image.new("RGBA", (4, 4), color=(255, 255, 255, 255))
        pixels = test_image.load()
        for y in range(4):
            for x in range(4):
                if (x + y) % 2 == 0:
                    pixels[x, y] = (0, 0, 0, 255)

        positions = self.importer._generate_occupied_positions(test_image, 1.0, 0.5)

        # Checkerboard should have 8 black squares
        self.assertEqual(len(positions), 8)

    def test_stripe_pattern(self):
        """Test position generation with vertical stripes."""
        # Create 6x4 image with vertical stripes
        test_image = Image.new("RGBA", (6, 4), color=(255, 255, 255, 255))
        pixels = test_image.load()
        for y in range(4):
            for x in range(0, 6, 2):  # Every other column
                pixels[x, y] = (0, 0, 0, 255)

        positions = self.importer._generate_occupied_positions(test_image, 1.0, 0.5)

        # Should have 12 positions (3 columns × 4 rows)
        self.assertEqual(len(positions), 12)

    def test_large_image_performance(self):
        """Test that large images can be processed without errors."""
        # Create a 100x100 image
        test_image = Image.new("RGBA", (100, 100), color=(255, 255, 255, 255))
        pixels = test_image.load()
        # Make diagonal black
        for i in range(100):
            pixels[i, i] = (0, 0, 0, 255)

        positions = self.importer._generate_occupied_positions(test_image, 0.05, 0.025)

        # Should have 100 positions (diagonal)
        self.assertEqual(len(positions), 100)

        # Verify that positions increase along diagonal
        sorted_positions = sorted(positions, key=lambda p: (p[0], -p[1]))
        for i in range(len(sorted_positions) - 1):
            # X should increase
            self.assertLess(sorted_positions[i][0], sorted_positions[i + 1][0])
