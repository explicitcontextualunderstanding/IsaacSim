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
import io
import os
import tempfile
from unittest.mock import patch

import numpy as np
import omni
import omni.kit.app
import omni.usd
from isaacsim.test.utils.image_capture import (
    capture_depth_data_async,
    capture_rgb_data_async,
    save_depth_image,
    save_rgb_image,
)
from isaacsim.test.utils.image_comparison import compare_images_within_tolerances
from isaacsim.test.utils.timed_async_test import TimedAsyncTestCase
from PIL import Image
from pxr import UsdGeom, UsdLux


class TestImageCapture(TimedAsyncTestCase):
    """Test suite for image capture utilities."""

    async def setUp(self):
        """Set up test fixtures."""
        await super().setUp()
        self.test_dir = tempfile.mkdtemp()

    async def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary files
        import shutil

        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        await super().tearDown()

    def get_image_capture_golden_dir(self):
        # Resolve path to golden assets for image capture tests.
        ext_manager = omni.kit.app.get_app().get_extension_manager()
        ext_id = ext_manager.get_enabled_extension_id("isaacsim.test.utils")
        extension_path = ext_manager.get_extension_path(ext_id)
        return os.path.join(extension_path, "data", "golden", "image_capture")

    async def setup_golden_stage(self):
        # Create a minimal scene for deterministic rendering.
        await omni.usd.get_context().new_stage_async()
        await omni.kit.app.get_app().next_update_async()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.Cube.Define(stage, "/World/cube")
        UsdLux.DistantLight.Define(stage, "/DistantLight")

        # Wait for scene to settle
        for _ in range(3):
            await omni.kit.app.get_app().next_update_async()

    def test_save_rgb_image_basic_functionality(self):
        """Test save_rgb_image with basic functionality and directory creation."""
        # Arrange RGB data and save to disk; verify file path and mode.
        resolution = (480, 320)
        rgb_data = np.random.randint(0, 255, (resolution[1], resolution[0], 3), dtype=np.uint8)
        file_name = "test_image.png"

        save_rgb_image(rgb_data, self.test_dir, file_name)

        file_path = os.path.join(self.test_dir, file_name)
        self.assertTrue(os.path.exists(file_path))

        saved_image = Image.open(file_path)
        self.assertEqual(saved_image.mode, "RGBA")  # PNG converts to RGBA

        # Verify nested directory creation when the path does not yet exist.
        nested_dir = os.path.join(self.test_dir, "nested", "subdir")
        nested_file = "nested_test.png"

        save_rgb_image(rgb_data, nested_dir, nested_file)

        self.assertTrue(os.path.exists(nested_dir))
        nested_path = os.path.join(nested_dir, nested_file)
        self.assertTrue(os.path.exists(nested_path))

    def test_save_rgb_image_formats(self):
        """Test save_rgb_image with different data types and file formats."""
        # Test PNG/JPG/JPEG writers with 3-channel data.
        resolution = (480, 320)
        rgb_data = np.random.randint(0, 255, (resolution[1], resolution[0], 3), dtype=np.uint8)
        formats = ["test.png", "test.jpg", "test.jpeg"]

        for file_name in formats:
            save_rgb_image(rgb_data, self.test_dir, file_name)

            file_path = os.path.join(self.test_dir, file_name)
            self.assertTrue(os.path.exists(file_path))
            saved_image = Image.open(file_path)
            self.assertIsNotNone(saved_image)

        # RGBA must be saved as PNG (alpha channel supported).
        rgba_data = np.random.randint(0, 255, (resolution[1], resolution[0], 4), dtype=np.uint8)
        save_rgb_image(rgba_data, self.test_dir, "rgba_test.png")

        rgba_path = os.path.join(self.test_dir, "rgba_test.png")
        self.assertTrue(os.path.exists(rgba_path))
        saved_rgba = Image.open(rgba_path)
        self.assertEqual(saved_rgba.mode, "RGBA")

        # JPEG cannot encode alpha; expect ValueError for RGBA inputs.
        for ext in ["jpg", "jpeg"]:
            with self.assertRaises(ValueError) as context:
                save_rgb_image(rgba_data, self.test_dir, f"rgba_test.{ext}")

            self.assertIn("Cannot save RGBA data", str(context.exception))
            self.assertIn("JPEG format doesn't support transparency", str(context.exception))

    def test_save_rgb_image_file_operations(self):
        """Test save_rgb_image with existing directories and file overwriting."""
        resolution = (480, 320)
        rgb_data = np.random.randint(0, 255, (resolution[1], resolution[0], 3), dtype=np.uint8)

        # Test saving to existing directory
        file_name = "existing_dir_test.png"
        save_rgb_image(rgb_data, self.test_dir, file_name)

        file_path = os.path.join(self.test_dir, file_name)
        self.assertTrue(os.path.exists(file_path))

        # Test file overwriting with different data
        black_data = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)  # All black
        white_data = np.ones((resolution[1], resolution[0], 3), dtype=np.uint8) * 255  # All white
        overwrite_file = "overwrite_test.png"
        overwrite_path = os.path.join(self.test_dir, overwrite_file)

        # Save first image (black)
        save_rgb_image(black_data, self.test_dir, overwrite_file)

        self.assertTrue(os.path.exists(overwrite_path))
        first_image = Image.open(overwrite_path)
        first_array = np.array(first_image.convert("RGB"))

        # Save second image (white) - should overwrite
        save_rgb_image(white_data, self.test_dir, overwrite_file)

        second_image = Image.open(overwrite_path)
        second_array = np.array(second_image.convert("RGB"))

        # Verify images are different (black vs white)
        self.assertFalse(np.array_equal(first_array, second_array))

    async def test_capture_rgb_data_async(self):
        """Test capture_rgb_data_async with basic functionality and compare to golden image."""
        # Set up minimal stage and resolve golden path.
        await self.setup_golden_stage()
        golden_dir = self.get_image_capture_golden_dir()
        golden_image_path = os.path.join(golden_dir, "rgb_data.png")

        # Tolerance for RGB comparison (scene is mostly black/white).
        # Use absolute tolerance for allclose; relative is ineffective near 0.
        rgb_rtol = 0.0
        rgb_atol = 255  # allow rare almost full black↔white flips without failing allclose
        rgb_mean_tol = 5.0  # keep average difference very small
        rgb_percentile_tol = (99, 15.0)  # 99th percentile of abs diff

        # Capture RGB image
        rgb_resolution = (1280, 720)
        rgb_data = await capture_rgb_data_async(resolution=rgb_resolution)

        # Verify image was captured with correct dimensions
        self.assertIsNotNone(rgb_data, "RGB data is None")
        self.assertEqual(rgb_data.shape, (rgb_resolution[1], rgb_resolution[0], 4))

        # Compare against golden; capture stdout and print all stats at failure only.
        stdout_capture = io.StringIO()
        with patch("sys.stdout", stdout_capture):
            # Save captured image for comparison
            captured_image_path = os.path.join(self.test_dir, "rgb_data.png")
            save_rgb_image(rgb_data, self.test_dir, "rgb_data.png")

            # Compare with golden image
            if not os.path.exists(golden_image_path):
                self.fail(
                    f"Golden image not found at {golden_image_path}. "
                    f"Captured image saved to {captured_image_path} for reference. "
                    f"Please review and copy to golden directory if correct."
                )

            res = compare_images_within_tolerances(
                golden_file_path=golden_image_path,
                test_file_path=captured_image_path,
                allclose_rtol=rgb_rtol,
                allclose_atol=rgb_atol,
                mean_tolerance=rgb_mean_tol,
                percentile_tolerance=rgb_percentile_tol,
                print_all_stats=True,
            )
            is_similar = bool(res["passed"])

        captured_stdout = stdout_capture.getvalue()
        self.assertTrue(
            is_similar,
            f"Captured RGB image does not match golden image within tolerances.\n"
            f"Captured image saved at: {captured_image_path}\n"
            f"Captured stdout:\n{captured_stdout}",
        )

    async def test_capture_rgb_data_with_existing_camera_async(self):
        """Test capture_rgb_data_async using an existing camera prim."""
        await self.setup_golden_stage()

        # Temporary camera at custom resolution.
        test_resolution = (496, 384)
        rgb_data_tmp_cam = await capture_rgb_data_async(resolution=test_resolution)
        self.assertIsNotNone(rgb_data_tmp_cam)
        self.assertEqual(rgb_data_tmp_cam.shape, (test_resolution[1], test_resolution[0], 4))

        # Use default `/OmniverseKit_Persp` camera at default resolution.
        default_resolution = (1280, 720)
        rgb_data_persp_cam = await capture_rgb_data_async(camera_prim_path="/OmniverseKit_Persp")
        self.assertIsNotNone(rgb_data_persp_cam)
        self.assertEqual(rgb_data_persp_cam.shape, (default_resolution[1], default_resolution[0], 4))

        # Use a user-defined camera prim and verify it is preserved.
        test_resolution = (640, 480)
        stage = omni.usd.get_context().get_stage()
        custom_cam = UsdGeom.Camera.Define(stage, "/World/TestCamera")
        rgb_data_custom_cam = await capture_rgb_data_async(
            camera_prim_path=custom_cam.GetPath(), resolution=test_resolution
        )
        self.assertIsNotNone(rgb_data_custom_cam)
        self.assertEqual(rgb_data_custom_cam.shape, (test_resolution[1], test_resolution[0], 4))
        # Ensure the existing camera prim remains.
        self.assertTrue(stage.GetPrimAtPath("/World/TestCamera").IsValid())

    async def test_distance_to_camera_metric_tiff_async(self):
        """Test saving distance_to_camera depth data as float32 TIFF (lossless metric)."""
        # Set up minimal stage and resolve golden path.
        await self.setup_golden_stage()
        golden_dir = self.get_image_capture_golden_dir()
        golden_tiff_path = os.path.join(golden_dir, "distance_to_camera_float32.tiff")

        # Tolerance settings for TIFF metric depth comparison (tighter for float32 data)
        tiff_rtol = 1e-5  # 0.001% relative tolerance
        tiff_atol = 1e-3  # Small absolute tolerance for float32 precision

        # Capture distance_to_camera depth data
        depth_resolution = (480, 320)
        depth_data = await capture_depth_data_async(depth_type="distance_to_camera", resolution=depth_resolution)
        self.assertIsNotNone(depth_data, "Failed to capture distance_to_camera")
        self.assertEqual(depth_data.dtype, np.float32)

        # Save as float32 TIFF (lossless metric depth)
        tiff_out_path = os.path.join(self.test_dir, "distance_to_camera_float32.tiff")
        save_depth_image(depth_data, self.test_dir, os.path.basename(tiff_out_path), normalize=False)
        self.assertTrue(os.path.exists(tiff_out_path))

        # Verify TIFF is float32 format
        tiff_img = Image.open(tiff_out_path)
        self.assertEqual(tiff_img.mode, "F")  # F mode = 32-bit float

        # Verify data integrity
        tiff_data = np.array(tiff_img)
        original_data = depth_data.squeeze() if len(depth_data.shape) == 3 else depth_data
        self.assertEqual(tiff_data.shape, original_data.shape)
        is_close = np.allclose(tiff_data, original_data, rtol=1e-5, atol=1e-5, equal_nan=True)
        self.assertTrue(is_close, "TIFF data should match original within tolerance")

        # Compare with golden TIFF
        if not os.path.exists(golden_tiff_path):
            self.fail(
                f"Golden TIFF not found at {golden_tiff_path}. "
                f"Captured TIFF saved to {tiff_out_path} for reference. "
                f"Please review and copy to golden directory if correct."
            )

        # Capture stdout to keep CI logs concise.
        stdout_capture = io.StringIO()
        with patch("sys.stdout", stdout_capture):
            res = compare_images_within_tolerances(
                golden_file_path=golden_tiff_path,
                test_file_path=tiff_out_path,
                allclose_rtol=tiff_rtol,
                allclose_atol=tiff_atol,
                print_all_stats=True,
            )
            is_similar = bool(res["passed"])

        captured_stdout = stdout_capture.getvalue()
        self.assertTrue(
            is_similar,
            f"Captured TIFF depth does not match golden TIFF within tolerances.\n"
            f"Captured TIFF saved at: {tiff_out_path}\n"
            f"Captured stdout:\n{captured_stdout}",
        )

    async def test_distance_to_camera_normalized_png_async(self):
        """Test saving distance_to_camera depth data as normalized PNG (8-bit grayscale visualization)."""
        # Set up minimal stage and resolve golden path.
        await self.setup_golden_stage()
        golden_dir = self.get_image_capture_golden_dir()
        golden_normalized_path = os.path.join(golden_dir, "distance_to_camera_auto_scaled_8bit.png")

        # Tolerance settings for 8-bit grayscale depth visualization
        depth_viz_rtol = 0.1  # 10% relative tolerance
        depth_viz_atol = 25  # ~10% absolute tolerance for 8-bit images
        depth_viz_mean_tol = 10.0  # Average pixel difference tolerance

        # Capture distance_to_camera depth data
        depth_resolution = (480, 320)
        depth_data = await capture_depth_data_async(depth_type="distance_to_camera", resolution=depth_resolution)
        self.assertIsNotNone(depth_data, "Failed to capture distance_to_camera")
        self.assertEqual(depth_data.dtype, np.float32)

        # Save as normalized PNG (8-bit grayscale visualization)
        normalized_out_path = os.path.join(self.test_dir, "distance_to_camera_auto_scaled_8bit.png")
        save_depth_image(depth_data, self.test_dir, os.path.basename(normalized_out_path), normalize=True)
        self.assertTrue(os.path.exists(normalized_out_path))

        normalized_img = Image.open(normalized_out_path)
        self.assertEqual(normalized_img.mode, "L")  # L mode = 8-bit grayscale

        # Compare with golden normalized PNG
        if not os.path.exists(golden_normalized_path):
            self.fail(
                f"Golden normalized PNG not found at {golden_normalized_path}. "
                f"Captured normalized PNG saved to {normalized_out_path} for reference. "
                f"Please review and copy to golden directory if correct."
            )

        # Capture stdout to keep CI logs concise and compare with golden.
        stdout_capture = io.StringIO()
        with patch("sys.stdout", stdout_capture):
            res = compare_images_within_tolerances(
                golden_file_path=golden_normalized_path,
                test_file_path=normalized_out_path,
                allclose_rtol=depth_viz_rtol,
                allclose_atol=depth_viz_atol,
                mean_tolerance=depth_viz_mean_tol,
                print_all_stats=True,
            )
            is_similar = bool(res["passed"])

        captured_stdout = stdout_capture.getvalue()
        self.assertTrue(
            is_similar,
            f"Captured normalized PNG does not match golden within tolerances.\n"
            f"Captured PNG saved at: {normalized_out_path}\n"
            f"Captured stdout:\n{captured_stdout}",
        )

    async def test_distance_to_camera_non_normalized_png_async(self):
        """Test saving distance_to_camera depth data as non-normalized PNG and JPEG formats."""
        # Set up minimal stage and resolve golden path.
        await self.setup_golden_stage()
        golden_dir = self.get_image_capture_golden_dir()
        golden_non_normalized_path = os.path.join(golden_dir, "distance_to_camera_manual_scaled_8bit.png")

        # Tolerance settings for 8-bit grayscale depth visualization (same as normalized)
        depth_viz_rtol = 0.1  # 10% relative tolerance
        depth_viz_atol = 25  # ~10% absolute tolerance for 8-bit images
        depth_viz_mean_tol = 10.0  # Average pixel difference tolerance

        # Capture distance_to_camera depth data
        depth_resolution = (480, 320)
        depth_data = await capture_depth_data_async(depth_type="distance_to_camera", resolution=depth_resolution)
        self.assertIsNotNone(depth_data, "Failed to capture distance_to_camera")
        self.assertEqual(depth_data.dtype, np.float32)

        # Scale depth data to [0, 1] range for the non-normalized path.
        depth_01 = depth_data.copy()
        valid_mask = np.isfinite(depth_01)
        if np.any(valid_mask):
            vmin, vmax = np.min(depth_01[valid_mask]), np.max(depth_01[valid_mask])
            if vmax > vmin:
                depth_01[valid_mask] = (depth_01[valid_mask] - vmin) / (vmax - vmin)

        # Save as non-normalized PNG (assumes 0-1 range)
        non_normalized_out_path = os.path.join(self.test_dir, "distance_to_camera_manual_scaled_8bit.png")
        save_depth_image(depth_01, self.test_dir, os.path.basename(non_normalized_out_path), normalize=False)
        self.assertTrue(os.path.exists(non_normalized_out_path))

        non_normalized_img = Image.open(non_normalized_out_path)
        self.assertEqual(non_normalized_img.mode, "L")

        # Compare with golden non-normalized PNG
        if not os.path.exists(golden_non_normalized_path):
            self.fail(
                f"Golden non-normalized PNG not found at {golden_non_normalized_path}. "
                f"Captured non-normalized PNG saved to {non_normalized_out_path} for reference. "
                f"Please review and copy to golden directory if correct."
            )

        # Capture stdout to keep CI logs concise and compare with golden.
        stdout_capture = io.StringIO()
        with patch("sys.stdout", stdout_capture):
            res = compare_images_within_tolerances(
                golden_file_path=golden_non_normalized_path,
                test_file_path=non_normalized_out_path,
                allclose_rtol=depth_viz_rtol,
                allclose_atol=depth_viz_atol,
                mean_tolerance=depth_viz_mean_tol,
                print_all_stats=True,
            )
            is_similar = bool(res["passed"])

        captured_stdout = stdout_capture.getvalue()
        self.assertTrue(
            is_similar,
            f"Captured non-normalized PNG does not match golden within tolerances.\n"
            f"Captured PNG saved at: {non_normalized_out_path}\n"
            f"Captured stdout:\n{captured_stdout}",
        )

        # Also test JPEG format with normalize=False
        jpg_request_path = os.path.join(self.test_dir, "distance_to_camera_manual_scaled_8bit.jpg")
        save_depth_image(depth_01, self.test_dir, os.path.basename(jpg_request_path), normalize=False)
        self.assertTrue(os.path.exists(jpg_request_path))

        # Verify the saved JPEG image
        jpg_img = Image.open(jpg_request_path)
        self.assertIn(jpg_img.mode, ("L", "RGB"))  # JPEG might convert to RGB

    def test_save_depth_image_grayscale(self):
        """Test save_depth_image with grayscale output."""
        # Test basic float32 depth data normalization
        resolution = (480, 320)
        depth_data = np.random.rand(resolution[1], resolution[0]).astype(np.float32) * 10.0
        file_name = "test_depth_gray.png"

        save_depth_image(depth_data, self.test_dir, file_name, normalize=True)

        file_path = os.path.join(self.test_dir, file_name)
        self.assertTrue(os.path.exists(file_path))

        # Check saved image
        saved_image = Image.open(file_path)
        self.assertEqual(saved_image.mode, "L")  # Grayscale
        self.assertEqual(saved_image.size, (resolution[0], resolution[1]))

    def test_save_depth_image_without_normalization(self):
        """Test save_depth_image without normalization."""
        # Test with uint8 data that doesn't need normalization
        resolution = (480, 320)
        depth_data = np.random.randint(0, 255, (resolution[1], resolution[0]), dtype=np.uint8)
        file_name = "test_depth_no_norm.png"

        save_depth_image(depth_data, self.test_dir, file_name, normalize=False)

        file_path = os.path.join(self.test_dir, file_name)
        self.assertTrue(os.path.exists(file_path))

        saved_image = Image.open(file_path)
        self.assertEqual(saved_image.mode, "L")  # Always L mode for non-TIFF

        # Test with float data in 0-1 range
        depth_data_float = np.random.rand(resolution[1], resolution[0]).astype(np.float32)
        file_name_float = "test_depth_float_no_norm.png"

        save_depth_image(depth_data_float, self.test_dir, file_name_float, normalize=False)

        file_path_float = os.path.join(self.test_dir, file_name_float)
        self.assertTrue(os.path.exists(file_path_float))

    def test_save_depth_image_tiff_with_normalize_warning(self):
        """Test that TIFF with normalize=True ignores normalize and saves as float32 TIFF."""
        resolution = (480, 320)
        depth_data = np.random.rand(resolution[1], resolution[0]).astype(np.float32) * 10.0
        tiff_file_name = "test_depth_normalize_ignored.tiff"

        # This should save as TIFF (ignoring normalize=True) and print a warning
        stdout_capture = io.StringIO()
        with patch("sys.stdout", stdout_capture):
            save_depth_image(depth_data, self.test_dir, tiff_file_name, normalize=True)
        captured_output = stdout_capture.getvalue()
        self.assertIn(
            "Warning: TIFF format requested with normalize=True",
            captured_output,
            "Should warn about normalize being ignored for TIFF",
        )

        # Check that TIFF was saved (not PNG)
        tiff_path = os.path.join(self.test_dir, tiff_file_name)
        self.assertTrue(os.path.exists(tiff_path), "TIFF file should exist")

        # Verify the saved TIFF is float32 format
        saved_image = Image.open(tiff_path)
        self.assertEqual(saved_image.mode, "F")  # F mode = 32-bit float

        # Verify data integrity - should be the same as original
        tiff_data = np.array(saved_image)
        self.assertEqual(tiff_data.shape, depth_data.shape, "TIFF shape should match original data")
        is_close = np.allclose(tiff_data, depth_data, rtol=1e-5, atol=1e-5, equal_nan=True)
        self.assertTrue(is_close, "TIFF should contain original float32 data, not normalized")
