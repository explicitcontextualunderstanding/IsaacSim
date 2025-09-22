import os
from typing import Any

import numpy as np


async def capture_annotator_data_async(
    annotator_name: str,
    camera_position: tuple[float, float, float] = (5, 5, 5),
    camera_look_at: tuple[float, float, float] = (0, 0, 0),
    resolution: tuple[int, int] = (1280, 720),
    camera_prim_path: str | None = None,
) -> Any:
    """Capture annotator data from a virtual camera in the simulation.

    Args:
        annotator_name: Name of the annotator to capture data from.
            See https://docs.omniverse.nvidia.com/py/replicator/1.11.35/source/extensions/omni.replicator.core/docs/API.html#default-annotators for available annotators.
        camera_position: 3D position coordinates for the camera placement (x, y, z).
        camera_look_at: 3D coordinates of the target point the camera should look at.
        resolution: Tuple specifying the image resolution (width, height).
        camera_prim_path: USD path to an existing camera prim to use for capture.
            If provided, camera_position and camera_look_at are ignored.

    Returns:
        Annotator data. The exact type depends on the annotator:
            - For image-based annotators (rgb, depth, normals, etc.): numpy array
            - For non-array annotators (camera_params, etc.): dict
            - Mixed annotators may return dict with 'data' key containing array

    Example:

    .. code-block:: python

        >>> from isaacsim.test.utils.image_capture import capture_annotator_data_async
        >>>
        >>> # Capture RGB data with temporary camera
        >>> rgb_data = await capture_annotator_data_async(
        ...     "rgb",
        ...     camera_position=(10, 10, 10),
        ...     camera_look_at=(0, 0, 0),
        ...     resolution=(512, 512)
        ... )
        >>> rgb_data.shape
        (512, 512, 4)
        >>>
        >>> # Capture distance to camera data using existing camera
        >>> distance_data = await capture_annotator_data_async(
        ...     "distance_to_camera",
        ...     resolution=(512, 512),
        ...     camera_prim_path="/World/Camera"
        ... )
        >>> distance_data.shape
        (512, 512, 1)
    """
    import omni.replicator.core as rep

    temp_cam = None
    camera_path = camera_prim_path
    if camera_path is None:
        # Create new camera
        temp_cam = rep.functional.create.camera(position=camera_position, look_at=camera_look_at)
        camera_path = temp_cam.GetPath()

    # Create render product once using either existing or temporary camera
    render_product = rep.create.render_product(camera_path, resolution)

    annot = rep.AnnotatorRegistry.get_annotator(annotator_name)
    annot.attach(render_product)
    await rep.orchestrator.step_async()
    annot_data = annot.get_data(do_array_copy=True)
    annot.detach()
    render_product.destroy()

    # Cleanup temporary camera
    if temp_cam is not None:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        stage.RemovePrim(camera_path)

    return annot_data


async def capture_rgb_data_async(
    camera_position: tuple[float, float, float] = (5, 5, 5),
    camera_look_at: tuple[float, float, float] = (0, 0, 0),
    resolution: tuple[int, int] = (1280, 720),
    camera_prim_path: str | None = None,
) -> np.ndarray:
    """Capture an RGB image from a virtual camera in the simulation.

    Args:
        camera_position: 3D position coordinates for the camera placement (x, y, z).
        camera_look_at: 3D coordinates of the target point the camera should look at.
        resolution: Tuple specifying the image resolution (width, height).
        camera_prim_path: USD path to an existing camera prim to use for capture.
            If provided, camera_position and camera_look_at are ignored.

    Returns:
        RGB image data as a numpy array with shape (height, width, 4) and dtype uint8.

    Example:

    .. code-block:: python

        >>> from isaacsim.test.utils.image_capture import capture_rgb_data_async
        >>>
        >>> # Using temporary camera creation
        >>> rgb_data = await capture_rgb_data_async(
        ...     camera_position=(10, 10, 10),
        ...     camera_look_at=(0, 0, 0),
        ...     resolution=(512, 512)
        ... )
        >>> rgb_data.shape
        (512, 512, 4)
        >>>
        >>> # Using existing camera prim
        >>> rgb_data = await capture_rgb_data_async(
        ...     resolution=(512, 512),
        ...     camera_prim_path="/World/Camera"
        ... )
        >>> rgb_data.shape
        (512, 512, 4)
    """
    return await capture_annotator_data_async("rgb", camera_position, camera_look_at, resolution, camera_prim_path)


async def capture_depth_data_async(
    depth_type: str = "distance_to_camera",
    camera_position: tuple[float, float, float] = (5, 5, 5),
    camera_look_at: tuple[float, float, float] = (0, 0, 0),
    resolution: tuple[int, int] = (1280, 720),
    camera_prim_path: str | None = None,
) -> np.ndarray:
    """Capture depth data from a virtual camera in the simulation.

    Args:
        depth_type: Type of depth measurement to capture. Must be either "distance_to_camera" or "distance_to_image_plane".
            - "distance_to_camera": Euclidean distance from each point to the camera origin (radial distance).
              This gives the actual 3D distance from the camera center to each point in the scene.
            - "distance_to_image_plane": Perpendicular distance from each point to the camera's image plane (Z-depth).
              This is the traditional depth buffer value used in computer graphics.
        camera_position: 3D position coordinates for the camera placement (x, y, z).
        camera_look_at: 3D coordinates of the target point the camera should look at.
        resolution: Tuple specifying the image resolution (width, height).
        camera_prim_path: USD path to an existing camera prim to use for capture.
            If provided, camera_position and camera_look_at are ignored.

    Returns:
        Depth data as a numpy array with shape (height, width) and dtype float32.

    Raises:
        ValueError: If depth_type is not "distance_to_camera" or "distance_to_image_plane".

    Example:

    .. code-block:: python

        >>> from isaacsim.test.utils.image_capture import capture_depth_data_async
        >>>
        >>> # Capture radial distance (distance to camera origin)
        >>> radial_depth = await capture_depth_data_async(
        ...     depth_type="distance_to_camera",
        ...     camera_position=(10, 10, 10),
        ...     camera_look_at=(0, 0, 0),
        ...     resolution=(512, 512)
        ... )
        >>> radial_depth.shape
        (512, 512, 1)
        >>>
        >>> # Capture Z-depth (distance to image plane)
        >>> z_depth = await capture_depth_data_async(
        ...     depth_type="distance_to_image_plane",
        ...     resolution=(512, 512),
        ...     camera_prim_path="/World/Camera"
        ... )
        >>> z_depth.shape
        (512, 512, 1)
    """
    valid_depth_types = ["distance_to_camera", "distance_to_image_plane"]
    if depth_type not in valid_depth_types:
        raise ValueError(
            f"Invalid depth_type '{depth_type}'. Must be one of {valid_depth_types}. "
            f"'distance_to_camera' provides radial distance from camera origin, "
            f"'distance_to_image_plane' provides perpendicular distance to camera plane (Z-depth)."
        )

    return await capture_annotator_data_async(depth_type, camera_position, camera_look_at, resolution, camera_prim_path)


def save_rgb_image(rgb_data: np.ndarray, out_dir: str, file_name: str) -> None:
    """Save RGB image data to a file on disk.

    This function converts numpy array RGB data to a PIL Image and saves it to the
    specified directory with the given filename. For RGBA data (4 channels), the file
    must be saved as PNG format since JPEG/JPG doesn't support transparency. The output
    directory is created if it doesn't exist.

    Args:
        rgb_data: RGB or RGBA image data as a numpy array with shape (height, width, channels).
        out_dir: Output directory path where the image file will be saved.
        file_name: Name of the output image file (including extension).

    Raises:
        ValueError: If trying to save RGBA data (4 channels) in JPEG/JPG format.

    Example:

    .. code-block:: python

        >>> import numpy as np
        >>> from isaacsim.test.utils.image_capture import save_rgb_image
        >>>
        >>> # Save RGB image
        >>> rgb_data = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        >>> save_rgb_image(rgb_data, "/tmp/test_images", "test_image.png")
        Saved image to /tmp/test_images/test_image.png with shape (512, 512, 3)
        >>>
        >>> # Save RGBA image (must use PNG format)
        >>> rgba_data = np.random.randint(0, 255, (512, 512, 4), dtype=np.uint8)
        >>> save_rgb_image(rgba_data, "/tmp/test_images", "test_rgba.png")
        Saved image to /tmp/test_images/test_rgba.png with shape (512, 512, 4)
    """
    from PIL import Image

    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

    # Check if input has alpha channel or if we're dealing with RGBA data
    has_alpha = len(rgb_data.shape) == 3 and rgb_data.shape[2] == 4

    # Get file extension to check format
    file_ext = os.path.splitext(file_name)[1].lower()

    # Verify PNG format for RGBA data
    if has_alpha and file_ext in [".jpg", ".jpeg"]:
        raise ValueError(
            f"Cannot save RGBA data (4 channels) as {file_ext.upper()} format. "
            "JPEG format doesn't support transparency. Use PNG format instead."
        )

    # Convert to appropriate format based on input data
    if has_alpha:
        rgb_img = Image.fromarray(rgb_data, mode="RGBA")
    else:
        rgb_img = Image.fromarray(rgb_data)
        if file_ext == ".png":
            rgb_img = rgb_img.convert("RGBA")  # Convert to RGBA for PNG

    file_path = os.path.join(out_dir, file_name)
    rgb_img.save(file_path)
    print(f"Saved image to {file_path} with shape {rgb_data.shape}")


def save_depth_image(
    depth_data: np.ndarray,
    out_dir: str,
    file_name: str,
    normalize: bool = False,
) -> None:
    """Save depth data as TIFF (float32) or grayscale visualization.

    This function supports two primary modes:
    1. Metric (float32 TIFF): When extension is .tif/.tiff
       - Always preserves exact depth values including NaN/Inf
       - Uses 32-bit float TIFF format with lossless compression
       - If normalize=True is passed, it will be ignored with a warning

    2. Visualization (8-bit grayscale): For all other extensions
       - Converts to 8-bit grayscale for viewing
       - Optionally normalizes values to full 0-255 range based on normalize flag

    Args:
        depth_data: Depth array with shape (H, W) or (H, W, 1).
        out_dir: Output directory path.
        file_name: Output filename with extension.
        normalize: If True, normalize valid values to 0-255 range for non-TIFF formats.
            Ignored for TIFF format which always saves raw float32 data.

    Raises:
        ValueError: If depth_data has invalid shape.

    Example:

    .. code-block:: python

        >>> import numpy as np
        >>> from isaacsim.test.utils.image_capture import save_depth_image
        >>>
        >>> # Save lossless float32 depth (normalize parameter is ignored for TIFF)
        >>> depth = np.random.rand(512, 512).astype(np.float32) * 10.0
        >>> save_depth_image(depth, "/tmp", "depth.tiff", normalize=False)
        Saved metric depth (float32 TIFF) to /tmp/depth.tiff
        >>>
        >>> # TIFF with normalize=True will ignore the normalize flag
        >>> save_depth_image(depth, "/tmp", "depth2.tiff", normalize=True)
        Warning: TIFF format requested with normalize=True. TIFF is intended for lossless float32 storage, ignoring normalize parameter.
        Saved metric depth (float32 TIFF) to /tmp/depth2.tiff
        >>>
        >>> # Save normalized grayscale visualization as PNG
        >>> save_depth_image(depth, "/tmp", "depth_viz.png", normalize=True)
        Saved grayscale depth to /tmp/depth_viz.png
    """
    from PIL import Image

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Validate and convert to 2D array
    if len(depth_data.shape) == 3 and depth_data.shape[2] == 1:
        depth_data = depth_data.squeeze(axis=2)
    elif len(depth_data.shape) != 2:
        raise ValueError(f"Expected depth data with shape (H, W) or (H, W, 1), got {depth_data.shape}")

    # Determine file path and extension
    file_path = os.path.join(out_dir, file_name)
    file_ext = os.path.splitext(file_name)[1].lower()

    # Route to appropriate format based on extension
    if file_ext in (".tif", ".tiff"):
        # TIFF is always saved as lossless float32, regardless of normalize flag
        if normalize:
            print(
                f"Warning: TIFF format requested with normalize=True. "
                f"TIFF is intended for lossless float32 storage, ignoring normalize parameter."
            )

        # Save as lossless float32 TIFF
        depth_f32 = depth_data.astype(np.float32, copy=False)
        img = Image.fromarray(depth_f32, mode="F")

        # Try compressed save first, fallback to uncompressed
        try:
            img.save(file_path, compression="tiff_deflate")
        except Exception:
            img.save(file_path)

        print(f"Saved metric depth (float32 TIFF) to {file_path}")
    else:
        # Convert to 8-bit grayscale for visualization
        if normalize:
            # Normalize based on valid min/max values
            valid_mask = np.isfinite(depth_data)

            if not np.any(valid_mask):
                # No valid values - use mid-gray
                depth_u8 = np.full_like(depth_data, 128, dtype=np.uint8)
            else:
                # Get min/max of valid values
                valid_values = depth_data[valid_mask]
                vmin, vmax = np.min(valid_values), np.max(valid_values)

                if vmax > vmin:
                    # Replace invalid values with vmin, then normalize
                    depth_clean = np.where(valid_mask, depth_data, vmin)
                    normalized = (depth_clean - vmin) / (vmax - vmin)
                    depth_u8 = (normalized * 255).astype(np.uint8)
                else:
                    # All values identical - use mid-gray
                    depth_u8 = np.full_like(depth_data, 128, dtype=np.uint8)
        else:
            # No normalization - direct conversion
            if depth_data.dtype in (np.float32, np.float64):
                # Assume 0-1 range for floats
                depth_u8 = (np.clip(depth_data, 0, 1) * 255).astype(np.uint8)
            else:
                # Direct conversion for integer types
                depth_u8 = depth_data.astype(np.uint8)

        # Save as grayscale image
        img = Image.fromarray(depth_u8, mode="L")
        img.save(file_path)
        print(f"Saved image as grayscale depth to {file_path}")
