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

"""
To generate several frames, create and run a bash script with the following content:

for i in $(seq 64 512); do
    echo ${i}x${i}
    PATH/TO/python.sh test_frame_delay.py --resolution ${i}x${i} --/app/updateOrder/checkForHydraRenderComplete=1000
done
"""


USE_REPLICATOR_WRITER = True
CAMERA_PATH = "/camera"
CAMERA_POS = [0, 0, 25]
COLLECTION_STEPS = 10

# parse any command-line arguments specific to the standalone application
import argparse
import os
import sys

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--resolution", type=str, default="256x256", help="Resolution (WxH)")
# Parse only known arguments, so that any (eg) Kit settings are passed through to the core Kit app
args, _ = parser.parse_known_args()

RESOLUTION = tuple([int(item) for item in args.resolution.split("x")])
PIXELS_PER_METER = 0.09765625 * RESOLUTION[0]

simulation_app = SimulationApp(
    {"headless": True}, experience=f'{os.environ["EXP_PATH"]}/isaacsim.exp.base.zero_delay.kit'
)

import pprint

import carb
import cv2
import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.transform as transform_utils
import numpy as np
import omni.replicator.core as rep
import omni.usd
from isaacsim.core.experimental.objects import Cube, GroundPlane
from isaacsim.core.experimental.prims import RigidPrim
from isaacsim.core.experimental.utils.semantics import add_labels
from isaacsim.core.rendering_manager import ViewportManager
from isaacsim.sensors.camera import Camera
from omni.replicator.core import AnnotatorRegistry, Writer
from pxr import UsdGeom, UsdPhysics

# rep.settings.set_render_rtx_realtime(antialiasing="DLAA")


class CustomWriter(Writer):
    def __init__(self):
        self.annotators = []
        self.annotators.append(AnnotatorRegistry.get_annotator("rgb"))
        self.annotators.append(AnnotatorRegistry.get_annotator("semantic_segmentation"))
        self.annotators.append(AnnotatorRegistry.get_annotator("bounding_box_2d_tight"))

    def write(self, data):
        # The base Writer class caches 'data' automatically, accessible via self.get_data()
        pass


def get_data(sensor: Camera | Writer) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get RGB, semantic segmentation and BBox from Camera or Writer (according to `USE_REPLICATOR_WRITER`)"""
    if USE_REPLICATOR_WRITER:
        rgb = sensor.get_data()["rgb"]
        semantic_segmentation = sensor.get_data()["semantic_segmentation"]["data"]
        bbox = sensor.get_data()["bounding_box_2d_tight"]["data"][0]
    else:
        rgb = sensor.get_rgba()
        semantic_segmentation = sensor._custom_annotators["semantic_segmentation"].get_data()["data"]
        bbox = sensor._custom_annotators["bounding_box_2d_tight"].get_data()["data"][0]

    semantic_segmentation = (semantic_segmentation * 255 / np.max(semantic_segmentation)).astype(np.uint8)
    semantic_segmentation = np.repeat(semantic_segmentation[:, :, np.newaxis], 3, axis=2)
    return rgb[:, :, :3], semantic_segmentation, bbox


def calculate_expected_bbox(position: np.ndarray) -> dict:
    """Calculate expected 2D bounding box from 3D world position.

    This uses the same projection logic as draw_data to compute where the cube
    should appear in the image based on its world position.

    Args:
        position: 3D world position of the cube center [x, y, z].

    Returns:
        Dictionary with x_min, y_min, x_max, y_max pixel coordinates.
    """
    # Project 3D position to 2D image coordinates
    # The cube has scale [1.0, 2.0, 0.2] and size 1.0, so actual dimensions are [1.0, 2.0, 0.2] meters
    # The bounding box should capture the full extent of the cube
    center_x = RESOLUTION[1] / 2 + PIXELS_PER_METER * position[0]
    center_y = RESOLUTION[1] / 2 + PIXELS_PER_METER * position[1]

    # Half-widths in pixels (cube scale is [1.0, 2.0, 0.2])
    half_width_x = PIXELS_PER_METER / 2  # 0.5 meters on each side
    half_width_y = PIXELS_PER_METER  # 1.0 meters on each side

    return {
        "x_min": int(center_x - half_width_x),
        "y_min": int(center_y - half_width_y),
        "x_max": int(center_x + half_width_x),
        "y_max": int(center_y + half_width_y),
    }


def validate_bbox(detected_bbox: dict, expected_bbox: dict, tolerance_pixels: float = 2.0) -> tuple[bool, float]:
    """Validate that detected bounding box matches expected position within tolerance.

    Args:
        detected_bbox: Detected bounding box from the sensor/writer.
        expected_bbox: Expected bounding box calculated from world position.
        tolerance_pixels: Maximum allowed error in pixels for bbox center.

    Returns:
        Tuple of (is_valid, error_distance) where error_distance is the pixel distance
        between detected and expected bbox centers.
    """
    # Calculate centers
    detected_center_x = (detected_bbox["x_min"] + detected_bbox["x_max"]) / 2
    detected_center_y = (detected_bbox["y_min"] + detected_bbox["y_max"]) / 2

    expected_center_x = (expected_bbox["x_min"] + expected_bbox["x_max"]) / 2
    expected_center_y = (expected_bbox["y_min"] + expected_bbox["y_max"]) / 2

    # Calculate Euclidean distance between centers
    error_distance = np.sqrt(
        (detected_center_x - expected_center_x) ** 2 + (detected_center_y - expected_center_y) ** 2
    )

    is_valid = error_distance <= tolerance_pixels

    return is_valid, error_distance


def draw_data(frame, position, bbox, label):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.rectangle(
        img=frame,
        pt1=(
            int(RESOLUTION[1] / 2 + PIXELS_PER_METER * position[0] - PIXELS_PER_METER / 2),
            int(RESOLUTION[1] / 2 + PIXELS_PER_METER * position[1] - PIXELS_PER_METER),
        ),
        pt2=(
            int(RESOLUTION[1] / 2 + PIXELS_PER_METER * position[0] + PIXELS_PER_METER / 2),
            int(RESOLUTION[1] / 2 + PIXELS_PER_METER * position[1] + PIXELS_PER_METER),
        ),
        color=(255, 255, 0),
        thickness=1,
        lineType=cv2.LINE_AA,
    )
    frame = cv2.rectangle(
        img=frame,
        pt1=(bbox["x_min"], bbox["y_min"]),
        pt2=(bbox["x_max"], bbox["y_max"]),
        color=(0, 255, 0),
        thickness=1,
        lineType=cv2.LINE_AA,
    )
    frame = cv2.putText(
        img=frame,
        text=label,
        org=(5, 15),
        fontFace=cv2.FONT_HERSHEY_PLAIN,
        fontScale=0.75,
        color=(0, 255, 255),
        thickness=1,
        lineType=cv2.LINE_AA,
    )
    frame = cv2.putText(
        img=frame,
        text=f"position: {(round(position[0], 2), round(position[1], 2), round(position[2], 2))}",
        org=(5, 30),
        fontFace=cv2.FONT_HERSHEY_PLAIN,
        fontScale=0.75,
        color=(255, 255, 0),
        thickness=1,
        lineType=cv2.LINE_AA,
    )
    frame = cv2.putText(
        img=frame,
        text=f'bbox: {(bbox["x_min"], bbox["y_min"])} {(bbox["x_max"], bbox["y_max"])}',
        org=(5, 45),
        fontFace=cv2.FONT_HERSHEY_PLAIN,
        fontScale=0.75,
        color=(0, 255, 0),
        thickness=1,
        lineType=cv2.LINE_AA,
    )
    return frame


def generate_result(data: list[dict], banner: list[str] = []):
    rgb_frames = []
    semantic_segmentation_frames = []
    for item in data:
        rgb_frames.append(draw_data(item["rgb"], item["position"], item["bbox"], item["label"]))
        semantic_segmentation_frames.append(
            draw_data(item["semantic_segmentation"], item["position"], item["bbox"], item["label"])
        )

    separator = np.full((RESOLUTION[0], 5, 3), 0, dtype=np.uint8)
    rgb_frames = [x for item in rgb_frames for x in (item, separator)][:-1]
    semantic_segmentation_frames = [x for item in semantic_segmentation_frames for x in (item, separator)][:-1]

    frame = cv2.vconcat([cv2.hconcat(rgb_frames), cv2.hconcat(semantic_segmentation_frames)])
    if banner:
        frame = cv2.copyMakeBorder(
            frame, top=25, bottom=0, left=0, right=0, borderType=cv2.BORDER_CONSTANT, value=[0] * 3
        )
        frame = cv2.putText(
            img=frame,
            text=", ".join(banner),
            org=(5, 15),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=0.75,
            color=(255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
    return frame


simulation_app.update()

# Setup scene
ViewportManager.set_camera_view("/OmniverseKit_Persp", eye=CAMERA_POS, target=[0, 0, 0])

GroundPlane("/World/ground_plane")

# Create cube with physics
cube_prim = Cube(
    "/cube",
    positions=[[-3.0, 0.0, 0.1]],
    scales=[[1.0, 2.0, 0.2]],
    sizes=1.0,
)
UsdPhysics.RigidBodyAPI.Apply(cube_prim.prims[0])
UsdPhysics.CollisionAPI.Apply(cube_prim.prims[0])
rigid_cube = RigidPrim("/cube")
add_labels(cube_prim.prims[0], labels=["cube"], taxonomy="class")

camera = None
writer = None
if USE_REPLICATOR_WRITER:
    stage = omni.usd.get_context().get_stage()
    camera_prim = stage.DefinePrim(CAMERA_PATH, "Camera")
    UsdGeom.Xformable(camera_prim).AddTranslateOp().Set(tuple(CAMERA_POS))
    render_product = rep.create.render_product(str(camera_prim.GetPrimPath()), resolution=RESOLUTION)
else:
    camera = Camera(
        prim_path=CAMERA_PATH,
        position=np.array(CAMERA_POS),
        resolution=RESOLUTION,
        orientation=transform_utils.euler_angles_to_quaternion(np.array([[0, 90, 90]]), degrees=True).numpy().flatten(),
    )

app_utils.play()

if USE_REPLICATOR_WRITER:
    rep.WriterRegistry.register(CustomWriter)
    writer = rep.WriterRegistry.get("CustomWriter")
    writer.initialize()
    writer.attach([render_product])
else:
    camera.initialize()
    camera.add_bounding_box_2d_tight_to_frame()
    camera.add_semantic_segmentation_to_frame()


# Do some warmup steps
for _ in range(5):
    simulation_app.update()

data = []
validation_errors = []

# Get data and object info before running the collection steps
position = rigid_cube.get_world_poses()[0].numpy().flatten()
rgb, semantic_segmentation, bbox = get_data(camera or writer)
data.append(
    {"position": position, "rgb": rgb, "semantic_segmentation": semantic_segmentation, "bbox": bbox, "label": "before"}
)

# Validate initial frame
expected_bbox = calculate_expected_bbox(position)
is_valid, error_distance = validate_bbox(bbox, expected_bbox, tolerance_pixels=2.0)
if not is_valid:
    error_msg = f"Frame 'before': BBox center error = {error_distance:.2f} pixels (tolerance: 2.0 pixels)"
    validation_errors.append(error_msg)
    print(f"[error] {error_msg}")

# Do some collection steps
for i in range(COLLECTION_STEPS):
    # Move object
    position = rigid_cube.get_world_poses()[0].numpy().flatten()
    position[0] += 0.5
    rigid_cube.set_world_poses(positions=[position])

    # Step the simulation
    simulation_app.update()

    # Get data and object info
    position = rigid_cube.get_world_poses()[0].numpy().flatten()
    rgb, semantic_segmentation, bbox = get_data(camera or writer)
    data.append(
        {
            "position": position,
            "rgb": rgb,
            "semantic_segmentation": semantic_segmentation,
            "bbox": bbox,
            "label": f"step {i + 1}",
        }
    )

    # Validate this frame
    expected_bbox = calculate_expected_bbox(position)
    is_valid, error_distance = validate_bbox(bbox, expected_bbox, tolerance_pixels=2.0)
    if not is_valid:
        error_msg = f"Frame {i + 1}: BBox center error = {error_distance:.2f} pixels (tolerance: 2.0 pixels)"
        validation_errors.append(error_msg)
        print(f"[error] {error_msg}")

# Export result
banner = [
    f"source: {'rep.Writer' if USE_REPLICATOR_WRITER else 'Camera'}",
    f"checkForHydraRenderComplete: {carb.settings.get_settings().get('/app/updateOrder/checkForHydraRenderComplete')}",
    f"app.hydraEngine.waitIdle: {carb.settings.get_settings().get('/app/hydraEngine/waitIdle')}",
    f"rtx.post.aa.op: {carb.settings.get_settings().get('/rtx/post/aa/op')}",
]
print("")
pprint.pprint(banner)
print("")
cv2.imwrite(f"result-{args.resolution}.png", generate_result(data, banner))

# Print validation summary
print("\n" + "=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)
if validation_errors:
    print(f"[fatal] {len(validation_errors)} frame(s) had bounding box errors exceeding tolerance")
    print("\nErrors:")
    for error in validation_errors:
        print(f"  - {error}")
    print("\nThis indicates frame delay between object motion and rendered image.")
    print("=" * 80)
    simulation_app.close()
    sys.exit(1)
else:
    print(f"SUCCESS: All {len(data)} frames passed validation")
    print("Bounding boxes matched expected positions within 2.0 pixel tolerance")
    print("No frame delay detected between object motion and rendered image")
    print("=" * 80)
    simulation_app.close()
    sys.exit(0)
