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

import os

from isaacsim import SimulationApp

simulation_app = SimulationApp(launch_config={"headless": False})

import carb.settings
import omni.replicator.core as rep
import omni.usd
from omni.replicator.core import Writer


# Create a custom writer to access annotator data
class MyWriter(Writer):
    def __init__(self, camera_params: bool = True, bounding_box_3d: bool = True):
        # Organize data from render product perspective (legacy, annotator, renderProduct)
        self.data_structure = "renderProduct"
        self.annotators = []
        if camera_params:
            self.annotators.append(rep.annotators.get("camera_params"))
        if bounding_box_3d:
            self.annotators.append(rep.annotators.get("bounding_box_3d"))
        self._frame_id = 0

    def write(self, data: dict):
        print(f"[MyWriter][{self._frame_id}] data:")
        for key, value in data.items():
            print(f"  {key}: {value}")
        self._frame_id += 1


# Register the writer
rep.writers.register_writer(MyWriter)


def run_example():
    # Create a new stage and disable capture on play
    omni.usd.get_context().new_stage()
    rep.orchestrator.set_capture_on_play(False)

    # Set DLSS to Quality mode (2) for best SDG results , options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
    carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

    # Setup stage
    rep.functional.create.xform(name="World")
    rep.functional.create.dome_light(intensity=500, parent="/World", name="DomeLight")
    cube = rep.functional.create.cube(parent="/World", name="Cube")
    rep.functional.modify.semantics(cube, {"class": "my_cube"}, mode="add")

    # Capture from two perspectives, a custom camera and a perspective camera
    top_cam = rep.functional.create.camera(position=(0, 0, 5), look_at=(0, 0, 0), parent="/World", name="TopCamera")
    persp_cam = rep.functional.create.camera(position=(5, 5, 5), look_at=(0, 0, 0), parent="/World", name="PerspCamera")

    # Create the render products
    rp_top = rep.create.render_product(top_cam.GetPath(), (400, 400), name="top_view")
    rp_persp = rep.create.render_product(persp_cam.GetPath(), (512, 512), name="persp_view")

    # Use the annotators to access the data directly, each annotator is attached to a render product
    rgb_annotator_top = rep.annotators.get("rgb")
    rgb_annotator_top.attach(rp_top)
    rgb_annotator_persp = rep.annotators.get("rgb")
    rgb_annotator_persp.attach(rp_persp)

    # Use the custom writer to access the annotator data
    custom_writer = rep.writers.get("MyWriter")
    custom_writer.initialize(camera_params=True, bounding_box_3d=True)
    custom_writer.attach([rp_top, rp_persp])

    # Use the pose writer to write the data to disk
    pose_writer = rep.WriterRegistry.get("PoseWriter")
    out_dir = os.path.join(os.getcwd(), "_out_pose_writer")
    print(f"Output directory: {out_dir}")
    pose_writer.initialize(output_dir=out_dir, write_debug_images=True)
    pose_writer.attach([rp_top, rp_persp])

    # Trigger a data capture request (data will be written to disk by the writer)
    for i in range(3):
        print(f"Step {i}")
        rep.orchestrator.step()

        # Get the data from the annotators
        rgb_data_top = rgb_annotator_top.get_data()
        rgb_data_persp = rgb_annotator_persp.get_data()
        print(f"[Annotator][Top][{i}] rgb_data_top shape: {rgb_data_top.shape}")
        print(f"[Annotator][Persp][{i}] rgb_data_persp shape: {rgb_data_persp.shape}")

    # Wait for the data to be written to disk and clean up resources
    rep.orchestrator.wait_until_complete()
    pose_writer.detach()
    custom_writer.detach()
    rgb_annotator_top.detach()
    rgb_annotator_persp.detach()
    rp_top.destroy()
    rp_persp.destroy()


run_example()

simulation_app.close()
