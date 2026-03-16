# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import os
import sys

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import omni.kit.app


def _enable_scene_optimizer_extension():
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    ext_manager.set_extension_enabled_immediate("omni.scene.optimizer.core", True)


_enable_scene_optimizer_extension()

from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig

parser = argparse.ArgumentParser(description="Import an MJCF file using Isaac Sim.")
parser.add_argument("--mjcf", required=False, default=None, help="Path to the MJCF file (.xml) to import.")
parser.add_argument(
    "--usd-path",
    required=False,
    default=None,
    help="Directory to write converted USD assets.",
)
parser.add_argument(
    "--import-scene",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Import the MJCF simulation settings along with the model.",
)
parser.add_argument(
    "--merge-mesh",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Merge meshes after conversion.",
)
parser.add_argument(
    "--debug-mode",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Enable debug mode and keep intermediate outputs.",
)
parser.add_argument(
    "--collision-from-visuals",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Generate collision geometry from visuals.",
)
parser.add_argument(
    "--collision-type",
    default=None,
    help="Collision geometry type (e.g. default, Convex Hull, Convex Decomposition).",
)
parser.add_argument(
    "--allow-self-collision",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Allow self-collision for the imported asset.",
)
parser.add_argument(
    "--test",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Run in test mode: uses nv_ant.xml test asset into a temp directory",
)
args, unknown = parser.parse_known_args()


def main():
    """Run the MJCF import workflow with CLI configuration.

    Returns:
        Exit code integer.

    """
    try:
        import_config = MJCFImporterConfig()

        if args.test:
            ext_manager = omni.kit.app.get_app().get_extension_manager()
            ext_id = ext_manager.get_enabled_extension_id("isaacsim.asset.importer.mjcf")
            extension_path = ext_manager.get_extension_path(ext_id)
            args.mjcf = os.path.join(extension_path, "data", "mjcf", "nv_ant.xml")
        else:
            if not os.path.isabs(args.mjcf):
                args.mjcf = os.path.abspath(args.mjcf)

            if not os.path.exists(args.mjcf):
                raise RuntimeError(f"MJCF file not found: {args.mjcf}")

        import_config.mjcf_path = args.mjcf

        if args.usd_path is not None:
            import_config.usd_path = os.path.abspath(args.usd_path)

        if args.import_scene is not None:
            import_config.import_scene = args.import_scene
        if args.merge_mesh is not None:
            import_config.merge_mesh = args.merge_mesh
        if args.debug_mode is not None:
            import_config.debug_mode = args.debug_mode
        if args.collision_from_visuals is not None:
            import_config.collision_from_visuals = args.collision_from_visuals
        if args.collision_type is not None:
            import_config.collision_type = args.collision_type
        if args.allow_self_collision is not None:
            import_config.allow_self_collision = args.allow_self_collision

        importer = MJCFImporter(import_config)
        output_usd = importer.import_mjcf()
        if not output_usd:
            raise RuntimeError("MJCF import failed.")

        print(f"MJCF import successful. Output USD file: {output_usd}")

        simulation_app.update()
        simulation_app.update()
        while omni.usd.get_context().get_stage_loading_status()[2] > 0:
            simulation_app.update()

        simulation_app.close()
    except Exception as e:
        print(f"Error: {e}")
        simulation_app.close()


if __name__ == "__main__":
    main()
