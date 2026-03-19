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

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import omni.kit.app


def _enable_scene_optimizer_extension():
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    ext_manager.set_extension_enabled_immediate("omni.scene.optimizer.core", True)


_enable_scene_optimizer_extension()

from isaacsim.asset.importer.urdf.impl import URDFImporter, URDFImporterConfig

parser = argparse.ArgumentParser(description="Import a URDF file using Isaac Sim.")
parser.add_argument("--urdf", required=False, default=None, help="Path to the URDF file (.urdf) to import.")
parser.add_argument(
    "--usd-path",
    required=False,
    default=None,
    help="Directory to write converted USD assets.",
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
    help="Run in test mode: uses carter.urdf test asset into a temp directory",
)
parser.add_argument(
    "--ros-package",
    action="append",
    metavar="NAME:PATH",
    help="ROS package mapping in format 'name:path'. Can be specified multiple times for multiple packages.",
)
args, unknown = parser.parse_known_args()


def main():
    """Run the URDF import workflow with CLI configuration.

    Returns:
        Exit code integer.

    """
    try:
        import_config = URDFImporterConfig()

        if args.test:
            ext_manager = omni.kit.app.get_app().get_extension_manager()
            ext_id = ext_manager.get_enabled_extension_id("isaacsim.asset.importer.urdf")
            extension_path = ext_manager.get_extension_path(ext_id)
            args.urdf = os.path.join(extension_path, "data", "urdf", "robots", "carter", "urdf", "carter.urdf")
        else:
            if args.urdf is None:
                raise RuntimeError(
                    "URDF file path is required. Use --urdf to specify a file or --test to use a test asset."
                )
            if not os.path.isabs(args.urdf):
                args.urdf = os.path.abspath(args.urdf)

            if not os.path.exists(args.urdf):
                raise RuntimeError(f"URDF file not found: {args.urdf}")

        import_config.urdf_path = args.urdf

        if args.usd_path is not None:
            import_config.usd_path = os.path.abspath(args.usd_path)

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

        if args.ros_package:
            ros_packages = []
            for package_spec in args.ros_package:
                if ":" not in package_spec:
                    raise ValueError(f"Invalid ROS package format: {package_spec}. Expected format: 'name:path'")
                name, path = package_spec.split(":", 1)
                ros_packages.append({"name": name.strip(), "path": path.strip()})
            import_config.ros_package_paths = ros_packages

        importer = URDFImporter(import_config)
        output_usd = importer.import_urdf()
        if not output_usd:
            raise RuntimeError("URDF import failed.")

        print(f"URDF import successful. Output USD file: {output_usd}")
        simulation_app.close()
    except Exception as e:
        print(f"Error: {e}")
        simulation_app.close()


if __name__ == "__main__":
    main()
