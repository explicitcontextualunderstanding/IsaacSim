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


"""MJCF importer configuration utilities."""

from dataclasses import dataclass


@dataclass
class MJCFImporterConfig:
    """Configuration for MJCF import operations.

    Stores settings that control how MJCF files are converted to USD.

    Args:
        mjcf_path: Path to the MJCF (.xml) file to import.
        usd_path: Directory path where the USD file will be saved.
        import_scene: If True, imports the MJCF simulation settings along with the model.
        merge_mesh: If True, merges meshes where possible to optimize the model.
        debug_mode: If True, enables debug mode with additional logging and visualization.
        collision_from_visuals: If True, collision geometry is generated from visual geometries.
        collision_type: Type of collision geometry to use. Options: "Convex Hull", "Convex Decomposition", "Bounding Sphere", "Bounding Cube".
        allow_self_collision: If True, allows the model to collide with itself.

    Example:

    .. code-block:: python

        >>> from isaacsim.asset.importer.mjcf import MJCFImporterConfig

        >>> config = MJCFImporterConfig(
        ...     mjcf_path="/tmp/robot.xml",
        ...     usd_path="/tmp/output",
        ...     merge_mesh=True
        ... )
        >>> config.mjcf_path
        '/tmp/robot.xml'
    """

    mjcf_path: str | None = None
    usd_path: str | None = None
    import_scene: bool = True
    merge_mesh: bool = False
    debug_mode: bool = False
    collision_from_visuals: bool = False
    collision_type: str = "Convex Hull"
    allow_self_collision: bool = False
