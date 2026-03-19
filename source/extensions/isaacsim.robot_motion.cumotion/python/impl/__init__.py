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
Motion generation extension: defines interfaces to work with IsaacSim.
"""

import os

from .configuration_loader import (
    CumotionRobot,
    load_cumotion_robot,
    load_cumotion_supported_robot,
)
from .cumotion_trajectory import CumotionTrajectory
from .cumotion_world_interface import CumotionWorldInterface
from .graph_based_motion_planner import GraphBasedMotionPlanner
from .rmp_flow_controller import RmpFlowController
from .trajectory_generator import TrajectoryGenerator

# temporary: TrajectoryOptimizer does not work on Windows.
if os.name != "nt":
    from .trajectory_optimizer import TrajectoryOptimizer
