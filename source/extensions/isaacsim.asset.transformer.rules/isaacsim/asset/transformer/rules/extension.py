# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Register transformer rules with the global registry."""

import carb
import omni.ext
from isaacsim.asset.transformer import RuleRegistry

from .core.prims import PrimRoutingRule
from .core.properties import PropertyRoutingRule
from .core.remove_schema import RemoveSchemaRule
from .core.schemas import SchemaRoutingRule
from .isaac_sim.make_lists_non_explicit import MakeListsNonExplicitRule
from .isaac_sim.physics_joint_pose_fix import PhysicsJointPoseFixRule
from .isaac_sim.robot_schema import RobotSchemaRule
from .perf.geometries import GeometriesRoutingRule
from .perf.materials import MaterialsRoutingRule
from .structure.flatten import FlattenRule
from .structure.interface import InterfaceConnectionRule
from .structure.variants import VariantRoutingRule


class Extension(omni.ext.IExt):
    """Extension that registers transformation rules."""

    def on_startup(self, ext_id: str):
        """Register rule implementations with the global registry.

        Args:
            ext_id: Fully qualified extension identifier.
        """
        self._ext_id = ext_id
        carb.log_info(f"[isaacsim.asset.transformer.rules] Startup: {ext_id}")
        registry = RuleRegistry()
        registry.register(FlattenRule)
        registry.register(GeometriesRoutingRule)
        registry.register(MaterialsRoutingRule)
        registry.register(PhysicsJointPoseFixRule)
        registry.register(MakeListsNonExplicitRule)
        registry.register(RemoveSchemaRule)
        registry.register(PrimRoutingRule)
        registry.register(SchemaRoutingRule)
        registry.register(PropertyRoutingRule)
        registry.register(VariantRoutingRule)
        registry.register(RobotSchemaRule)
        registry.register(InterfaceConnectionRule)
        carb.log_info("[isaacsim.asset.transformer.rules] Rules registered")

    def on_shutdown(self):
        """Log shutdown for the rules extension."""
        carb.log_info(f"[isaacsim.asset.transformer.rules] Shutdown: {getattr(self, '_ext_id', '')}")
