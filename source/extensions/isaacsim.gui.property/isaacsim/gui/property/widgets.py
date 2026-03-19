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

"""Extension entry point that registers Isaac property widgets."""

import omni.usd

from .array_widget import ArrayPropertiesWidget
from .custom_data import CustomDataWidget
from .motion_planning_schema import MotionPlanningAPIWidget
from .name_override import NameOverrideWidget
from .namespace import NamespaceWidget
from .robot_schema import JointAPIWidget, LinkAPIWidget, RobotAPIWidget, SiteAPIWidget


class IsaacPropertyWidgets(omni.ext.IExt):
    """Extension that provides custom property widgets for Isaac Sim robotics workflows.

    This extension registers specialized property panel widgets in the USD Property Window to enhance
    the editing experience for robotics-specific USD prims and schemas. The widgets provide intuitive
    interfaces for configuring robot components, motion planning parameters, and other Isaac Sim-specific
    attributes.

    The extension registers the following property widgets:

    - Array Properties: Manages array-based properties on USD prims
    - Prim Custom Data: Handles custom metadata stored on prims
    - Name Override: Provides name override functionality for prims
    - Namespace: Manages namespace properties for organizational purposes
    - Robot Schema: Configures robot-level properties and parameters
    - Robot Link: Handles link-specific properties for articulated robots
    - Robot Joint: Manages joint properties including limits, drives, and DOF settings
    - Motion Planning: Configures motion planning API parameters and constraints

    These widgets appear in the Property Window when relevant USD prims are selected, providing
    context-sensitive editing capabilities for robotics applications in Isaac Sim.
    """

    def __init__(self):
        super().__init__()
        self._registered = False
        self._robot_api_widget = None
        self._link_api_widget = None
        self._joint_api_widget = None
        self._site_api_widget = None
        self._motion_planning_api_widget = None

    def on_startup(self, ext_id: str) -> None:
        """Register all Isaac property widgets with the property window.

        Args:
            ext_id: The extension identifier.
        """
        self._register_widget()

    def on_shutdown(self) -> None:
        """Unregister all Isaac property widgets from the property window."""
        self._unregister_widget()

    def _register_widget(self):
        """Registers Isaac property widgets with the property window.

        Creates and registers widgets for array properties, custom data, name override, namespace,
        robot schema, robot link, robot joint, and motion planning.
        """
        import omni.kit.window.property as p

        w = p.get_window()
        w.register_widget("prim", "isaac_array", ArrayPropertiesWidget(title="Array Properties", collapsed=True), False)
        w.register_widget(
            "prim", "isaac_custom_data", CustomDataWidget(title="Prim Custom Data", collapsed=True), False
        )
        self._isaac_name_override = NameOverrideWidget(title="Name Override", collapsed=False)
        w.register_widget("prim", "isaac_name_override", self._isaac_name_override, False)
        self._isaac_namespace = NamespaceWidget(title="Namespace", collapsed=False)
        w.register_widget("prim", "isaac_namespace", self._isaac_namespace, False)
        self._robot_api_widget = RobotAPIWidget(title="Robot Schema", collapsed=False)
        w.register_widget("prim", "isaac_robot_api", self._robot_api_widget, False)
        self._link_api_widget = LinkAPIWidget(title="Robot Link", collapsed=False)
        w.register_widget("prim", "isaac_link_api", self._link_api_widget, False)
        self._joint_api_widget = JointAPIWidget(title="Robot Joint", collapsed=False)
        w.register_widget("prim", "isaac_joint_api", self._joint_api_widget, False)
        self._site_api_widget = SiteAPIWidget(title="Robot Site", collapsed=False)
        w.register_widget("prim", "isaac_site_api", self._site_api_widget, False)
        self._motion_planning_api_widget = MotionPlanningAPIWidget(title="Motion Planning", collapsed=False)
        w.register_widget("prim", "isaac_motion_planning_api", self._motion_planning_api_widget, False)

    def _unregister_widget(self):
        """Unregisters Isaac property widgets from the property window.

        Removes all registered widgets and destroys their instances to free up resources.
        """
        import omni.kit.window.property as p

        w = p.get_window()
        if w:
            w.unregister_widget("prim", "isaac_array")
            w.unregister_widget("prim", "isaac_custom_data")
            w.unregister_widget("prim", "isaac_name_override")
            w.unregister_widget("prim", "isaac_namespace")
            w.unregister_widget("prim", "isaac_robot_api")
            w.unregister_widget("prim", "isaac_link_api")
            w.unregister_widget("prim", "isaac_joint_api")
            w.unregister_widget("prim", "isaac_site_api")
            w.unregister_widget("prim", "isaac_motion_planning_api")
            self._isaac_name_override.destroy()
            self._isaac_namespace.destroy()
            if self._robot_api_widget:
                self._robot_api_widget.destroy()
            if self._link_api_widget:
                self._link_api_widget.destroy()
            if self._joint_api_widget:
                self._joint_api_widget.destroy()
            if self._site_api_widget:
                self._site_api_widget.destroy()
            if self._motion_planning_api_widget:
                self._motion_planning_api_widget.destroy()
            self._registered = False
