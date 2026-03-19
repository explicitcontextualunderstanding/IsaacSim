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

"""UI extension for Isaac Sim Replicator behavior management with property widgets for exposed variables."""


import omni.ext
import omni.kit.window.property
from isaacsim.replicator.behavior.global_variables import EXPOSED_ATTR_NS

from .exposed_variables_widget import ExposedVariablesPropertyWidget
from .global_variables import WIDGET_NAME, WIDGET_TITLE


class Extension(omni.ext.IExt):
    """Extension for Isaac Sim Replicator Behavior UI integration.

    This extension provides user interface components for managing behavior-related functionality within Isaac Sim's Replicator framework. It registers a custom property widget that allows users to interact with exposed variables in the Property window, enhancing the workflow for behavior configuration and monitoring.

    The extension integrates with the Property window by adding an ExposedVariablesPropertyWidget that filters and displays attributes based on the exposed attribute namespace, providing a streamlined interface for behavior parameter management.
    """

    def __init__(self):
        super().__init__()
        self._registered = False

    def on_startup(self, ext_id):
        """Called when the extension is starting up.

        Args:
            ext_id: The extension identifier.
        """
        self._register_widget()

    def on_shutdown(self):
        """Called when the extension is shutting down."""
        if self._registered:
            self._unregister_widget()

    def _register_widget(self):
        """Registers the exposed variables property widget with the property window."""
        property_window = omni.kit.window.property.get_window()
        if property_window:
            property_window.register_widget(
                "prim",
                WIDGET_NAME,
                ExposedVariablesPropertyWidget(title=WIDGET_TITLE, attribute_namespace_filter=[EXPOSED_ATTR_NS]),
            )
            self._registered = True

    def _unregister_widget(self):
        """Unregisters the exposed variables property widget from the property window."""
        property_window = omni.kit.window.property.get_window()
        if property_window:
            property_window.unregister_widget("prim", WIDGET_NAME)
            self._registered = False
