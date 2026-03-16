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


"""Extension for Isaac Sim wheeled robots user interface components and differential drive controller configuration."""


import gc

import omni.ext
from omni.kit.menu.utils import MenuHelperExtensionFull

from .menu_graphs import DifferentialControllerWindow


class Extension(omni.ext.IExt, MenuHelperExtensionFull):
    """Extension for the Isaac Sim Wheeled Robots UI.

    This extension provides user interface components for working with wheeled robots in Isaac Sim.
    It adds a "Differential Controller" menu item under "Tools/Robotics/OmniGraph Controllers"
    that opens a window for configuring differential drive controllers for wheeled robots.
    """

    def on_startup(self, ext_id: str):
        """Called when the extension starts up.

        Creates menu entry for the Differential Controller window under Tools/Robotics/OmniGraph Controllers.

        Args:
            ext_id: The extension identifier.
        """

        # Create menu using MenuHelperExtensionFull
        self.menu_startup(
            lambda: DifferentialControllerWindow(),
            "Differential Controller",
            "Differential Controller",
            "Tools/Robotics/OmniGraph Controllers",
        )

    def on_shutdown(self):
        """Called when the extension shuts down.

        Cleans up menu entries and performs garbage collection.
        """
        self.menu_shutdown()
        gc.collect()
