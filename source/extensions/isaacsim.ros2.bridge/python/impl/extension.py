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

import carb
import omni.ext


class ROS2BridgeExtension(omni.ext.IExt):
    """ROS 2 Bridge Extension - Extension for ROS 2 integration.

    This extension brings together all ROS 2 extensions required for the ROS 2 bridge extension.
    """

    def on_startup(self, ext_id):

        carb.log_info("Starting ROS 2 Bridge extension")

    def on_shutdown(self):

        carb.log_info("Shutting down ROS 2 Bridge extension")
