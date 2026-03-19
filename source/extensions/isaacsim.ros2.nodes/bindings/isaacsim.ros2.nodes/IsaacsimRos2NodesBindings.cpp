// SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <carb/BindingsPythonUtils.h>

#include <isaacsim/ros2/nodes/IRos2Nodes.h>

CARB_BINDINGS("isaacsim.ros2.nodes.python")

namespace
{

/**
 * @brief Python bindings for the ROS2 Nodes module
 *
 * Provides Python interface access to the ROS 2 nodes functionality
 * through pybind11 bindings.
 */
PYBIND11_MODULE(_ros2_nodes, m)
{
    // clang-format off
    using namespace carb;
    using namespace isaacsim::ros2::nodes;

    m.doc() = R"pbdoc(
        Internal interface that is automatically called when the extension is loaded so that Omnigraph nodes are registered.

        Example:

            # import  isaacsim.ros2.nodes.bindings._ros2_nodes as _ros2_nodes

            # Acquire the interface
            interface = _ros2_nodes.acquire_interface()

            # Use the interface
            # ...

            # Release the interface
            _ros2_nodes.release_interface(interface)
    )pbdoc";

    defineInterfaceClass<IRos2Nodes>(
        m,
        "IRos2Nodes",
        "acquire_interface",
        "release_interface"
    );
}
} // namespace anonymous
