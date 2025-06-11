// SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <carb/BindingsUtils.h>

#include <isaacsim/robot/surface_gripper/ISurfaceGripper.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

CARB_BINDINGS("isaacsim.robot.surface_gripper.python")

namespace isaacsim
{
namespace robot
{
namespace surface_gripper
{
} // namespace surface_gripper
} // namespace robot
} // namespace isaacsim

namespace
{

namespace py = pybind11;

template <typename InterfaceType>
py::class_<InterfaceType> defineInterfaceClass(py::module& m,
                                               const char* className,
                                               const char* acquireFuncName,
                                               const char* releaseFuncName = nullptr,
                                               const char* docString = "")
{
    m.def(
        acquireFuncName,
        [](const char* pluginName, const char* libraryPath)
        {
            return libraryPath ? carb::acquireInterfaceFromLibraryForBindings<InterfaceType>(libraryPath) :
                                 carb::acquireInterfaceForBindings<InterfaceType>(pluginName);
        },
        py::arg("plugin_name") = nullptr, py::arg("library_path") = nullptr, py::return_value_policy::reference,
        "Acquire Surface Gripper interface. This is the base object that all of the Surface Gripper functions are defined on");

    if (releaseFuncName)
    {
        m.def(
            releaseFuncName, [](InterfaceType* iface) { carb::getFramework()->releaseInterface(iface); },
            "Release Surface Gripper interface. Generally this does not need to be called, the Surface Gripper interface is released on extension shutdown");
    }

    return py::class_<InterfaceType>(m, className, docString);
}


PYBIND11_MODULE(_surface_gripper, m)
{
    using namespace carb;
    using namespace isaacsim::robot::surface_gripper;
    // We use carb data types, must import bindings for them
    auto carbModule = py::module::import("carb");

    defineInterfaceClass<SurfaceGripperInterface>(
        m, "SurfaceGripperInterface", "acquire_surface_gripper_interface", "release_surface_gripper_interface")
        .def(
            "set_gripper_action",
            [](const SurfaceGripperInterface* iface, const char* primPath, const float action) -> bool
            { return iface ? iface->SetGripperAction(primPath, action) : false; },
            py::arg("prim_path"), py::arg("action"), "Sets the action of a surface gripper at the specified USD path.")
        .def(
            "open_gripper",
            [](const SurfaceGripperInterface* iface, const char* primPath) -> bool
            { return iface ? iface->OpenGripper(primPath) : false; },
            py::arg("prim_path"), "Opens a surface gripper at the specified USD path.")
        .def(
            "close_gripper",
            [](const SurfaceGripperInterface* iface, const char* primPath) -> bool
            { return iface ? iface->CloseGripper(primPath) : false; },
            py::arg("prim_path"), "Closes a surface gripper at the specified USD path.")
        .def(
            "get_gripper_status",
            [](const SurfaceGripperInterface* iface, const char* primPath) -> std::string
            { return iface ? iface->GetGripperStatus(primPath) : "<FAILURE TO LOAD INTERFACE>"; },
            py::arg("prim_path"), "Gets the status of a surface gripper at the specified USD path.")
        // .def(
        //     "get_all_grippers",
        //     [](const SurfaceGripperInterface* iface) -> std::vector<std::string>
        //     {
        //         return iface ? iface->getAllGrippers() : std::vector<std::string>();
        //     },
        //     "Gets a list of all surface grippers in the scene.")
        .def(
            "get_gripped_objects",
            [](const SurfaceGripperInterface* iface, const char* primPath) -> std::vector<std::string>
            { return iface ? iface->GetGrippedObjects(primPath) : std::vector<std::string>(); },
            py::arg("prim_path"), "Gets a list of objects currently gripped by the specified gripper.");
}
}
