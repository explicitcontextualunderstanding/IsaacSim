// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <pch/UsdPCH.h>
// clang-format on

#include <carb/BindingsPythonUtils.h>

#include <isaacsim/core/simulation_manager/ISimulationManager.h>
#include <pybind11/functional.h>


CARB_BINDINGS("isaacsim.core.simulation_manager.python")

namespace
{

PYBIND11_MODULE(_simulation_manager, m)
{
    using namespace isaacsim::core::simulation_manager;

    m.doc() = R"pbdoc(Omniverse Isaac Sim Simulation Manager Interface

This module provides access to the Simulation Manager which handles events and callbacks
for simulation-related activities, such as physics scene additions and object deletions.
It also manages USD notice handling to track stage changes.)pbdoc";

    // carb interface
    carb::defineInterfaceClass<ISimulationManager>(
        m, "ISimulationManager", "acquire_simulation_manager_interface", "release_simulation_manager_interface")
        .def("register_deletion_callback", &ISimulationManager::registerDeletionCallback,
             R"pbdoc(
                Register a callback for deletion events.

                Args:
                    callback: Function to be called when an object is deleted. Takes a string path parameter.

                Returns:
                    int: Unique identifier for the registered callback.
             )pbdoc")
        .def("register_physics_scene_addition_callback", &ISimulationManager::registerPhysicsSceneAdditionCallback,
             R"pbdoc(
                Register a callback for physics scene addition events.

                Args:
                    callback: Function to be called when a physics scene is added. Takes a string path parameter.

                Returns:
                    int: Unique identifier for the registered callback.
             )pbdoc")
        .def("deregister_callback", &ISimulationManager::deregisterCallback,
             R"pbdoc(
                Deregister a previously registered callback.

                Args:
                    callback_id: The unique identifier of the callback to deregister.

                Returns:
                    bool: True if callback was successfully deregistered, False otherwise.
             )pbdoc")
        .def("reset", &ISimulationManager::reset,
             R"pbdoc(
                Reset the simulation manager to its initial state.

                Calls all registered deletion callbacks with the root path ('/'),
                clears all registered callbacks, clears the physics scenes list,
                and resets the callback iterator to 0.
             )pbdoc")
        .def("set_callback_iter", &ISimulationManager::setCallbackIter,
             R"pbdoc(
                Set the callback iteration counter.

                Args:
                    val: New value for the callback iteration counter.
             )pbdoc")
        .def("get_callback_iter", &ISimulationManager::getCallbackIter,
             R"pbdoc(
                Get the current callback iteration counter.

                Returns:
                    int: The current callback iteration counter value.
             )pbdoc")
        .def("enable_usd_notice_handler", &ISimulationManager::enableUsdNoticeHandler,
             R"pbdoc(
                Enable or disable the USD notice handler.

                Args:
                    flag: True to enable the handler, False to disable.
             )pbdoc")
        .def("enable_fabric_usd_notice_handler", &ISimulationManager::enableFabricUsdNoticeHandler,
             R"pbdoc(
                Enable or disable the USD notice handler for a specific fabric stage.

                Args:
                    stage_id: ID of the fabric stage.
                    flag: True to enable the handler, False to disable.
             )pbdoc")
        .def("is_fabric_usd_notice_handler_enabled", &ISimulationManager::isFabricUsdNoticeHandlerEnabled,
             R"pbdoc(
                Check if the USD notice handler is enabled for a specific fabric stage.

                Args:
                    stage_id: ID of the fabric stage to check.

                Returns:
                    bool: True if the handler is enabled for the stage, False otherwise.
             )pbdoc")
        .def("get_simulation_time", &ISimulationManager::getSimulationTime,
             R"pbdoc(
                Get the current simulation time.

                Returns:
                    double: The current simulation time.
             )pbdoc")
        .def("get_simulation_time_monotonic", &ISimulationManager::getSimulationTimeMonotonic,
             R"pbdoc(
                Get the current simulation time. This time does not reset when the simulation is stopped.

                Returns:
                    double: The current simulation time.
             )pbdoc")
        .def("get_system_time", &ISimulationManager::getSystemTime,
             R"pbdoc(
                Get the current system time.

                Returns:
                    double: The current system time.
             )pbdoc")
        .def("get_num_physics_steps", &ISimulationManager::getNumPhysicsSteps,
             R"pbdoc(
                Get the current physics step count.

                Returns:
                    int: The current physics step count.
             )pbdoc")
        .def("is_simulating", &ISimulationManager::isSimulating,
             R"pbdoc(
                Get the current simulation pause state.

                Returns:
                    bool: True if the simulation is paused, False otherwise.
             )pbdoc")
        .def("is_paused", &ISimulationManager::isPaused,
             R"pbdoc(
                Get the current simulation pause state.

                Returns:
                    bool: True if the simulation is paused, False otherwise.
             )pbdoc")
        .def("get_simulation_time_at_time", &ISimulationManager::getSimulationTimeAtTime,
             R"pbdoc(
                Gets simulation time in seconds at a specific rational time.

                Args:
                    time (omni.fabric.RationalTime): The rational time to query.

                Returns:
                    float: The simulation time in seconds at the specified time.
             )pbdoc")
        .def(
            "get_simulation_time_at_time",
            [](ISimulationManager* iface, const std::tuple<int64_t, uint64_t>& timeTuple)
            {
                int64_t numerator = std::get<0>(timeTuple);
                uint64_t denominator = std::get<1>(timeTuple);
                omni::fabric::RationalTime time(numerator, denominator);
                return iface->getSimulationTimeAtTime(time);
            },
            R"pbdoc(
                Gets simulation time in seconds at a specific rational time.

                Args:
                    time (tuple): A tuple of (numerator, denominator) representing the rational time.

                Returns:
                    float: The simulation time in seconds at the specified time.
             )pbdoc")
        .def("get_simulation_time_monotonic_at_time", &ISimulationManager::getSimulationTimeMonotonicAtTime,
             R"pbdoc(
                Gets monotonic simulation time in seconds at a specific rational time.

                Args:
                    time (omni.fabric.RationalTime): The rational time to query.

                Returns:
                    float: The monotonic simulation time in seconds at the specified time.
             )pbdoc")
        .def(
            "get_simulation_time_monotonic_at_time",
            [](ISimulationManager* iface, const std::tuple<int64_t, uint64_t>& timeTuple)
            {
                int64_t numerator = std::get<0>(timeTuple);
                uint64_t denominator = std::get<1>(timeTuple);
                omni::fabric::RationalTime time(numerator, denominator);
                return iface->getSimulationTimeMonotonicAtTime(time);
            },
            R"pbdoc(
                Gets monotonic simulation time in seconds at a specific rational time.

                Args:
                    time (tuple): A tuple of (numerator, denominator) representing the rational time.

                Returns:
                    float: The monotonic simulation time in seconds at the specified time.
             )pbdoc")
        .def("get_system_time_at_time", &ISimulationManager::getSystemTimeAtTime,
             R"pbdoc(
                Gets system (real-world) time in seconds at a specific rational time.

                Args:
                    time (omni.fabric.RationalTime): The rational time to query.

                Returns:
                    float: The system time in seconds at the specified time.
             )pbdoc")
        .def(
            "get_system_time_at_time",
            [](ISimulationManager* iface, const std::tuple<int64_t, uint64_t>& timeTuple)
            {
                int64_t numerator = std::get<0>(timeTuple);
                uint64_t denominator = std::get<1>(timeTuple);
                omni::fabric::RationalTime time(numerator, denominator);
                return iface->getSystemTimeAtTime(time);
            },
            R"pbdoc(
                Gets system (real-world) time in seconds at a specific rational time.

                Args:
                    time (tuple): A tuple of (numerator, denominator) representing the rational time.

                Returns:
                    float: The system time in seconds at the specified time.
             )pbdoc");
}

}
