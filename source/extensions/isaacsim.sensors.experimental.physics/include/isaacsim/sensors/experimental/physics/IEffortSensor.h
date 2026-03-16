// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include <carb/Interface.h>

#include <cstdint>

namespace isaacsim
{
namespace sensors
{
namespace experimental
{
namespace physics
{

/**
 * @struct EffortSensorReading
 * @brief Effort sensor reading with joint effort value, time, and validity.
 */
struct EffortSensorReading
{
    float value{ 0.0f }; ///< Joint effort (torque/force) value.
    float time{ 0.0f }; ///< Simulation time of this reading in seconds.
    bool isValid{ false }; ///< Whether this reading contains valid data.
};

/**
 * @struct IEffortSensor
 * @brief Carbonite interface for managing C++ effort sensors.
 * @details Provides engine-agnostic joint effort reading using IPrimDataReader
 * for articulation DOF efforts. Reads projected joint forces from the physics
 * engine for a specified joint.
 *
 * The plugin is self-driving: it subscribes to PhysX simulation events
 * (eResumed / eStopped) and physics step events internally, so it
 * initializes and processes each substep without any Python relay.
 *
 * Lifecycle:
 * 1. acquire via carb::getCachedInterface or pybind11 acquire function
 * 2. createSensor() for each joint prim path
 * 3. getSensorReading() to read data (sensors update automatically each substep)
 * 4. shutdown() on extension unload
 */
struct IEffortSensor
{
    CARB_PLUGIN_INTERFACE("isaacsim::sensors::experimental::physics::IEffortSensor", 1, 0);

    /**
     * @brief Shut down the manager, destroying all sensors and freeing resources.
     */
    virtual void shutdown() = 0;

    /**
     * @brief Create an effort sensor for the given joint prim.
     * @details Parses the joint path to find the parent articulation root,
     * creates an IArticulationDataView, and resolves the DOF index for the joint.
     * @param jointPrimPath USD path to the joint prim (e.g., "/Robot/Arm/RevoluteJoint").
     * @return Unique sensor ID (>= 0), or -1 on failure.
     */
    virtual int64_t createSensor(const char* jointPrimPath) = 0;

    /**
     * @brief Remove a sensor and free its resources.
     * @param sensorId ID returned by createSensor().
     */
    virtual void removeSensor(int64_t sensorId) = 0;

    /**
     * @brief Get the latest reading for a sensor.
     * @param sensorId ID returned by createSensor().
     * @return Sensor reading. isValid is false if sensor is disabled or not found.
     */
    virtual EffortSensorReading getSensorReading(int64_t sensorId) = 0;
};

} // namespace physics
} // namespace experimental
} // namespace sensors
} // namespace isaacsim
