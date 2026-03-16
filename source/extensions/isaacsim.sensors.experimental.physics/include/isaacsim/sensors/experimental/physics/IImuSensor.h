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
 * @struct ImuSensorReading
 * @brief IMU sensor reading with acceleration, angular velocity, and orientation.
 * @details All values are in the sensor's local frame except orientation, which is
 * in world frame. Quaternion is stored in wxyz order.
 */
struct ImuSensorReading
{
    float linearAccelerationX{ 0.0f }; ///< Linear acceleration along X axis in m/s^2.
    float linearAccelerationY{ 0.0f }; ///< Linear acceleration along Y axis in m/s^2.
    float linearAccelerationZ{ 0.0f }; ///< Linear acceleration along Z axis in m/s^2.
    float angularVelocityX{ 0.0f }; ///< Angular velocity around X axis in rad/s.
    float angularVelocityY{ 0.0f }; ///< Angular velocity around Y axis in rad/s.
    float angularVelocityZ{ 0.0f }; ///< Angular velocity around Z axis in rad/s.
    float orientationW{ 1.0f }; ///< Quaternion w component (world frame).
    float orientationX{ 0.0f }; ///< Quaternion x component (world frame).
    float orientationY{ 0.0f }; ///< Quaternion y component (world frame).
    float orientationZ{ 0.0f }; ///< Quaternion z component (world frame).
    float time{ 0.0f }; ///< Simulation time of this reading in seconds.
    bool isValid{ false }; ///< Whether this reading contains valid data.
};

/**
 * @struct ImuRawData
 * @brief Raw per-step IMU data stored in the circular buffer before filtering.
 */
struct ImuRawData
{
    float time{ 0.0f };
    float dt{ 0.0f };
    float linearVelocityX{ 0.0f };
    float linearVelocityY{ 0.0f };
    float linearVelocityZ{ 0.0f };
    float angularVelocityX{ 0.0f };
    float angularVelocityY{ 0.0f };
    float angularVelocityZ{ 0.0f };
    float orientationW{ 1.0f };
    float orientationX{ 0.0f };
    float orientationY{ 0.0f };
    float orientationZ{ 0.0f };
};

/**
 * @struct IImuSensor
 * @brief Carbonite interface for managing C++ IMU sensors.
 * @details Provides engine-agnostic IMU simulation using IPrimDataReader for
 * rigid body velocities and Pose.h for sensor world transforms. Supports
 * configurable rolling average filters and gravity inclusion.
 *
 * The plugin is self-driving: it subscribes to PhysX simulation events
 * (eResumed / eStopped) and physics step events internally, so it
 * initializes, discovers sensors, and processes each substep without
 * any Python relay.
 *
 * Lifecycle:
 * 1. acquire via carb::getCachedInterface or pybind11 acquire function
 * 2. createSensor() for each IsaacImuSensor prim (plugin self-initializes on eResumed)
 * 3. getSensorReading() to read data (sensors update automatically each substep)
 * 4. shutdown() on extension unload
 */
struct IImuSensor
{
    CARB_PLUGIN_INTERFACE("isaacsim::sensors::experimental::physics::IImuSensor", 1, 0);

    /**
     * @brief Shut down the manager, destroying all sensors and freeing resources.
     */
    virtual void shutdown() = 0;

    /**
     * @brief Create an IMU sensor for the given IsaacImuSensor prim.
     * @details Finds the parent rigid body by walking up the prim hierarchy and
     * creates an IRigidBodyDataView for velocity data.
     * @param primPath USD path to the IsaacImuSensor prim.
     * @return Unique sensor ID (>= 0), or -1 on failure.
     */
    virtual int64_t createSensor(const char* primPath) = 0;

    /**
     * @brief Remove a sensor and free its resources.
     * @param sensorId ID returned by createSensor().
     */
    virtual void removeSensor(int64_t sensorId) = 0;

    /**
     * @brief Get the latest reading for a sensor.
     * @param sensorId ID returned by createSensor().
     * @param readGravity If true, include gravity in acceleration output.
     * @return Sensor reading. isValid is false if sensor is disabled or not found.
     */
    virtual ImuSensorReading getSensorReading(int64_t sensorId, bool readGravity) = 0;
};

} // namespace physics
} // namespace experimental
} // namespace sensors
} // namespace isaacsim
