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

// clang-format off
#include <pch/UsdPCH.h>
// clang-format on

#include <carb/Types.h>

#include <isaacsim/core/includes/Math.h>
#include <isaacsim/ucx/nodes/UcxPublishOdometryNodeBase.h>

#include <OgnUCXPublishOdometryDatabase.h>

using isaacsim::core::includes::math::operator*; // NOLINT(misc-unused-using-decls)

/**
 * @class OgnUCXPublishOdometry
 * @brief OmniGraph node for publishing odometry data via UCX.
 * @details
 * This node takes position, orientation, and velocity inputs and publishes odometry data
 * (pose and velocities relative to starting position) over UCX using tagged communication.
 * The odometry includes:
 * - Relative position and orientation (from starting pose)
 * - Body-frame linear and angular velocities
 * - Body-frame linear and angular accelerations
 */
class OgnUCXPublishOdometry : public UCXPublishOdometryNodeBase<OgnUCXPublishOdometryDatabase>
{
public:
    /**
     * @brief Release the node instance.
     * @details
     * Cleans up resources when the node instance is destroyed.
     *
     * @param[in] nodeObj The node object
     * @param[in] instanceId The instance ID
     */
    static void releaseInstance(NodeObj const& nodeObj, GraphInstanceID instanceId)
    {
        auto& state = OgnUCXPublishOdometryDatabase::sPerInstanceState<OgnUCXPublishOdometry>(nodeObj, instanceId);
        state.reset();
    }

    /**
     * @brief Compute function - called when node is executed.
     * @details
     * Extracts inputs, gets the per-instance state, and delegates to the base class logic.
     *
     * @param[in] db Database accessor for node inputs/outputs
     * @return bool True if execution succeeded, false otherwise
     */
    static bool compute(OgnUCXPublishOdometryDatabase& db)
    {
        const uint16_t port = static_cast<uint16_t>(db.inputs.port());
        const uint64_t tag = db.inputs.tag();
        const uint32_t timeoutMs = db.inputs.timeoutMs();

        // Get the per-instance state and call the instance method
        auto& state = db.template perInstanceState<OgnUCXPublishOdometry>();
        return state.computeImpl(db, port, tag, timeoutMs);
    }

protected:
    /**
     * @brief Extract odometry data from node inputs.
     * @details
     * Takes position, orientation, and velocity inputs, computes odometry relative
     * to starting pose, and returns the computed data.
     *
     * @param[in] db Database accessor for node inputs
     * @return OdometryData Computed odometry data
     */
    isaacsim::ucx::nodes::OdometryData extractData(OgnUCXPublishOdometryDatabase& db) override
    {
        isaacsim::ucx::nodes::OdometryData data;
        data.timestamp = db.inputs.timeStamp();

        // Get direct inputs from node
        auto& position = db.inputs.position();
        auto& orientation = db.inputs.orientation();
        auto& linearVelocity = db.inputs.linearVelocity();
        auto& angularVelocity = db.inputs.angularVelocity();

        // Current world pose and velocities
        float currentPosX = static_cast<float>(position[0]);
        float currentPosY = static_cast<float>(position[1]);
        float currentPosZ = static_cast<float>(position[2]);

        // Orientation is in IJKR format (x, y, z, w)
        float currentQuatX = static_cast<float>(orientation.GetImaginary()[0]);
        float currentQuatY = static_cast<float>(orientation.GetImaginary()[1]);
        float currentQuatZ = static_cast<float>(orientation.GetImaginary()[2]);
        float currentQuatW = static_cast<float>(orientation.GetReal());

        float linearVelX = static_cast<float>(linearVelocity[0]);
        float linearVelY = static_cast<float>(linearVelocity[1]);
        float linearVelZ = static_cast<float>(linearVelocity[2]);

        float angularVelX = static_cast<float>(angularVelocity[0]);
        float angularVelY = static_cast<float>(angularVelocity[1]);
        float angularVelZ = static_cast<float>(angularVelocity[2]);

        // Initialize starting pose on first frame
        if (m_firstFrame)
        {
            m_startingPosition[0] = currentPosX;
            m_startingPosition[1] = currentPosY;
            m_startingPosition[2] = currentPosZ;
            m_startingOrientation[0] = currentQuatX;
            m_startingOrientation[1] = currentQuatY;
            m_startingOrientation[2] = currentQuatZ;
            m_startingOrientation[3] = currentQuatW;
        }

        // Compute relative position (world frame displacement)
        float deltaPosX = currentPosX - m_startingPosition[0];
        float deltaPosY = currentPosY - m_startingPosition[1];
        float deltaPosZ = currentPosZ - m_startingPosition[2];

        // Transform displacement to starting body frame
        carb::Float4 startingQuatInv = isaacsim::core::includes::math::inverse(carb::Float4{
            m_startingOrientation[0], m_startingOrientation[1], m_startingOrientation[2], m_startingOrientation[3] });

        carb::Float3 relativePosition =
            isaacsim::core::includes::math::rotate(startingQuatInv, carb::Float3{ deltaPosX, deltaPosY, deltaPosZ });

        // Compute relative orientation (current * starting^-1)
        carb::Float4 currentQuat = { currentQuatX, currentQuatY, currentQuatZ, currentQuatW };
        carb::Float4 relativeQuat = currentQuat * startingQuatInv;

        // Get velocities (world frame)
        carb::Float3 linearVelocityWorld = { linearVelX, linearVelY, linearVelZ };
        carb::Float3 angularVelocityWorld = { angularVelX, angularVelY, angularVelZ };

        // Transform velocities to body frame
        carb::Float4 currentQuatInv = isaacsim::core::includes::math::inverse(currentQuat);
        carb::Float3 linearVelocityBody = isaacsim::core::includes::math::rotate(currentQuatInv, linearVelocityWorld);
        carb::Float3 angularVelocityBody = isaacsim::core::includes::math::rotate(currentQuatInv, angularVelocityWorld);

        // Compute accelerations (body frame)
        carb::Float3 linearAcceleration = { 0.0f, 0.0f, 0.0f };
        carb::Float3 angularAcceleration = { 0.0f, 0.0f, 0.0f };

        if (m_firstFrame)
        {
            m_firstFrame = false;
        }
        else
        {
            // Simple finite difference
            const float dt = 0.016f; // Assume ~60Hz, could be improved with actual time tracking
            linearAcceleration.x = (linearVelocityBody.x - m_prevLinearVelocity.x) / dt;
            linearAcceleration.y = (linearVelocityBody.y - m_prevLinearVelocity.y) / dt;
            linearAcceleration.z = (linearVelocityBody.z - m_prevLinearVelocity.z) / dt;

            angularAcceleration.x = (angularVelocityBody.x - m_prevAngularVelocity.x) / dt;
            angularAcceleration.y = (angularVelocityBody.y - m_prevAngularVelocity.y) / dt;
            angularAcceleration.z = (angularVelocityBody.z - m_prevAngularVelocity.z) / dt;
        }

        // Store for next frame
        m_prevLinearVelocity = linearVelocityBody;
        m_prevAngularVelocity = angularVelocityBody;

        // Fill output data structure
        data.position = { static_cast<double>(relativePosition.x), static_cast<double>(relativePosition.y),
                          static_cast<double>(relativePosition.z) };
        data.orientation = { static_cast<double>(relativeQuat.w), static_cast<double>(relativeQuat.x),
                             static_cast<double>(relativeQuat.y), static_cast<double>(relativeQuat.z) };
        data.linearVelocity = { static_cast<double>(linearVelocityBody.x), static_cast<double>(linearVelocityBody.y),
                                static_cast<double>(linearVelocityBody.z) };
        data.angularVelocity = { static_cast<double>(angularVelocityBody.x), static_cast<double>(angularVelocityBody.y),
                                 static_cast<double>(angularVelocityBody.z) };
        data.linearAcceleration = { static_cast<double>(linearAcceleration.x), static_cast<double>(linearAcceleration.y),
                                    static_cast<double>(linearAcceleration.z) };
        data.angularAcceleration = { static_cast<double>(angularAcceleration.x),
                                     static_cast<double>(angularAcceleration.y),
                                     static_cast<double>(angularAcceleration.z) };

        return data;
    }

    /**
     * @brief Generate message from odometry data.
     * @details
     * Serializes odometry data into message format.
     *
     * Message format:
     * - timestamp (double, 8 bytes)
     * - position (3 doubles, 24 bytes) - relative position in body frame
     * - orientation (4 doubles, 32 bytes) - relative quaternion (w, x, y, z)
     * - linear_velocity (3 doubles, 24 bytes) - body frame
     * - angular_velocity (3 doubles, 24 bytes) - body frame
     * - linear_acceleration (3 doubles, 24 bytes) - body frame
     * - angular_acceleration (3 doubles, 24 bytes) - body frame
     *
     * @param[in] data Odometry data to serialize
     * @return std::vector<uint8_t> Serialized message data
     */
    std::vector<uint8_t> generateMessage(const isaacsim::ucx::nodes::OdometryData& data) override
    {
        // Calculate message size
        const size_t messageSize = sizeof(double) + // timestamp
                                   sizeof(double) * 3 + // position
                                   sizeof(double) * 4 + // orientation (quaternion)
                                   sizeof(double) * 3 + // linear_velocity
                                   sizeof(double) * 3 + // angular_velocity
                                   sizeof(double) * 3 + // linear_acceleration
                                   sizeof(double) * 3; // angular_acceleration

        std::vector<uint8_t> messageData(messageSize);
        size_t offset = 0;

        // Write timestamp
        std::memcpy(messageData.data() + offset, &data.timestamp, sizeof(double));
        offset += sizeof(double);

        // Write relative position
        std::memcpy(messageData.data() + offset, data.position.data(), sizeof(double) * 3);
        offset += sizeof(double) * 3;

        // Write relative orientation (w, x, y, z)
        std::memcpy(messageData.data() + offset, data.orientation.data(), sizeof(double) * 4);
        offset += sizeof(double) * 4;

        // Write linear velocity (body frame)
        std::memcpy(messageData.data() + offset, data.linearVelocity.data(), sizeof(double) * 3);
        offset += sizeof(double) * 3;

        // Write angular velocity (body frame)
        std::memcpy(messageData.data() + offset, data.angularVelocity.data(), sizeof(double) * 3);
        offset += sizeof(double) * 3;

        // Write linear acceleration (body frame)
        std::memcpy(messageData.data() + offset, data.linearAcceleration.data(), sizeof(double) * 3);
        offset += sizeof(double) * 3;

        // Write angular acceleration (body frame)
        std::memcpy(messageData.data() + offset, data.angularAcceleration.data(), sizeof(double) * 3);
        offset += sizeof(double) * 3;

        return messageData;
    }

private:
    float m_startingPosition[3] = { 0.0f, 0.0f, 0.0f }; //!< Starting position for odometry reference
    float m_startingOrientation[4] = { 0.0f, 0.0f, 0.0f, 1.0f }; //!< Starting orientation (x, y, z, w)
    carb::Float3 m_prevLinearVelocity = { 0.0f, 0.0f, 0.0f }; //!< Previous linear velocity for acceleration calculation
    carb::Float3 m_prevAngularVelocity = { 0.0f, 0.0f, 0.0f }; //!< Previous angular velocity for acceleration
                                                               //!< calculation
    bool m_firstFrame = true; //!< Flag indicating first frame to initialize starting pose
};

REGISTER_OGN_NODE()
