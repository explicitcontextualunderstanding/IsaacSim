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

#include <isaacsim/ucx/nodes/UcxPublishImuNodeBase.h>

#include <OgnUCXPublishImuDatabase.h>

/**
 * @class OgnUCXPublishImu
 * @brief OmniGraph node for publishing IMU data via UCX.
 * @details
 * This node publishes IMU sensor data over UCX using tagged communication.
 * It supports orientation, linear acceleration, and angular velocity data.
 */
class OgnUCXPublishImu : public UCXPublishImuNodeBase<OgnUCXPublishImuDatabase>
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
        auto& state = OgnUCXPublishImuDatabase::sPerInstanceState<OgnUCXPublishImu>(nodeObj, instanceId);
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
    static bool compute(OgnUCXPublishImuDatabase& db)
    {
        const uint16_t port = static_cast<uint16_t>(db.inputs.port());
        const uint64_t tag = db.inputs.tag();
        const uint32_t timeoutMs = db.inputs.timeoutMs();

        // Get the per-instance state and call the instance method
        auto& state = db.template perInstanceState<OgnUCXPublishImu>();
        return state.computeImpl(db, port, tag, timeoutMs);
    }

protected:
    /**
     * @brief Generate message from IMU data.
     * @details
     * Serializes IMU metadata and sensor data.
     * Message format:
     * - timestamp (double, 8 bytes)
     * - frameId_length (uint32_t, 4 bytes)
     * - frameId (variable length string)
     * - publish_flags (uint8_t, 1 byte): bit 0=orientation, bit 1=linear_acceleration, bit 2=angular_velocity
     * - orientation (4 doubles, 32 bytes) - if enabled
     * - linear_acceleration (3 doubles, 24 bytes) - if enabled
     * - angular_velocity (3 doubles, 24 bytes) - if enabled
     *
     * @param[in] data IMU data to serialize
     * @return std::vector<uint8_t> Serialized message data
     */
    std::vector<uint8_t> generateMessage(const isaacsim::ucx::nodes::ImuData& data) override
    {
        // Calculate message size
        size_t messageSize = sizeof(double) + sizeof(uint32_t) + data.frameId.length() + sizeof(uint8_t);

        if (data.publishOrientation)
        {
            messageSize += sizeof(double) * 4; // Quaternion (x, y, z, w)
        }
        if (data.publishLinearAcceleration)
        {
            messageSize += sizeof(double) * 3; // Vector3
        }
        if (data.publishAngularVelocity)
        {
            messageSize += sizeof(double) * 3; // Vector3
        }

        std::vector<uint8_t> messageData(messageSize);
        size_t offset = 0;

        // Write timestamp
        std::memcpy(messageData.data() + offset, &data.timestamp, sizeof(double));
        offset += sizeof(double);

        // Write frameId length
        const uint32_t frameIdLength = static_cast<uint32_t>(data.frameId.length());
        std::memcpy(messageData.data() + offset, &frameIdLength, sizeof(uint32_t));
        offset += sizeof(uint32_t);

        // Write frameId string
        std::memcpy(messageData.data() + offset, data.frameId.c_str(), data.frameId.length());
        offset += data.frameId.length();

        // Write publish flags
        uint8_t publishFlags = 0;
        if (data.publishOrientation)
        {
            publishFlags |= 0x01;
        }
        if (data.publishLinearAcceleration)
        {
            publishFlags |= 0x02;
        }
        if (data.publishAngularVelocity)
        {
            publishFlags |= 0x04;
        }
        std::memcpy(messageData.data() + offset, &publishFlags, sizeof(uint8_t));
        offset += sizeof(uint8_t);

        // Write orientation if enabled
        if (data.publishOrientation)
        {
            std::memcpy(messageData.data() + offset, data.orientation.data(), sizeof(double) * 4);
            offset += sizeof(double) * 4;
        }

        // Write linear acceleration if enabled
        if (data.publishLinearAcceleration)
        {
            std::memcpy(messageData.data() + offset, data.linearAcceleration.data(), sizeof(double) * 3);
            offset += sizeof(double) * 3;
        }

        // Write angular velocity if enabled
        if (data.publishAngularVelocity)
        {
            std::memcpy(messageData.data() + offset, data.angularVelocity.data(), sizeof(double) * 3);
            offset += sizeof(double) * 3;
        }

        return messageData;
    }
};

REGISTER_OGN_NODE()
