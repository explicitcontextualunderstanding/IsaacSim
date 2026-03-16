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

#include <isaacsim/ucx/nodes/UcxPublishClockNodeBase.h>

#include <OgnUCXPublishClockDatabase.h>

/**
 * @class OgnUCXPublishClock
 * @brief OmniGraph node for publishing clock messages via UCX.
 * @details
 * This node publishes timestamp data over UCX using tagged communication.
 * It uses the UCXPublishClockNodeBase template to minimize code duplication.
 */
class OgnUCXPublishClock : public UCXPublishClockNodeBase<OgnUCXPublishClockDatabase>
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
        auto& state = OgnUCXPublishClockDatabase::sPerInstanceState<OgnUCXPublishClock>(nodeObj, instanceId);
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
    static bool compute(OgnUCXPublishClockDatabase& db)
    {
        const uint16_t port = static_cast<uint16_t>(db.inputs.port());
        const uint64_t tag = db.inputs.tag();
        const uint32_t timeoutMs = db.inputs.timeoutMs();

        // Get the per-instance state and call the instance method
        auto& state = db.template perInstanceState<OgnUCXPublishClock>();
        return state.computeImpl(db, port, tag, timeoutMs);
    }

protected:
    /**
     * @brief Generate message from clock data.
     * @details
     * Serializes the clock data as a double (8 bytes) in native byte order.
     *
     * @param[in] data Clock data to serialize
     * @return std::vector<uint8_t> Serialized message data
     */
    std::vector<uint8_t> generateMessage(const isaacsim::ucx::nodes::ClockData& data) override
    {
        std::vector<uint8_t> messageData(sizeof(double));
        std::memcpy(messageData.data(), &data.timestamp, sizeof(double));
        return messageData;
    }
};

REGISTER_OGN_NODE()
