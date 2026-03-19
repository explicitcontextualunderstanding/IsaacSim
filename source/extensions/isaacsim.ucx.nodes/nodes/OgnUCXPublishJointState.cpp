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

#include <carb/Framework.h>
#include <carb/Types.h>

#include <isaacsim/core/includes/Math.h>
#include <isaacsim/core/includes/UsdUtilities.h>
#include <isaacsim/ucx/nodes/UcxPublishJointStateNodeBase.h>
#include <omni/fabric/FabricUSD.h>

#include <OgnUCXPublishJointStateDatabase.h>

using namespace isaacsim::ucx::nodes;

/**
 * @class OgnUCXPublishJointState
 * @brief OmniGraph node for publishing joint states via UCX.
 * @details
 * This node publishes robot joint state data over UCX using tagged communication.
 * It reads joint data from an articulation using the physics tensors API.
 */
class OgnUCXPublishJointState : public UCXPublishJointStateNodeBase<OgnUCXPublishJointStateDatabase>
{
public:
    /**
     * @brief Initialize the node instance.
     * @details
     * Acquires the physics tensor API interface needed for reading joint data.
     *
     * @param[in] nodeObj The node object
     * @param[in] instanceId The instance ID
     */
    static void initInstance(NodeObj const& nodeObj, GraphInstanceID instanceId)
    {
        auto& state = OgnUCXPublishJointStateDatabase::sPerInstanceState<OgnUCXPublishJointState>(nodeObj, instanceId);
        state.m_tensorInterface = carb::getCachedInterface<omni::physics::tensors::TensorApi>();
        if (!state.m_tensorInterface)
        {
            CARB_LOG_ERROR("Failed to acquire Tensor Api interface\n");
            return;
        }
    }

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
        auto& state = OgnUCXPublishJointStateDatabase::sPerInstanceState<OgnUCXPublishJointState>(nodeObj, instanceId);
        state.reset();
    }

    /**
     * @brief Compute function - called when node is executed.
     * @details
     * Extracts inputs, gets the per-instance state, and delegates to the base class logic.
     * Triggers the execution output port upon successful execution.
     *
     * @param[in] db Database accessor for node inputs/outputs
     * @return bool True if execution succeeded, false otherwise
     */
    static bool compute(OgnUCXPublishJointStateDatabase& db)
    {
        const GraphContextObj& context = db.abi_context();
        auto& state = db.template perInstanceState<OgnUCXPublishJointState>();

        const uint16_t port = static_cast<uint16_t>(db.inputs.port());
        const uint64_t tag = db.inputs.tag();
        const uint32_t timeoutMs = db.inputs.timeoutMs();

        bool success = state.computeImpl(db, context, port, tag, timeoutMs);

        if (success)
        {
            db.outputs.execOut() = kExecutionAttributeStateEnabled;
        }

        return success;
    }

protected:
    /**
     * @brief Extract joint state data from articulation.
     * @details
     * Reads joint state data from the articulation using tensor API.
     *
     * @param[in] db Database accessor for node inputs
     * @return JointStateData Extracted joint state data
     */
    isaacsim::ucx::nodes::JointStateData extractData(OgnUCXPublishJointStateDatabase& db) override
    {
        isaacsim::ucx::nodes::JointStateData data;
        data.timestamp = db.inputs.timeStamp();
        data.numJoints = 0;

        if (!m_articulation)
        {
            return data;
        }

        double stageUnits = 1.0 / m_unitScale;

        // Get number of DOFs
        uint32_t numDofs = m_articulation->getMaxDofs();
        data.numJoints = numDofs;

        // Resize vectors
        m_jointPositions.resize(numDofs);
        m_jointVelocities.resize(numDofs);
        m_jointEfforts.resize(numDofs);

        // Create tensor descriptors
        omni::physics::tensors::TensorDesc positionTensor;
        omni::physics::tensors::TensorDesc velocityTensor;
        omni::physics::tensors::TensorDesc effortTensor;

        createTensorDesc(
            positionTensor, m_jointPositions.data(), numDofs, omni::physics::tensors::TensorDataType::eFloat32);
        createTensorDesc(
            velocityTensor, m_jointVelocities.data(), numDofs, omni::physics::tensors::TensorDataType::eFloat32);
        createTensorDesc(effortTensor, m_jointEfforts.data(), numDofs, omni::physics::tensors::TensorDataType::eFloat32);

        // Get joint data from articulation
        if (!m_articulation->getDofPositions(&positionTensor))
        {
            db.logError("Failed to get DOF positions");
            data.numJoints = 0;
            return data;
        }
        if (!m_articulation->getDofVelocities(&velocityTensor))
        {
            db.logError("Failed to get DOF velocities");
            data.numJoints = 0;
            return data;
        }
        if (!m_articulation->getDofProjectedJointForces(&effortTensor))
        {
            db.logError("Failed to get DOF forces");
            data.numJoints = 0;
            return data;
        }

        // Convert to doubles with unit scaling
        data.positions.resize(numDofs);
        data.velocities.resize(numDofs);
        data.efforts.resize(numDofs);

        for (uint32_t i = 0; i < numDofs; ++i)
        {
            data.positions[i] = static_cast<double>(m_jointPositions[i]) * stageUnits;
            data.velocities[i] = static_cast<double>(m_jointVelocities[i]) * stageUnits;
            data.efforts[i] = static_cast<double>(m_jointEfforts[i]);
        }

        return data;
    }

    /**
     * @brief Generate message from joint state data.
     * @details
     * Serializes joint state data into message format.
     * Message format:
     * - timestamp (double, 8 bytes)
     * - num_joints (uint32_t, 4 bytes)
     * - For each joint:
     *   - position (double, 8 bytes)
     *   - velocity (double, 8 bytes)
     *   - effort (double, 8 bytes)
     *
     * @param[in] data Joint state data to serialize
     * @return std::vector<uint8_t> Serialized message data
     */
    std::vector<uint8_t> generateMessage(const isaacsim::ucx::nodes::JointStateData& data) override
    {
        if (data.numJoints == 0)
        {
            return {};
        }

        // Calculate message size (using doubles for better precision)
        const size_t messageSize = sizeof(double) + sizeof(uint32_t) + data.numJoints * (sizeof(double) * 3);

        std::vector<uint8_t> messageData(messageSize);
        size_t offset = 0;

        // Write timestamp
        std::memcpy(messageData.data() + offset, &data.timestamp, sizeof(double));
        offset += sizeof(double);

        // Write number of joints
        std::memcpy(messageData.data() + offset, &data.numJoints, sizeof(uint32_t));
        offset += sizeof(uint32_t);

        // Write joint data (interleaved position, velocity, effort)
        for (uint32_t i = 0; i < data.numJoints; ++i)
        {
            std::memcpy(messageData.data() + offset, &data.positions[i], sizeof(double));
            offset += sizeof(double);

            std::memcpy(messageData.data() + offset, &data.velocities[i], sizeof(double));
            offset += sizeof(double);

            std::memcpy(messageData.data() + offset, &data.efforts[i], sizeof(double));
            offset += sizeof(double);
        }

        return messageData;
    }
};

REGISTER_OGN_NODE()
