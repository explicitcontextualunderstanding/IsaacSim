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

#include <isaacsim/core/includes/UsdUtilities.h>
#include <isaacsim/ucx/core/UcxListenerRegistry.h>
#include <isaacsim/ucx/core/UcxUtils.h>
#include <isaacsim/ucx/nodes/UcxNode.h>
#include <omni/fabric/FabricUSD.h>
#include <omni/graph/core/CppWrappers.h>
#include <omni/graph/core/iComputeGraph.h>
#include <omni/physics/tensors/IArticulationView.h>
#include <omni/physics/tensors/ISimulationView.h>
#include <omni/physics/tensors/TensorApi.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usdPhysics/articulationRootAPI.h>

#include <cstring>
#include <vector>

using omni::graph::core::GraphInstanceID;
using omni::graph::core::NodeObj;

namespace isaacsim::ucx::nodes
{

/**
 * @struct JointStateData
 * @brief Data structure for joint state message payload.
 * @details
 * Contains robot joint state data including positions, velocities, and efforts
 * for all joints in the articulation.
 */
struct JointStateData
{
    double timestamp; //!< Timestamp value in seconds
    uint32_t numJoints; //!< Number of joints in the articulation
    std::vector<double> positions; //!< Joint positions (scaled to meters)
    std::vector<double> velocities; //!< Joint velocities (scaled)
    std::vector<double> efforts; //!< Joint efforts/forces
};

} // namespace isaacsim::ucx::nodes

/**
 * @class UCXPublishJointStateNodeBase
 * @brief Templated base class for UCX joint state publishing nodes.
 * @details
 * This template provides common functionality for publishing robot joint state data over UCX.
 * Derived classes implement message generation logic via generateMessage().
 *
 * @tparam DatabaseT The OGN database type for the node
 */
template <typename DatabaseT>
class UCXPublishJointStateNodeBase : public isaacsim::ucx::nodes::UcxNode
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
        // Derived class should override if it needs to acquire the tensor interface
    }

    /**
     * @brief Reset the node state.
     * @details
     * Releases physics resources and clears cached data.
     */
    virtual void reset() override
    {
        if (m_articulation)
        {
            m_articulation->release();
            m_articulation = nullptr;
        }
        if (m_simView)
        {
            m_simView->release(true);
            m_simView = nullptr;
        }

        m_stage = nullptr;
        m_jointPositions.clear();
        m_jointVelocities.clear();
        m_jointEfforts.clear();

        UcxNode::reset();
    }

protected:
    /**
     * @brief Common compute logic for joint state publishing nodes.
     * @details
     * Handles listener initialization, connection checking, and message publishing.
     * Initializes the articulation view if needed. Extracts data using extractData()
     * and serializes using generateMessage().
     *
     * @param[in] db Database accessor for node inputs/outputs
     * @param[in] context Graph context for accessing stage
     * @param[in] port Port number for UCX listener
     * @param[in] tag UCX tag for message identification
     * @param[in] timeoutMs Timeout in milliseconds for send request (0 = infinite)
     * @return bool True if execution succeeded, false otherwise
     */
    bool computeImpl(DatabaseT& db, const GraphContextObj& context, uint16_t port, uint64_t tag, uint32_t timeoutMs)
    {
        if (!this->ensureListenerReady(db, port))
        {
            return false;
        }

        if (!this->waitForConnection())
        {
            return true;
        }

        if (!m_articulation)
        {
            if (!initializeArticulation(db, context))
            {
                return false;
            }
        }

        isaacsim::ucx::nodes::JointStateData data = extractData(db);
        return this->publishMessage(db, generateMessage(data), tag, timeoutMs);
    }

    /**
     * @brief Find articulation root under a given prim.
     * @details
     * Recursively searches for a prim with ArticulationRootAPI applied.
     * If the given prim has the API, it's returned. Otherwise, searches children.
     *
     * @param[in] stage USD stage
     * @param[in] startPrim Prim to start searching from
     * @return pxr::UsdPrim The articulation root prim, or invalid prim if not found
     */
    static pxr::UsdPrim findArticulationRoot(const pxr::UsdStageWeakPtr& stage, const pxr::UsdPrim& startPrim)
    {
        if (!startPrim.IsValid())
        {
            return pxr::UsdPrim();
        }

        // Check if this prim is the articulation root
        if (startPrim.HasAPI<pxr::UsdPhysicsArticulationRootAPI>())
        {
            return startPrim;
        }

        // Search children recursively
        for (const auto& child : startPrim.GetChildren())
        {
            auto result = findArticulationRoot(stage, child);
            if (result.IsValid())
            {
                return result;
            }
        }

        return pxr::UsdPrim();
    }

    /**
     * @brief Initialize the articulation view.
     * @details
     * Creates the simulation view and articulation view for the target prim.
     * If the target prim is not an articulation root, searches its children recursively.
     *
     * @param[in] db Database accessor for node inputs
     * @param[in] context Graph context for accessing stage
     * @return bool True if initialization succeeded, false otherwise
     */
    virtual bool initializeArticulation(DatabaseT& db, const GraphContextObj& context)
    {
        if (!m_tensorInterface)
        {
            db.logError("Tensor API interface not initialized");
            return false;
        }

        long stageId = context.iContext->getStageId(context);
        auto stage = pxr::UsdUtilsStageCache::Get().Find(pxr::UsdStageCache::Id::FromLongInt(stageId));
        if (!stage)
        {
            db.logError("Could not find USD stage %ld", stageId);
            return false;
        }

        m_simView = m_tensorInterface->createSimulationView(stageId);

        const auto& prim = db.inputs.targetPrim();
        if (prim.empty())
        {
            db.logError("Could not find target prim");
            return false;
        }

        auto targetPrim = stage->GetPrimAtPath(omni::fabric::toSdfPath(prim[0]));
        if (!targetPrim)
        {
            db.logError(
                "The prim %s is not valid. Please specify a valid prim", omni::fabric::toSdfPath(prim[0]).GetText());
            return false;
        }

        // Find articulation root (may be the target prim itself, or a child)
        auto articulationRoot = findArticulationRoot(stage, targetPrim);
        if (!articulationRoot.IsValid())
        {
            db.logError("No articulation root found under prim %s. Please specify a prim that contains an articulation.",
                        omni::fabric::toSdfPath(prim[0]).GetText());
            return false;
        }

        const char* articulationPath = articulationRoot.GetPath().GetText();

        m_unitScale = UsdGeomGetStageMetersPerUnit(stage);
        m_stage = stage;

        // Create articulation view
        m_articulation = m_simView->createArticulationView(std::vector<std::string>{ articulationPath });
        if (!m_articulation)
        {
            db.logError("Failed to create articulation view for %s", articulationPath);
            return false;
        }

        return true;
    }

    /**
     * @brief Extract joint state data from articulation.
     * @details
     * Pure virtual function that derived classes must implement to read
     * joint data from the articulation and return it as JointStateData.
     *
     * @param[in] db Database accessor for node inputs
     * @return JointStateData Extracted joint state data
     */
    virtual isaacsim::ucx::nodes::JointStateData extractData(DatabaseT& db) = 0;

    /**
     * @brief Generate message from joint state data.
     * @details
     * Pure virtual function that derived classes must implement to serialize
     * joint state data into the appropriate message format.
     *
     * @param[in] data Joint state data to serialize
     * @return std::vector<uint8_t> Serialized message data
     */
    virtual std::vector<uint8_t> generateMessage(const isaacsim::ucx::nodes::JointStateData& data) = 0;

    /**
     * @brief Helper to create tensor descriptor.
     */
    static void createTensorDesc(omni::physics::tensors::TensorDesc& tensorDesc,
                                 void* data,
                                 uint32_t count,
                                 omni::physics::tensors::TensorDataType type)
    {
        tensorDesc.data = data;
        tensorDesc.dtype = type;
        tensorDesc.numDims = 1;
        tensorDesc.dims[0] = count;
    }

    pxr::UsdStageWeakPtr m_stage = nullptr;
    omni::physics::tensors::TensorApi* m_tensorInterface = nullptr;
    omni::physics::tensors::ISimulationView* m_simView = nullptr;
    omni::physics::tensors::IArticulationView* m_articulation = nullptr;
    std::vector<float> m_jointPositions;
    std::vector<float> m_jointVelocities;
    std::vector<float> m_jointEfforts;

    double m_unitScale = 1;
};

// NOTE: To use this base class:
// 1. Derive your OGN node class from UCXPublishJointStateNodeBase<YourDatabase>
// 2. Implement static void initInstance() to acquire the tensor interface
// 3. Implement static void releaseInstance(NodeObj const& nodeObj, GraphInstanceID instanceId)
//    - Get state: auto& state = YourDatabase::template sPerInstanceState<YourClass>(nodeObj, instanceId)
//    - Call state.reset()
// 4. Implement virtual JointStateData extractData(DatabaseT& db) override
//    - Read joint data from articulation and return as JointStateData
// 5. Implement virtual std::vector<uint8_t> generateMessage(const JointStateData& data) override
//    - Serialize the joint state data into message format
// 6. Implement static bool compute(YourDatabase& db) that:
//    - Extracts inputs from db
//    - Gets the graph context
//    - Gets the per-instance state: auto& state = db.template perInstanceState<YourClass>()
//    - Calls state.computeImpl(db, context, port, tag, timeoutMs)
// 7. See OgnUCXPublishJointState.cpp for examples
