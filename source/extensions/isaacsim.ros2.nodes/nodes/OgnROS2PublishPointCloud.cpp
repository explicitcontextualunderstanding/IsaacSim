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

#ifdef _WIN32
#    pragma warning(push)
#    pragma warning(disable : 4996)
#endif

// clang-format off
#include <pch/UsdPCH.h>
// clang-format on
#include "ROS2PublishPointCloud.cuh"
#include "isaacsim/core/includes/UsdUtilities.h"

#include <carb/profiler/Profile.h>
#include <carb/tasking/ITasking.h>
#include <carb/tasking/TaskingUtils.h>

#include <isaacsim/core/includes/ScopedCudaDevice.h>
#include <isaacsim/ros2/core/Ros2Node.h>

#include <OgnROS2PublishPointCloudDatabase.h>

using namespace isaacsim::ros2::core;

class PublishPointCloudThreadData
{
public:
    PublishPointCloudThreadData(uint8_t*& messageBufferDevice, size_t& messageBufferDeviceSize)
        : inputDataPtr(nullptr),
          outputDataPtr(nullptr),
          bufferSize(0),
          totalBytes(0),
          cudaDeviceIndex(-1),
          stream(nullptr),
          streamDevice(nullptr),
          mStreamNotCreated(nullptr),
          messageBufferDevice(messageBufferDevice),
          messageBufferDeviceSize(messageBufferDeviceSize)
    {
    }

    void* inputDataPtr;
    void* outputDataPtr;
    size_t bufferSize;
    size_t totalBytes;
    int cudaDeviceIndex;

    cudaStream_t* stream;
    int* streamDevice;
    bool* mStreamNotCreated;
    int maxThreadsPerBlock{ 0 };
    int multiProcessorCount{ 0 };

    uint8_t*& messageBufferDevice;
    size_t& messageBufferDeviceSize;

    std::shared_ptr<Ros2Publisher> publisher;
    std::shared_ptr<Ros2PointCloudMessage> message;
};

class OgnROS2PublishPointCloud : public Ros2Node
{
public:
    bool initializeCudaProperties(int cudaDeviceIndex)
    {
        if (cudaDeviceIndex == -1)
        {
            return true;
        }
        isaacsim::core::includes::ScopedDevice scopedDev(cudaDeviceIndex);

        // Query for device props
        try
        {
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDeviceProperties(&prop, cudaDeviceIndex));

            // cache results
            m_maxThreadsPerBlock = prop.maxThreadsPerBlock;
            m_multiProcessorCount = prop.multiProcessorCount;
        }
        catch (const std::exception& e)
        {
            CARB_LOG_ERROR("Failed to get device properties for GPU %d: %s", cudaDeviceIndex, e.what());
            return false;
        }
        return true;
    }

    static bool compute(OgnROS2PublishPointCloudDatabase& db)
    {
        auto& state = db.perInstanceState<OgnROS2PublishPointCloud>();

        // Spin once calls reset automatically if it was not successful
        const auto& nodeObj = db.abi_node();
        if (!state.isInitialized())
        {
            const GraphContextObj& context = db.abi_context();
            // Find our stage
            long stageId = context.iContext->getStageId(context);
            auto stage = pxr::UsdUtilsStageCache::Get().Find(pxr::UsdStageCache::Id::FromLongInt(stageId));

            if (!state.initializeNodeHandle(
                    std::string(nodeObj.iNode->getPrimPath(nodeObj)),
                    collectNamespace(db.inputs.nodeNamespace(),
                                     stage->GetPrimAtPath(pxr::SdfPath(nodeObj.iNode->getPrimPath(nodeObj)))),
                    db.inputs.context()))
            {
                db.logError("Unable to create ROS2 node, please check that namespace is valid");
                return false;
            }
            if (!state.initializeCudaProperties(db.inputs.cudaDeviceIndex()))
            {
                db.logError("Failed to initialize CUDA properties for GPU %d", db.inputs.cudaDeviceIndex());
                return false;
            }
        }

        // Publisher was not valid, create a new one
        if (!state.m_publisher)
        {
            CARB_PROFILE_ZONE(0, "[IsaacSim] setup point cloud publisher");
            // Setup ROS publisher
            const std::string& topicName = db.inputs.topicName();
            std::string fullTopicName = addTopicPrefix(state.m_namespaceName, topicName);
            if (!state.m_factory->validateTopicName(fullTopicName))
            {
                db.logError("Unable to create ROS2 publisher, invalid topic name");
                return false;
            }

            state.m_message = state.m_factory->createPointCloudMessage();

            Ros2QoSProfile qos;
            const std::string& qosProfile = db.inputs.qosProfile();
            if (qosProfile.empty())
            {
                qos.depth = db.inputs.queueSize();
            }
            else
            {
                if (!jsonToRos2QoSProfile(qos, qosProfile))
                {
                    return false;
                }
            }

            state.m_publisher = state.m_factory->createPublisher(
                state.m_nodeHandle.get(), fullTopicName.c_str(), state.m_message->getTypeSupportHandle(), qos);

            state.m_frameId = db.inputs.frameId();

            // Get extension settings for multithreading
            carb::settings::ISettings* threadSettings = carb::getCachedInterface<carb::settings::ISettings>();
            static constexpr char s_kThreadDisable[] = "/exts/isaacsim.ros2.bridge/publish_multithreading_disabled";
            state.m_multithreadingDisabled = threadSettings->getAsBool(s_kThreadDisable);
            return true;
        }

        return state.publishLidar(db);
    }

    PublishPointCloudThreadData buildThreadData(OgnROS2PublishPointCloudDatabase& db,
                                                OgnROS2PublishPointCloud& state,
                                                void* dataPtr,
                                                size_t totalBytes)
    {
        PublishPointCloudThreadData threadData(state.m_messageBufferDevice, state.m_messageBufferDeviceSize);

        threadData.inputDataPtr = reinterpret_cast<void*>(db.inputs.dataPtr());
        threadData.outputDataPtr = dataPtr;
        threadData.bufferSize = db.inputs.bufferSize();
        threadData.totalBytes = totalBytes;
        threadData.cudaDeviceIndex = db.inputs.cudaDeviceIndex();

        threadData.stream = &state.m_stream;
        threadData.streamDevice = &state.m_streamDevice;
        threadData.mStreamNotCreated = &state.m_streamNotCreated;

        threadData.publisher = state.m_publisher;
        threadData.message = state.m_message;

        threadData.maxThreadsPerBlock = state.m_maxThreadsPerBlock;
        threadData.multiProcessorCount = state.m_multiProcessorCount;

        return threadData;
    }

    static bool publishPointCloudHelper(PublishPointCloudThreadData& data)
    {
        CARB_PROFILE_ZONE(1, "[IsaacSim] Publish PointCloud Thread");
        isaacsim::core::includes::ScopedDevice scopedDev(data.cudaDeviceIndex);

        // If the device doesn't match and we have created a stream, destroy it
        if (*data.streamDevice != data.cudaDeviceIndex && *data.mStreamNotCreated == false)
        {
            CARB_PROFILE_ZONE(1, "[IsaacSim] Destroy stream");
            CUDA_CHECK(cudaStreamDestroy(*data.stream));
            *data.mStreamNotCreated = true;
            *data.streamDevice = -1;
        }
        // Create a stream if it does not exist
        if (*data.mStreamNotCreated)
        {
            CARB_PROFILE_ZONE(1, "[IsaacSim] Create stream");
            CUDA_CHECK(cudaStreamCreate(data.stream));
            *data.mStreamNotCreated = false;
            *data.streamDevice = data.cudaDeviceIndex;
        }

        CARB_PROFILE_ZONE(1, "[IsaacSim] data in cuda memory");
        if (data.message->getOrderedFields().empty())
        {
            // Direct copy when no metadata is needed
            CUDA_CHECK(cudaMemcpyAsync(
                data.outputDataPtr, data.inputDataPtr, data.bufferSize, cudaMemcpyDeviceToHost, *data.stream));
        }
        else
        {
            // Use kernel to fill the buffer with correctly-ordered metadata
            if (data.messageBufferDeviceSize < data.message->getTotalBytes())
            {
                // Reallocate the buffer if it's too small
                CUDA_CHECK(cudaFree(data.messageBufferDevice));
                data.messageBufferDevice = nullptr;
                data.messageBufferDeviceSize = 0;
            }
            if (data.messageBufferDevice == nullptr)
            {
                data.messageBufferDeviceSize = data.message->getTotalBytes();
                CUDA_CHECK(cudaMalloc(&data.messageBufferDevice, data.messageBufferDeviceSize));
            }
            isaacsim::ros2::nodes::fillPointCloudBuffer(
                data.messageBufferDevice, reinterpret_cast<const float3*>(data.inputDataPtr),
                data.message->getOrderedFields(), data.message->getPointStep(), data.message->getNumPoints(),
                data.maxThreadsPerBlock, data.multiProcessorCount, data.cudaDeviceIndex, *data.stream);
            CUDA_CHECK(cudaMemcpyAsync(data.outputDataPtr, data.messageBufferDevice, data.message->getTotalBytes(),
                                       cudaMemcpyDeviceToHost, *data.stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(*data.stream));

        {
            CARB_PROFILE_ZONE(1, "[IsaacSim] pcl publisher publish");
            data.publisher.get()->publish(data.message->getPtr());
        }
        return true;
    }

    bool publishLidar(OgnROS2PublishPointCloudDatabase& db)
    {
        CARB_PROFILE_ZONE(0, "[IsaacSim] Lidar Point Cloud Pub");
        auto& state = db.perInstanceState<OgnROS2PublishPointCloud>();
        auto tasking = carb::getCachedInterface<carb::tasking::ITasking>();

        {
            CARB_PROFILE_ZONE(1, "[IsaacSim] wait for previous publish");
            // Wait for last message publish
            state.m_tasks.wait();
        }

        // Check if subscription count is 0
        if (!state.m_publishWithoutVerification && !state.m_publisher.get()->getSubscriptionCount())
        {
            return false;
        }

        float* intensityPtr = reinterpret_cast<float*>(db.inputs.intensityPtr());
        uint64_t* timestampPtr = reinterpret_cast<uint64_t*>(db.inputs.timestampPtr());
        uint32_t* emitterIdPtr = reinterpret_cast<uint32_t*>(db.inputs.emitterIdPtr());
        uint32_t* channelIdPtr = reinterpret_cast<uint32_t*>(db.inputs.channelIdPtr());
        uint32_t* materialIdPtr = reinterpret_cast<uint32_t*>(db.inputs.materialIdPtr());
        uint32_t* tickIdPtr = reinterpret_cast<uint32_t*>(db.inputs.tickIdPtr());
        GfVec3f* hitNormalPtr = reinterpret_cast<GfVec3f*>(db.inputs.hitNormalPtr());
        GfVec3f* velocityPtr = reinterpret_cast<GfVec3f*>(db.inputs.velocityPtr());
        uint32_t* objectIdPtr = reinterpret_cast<uint32_t*>(db.inputs.objectIdPtr());
        uint8_t* echoIdPtr = reinterpret_cast<uint8_t*>(db.inputs.echoIdPtr());
        uint8_t* tickStatePtr = reinterpret_cast<uint8_t*>(db.inputs.tickStatePtr());
        float* radialVelocityMSPtr = reinterpret_cast<float*>(db.inputs.radialVelocityMSPtr());

        if (db.inputs.cudaDeviceIndex() == -1)
        {
            state.m_message->setUsePinnedBuffer(false);
            if (db.inputs.dataPtr() != 0)
            {
                if (db.inputs.bufferSize() == 0)
                {
                    return false;
                }
                state.m_message->generateBuffer(db.inputs.timeStamp(), state.m_frameId, db.inputs.bufferSize(),
                                                intensityPtr, timestampPtr, emitterIdPtr, channelIdPtr, materialIdPtr,
                                                tickIdPtr, hitNormalPtr, velocityPtr, objectIdPtr, echoIdPtr,
                                                tickStatePtr, radialVelocityMSPtr);
                // Data is on host as ptr
                if (state.m_message->getOrderedFields().empty())
                {
                    // Direct copy when no metadata interleaving is needed
                    memcpy(state.m_message->getBufferPtr(), reinterpret_cast<void*>(db.inputs.dataPtr()),
                           state.m_message->getTotalBytes());
                }
                else
                {
                    // Host interleave: fill buffer with xyz + metadata per point
                    isaacsim::ros2::nodes::fillPointCloudBufferHost(
                        reinterpret_cast<uint8_t*>(state.m_message->getBufferPtr()),
                        reinterpret_cast<const float3*>(db.inputs.dataPtr()), state.m_message->getOrderedFields(),
                        state.m_message->getPointStep(), state.m_message->getNumPoints());
                }
            }
            else if (db.inputs.dataPtr() == 0)
            {
                const size_t totalBytes = sizeof(GfVec3f) * db.inputs.data.size();
                if (totalBytes == 0)
                {
                    return false;
                }
                state.m_message->generateBuffer(db.inputs.timeStamp(), state.m_frameId, totalBytes, intensityPtr,
                                                timestampPtr, emitterIdPtr, channelIdPtr, materialIdPtr, tickIdPtr,
                                                hitNormalPtr, velocityPtr, objectIdPtr, echoIdPtr, tickStatePtr,
                                                radialVelocityMSPtr);
                // Data is on host as ogn data, copy from cpu
                {
                    memcpy(state.m_message->getBufferPtr(), reinterpret_cast<const uint8_t*>(db.inputs.data.cpu().data()),
                           state.m_message->getTotalBytes());
                }
            }

            if (state.m_multithreadingDisabled)
            {
                CARB_PROFILE_ZONE(1, "[IsaacSim] Publish PCL");
                state.m_publisher.get()->publish(state.m_message->getPtr());
            }
            else
            {
                tasking->addTask(carb::tasking::Priority::eHigh, state.m_tasks,
                                 [&state]
                                 {
                                     CARB_PROFILE_ZONE(1, "[IsaacSim] Publish PCL Thread");
                                     state.m_publisher.get()->publish(state.m_message->getPtr());
                                 });
            }
        }
        else if (db.inputs.dataPtr() != 0)
        {
            state.m_message->setUsePinnedBuffer(true);
            state.m_message->generateBuffer(db.inputs.timeStamp(), state.m_frameId, db.inputs.bufferSize(),
                                            intensityPtr, timestampPtr, emitterIdPtr, channelIdPtr, materialIdPtr,
                                            tickIdPtr, hitNormalPtr, velocityPtr, objectIdPtr, echoIdPtr, tickStatePtr,
                                            radialVelocityMSPtr);

            PublishPointCloudThreadData publishPointCloudThreadData =
                buildThreadData(db, state, state.m_message->getBufferPtr(), state.m_message->getTotalBytes());

            if (state.m_multithreadingDisabled)
            {
                return publishPointCloudHelper(publishPointCloudThreadData);
            }
            else
            {
                // In order to get the benefits of using a separate stream, do the work in a new thread
                tasking->addTask(carb::tasking::Priority::eHigh, state.m_tasks,
                                 [data = publishPointCloudThreadData]() mutable
                                 { return publishPointCloudHelper(data); });
            }
        }

        return true;
    }

    static void releaseInstance(NodeObj const& nodeObj, GraphInstanceID instanceId)
    {
        auto& state = OgnROS2PublishPointCloudDatabase::sPerInstanceState<OgnROS2PublishPointCloud>(nodeObj, instanceId);
        state.reset();
    }

    virtual void reset()
    {
        if (m_messageBufferDevice)
        {
            CUDA_CHECK(cudaFree(m_messageBufferDevice));
            m_messageBufferDevice = nullptr;
        }
        m_messageBufferDeviceSize = 0;
        m_maxThreadsPerBlock = 0;
        m_multiProcessorCount = 0;

        m_publisher.reset(); // This should be reset before we reset the handle.
        Ros2Node::reset();
    }

private:
    std::shared_ptr<Ros2Publisher> m_publisher = nullptr;
    std::shared_ptr<Ros2PointCloudMessage> m_message = nullptr;

    std::string m_frameId = "sim_lidar";

    carb::tasking::TaskGroup m_tasks;
    cudaStream_t m_stream;
    int m_streamDevice = -1;
    bool m_streamNotCreated = true;
    int m_maxThreadsPerBlock{ 0 };
    int m_multiProcessorCount{ 0 };

    bool m_multithreadingDisabled = false;

    uint8_t* m_messageBufferDevice = nullptr;
    size_t m_messageBufferDeviceSize = 0;
};

REGISTER_OGN_NODE()

#ifdef _WIN32
#    pragma warning(pop)
#endif
