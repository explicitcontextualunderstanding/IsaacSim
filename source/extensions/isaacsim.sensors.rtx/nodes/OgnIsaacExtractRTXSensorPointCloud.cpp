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

#include "GenericModelOutput.h"
#include "IsaacSimSensorsRTXCuda.cuh"
#include "OgnIsaacExtractRTXSensorPointCloudDatabase.h"
#include "SensorNodeUtils.h"
#include "isaacsim/core/includes/Buffer.h"
#include "isaacsim/core/includes/ScopedCudaDevice.h"
#include "isaacsim/core/includes/UsdUtilities.h"

namespace isaacsim
{
namespace sensors
{
namespace rtx
{

class OgnIsaacExtractRTXSensorPointCloud
{
private:
    isaacsim::core::includes::HostBufferBase<float3> hostPcBuffer;
    isaacsim::core::includes::DeviceBufferBase<float3> m_pcBuffer; // 3d point cloud
    isaacsim::core::includes::DeviceBufferBase<float> m_scratchBuffer1;
    isaacsim::core::includes::DeviceBufferBase<float> m_scratchBuffer2;
    isaacsim::core::includes::DeviceBufferBase<float> m_scratchBuffer3;
    isaacsim::core::includes::DeviceBufferBase<float> m_scratchBuffer4;

public:
    static bool compute(OgnIsaacExtractRTXSensorPointCloudDatabase& db)
    {
        // Disable downstream execution by default
        db.outputs.exec() = kExecutionAttributeStateDisabled;

        // Retrieve and test input data pointer
        void* dataPtr = reinterpret_cast<void*>(db.inputs.dataPtr());
        if (!dataPtr)
        {
            CARB_LOG_INFO("IsaacExtractRTXSensorPointCloud: dataPtr input is empty.");
            return false;
        }

        // Determine if input data pointer is on device or host
        cudaStream_t cudaStream = (cudaStream_t)db.inputs.cudaStream();
        cudaPointerAttributes attributes;
        CUDA_CHECK(cudaPointerGetAttributes(&attributes, dataPtr));
        const bool isDevicePtr{ attributes.type == cudaMemoryTypeDevice };

        if (isDevicePtr)
        {
            CARB_LOG_ERROR(
                "IsaacExtractRTXSensorPointCloud: dataPtr input unexpectedly on device. Expecting input on host.");
            return false;
        }

        // Retrieve GMO struct from buffer, then validate it
        omni::sensors::GenericModelOutput* gmo = omni::sensors::getModelOutputPtrFromBuffer(dataPtr);
        if (gmo->numElements == 0)
        {
            CARB_LOG_WARN("IsaacExtractRTXSensorPointCloud: gmo->numElements is 0. Skipping execution.");
            return false;
        }
        if (gmo->modality != omni::sensors::Modality::LIDAR && gmo->modality != omni::sensors::Modality::RADAR)
        {
            CARB_LOG_WARN(
                "IsaacExtractRTXSensorPointCloud: dataPtr input is not from a Lidar or Radar prim. Buffer will not be parsed.");
            return false;
        }
        auto& state = db.perInstanceState<OgnIsaacExtractRTXSensorPointCloud>();

        // Reference to output transform matrix
        auto& matrixOutput = *reinterpret_cast<omni::math::linalg::matrix4d*>(&db.outputs.transform());
        getTransformFromSensorPose(gmo->frameEnd, matrixOutput);

        // If the source GMO buffer is on the host, we'll use the first device (0) for the scratch buffers
        int localGmoDeviceIndex = attributes.device == -1 ? 0 : attributes.device;
        // Set cudaMemcpyKinds based on whether the source GMO buffer is on the host or device
        cudaMemcpyKind cudaMemcpyKindInput = attributes.device == -1 ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;
        cudaMemcpyKind cudaMemcpyKindOutput = attributes.device == -1 ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice;

        isaacsim::core::includes::ScopedDevice scopedDev(localGmoDeviceIndex);

        state.m_pcBuffer.setDevice(localGmoDeviceIndex);
        state.m_scratchBuffer1.setDevice(localGmoDeviceIndex);
        state.m_scratchBuffer2.setDevice(localGmoDeviceIndex);
        state.m_scratchBuffer3.setDevice(localGmoDeviceIndex);
        state.m_scratchBuffer4.setDevice(localGmoDeviceIndex);

        state.hostPcBuffer.resize(gmo->numElements, make_float3(0.0f, 0.0f, 0.0f));
        state.m_pcBuffer.resize(gmo->numElements);
        state.m_scratchBuffer1.resize(gmo->numElements);
        state.m_scratchBuffer2.resize(gmo->numElements);
        state.m_scratchBuffer3.resize(gmo->numElements);
        state.m_scratchBuffer4.resize(gmo->numElements);

        if (gmo->elementsCoordsType == omni::sensors::CoordsType::SPHERICAL)
        {

            // TODO (adevalla): Assumes incoming GMO is on host, but that's not guaranteed
            // Copy x, y, z to scratch buffers (az/el/dist)
            state.m_scratchBuffer1.copyAsync(gmo->elements.x, gmo->numElements, cudaMemcpyKindInput, cudaStream);
            state.m_scratchBuffer2.copyAsync(gmo->elements.y, gmo->numElements, cudaMemcpyKindInput, cudaStream);
            state.m_scratchBuffer3.copyAsync(gmo->elements.z, gmo->numElements, cudaMemcpyKindInput, cudaStream);

            // Store sin(elevation) in pcBuffer.z, and cos(elevation) in scratchBuffer4
            elevation(state.m_scratchBuffer2.data(), state.m_pcBuffer.data(), state.m_scratchBuffer4.data(), 0.0f,
                      gmo->numElements, localGmoDeviceIndex, cudaStream);

            // Store sin(azimuth) in pcBuffer.x, and cos(azimuth) in pcBuffer.y
            azimuthDegToRad(state.m_scratchBuffer1.data(), state.m_pcBuffer.data(), 0.0f, gmo->numElements,
                            localGmoDeviceIndex, cudaStream);

            // Store transformed Cartesian points in pcBuffer
            pointCloudWithTransform(state.m_pcBuffer.data(), state.m_scratchBuffer4.data(),
                                    state.m_scratchBuffer3.data(), make_float3(0.0f, 0.0f, 0.0f), nullptr,
                                    gmo->numElements, localGmoDeviceIndex, cudaStream);

            // Copy pcBuffer to host
            // TODO (adevalla): Keep buffer on device
            cudaMemcpyAsync(state.hostPcBuffer.data(), state.m_pcBuffer.data(), gmo->numElements * sizeof(float3),
                            cudaMemcpyKindOutput, cudaStream);
        }
        else
        {
            // TODO (adevalla): Assumes incoming GMO is on host, but that's not guaranteed
            for (size_t i = 0; i < gmo->numElements; i++)
            {
                // // skip invalid points
                // if ((gmo->elements.flags[i] & omni::sensors::ElementFlags::VALID) !=
                // omni::sensors::ElementFlags::VALID)
                // {
                //     continue;
                // }
                state.hostPcBuffer.data()[i] = make_float3(gmo->elements.x[i], gmo->elements.y[i], gmo->elements.z[i]);
            }
        }

        // Set output metadata
        db.outputs.dataPtr() = reinterpret_cast<uint64_t>(state.hostPcBuffer.data());
        db.outputs.sensorOutputBuffer() = db.inputs.dataPtr();
        db.outputs.cudaDeviceIndex() = isDevicePtr ? attributes.device : -1;
        db.outputs.cudaStream() = db.inputs.cudaStream();
        db.outputs.bufferSize() = gmo->numElements * sizeof(float3);
        db.outputs.width() = gmo->numElements;
        db.outputs.height() = 1;

        // Return success and enable downstream execution
        db.outputs.exec() = kExecutionAttributeStateEnabled;
        return true;
    }
};

REGISTER_OGN_NODE()
} // rtx
} // sensors
} // isaacsim
