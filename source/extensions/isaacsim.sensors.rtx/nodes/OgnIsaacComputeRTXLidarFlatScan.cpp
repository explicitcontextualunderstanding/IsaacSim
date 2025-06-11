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
#include <pch/UsdPCH.h>
// clang-format on

#include "GenericModelOutput.h"
#include "OgnIsaacComputeRTXLidarFlatScanDatabase.h"
#include "SensorNodeUtils.h"
#include "isaacsim/core/includes/ScopedCudaDevice.h"
#include "isaacsim/core/includes/UsdUtilities.h"

#include <math.h>

namespace isaacsim
{
namespace sensors
{
namespace rtx
{

class OgnIsaacComputeRTXLidarFlatScan : public LidarConfigHelper
{
public:
    struct FlatScanDataHandle
    {
        void* hostGMO{ nullptr }; // pointer to GMO buffer on host
        OgnIsaacComputeRTXLidarFlatScanDatabase* db{ nullptr }; // pointer to node state
        bool returnVal{ false }; // return value for node
    };

    static bool compute(OgnIsaacComputeRTXLidarFlatScanDatabase& db)
    {
        // Disable downstream execution by default
        db.outputs.exec() = kExecutionAttributeStateDisabled;

        // Retrieve and test input data pointer
        void* dataPtr = reinterpret_cast<void*>(db.inputs.dataPtr());
        if (!dataPtr)
        {
            CARB_LOG_INFO("IsaacComputeRTXLidarFlatScan: dataPtr input is empty.");
            return false;
        }

        // Determine if input data pointer is on device or host, based on associated CUDA stream
        cudaStream_t cudaStream = (cudaStream_t)db.inputs.cudaStream();
        cudaPointerAttributes attributes;
        CUDA_CHECK(cudaPointerGetAttributes(&attributes, dataPtr));
        const bool isDevicePtr{ attributes.type == cudaMemoryTypeDevice };

        if (isDevicePtr)
        {
            CARB_LOG_ERROR("IsaacComputeRTXLidarFlatScan.inputs:dataPtr unexpectedly on device. Expecting input on host.");
            return false;
        }

        // Allocate a data handle on host via CUDA
        FlatScanDataHandle* dataHandle;
        CUDA_CHECK(cudaMallocHost((void**)&dataHandle, sizeof(FlatScanDataHandle)));
        dataHandle->hostGMO = dataPtr;
        dataHandle->db = &db;
        dataHandle->returnVal = false;

        // Launch a host function in the provided stream to actually perform the node's work
        CUDA_CHECK(cudaLaunchHostFunc(
            cudaStream,
            [](void* userData)
            {
                // Decompose the user data struct
                auto flatScanDataHandle = reinterpret_cast<FlatScanDataHandle*>(userData);
                auto hostGMO = omni::sensors::getModelOutputPtrFromBuffer(flatScanDataHandle->hostGMO);
                auto db = flatScanDataHandle->db;
                bool& returnVal = flatScanDataHandle->returnVal;

                if (hostGMO->numElements == 0)
                {
                    CARB_LOG_INFO("IsaacComputeRTXLidarFlatScan: hostGMO->numElements is 0. Skipping execution.");
                    returnVal = false;
                    return;
                }

                // Verify that we have a supported modality
                if (hostGMO->modality != omni::sensors::Modality::LIDAR)
                {
                    CARB_LOG_WARN(
                        "IsaacComputeRTXLidarFlatScan: Unsupported sensor modality: %d. Only LIDAR is supported.",
                        static_cast<int>(hostGMO->modality));
                    returnVal = false;
                    return;
                }

                // Retrieve lidar prim from render product path, then validate its attributes
                const std::string renderProductPath = std::string(db->tokenToString(db->inputs.renderProductPath()));
                if (renderProductPath.length() == 0)
                {
                    CARB_LOG_WARN("IsaacComputeRTXLidarFlatScan: renderProductPath input is empty. Skipping execution.");
                    returnVal = false;
                    return;
                }
                float rotationRate, horizontalFov, horizontalResolution, azimuthRangeStart, azimuthRangeEnd, nearRangeM,
                    farRangeM;
                pxr::UsdPrim lidarPrim = isaacsim::core::includes::getCameraPrimFromRenderProduct(renderProductPath);
                if (lidarPrim.IsA<pxr::UsdGeomCamera>())
                {
                    auto& state = db->perInstanceState<OgnIsaacComputeRTXLidarFlatScan>();
                    CARB_LOG_INFO(
                        "RTX sensors as camera prims are deprecated as of Isaac Sim 5.0, and support will be removed in a future release. Please use an OmniLidar prim with the new OmniSensorGenericLidarCoreAPI schema.");
                    bool updatedConfig = state.updateLidarConfig(renderProductPath.c_str());
                    if (state.scanType == LidarScanType::kUnknown)
                    {
                        if (updatedConfig)
                        {
                            CARB_LOG_WARN(
                                "IsaacComputeRTXLidarFlatScan: Lidar prim is not a valid Lidar. Skipping execution.");
                        }
                        returnVal = false;
                        return;
                    }
                    if (!state.is2D)
                    {
                        if (updatedConfig)
                        {
                            CARB_LOG_WARN(
                                "IsaacComputeRTXLidarFlatScan: Lidar prim is not a 2D Lidar. Skipping execution.");
                        }
                        returnVal = false;
                        return;
                    }
                    if (state.scanType == LidarScanType::kSolidState)
                    {
                        azimuthRangeStart = state.azimuthStartDeg;
                        azimuthRangeEnd = state.azimuthEndDeg;
                        horizontalFov = azimuthRangeStart - azimuthRangeEnd;
                        horizontalResolution = horizontalFov / static_cast<float>(state.numberOfEmitters);
                    }
                    else
                    {
                        horizontalResolution = 360.0f * state.scanRateBaseHz / state.reportRateBaseHz;
                        azimuthRangeStart = -180.0f;
                        azimuthRangeEnd = 180.0f - horizontalResolution;
                        horizontalFov = 360.0f;
                    }
                    nearRangeM = state.nearRangeM;
                    farRangeM = state.farRangeM;
                    rotationRate = static_cast<float>(state.scanRateBaseHz);
                }
                else
                {
                    pxr::TfToken elementsCoordsType;
                    if (lidarPrim.GetAttribute(pxr::TfToken("omni:sensor:Core:elementsCoordsType")).Get(&elementsCoordsType) &&
                        elementsCoordsType != pxr::TfToken("SPHERICAL"))
                    {
                        CARB_LOG_WARN(
                            "IsaacComputeRTXLidarFlatScan: Lidar prim elementsCoordsType is not set to SPHERICAL. Skipping execution.");
                        returnVal = false;
                        return;
                    }
                    pxr::VtFloatArray elevationDeg;
                    if (lidarPrim.GetAttribute(pxr::TfToken("omni:sensor:Core:emitterState:s001:elevationDeg"))
                            .Get(&elevationDeg))
                    {
                        const float epsilon = 1e-3f; // Tolerance of 0.001 degrees
                        for (const float elev : elevationDeg)
                        {
                            if (::fabs(elev) > epsilon)
                            {
                                CARB_LOG_WARN(
                                    "IsaacComputeRTXLidarFlatScan: Lidar prim elevationDeg contains nonzero value %f, indicating a 3D lidar. Skipping execution.",
                                    elev);
                                returnVal = false;
                                return;
                            }
                        }
                    }
                    // Populate any prim-specific outputs
                    uint32_t rotationRateAsInt;
                    lidarPrim.GetAttribute(pxr::TfToken("omni:sensor:Core:scanRateBaseHz")).Get(&rotationRateAsInt);
                    rotationRate = static_cast<float>(rotationRateAsInt);
                    lidarPrim.GetAttribute(pxr::TfToken("omni:sensor:Core:nearRangeM")).Get(&nearRangeM);
                    lidarPrim.GetAttribute(pxr::TfToken("omni:sensor:Core:farRangeM")).Get(&farRangeM);

                    pxr::TfToken outputType;
                    lidarPrim.GetAttribute(pxr::TfToken("omni:sensor:Core:scanType")).Get(&outputType);
                    if (outputType == pxr::TfToken("SOLID_STATE"))
                    {
                        pxr::VtFloatArray azimuthDeg;
                        lidarPrim.GetAttribute(pxr::TfToken("omni:sensor:Core:emitterState:s001:azimuthDeg")).Get(&azimuthDeg);

                        azimuthRangeStart = *std::min_element(azimuthDeg.begin(), azimuthDeg.end());
                        azimuthRangeEnd = *std::max_element(azimuthDeg.begin(), azimuthDeg.end());
                        horizontalFov = azimuthRangeEnd - azimuthRangeStart;
                        horizontalResolution = horizontalFov / static_cast<float>(azimuthDeg.size());
                    }
                    else
                    {
                        // Set useful state variables
                        uint32_t reportRateBaseHzAsInt;
                        lidarPrim.GetAttribute(pxr::TfToken("omni:sensor:Core:reportRateBaseHz")).Get(&reportRateBaseHzAsInt);
                        float reportRateBaseHz = static_cast<float>(reportRateBaseHzAsInt);
                        horizontalResolution = 360.0f * rotationRate / reportRateBaseHz;

                        azimuthRangeStart = -180.0f;
                        azimuthRangeEnd = 180.0f - horizontalResolution;
                        horizontalFov = 360.0;
                    }
                }

                db->outputs.horizontalFov() = horizontalFov;
                db->outputs.horizontalResolution() = horizontalResolution;
                db->outputs.azimuthRange() = { azimuthRangeStart, azimuthRangeEnd };
                db->outputs.rotationRate() = rotationRate;
                db->outputs.depthRange() = { nearRangeM, farRangeM };
                db->outputs.numRows() = 1;
                db->outputs.numCols() = hostGMO->numElements;

                // Populate output buffers
                db->outputs.linearDepthData().resize(hostGMO->numElements);
                db->outputs.intensitiesData().resize(hostGMO->numElements);

                // Create a map of azimuth to depth and intensity, to automatically sort by azimuth
                std::map<float, std::pair<float, uint8_t>> azimuthToDepthAndIntensity;
                for (size_t i = 0; i < hostGMO->numElements; i++)
                {
                    // Skip invalid returns
                    if ((hostGMO->elements.flags[i] & omni::sensors::ElementFlags::VALID) !=
                        omni::sensors::ElementFlags::VALID)
                    {
                        continue;
                    }
                    azimuthToDepthAndIntensity[hostGMO->elements.x[i]] = {
                        hostGMO->elements.z[i], static_cast<uint8_t>(hostGMO->elements.scalar[i] * 255.0f)
                    };
                }
                // Copy sorted values into output buffers
                db->outputs.linearDepthData().resize(azimuthToDepthAndIntensity.size());
                db->outputs.intensitiesData().resize(azimuthToDepthAndIntensity.size());
                db->outputs.numCols() = static_cast<int>(azimuthToDepthAndIntensity.size());
                size_t index = 0;
                for (const auto& [azimuth, depthAndIntensity] : azimuthToDepthAndIntensity)
                {
                    db->outputs.linearDepthData()[index] = depthAndIntensity.first;
                    db->outputs.intensitiesData()[index] = depthAndIntensity.second;
                    index++;
                }

                // returnVal=success and enable downstream execution
                db->outputs.exec() = kExecutionAttributeStateEnabled;
                returnVal = true;
                return;
            },
            dataHandle));

        bool returnVal = dataHandle->returnVal;
        CUDA_CHECK(cudaFreeHost(dataHandle));
        return returnVal;
    }
};

REGISTER_OGN_NODE()
} // rtx
} // sensors
} // isaacsim
