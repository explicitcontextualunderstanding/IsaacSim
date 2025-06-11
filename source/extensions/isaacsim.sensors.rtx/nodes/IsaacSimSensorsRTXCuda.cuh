// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#pragma once

#include "GenericModelOutput.h"

namespace isaacsim
{
namespace sensors
{
namespace rtx
{

void retrieveGMOBufferFromDevice(void* dataPtr, omni::sensors::GenericModelOutput& outBuffer, const cudaStream_t& stream) noexcept;

void getGMOBufferFromDevice(void* dataPtr, omni::sensors::GenericModelOutput& hostBuffer, const int cudaDevice, const cudaStream_t& cudaStream) noexcept;

void azimuthDegToRad(float* srcDest, float3* scratch, float accuracyError, int n, int cdi, const cudaStream_t& cudaStream);

void elevation(float* srcDest, float3* scratch, float* scratch2, float accuracyError, int n, int cdi, const cudaStream_t& cudaStream);

void pointCloudWithTransform(float3* srcDest, const float* cosEle, const float* dist, const float3& accuracyError, const double* t, int n, int cdi, const cudaStream_t& cudaStream);

void cartesianToSpherical(float3* srcDest, float* azimuth, float* elevation, float* range, int N, int cdi, const cudaStream_t& cudaStream);

void timestamp(int32_t* dest, int32_t* src, uint64_t tickStartTime, int N, int cdi, const cudaStream_t& cudaStream);

}
}
}

