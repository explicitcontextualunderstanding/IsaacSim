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

