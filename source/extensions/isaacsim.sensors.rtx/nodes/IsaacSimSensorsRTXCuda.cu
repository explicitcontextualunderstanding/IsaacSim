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

#include <cuda.h>
#include <stdint.h>
#include <stdio.h>

#include "GenericModelOutput.h"
#include "isaacsim/core/includes/ScopedCudaDevice.h"

namespace isaacsim
{
namespace sensors
{
namespace rtx
{

#define DEG2RAD(deg) ((deg) / 180.f * 3.14159265358979323846f)
#define RAD2DEG(rad) ((rad) / 3.14159265358979323846f * 180.f)

__global__ void retrieveGMOBuffer_CUDA(void* dataPtr, omni::sensors::GenericModelOutput& outBuffer) {
    outBuffer = omni::sensors::getModelOutputFromBuffer(dataPtr);
}

void retrieveGMOBufferFromDevice(void* dataPtr, omni::sensors::GenericModelOutput outBuffer, const cudaStream_t& cudaStream) {
    // cudaMalloc some memory for the output buffer
    // cuda copy everything in dataPtr into the output buffer
    // then get that buffer?

    void* gmoBufferDevice;
    CUDA_CHECK(cudaMalloc(&gmoBufferDevice, omni::sensors::sizeBasics()));
    CUDA_CHECK(cudaMemcpyAsync(gmoBufferDevice, dataPtr, omni::sensors::sizeBasics(), cudaMemcpyKind::cudaMemcpyDeviceToDevice, cudaStream));
    CUDA_CHECK(cudaStreamSynchronize(cudaStream));
    retrieveGMOBuffer_CUDA<<<1, 1, 0, cudaStream>>>(gmoBufferDevice, outBuffer);

    // retrieveGMOBuffer_CUDA<<<1, 1, 0, cudaStream>>>(dataPtr, outBuffer);
}

__global__ void getGMODevicePointer_CUDA(void* dataPtr, omni::sensors::GenericModelOutput** devicePtr) {
    // Get a device pointer to the GMO buffer
    *devicePtr = omni::sensors::getModelOutputPtrFromBuffer(dataPtr);
}

void getGMOBufferFromDevice(void* dataPtr, omni::sensors::GenericModelOutput& hostBuffer, const int cudaDevice, const cudaStream_t& cudaStream) {
    omni::sensors::GenericModelOutput** devicePtr;
    CUDA_CHECK(cudaMallocAsync((void**)&devicePtr, sizeof(omni::sensors::GenericModelOutput*), cudaStream));
    getGMODevicePointer_CUDA<<<1, 1, 0, cudaStream>>>(dataPtr, devicePtr);
    omni::sensors::GenericModelOutput* hostPointerToDeviceBuffer;
    CUDA_CHECK(cudaMemcpyAsync(&hostPointerToDeviceBuffer, devicePtr, sizeof(omni::sensors::GenericModelOutput*), cudaMemcpyKind::cudaMemcpyDeviceToHost, cudaStream));
    CUDA_CHECK(cudaStreamSynchronize(cudaStream));
    omni::sensors::cpyGMOToGMO(hostBuffer, *hostPointerToDeviceBuffer, cudaDevice, cudaStream);
    CUDA_CHECK(cudaFreeAsync(devicePtr, cudaStream));
}

__global__ void cartesianToSphericalKernel(float3* srcDest, float* azimuth, float* elevation, float* range, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    float x = srcDest[idx].x;
    float y = srcDest[idx].y;
    float z = srcDest[idx].z;

    float rangeRad = sqrtf(x * x + y * y + z * z);
    float azimuthRad = atan2f(y, x);
    float elevationRad = asinf(z / rangeRad);

    azimuth[idx] = RAD2DEG(azimuthRad);
    elevation[idx] = RAD2DEG(elevationRad);
    range[idx] = RAD2DEG(rangeRad);
}

void cartesianToSpherical(float3* srcDest, float* azimuth, float* elevation, float* range, int N, int cdi, const cudaStream_t& cudaStream)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, cdi);
    const int nt = prop.maxThreadsPerBlock;
    const int nb = (N + nt - 1) / nt;

    cartesianToSphericalKernel<<<nb, nt, 0, cudaStream>>>(srcDest, azimuth, elevation, range, N);
}

// uses the destination and a scratch area to compte and store for use computing pc
// omni.sensor v1.0.0+ provides azimuth in [-180, 180), +CCW
// omni.sensor v0.4.x  provided azimuth in [0, 360), +CW
// srcDest computes final azimuth
// scratch.x = sinAzimuth
// scratch.y = conAzimuth
__global__ void azimuthDegToRadKernel(float* srcDest, float3* scratch, float accuracyError, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;
    srcDest[idx] = DEG2RAD(srcDest[idx] + accuracyError);

    scratch[idx].x = sinf(srcDest[idx]); // notice.x
    scratch[idx].y = cosf(srcDest[idx]); // notice.y
}

void azimuthDegToRad(float* srcDest, float3* scratch, float accuracyError, int N, int cdi, const cudaStream_t& cudaStream)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, cdi);
    const int nt = prop.maxThreadsPerBlock;
    const int nb = (N + nt - 1) / nt;

    azimuthDegToRadKernel<<<nb, nt, 0, cudaStream>>>(srcDest, scratch, accuracyError, N);
}

// uses destination and scrath to compute and store
// const float elevationDeg{ lidarReturns.elevations[idx] + accuracyErrorElevationDeg };
// const float elevationRad{ deg2Rad(elevationDeg) };
// const float sinElevation{ ::sinf(elevationRad) };
// const float cosElevation{ ::cosf(elevationRad) };
// srcDest computes final elevation
// scratch.z = sinElevation
// scratch2 = cosElevation
__global__ void elevationKernel(float* srcDest, float3* scratch, float* scratch2, float accuracyError, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    srcDest[idx] = DEG2RAD(srcDest[idx] + accuracyError);

    const float elevationRad = srcDest[idx];
    scratch[idx].z = sinf(elevationRad); // notice .z
    scratch2[idx] = cosf(elevationRad);
}

void elevation(float* srcDest, float3* scratch, float* scratch2, float accuracyError, int N, int cdi, const cudaStream_t& cudaStream)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, cdi);
    const int nt = prop.maxThreadsPerBlock;
    const int nb = (N + nt - 1) / nt;

    elevationKernel<<<nb, nt, 0, cudaStream>>>(srcDest, scratch, scratch2, accuracyError, N);
}

//    const float rayDirectionX{ cosElevation * cosAzimuth };
//    const float rayDirectionY{ cosElevation * sinAzimuth };
//    const float rayDirectionZ{ sinElevation };
// srdDest has the point cloud location with transforms at the end.
__global__ void pointCloudWithTransformKernel(
    float3* srcDest, const float* cosEle, const float* dist, const float4 t1, const float4 t2, const float4 t3, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;
    const float sinAzimuth = srcDest[idx].x;
    const float cosAzimuth = srcDest[idx].y;
    const float sinElevation = srcDest[idx].z;
    const float cosElevation = cosEle[idx];
    const float distance = dist[idx];
    const float X{ distance * cosElevation * cosAzimuth };
    const float Y{ distance * cosElevation * sinAzimuth };
    const float Z{ distance * sinElevation };
    srcDest[idx] = make_float3(t1.x * X + t1.y * Y + t1.z * Z + t1.w, t2.x * X + t2.y * Y + t2.z * Z + t2.w,
                               t3.x * X + t3.y * Y + t3.z * Z + t3.w);
}

void pointCloudWithTransform(
    float3* srcDest, const float* cosEle, const float* dist, const float3& accuracyError, const double* T, int N, int cdi, const cudaStream_t& cudaStream)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, cdi);
    const int nt = prop.maxThreadsPerBlock;
    const int nb = (N + nt - 1) / nt;

    float4 t1, t2, t3;
    if (T)
    {
        t1 = make_float4(T[0], T[4], T[8], T[12]);
        t2 = make_float4(T[1], T[5], T[9], T[13]);
        t3 = make_float4(T[2], T[6], T[10], T[14]);
    }
    else
    {
        t1 = make_float4(1, 0, 0, 0);
        t2 = make_float4(0, 1, 0, 0);
        t3 = make_float4(0, 0, 1, 0);
    }
    t1.w += accuracyError.x;
    t2.w += accuracyError.y;
    t3.w += accuracyError.z;
    pointCloudWithTransformKernel<<<nb, nt, 0, cudaStream>>>(srcDest, cosEle, dist, t1, t2, t3, N);
}

__global__ void timestampKernel(int32_t* dest, int32_t* src, uint64_t tickStartTime, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    dest[idx] = src[idx] + tickStartTime;
}

void timestamp(int32_t* dest, int32_t* src, uint64_t tickStartTime, int N, int cdi, const cudaStream_t& cudaStream)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, cdi);
    const int nt = prop.maxThreadsPerBlock;
    const int nb = (N + nt - 1) / nt;

    timestampKernel<<<nb, nt, 0, cudaStream>>>(dest, src, tickStartTime, N);
}

}   // namespace isaacsim
}   // namespace sensors
}   // namespace rtx
