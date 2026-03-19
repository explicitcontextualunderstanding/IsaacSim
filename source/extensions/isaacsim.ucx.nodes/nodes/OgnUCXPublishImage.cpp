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

#include <carb/RenderingTypes.h>

#include <isaacsim/ucx/nodes/UcxPublishImageNodeBase.h>

#include <OgnUCXPublishImageDatabase.h>

/**
 * @class OgnUCXPublishImage
 * @brief OmniGraph node for publishing camera images via UCX.
 * @details
 * This node publishes camera image data over UCX using tagged communication.
 * It supports RGB8, RGBA8, and BGRA8 image formats.
 * Supports both CPU and GPU memory sources.
 */
class OgnUCXPublishImage : public UCXPublishImageNodeBase<OgnUCXPublishImageDatabase>
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
        auto& state = OgnUCXPublishImageDatabase::sPerInstanceState<OgnUCXPublishImage>(nodeObj, instanceId);
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
    static bool compute(OgnUCXPublishImageDatabase& db)
    {
        const uint16_t port = static_cast<uint16_t>(db.inputs.port());
        const uint64_t tag = db.inputs.tag();

        // Get the per-instance state and call the instance method
        auto& state = db.template perInstanceState<OgnUCXPublishImage>();
        return state.computeImpl(db, port, tag);
    }

protected:
    /**
     * @brief Generate message from image metadata and pixel data.
     * @details
     * Serializes image metadata and data, handling both CPU and GPU memory sources.
     * Message format (ROS2-compatible):
     * - timestamp (double, 8 bytes)
     * - width (uint32_t, 4 bytes)
     * - height (uint32_t, 4 bytes)
     * - encoding_length (uint32_t, 4 bytes)
     * - encoding (variable length string, padded to 4-byte boundary)
     * - step (uint32_t, 4 bytes) - row length in bytes
     * - image_data (variable length)
     *
     * @param[in] metadata Image metadata
     * @param[in] db Database accessor for pixel data access
     * @return std::vector<uint8_t> Serialized message data
     */
    std::vector<uint8_t> generateMessage(const isaacsim::ucx::nodes::ImageMetadata& metadata,
                                         OgnUCXPublishImageDatabase& db) override
    {
        const int cudaDeviceIndex = db.inputs.cudaDeviceIndex();
        const size_t bufferSize = db.inputs.bufferSize();

        // Determine data size - may need to calculate for GPU textures
        size_t dataSize = metadata.dataSize;
        if (dataSize == 0 && cudaDeviceIndex != -1)
        {
            // GPU texture - calculate size based on format
            const carb::Format resourceFormat = static_cast<carb::Format>(db.inputs.format());
            if (resourceFormat == carb::Format::eR32_SFLOAT)
            {
                dataSize = metadata.width * metadata.height * sizeof(float);
            }
            else
            {
                db.logError("GPU texture with unknown format: %d", static_cast<int>(resourceFormat));
                return {};
            }
        }

        if (dataSize == 0)
        {
            db.logError("Cannot determine image data size - no valid data source");
            return {};
        }

        // Calculate step if not set
        uint32_t step = metadata.step;
        if (step == 0 && metadata.height > 0)
        {
            step = static_cast<uint32_t>(dataSize / metadata.height);
        }

        // Calculate message size with 4-byte alignment after encoding
        const size_t encodingLength = metadata.encoding.length();
        const size_t encodingPadded = (encodingLength + 3) & ~3; // Round up to 4-byte boundary
        const size_t messageSize = sizeof(double) + sizeof(uint32_t) * 4 + encodingPadded + dataSize;

        // Allocate message buffer
        std::vector<uint8_t> messageData(messageSize);
        size_t offset = 0;

        // Write header
        std::memcpy(messageData.data() + offset, &metadata.timestamp, sizeof(double));
        offset += sizeof(double);

        std::memcpy(messageData.data() + offset, &metadata.width, sizeof(uint32_t));
        offset += sizeof(uint32_t);

        std::memcpy(messageData.data() + offset, &metadata.height, sizeof(uint32_t));
        offset += sizeof(uint32_t);

        const uint32_t encLen = static_cast<uint32_t>(encodingLength);
        std::memcpy(messageData.data() + offset, &encLen, sizeof(uint32_t));
        offset += sizeof(uint32_t);

        std::memcpy(messageData.data() + offset, metadata.encoding.c_str(), encodingLength);
        offset += encodingLength;

        // Pad to 4-byte boundary (fill with zeros)
        while (offset % 4 != 0)
        {
            messageData[offset] = 0;
            offset++;
        }

        // Write step (bytes per row)
        std::memcpy(messageData.data() + offset, &step, sizeof(uint32_t));
        offset += sizeof(uint32_t);

        // Copy image data based on source location
        uint8_t* imageDataDest = messageData.data() + offset;

        if (cudaDeviceIndex == -1)
        {
            // Data is on CPU
            if (db.inputs.dataPtr() != 0 && bufferSize > 0)
            {
                // Use pointer-based input
                std::memcpy(imageDataDest, reinterpret_cast<const void*>(db.inputs.dataPtr()), dataSize);
            }
            else if (db.inputs.data.size() > 0)
            {
                // Use array-based input
                const auto& imageData = db.inputs.data.cpu();
                std::memcpy(imageDataDest, imageData.data(), dataSize);
            }
            else
            {
                db.logError("No valid CPU image data source available");
                return {};
            }
        }
        else
        {
            // Data is on GPU - copy to host
            isaacsim::core::includes::ScopedDevice scopedDev(cudaDeviceIndex);

            // Manage CUDA stream
            ensureCudaStream(cudaDeviceIndex);

            if (bufferSize == 0)
            {
                // Data in GPU texture
                cudaArray_t levelArray = nullptr;
                CUDA_CHECK(cudaGetMipmappedArrayLevel(
                    &levelArray, reinterpret_cast<cudaMipmappedArray_t>(db.inputs.dataPtr()), 0));

                const carb::Format resourceFormat = static_cast<carb::Format>(db.inputs.format());
                switch (resourceFormat)
                {
                case carb::Format::eR32_SFLOAT:
                    if (metadata.width * metadata.height * sizeof(float) != dataSize)
                    {
                        db.logError("Data size mismatch for eR32_SFLOAT format");
                        return {};
                    }
                    CUDA_CHECK(cudaMemcpy2DFromArrayAsync(imageDataDest, metadata.width * sizeof(float), levelArray, 0,
                                                          0, metadata.width * sizeof(float), metadata.height,
                                                          cudaMemcpyDeviceToHost, getCudaStream()));
                    CUDA_CHECK(cudaStreamSynchronize(getCudaStream()));
                    break;
                default:
                    db.logError("GPU texture format (%d) is not supported", static_cast<int>(resourceFormat));
                    return {};
                }
            }
            else
            {
                // Data in GPU buffer
                CUDA_CHECK(cudaMemcpyAsync(imageDataDest, reinterpret_cast<void*>(db.inputs.dataPtr()), bufferSize,
                                           cudaMemcpyDeviceToHost, getCudaStream()));
                CUDA_CHECK(cudaStreamSynchronize(getCudaStream()));
            }
        }

        return messageData;
    }
};


REGISTER_OGN_NODE()
