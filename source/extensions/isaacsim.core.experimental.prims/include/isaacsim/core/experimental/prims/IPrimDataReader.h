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

#include <carb/Interface.h>

#include <cstddef>
#include <cstdint>
#include <functional>

namespace isaacsim
{
namespace core
{
namespace experimental
{
namespace prims
{

/**
 * @struct IXformDataView
 * @brief Read-only view for XformPrim data (positions, orientations, scales).
 * @details Engine-agnostic transform data read via IFabricHierarchy.
 * All getters use lazy fetch with step-based staleness detection via
 * ISimulationManager::getNumPhysicsSteps(). Safe to call from multiple
 * onPhysicsStep callbacks.
 */
struct IXformDataView
{
    virtual ~IXformDataView() = default;

    // Transform getters (native device pointers)
    virtual const float* getWorldPositions(int* outCount) = 0;
    virtual const float* getWorldOrientations(int* outCount) = 0;
    virtual const float* getLocalTranslations(int* outCount) = 0;
    virtual const float* getLocalOrientations(int* outCount) = 0;
    virtual const float* getLocalScales(int* outCount) = 0;

    // Host variants (always CPU pointers, copy from GPU if needed)
    virtual const float* getWorldPositionsHost(int* outCount) = 0;
    virtual const float* getWorldOrientationsHost(int* outCount) = 0;
    virtual const float* getLocalTranslationsHost(int* outCount) = 0;
    virtual const float* getLocalOrientationsHost(int* outCount) = 0;
    virtual const float* getLocalScalesHost(int* outCount) = 0;

    // Batch pre-fetch all fields for this view
    virtual bool update() = 0;

    // Buffer / callback management (used by Python during setup)
    virtual bool allocateBufferFloat(const char* fieldName, size_t count) = 0;
    virtual bool allocateBufferUint8(const char* fieldName, size_t count) = 0;
    virtual uintptr_t getBufferPtr(const char* fieldName) = 0;
    virtual size_t getBufferSize(const char* fieldName) = 0;
    virtual int getBufferDevice() = 0;
    virtual void registerFieldCallback(const char* fieldName, std::function<void()> callback) = 0;
};

/**
 * @struct IRigidBodyDataView
 * @brief Read-only view for RigidPrim data. Extends IXformDataView with physics state.
 */
struct IRigidBodyDataView : public IXformDataView
{
    virtual const float* getLinearVelocities(int* outCount) = 0;
    virtual const float* getAngularVelocities(int* outCount) = 0;

    // Host variants
    virtual const float* getLinearVelocitiesHost(int* outCount) = 0;
    virtual const float* getAngularVelocitiesHost(int* outCount) = 0;
};

/**
 * @struct IArticulationDataView
 * @brief Read-only view for Articulation data. Extends IXformDataView with DOF/link/dynamics.
 */
struct IArticulationDataView : public IXformDataView
{
    virtual const float* getDofPositions(int* outCount) = 0;
    virtual const float* getDofVelocities(int* outCount) = 0;
    virtual const float* getDofEfforts(int* outCount) = 0;
    virtual const float* getRootTransforms(int* outCount) = 0;
    virtual const float* getRootVelocities(int* outCount) = 0;
    virtual const uint8_t* getDofTypes(int* outCount) = 0;

    // Host variants
    virtual const float* getDofPositionsHost(int* outCount) = 0;
    virtual const float* getDofVelocitiesHost(int* outCount) = 0;
    virtual const float* getDofEffortsHost(int* outCount) = 0;
    virtual const float* getRootTransformsHost(int* outCount) = 0;
    virtual const float* getRootVelocitiesHost(int* outCount) = 0;
    virtual const uint8_t* getDofTypesHost(int* outCount) = 0;

    /**
     * @brief Resolve the DOF index for a joint given its USD prim path.
     * @param dofPrimPath USD path to the joint prim.
     * @return DOF index (>= 0), or -1 if the joint is not found in this view.
     */
    virtual int getDofIndex(const char* dofPrimPath) = 0;

    /**
     * @brief Get DOF names in articulation order (index 0 .. N-1).
     * @param outCount Receives the number of DOFs.
     * @return Pointer to array of C-strings, or nullptr if none. Valid until view is removed or reader re-initialized.
     */
    virtual const char* const* getDofNames(int* outCount) = 0;
};

/**
 * @struct IPrimDataReader
 * @brief Factory interface for creating typed read-only prim data views.
 * @details Carbonite plugin interface that creates typed views for XformPrim,
 * RigidPrim, and Articulation data. Views provide compile-time safety:
 * only getters valid for the view type are available.
 *
 * Supports both PhysX (direct C++ TensorApi) and Newton (Python callback)
 * backends transparently.
 *
 * For multi-plugin usage, prefer acquiring and using IPrimDataReaderManager
 * so lifecycle initialization is centralized across sensors/nodes. In new
 * call sites, initialize() should be treated as manager-owned API surface.
 *
 * **Threading:** All methods must be called from the main thread (the
 * same thread that drives the simulation). Getter callbacks registered
 * on views may be invoked from physics step callbacks on that thread.
 *
 * **View lifetime:** Pointers returned by create*View() are owned by
 * the plugin. They remain valid until removeView() is called for the
 * same viewId, or until a generation change (detectable via
 * getGeneration()). Consumers should snapshot the generation when
 * creating a view and recheck before dereferencing cached pointers.
 */
struct IPrimDataReader
{
    CARB_PLUGIN_INTERFACE("isaacsim::core::experimental::prims::IPrimDataReader", 1, 0);

    /**
     * @brief Initialize the reader with the current USD stage and simulation device.
     * @details Each call destroys all existing views, recreates the internal
     * PhysX simulation view, and increments the generation counter. This is
     * required because PhysX invalidates simulation views on timeline
     * stop/play even when the stageId is unchanged. Consumers must
     * recreate their views after each initialize() call. Prefer calling
     * IPrimDataReaderManager::ensureInitialized() instead of calling this
     * directly from sensor/node plugins.
     * @param stageId Fabric stage ID (from UsdUtilsStageCache).
     * @param deviceOrdinal CUDA device ordinal (-1 for CPU, >=0 for GPU).
     */
    virtual void initialize(long stageId, int deviceOrdinal) = 0;

    /**
     * @brief Shut down the reader, destroying all views and freeing all buffers.
     */
    virtual void shutdown() = 0;

    /**
     * @brief Create a typed XformPrim data view.
     * @param viewId Unique string identifier for this view.
     * @param paths Array of prim path strings (may include regex patterns).
     * @param numPaths Number of paths in the array.
     * @param engineType Physics engine backend: "physx" or "newton".
     * @return Pointer to view, owned by the plugin. Valid until removeView() or shutdown().
     */
    virtual IXformDataView* createXformView(const char* viewId,
                                            const char** paths,
                                            size_t numPaths,
                                            const char* engineType) = 0;

    /**
     * @brief Create a typed RigidPrim data view (includes XformPrim getters).
     * @param viewId Unique string identifier for this view.
     * @param paths Array of prim path strings.
     * @param numPaths Number of paths in the array.
     * @param engineType Physics engine backend: "physx" or "newton".
     * @return Pointer to view, owned by the plugin. Valid until removeView() or shutdown().
     */
    virtual IRigidBodyDataView* createRigidBodyView(const char* viewId,
                                                    const char** paths,
                                                    size_t numPaths,
                                                    const char* engineType) = 0;

    /**
     * @brief Create a typed Articulation data view (includes XformPrim getters).
     * @param viewId Unique string identifier for this view.
     * @param paths Array of prim path strings.
     * @param numPaths Number of paths in the array.
     * @param engineType Physics engine backend: "physx" or "newton".
     * @return Pointer to view, owned by the plugin. Valid until removeView() or shutdown().
     */
    virtual IArticulationDataView* createArticulationView(const char* viewId,
                                                          const char** paths,
                                                          size_t numPaths,
                                                          const char* engineType) = 0;

    /**
     * @brief Destroy a view and free its buffers.
     * @param viewId Identifier of the view to remove.
     */
    virtual void removeView(const char* viewId) = 0;

    /**
     * @brief Set DOF names and types for an articulation view (used by Newton backend from Python).
     * @param viewId Identifier of the articulation view.
     * @param names Array of DOF name C-strings.
     * @param numNames Number of names.
     * @param types Array of DOF types (0 = rotation, 1 = translation).
     * @param numTypes Number of types.
     */
    virtual void setArticulationDofMetadata(
        const char* viewId, const char** names, size_t numNames, const uint8_t* types, size_t numTypes) = 0;

    /**
     * @brief Monotonically increasing counter incremented on each initialize() call.
     * @details Consumers can snapshot this value when creating views and compare
     * later to detect whether the underlying simulation view has been recreated,
     * which invalidates all previously returned data view pointers.
     */
    virtual uint64_t getGeneration() const = 0;

    /**
     * @brief Get the Fabric stage ID passed to the last initialize() call.
     * @return Stage ID, or 0 if not yet initialized.
     */
    virtual long getStageId() const = 0;

    /**
     * @brief Get the effective CUDA device ordinal.
     * @details After initialize(), this reflects the device ordinal reported by
     * the underlying simulation view (which may differ from the value originally
     * passed to initialize()).
     * @return CUDA device ordinal (>=0 for GPU, -1 for CPU).
     */
    virtual int getDeviceOrdinal() const = 0;
};

} // namespace prims
} // namespace experimental
} // namespace core
} // namespace isaacsim
