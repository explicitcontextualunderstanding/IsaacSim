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

#include <carb/PluginUtils.h>
#include <carb/events/EventsUtils.h>

#include <isaacsim/core/experimental/prims/BufferRegistry.h>
#include <isaacsim/core/experimental/prims/IPrimDataReader.h>
#include <isaacsim/core/experimental/prims/IPrimDataReaderManager.h>
#include <isaacsim/core/includes/Pose.h>
#include <isaacsim/core/simulation_manager/ISimulationManager.h>
#include <omni/ext/IExt.h>
#include <omni/fabric/FabricUSD.h>
#include <omni/physics/simulation/IPhysics.h>
#include <omni/physics/simulation/IPhysicsStageUpdate.h>
#include <omni/physics/tensors/IArticulationView.h>
#include <omni/physics/tensors/IRigidBodyView.h>
#include <omni/physics/tensors/ISimulationView.h>
#include <omni/physics/tensors/TensorApi.h>
#include <omni/usd/UsdContext.h>

#if defined(_WIN32)
#    include <usdrt/scenegraph/usd/usd/stage.h>
#else
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wunused-variable"
#    pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#    include <usdrt/scenegraph/usd/usd/stage.h>
#    pragma GCC diagnostic pop
#endif

#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>

const struct carb::PluginImplDesc g_kPluginDesc = { "isaacsim.core.experimental.prims.plugin",
                                                    "C++ read-only prim data reader", "NVIDIA",
                                                    carb::PluginHotReload::eEnabled, "dev" };

namespace isaacsim
{
namespace core
{
namespace experimental
{
namespace prims
{

namespace
{

using simulation_manager::ISimulationManager;
using namespace omni::physics::tensors;

static bool isPhysxActive()
{
    auto* physics = carb::getCachedInterface<omni::physics::IPhysics>();
    if (!physics)
        return false;

    size_t numSims = physics->getNumSimulations();
    std::vector<omni::physics::SimulationId> ids(numSims);
    physics->getSimulationIds(ids.data(), numSims);
    for (const auto& id : ids)
    {
        const char* name = physics->getSimulationName(id);
        if (name && std::string(name) == "PhysX" && physics->isSimulationActive(id))
            return true;
    }
    return false;
}

static void fillTensorDesc(TensorDesc& desc, void* dataPtr, int numElements, TensorDataType type, int device)
{
    desc.dtype = type;
    desc.numDims = 1;
    desc.dims[0] = numElements;
    desc.data = dataPtr;
    desc.ownData = true;
    desc.device = device;
}

/**
 * @brief Decompose a 4x4 matrix into a position (3 floats) and a wxyz quaternion (4 floats).
 */
static void decomposeMatrix(const usdrt::GfMatrix4d& m, float* outputPosition, float* outputQuaternion)
{
    auto translation = m.ExtractTranslation();
    outputPosition[0] = static_cast<float>(translation[0]);
    outputPosition[1] = static_cast<float>(translation[1]);
    outputPosition[2] = static_cast<float>(translation[2]);

    usdrt::GfMatrix4d noScale = m;
    noScale.Orthonormalize();
    auto q = noScale.ExtractRotation();
    outputQuaternion[0] = static_cast<float>(q.GetReal());
    auto imaginary = q.GetImaginary();
    outputQuaternion[1] = static_cast<float>(imaginary[0]);
    outputQuaternion[2] = static_cast<float>(imaginary[1]);
    outputQuaternion[3] = static_cast<float>(imaginary[2]);
}

/**
 * @class BaseDataView
 * @brief Shared implementation of buffer fetch, dirty tracking, and callback dispatch.
 * @details The callback stored in each @c FieldEntry is either a C++ lambda (for PhysX
 * and Fabric paths) or a Python callable wrapped by pybind11 (for Newton). The dirty
 * tracking logic is identical for both.
 */
class BaseDataView
{
protected:
    ViewData* m_data = nullptr;
    ISimulationManager* m_simulationManager = nullptr;

    /**
     * @brief Look up a field by name, run its callback if stale, and return the device buffer pointer.
     * @tparam T Element type (float or uint8_t).
     * @param[in] fields   Map of field name to FieldEntry<T> (fieldsF or fieldsU8).
     * @param[in] name    Field name (e.g. "dof_positions").
     * @param[out] outCount If non-null, receives the number of elements; set to 0 if field not found.
     * @return Pointer to the buffer data, or nullptr if the field does not exist or has no buffer.
     */
    template <typename T>
    const T* _fetchFieldImpl(std::unordered_map<std::string, FieldEntry<T>>& fields, const std::string& name, int* outCount)
    {
        auto it = fields.find(name);
        if (it == fields.end())
        {
            if (outCount)
                *outCount = 0;
            return nullptr;
        }
        auto& field = it->second;
        int64_t step = m_simulationManager ? static_cast<int64_t>(m_simulationManager->getNumPhysicsSteps()) : 0;
        if (field.lastStep < step)
        {
            if (field.callback)
                field.callback();
            field.lastStep = step;
        }
        if (outCount)
            *outCount = static_cast<int>(field.count);
        return field.buffer ? field.buffer->data() : nullptr;
    }

    /**
     * @brief Ensure host staging buffer exists for the field, copy from device if needed, and return host pointer.
     * @tparam T Element type (float or uint8_t).
     * @param[in] fields Map of field name to FieldEntry<T> (fieldsF or fieldsU8).
     * @param[in] name  Field name.
     * @return Pointer to the host staging data, or nullptr if the field does not exist.
     */
    template <typename T>
    const T* _copyToHostAndGet(std::unordered_map<std::string, FieldEntry<T>>& fields, const std::string& name)
    {
        auto it = fields.find(name);
        if (it == fields.end())
            return nullptr;
        auto& field = it->second;
        if (!field.hostStaging)
            field.hostStaging = std::make_unique<includes::GenericBufferBase<T>>(field.count, -1);
        if (field.hostLastStep < field.lastStep)
        {
            field.buffer->copyTo(field.hostStaging->data(), field.count);
            field.hostLastStep = field.lastStep;
        }
        return field.hostStaging->data();
    }

    /**
     * @brief Fetch field data on device (runs callback if stale). Dispatches to float or uint8_t map.
     * @tparam T Element type (float or uint8_t).
     * @param[in] name     Field name (e.g. "dof_positions", "dof_types").
     * @param[out] outCount If non-null, receives the number of elements.
     * @return Pointer to the buffer data, or nullptr if not found.
     */
    template <typename T>
    const T* _fetchFieldT(const std::string& name, int* outCount)
    {
        if constexpr (std::is_same_v<T, float>)
            return _fetchFieldImpl(m_data->fieldsF, name, outCount);
        else if constexpr (std::is_same_v<T, uint8_t>)
            return _fetchFieldImpl(m_data->fieldsU8, name, outCount);
        return nullptr;
    }

    /**
     * @brief Fetch field data on host. If device is GPU, copies to host staging and returns that pointer.
     * @tparam T Element type (float or uint8_t).
     * @param[in] name     Field name.
     * @param[out] outCount If non-null, receives the number of elements.
     * @return Pointer to host-accessible data, or nullptr if field not found.
     */
    template <typename T>
    const T* _fetchFieldHostT(const std::string& name, int* outCount)
    {
        const T* ptr = _fetchFieldT<T>(name, outCount);
        if (!ptr)
            return nullptr;
        if (m_data->deviceOrdinal >= 0)
        {
            if constexpr (std::is_same_v<T, float>)
                return _copyToHostAndGet(m_data->fieldsF, name);
            else if constexpr (std::is_same_v<T, uint8_t>)
                return _copyToHostAndGet(m_data->fieldsU8, name);
            return nullptr;
        }
        return ptr;
    }

    const float* _fetchField(const std::string& name, int* outCount)
    {
        return _fetchFieldT<float>(name, outCount);
    }

    const float* _fetchFieldHost(const std::string& name, int* outCount)
    {
        return _fetchFieldHostT<float>(name, outCount);
    }

    const uint8_t* _fetchFieldU8(const std::string& name, int* outCount)
    {
        return _fetchFieldT<uint8_t>(name, outCount);
    }

    const uint8_t* _fetchFieldU8Host(const std::string& name, int* outCount)
    {
        return _fetchFieldHostT<uint8_t>(name, outCount);
    }

    template <typename T>
    static void _runFieldCallbacksForStep(std::unordered_map<std::string, FieldEntry<T>>& map, int64_t step)
    {
        for (auto& [name, field] : map)
        {
            if (field.lastStep < step && field.callback)
            {
                field.callback();
                field.lastStep = step;
            }
        }
    }

    bool _updateImpl()
    {
        int64_t step = m_simulationManager ? static_cast<int64_t>(m_simulationManager->getNumPhysicsSteps()) : 0;
        _runFieldCallbacksForStep(m_data->fieldsF, step);
        _runFieldCallbacksForStep(m_data->fieldsU8, step);
        return true;
    }

    template <typename T>
    bool _allocateBufferImpl(const char* fieldName, size_t count)
    {
        m_data->getOrCreateField<T>(std::string(fieldName), count, m_data->deviceOrdinal);
        return true;
    }

    uintptr_t _getBufferPtrImpl(const char* fieldName)
    {
        const std::string name(fieldName);
        auto itF = m_data->fieldsF.find(name);
        if (itF != m_data->fieldsF.end() && itF->second.buffer)
            return reinterpret_cast<uintptr_t>(itF->second.buffer->data());
        auto itU8 = m_data->fieldsU8.find(name);
        if (itU8 != m_data->fieldsU8.end() && itU8->second.buffer)
            return reinterpret_cast<uintptr_t>(itU8->second.buffer->data());
        return 0;
    }

    size_t _getBufferSizeImpl(const char* fieldName)
    {
        const std::string name(fieldName);
        auto itF = m_data->fieldsF.find(name);
        if (itF != m_data->fieldsF.end())
            return itF->second.count;
        auto itU8 = m_data->fieldsU8.find(name);
        if (itU8 != m_data->fieldsU8.end())
            return itU8->second.count;
        return 0;
    }

    int _getBufferDeviceImpl()
    {
        return m_data->deviceOrdinal;
    }

    void _registerFieldCallbackImpl(const char* fieldName, std::function<void()> callback)
    {
        const std::string name(fieldName);
        auto itF = m_data->fieldsF.find(name);
        if (itF != m_data->fieldsF.end())
        {
            itF->second.callback = std::move(callback);
            return;
        }
        auto itU8 = m_data->fieldsU8.find(name);
        if (itU8 != m_data->fieldsU8.end())
        {
            itU8->second.callback = std::move(callback);
            return;
        }
        FieldEntry<float> entry;
        entry.callback = std::move(callback);
        m_data->fieldsF.emplace(name, std::move(entry));
    }
};

// Macro to avoid duplicating the IXformDataView boilerplate across three classes.
#define IMPL_XFORM_DATA_VIEW                                                                                           \
    const float* getWorldPositions(int* outCount) override                                                             \
    {                                                                                                                  \
        return _fetchField("world_positions", outCount);                                                               \
    }                                                                                                                  \
    const float* getWorldOrientations(int* outCount) override                                                          \
    {                                                                                                                  \
        return _fetchField("world_orientations", outCount);                                                            \
    }                                                                                                                  \
    const float* getLocalTranslations(int* outCount) override                                                          \
    {                                                                                                                  \
        return _fetchField("local_translations", outCount);                                                            \
    }                                                                                                                  \
    const float* getLocalOrientations(int* outCount) override                                                          \
    {                                                                                                                  \
        return _fetchField("local_orientations", outCount);                                                            \
    }                                                                                                                  \
    const float* getLocalScales(int* outCount) override                                                                \
    {                                                                                                                  \
        return _fetchField("local_scales", outCount);                                                                  \
    }                                                                                                                  \
    const float* getWorldPositionsHost(int* outCount) override                                                         \
    {                                                                                                                  \
        return _fetchFieldHost("world_positions", outCount);                                                           \
    }                                                                                                                  \
    const float* getWorldOrientationsHost(int* outCount) override                                                      \
    {                                                                                                                  \
        return _fetchFieldHost("world_orientations", outCount);                                                        \
    }                                                                                                                  \
    const float* getLocalTranslationsHost(int* outCount) override                                                      \
    {                                                                                                                  \
        return _fetchFieldHost("local_translations", outCount);                                                        \
    }                                                                                                                  \
    const float* getLocalOrientationsHost(int* outCount) override                                                      \
    {                                                                                                                  \
        return _fetchFieldHost("local_orientations", outCount);                                                        \
    }                                                                                                                  \
    const float* getLocalScalesHost(int* outCount) override                                                            \
    {                                                                                                                  \
        return _fetchFieldHost("local_scales", outCount);                                                              \
    }

// Macro for the buffer/callback management methods shared by all view types.
#define IMPL_BUFFER_MANAGEMENT                                                                                         \
    bool update() override                                                                                             \
    {                                                                                                                  \
        return _updateImpl();                                                                                          \
    }                                                                                                                  \
    bool allocateBufferFloat(const char* fieldName, size_t count) override                                             \
    {                                                                                                                  \
        return _allocateBufferImpl<float>(fieldName, count);                                                           \
    }                                                                                                                  \
    bool allocateBufferUint8(const char* fieldName, size_t count) override                                             \
    {                                                                                                                  \
        return _allocateBufferImpl<uint8_t>(fieldName, count);                                                         \
    }                                                                                                                  \
    uintptr_t getBufferPtr(const char* fieldName) override                                                             \
    {                                                                                                                  \
        return _getBufferPtrImpl(fieldName);                                                                           \
    }                                                                                                                  \
    size_t getBufferSize(const char* fieldName) override                                                               \
    {                                                                                                                  \
        return _getBufferSizeImpl(fieldName);                                                                          \
    }                                                                                                                  \
    int getBufferDevice() override                                                                                     \
    {                                                                                                                  \
        return _getBufferDeviceImpl();                                                                                 \
    }                                                                                                                  \
    void registerFieldCallback(const char* fieldName, std::function<void()> callback) override                         \
    {                                                                                                                  \
        _registerFieldCallbackImpl(fieldName, std::move(callback));                                                    \
    }


/**
 * @class XformDataView
 * @brief Concrete view providing read-only access to XformPrim transform data.
 */
class XformDataView final : public IXformDataView, public BaseDataView
{
public:
    XformDataView(ViewData* data, ISimulationManager* simulationManager)
    {
        m_data = data;
        m_simulationManager = simulationManager;
    }
    IMPL_XFORM_DATA_VIEW
    IMPL_BUFFER_MANAGEMENT
};

/**
 * @class RigidBodyDataView
 * @brief Concrete view providing read-only access to RigidPrim data (transforms + velocities).
 */
class RigidBodyDataView final : public IRigidBodyDataView, public BaseDataView
{
public:
    RigidBodyDataView(ViewData* data, ISimulationManager* simulationManager)
    {
        m_data = data;
        m_simulationManager = simulationManager;
    }
    IMPL_XFORM_DATA_VIEW

    const float* getLinearVelocities(int* outCount) override
    {
        return _fetchField("linear_velocities", outCount);
    }
    const float* getAngularVelocities(int* outCount) override
    {
        return _fetchField("angular_velocities", outCount);
    }

    const float* getLinearVelocitiesHost(int* outCount) override
    {
        return _fetchFieldHost("linear_velocities", outCount);
    }
    const float* getAngularVelocitiesHost(int* outCount) override
    {
        return _fetchFieldHost("angular_velocities", outCount);
    }

    IMPL_BUFFER_MANAGEMENT
};

/**
 * @class ArticulationDataView
 * @brief Concrete view providing read-only access to Articulation data (transforms + DOF/link/dynamics).
 */
class ArticulationDataView final : public IArticulationDataView, public BaseDataView
{
public:
    ArticulationDataView(ViewData* data, ISimulationManager* simulationManager)
    {
        m_data = data;
        m_simulationManager = simulationManager;
    }
    IMPL_XFORM_DATA_VIEW

    const float* getDofPositions(int* outCount) override
    {
        return _fetchField("dof_positions", outCount);
    }
    const float* getDofVelocities(int* outCount) override
    {
        return _fetchField("dof_velocities", outCount);
    }
    const float* getDofEfforts(int* outCount) override
    {
        return _fetchField("dof_efforts", outCount);
    }
    const float* getRootTransforms(int* outCount) override
    {
        return _fetchField("root_transforms", outCount);
    }
    const float* getRootVelocities(int* outCount) override
    {
        return _fetchField("root_velocities", outCount);
    }
    const uint8_t* getDofTypes(int* outCount) override
    {
        return _fetchFieldU8("dof_types", outCount);
    }

    const float* getDofPositionsHost(int* outCount) override
    {
        return _fetchFieldHost("dof_positions", outCount);
    }
    const float* getDofVelocitiesHost(int* outCount) override
    {
        return _fetchFieldHost("dof_velocities", outCount);
    }
    const float* getDofEffortsHost(int* outCount) override
    {
        return _fetchFieldHost("dof_efforts", outCount);
    }
    const float* getRootTransformsHost(int* outCount) override
    {
        return _fetchFieldHost("root_transforms", outCount);
    }
    const float* getRootVelocitiesHost(int* outCount) override
    {
        return _fetchFieldHost("root_velocities", outCount);
    }
    const uint8_t* getDofTypesHost(int* outCount) override
    {
        return _fetchFieldU8Host("dof_types", outCount);
    }

    int getDofIndex(const char* dofPrimPath) override
    {
        if (!m_data || !m_data->physxArticulationView || !dofPrimPath)
            return -1;

        auto* articulationView = m_data->physxArticulationView;
        uint32_t maxDofs = articulationView->getMaxDofs();
        std::string target(dofPrimPath);

        for (uint32_t i = 0; i < maxDofs; ++i)
        {
            const char* path = articulationView->getUsdDofPath(0, i);
            if (path && target == path)
                return static_cast<int>(i);
        }
        return -1;
    }

    const char* const* getDofNames(int* outCount) override
    {
        if (outCount)
            *outCount = 0;
        if (!m_data || m_data->dofNamePtrs.empty())
            return nullptr;
        if (outCount)
            *outCount = static_cast<int>(m_data->dofNamePtrs.size());
        return m_data->dofNamePtrs.data();
    }

    IMPL_BUFFER_MANAGEMENT
};

#undef IMPL_XFORM_DATA_VIEW
#undef IMPL_BUFFER_MANAGEMENT

} // anonymous namespace


/**
 * @class PrimDataReaderImpl
 * @brief Implementation of the @c IPrimDataReader Carbonite interface.
 * @details Routes data access by engine:
 * - **PhysX**: Direct C++ TensorApi calls (no Python).
 * - **Newton**: Python callbacks fill C++-owned buffers.
 * - **Transforms**: IFabricHierarchy in C++ (engine-agnostic).
 */
class PrimDataReaderImpl : public IPrimDataReader
{
public:
    void initialize(long stageId, int deviceOrdinal) override
    {
        ++m_generation;
        m_stageId = stageId;
        m_deviceOrdinal = deviceOrdinal;
        m_simulationManager = carb::getCachedInterface<ISimulationManager>();
        m_tensorApi = carb::getCachedInterface<TensorApi>();

        PXR_NS::UsdStageCache& cache = PXR_NS::UsdUtilsStageCache::Get();
        m_usdStage = cache.Find(PXR_NS::UsdStageCache::Id::FromLongInt(stageId));

        if (m_usdStage)
        {
            omni::fabric::UsdStageId fabricStageId = { static_cast<uint64_t>(stageId) };
            omni::fabric::IStageReaderWriter* iStageReaderWriter =
                carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
            if (iStageReaderWriter)
            {
                omni::fabric::StageReaderWriterId stageInProgress = iStageReaderWriter->get(fabricStageId);
                m_usdrtStage = usdrt::UsdStage::Attach(fabricStageId, stageInProgress);
            }
        }

        // Release stale views and simulation view before recreating.
        // After a timeline stop/play cycle within the same stage, the
        // PhysX simulation view is invalidated; we must recreate it.
        for (auto& [id, data] : m_viewData)
            _releasePhysxHandles(data);
        m_views.clear();
        m_viewData.clear();

        if (m_simulationView)
        {
            m_simulationView->release(true);
            m_simulationView = nullptr;
        }

        if (m_tensorApi && stageId != 0 && isPhysxActive())
        {
            m_simulationView = m_tensorApi->createSimulationView(stageId);
            if (m_simulationView)
            {
                m_deviceOrdinal = m_simulationView->getDeviceOrdinal();
            }
        }
    }

    void shutdown() override
    {
        for (auto& [id, data] : m_viewData)
            _releasePhysxHandles(data);
        m_views.clear();
        m_viewData.clear();

        if (m_simulationView)
        {
            m_simulationView->release(true);
            m_simulationView = nullptr;
        }
        m_usdrtStage = nullptr;
        m_usdStage = nullptr;
        m_tensorApi = nullptr;
        m_simulationManager = nullptr;
    }

    IXformDataView* createXformView(const char* viewId, const char** paths, size_t numPaths, const char* engineType) override
    {
        auto& data = _setupViewData(viewId, paths, numPaths, engineType, ViewType::eXform);
        _setupTransformCallbacks(data);

        auto view = std::make_unique<XformDataView>(&data, m_simulationManager);
        auto* ptr = view.get();
        m_views[viewId] = std::move(view);
        return ptr;
    }

    IRigidBodyDataView* createRigidBodyView(const char* viewId,
                                            const char** paths,
                                            size_t numPaths,
                                            const char* engineType) override
    {
        auto& data = _setupViewData(viewId, paths, numPaths, engineType, ViewType::eRigidBody);
        _setupTransformCallbacks(data);

        if (data.engine == EngineType::ePhysX)
            _setupPhysxRigidBodyCallbacks(data);

        auto view = std::make_unique<RigidBodyDataView>(&data, m_simulationManager);
        auto* ptr = view.get();
        m_views[viewId] = std::move(view);
        return ptr;
    }

    IArticulationDataView* createArticulationView(const char* viewId,
                                                  const char** paths,
                                                  size_t numPaths,
                                                  const char* engineType) override
    {
        auto& data = _setupViewData(viewId, paths, numPaths, engineType, ViewType::eArticulation);
        _setupTransformCallbacks(data);

        if (data.engine == EngineType::ePhysX)
            _setupPhysxArticulationCallbacks(data);

        auto view = std::make_unique<ArticulationDataView>(&data, m_simulationManager);
        auto* ptr = view.get();
        m_views[viewId] = std::move(view);
        return ptr;
    }

    void removeView(const char* viewId) override
    {
        auto dataIt = m_viewData.find(viewId);
        if (dataIt != m_viewData.end())
        {
            _releasePhysxHandles(dataIt->second);
        }
        m_views.erase(viewId);
        m_viewData.erase(viewId);
    }

    void setArticulationDofMetadata(
        const char* viewId, const char** names, size_t numNames, const uint8_t* types, size_t numTypes) override
    {
        if (!viewId)
            return;
        auto dataIt = m_viewData.find(viewId);
        if (dataIt == m_viewData.end() || dataIt->second.type != ViewType::eArticulation)
            return;
        ViewData& data = dataIt->second;
        data.dofNames.clear();
        data.dofNamePtrs.clear();
        if (names && numNames > 0)
        {
            data.dofNames.reserve(numNames);
            for (size_t i = 0; i < numNames; ++i)
                data.dofNames.push_back(names[i] ? names[i] : std::string());
            data.dofNamePtrs.resize(data.dofNames.size());
            for (size_t i = 0; i < data.dofNames.size(); ++i)
                data.dofNamePtrs[i] = data.dofNames[i].c_str();
        }
        if (types && numTypes > 0)
        {
            auto& dofTypesField = data.getOrCreateField<uint8_t>("dof_types", numTypes, -1);
            dofTypesField.buffer->copyFrom(types, numTypes);
            int64_t step = m_simulationManager ? static_cast<int64_t>(m_simulationManager->getNumPhysicsSteps()) : 0;
            dofTypesField.lastStep = step;
        }
    }

    uint64_t getGeneration() const override
    {
        return m_generation;
    }

    long getStageId() const override
    {
        return m_stageId;
    }

    int getDeviceOrdinal() const override
    {
        return m_deviceOrdinal;
    }

private:
    static EngineType _parseEngine(const char* engineType)
    {
        if (engineType && std::string(engineType) == "newton")
            return EngineType::eNewton;
        return EngineType::ePhysX;
    }

    ViewData& _setupViewData(const char* viewId, const char** paths, size_t numPaths, const char* engineType, ViewType viewType)
    {
        auto& data = m_viewData[viewId];
        data.engine = _parseEngine(engineType);
        data.type = viewType;
        data.deviceOrdinal = m_deviceOrdinal;
        data.primPaths.clear();
        for (size_t i = 0; i < numPaths; ++i)
            data.primPaths.emplace_back(paths[i]);
        return data;
    }

    static void _releasePhysxHandles(ViewData& data)
    {
        if (data.physxArticulationView)
        {
            data.physxArticulationView->release();
            data.physxArticulationView = nullptr;
        }
        if (data.physxRigidBodyView)
        {
            data.physxRigidBodyView->release();
            data.physxRigidBodyView = nullptr;
        }
    }

    // ---- Fabric transform callbacks (engine-agnostic) ----

    void _setupTransformCallbacks(ViewData& data)
    {
        if (!m_usdStage || !m_usdrtStage)
            return;

        size_t numPrims = data.primPaths.size();
        auto& worldPositionsField = data.getOrCreateField<float>("world_positions", numPrims * 3, -1);
        auto& worldOrientationsField = data.getOrCreateField<float>("world_orientations", numPrims * 4, -1);

        // Capture by value: copies of smart pointers and path list are safe across calls.
        pxr::UsdStageRefPtr usdStage = m_usdStage;
        usdrt::UsdStageRefPtr usdrtStage = m_usdrtStage;
        std::vector<std::string> primPaths = data.primPaths;

        auto fillTransforms = [usdStage, usdrtStage, primPaths, &worldPositionsField, &worldOrientationsField]()
        {
            for (size_t i = 0; i < primPaths.size(); ++i)
            {
                auto xform = includes::pose::computeWorldXformNoCache(usdStage, usdrtStage, pxr::SdfPath(primPaths[i]));
                decomposeMatrix(
                    xform, worldPositionsField.buffer->data() + i * 3, worldOrientationsField.buffer->data() + i * 4);
            }
        };
        worldPositionsField.callback = fillTransforms;
        worldOrientationsField.callback = fillTransforms;
    }

    // ---- PhysX direct C++ callbacks ----

    /**
     * @brief Helper: create a C++ lambda that calls a PhysX TensorApi getter.
     * @details The lambda fills the FieldEntry buffer directly via a TensorDesc
     * pointing to the buffer's data pointer.
     */
    template <typename T>
    static std::function<void()> _makePhysxFieldCallbackT(FieldEntry<T>& field,
                                                          int device,
                                                          std::function<void(TensorDesc*)> tensorGetter)
    {
        return [&field, device, tensorGetter = std::move(tensorGetter)]()
        {
            TensorDesc desc;
            if constexpr (std::is_same_v<T, float>)
                fillTensorDesc(
                    desc, field.buffer->data(), static_cast<int>(field.count), TensorDataType::eFloat32, device);
            else if constexpr (std::is_same_v<T, uint8_t>)
                fillTensorDesc(desc, field.buffer->data(), static_cast<int>(field.count), TensorDataType::eUint8, -1);
            tensorGetter(&desc);
        };
    }

    void _setupPhysxArticulationCallbacks(ViewData& data)
    {
        if (!m_simulationView || data.primPaths.empty())
            return;

        data.physxArticulationView = m_simulationView->createArticulationView(data.primPaths);
        if (!data.physxArticulationView)
            return;

        IArticulationView* articulationView = data.physxArticulationView;
        int device = data.deviceOrdinal;
        uint32_t count = articulationView->getCount();
        uint32_t maxDofs = articulationView->getMaxDofs();
        uint32_t maxLinks = articulationView->getMaxLinks();

        auto& dofPositions = data.getOrCreateField<float>("dof_positions", count * maxDofs, device);
        dofPositions.callback = _makePhysxFieldCallbackT<float>(
            dofPositions, device, [articulationView](TensorDesc* d) { articulationView->getDofPositions(d); });

        auto& dofVelocities = data.getOrCreateField<float>("dof_velocities", count * maxDofs, device);
        dofVelocities.callback = _makePhysxFieldCallbackT<float>(
            dofVelocities, device, [articulationView](TensorDesc* d) { articulationView->getDofVelocities(d); });

        auto& dofEfforts = data.getOrCreateField<float>("dof_efforts", count * maxDofs, device);
        dofEfforts.callback = _makePhysxFieldCallbackT<float>(
            dofEfforts, device, [articulationView](TensorDesc* d) { articulationView->getDofProjectedJointForces(d); });

        auto& rootTransforms = data.getOrCreateField<float>("root_transforms", count * 7, device);
        rootTransforms.callback = _makePhysxFieldCallbackT<float>(
            rootTransforms, device, [articulationView](TensorDesc* d) { articulationView->getRootTransforms(d); });

        auto& rootVelocities = data.getOrCreateField<float>("root_velocities", count * 6, device);
        rootVelocities.callback = _makePhysxFieldCallbackT<float>(
            rootVelocities, device, [articulationView](TensorDesc* d) { articulationView->getRootVelocities(d); });

        auto& linkMasses = data.getOrCreateField<float>("link_masses", count * maxLinks, device);
        linkMasses.callback = _makePhysxFieldCallbackT<float>(
            linkMasses, device, [articulationView](TensorDesc* d) { articulationView->getMasses(d); });

        uint32_t jacobianRows = 0;
        uint32_t jacobianColumns = 0;
        if (articulationView->getJacobianShape(&jacobianRows, &jacobianColumns))
        {
            auto& jacobians = data.getOrCreateField<float>("jacobians", count * jacobianRows * jacobianColumns, device);
            jacobians.callback = _makePhysxFieldCallbackT<float>(
                jacobians, device, [articulationView](TensorDesc* d) { articulationView->getJacobians(d); });
        }

        uint32_t massMatrixRows = 0;
        uint32_t massMatrixColumns = 0;
        if (articulationView->getGeneralizedMassMatrixShape(&massMatrixRows, &massMatrixColumns))
        {
            auto& massMatrices =
                data.getOrCreateField<float>("mass_matrices", count * massMatrixRows * massMatrixColumns, device);
            massMatrices.callback = _makePhysxFieldCallbackT<float>(
                massMatrices, device,
                [articulationView](TensorDesc* d) { articulationView->getGeneralizedMassMatrices(d); });
        }

        // DOF metadata: names from USD, types via field callback (same pattern as other getters).
        data.dofNames.clear();
        data.dofNamePtrs.clear();
        if (m_usdStage)
        {
            for (uint32_t j = 0; j < maxDofs; ++j)
            {
                const char* path = articulationView->getUsdDofPath(0, j);
                if (path)
                {
                    pxr::UsdPrim prim = m_usdStage->GetPrimAtPath(pxr::SdfPath(path));
                    if (prim.IsValid())
                        data.dofNames.push_back(prim.GetName());
                    else
                        data.dofNames.push_back(std::string());
                }
                else
                {
                    data.dofNames.push_back(std::string());
                }
            }
            data.dofNamePtrs.resize(data.dofNames.size());
            for (size_t i = 0; i < data.dofNames.size(); ++i)
                data.dofNamePtrs[i] = data.dofNames[i].c_str();
        }
        if (maxDofs > 0)
        {
            auto& dofTypesField = data.getOrCreateField<uint8_t>("dof_types", maxDofs, -1);
            dofTypesField.callback = _makePhysxFieldCallbackT<uint8_t>(
                dofTypesField, -1, [articulationView](TensorDesc* d) { articulationView->getDofTypes(d); });
        }
    }

    void _setupPhysxRigidBodyCallbacks(ViewData& data)
    {
        if (!m_simulationView)
            return;

        data.physxRigidBodyView = m_simulationView->createRigidBodyView(data.primPaths);
        if (!data.physxRigidBodyView)
            return;

        IRigidBodyView* rigidBody = data.physxRigidBodyView;
        int device = data.deviceOrdinal;
        uint32_t count = rigidBody->getCount();

        // Transforms: float[N][7] -- split into separate velocity fields after fetch
        auto& transforms = data.getOrCreateField<float>("rigid_transforms_raw", count * 7, device);
        transforms.callback = _makePhysxFieldCallbackT<float>(
            transforms, device, [rigidBody](TensorDesc* d) { rigidBody->getTransforms(d); });

        // Velocities: float[N][6]
        auto& velocities = data.getOrCreateField<float>("rigid_velocities_raw", count * 6, device);
        velocities.callback = _makePhysxFieldCallbackT<float>(
            velocities, device, [rigidBody](TensorDesc* d) { rigidBody->getVelocities(d); });

        // Expose split linear/angular velocity fields by referencing the raw buffer
        auto& linearVelocity = data.getOrCreateField<float>("linear_velocities", count * 3, device);
        auto& angularVelocity = data.getOrCreateField<float>("angular_velocities", count * 3, device);
        linearVelocity.callback = [&velocities, &linearVelocity, &angularVelocity, count, device]()
        {
            if (velocities.callback)
                velocities.callback();
            float* source = velocities.buffer->data();
            float* destinationLinear = linearVelocity.buffer->data();
            float* destinationAngular = angularVelocity.buffer->data();
            if (device >= 0)
            {
                includes::ScopedDevice scopedDevice(device);
                CUDA_CHECK(cudaMemcpy2D(destinationLinear, 3 * sizeof(float), source, 6 * sizeof(float),
                                        3 * sizeof(float), count, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy2D(destinationAngular, 3 * sizeof(float), source + 3, 6 * sizeof(float),
                                        3 * sizeof(float), count, cudaMemcpyDeviceToDevice));
            }
            else
            {
                for (size_t i = 0; i < count; ++i)
                {
                    destinationLinear[i * 3 + 0] = source[i * 6 + 0];
                    destinationLinear[i * 3 + 1] = source[i * 6 + 1];
                    destinationLinear[i * 3 + 2] = source[i * 6 + 2];
                    destinationAngular[i * 3 + 0] = source[i * 6 + 3];
                    destinationAngular[i * 3 + 1] = source[i * 6 + 4];
                    destinationAngular[i * 3 + 2] = source[i * 6 + 5];
                }
            }
        };
        angularVelocity.callback = linearVelocity.callback;
    }

    long m_stageId = 0;
    int m_deviceOrdinal = -1;
    uint64_t m_generation = 0;
    ISimulationManager* m_simulationManager = nullptr;
    TensorApi* m_tensorApi = nullptr;
    ISimulationView* m_simulationView = nullptr;
    pxr::UsdStageRefPtr m_usdStage;
    usdrt::UsdStageRefPtr m_usdrtStage;

    std::unordered_map<std::string, ViewData> m_viewData;
    std::unordered_map<std::string, std::unique_ptr<IXformDataView>> m_views;
};

/**
 * @class PrimDataReaderManagerImpl
 * @brief Implementation of @c IPrimDataReaderManager.
 * @details Centralizes lifecycle management for the shared @c IPrimDataReader instance,
 * ensuring that sensor plugins and nodes do not call initialize() independently.
 * Subscribes to physics simulation events to trigger reinitialization on timeline
 * stop/resume.
 */
class PrimDataReaderManagerImpl : public IPrimDataReaderManager
{
public:
    PrimDataReaderManagerImpl()
    {
        _subscribeToPhysicsEvents();
    }

    ~PrimDataReaderManagerImpl()
    {
        m_physicsEventSubscription.reset();
    }

    bool ensureInitialized(long stageId, int deviceOrdinal) override
    {
        if (stageId == 0)
            return false;

        if (!m_reader)
            m_reader = carb::getCachedInterface<IPrimDataReader>();

        if (!m_reader)
            return false;

        const bool needsInit =
            !m_initialized || m_forceReinitialize || stageId != m_lastStageId || deviceOrdinal != m_lastDeviceOrdinal;
        if (needsInit)
        {
            m_reader->initialize(stageId, deviceOrdinal);
            m_lastStageId = stageId;
            m_lastDeviceOrdinal = deviceOrdinal;
            m_lastGeneration = m_reader->getGeneration();
            m_initialized = true;
            m_forceReinitialize = false;
        }
        else
        {
            m_lastGeneration = m_reader->getGeneration();
        }

        return true;
    }

    IPrimDataReader* getReader() override
    {
        if (!m_reader)
            m_reader = carb::getCachedInterface<IPrimDataReader>();
        return m_reader;
    }

    uint64_t getGeneration() const override
    {
        if (!m_reader)
            return m_lastGeneration;
        return m_reader->getGeneration();
    }

private:
    void _subscribeToPhysicsEvents()
    {
        if (m_physicsEventSubscription)
            return;

        auto* physicsStageUpdate = carb::getCachedInterface<omni::physics::IPhysicsStageUpdate>();
        if (!physicsStageUpdate)
            return;

        m_physicsEventSubscription = carb::events::createSubscriptionToPop(
            physicsStageUpdate->getSimulationEventStream().get(),
            [this](carb::events::IEvent* e)
            {
                if (e->type == omni::physics::SimulationEvent::eStopped ||
                    e->type == omni::physics::SimulationEvent::eResumed)
                {
                    m_forceReinitialize = true;
                }
            },
            0, "IsaacSim.Core.Experimental.Prims.ReaderManager.SimulationEvent");
    }

    IPrimDataReader* m_reader = nullptr;
    carb::events::ISubscriptionPtr m_physicsEventSubscription;
    long m_lastStageId = 0;
    int m_lastDeviceOrdinal = -1;
    uint64_t m_lastGeneration = 0;
    bool m_initialized = false;
    bool m_forceReinitialize = true;
};

/**
 * @class Extension
 * @brief Omniverse extension entry point for the prim data reader plugin.
 */
class Extension : public omni::ext::IExt
{
public:
    void onStartup(const char* extId) override
    {
    }

    void onShutdown() override
    {
    }
};

} // namespace prims
} // namespace experimental
} // namespace core
} // namespace isaacsim

CARB_EXPORT void carbOnPluginStartup()
{
}

CARB_EXPORT void carbOnPluginShutdown()
{
}

CARB_PLUGIN_IMPL(g_kPluginDesc,
                 isaacsim::core::experimental::prims::PrimDataReaderImpl,
                 isaacsim::core::experimental::prims::PrimDataReaderManagerImpl,
                 isaacsim::core::experimental::prims::Extension)
CARB_PLUGIN_IMPL_DEPS(omni::physics::IPhysics)

void fillInterface(isaacsim::core::experimental::prims::PrimDataReaderImpl& iface)
{
}

void fillInterface(isaacsim::core::experimental::prims::Extension& iface)
{
}

void fillInterface(isaacsim::core::experimental::prims::PrimDataReaderManagerImpl& iface)
{
}
