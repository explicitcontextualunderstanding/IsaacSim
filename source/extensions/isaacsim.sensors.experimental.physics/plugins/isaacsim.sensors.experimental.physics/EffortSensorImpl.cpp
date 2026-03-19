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

#include "EffortSensorImpl.h"

#include <carb/events/EventsUtils.h>
#include <carb/logging/Log.h>
#include <carb/settings/ISettings.h>

#include <isaacsim/core/experimental/prims/IPrimDataReader.h>
#include <isaacsim/core/experimental/prims/IPrimDataReaderManager.h>
#include <isaacsim/core/simulation_manager/ISimulationManager.h>
#include <omni/fabric/FabricUSD.h>
#include <omni/physics/simulation/IPhysicsSimulation.h>
#include <omni/physics/simulation/IPhysicsStageUpdate.h>
#include <omni/usd/UsdContext.h>
#include <pxr/usd/usdPhysics/articulationRootAPI.h>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

namespace isaacsim
{
namespace sensors
{
namespace experimental
{
namespace physics
{
namespace
{

using core::experimental::prims::IArticulationDataView;
using core::experimental::prims::IPrimDataReader;
using core::experimental::prims::IPrimDataReaderManager;
using core::simulation_manager::ISimulationManager;

static std::string findArticulationRoot(pxr::UsdStageRefPtr stage, const pxr::SdfPath& startPath)
{
    pxr::UsdPrim prim = stage->GetPrimAtPath(startPath);
    while (prim.IsValid() && prim.GetPath() != pxr::SdfPath::AbsoluteRootPath())
    {
        if (prim.HasAPI<pxr::UsdPhysicsArticulationRootAPI>())
            return prim.GetPath().GetString();
        prim = prim.GetParent();
    }
    return {};
}

struct EffortSensorData
{
    std::string jointPrimPath;
    std::string articulationRootPath;
    std::string viewId;
    std::string dofName;
    IArticulationDataView* articulationView = nullptr;
    int dofIndex = -1;

    EffortSensorReading latestReading;
    bool enabled = true;
};

} // namespace

struct EffortSensorImpl::ImplData
{
    long stageId = 0;
    std::string engineType = "physx";
    int64_t nextSensorId = 0;
    float lastDt = 0.0f;
    int stepCount = 0;
    uint64_t readerGeneration = 0;

    ISimulationManager* simManager = nullptr;
    IPrimDataReaderManager* readerManager = nullptr;
    IPrimDataReader* reader = nullptr;
    omni::physics::IPhysicsSimulation* physicsSimulation = nullptr;
    omni::physics::SubscriptionId physicsStepSub = omni::physics::kInvalidSubscriptionId;
    carb::events::ISubscriptionPtr physicsEventSub;

    pxr::UsdStageRefPtr usdStage;
    std::unordered_map<int64_t, EffortSensorData> sensors;
};

EffortSensorImpl::EffortSensorImpl() : m_impl(std::make_unique<ImplData>())
{
    m_impl->simManager = carb::getCachedInterface<ISimulationManager>();
    m_impl->readerManager = carb::getCachedInterface<IPrimDataReaderManager>();
    m_impl->reader = m_impl->readerManager ? m_impl->readerManager->getReader() : nullptr;
    _subscribeToPhysicsEvents();
}

EffortSensorImpl::~EffortSensorImpl()
{
    shutdown();
}

void EffortSensorImpl::shutdown()
{
    _unsubscribeFromPhysicsStepEvents();
    m_impl->physicsEventSub.reset();
    _clearSensors();
    m_impl->readerManager = nullptr;
    m_impl->reader = nullptr;
    m_impl->simManager = nullptr;
    m_impl->physicsSimulation = nullptr;
    m_impl->usdStage = nullptr;
    m_impl->stageId = 0;
    m_impl->stepCount = 0;
    m_impl->lastDt = 0.0f;
    m_impl->readerGeneration = 0;
}

void EffortSensorImpl::_initializeFromContext()
{
    auto* usdContext = omni::usd::UsdContext::getContext();
    if (!usdContext)
        return;

    pxr::UsdStageRefPtr stage = usdContext->getStage();
    if (!stage)
        return;

    pxr::UsdStageCache& cache = pxr::UsdUtilsStageCache::Get();
    const long stageId = cache.GetId(stage).ToLongInt();
    if (stageId == 0)
        return;

    auto* settings = carb::getCachedInterface<carb::settings::ISettings>();
    if (settings)
    {
        const char* engineSetting = settings->getStringBuffer("/exts/isaacsim.core.simulation_manager/default_engine");
        if (engineSetting && engineSetting[0] != '\0')
            m_impl->engineType = engineSetting;
    }

    _initializeStage(stageId);
}

void EffortSensorImpl::_initializeStage(long stageId)
{
    if (m_impl->stageId == stageId && m_impl->usdStage)
        return;

    if (m_impl->stageId != 0 && m_impl->stageId != stageId)
        _clearSensors();

    m_impl->stageId = stageId;
    m_impl->stepCount = 0;
    m_impl->lastDt = 0.0f;

    m_impl->simManager = carb::getCachedInterface<ISimulationManager>();
    m_impl->readerManager = carb::getCachedInterface<IPrimDataReaderManager>();
    if (m_impl->readerManager)
    {
        if (!m_impl->readerManager->ensureInitialized(stageId, -1))
            return;
        m_impl->reader = m_impl->readerManager->getReader();
    }
    else
    {
        m_impl->reader = nullptr;
    }

    pxr::UsdStageCache& cache = pxr::UsdUtilsStageCache::Get();
    m_impl->usdStage = cache.Find(pxr::UsdStageCache::Id::FromLongInt(stageId));

    _subscribeToPhysicsStepEvents();
}

int64_t EffortSensorImpl::createSensor(const char* jointPrimPath)
{
    if (!m_impl->usdStage || !m_impl->reader)
        return -1;

    for (auto& [id, s] : m_impl->sensors)
    {
        if (s.jointPrimPath == jointPrimPath)
            return id;
    }

    pxr::SdfPath sdfPath(jointPrimPath);
    pxr::UsdPrim prim = m_impl->usdStage->GetPrimAtPath(sdfPath);
    if (!prim.IsValid())
        return -1;

    std::string dofName = sdfPath.GetName();
    pxr::SdfPath parentPath = sdfPath.GetParentPath();

    std::string articulationRootPath = findArticulationRoot(m_impl->usdStage, parentPath);
    if (articulationRootPath.empty())
        return -1;

    int64_t sensorId = m_impl->nextSensorId++;
    EffortSensorData& sensor = m_impl->sensors[sensorId];
    sensor.jointPrimPath = jointPrimPath;
    sensor.articulationRootPath = articulationRootPath;
    sensor.dofName = dofName;
    sensor.viewId = "effort_art_" + std::to_string(sensorId);

    const char* pathStr = articulationRootPath.c_str();
    sensor.articulationView =
        m_impl->reader->createArticulationView(sensor.viewId.c_str(), &pathStr, 1, m_impl->engineType.c_str());
    if (!sensor.articulationView)
    {
        m_impl->sensors.erase(sensorId);
        return -1;
    }

    sensor.dofIndex = sensor.articulationView->getDofIndex(jointPrimPath);
    if (sensor.dofIndex < 0)
    {
        CARB_LOG_WARN("EffortSensor: could not resolve DOF index for joint '%s' in articulation '%s'", jointPrimPath,
                      articulationRootPath.c_str());
        m_impl->reader->removeView(sensor.viewId.c_str());
        m_impl->sensors.erase(sensorId);
        return -1;
    }

    m_impl->readerGeneration = m_impl->reader->getGeneration();
    return sensorId;
}

void EffortSensorImpl::removeSensor(int64_t sensorId)
{
    auto it = m_impl->sensors.find(sensorId);
    if (it == m_impl->sensors.end())
        return;
    if (m_impl->reader && !it->second.viewId.empty())
        m_impl->reader->removeView(it->second.viewId.c_str());
    m_impl->sensors.erase(it);
}

EffortSensorReading EffortSensorImpl::getSensorReading(int64_t sensorId)
{
    auto it = m_impl->sensors.find(sensorId);
    if (it == m_impl->sensors.end())
        return EffortSensorReading();

    if (m_impl->reader && m_impl->reader->getGeneration() != m_impl->readerGeneration)
        _recreateSensorViews();

    EffortSensorData& sensor = it->second;

    // On-the-fly processing if the reading isn't valid yet but we have data
    if (sensor.enabled && !sensor.latestReading.isValid && sensor.articulationView && m_impl->simManager &&
        m_impl->usdStage && m_impl->lastDt > 0.0f)
    {
        double simTime = m_impl->simManager->getSimulationTime();
        _processSensor(*m_impl, sensorId, m_impl->lastDt, simTime);
    }

    if (!sensor.enabled || !sensor.latestReading.isValid)
        return EffortSensorReading();

    return sensor.latestReading;
}

void EffortSensorImpl::_clearSensors()
{
    for (auto& [id, sensor] : m_impl->sensors)
    {
        (void)id;
        if (m_impl->reader && !sensor.viewId.empty())
            m_impl->reader->removeView(sensor.viewId.c_str());
    }
    m_impl->sensors.clear();
}

void EffortSensorImpl::_recreateSensorViews()
{
    if (!m_impl->reader)
        return;

    for (auto& [id, sensor] : m_impl->sensors)
    {
        sensor.articulationView = nullptr;
        sensor.dofIndex = -1;
        if (sensor.viewId.empty() || sensor.articulationRootPath.empty())
            continue;

        const char* pathStr = sensor.articulationRootPath.c_str();
        sensor.articulationView =
            m_impl->reader->createArticulationView(sensor.viewId.c_str(), &pathStr, 1, m_impl->engineType.c_str());
        if (sensor.articulationView)
            sensor.dofIndex = sensor.articulationView->getDofIndex(sensor.jointPrimPath.c_str());
    }
    m_impl->readerGeneration = m_impl->reader->getGeneration();
}

void EffortSensorImpl::_subscribeToPhysicsEvents()
{
    if (m_impl->physicsEventSub)
        return;

    auto* physicsStageUpdate = carb::getCachedInterface<omni::physics::IPhysicsStageUpdate>();
    if (!physicsStageUpdate)
        return;

    m_impl->physicsEventSub = carb::events::createSubscriptionToPop(
        physicsStageUpdate->getSimulationEventStream().get(),
        [this](carb::events::IEvent* e)
        {
            if (e->type == omni::physics::SimulationEvent::eStopped)
            {
                _clearSensors();
                m_impl->usdStage = nullptr;
                m_impl->stageId = 0;
                m_impl->stepCount = 0;
                m_impl->lastDt = 0.0f;
            }
            else if (e->type == omni::physics::SimulationEvent::eResumed)
            {
                _initializeFromContext();
            }
        },
        0, "IsaacSim.Sensors.Experimental.Physics.EffortSensor.SimulationEvent");
}

void EffortSensorImpl::_subscribeToPhysicsStepEvents()
{
    if (m_impl->physicsStepSub != omni::physics::kInvalidSubscriptionId)
        return;

    m_impl->physicsSimulation = carb::getCachedInterface<omni::physics::IPhysicsSimulation>();
    if (!m_impl->physicsSimulation)
        return;

    m_impl->physicsStepSub = m_impl->physicsSimulation->subscribePhysicsOnStepEvents(
        false, 1,
        [this](float elapsedTime, const omni::physics::PhysicsStepContext& /*context*/) { _stepSensors(elapsedTime); });
}

void EffortSensorImpl::_unsubscribeFromPhysicsStepEvents()
{
    if (m_impl->physicsSimulation && m_impl->physicsStepSub != omni::physics::kInvalidSubscriptionId)
    {
        m_impl->physicsSimulation->unsubscribePhysicsOnStepEvents(m_impl->physicsStepSub);
        m_impl->physicsStepSub = omni::physics::kInvalidSubscriptionId;
    }
}

void EffortSensorImpl::_stepSensors(float dt)
{
    m_impl->lastDt = dt;
    m_impl->stepCount++;

    if (!m_impl->simManager || !m_impl->usdStage || m_impl->sensors.empty())
        return;

    if (m_impl->reader && m_impl->reader->getGeneration() != m_impl->readerGeneration)
        _recreateSensorViews();

    const double simTime = m_impl->simManager->getSimulationTime();
    for (auto& [id, sensor] : m_impl->sensors)
    {
        (void)sensor;
        _processSensor(*m_impl, id, dt, simTime);
    }
}

void EffortSensorImpl::_processSensor(ImplData& impl, int64_t sensorId, float dt, double simTime)
{
    auto it = impl.sensors.find(sensorId);
    if (it == impl.sensors.end())
        return;
    EffortSensorData& sensor = it->second;

    if (!sensor.enabled || !sensor.articulationView || sensor.dofIndex < 0)
        return;

    int effortCount = 0;
    const float* efforts = sensor.articulationView->getDofEffortsHost(&effortCount);

    if (!efforts || sensor.dofIndex >= effortCount)
    {
        sensor.latestReading = EffortSensorReading();
        return;
    }

    sensor.latestReading.value = efforts[sensor.dofIndex];
    sensor.latestReading.time = static_cast<float>(simTime);
    sensor.latestReading.isValid = true;
}

} // namespace physics
} // namespace experimental
} // namespace sensors
} // namespace isaacsim
