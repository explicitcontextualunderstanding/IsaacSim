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

#include "ContactSensorImpl.h"

#include <carb/events/EventsUtils.h>
#include <carb/logging/Log.h>
#include <carb/settings/ISettings.h>

#include <isaacSensorSchema/isaacContactSensor.h>
#include <isaacsim/core/includes/Pose.h>
#include <isaacsim/core/includes/UsdUtilities.h>
#include <isaacsim/core/simulation_manager/ISimulationManager.h>
#include <omni/fabric/FabricUSD.h>
#include <omni/physics/simulation/IPhysicsSimulation.h>
#include <omni/physics/simulation/IPhysicsStageUpdate.h>
#include <omni/physx/ContactEvent.h>
#include <omni/physx/IPhysxSimulation.h>
#include <omni/usd/UsdContext.h>
#include <physxSchema/physxContactReportAPI.h>
#include <physxSchema/physxRigidBodyAPI.h>
#include <pxr/usd/usdPhysics/rigidBodyAPI.h>

#if defined(_WIN32)
#    include <usdrt/scenegraph/usd/usd/stage.h>
#else
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wunused-variable"
#    pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#    include <usdrt/scenegraph/usd/usd/stage.h>
#    pragma GCC diagnostic pop
#endif

#include <algorithm>
#include <cmath>
#include <map>
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

using core::simulation_manager::ISimulationManager;

static std::string findParentRigidBody(pxr::UsdStageRefPtr stage, const pxr::SdfPath& sensorPath)
{
    pxr::UsdPrim prim = stage->GetPrimAtPath(sensorPath);
    if (!prim.IsValid())
        return {};

    prim = prim.GetParent();
    while (prim.IsValid() && prim.GetPath() != pxr::SdfPath::AbsoluteRootPath())
    {
        bool enabled = false;
        pxr::UsdAttribute attr = prim.GetAttribute(pxr::TfToken("physics:rigidBodyEnabled"));
        bool hasRigidBodyAPI = prim.HasAPI<pxr::UsdPhysicsRigidBodyAPI>();
        bool attrValid = attr.IsValid();
        if (attrValid)
            attr.Get(&enabled);

        if (enabled)
            return prim.GetPath().GetString();

        if (hasRigidBodyAPI && !attrValid)
            return prim.GetPath().GetString();

        prim = prim.GetParent();
    }
    return {};
}

inline uint64_t sdfPathToToken(const pxr::SdfPath& path)
{
    static_assert(sizeof(pxr::SdfPath) == sizeof(uint64_t), "SdfPath size mismatch");
    uint64_t ret;
    std::memcpy(&ret, &path, sizeof(pxr::SdfPath));
    return ret;
}

class ContactDataStore
{
public:
    std::vector<ContactRawData> rawContacts;
    std::map<uint64_t, std::vector<ContactRawData>> perBodyMap;

    void clear()
    {
        rawContacts.clear();
        perBodyMap.clear();
    }

    void removeContactPair(uint64_t body0, uint64_t body1)
    {
        uint64_t lower = std::min(body0, body1);
        uint64_t higher = std::max(body0, body1);

        rawContacts.erase(std::remove_if(rawContacts.begin(), rawContacts.end(),
                                         [lower, higher](const ContactRawData& e)
                                         {
                                             uint64_t entryLower = std::min(e.body0, e.body1);
                                             uint64_t entryHigher = std::max(e.body0, e.body1);
                                             return entryLower == lower && entryHigher == higher;
                                         }),
                          rawContacts.end());

        perBodyMap.erase(body0);
        perBodyMap.erase(body1);
    }

    const std::vector<ContactRawData>& getForBody(uint64_t token)
    {
        auto it = perBodyMap.find(token);
        if (it != perBodyMap.end() && !it->second.empty())
            return it->second;

        auto& vec = perBodyMap[token];
        vec.clear();
        for (const auto& e : rawContacts)
        {
            if (e.body0 == token || e.body1 == token)
                vec.push_back(e);
        }
        return vec;
    }
};

class SensorData
{
public:
    std::string sensorPrimPath;
    std::string parentRigidBodyPath;
    uint64_t parentToken = 0;

    float radius = -1.0f;
    float minThreshold = 0.0f;
    float maxThreshold = 100000.0f;
    bool enabled = true;
    bool previousEnabled = true;

    ContactSensorReading latestReading;
    std::vector<ContactRawData> latestRawContacts;

    void refreshConfig(pxr::UsdStageRefPtr stage)
    {
        pxr::UsdPrim prim = stage->GetPrimAtPath(pxr::SdfPath(sensorPrimPath));
        if (!prim.IsValid())
            return;

        pxr::IsaacSensorIsaacContactSensor typedPrim(prim);

        pxr::UsdAttribute enabledAttr = typedPrim.GetEnabledAttr();
        if (enabledAttr.IsValid())
        {
            bool val = true;
            enabledAttr.Get(&val);
            enabled = val;
        }
        else
        {
            enabled = true;
        }

        pxr::GfVec2f thresholdAttr(0.0f, 100000.0f);
        isaacsim::core::includes::safeGetAttribute(typedPrim.GetThresholdAttr(), thresholdAttr);
        const float* t = thresholdAttr.GetArray();
        minThreshold = t[0];
        maxThreshold = t[1];

        float r = -1.0f;
        isaacsim::core::includes::safeGetAttribute(typedPrim.GetRadiusAttr(), r);
        radius = r;
    }
};

} // namespace

struct ContactSensorImpl::ImplData
{
    long stageId = 0;
    int64_t nextSensorId = 0;
    float lastDt = 0.0f;
    int stepCount = 0;

    ISimulationManager* simManager = nullptr;
    omni::physics::IPhysicsSimulation* physicsSimulation = nullptr;
    omni::physics::SubscriptionId physicsStepSub = omni::physics::kInvalidSubscriptionId;
    carb::events::ISubscriptionPtr physicsEventSub;

    pxr::UsdStageRefPtr usdStage;
    usdrt::UsdStageRefPtr usdrtStage;
    std::unordered_map<int64_t, SensorData> sensors;
    ContactDataStore contactStore;
};

ContactSensorImpl::ContactSensorImpl() : m_impl(std::make_unique<ImplData>())
{
    m_impl->simManager = carb::getCachedInterface<ISimulationManager>();
    _subscribeToPhysicsStepEvents();
    _subscribeToPhysicsEvents();
}

ContactSensorImpl::~ContactSensorImpl()
{
    shutdown();
}

void ContactSensorImpl::shutdown()
{
    _unsubscribeFromPhysicsStepEvents();
    m_impl->physicsEventSub.reset();
    _clearSensors();
    m_impl->simManager = nullptr;
    m_impl->physicsSimulation = nullptr;
    m_impl->usdStage = nullptr;
    m_impl->usdrtStage = nullptr;
    m_impl->stageId = 0;
    m_impl->stepCount = 0;
    m_impl->lastDt = 0.0f;
    m_impl->contactStore.clear();
}

void ContactSensorImpl::_initializeFromContext()
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

    _initializeStage(stageId);
    _discoverSensorsFromStage();
}

void ContactSensorImpl::_initializeStage(long stageId)
{
    if (m_impl->stageId == stageId && m_impl->usdStage)
        return;

    if (m_impl->stageId != 0 && m_impl->stageId != stageId)
        _clearSensors();

    m_impl->stageId = stageId;
    m_impl->stepCount = 0;
    m_impl->lastDt = 0.0f;
    m_impl->contactStore.clear();

    m_impl->simManager = carb::getCachedInterface<ISimulationManager>();

    pxr::UsdStageCache& cache = pxr::UsdUtilsStageCache::Get();
    m_impl->usdStage = cache.Find(pxr::UsdStageCache::Id::FromLongInt(stageId));

    if (m_impl->usdStage)
    {
        omni::fabric::UsdStageId fabricStageId = { static_cast<uint64_t>(stageId) };
        omni::fabric::IStageReaderWriter* iStageReaderWriter =
            carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
        if (iStageReaderWriter)
        {
            omni::fabric::StageReaderWriterId stageInProgress = iStageReaderWriter->get(fabricStageId);
            m_impl->usdrtStage = usdrt::UsdStage::Attach(fabricStageId, stageInProgress);
        }
    }
}

int64_t ContactSensorImpl::createSensor(const char* primPath)
{
    if (!m_impl->usdStage)
        return -1;

    for (auto& [id, s] : m_impl->sensors)
    {
        if (s.sensorPrimPath == primPath)
            return id;
    }

    pxr::SdfPath sdfPath(primPath);
    pxr::UsdPrim prim = m_impl->usdStage->GetPrimAtPath(sdfPath);
    if (!prim.IsValid())
        return -1;

    if (prim.GetTypeName() != "IsaacContactSensor")
        return -1;

    std::string parentPath = findParentRigidBody(m_impl->usdStage, sdfPath);
    if (parentPath.empty())
        return -1;

    // SIDE EFFECT: createSensor() modifies the USD stage on the parent rigid body.
    //
    // PhysX's getFullContactReport() only returns data for bodies that have
    // PhysxContactReportAPI applied. Without it, numHeaders == 0 and no contacts
    // are reported regardless of actual collisions. This mirrors the behavior of
    // the legacy ContactSensor::setContactReportApi() in isaacsim.sensors.physics.
    //
    // Modifications made:
    //   1. Applies PhysxSchemaPhysxContactReportAPI if not present
    //   2. Creates ReportPairsRel if missing
    //   3. Sets contact report ThresholdAttr to 0.0 (report all contacts)
    //   4. Sets PhysxRigidBodyAPI sleepThreshold to 0.0 (prevent sleeping bodies
    //      from being excluded from contact reports)
    //
    // These changes persist on the USD stage for the lifetime of the session.
    pxr::SdfPath parentSdfPath(parentPath);
    pxr::UsdPrim parentPrim = m_impl->usdStage->GetPrimAtPath(parentSdfPath);
    if (parentPrim.IsValid())
    {
        pxr::PhysxSchemaPhysxContactReportAPI contactReportAPI =
            pxr::PhysxSchemaPhysxContactReportAPI::Get(m_impl->usdStage, parentSdfPath);
        if (!contactReportAPI)
            contactReportAPI = pxr::PhysxSchemaPhysxContactReportAPI::Apply(parentPrim);

        if (contactReportAPI)
        {
            if (!contactReportAPI.GetReportPairsRel())
                contactReportAPI.CreateReportPairsRel();
            contactReportAPI.GetThresholdAttr().Set(0.0f);
        }

        pxr::PhysxSchemaPhysxRigidBodyAPI rigidBodyAPI =
            pxr::PhysxSchemaPhysxRigidBodyAPI::Get(m_impl->usdStage, parentSdfPath);
        if (rigidBodyAPI)
            rigidBodyAPI.CreateSleepThresholdAttr(pxr::VtValue(0.0f));
    }

    int64_t sensorId = m_impl->nextSensorId++;
    SensorData& sensor = m_impl->sensors[sensorId];
    sensor.sensorPrimPath = primPath;
    sensor.parentRigidBodyPath = parentPath;
    sensor.parentToken = sdfPathToToken(pxr::SdfPath(parentPath));
    sensor.refreshConfig(m_impl->usdStage);

    return sensorId;
}

void ContactSensorImpl::removeSensor(int64_t sensorId)
{
    m_impl->sensors.erase(sensorId);
}

ContactSensorReading ContactSensorImpl::getSensorReading(int64_t sensorId)
{
    auto it = m_impl->sensors.find(sensorId);
    if (it == m_impl->sensors.end())
        return ContactSensorReading();

    return it->second.latestReading;
}

void ContactSensorImpl::getRawContacts(int64_t sensorId, const ContactRawData** outData, int32_t* outCount)
{
    if (!outData || !outCount)
        return;

    *outData = nullptr;
    *outCount = 0;

    auto it = m_impl->sensors.find(sensorId);
    if (it == m_impl->sensors.end())
        return;

    const auto& contacts = it->second.latestRawContacts;
    if (!contacts.empty())
    {
        *outData = contacts.data();
        *outCount = static_cast<int32_t>(contacts.size());
    }
}

void ContactSensorImpl::_discoverSensorsFromStage()
{
    if (!m_impl->usdStage)
        return;

    int found = 0;
    for (auto prim : m_impl->usdStage->Traverse())
    {
        if (prim.GetTypeName() == "IsaacContactSensor")
        {
            found++;
            (void)createSensor(prim.GetPath().GetString().c_str());
        }
    }
}

void ContactSensorImpl::_clearSensors()
{
    m_impl->sensors.clear();
    m_impl->contactStore.clear();
}

void ContactSensorImpl::_subscribeToPhysicsEvents()
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
                m_impl->usdrtStage = nullptr;
                m_impl->stageId = 0;
                m_impl->stepCount = 0;
                m_impl->lastDt = 0.0f;
            }
            else if (e->type == omni::physics::SimulationEvent::eResumed)
            {
                _initializeFromContext();
            }
        },
        0, "IsaacSim.Sensors.Experimental.Physics.ContactSensor.SimulationEvent");
}

void ContactSensorImpl::_subscribeToPhysicsStepEvents()
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

void ContactSensorImpl::_unsubscribeFromPhysicsStepEvents()
{
    if (m_impl->physicsSimulation && m_impl->physicsStepSub != omni::physics::kInvalidSubscriptionId)
    {
        m_impl->physicsSimulation->unsubscribePhysicsOnStepEvents(m_impl->physicsStepSub);
        m_impl->physicsStepSub = omni::physics::kInvalidSubscriptionId;
    }
}

void ContactSensorImpl::_pullContactData(float dt)
{
    m_impl->contactStore.rawContacts.clear();
    for (auto& it : m_impl->contactStore.perBodyMap)
        it.second.clear();

    auto* physxSim = carb::getCachedInterface<omni::physx::IPhysxSimulation>();
    if (!physxSim)
    {
        CARB_LOG_WARN("ContactSensorImpl: IPhysxSimulation not available");
        return;
    }

    const omni::physx::ContactEventHeader* headers = nullptr;
    const omni::physx::ContactData* data = nullptr;
    const omni::physx::FrictionAnchor* frictionData = nullptr;
    uint32_t numContactData = 0;
    uint32_t numFrictionData = 0;
    uint32_t numHeaders = physxSim->getFullContactReport(&headers, &data, numContactData, &frictionData, numFrictionData);

    float simTime = m_impl->simManager ? static_cast<float>(m_impl->simManager->getSimulationTime()) : 0.0f;

    uint32_t dataIndex = 0;
    for (uint32_t h = 0; h < numHeaders; h++)
    {
        const auto& header = headers[h];
        uint64_t body0 = header.actor0;
        uint64_t body1 = header.actor1;

        if (header.type == omni::physx::ContactEventType::Enum::eCONTACT_FOUND ||
            header.type == omni::physx::ContactEventType::Enum::eCONTACT_PERSIST)
        {
            m_impl->contactStore.removeContactPair(body0, body1);

            for (uint32_t i = 0; i < header.numContactData; i++)
            {
                const auto& contactData = data[dataIndex + i];
                ContactRawData entry;
                entry.body0 = body0;
                entry.body1 = body1;
                entry.positionX = contactData.position.x;
                entry.positionY = contactData.position.y;
                entry.positionZ = contactData.position.z;
                entry.normalX = contactData.normal.x;
                entry.normalY = contactData.normal.y;
                entry.normalZ = contactData.normal.z;
                entry.impulseX = contactData.impulse.x;
                entry.impulseY = contactData.impulse.y;
                entry.impulseZ = contactData.impulse.z;
                entry.time = simTime;
                entry.dt = dt;
                m_impl->contactStore.rawContacts.push_back(entry);
            }
            dataIndex += header.numContactData;
        }
        else if (header.type == omni::physx::ContactEventType::Enum::eCONTACT_LOST)
        {
            m_impl->contactStore.removeContactPair(body0, body1);
        }
    }
}

void ContactSensorImpl::_stepSensors(float dt)
{
    m_impl->lastDt = dt;
    m_impl->stepCount++;

    if (!m_impl->simManager || !m_impl->usdStage)
        return;

    _pullContactData(dt);

    if (m_impl->sensors.empty())
        return;

    const double simTime = m_impl->simManager->getSimulationTime();
    for (auto& [id, sensor] : m_impl->sensors)
    {
        (void)sensor;
        _processSensor(*m_impl, id, dt, simTime);
    }
}

void ContactSensorImpl::_processSensor(ImplData& impl, int64_t sensorId, float dt, double simTime)
{
    auto it = impl.sensors.find(sensorId);
    if (it == impl.sensors.end())
        return;
    SensorData& sensor = it->second;

    sensor.refreshConfig(impl.usdStage);

    if (sensor.previousEnabled != sensor.enabled)
    {
        if (!sensor.enabled)
        {
            sensor.latestReading = ContactSensorReading();
            sensor.latestRawContacts.clear();
        }
        sensor.previousEnabled = sensor.enabled;
    }

    if (!sensor.enabled)
        return;

    ContactSensorReading reading;
    reading.time = static_cast<float>(simTime);

    const auto& contacts = impl.contactStore.getForBody(sensor.parentToken);

    // Snapshot raw contacts for this sensor so they persist for Python access
    sensor.latestRawContacts.assign(contacts.begin(), contacts.end());

    if (contacts.empty())
    {
        reading.isValid = true;
        sensor.latestReading = reading;
        return;
    }

    usdrt::GfMatrix4d sensorXform = core::includes::pose::computeWorldXformNoCache(
        impl.usdStage, impl.usdrtStage, pxr::SdfPath(sensor.sensorPrimPath));
    usdrt::GfVec3d sensorPos = sensorXform.ExtractTranslation();

    double totalImpulseX = 0.0, totalImpulseY = 0.0, totalImpulseZ = 0.0;
    float contactDt = dt;

    for (const auto& c : contacts)
    {
        if (sensor.radius > 0.0f)
        {
            double dx = sensorPos[0] - c.positionX;
            double dy = sensorPos[1] - c.positionY;
            double dz = sensorPos[2] - c.positionZ;
            double distance = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (distance >= static_cast<double>(sensor.radius))
                continue;
        }

        double impulseX = static_cast<double>(c.impulseX);
        double impulseY = static_cast<double>(c.impulseY);
        double impulseZ = static_cast<double>(c.impulseZ);

        if (c.body1 == sensor.parentToken)
        {
            impulseX = -impulseX;
            impulseY = -impulseY;
            impulseZ = -impulseZ;
        }

        totalImpulseX += impulseX;
        totalImpulseY += impulseY;
        totalImpulseZ += impulseZ;

        if (c.dt > 0.0f)
            contactDt = c.dt;
    }

    double impulseMagnitude =
        std::sqrt(totalImpulseX * totalImpulseX + totalImpulseY * totalImpulseY + totalImpulseZ * totalImpulseZ);

    if (impulseMagnitude <= 0.0)
    {
        reading.isValid = true;
        sensor.latestReading = reading;
        return;
    }

    if (contactDt <= 0.0f)
        contactDt = dt > 0.0f ? dt : 1.0f / 60.0f;

    float forceValue = static_cast<float>(impulseMagnitude / static_cast<double>(contactDt));

    forceValue = std::min(forceValue, sensor.maxThreshold);
    if (forceValue < sensor.minThreshold)
    {
        reading.isValid = true;
        sensor.latestReading = reading;
        return;
    }

    reading.value = forceValue;
    reading.inContact = true;
    reading.isValid = true;
    sensor.latestReading = reading;
}

} // namespace physics
} // namespace experimental
} // namespace sensors
} // namespace isaacsim
