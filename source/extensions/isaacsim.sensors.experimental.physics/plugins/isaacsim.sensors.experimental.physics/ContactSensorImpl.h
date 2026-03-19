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

#include <isaacsim/sensors/experimental/physics/IContactSensor.h>

#include <memory>

namespace isaacsim
{
namespace sensors
{
namespace experimental
{
namespace physics
{

class ContactSensorImpl : public IContactSensor
{
public:
    ContactSensorImpl();
    ~ContactSensorImpl();

    void shutdown() override;
    int64_t createSensor(const char* primPath) override;
    void removeSensor(int64_t sensorId) override;
    ContactSensorReading getSensorReading(int64_t sensorId) override;
    void getRawContacts(int64_t sensorId, const ContactRawData** outData, int32_t* outCount) override;

private:
    struct ImplData;
    std::unique_ptr<ImplData> m_impl;

    void _initializeFromContext();
    void _initializeStage(long stageId);
    void _discoverSensorsFromStage();
    void _clearSensors();
    void _subscribeToPhysicsEvents();
    void _subscribeToPhysicsStepEvents();
    void _unsubscribeFromPhysicsStepEvents();
    void _pullContactData(float dt);
    void _stepSensors(float dt);
    void _processSensor(ImplData& impl, int64_t sensorId, float dt, double simTime);
};

} // namespace physics
} // namespace experimental
} // namespace sensors
} // namespace isaacsim
