// SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "isaacsim/robot/schema/robot_schema.h"
#include "isaacsim/robot/surface_gripper/SurfaceGripperManager.h"

namespace isaacsim
{
namespace robot
{
namespace surface_gripper
{

std::vector<std::string> SurfaceGripperManager::getComponentIsAVector() const
{
    return { isaacsim::robot::schema::className(isaacsim::robot::schema::Classes::SURFACE_GRIPPER).GetString() };
}

void SurfaceGripperManager::onStop()
{
    for (auto& component : m_components)
    {
        component.second->mDoStart = true;
        component.second->onStop();
    }
    if (m_gripperLayer)
    {
        m_gripperLayer->Clear();
    }

    // Reset timers when stopped
    this->m_timeSeconds = 0;
    this->m_timeNanoSeconds = 0;
}

void SurfaceGripperManager::onComponentAdd(const pxr::UsdPrim& prim)
{
    static const pxr::TfToken surfaceGripperToken(
        isaacsim::robot::schema::className(isaacsim::robot::schema::Classes::SURFACE_GRIPPER));

    if (pxr::TfToken(prim.GetTypeName()) == surfaceGripperToken)
    {
        std::unique_ptr<SurfaceGripperComponent> component = std::make_unique<SurfaceGripperComponent>();
        component->initialize(prim, m_stage);

        m_components[prim.GetPath().GetString()] = std::move(component);
    }
    if (m_components.size() > 0)
    {
        if (!m_gripperLayer)
        {
            m_gripperLayer = pxr::SdfLayer::CreateAnonymous("anon_gripper_ops_");
            // Create a new anonymous layer for gripper operations to not persist after simulation stops
            m_gripperLayer = pxr::SdfLayer::CreateAnonymous("anon_gripper_ops_");

            auto sessionLayer = m_stage->GetSessionLayer();
            if (sessionLayer)
            {
                sessionLayer->GetSubLayerPaths().push_back(m_gripperLayer->GetIdentifier());
            }

            if (m_gripperLayer)
            {
                m_gripperLayer->Clear();
            }
        }
    }
}

void SurfaceGripperManager::initialize(const pxr::UsdStageWeakPtr stage)
{
    isaacsim::core::includes::PrimManagerBase<SurfaceGripperComponent>::initialize(stage);
    m_stage = stage;
    m_gripperLayer = nullptr;
}

void SurfaceGripperManager::onComponentChange(const pxr::UsdPrim& prim)
{
    isaacsim::core::includes::PrimManagerBase<SurfaceGripperComponent>::onComponentChange(prim);

    // Update properties of this prim
    if (m_components.find(prim.GetPath().GetString()) != m_components.end())
    {
        m_components[prim.GetPath().GetString()]->onComponentChange();
    }
}

void SurfaceGripperManager::onPhysicsStep(const double& dt)
{
    pxr::SdfChangeBlock changeBlock;
    auto layer = m_stage->GetRootLayer();
    pxr::SdfLayerRefPtr refPtr(layer.operator->());
    pxr::UsdEditContext context(m_stage, m_gripperLayer ? m_gripperLayer : refPtr);
    for (auto& component : m_components)
    {
        // First run initialization for all components
        if (component.second->mDoStart == true)
        {
            component.second->mDoStart = false;
        }

        // Process physics update for all components
        component.second->onPhysicsStep(dt);
    }
}

void SurfaceGripperManager::onStart()
{
    if (!m_stage)
    {
        return;
    }
    for (auto& component : m_components)
    {
        if (component.second->mDoStart == true)
        {
            component.second->onStart();
            component.second->mDoStart = false;
        }
    }
}
void SurfaceGripperManager::tick(double dt)
{
    if (!m_stage)
    {
        return;
    }

    for (auto& component : m_components)
    {
        component.second->preTick();
        component.second->tick();
    }
}

bool SurfaceGripperManager::setGripperStatus(const std::string& primPath, const std::string& status)
{
    pxr::SdfChangeBlock changeBlock;
    auto layer = m_stage->GetRootLayer();
    pxr::SdfLayerRefPtr refPtr(layer.operator->());
    pxr::UsdEditContext context(m_stage, m_gripperLayer ? m_gripperLayer : refPtr);
    auto it = m_components.find(primPath);
    if (it != m_components.end())
    {
        return it->second->setGripperStatus(status);
    }
    return false;
}

std::string SurfaceGripperManager::getGripperStatus(const std::string& primPath)
{
    auto it = m_components.find(primPath);
    if (it != m_components.end())
    {
        return it->second->getGripperStatus();
    }
    else
    {
        CARB_LOG_ERROR("Gripper not found: %s", primPath.c_str());
    }
    return "";
}

std::vector<std::string> SurfaceGripperManager::getAllGrippers() const
{
    std::vector<std::string> result;
    for (const auto& component : m_components)
    {
        result.push_back(component.first);
    }
    return result;
}

SurfaceGripperComponent* SurfaceGripperManager::getGripper(const std::string& primPath)
{
    auto it = m_components.find(primPath);
    if (it != m_components.end())
    {
        return it->second.get();
    }
    return nullptr;
}

SurfaceGripperComponent* SurfaceGripperManager::getGripper(const pxr::UsdPrim& prim)
{
    return getGripper(prim.GetPath().GetString());
}

} // namespace surface_gripper
} // namespace robot
} // namespace isaacsim
