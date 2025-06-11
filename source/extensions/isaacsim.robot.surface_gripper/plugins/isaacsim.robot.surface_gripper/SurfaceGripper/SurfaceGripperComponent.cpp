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
#include "isaacsim/robot/surface_gripper/SurfaceGripperComponent.h"

#include <extensions/PxJoint.h>
#include <omni/physics/tensors/BodyTypes.h>
#include <omni/physx/IPhysx.h>
#include <omni/physx/IPhysxSceneQuery.h>
#include <pxr/usd/sdf/layer.h>
#include <pxr/usd/usdPhysics/filteredPairsAPI.h>

#include <PxConstraint.h>
#include <PxRigidActor.h>


namespace
{
omni::physx::IPhysx* g_physx = nullptr;
}

namespace isaacsim
{
namespace robot
{
namespace surface_gripper
{

inline const pxr::SdfPath& intToPath(const uint64_t& path)
{
    static_assert(sizeof(pxr::SdfPath) == sizeof(uint64_t), "Change to make the same size as pxr::SdfPath");

    return reinterpret_cast<const pxr::SdfPath&>(path);
}


void SurfaceGripperComponent::initialize(const pxr::UsdPrim& prim, const pxr::UsdStageWeakPtr stage)
{
    g_physx = carb::getCachedInterface<omni::physx::IPhysx>();
    isaacsim::core::includes::ComponentBase<pxr::UsdPrim>::initialize(prim, stage);
    m_primPath = prim.GetPath();

    mDoStart = true;

    updateGripperProperties();
}

void SurfaceGripperComponent::onComponentChange()
{
    // Update surface gripper properties from the prim
    updateGripperProperties();
}

void SurfaceGripperComponent::onStart()
{

    onComponentChange();
    // Get attachment points (D6 joints)
    updateAttachmentPoints();
    if (m_status == GripperStatus::Closed)
    {
        updateClosedGripper();
    }
    else
    {
        updateOpenGripper();
    }
    m_isInitialized = true;
}

void SurfaceGripperComponent::onPhysicsStep(double dt)
{
    // Early return if component is not initialized
    if (!m_isInitialized)
        return;

    // Use SdfChangeBlock to batch USD changes for better performance
    // pxr::SdfChangeBlock changeBlock;


    // Update joint settling counters for inactive attachment points
    for (const auto& attachmentPath : m_inactiveAttachmentPoints)
    {
        if (m_jointSettlingCounters[attachmentPath] > 0)
        {
            m_jointSettlingCounters[attachmentPath]--;
        }
    }

    // Handle retry timeout for closing gripper
    if (m_status == GripperStatus::Closing && m_retryInterval > 0 && m_retryCloseActive &&
        !m_inactiveAttachmentPoints.empty())
    {
        m_retryElapsed += dt;
        if (m_retryElapsed > m_retryInterval)
        {
            // Timeout reached, stop trying to close
            m_retryCloseActive = false;
            m_retryElapsed = 0.0;
        }
    }

    // Update gripper state based on current status
    if (m_status == GripperStatus::Closed || m_status == GripperStatus::Closing)
    {
        if (m_retryCloseActive || !m_activeAttachmentPoints.empty())
        {
            updateClosedGripper();
        }
        else
        {
            updateOpenGripper();
        }
    }
    else
    {
        // For GripperStatus::Open or any other state
        updateOpenGripper();
    }
}

void SurfaceGripperComponent::preTick()
{
    // Nothing to do in preTick for now
}

void SurfaceGripperComponent::tick()
{
    if (!m_isInitialized || !m_isEnabled)
        return;

    // Update the visualization or any non-physics aspects
    // For example, we could update visual indicators of gripper state
}

void SurfaceGripperComponent::onStop()
{
    // First release all objects with physics changes
    releaseAllObjects();


    m_activeAttachmentPoints.clear();
    m_inactiveAttachmentPoints = m_attachmentPoints;
    m_grippedObjects.clear();
    m_isInitialized = false;
    mDoStart = true;
}

bool SurfaceGripperComponent::setGripperStatus(const std::string& status)
{
    auto gripperStatus = GripperStatusFromToken(pxr::TfToken(status));
    pxr::UsdAttribute gripperStatusAttr =
        m_stage->GetPrimAtPath(m_primPath)
            .GetAttribute(isaacsim::robot::schema::getAttributeName(isaacsim::robot::schema::Attributes::STATUS));
    if (gripperStatus != GripperStatus::Open && gripperStatus != GripperStatus::Closed &&
        gripperStatus != GripperStatus::Closing)
        return false;
    if (gripperStatus != m_status)
    {

        if (gripperStatus == GripperStatus::Open)
        {
            updateOpenGripper();
            m_status = GripperStatus::Open;
            m_retryCloseActive = false;
        }
        else if (gripperStatus == GripperStatus::Closed || gripperStatus == GripperStatus::Closing)
        {
            m_status = GripperStatus::Closing;
            m_retryCloseActive = true;
            updateClosedGripper();
            if (m_retryInterval > 0)
            {
                gripperStatusAttr.Set(pxr::TfToken("Closing"));
                // Start retry timer
                m_retryElapsed = 0.0;
            }
            else
            {
                m_retryCloseActive = false;
            }
        }
    }

    return true;
}

void SurfaceGripperComponent::updateGripperProperties()
{
    auto prim = m_stage->GetPrimAtPath(m_primPath);
    if (!prim)
    {
        return;
    }
    // Get forward axis
    pxr::UsdAttribute forwardAxisAttr =
        prim.GetAttribute(isaacsim::robot::schema::getAttributeName(isaacsim::robot::schema::Attributes::FORWARD_AXIS));
    if (forwardAxisAttr)
    {
        pxr::TfToken forwardAxis;
        forwardAxisAttr.Get(&forwardAxis);
        m_forwardAxis = forwardAxis.GetString();
    }

    // Get status
    pxr::UsdAttribute statusAttr =
        prim.GetAttribute(isaacsim::robot::schema::getAttributeName(isaacsim::robot::schema::Attributes::STATUS));
    if (statusAttr)
    {
        pxr::TfToken status;
        statusAttr.Get(&status);
        m_status = GripperStatusFromToken(status);
    }

    // Get retry interval
    pxr::UsdAttribute retryIntervalAttr =
        prim.GetAttribute(isaacsim::robot::schema::getAttributeName(isaacsim::robot::schema::Attributes::RETRY_INTERVAL));
    if (retryIntervalAttr)
    {
        retryIntervalAttr.Get(&m_retryInterval);
    }

    // Get force limits
    pxr::UsdAttribute shearForceLimitAttr = prim.GetAttribute(
        isaacsim::robot::schema::getAttributeName(isaacsim::robot::schema::Attributes::SHEAR_FORCE_LIMIT));
    if (shearForceLimitAttr)
    {
        shearForceLimitAttr.Get(&m_shearForceLimit);
    }

    pxr::UsdAttribute coaxialForceLimitAttr = prim.GetAttribute(
        isaacsim::robot::schema::getAttributeName(isaacsim::robot::schema::Attributes::COAXIAL_FORCE_LIMIT));
    if (coaxialForceLimitAttr)
    {
        coaxialForceLimitAttr.Get(&m_coaxialForceLimit);
    }

    pxr::UsdAttribute maxGripDistanceAttr = prim.GetAttribute(
        isaacsim::robot::schema::getAttributeName(isaacsim::robot::schema::Attributes::MAX_GRIP_DISTANCE));
    if (maxGripDistanceAttr)
    {
        maxGripDistanceAttr.Get(&m_maxGripDistance);
    }
}

void SurfaceGripperComponent::updateAttachmentPoints()
{
    auto prim = m_stage->GetPrimAtPath(m_primPath);
    m_attachmentPoints.clear();

    pxr::UsdRelationship attachmentPointsRel = prim.GetRelationship(
        isaacsim::robot::schema::relationNames.at(isaacsim::robot::schema::Relations::ATTACHMENT_POINTS));
    if (!attachmentPointsRel)
        return;

    std::vector<pxr::SdfPath> attachmentPaths;
    attachmentPointsRel.GetTargets(&attachmentPaths);
    // Preallocate buffers
    m_attachmentPoints.reserve(attachmentPaths.size());
    m_grippedObjectsBuffer.reserve(attachmentPaths.size());
    m_grippedObjects.reserve(attachmentPaths.size());
    m_activeAttachmentPoints.reserve(attachmentPaths.size());
    m_inactiveAttachmentPoints.reserve(attachmentPaths.size());
    m_jointSettlingCounters.reserve(attachmentPaths.size());
    for (const auto& path : attachmentPaths)
    {
        pxr::UsdPrim attachmentPrim = prim.GetPrimAtPath(path);
        if (attachmentPrim && attachmentPrim.IsA<pxr::UsdPhysicsJoint>())
        {
            if (!attachmentPrim.HasAPI(
                    isaacsim::robot::schema::className(isaacsim::robot::schema::Classes::ATTACHMENT_POINT_API)))
            {
                pxr::UsdEditContext context(m_stage, m_stage->GetRootLayer());
                isaacsim::robot::schema::ApplyAttachmentPointAPI(attachmentPrim);
            }
            pxr::UsdPhysicsJoint joint(attachmentPrim);

            bool excludeFromArticulation;
            joint.GetExcludeFromArticulationAttr().Get(&excludeFromArticulation);
            if (!excludeFromArticulation)
            {
                joint.GetExcludeFromArticulationAttr().Set(true);
            }
            physx::PxJoint* px_joint = (physx::PxJoint*)g_physx->getPhysXPtr((path), omni::physx::PhysXType::ePTJoint);
            if (!px_joint)
            {
                CARB_LOG_WARN("   Gripper %s has no joint at attachment point %s", m_primPath.GetText(), path.GetText());
                continue;
            }
            px_joint->setConstraintFlag(physx::PxConstraintFlag::eDISABLE_CONSTRAINT, m_status == GripperStatus::Open);
            m_attachmentPoints.insert(path.GetString());
            m_jointSettlingCounters[path.GetString()] = 0;
        }
    }
    m_activeAttachmentPoints.clear();
    m_inactiveAttachmentPoints = m_attachmentPoints;
}

void SurfaceGripperComponent::updateGrippedObjectsList()
{
    pxr::SdfPathVector objectPathsVec;
    // Early return if gripper is open - no need to track gripped objects
    if (m_activeAttachmentPoints.empty())
    {
        if (!m_grippedObjects.empty())
        {
            m_grippedObjects.clear();

            // Clear the GRIPPED_OBJECTS relationship if it exists
            auto grippedObjectsRel = m_stage->GetPrimAtPath(m_primPath)
                                         .GetRelationship(isaacsim::robot::schema::relationNames.at(
                                             isaacsim::robot::schema::Relations::GRIPPED_OBJECTS));
            if (grippedObjectsRel)
            {
                grippedObjectsRel.ClearTargets(true);
            }
            pxr::UsdPrim jointPrim = m_stage->GetPrimAtPath(pxr::SdfPath(m_attachmentPoints.begin()->c_str()));
            pxr::UsdPhysicsJoint joint(jointPrim);

            // Get body0 targets
            pxr::SdfPathVector targets0;
            joint.GetBody0Rel().GetTargets(&targets0);

            if (!targets0.empty())
            {
                // Update filtered pairs to disable collision with gripped objects
                pxr::UsdPhysicsFilteredPairsAPI filterPairsAPI =
                    pxr::UsdPhysicsFilteredPairsAPI(m_stage->GetPrimAtPath(targets0[0]));
                if (filterPairsAPI)
                {
                    filterPairsAPI.GetFilteredPairsRel().ClearTargets(true);
                }
            }
        }
    }
    else
    {
        // Create a new set for current gripped objects
        m_grippedObjectsBuffer.clear();

        // Iterate through active attachment points to find gripped objects
        for (const auto& attachmentPath : m_activeAttachmentPoints)
        {
            physx::PxJoint* px_joint = static_cast<physx::PxJoint*>(
                g_physx->getPhysXPtr(pxr::SdfPath(attachmentPath), omni::physx::PhysXType::ePTJoint));

            // Skip invalid or broken joints
            if (!px_joint || (px_joint->getConstraintFlags() &
                              (physx::PxConstraintFlag::eBROKEN | physx::PxConstraintFlag::eDISABLE_CONSTRAINT)))
            {
                continue;
            }

            // Get actors attached to the joint
            physx::PxRigidActor *px_actor0, *px_actor1;
            px_joint->getActors(px_actor0, px_actor1);

            // Add the second actor to gripped objects if it exists and has a name
            if (px_actor1 && px_actor1->getName())
            {
                m_grippedObjectsBuffer.insert(px_actor1->getName());
            }
        }

        // Check if the set of gripped objects has changed
        if (m_grippedObjects == m_grippedObjectsBuffer)
        {
            // No change in gripped objects, nothing to update
            return;
        }

        // Update the gripped objects
        m_grippedObjects = m_grippedObjectsBuffer;


        // Convert to path vectors for USD API

        objectPathsVec.reserve(m_grippedObjects.size());

        for (const auto& objectPath : m_grippedObjects)
        {
            objectPathsVec.push_back(pxr::SdfPath(objectPath));
        }
        // Get the first joint to find the body to apply filtered pairs to
        pxr::UsdPrim jointPrim = m_stage->GetPrimAtPath(pxr::SdfPath(m_attachmentPoints.begin()->c_str()));
        pxr::UsdPhysicsJoint joint(jointPrim);

        // Get body0 targets
        pxr::SdfPathVector targets0;
        joint.GetBody0Rel().GetTargets(&targets0);

        if (!targets0.empty())
        {
            // Update filtered pairs to disable collision with gripped objects
            pxr::UsdPhysicsFilteredPairsAPI filterPairsAPI =
                pxr::UsdPhysicsFilteredPairsAPI(m_stage->GetPrimAtPath(targets0[0]));
            if (!filterPairsAPI)
            {
                filterPairsAPI = pxr::UsdPhysicsFilteredPairsAPI::Apply(m_stage->GetPrimAtPath(targets0[0]));
            }
            filterPairsAPI.GetFilteredPairsRel().SetTargets(objectPathsVec);
        }

        // Update the gripper's GRIPPED_OBJECTS relationship
        m_stage->GetPrimAtPath(m_primPath)
            .GetRelationship(
                isaacsim::robot::schema::relationNames.at(isaacsim::robot::schema::Relations::GRIPPED_OBJECTS))
            .SetTargets(objectPathsVec);
    }
}

void SurfaceGripperComponent::updateClosedGripper()
{
    // If we have no attachment points, we can't do anything
    if (m_attachmentPoints.empty())
    {
        return;
    }

    // Get status attribute once for efficiency
    pxr::UsdAttribute gripperStatusAttr =
        m_stage->GetPrimAtPath(m_primPath)
            .GetAttribute(isaacsim::robot::schema::getAttributeName(isaacsim::robot::schema::Attributes::STATUS));
    if (!m_inactiveAttachmentPoints.empty() && (m_status == GripperStatus::Closing || m_retryCloseActive))
    {
        findObjectsToGrip();
    }

    checkForceLimits();

    updateGrippedObjectsList();

    GripperStatus newStatus = m_status;

    if (m_inactiveAttachmentPoints.empty())
    {
        newStatus = GripperStatus::Closed;
    }
    else if (m_retryCloseActive)
    {
        newStatus = GripperStatus::Closing;
    }
    else if (m_activeAttachmentPoints.empty())
    {
        newStatus = GripperStatus::Open;
    }

    // Update status if it changed
    if (newStatus != m_status)
    {
        m_status = newStatus;
        gripperStatusAttr.Set(GripperStatusToToken(newStatus));

        // Reset retry elapsed time if we've finished closing
        if (newStatus == GripperStatus::Closed)
        {
            m_retryElapsed = 0.0f;
        }
    }
}

void SurfaceGripperComponent::checkForceLimits()
{
    // Pre-allocate the vector to avoid resizing
    std::vector<std::string> ap_to_remove;
    ap_to_remove.reserve(m_activeAttachmentPoints.size());

    // Only check force limits if they are set
    const bool checkShearForce = m_shearForceLimit > 0.0f;
    const bool checkCoaxialForce = m_coaxialForceLimit > 0.0f;
    const bool checkForces = checkShearForce || checkCoaxialForce;

    for (const auto& attachmentPath : m_activeAttachmentPoints)
    {
        // Skip joints that are still settling
        if (m_jointSettlingCounters[attachmentPath] > 0)
        {
            m_jointSettlingCounters[attachmentPath]--;
            continue;
        }

        // Get the joint prim and PhysX joint
        pxr::UsdPrim jointPrim = m_stage->GetPrimAtPath(pxr::SdfPath(attachmentPath));
        physx::PxJoint* px_joint =
            (physx::PxJoint*)g_physx->getPhysXPtr(jointPrim.GetPath(), omni::physx::PhysXType::ePTJoint);

        if (!px_joint)
            continue;

        // Check if the joint is already disabled
        auto flags = px_joint->getConstraintFlags();
        if (flags & (physx::PxConstraintFlag::eDISABLE_CONSTRAINT | physx::PxConstraintFlag::eBROKEN))
        {
            ap_to_remove.push_back(attachmentPath);
            continue;
        }

        // Skip force checking if no limits are set
        if (!checkForces)
            continue;

        // Get force and actors
        physx::PxVec3 force, torque;
        physx::PxRigidActor *px_actor0, *px_actor1;
        px_joint->getActors(px_actor0, px_actor1);
        px_joint->getConstraint()->getForce(force, torque);


        // Get joint axis direction
        pxr::TfToken jointAxis;
        jointPrim
            .GetAttribute(isaacsim::robot::schema::getAttributeName(isaacsim::robot::schema::Attributes::FORWARD_AXIS))
            .Get(&jointAxis);

        // Create direction vector based on joint axis
        physx::PxVec3 direction(0.0f, 0.0f, 0.0f);
        char axis = jointAxis.GetText()[0];
        if (axis == 'X')
            direction.x = 1.0f;
        else if (axis == 'Y')
            direction.y = 1.0f;
        else
            direction.z = 1.0f;

        // Transform direction to world space
        physx::PxTransform localPose0 = px_joint->getLocalPose(physx::PxJointActorIndex::eACTOR0);
        physx::PxTransform actorPose = px_actor0->getGlobalPose();
        physx::PxTransform worldTransform = actorPose * localPose0;
        direction = worldTransform.q.rotate(direction);
        direction.normalize();

        // Calculate force components
        float coaxialForce = force.dot(direction);
        physx::PxVec3 shearForce;

        bool shouldRelease = false;

        // Only calculate shear force if needed
        if (checkShearForce)
        {
            shearForce = force - coaxialForce * direction;
            if (shearForce.magnitude() > m_shearForceLimit)
                shouldRelease = true;
        }

        // Check coaxial force if needed
        if (!shouldRelease && checkCoaxialForce && (coaxialForce) > m_coaxialForceLimit)
            shouldRelease = true;

        if (shouldRelease)
        {
            ap_to_remove.push_back(attachmentPath);
            px_joint->setConstraintFlag(physx::PxConstraintFlag::eDISABLE_CONSTRAINT, true);
            px_joint->setConstraintFlag(physx::PxConstraintFlag::eCOLLISION_ENABLED, true);
        }
    }

    // Process all joints to be removed
    for (const auto& ap : ap_to_remove)
    {
        m_activeAttachmentPoints.erase(ap);
        m_inactiveAttachmentPoints.insert(ap);
        m_jointSettlingCounters[ap] = m_settlingDelay;
    }
}

void SurfaceGripperComponent::findObjectsToGrip()
{
    // Get physics query interface
    auto physxQuery = carb::getCachedInterface<omni::physx::IPhysxSceneQuery>();
    if (!physxQuery)
        return;

    std::set<pxr::SdfPath> targetSet(m_grippedObjects.begin(), m_grippedObjects.end());

    // Iterate through each attachment point
    std::vector<std::string> ap_to_remove;
    physx::PxRigidActor *actor0, *actor1;
    for (const auto& attachmentPath : m_inactiveAttachmentPoints)
    {
        if (m_jointSettlingCounters[attachmentPath] > 0)
        {
            m_jointSettlingCounters[attachmentPath]--;
            continue;
        }
        pxr::UsdPrim jointPrim = m_stage->GetPrimAtPath(pxr::SdfPath(attachmentPath));

        pxr::UsdPhysicsJoint joint(jointPrim);

        // Get PhysX joint pointer
        physx::PxJoint* px_joint =
            (physx::PxJoint*)g_physx->getPhysXPtr(pxr::SdfPath(attachmentPath), omni::physx::PhysXType::ePTJoint);
        if (!px_joint)
            continue;

        // Get actors connected to joint
        px_joint->getActors(actor0, actor1);
        if (!actor0)
            continue;

        // Get joint's local poses
        physx::PxTransform localPose0 = px_joint->getLocalPose(physx::PxJointActorIndex::eACTOR0);

        // Get actor's global pose
        physx::PxTransform actorPose = actor0->getGlobalPose();

        // Calculate world transform
        physx::PxTransform worldTransform = actorPose * localPose0;

        // Get joint axis (assuming Z is default)
        pxr::GfVec3f direction(0.0, 0.0, 1.0);

        pxr::TfToken jointAxis;
        jointPrim
            .GetAttribute(isaacsim::robot::schema::getAttributeName(isaacsim::robot::schema::Attributes::FORWARD_AXIS))
            .Get(&jointAxis);
        switch (jointAxis.GetText()[0])
        {
        case 'X':
            direction = pxr::GfVec3f(1.0, 0.0, 0.0);
            break;
        case 'Y':
            direction = pxr::GfVec3f(0.0, 1.0, 0.0);
            break;
        default:
            direction = pxr::GfVec3f(0.0, 0.0, 1.0);
            break;
        }

        // Transform direction to world space
        auto rotated = worldTransform.q.rotate(physx::PxVec3(direction[0], direction[1], direction[2]));
        direction = pxr::GfVec3f(rotated.x, rotated.y, rotated.z);
        direction.Normalize();

        // Get world position
        pxr::GfVec3f worldPos(worldTransform.p.x, worldTransform.p.y, worldTransform.p.z);
        float clearanceOffset = 0.0f;
        pxr::UsdAttribute clearanceOffsetAttr = jointPrim.GetAttribute(
            isaacsim::robot::schema::getAttributeName(isaacsim::robot::schema::Attributes::CLEARANCE_OFFSET));
        if (clearanceOffsetAttr)
        {
            clearanceOffsetAttr.Get(&clearanceOffset);
        }
        else
        {
            jointPrim
                .CreateAttribute(
                    isaacsim::robot::schema::getAttributeName(isaacsim::robot::schema::Attributes::CLEARANCE_OFFSET),
                    pxr::SdfValueTypeNames->Float, false)
                .Set(clearanceOffset);
        }
        bool selfCollision = true;
        while (selfCollision)
        {
            pxr::GfVec3f rayStart = worldPos + direction * static_cast<float>(clearanceOffset);

            // Convert to carb types for raycast
            carb::Float3 _rayStart{ static_cast<float>(rayStart[0]), static_cast<float>(rayStart[1]),
                                    static_cast<float>(rayStart[2]) };

            // Scale distance before adding to start to get the correct starting point.
            carb::Float3 _rayDir{ static_cast<float>(direction[0]), static_cast<float>(direction[1]),
                                  static_cast<float>(direction[2]) };

            // Perform raycast
            omni::physx::RaycastHit result;
            float rayLength = static_cast<float>(m_maxGripDistance) - clearanceOffset;
            if (rayLength <= 0.0f)
            {
                rayLength = 0.001f;
            }
            bool hit = physxQuery->raycastClosest(_rayStart, _rayDir, rayLength, result, false);

            if (hit)
            {
                // Get the hit object's prim path
                pxr::SdfPath hitPath = pxr::SdfPath(intToPath(result.rigidBody).GetString());
                // Skip if we hit the gripper itself or its parent
                if (hitPath == pxr::SdfPath(actor0->getName()))
                {
                    selfCollision = true;
                    clearanceOffset += 0.001f;
                    continue;
                }
                selfCollision = false;
                if (clearanceOffset > 0.0f)
                {
                    float originalOffset;
                    clearanceOffsetAttr.Get(&originalOffset);
                    if (originalOffset != clearanceOffset)
                    {
                        CARB_LOG_WARN("   Gripper Attachment %s hit itself, adjust Clearance Offset at the joint to %f",
                                      attachmentPath.c_str(), clearanceOffset);
                    }
                    clearanceOffsetAttr.Set(clearanceOffset);
                }
                // Calculate the relative transform for the joint connection
                physx::PxRigidActor* hitActor =
                    (physx::PxRigidActor*)g_physx->getPhysXPtr(hitPath, omni::physx::PhysXType::ePTActor);
                if (!hitActor)
                    continue;
                // Calculate the relative transform for the joint connection
                physx::PxTransform hitWorldTransform = hitActor->getGlobalPose();

                // Calculate offset transform to place object at grip distance

                physx::PxVec3 offsetTranslation =
                    -physx::PxVec3(direction[0], direction[1], direction[2]) * (result.distance - clearanceOffset);
                physx::PxTransform offsetTransform(offsetTranslation, physx::PxQuat(physx::PxIdentity));

                // Apply offset to get the desired world transform
                physx::PxTransform adjustedWorldTransform = offsetTransform * worldTransform;

                // Calculate the local transform for body1 (relative to hit actor)
                physx::PxTransform hitLocalTransform = hitWorldTransform.transformInv(adjustedWorldTransform);

                // Update joint's Body1 actor and transform
                px_joint->setActors(actor0, hitActor);
                px_joint->setLocalPose(physx::PxJointActorIndex::eACTOR1, hitLocalTransform);
                auto constraintFlags = px_joint->getConstraintFlags();
                px_joint->setConstraintFlags(constraintFlags);
                // Enable the joint
                px_joint->setConstraintFlag(physx::PxConstraintFlag::eDISABLE_CONSTRAINT, false);
                // This should be the way to disable collision, but it doesn't work
                px_joint->setConstraintFlag(physx::PxConstraintFlag::eCOLLISION_ENABLED, false);


                ap_to_remove.push_back(attachmentPath);
            }
            else
            {
                break;
            }
        }
    }
    for (const auto& ap : ap_to_remove)
    {
        m_inactiveAttachmentPoints.erase(ap);
        m_activeAttachmentPoints.insert(ap);

        m_jointSettlingCounters[ap] = m_settlingDelay; // Initialize settling counter
    }
}

void SurfaceGripperComponent::updateOpenGripper()
{

    // Make sure we've released any gripped objects
    if (!m_grippedObjects.empty())
    {
        releaseAllObjects();
        updateGrippedObjectsList();
        m_activeAttachmentPoints.clear();
    }
    if (m_status != GripperStatus::Open)
    {
        m_status = GripperStatus::Open;
        m_stage->GetPrimAtPath(m_primPath)
            .GetAttribute(isaacsim::robot::schema::getAttributeName(isaacsim::robot::schema::Attributes::STATUS))
            .Set(pxr::TfToken("Open"));
    }
}

void SurfaceGripperComponent::releaseAllObjects()
{
    // Early return if no attachment points exist
    if (m_attachmentPoints.empty())
    {
        return;
    }

    // Release all objects by disabling constraints on all attachment points
    for (const auto& attachmentPath : m_attachmentPoints)
    {
        // Get the PhysX joint directly
        physx::PxJoint* px_joint =
            (physx::PxJoint*)g_physx->getPhysXPtr(pxr::SdfPath(attachmentPath), omni::physx::PhysXType::ePTJoint);

        if (px_joint)
        {
            // Disable the constraint and enable collision
            px_joint->setConstraintFlag(physx::PxConstraintFlag::eDISABLE_CONSTRAINT, true);
            px_joint->setConstraintFlag(physx::PxConstraintFlag::eCOLLISION_ENABLED, true);
        }
    }
    m_activeAttachmentPoints.clear();
    m_inactiveAttachmentPoints = m_attachmentPoints;


    // Reset settling counters for all attachment points
    for (const auto& ap : m_attachmentPoints)
    {
        m_jointSettlingCounters[ap] = m_settlingDelay;
    }
}

} // namespace surface_gripper
} // namespace robot
} // namespace isaacsim
