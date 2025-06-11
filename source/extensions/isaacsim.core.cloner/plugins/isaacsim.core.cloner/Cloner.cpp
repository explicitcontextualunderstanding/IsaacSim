// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <isaacsim/core/cloner/Cloner.h>
#include <omni/fabric/FabricUSD.h>
#include <omni/fabric/IFabric.h>
#include <omni/fabric/connectivity/Connectivity.h>
#include <usdrt/hierarchy/IFabricHierarchy.h>
#include <usdrt/population/IUtils.h>
#include <usdrt/scenegraph/usd/usd/stage.h>


/**
 * @brief Creates clones of a source prim at specified target paths in the USD stage
 * @details This function performs a fabric clone operation, creating copies of a source prim at multiple
 * target locations within the same stage.
 *
 * @param[in] stageId The unique identifier of the USD stage where cloning will occur
 * @param[in] source_prim_path The path to the source prim that will be cloned, this prim should be a valid USD prim
 * @param[in] prim_paths Vector of target paths where clones will be created, these prims will be created in Fabric
 * stage not in USD stage
 *
 * @return true if cloning was successful, false otherwise
 *
 * @warning The source prim must exist at the specified path
 */
bool isaacsim::core::cloner::fabricClone(long int stageId,
                                         const std::string& source_prim_path,
                                         const std::vector<std::string>& prim_paths)
{

    omni::fabric::IStageReaderWriter* iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
    if (!iStageReaderWriter)
    {
        CARB_LOG_ERROR("fabricClone - IStageReaderWriter not found");
        return false;
    }

    {
        CARB_PROFILE_ZONE(0, "fabricClone - fabric get or create");

        omni::fabric::FabricId fabricId{};
        omni::fabric::StageReaderWriterId stageReaderWriterId = iStageReaderWriter->get(stageId);
        if (!stageReaderWriterId.id)
        {
            // populate fabric
            auto iSimStageWithHistory = carb::getCachedInterface<omni::fabric::ISimStageWithHistory>();

            if (!iSimStageWithHistory)
            {
                CARB_LOG_ERROR("fabricClone - ISimStageWithHistory not found");
                return false;
            }

            iSimStageWithHistory->getOrCreate(stageId, 1, { 1, 30 }, omni::fabric::GpuComputeType::eCuda);
            iStageReaderWriter->create(stageId, 0);

            stageReaderWriterId = iStageReaderWriter->get(stageId);
            fabricId = iStageReaderWriter->getFabricId(stageReaderWriterId);
        }
        else
        {
            fabricId = iStageReaderWriter->getFabricId(stageReaderWriterId);
        }


        auto iFabricUsd = carb::getCachedInterface<omni::fabric::IFabricUsd>();
        if (!iFabricUsd)
        {
            CARB_LOG_ERROR("fabricClone - IFabricUsd not found");
            return false;
        }
        iFabricUsd->setEnableChangeNotifies(fabricId, false);

        auto populationUtils = omni::core::createType<usdrt::population::IUtils>();
        if (!populationUtils)
        {
            CARB_LOG_ERROR("fabricClone - IUtils not found");
            return false;
        }
        populationUtils->setEnableUsdNoticeHandling(stageId, fabricId, true);

        // Fill the stage in progress with USD values
        {
            CARB_PROFILE_ZONE(0, "fabricClone - fabric populate");
            populationUtils->populateFromUsd(
                stageReaderWriterId, stageId, omni::fabric::asInt(PXR_NS::SdfPath::AbsoluteRootPath()), nullptr, 0.0);
        }
    }

    {
        omni::fabric::StageReaderWriterId stageInProgress = iStageReaderWriter->get(stageId);
        omni::fabric::IStageReaderWriterLegacy* isrwLegacy =
            carb::getCachedInterface<omni::fabric::IStageReaderWriterLegacy>();
        if (!isrwLegacy)
        {
            CARB_LOG_ERROR("fabricClone - IStageReaderWriterLegacy not found");
            return false;
        }

        omni::fabric::Path world_envs_env_0(source_prim_path.c_str());

        // convert prim_paths to fabric paths, this is quite unfortunate and will slow down the cloning process
        std::vector<omni::fabric::Path> list_of_clones;
        {
            CARB_PROFILE_ZONE(0, "fabricClone - prim_paths to fabric paths");
            for (const auto& prim_path : prim_paths)
            {
                // compare if its not the same as the source prim path
                if (prim_path != source_prim_path)
                {
                    list_of_clones.push_back(omni::fabric::Path(prim_path.c_str()));
                }
            }
        }

        if (!list_of_clones.empty())
        {
            CARB_PROFILE_ZONE(0, "fabricClone - fabric batch clone");
            isrwLegacy->batchClone(iStageReaderWriter->getFabricId(stageInProgress), world_envs_env_0,
                                   { (const omni::fabric::PathC*)list_of_clones.data(), list_of_clones.size() });
        }
        else
        {
            return false;
        }
    }

    return true;
}
