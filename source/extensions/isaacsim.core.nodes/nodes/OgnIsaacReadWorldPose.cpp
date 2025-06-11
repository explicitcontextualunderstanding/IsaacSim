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


#include <isaacsim/core/includes/Pose.h>
#include <omni/fabric/FabricUSD.h>
#include <omni/renderer/IDebugDraw.h>
#include <omni/usd/UsdContext.h>
#include <omni/usd/UsdContextIncludes.h>
#include <pxr/usd/usd/inherits.h>
#include <usdrt/gf/matrix.h>
#include <usdrt/gf/transform.h>
#include <usdrt/gf/vec.h>

#include <OgnIsaacReadWorldPoseDatabase.h>


namespace isaacsim
{
namespace core
{
namespace nodes
{

class OgnIsaacReadWorldPose
{
public:
    static bool compute(OgnIsaacReadWorldPoseDatabase& db)
    {
        auto& context = db.abi_context();

        const auto* const iToken = context.iToken;
        const auto* const iBundle = context.iBundle;

        auto& state = db.perInstanceState<OgnIsaacReadWorldPose>();
        const auto& inputPrim = db.inputs.prim();
        const auto& includeScale = db.inputs.includeScale();
        pxr::SdfPath primPath;
        if (!inputPrim.empty())
        {
            primPath = omni::fabric::toSdfPath(inputPrim[0]);
        }
        else
        {
            db.logError("Omnigraph Error: no input prim");
            return false;
        }
        if (state.m_stage == nullptr)
        {
            //  Find our stage
            const GraphContextObj& context = db.abi_context();
            state.m_stageId = context.iContext->getStageId(context);
            state.m_stage = pxr::UsdUtilsStageCache::Get().Find(pxr::UsdStageCache::Id::FromLongInt(state.m_stageId));
            omni::fabric::IStageReaderWriter* iStageReaderWriter =
                carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
            omni::fabric::StageReaderWriterId stageInProgress = iStageReaderWriter->get(state.m_stageId);
            state.m_usdrtStage = usdrt::UsdStage::Attach(state.m_stageId, stageInProgress);
            if (!state.m_stage)
            {
                db.logError("Could not find USD stage %ld", state.m_stageId);
                return false;
            }
        }
        usdrt::GfMatrix4d usdTransform =
            isaacsim::core::includes::pose::computeWorldXformNoCache(state.m_stage, state.m_usdrtStage, primPath);
        const double* sourceOrientationI =
            usdTransform.ExtractRotationMatrix().ExtractRotation().GetImaginary().GetArray();
        const double sourceOrientationR = usdTransform.ExtractRotationMatrix().ExtractRotation().GetReal();
        const double* sourceTranslation = usdTransform.ExtractTranslation().GetArray();
        auto& bundleContents = db.outputs.primsBundle();
        auto bundleHandle = bundleContents.abi_bundleHandle();

        AttributeDataHandle orientationAttr =
            iBundle->addAttribute(context, bundleHandle, iToken->getHandle("xformOp:orient"),
                                  Type(BaseDataType::eDouble, 4, 0, AttributeRole::eQuaternion));
        AttributeDataHandle positionAttr =
            iBundle->addAttribute(context, bundleHandle, iToken->getHandle("xformOp:translate"),
                                  Type(BaseDataType::eDouble, 3, 0, AttributeRole::eNone));

        double* orientationData = getDataW<double>(context, orientationAttr);
        double* positionData = getDataW<double>(context, positionAttr);

        if (includeScale)
        {
            auto transform = usdrt::GfTransform(usdTransform);
            AttributeDataHandle scaleAttr =
                iBundle->addAttribute(context, bundleHandle, iToken->getHandle("xformOp:scale"),
                                      Type(BaseDataType::eDouble, 3, 0, AttributeRole::eNone));
            double* scaleData = getDataW<double>(context, scaleAttr);
            const double* sourceScale = transform.GetScale().data();
            scaleData[0] = sourceScale[0];
            scaleData[1] = sourceScale[1];
            scaleData[2] = sourceScale[2];
        }

        orientationData[0] = sourceOrientationR;
        orientationData[1] = sourceOrientationI[0];
        orientationData[2] = sourceOrientationI[1];
        orientationData[3] = sourceOrientationI[2];

        positionData[0] = sourceTranslation[0];
        positionData[1] = sourceTranslation[1];
        positionData[2] = sourceTranslation[2];

        return true;
    }

private:
    pxr::UsdStageRefPtr m_stage = nullptr;
    long m_stageId;
    usdrt::UsdStageRefPtr m_usdrtStage{ nullptr };
};
REGISTER_OGN_NODE()
}
}
}
