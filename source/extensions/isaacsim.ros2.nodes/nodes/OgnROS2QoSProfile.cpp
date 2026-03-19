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

// clang-format off
#include <pch/UsdPCH.h>
// clang-format on

#include <isaacsim/core/includes/BaseResetNode.h>
#include <nlohmann/json.hpp>

#include <OgnROS2QoSProfileDatabase.h>
#include <string>
#include <unordered_map>

namespace
{
constexpr int kNumPolicyInputs = 8;

// Per-node semaphore keyed by prim path, prevents cascading callbacks
// when applying a preset that modifies all individual QoS policy inputs.
std::unordered_map<std::string, int> s_presetSemaphores;
} // namespace

class OgnROS2QoSProfile : public isaacsim::core::includes::BaseResetNode
{
public:
    static void initInstance(NodeObj const& nodeObj, GraphInstanceID instanceId)
    {
        auto& state = OgnROS2QoSProfileDatabase::sPerInstanceState<OgnROS2QoSProfile>(nodeObj, instanceId);
        state.m_nodeObj = nodeObj;

        std::string primPath = nodeObj.iNode->getPrimPath(nodeObj);
        s_presetSemaphores[primPath] = 0;

        AttributeObj createProfileAttr = nodeObj.iNode->getAttribute(nodeObj, "inputs:createProfile");
        createProfileAttr.iAttribute->registerValueChangedCallback(createProfileAttr, onCreateProfileChanged, true);

        const char* policyAttrs[] = { "inputs:history",  "inputs:depth",    "inputs:reliability", "inputs:durability",
                                      "inputs:deadline", "inputs:lifespan", "inputs:liveliness",  "inputs:leaseDuration" };
        for (const auto& attrName : policyAttrs)
        {
            AttributeObj attr = nodeObj.iNode->getAttribute(nodeObj, attrName);
            attr.iAttribute->registerValueChangedCallback(attr, onQoSPolicyChanged, true);
        }

        applyPreset(nodeObj);
    }

    static bool compute(OgnROS2QoSProfileDatabase& db)
    {
        auto& state = db.perInstanceState<OgnROS2QoSProfile>();
        if (state.m_firstFrame)
        {
            nlohmann::json qosProfile;
            qosProfile["history"] = db.tokenToString(db.inputs.history());
            qosProfile["depth"] = db.inputs.depth();
            qosProfile["reliability"] = db.tokenToString(db.inputs.reliability());
            qosProfile["durability"] = db.tokenToString(db.inputs.durability());
            qosProfile["deadline"] = db.inputs.deadline();
            qosProfile["lifespan"] = db.inputs.lifespan();
            qosProfile["liveliness"] = db.tokenToString(db.inputs.liveliness());
            qosProfile["leaseDuration"] = db.inputs.leaseDuration();

            std::string jsonStr = qosProfile.dump();
            db.outputs.qosProfile() = jsonStr;
            state.m_firstFrame = false;
        }
        return true;
    }

    static void releaseInstance(NodeObj const& nodeObj, GraphInstanceID instanceId)
    {
        auto& state = OgnROS2QoSProfileDatabase::sPerInstanceState<OgnROS2QoSProfile>(nodeObj, instanceId);
        std::string primPath = nodeObj.iNode->getPrimPath(nodeObj);
        s_presetSemaphores.erase(primPath);
        state.reset();
    }

    virtual void reset()
    {
        m_firstFrame = true;
    }

private:
    static void setTokenAttr(const NodeObj& nodeObj, GraphContextObj& context, const char* attrName, const char* value)
    {
        AttributeObj attr = nodeObj.iNode->getAttribute(nodeObj, attrName);
        auto handle = attr.iAttribute->getAttributeDataHandle(attr, kAccordingToContextIndex);
        NameToken* ptr = getDataW<NameToken>(context, handle);
        if (ptr)
        {
            auto db = OgnROS2QoSProfileDatabase(nodeObj);
            *ptr = db.stringToToken(value);
        }
    }

    static void setUint64Attr(const NodeObj& nodeObj, GraphContextObj& context, const char* attrName, uint64_t value)
    {
        AttributeObj attr = nodeObj.iNode->getAttribute(nodeObj, attrName);
        auto handle = attr.iAttribute->getAttributeDataHandle(attr, kAccordingToContextIndex);
        uint64_t* ptr = getDataW<uint64_t>(context, handle);
        if (ptr)
        {
            *ptr = value;
        }
    }

    static void setDoubleAttr(const NodeObj& nodeObj, GraphContextObj& context, const char* attrName, double value)
    {
        AttributeObj attr = nodeObj.iNode->getAttribute(nodeObj, attrName);
        auto handle = attr.iAttribute->getAttributeDataHandle(attr, kAccordingToContextIndex);
        double* ptr = getDataW<double>(context, handle);
        if (ptr)
        {
            *ptr = value;
        }
    }

    static void setTimeDefaults(const NodeObj& nodeObj, GraphContextObj& context)
    {
        setDoubleAttr(nodeObj, context, "inputs:deadline", 0.0);
        setDoubleAttr(nodeObj, context, "inputs:lifespan", 0.0);
        setDoubleAttr(nodeObj, context, "inputs:leaseDuration", 0.0);
    }

    static void applyDefaultProfile(const NodeObj& nodeObj, GraphContextObj& context)
    {
        setTokenAttr(nodeObj, context, "inputs:history", "keepLast");
        setUint64Attr(nodeObj, context, "inputs:depth", 10);
        setTokenAttr(nodeObj, context, "inputs:reliability", "reliable");
        setTokenAttr(nodeObj, context, "inputs:durability", "volatile");
        setTokenAttr(nodeObj, context, "inputs:liveliness", "systemDefault");
        setTimeDefaults(nodeObj, context);
    }

    static std::string readTokenAttr(const NodeObj& nodeObj, GraphContextObj& context, const char* attrName)
    {
        AttributeObj attr = nodeObj.iNode->getAttribute(nodeObj, attrName);
        ConstAttributeDataHandle handle = attr.iAttribute->getConstAttributeDataHandle(attr, kAccordingToContextIndex);
        auto const token = getDataR<NameToken>(context, handle);
        if (token)
        {
            return token->getText();
        }
        return {};
    }

    static void applyPreset(const NodeObj& nodeObj)
    {
        GraphObj graphObj = nodeObj.iNode->getGraph(nodeObj);
        GraphContextObj context = graphObj.iGraph->getDefaultGraphContext(graphObj);

        std::string profileValue = readTokenAttr(nodeObj, context, "inputs:createProfile");

        if (profileValue == "Custom")
        {
            return;
        }

        std::string primPath = nodeObj.iNode->getPrimPath(nodeObj);
        s_presetSemaphores[primPath] = kNumPolicyInputs;

        if (profileValue == "Default for publishers/subscribers" || profileValue == "Services")
        {
            applyDefaultProfile(nodeObj, context);
        }
        else if (profileValue == "System Default")
        {
            setTokenAttr(nodeObj, context, "inputs:history", "systemDefault");
            setUint64Attr(nodeObj, context, "inputs:depth", 0);
            setTokenAttr(nodeObj, context, "inputs:reliability", "systemDefault");
            setTokenAttr(nodeObj, context, "inputs:durability", "systemDefault");
            setTokenAttr(nodeObj, context, "inputs:liveliness", "systemDefault");
            setTimeDefaults(nodeObj, context);
        }
        else if (profileValue == "Sensor Data")
        {
            setTokenAttr(nodeObj, context, "inputs:history", "keepLast");
            setUint64Attr(nodeObj, context, "inputs:depth", 5);
            setTokenAttr(nodeObj, context, "inputs:reliability", "bestEffort");
            setTokenAttr(nodeObj, context, "inputs:durability", "volatile");
            setTokenAttr(nodeObj, context, "inputs:liveliness", "systemDefault");
            setTimeDefaults(nodeObj, context);
        }
    }

    static void onCreateProfileChanged(AttributeObj const& attrObj, void const* userData)
    {
        NodeObj nodeObj = attrObj.iAttribute->getNode(attrObj);
        applyPreset(nodeObj);
    }

    static void onQoSPolicyChanged(AttributeObj const& attrObj, void const* userData)
    {
        NodeObj nodeObj = attrObj.iAttribute->getNode(attrObj);
        std::string primPath = nodeObj.iNode->getPrimPath(nodeObj);

        auto it = s_presetSemaphores.find(primPath);
        if (it != s_presetSemaphores.end() && it->second > 0)
        {
            it->second--;
            return;
        }

        GraphObj graphObj = nodeObj.iNode->getGraph(nodeObj);
        GraphContextObj context = graphObj.iGraph->getDefaultGraphContext(graphObj);
        setTokenAttr(nodeObj, context, "inputs:createProfile", "Custom");
    }

    bool m_firstFrame = true;
    NodeObj m_nodeObj;
};

REGISTER_OGN_NODE()
