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

#include "isaacsim/ucx/core/UcxListenerRegistry.h"

#include <carb/logging/Log.h>

#include <ucxx/api.h>

#include <stdexcept>

namespace isaacsim::ucx::core
{
std::unordered_map<uint16_t, std::shared_ptr<UCXListener>> UCXListenerRegistry::g_listeners;
std::mutex UCXListenerRegistry::g_registryMutex;

std::shared_ptr<UCXListener> UCXListenerRegistry::addListener(uint16_t port)
{
    std::lock_guard<std::mutex> lock(g_registryMutex);

    // When port is 0, always create a new listener on an ephemeral port
    if (port == 0)
    {
        CARB_LOG_INFO("UCXListenerRegistry::addListener: creating new listener on ephemeral port");
        try
        {
            auto context = ucxx::createContext({}, ucxx::Context::defaultFeatureFlags);
            if (!context)
            {
                throw std::runtime_error("UCXListenerRegistry: failed to create UCX context");
            }

            // Always create on ephemeral port (0)
            auto listener = std::make_shared<UCXListener>(context, 0);
            uint16_t actualPort = listener->getPort();
            g_listeners[actualPort] = listener;

            CARB_LOG_INFO("UCXListenerRegistry::addListener: listener created on port %u", actualPort);
            return listener;
        }
        catch (const std::exception& e)
        {
            throw std::runtime_error(std::string("UCXListenerRegistry: failed to create listener - ") + e.what());
        }
    }

    // When port is non-zero, return existing listener or create new one
    // Warn about privileged ports (requires root/administrator)
    if (port < 1024)
    {
        CARB_LOG_WARN("Using privileged port %u (ports 1-1023 require elevated privileges)", port);
    }

    if (g_listeners.find(port) == g_listeners.end())
    {
        CARB_LOG_INFO("UCXListenerRegistry::addListener: port %u", port);
        try
        {
            auto context = ucxx::createContext({}, ucxx::Context::defaultFeatureFlags);
            if (!context)
            {
                throw std::runtime_error("UCXListenerRegistry: failed to create UCX context");
            }

            auto listener = std::make_shared<UCXListener>(context, port);
            g_listeners[port] = listener;
            CARB_LOG_INFO("UCXListenerRegistry::addListener: listener created on port %u", port);
        }
        catch (const std::exception& e)
        {
            throw std::runtime_error(std::string("UCXListenerRegistry: failed to create listener - ") + e.what());
        }
    }
    return g_listeners[port];
}

void UCXListenerRegistry::removeListener(uint16_t port)
{
    std::shared_ptr<UCXListener> listener;
    {
        std::lock_guard<std::mutex> lock(g_registryMutex);
        auto it = g_listeners.find(port);
        if (it == g_listeners.end())
        {
            return;
        }

        listener = std::move(it->second);
        g_listeners.erase(it);
    }

    try
    {
        CARB_LOG_INFO("UCXListenerRegistry::removeListener: shutting down listener on port %u", port);
        listener->shutdown();
    }
    catch (const std::exception& e)
    {
        // Log error but continue with removal
        CARB_LOG_ERROR("Error shutting down listener on port %u: %s", port, e.what());
    }
}

bool UCXListenerRegistry::isListenerRegistered(uint16_t port)
{
    std::lock_guard<std::mutex> lock(g_registryMutex);
    return g_listeners.find(port) != g_listeners.end();
}

void UCXListenerRegistry::shutdown()
{
    std::lock_guard<std::mutex> lock(g_registryMutex);
    for (auto& [port, listener] : g_listeners)
    {
        try
        {
            if (listener)
            {
                listener->shutdown();
            }
        }
        catch (const std::exception& e)
        {
            // Log error but continue with shutdown
            CARB_LOG_ERROR("Error shutting down listener on port %u: %s", port, e.what());
        }
    }
    g_listeners.clear();
}

bool UCXListenerRegistry::tryRemoveListener(uint16_t port)
{
    std::shared_ptr<UCXListener> listener;
    {
        std::lock_guard<std::mutex> lock(g_registryMutex);
        auto it = g_listeners.find(port);
        if (it == g_listeners.end())
        {
            return false;
        }
        if (it->second.use_count() > 1)
        {
            // Other nodes still hold references
            return false;
        }
        CARB_LOG_INFO("UCXListenerRegistry::tryRemoveListener: removing listener on port %u", port);
        listener = std::move(it->second);
        g_listeners.erase(it);
    }

    // Shutdown outside the lock
    try
    {
        listener->shutdown();
    }
    catch (const std::exception& e)
    {
        CARB_LOG_ERROR("Error shutting down listener on port %u: %s", port, e.what());
    }
    return true;
}

} // namespace isaacsim::ucx::core