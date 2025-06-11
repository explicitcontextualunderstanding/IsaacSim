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

#include <carb/Defines.h>
#include <carb/Types.h>

#include <usdrt/gf/matrix.h>

namespace isaacsim
{
namespace robot
{
namespace surface_gripper
{

/**
 * @struct SurfaceGripperInterface
 * @brief Interface for controlling surface gripper functionality.
 * @details
 * Provides function pointers for controlling surface grippers in the simulation.
 * Surface grippers can attach to and manipulate objects through surface contact
 * rather than traditional mechanical gripping.
 *
 * All functions operate on grippers identified by their USD prim path.
 */
struct SurfaceGripperInterface
{
    CARB_PLUGIN_INTERFACE("isaacsim::robot::surface_gripper::ISurfaceGripper", 0, 1);

    /** @brief Function pointer to get the current status of a surface gripper */
    const char*(CARB_ABI* GetGripperStatus)(const char* primPath);

    /** @brief Function pointer to open/release a surface gripper */
    bool(CARB_ABI* OpenGripper)(const char* primPath);

    /** @brief Function pointer to close/activate a surface gripper */
    bool(CARB_ABI* CloseGripper)(const char* primPath);

    /** @brief Function pointer to set a specific gripper action value */
    bool(CARB_ABI* SetGripperAction)(const char* primPath, const float action);

    /** @brief Function pointer to get the list of objects currently gripped */
    std::vector<std::string>(CARB_ABI* GetGrippedObjects)(const char* primPath);
};

} // namespace surface_gripper
} // namespace robot
} // namespace isaacsim
