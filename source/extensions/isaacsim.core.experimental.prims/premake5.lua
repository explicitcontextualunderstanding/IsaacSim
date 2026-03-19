-- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
-- http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

-- Setup the basic extension variables
local ext = get_current_extension_info()
-- Set up the basic shared project information
project_ext(ext)

-- -------------------------------------
-- Build the C++ plugin that will be loaded by the extension
project_ext_plugin(ext, "isaacsim.core.experimental.prims.plugin")
defines { "ISAACSIM_CORE_EXPERIMENTAL_PRIMS_EXPORT" }

add_files("include", "include/isaacsim/core/experimental/prims")
add_files("source", "plugins/isaacsim.core.experimental.prims")
add_cuda_dependencies()
include_physx()
includedirs {
    "include",
    "plugins/isaacsim.core.experimental.prims",
    "%{root}/source/extensions/isaacsim.core.includes/include",
    "%{root}/source/extensions/isaacsim.core.simulation_manager/include",
    target_deps .. "/omni_physics/%{config}/include",
    "%{kit_sdk_bin_dir}/dev/fabric/include/",
    "%{target_deps}/usd/%{cfg.buildcfg}/include",
    "%{target_deps}/usd/%{cfg.buildcfg}/include/boost",
    "%{target_deps}/python/include/python3.12",
    extsbuild_dir .. "/usdrt.scenegraph/include",
    target_deps .. "/usd_ext_physics/%{cfg.buildcfg}/include",
}

libdirs {
    target_deps .. "/usd/%{cfg.buildcfg}/lib",
    target_deps .. "/usd_ext_physics/%{cfg.buildcfg}/lib",
    extsbuild_dir .. "/omni.usd.core/bin",
}
links { "carb", "omni.usd", "physxSchema" }

extra_usd_libs = { "usdGeom", "usdUtils", "usdPhysics", "ts" }
add_usd(extra_usd_libs)

filter { "configurations:debug" }
defines { "_DEBUG" }
filter { "configurations:release" }
defines { "NDEBUG" }
filter {}

-- -------------------------------------
-- Build Python bindings that will be loaded by the extension
project_ext_bindings {
    ext = ext,
    project_name = "isaacsim.core.experimental.prims.python",
    module = "_prims_reader",
    src = "bindings/isaacsim.core.experimental.prims",
    target_subdir = "isaacsim/core/experimental/prims",
}
dependson { "isaacsim.core.experimental.prims.plugin" }
links { "isaacsim.core.experimental.prims.plugin" }

includedirs {
    "include",
    "%{root}/source/extensions/isaacsim.core.includes/include",
    "%{root}/source/extensions/isaacsim.core.simulation_manager/include",
    "%{root}/_build/target-deps/usd/%{cfg.buildcfg}/include",
    "%{kit_sdk_bin_dir}/dev/fabric/include/",
}

libdirs {
    "%{root}/_build/target-deps/usd/%{cfg.buildcfg}/lib",
    extsbuild_dir .. "/omni.usd.core/bin",
}

extra_usd_libs = { "usdGeom", "usdUtils", "usdPhysics", "ts" }
add_usd(extra_usd_libs)

filter { "system:linux", "platforms:x86_64", "configurations:release" }
links { "tbb" }
filter { "system:linux", "platforms:x86_64", "configurations:debug" }
links { "tbb_debug" }
filter {}

filter { "configurations:debug" }
defines { "_DEBUG" }
filter { "configurations:release" }
defines { "NDEBUG" }
filter {}

-- -------------------------------------
-- Build C++ doctest tests
project_ext_tests(ext, "isaacsim.core.experimental.prims.tests")
add_files("source", "plugins/isaacsim.core.experimental.prims.tests")
add_cuda_dependencies()
includedirs {
    "include",
    "plugins/",
    "%{target_deps}/doctest/include",
    "%{root}/source/extensions/isaacsim.core.includes/include",
    "%{root}/source/extensions/isaacsim.core.simulation_manager/include",
}
libdirs {
    extsbuild_dir .. "/omni.kit.test/bin",
}

filter { "configurations:debug" }
defines { "_DEBUG" }
filter { "configurations:release" }
defines { "NDEBUG" }
filter {}

-- -------------------------------------
-- Link/copy folders and files to be packaged with the extension
repo_build.prebuild_link {
    { "data", ext.target_dir .. "/data" },
    { "docs", ext.target_dir .. "/docs" },
    { "python/impl", ext.target_dir .. "/isaacsim/core/experimental/prims/impl" },
    { "python/tests", ext.target_dir .. "/isaacsim/core/experimental/prims/tests" },
    { "include", ext.target_dir .. "/include" },
}

repo_build.prebuild_copy {
    { "python/*.py", ext.target_dir .. "/isaacsim/core/experimental/prims" },
}
