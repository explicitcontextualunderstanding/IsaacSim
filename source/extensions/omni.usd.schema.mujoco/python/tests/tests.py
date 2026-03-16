# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

import omni.kit.test
from pxr import Plug, Tf, Usd


class MujocoSchemaTests(omni.kit.test.AsyncTestCaseFailOnLogError):
    async def setUp(self):
        pass

    async def tearDown(self):
        pass

    async def test_schema_api_types(self):
        physics_plugin = Plug.Registry().GetPluginWithName("mjcPhysics")
        self.assertTrue(physics_plugin != None)

        reg = Usd.SchemaRegistry()

        typeName = "MjcSceneAPI"
        self.assertTrue(reg.IsAppliedAPISchema(typeName))
