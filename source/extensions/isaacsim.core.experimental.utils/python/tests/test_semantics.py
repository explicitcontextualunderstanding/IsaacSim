# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import isaacsim.core.experimental.utils.semantics as semantics_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import omni.kit.test


class TestSemantics(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        """Method called to prepare the test fixture"""
        super().setUp()
        # create new stage
        await stage_utils.create_new_stage_async()

    async def tearDown(self):
        """Method called immediately after the test method has been called"""
        super().tearDown()

    # --------------------------------------------------------------------

    async def test_add_labels(self):
        prim = stage_utils.define_prim("/World/A", "Cube")
        self.assertDictEqual(semantics_utils.get_labels(prim), {})
        # test cases
        # - add labels
        semantics_utils.add_labels(prim, labels=["label_0", "label_1"])
        semantics_utils.add_labels(prim, labels=["label_1", "label_2", "label_3"], taxonomy="test")
        match = {"class": ["label_0", "label_1"], "test": ["label_1", "label_2", "label_3"]}
        self.assertDictEqual(semantics_utils.get_labels(prim), match)
        # - add a new label to existing ones
        semantics_utils.add_labels(prim, labels="label_4")
        semantics_utils.add_labels(prim, labels="label_4")  # add same label again
        match = {"class": ["label_0", "label_1", "label_4"], "test": ["label_1", "label_2", "label_3"]}
        self.assertDictEqual(semantics_utils.get_labels(prim), match)

    async def test_get_labels(self):
        stage_utils.define_prim("/World/A", "Cube")
        stage_utils.define_prim("/World/B", "Xform")
        self.assertDictEqual(semantics_utils.get_labels("/World/A"), {})
        self.assertDictEqual(semantics_utils.get_labels("/World/B"), {})
        # add labels
        semantics_utils.add_labels("/World/A", labels=["label_0", "label_1"])
        semantics_utils.add_labels("/World/B", labels=["label_1", "label_2", "label_3"], taxonomy="test")
        # test cases
        # - get labels from specific prims
        match = {"class": ["label_0", "label_1"]}
        self.assertDictEqual(semantics_utils.get_labels("/World/A"), match)
        match = {"test": ["label_1", "label_2", "label_3"]}
        self.assertDictEqual(semantics_utils.get_labels("/World/B"), match)
        # - get labels from prim without semantics applied
        self.assertDictEqual(semantics_utils.get_labels("/World"), {})
        # - get labels from prim without semantics applied (but with descendants)
        match = {"class": ["label_0", "label_1"], "test": ["label_1", "label_2", "label_3"]}
        self.assertDictEqual(semantics_utils.get_labels("/World", include_descendants=True), match)

    async def test_remove_labels(self):
        prim = stage_utils.define_prim("/World/A", "Cube")
        self.assertDictEqual(semantics_utils.get_labels(prim), {})
        # add labels
        semantics_utils.add_labels(prim, labels=["label_0", "label_1", "label_4"])
        semantics_utils.add_labels(prim, labels=["label_1", "label_2", "label_3"], taxonomy="test")
        match = {"class": ["label_0", "label_1", "label_4"], "test": ["label_1", "label_2", "label_3"]}
        self.assertDictEqual(semantics_utils.get_labels(prim), match)
        # test cases
        # - remove label from specific instance
        semantics_utils.remove_labels(prim, labels="label_2", taxonomy="test")
        match = {"class": ["label_0", "label_1", "label_4"], "test": ["label_1", "label_3"]}
        self.assertDictEqual(semantics_utils.get_labels(prim), match)
        # - remove label from all instances
        semantics_utils.remove_labels(prim, labels="label_1")
        match = {"class": ["label_0", "label_4"], "test": ["label_3"]}
        self.assertDictEqual(semantics_utils.get_labels(prim), match)
        # - remove label from descendants
        semantics_utils.remove_labels("/World", labels="label_4", include_descendants=True)
        match = {"class": ["label_0"], "test": ["label_3"]}
        self.assertDictEqual(semantics_utils.get_labels(prim), match)
        # remove unexisting label
        semantics_utils.remove_labels(prim, labels="label_5")
        match = {"class": ["label_0"], "test": ["label_3"]}
        self.assertDictEqual(semantics_utils.get_labels(prim), match)

    async def test_remove_all_labels(self):
        prim = stage_utils.define_prim("/World/A", "Cube")
        self.assertDictEqual(semantics_utils.get_labels(prim), {})
        # add labels
        semantics_utils.add_labels(prim, labels=["label_0", "label_1", "label_4"])
        semantics_utils.add_labels(prim, labels=["label_1", "label_2", "label_3"], taxonomy="test")
        match = {"class": ["label_0", "label_1", "label_4"], "test": ["label_1", "label_2", "label_3"]}
        self.assertDictEqual(semantics_utils.get_labels(prim), match)
        # test cases
        # - remove all labels (keep taxonomies)
        semantics_utils.remove_all_labels(prim)
        match = {"class": [], "test": []}
        self.assertDictEqual(semantics_utils.get_labels(prim), match)
        # - remove all labels (remove taxonomies)
        # -- current prim
        semantics_utils.remove_all_labels("/World", remove_taxonomies=True)
        self.assertDictEqual(semantics_utils.get_labels(prim), match)
        # -- current prim and descendants
        semantics_utils.remove_all_labels("/World", remove_taxonomies=True, include_descendants=True)
        self.assertDictEqual(semantics_utils.get_labels(prim), {})
