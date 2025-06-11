# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import omni.ui as ui
from isaacsim.gui.components.element_wrappers import DropDown, Frame
from isaacsim.gui.components.widgets import DynamicComboBoxModel

LABEL_WIDTH = 90


class CustomDropDown(DropDown):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _create_ui_widget(self, label, tooltip):
        items = []
        combobox_model = DynamicComboBoxModel(items)
        containing_frame = Frame().frame
        with containing_frame:
            with ui.HStack():
                self._label = ui.Label(
                    label, name="dropdown_label", width=LABEL_WIDTH, alignment=ui.Alignment.LEFT_CENTER, tooltip=tooltip
                )
                self._combobox = ui.ComboBox(combobox_model)

        return containing_frame
