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

from __future__ import annotations

import omni.ext

from .. import _prims_reader

_reader_interface = None


def get_prim_data_reader() -> object | None:
    """Get the IPrimDataReader Carbonite interface singleton.

    Returns:
        The acquired IPrimDataReader interface, or None if the extension has not started.
    """
    return _reader_interface


class Extension(omni.ext.IExt):
    """Extension lifecycle handler for the C++ prim data reader plugin."""

    def on_startup(self, ext_id: str) -> None:
        """Acquire the IPrimDataReader Carbonite interface on extension load."""
        global _reader_interface
        _reader_interface = _prims_reader.acquire_prim_data_reader_interface()

    def on_shutdown(self) -> None:
        """Release the IPrimDataReader Carbonite interface on extension unload."""
        global _reader_interface
        if _reader_interface is not None:
            _prims_reader.release_prim_data_reader_interface(_reader_interface)
            _reader_interface = None
