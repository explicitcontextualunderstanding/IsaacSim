# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extension entry point for asset transformer rules."""

import omni.ext

from .manager import RuleRegistry


class Extension(omni.ext.IExt):
    """Extension that initializes the transformer rule registry."""

    def on_startup(self, ext_id: str):
        """Initialize the extension.

        Args:
            ext_id: Fully qualified extension identifier.
        """
        self._ext_id = ext_id
        # Initialize the singleton rule registry so other modules can register rules.
        self._registry = RuleRegistry()

    def on_shutdown(self):
        """Tear down the extension and release resources."""
