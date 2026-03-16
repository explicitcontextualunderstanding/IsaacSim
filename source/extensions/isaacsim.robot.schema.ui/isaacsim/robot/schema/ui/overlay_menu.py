# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Overlay context menu helpers for joint selection."""

from typing import Any

__all__ = ["OverlayMenu"]

import carb
import omni.kit
import omni.kit.commands
import omni.usd
from omni.kit.context_menu import get_instance as get_context_menu_instance
from omni.kit.widget.context_menu import DefaultMenuDelegate


class OverlayMenu:
    """Context menu for selecting from overlapping joints.

    Provides a menu interface for selecting one of multiple joints
    that share the same screen position in the viewport.
    """

    @staticmethod
    def show_menu(connection: Any):
        """Display the overlay selection menu.

        Args:
            connection: Overlay connection group to display.

        Returns:
            None.

        Example:

        .. code-block:: python

            OverlayMenu.show_menu(connection)
        """
        context_menu = OverlayMenu._get_context_menu()
        if context_menu is None:
            carb.log_info("context_menu is disabled!")
            return

        stage = OverlayMenu._get_stage()
        if stage is None:
            carb.log_error("stage not available")
            return

        joint_paths, joint_names = OverlayMenu._build_joint_lists(connection)

        menu_context = {
            "stage": stage,
            "connection": connection,
            "joint_paths": joint_paths,
        }

        menu_entries = OverlayMenu._build_menu_entries(joint_names)
        context_menu.show_context_menu("overlay_menu", menu_context, menu_entries, delegate=DefaultMenuDelegate())

    @staticmethod
    def _get_context_menu() -> Any | None:
        """Get the context menu instance if available.

        Returns:
            The context menu instance, or None if unavailable.
        """
        if hasattr(omni.kit, "context_menu"):
            return get_context_menu_instance()
        return None

    @staticmethod
    def _get_stage() -> Any | None:
        """Get the current USD stage.

        Returns:
            The current USD stage, or None if unavailable.
        """
        usd_context = omni.usd.get_context()
        return usd_context.get_stage() if usd_context else None

    @staticmethod
    def _build_joint_lists(connection: Any) -> tuple[list[str], list[str]]:
        """Build lists of joint paths and names from a connection.

        Args:
            connection: The connection with overlay information.

        Returns:
            A tuple of (joint_paths, joint_names) lists.
        """
        joint_paths = [connection.joint_prim_path.pathString]
        joint_names = [connection.joint_prim_path.name]

        for path in connection.overlay_paths:
            joint_paths.append(path.pathString)
            joint_names.append(path.name)

        return joint_paths, joint_names

    @staticmethod
    def _build_menu_entries(joint_names: list[str]) -> list[dict[str, Any]]:
        """Build menu dictionary entries for each joint.

        Args:
            joint_names: List of joint display names.

        Returns:
            List of menu entry dictionaries.
        """
        menu_entries = []
        for index, name in enumerate(joint_names):
            menu_entries.append(
                {
                    "name": name,
                    "glyph": "menu_search.svg",
                    "show_fn": [],
                    "onclick_fn": lambda context, i=index: OverlayMenu._select_joint(context, i),
                }
            )
        return menu_entries

    @staticmethod
    def _select_joint(menu_context: dict[str, Any], index: int):
        """Select a joint by index from the menu.

        Args:
            menu_context: The menu context objects dictionary.
            index: The index of the joint to select.
        """
        paths = menu_context.get("joint_paths", [])
        if index < len(paths):
            omni.kit.commands.execute(
                "SelectPrims",
                old_selected_paths=[],
                new_selected_paths=[paths[index]],
                expand_in_stage=False,
            )
