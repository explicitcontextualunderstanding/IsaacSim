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
"""State management for robot masking (deactivation of joints and links)."""

from __future__ import annotations

import traceback
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import carb
import usd.schema.isaac.robot_schema as robot_schema
from pxr import UsdPhysics

from .utils import PathMap

if TYPE_CHECKING:
    from pxr import Usd


def is_maskable_type(prim: Usd.Prim) -> bool:
    """Check if a prim is maskable (can be deactivated).

    Args:
        prim: The prim to check.

    Returns:
        True if the prim is a link or joint that can be masked.
    """
    schemas = prim.GetAppliedSchemas()
    return robot_schema.Classes.JOINT_API.value in schemas or robot_schema.Classes.LINK_API.value in schemas


def get_selected_maskable_paths() -> list[str]:
    """Return the original stage prim paths of currently selected maskable prims.

    Returns:
        List of path strings for selected prims that are joints or links.
    """
    import omni.usd

    usd_context = omni.usd.get_context()
    stage = usd_context.get_stage()
    if not stage:
        return []
    selection = usd_context.get_selection()
    if not selection:
        return []
    return [
        path_str for path_str in selection.get_selected_prim_paths() if is_maskable_type(stage.GetPrimAtPath(path_str))
    ]


def is_joint_type(prim: Usd.Prim) -> bool:
    """Check if a prim is a joint type.

    Args:
        prim: The prim to check.

    Returns:
        True if the prim is a physics joint.
    """
    schemas = prim.GetAppliedSchemas()
    return robot_schema.Classes.JOINT_API.value in schemas


def is_link_type(prim: Usd.Prim) -> bool:
    """Check if a prim is a robot link (not a joint).

    Args:
        prim: The prim to check.

    Returns:
        True if the prim has the link API applied.
    """
    schemas = prim.GetAppliedSchemas()
    return robot_schema.Classes.LINK_API.value in schemas


def is_anchorable_link(prim: Usd.Prim) -> bool:
    """Check if a prim can be anchored to the world (link with RigidBodyAPI).

    Args:
        prim: The prim to check.

    Returns:
        True if the prim has both the link API and RigidBodyAPI applied.
    """
    return is_link_type(prim) and prim.HasAPI(UsdPhysics.RigidBodyAPI)


def get_selected_link_paths() -> list[str]:
    """Return the original stage prim paths of currently selected link prims.

    Returns:
        List of path strings for selected prims that are links.
    """
    import omni.usd

    usd_context = omni.usd.get_context()
    stage = usd_context.get_stage()
    if not stage:
        return []
    selection = usd_context.get_selection()
    if not selection:
        return []
    return [path_str for path_str in selection.get_selected_prim_paths() if is_link_type(stage.GetPrimAtPath(path_str))]


def get_selected_anchorable_link_paths() -> list[str]:
    """Return the original stage prim paths of currently selected anchorable link prims.

    Returns:
        List of path strings for selected prims that are links with RigidBodyAPI.
    """
    import omni.usd

    usd_context = omni.usd.get_context()
    stage = usd_context.get_stage()
    if not stage:
        return []
    selection = usd_context.get_selection()
    if not selection:
        return []
    return [
        path_str
        for path_str in selection.get_selected_prim_paths()
        if is_anchorable_link(stage.GetPrimAtPath(path_str))
    ]


class MaskingState:
    """Singleton tracking which prims are deactivated (masked/bypassed).

    Tracks two distinct states:

    * **masked** -- the element is disabled for simulation.
    * **bypassed** -- the element is disabled AND the kinematic chain is
      reconnected around it.

    Notifies subscribers when any state changes.
    """

    _instance: MaskingState | None = None

    @classmethod
    def get_instance(cls) -> MaskingState:
        """Return the singleton `MaskingState` instance, creating it if needed.

        Returns:
            The shared ``MaskingState`` instance.
        """
        if cls._instance is None:
            cls._instance = MaskingState()
        return cls._instance

    def __init__(self) -> None:
        self._deactivated_paths: set[str] = set()
        self._bypassed_paths: set[str] = set()
        self._anchored_paths: set[str] = set()
        # Maps bypassed-link path → (backward_joint_path, prev_state).
        # prev_state is "enabled", "masked", or "bypassed".
        # Used to lock the joint while the forward link is bypassed and to
        # restore its UI representation correctly on unbypass.
        self._bypass_deactivated_joints: dict[str, tuple[str, str]] = {}
        self._path_map: PathMap | None = None
        self._operations: Any = None
        self._on_changed_callbacks: list[Callable[[], None]] = []

    # -- properties -------------------------------------------------------

    @property
    def path_map(self) -> PathMap | None:
        """Path mapping between hierarchy and original stage; used for lookups."""
        return self._path_map

    @path_map.setter
    def path_map(self, value: PathMap | None) -> None:
        """Set the path mapping between hierarchy and original stage.

        Args:
            value: New ``PathMap`` instance, or ``None`` to clear.
        """
        self._path_map = value

    @property
    def operations(self) -> Any:
        """Masking operations backend (e.g. `MaskingOperations`); None if not set."""
        return self._operations

    @operations.setter
    def operations(self, value: Any) -> None:
        """Set the masking operations backend.

        Args:
            value: ``MaskingOperations`` instance, or ``None`` to clear.
        """
        self._operations = value

    # -- queries ----------------------------------------------------------

    def is_deactivated(self, original_path: str) -> bool:
        """Return True if the prim is masked or bypassed.

        Args:
            original_path: Original stage prim path string.

        Returns:
            ``True`` if the path is in the deactivated set.
        """
        return original_path in self._deactivated_paths

    def is_bypassed(self, original_path: str) -> bool:
        """Return True only if the prim is in the bypassed state.

        Args:
            original_path: Original stage prim path string.

        Returns:
            ``True`` if the path is in the bypassed set.
        """
        return original_path in self._bypassed_paths

    def is_bypass_controlled(self, original_path: str) -> bool:
        """Return True if ``original_path`` is a backward joint locked by an active link bypass.

        While locked, the joint's bypass and deactivation state must not be
        changed through the UI; the owning link bypass owns its state.

        Args:
            original_path: Original stage prim path string.

        Returns:
            ``True`` if the joint is currently locked by a link bypass.
        """
        return any(joint_path == original_path for joint_path, _ in self._bypass_deactivated_joints.values())

    def get_masking_layer_id(self) -> str | None:
        """Return the masking sublayer identifier, or None if absent.

        Returns:
            Layer identifier string, or ``None`` if no masking layer exists.
        """
        if self._operations:
            return self._operations.get_masking_layer_id()
        return None

    def get_deactivated_paths(self) -> set[str]:
        """Return a copy of the set of deactivated (masked or bypassed) prim paths.

        Returns:
            Snapshot of the deactivated path set.
        """
        return set(self._deactivated_paths)

    def get_bypassed_paths(self) -> set[str]:
        """Return a copy of the set of bypassed prim paths.

        Returns:
            Snapshot of the bypassed path set.
        """
        return set(self._bypassed_paths)

    # -- mask toggle ------------------------------------------------------

    def toggle_deactivated(self, original_path: str) -> bool:
        """Toggle plain mask; unmasking a bypassed prim unbypasses it first.

        Args:
            original_path: Original stage prim path string.

        Returns:
            False if the prim is a backward joint locked by an active link
            bypass (no change); True if now deactivated, False if now active.
        """
        if self.is_bypass_controlled(original_path):
            return False
        if original_path in self._deactivated_paths:
            if original_path in self._bypassed_paths:
                self._do_unbypass(original_path)
                self._discard_bypass_joint(original_path)
            else:
                self._do_unmask(original_path)
            self._deactivated_paths.discard(original_path)
            self._bypassed_paths.discard(original_path)
            result = False
        else:
            self._do_mask(original_path)
            self._deactivated_paths.add(original_path)
            result = True
        self._notify_changed()
        return result

    def _set_deactivated_impl(self, original_path: str, deactivated: bool) -> None:
        """Apply deactivation state without notifying subscribers.

        Skips bypass-controlled joints (those locked by an active link bypass).

        Args:
            original_path: Original stage prim path string.
            deactivated: True to deactivate, False to reactivate.
        """
        if self.is_bypass_controlled(original_path):
            return
        if deactivated:
            if original_path not in self._deactivated_paths:
                self._do_mask(original_path)
                self._deactivated_paths.add(original_path)
        else:
            if original_path in self._deactivated_paths:
                if original_path in self._bypassed_paths:
                    self._do_unbypass(original_path)
                    self._discard_bypass_joint(original_path)
                else:
                    self._do_unmask(original_path)
                self._deactivated_paths.discard(original_path)
                self._bypassed_paths.discard(original_path)

    def set_deactivated(self, original_path: str, deactivated: bool) -> None:
        """Set deactivation state for a single prim and notify subscribers.

        Args:
            original_path: Original stage prim path string.
            deactivated: True to deactivate, False to reactivate.
        """
        self._set_deactivated_impl(original_path, deactivated)
        self._notify_changed()

    def set_deactivated_batch(self, paths: list[str], deactivated: bool) -> None:
        """Set deactivation state for multiple prims with a single change notification.

        Args:
            paths: Original stage prim path strings to update.
            deactivated: True to deactivate, False to reactivate.
        """
        for path in paths:
            self._set_deactivated_impl(path, deactivated)
        self._notify_changed()

    # -- bypass toggle ----------------------------------------------------

    def toggle_bypassed(self, original_path: str) -> bool:
        """Toggle bypass state.

        Args:
            original_path: Original stage prim path string.

        Returns:
            False if the prim is a backward joint locked by an active link
            bypass (no change); True if now bypassed, False if now unbypassed.
        """
        if self.is_bypass_controlled(original_path):
            return False
        if original_path in self._bypassed_paths:
            self._do_unbypass(original_path)
            self._discard_bypass_joint(original_path)
            self._deactivated_paths.discard(original_path)
            self._bypassed_paths.discard(original_path)
            result = False
        else:
            # If already masked, unmask first so bypass starts clean
            if original_path in self._deactivated_paths:
                self._do_unmask(original_path)
                self._deactivated_paths.discard(original_path)
            result_data = self._do_bypass(original_path)
            self._deactivated_paths.add(original_path)
            self._bypassed_paths.add(original_path)
            if result_data:
                joint_path, prev_state = result_data
                self._bypass_deactivated_joints[original_path] = (joint_path, prev_state)
                if prev_state == "enabled":
                    # Only newly deactivated joints need to be added
                    self._deactivated_paths.add(joint_path)
                # "masked" and "bypassed": joint already in deactivated/bypassed paths
            result = True
        self._notify_changed()
        return result

    def _set_bypassed_impl(self, original_path: str, bypassed: bool) -> None:
        """Apply bypass state without notifying subscribers.

        Skips bypass-controlled joints (those locked by an active link bypass).

        Args:
            original_path: Original stage prim path string.
            bypassed: True to bypass, False to unbypass.
        """
        if self.is_bypass_controlled(original_path):
            return
        if bypassed:
            if original_path not in self._bypassed_paths:
                if original_path in self._deactivated_paths:
                    self._do_unmask(original_path)
                    self._deactivated_paths.discard(original_path)
                result_data = self._do_bypass(original_path)
                self._deactivated_paths.add(original_path)
                self._bypassed_paths.add(original_path)
                if result_data:
                    joint_path, prev_state = result_data
                    self._bypass_deactivated_joints[original_path] = (joint_path, prev_state)
                    if prev_state == "enabled":
                        self._deactivated_paths.add(joint_path)
        else:
            if original_path in self._bypassed_paths:
                self._do_unbypass(original_path)
                self._discard_bypass_joint(original_path)
                self._deactivated_paths.discard(original_path)
                self._bypassed_paths.discard(original_path)

    def set_bypassed(self, original_path: str, bypassed: bool) -> None:
        """Set bypass state explicitly for a single prim.

        Args:
            original_path: Original stage prim path string.
            bypassed: True to bypass, False to unbypass.
        """
        self._set_bypassed_impl(original_path, bypassed)
        self._notify_changed()

    def set_bypassed_batch(self, paths: list[str], bypassed: bool) -> None:
        """Set bypass state for multiple prims with a single change notification.

        Args:
            paths: Original stage prim path strings to update.
            bypassed: True to bypass, False to unbypass.
        """
        for path in paths:
            self._set_bypassed_impl(path, bypassed)
        self._notify_changed()

    # -- anchor toggle ----------------------------------------------------

    def is_anchored(self, original_path: str) -> bool:
        """Return True if the link is currently anchored to the world.

        Args:
            original_path: Original stage prim path string.

        Returns:
            ``True`` if the path is in the anchored set.
        """
        return original_path in self._anchored_paths

    def get_anchored_paths(self) -> set[str]:
        """Return a copy of the set of anchored link paths.

        Returns:
            Snapshot of the anchored path set.
        """
        return set(self._anchored_paths)

    def toggle_anchored(self, original_path: str) -> bool:
        """Toggle anchor state on a link.

        Args:
            original_path: Original stage prim path string.

        Returns:
            True if now anchored, False if now unanchored.
        """
        if original_path in self._anchored_paths:
            self._do_unanchor(original_path)
            self._anchored_paths.discard(original_path)
            result = False
        else:
            self._do_anchor(original_path)
            self._anchored_paths.add(original_path)
            result = True
        self._notify_changed()
        return result

    def _set_anchored_impl(self, original_path: str, anchored: bool) -> None:
        if anchored:
            if original_path not in self._anchored_paths:
                self._do_anchor(original_path)
                self._anchored_paths.add(original_path)
        else:
            if original_path in self._anchored_paths:
                self._do_unanchor(original_path)
                self._anchored_paths.discard(original_path)

    def set_anchored(self, original_path: str, anchored: bool) -> None:
        """Set anchor state for a single link and notify subscribers.

        Args:
            original_path: Original stage prim path string.
            anchored: True to anchor, False to unanchor.
        """
        self._set_anchored_impl(original_path, anchored)
        self._notify_changed()

    def set_anchored_batch(self, paths: list[str], anchored: bool) -> None:
        """Set anchor state for multiple links with a single change notification.

        Args:
            paths: Original stage prim path strings to update.
            anchored: True to anchor, False to unanchor.
        """
        for path in paths:
            self._set_anchored_impl(path, anchored)
        self._notify_changed()

    # -- clear ------------------------------------------------------------

    def clear(self) -> None:
        """Clear all masking, bypass, and anchor state and notify subscribers."""
        self._deactivated_paths.clear()
        self._bypassed_paths.clear()
        self._anchored_paths.clear()
        self._bypass_deactivated_joints.clear()
        self._notify_changed()

    # -- subscriptions ----------------------------------------------------

    def subscribe_changed(self, callback: Callable[[], None]) -> None:
        """Register a callback to be invoked when any masking state changes.

        Args:
            callback: No-argument callable invoked on change.
        """
        self._on_changed_callbacks.append(callback)

    def unsubscribe_changed(self, callback: Callable[[], None]) -> None:
        """Unregister a previously registered change callback.

        Args:
            callback: Callable to remove.
        """
        if callback in self._on_changed_callbacks:
            self._on_changed_callbacks.remove(callback)

    # -- internal ---------------------------------------------------------

    def _do_mask(self, path: str) -> None:
        if self._operations:
            try:
                self._operations.mask_prim(path)
            except Exception as exc:
                carb.log_error(f"Failed to mask prim at {path}: {exc}\n{traceback.format_exc()}")

    def _do_unmask(self, path: str) -> None:
        if self._operations:
            try:
                self._operations.unmask_prim(path)
            except Exception as exc:
                carb.log_error(f"Failed to unmask prim at {path}: {exc}\n{traceback.format_exc()}")

    def _do_bypass(self, path: str) -> tuple[str, str] | None:
        """Call bypass operation; returns (joint_path, prev_state) for link bypass.

        Args:
            path: Original stage prim path string.

        Returns:
            ``(backward_joint_path, prev_state)`` for a link bypass, or ``None``.
        """
        if self._operations:
            try:
                return self._operations.bypass_prim(path)
            except Exception as exc:
                carb.log_error(f"Failed to bypass prim at {path}: {exc}\n{traceback.format_exc()}")
        return None

    def _do_unbypass(self, path: str) -> tuple[str, str] | None:
        """Call unbypass operation; returns (joint_path, prev_state) for link unbypass.

        Args:
            path: Original stage prim path string.

        Returns:
            ``(backward_joint_path, prev_state)`` for a link unbypass, or ``None``.
        """
        if self._operations:
            try:
                return self._operations.unbypass_prim(path)
            except Exception as exc:
                carb.log_error(f"Failed to unbypass prim at {path}: {exc}\n{traceback.format_exc()}")
        return None

    def _do_anchor(self, path: str) -> None:
        if self._operations:
            try:
                self._operations.anchor_link(path)
            except Exception as exc:
                carb.log_error(f"Failed to anchor link at {path}: {exc}\n{traceback.format_exc()}")

    def _do_unanchor(self, path: str) -> None:
        if self._operations:
            try:
                self._operations.unanchor_link(path)
            except Exception as exc:
                carb.log_error(f"Failed to unanchor link at {path}: {exc}\n{traceback.format_exc()}")

    def _discard_bypass_joint(self, link_path: str) -> None:
        """Restore backward joint UI state when a link is unbypassed.

        Args:
            link_path: Original stage path string of the link being unbypassed.
        """
        entry = self._bypass_deactivated_joints.pop(link_path, None)
        if entry:
            joint_path, prev_state = entry
            if prev_state == "enabled":
                # We added this joint; remove it from deactivated
                self._deactivated_paths.discard(joint_path)
            # "masked" and "bypassed": joint stays in its pre-bypass tracking sets

    def _notify_changed(self) -> None:
        for callback in self._on_changed_callbacks:
            try:
                callback()
            except Exception:
                pass
