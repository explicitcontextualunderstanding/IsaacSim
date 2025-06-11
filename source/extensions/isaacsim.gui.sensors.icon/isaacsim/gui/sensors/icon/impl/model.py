# SPDX-FileCopyrightText: Copyright (c) 2018-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
__all__ = ["IconModel"]

from pathlib import Path

import carb
import omni.kit.app
import omni.usd
import usdrt.Usd
from omni.ui import scene as sc
from pxr import Gf, Sdf, Tf, Trace, Usd, UsdGeom

ICON_POSITION_ATTR = "xformOp:translate"
ICON_ORIENTATION_ATTR = "xformOp:orient"
ICON_TRANSFORM_ATTR = "xformOp:transform"


class IconModel(sc.AbstractManipulatorModel):

    SENSOR_TYPES = ["Lidar", "OmniLidar", "IsaacContactSensor", "IsaacLightBeamSensor", "IsaacImuSensor", "Generic"]

    class IconItem(sc.AbstractManipulatorItem):

        def __init__(self, prim_path, icon_url):
            super().__init__()
            self.icon_url = icon_url
            self.prim_path = prim_path
            self.on_click = None
            self.removed = False
            self.visible = True

    def __init__(self):
        super().__init__()
        self._usd_listening_active = True
        self._sensor_icon_dir = omni.kit.app.get_app().get_extension_manager().get_extension_path_by_module(__name__)
        self._sensor_icon_path = str(Path(self._sensor_icon_dir).joinpath("icons/icoSensors.svg"))
        self._usd_context = omni.usd.get_context()
        self._usdrt_stage = None
        self._world_unit = 0.1
        self._icons = {}
        self._usd_listener = None

        bus = carb.eventdispatcher.get_eventdispatcher()

        self._stage_open_sub = bus.observe_event(
            event_name=omni.usd.get_context().stage_event_name(omni.usd.StageEventType.OPENED),
            on_event=self._on_stage_opened,
            observer_name="IconModel._on_stage_opened",
        )

        self._stage_close_sub = bus.observe_event(
            event_name=omni.usd.get_context().stage_event_name(omni.usd.StageEventType.CLOSED),
            on_event=self._on_stage_closed,
            observer_name="IconModel._on_stage_closed",
        )

        self._connect_to_stage()

    def _connect_to_stage(self):
        stage = self._usd_context.get_stage()
        if stage:
            stage_id = self._usd_context.get_stage_id()
            try:
                self._usdrt_stage = usdrt.Usd.Stage.Attach(stage_id)
            except Exception as e:
                print(f"[Error] Failed to attach usdrt stage: {e}")
                self._usdrt_stage = None

            self._world_unit = UsdGeom.GetStageMetersPerUnit(stage)

            if self._world_unit == 0.0:
                self._world_unit = 0.1

            if self._usd_listening_active:
                self._usd_listener = Tf.Notice.Register(Usd.Notice.ObjectsChanged, self._on_usd_changed, stage)
                self._populate_initial_icons()
            else:
                if self._usd_listener:
                    self._usd_listener.Revoke()
                    self._usd_listener = None

                # populate icons with hidden state
                self._populate_initial_icons()
                for item in self._icons.values():
                    item.visible = False
                self._item_changed(None)
        else:
            self._usdrt_stage = None
            if self._usd_listener:
                self._usd_listener.Revoke()
            self._usd_listener = None
            self.clear()

    def _populate_initial_icons(self):
        """Populate icons by querying the USDrt stage."""
        self.clear()
        if not self._usdrt_stage:
            return

        stage = self._usd_context.get_stage()
        if not stage:
            return

        all_sensor_paths = set()
        for sensor_type_str in self.SENSOR_TYPES:
            try:
                paths = self._usdrt_stage.GetPrimsWithTypeName(sensor_type_str)
                all_sensor_paths.update(paths)
            except Exception as e:
                carb.log_warn(f"Failed querying usdrt for type {sensor_type_str}: {e}")

        for prim_path_obj in all_sensor_paths:
            prim_path = Sdf.Path(str(prim_path_obj))
            prim = stage.GetPrimAtPath(prim_path)
            if not prim:
                continue

            # check visibility and activation status
            is_active = prim.IsActive()
            should_be_visible = False
            try:
                visibility = UsdGeom.Imageable(prim).ComputeVisibility()
                should_be_visible = is_active and visibility != UsdGeom.Tokens.invisible
            except Exception as e:
                carb.log_warn(f"[Warning] Failed to compute visibility/activation for {prim_path}: {e}")

            # Ignore hidden sensors
            if not should_be_visible:
                continue

            item = IconModel.IconItem(prim_path, self._sensor_icon_path)
            item.visible = should_be_visible if self._usd_listening_active else False
            self._icons[prim_path] = item

        self._item_changed(None)

    def _on_stage_opened(self, event):
        # Revoke existing listener before reconnecting
        if self._usd_listener:
            self._usd_listener.Revoke()
            self._usd_listener = None
        self._connect_to_stage()

    def _on_stage_closed(self, event):
        self.clear()
        self._usdrt_stage = None
        if self._usd_listener:
            self._usd_listener.Revoke()
            self._usd_listener = None

    def get_world_unit(self):
        return max(self._world_unit, 0.1)

    def __del__(self):
        self._stage_open_sub = None
        self._stage_close_sub = None
        self._usd_listener = None
        self.destroy()

    def destroy(self):
        self._icons = {}
        if self._usd_listener:
            self._usd_listener.Revoke()
        self._usd_listener = None

    def get_item(self, identifier):
        if isinstance(identifier, str):
            identifier = Sdf.Path(identifier)
        return self._icons.get(identifier)

    def get_prim_paths(self):
        return list(self._icons.keys())

    def get_position(self, prim_path):
        if not isinstance(prim_path, Sdf.Path):
            prim_path = Sdf.Path(prim_path)

        if prim_path in self._icons:
            stage = self._usd_context.get_stage()
            if not stage:
                return None
            prim = stage.GetPrimAtPath(prim_path)
            if prim and prim.IsValid():
                try:
                    xformCache = UsdGeom.XformCache(Usd.TimeCode.Default())
                    worldTransform = xformCache.GetLocalToWorldTransform(prim)
                    translation = worldTransform.ExtractTranslation()
                    return Gf.Vec3d(translation[0], translation[1], translation[2])
                except Exception as e:
                    carb.log_warn(f"Failed to compute transform for {prim_path}: {e}")
                    return None
        return None

    def get_on_click(self, prim_path):
        if not isinstance(prim_path, Sdf.Path):
            prim_path = Sdf.Path(prim_path)
        item = self._icons.get(prim_path)
        return item.on_click if item else None

    def get_icon_url(self, prim_path):
        if not isinstance(prim_path, Sdf.Path):
            prim_path = Sdf.Path(prim_path)
        item = self._icons.get(prim_path)
        return item.icon_url if item else ""

    @Trace.TraceFunction
    def _on_usd_changed(self, notice, stage):
        if not self._usd_listening_active:
            return

        if not self._usdrt_stage and not stage:
            return

        try:
            _ = UsdGeom.Tokens.invisible
        except AttributeError:
            # carb.log_warn("UsdGeom.Tokens not available. Icon visibility checks will be skipped.")
            return

        current_prim_paths = set(self._icons.keys())
        added_paths = set()
        removed_paths = set()
        changed_paths_to_recheck = set()

        # Current sensor states
        current_sensor_paths = set()
        if self._usdrt_stage:
            for sensor_type_str in self.SENSOR_TYPES:
                try:
                    paths = self._usdrt_stage.GetPrimsWithTypeName(sensor_type_str)
                    current_sensor_paths.update(Sdf.Path(str(p)) for p in paths if p)
                except Exception as e:
                    # carb.log_warn(f"Failed querying usdrt during change notification for type {sensor_type_str}: {e}")
                    pass
        else:
            # Fallback to USD stage query if usdrt fails or is unavailable
            for prim in Usd.PrimRange.AllPrims(stage.GetPseudoRoot()):
                prim_type = str(prim.GetTypeName())
                if prim_type in self.SENSOR_TYPES:
                    current_sensor_paths.add(prim.GetPath())

        # Verify additions and deletions
        added_paths = current_sensor_paths - current_prim_paths
        removed_paths = current_prim_paths - current_sensor_paths

        # Recheck paths based on notice
        potentially_changed_prims_from_notice = set()
        resynced_paths = notice.GetResyncedPaths()
        changed_info_paths = notice.GetChangedInfoOnlyPaths()

        all_notice_paths = resynced_paths + changed_info_paths
        for path in all_notice_paths:
            prim_path = path.GetPrimPath()
            if prim_path and prim_path != Sdf.Path.emptyPath:
                potentially_changed_prims_from_notice.add(prim_path)

        # Recheck tracked icons if ancestor or prim itself in notice
        tracked_icons_to_consider = current_prim_paths - removed_paths
        for tracked_path in tracked_icons_to_consider:
            if tracked_path in potentially_changed_prims_from_notice:
                changed_paths_to_recheck.add(tracked_path)
                continue
            for notice_prim_path in potentially_changed_prims_from_notice:
                if tracked_path.HasPrefix(notice_prim_path) and tracked_path != notice_prim_path:
                    changed_paths_to_recheck.add(tracked_path)
                    break  # No need to check other notice paths for this tracked_path

        # property change triggering updates
        transform_props = {ICON_POSITION_ATTR, ICON_ORIENTATION_ATTR, ICON_TRANSFORM_ATTR, "xformOpOrder"}
        visibility_props = {"visibility", UsdGeom.Tokens.visibility}
        activation_props = {"active"}

        for path in changed_info_paths:
            if path.IsPropertyPath():
                prim_path = path.GetPrimPath()
                prop_name = path.name

                if prim_path in tracked_icons_to_consider:
                    if prop_name in transform_props or prop_name in visibility_props or prop_name in activation_props:
                        changed_paths_to_recheck.add(prim_path)

                for tracked_path in tracked_icons_to_consider:
                    if tracked_path.HasPrefix(prim_path) and tracked_path != prim_path:
                        if (
                            prop_name in visibility_props
                            or prop_name in activation_props
                            or prop_name in transform_props
                        ):
                            changed_paths_to_recheck.add(tracked_path)

        # handle deletions
        items_changed_notification = []
        for prim_path in removed_paths:
            if prim_path in self._icons:
                item = self._icons.pop(prim_path)
                if not item.removed:
                    item.removed = True
                    items_changed_notification.append(item)

        # handle additions
        for prim_path in added_paths:
            if prim_path not in self._icons:
                prim = stage.GetPrimAtPath(prim_path)
                # verify valid prim and valid type
                if prim and str(prim.GetTypeName()) in self.SENSOR_TYPES:
                    try:
                        is_active = prim.IsActive()
                        visibility = UsdGeom.Imageable(prim).ComputeVisibility()
                        should_be_visible = is_active and visibility != UsdGeom.Tokens.invisible
                    except Exception as e:
                        carb.log_warn(f"Failed checking state for new prim {prim_path}: {e}")
                        should_be_visible = False

                    item = IconModel.IconItem(prim_path, self._sensor_icon_path)
                    item.visible = should_be_visible
                    self._icons[prim_path] = item
                    items_changed_notification.append(item)

        # handle potential updates
        paths_to_evaluate = changed_paths_to_recheck - removed_paths - added_paths

        for prim_path in paths_to_evaluate:
            item = self._icons.get(prim_path)
            if not item or item.removed:
                continue

            prim = stage.GetPrimAtPath(prim_path)
            if not prim or not prim.IsValid():
                if prim_path not in removed_paths:
                    removed_paths.add(prim_path)
                    item = self._icons.pop(prim_path)
                    item.removed = True
                    items_changed_notification.append(item)
                continue

            # Prim exists, re-compute its visibility state
            needs_update = False
            try:
                is_active = prim.IsActive()
                visibility = UsdGeom.Imageable(prim).ComputeVisibility()
                should_be_visible = is_active and visibility != UsdGeom.Tokens.invisible

                if item.visible != should_be_visible:
                    item.visible = should_be_visible
                    needs_update = True  # Visibility state changed

            except Exception as e:
                if item.visible:  # if it was visible, hide it on error
                    item.visible = False
                    needs_update = True

            # path was added to changed_paths_to_recheck due to property change
            if not needs_update:
                for path in changed_info_paths:
                    if path.IsPropertyPath() and path.GetPrimPath() == prim_path:
                        if path.name in transform_props:
                            needs_update = True
                            break
                    elif prim_path.HasPrefix(path.GetPrimPath()) and path.GetPrimPath() != prim_path:
                        if path.IsPropertyPath() and path.name in transform_props:
                            needs_update = True
                            break

            if needs_update:
                items_changed_notification.append(item)

        notified_items = set()
        if items_changed_notification:
            for item in items_changed_notification:
                item_id = id(item)
                if item_id not in notified_items:
                    # update UI
                    self._item_changed(item)
                    notified_items.add(item_id)

    def clear(self):
        if self._icons:
            self._icons = {}
            self._item_changed(None)

    def add_sensor_icon(self, prim_path, icon_url=None):
        if not isinstance(prim_path, Sdf.Path):
            prim_path = Sdf.Path(prim_path)

        if prim_path in self._icons:
            return

        is_sensor = False
        if self._usdrt_stage:
            try:
                prim_type_token = self._usdrt_stage.GetPrimAtPath(str(prim_path)).GetTypeName()
                if str(prim_type_token) in self.SENSOR_TYPES:
                    is_sensor = True
            except Exception:
                pass

        if not is_sensor:
            stage = self._usd_context.get_stage()
            if stage:
                prim = stage.GetPrimAtPath(prim_path)
                if prim and str(prim.GetTypeName()) in self.SENSOR_TYPES:
                    is_sensor = True

        # If confirmed as a sensor by either method, add the icon
        if is_sensor:
            icon_url = icon_url or self._sensor_icon_path
            item = IconModel.IconItem(prim_path, icon_url)

            # Check initial visibility and activation state
            should_be_visible = True  # Default to visible
            stage = self._usd_context.get_stage()
            if stage:
                prim = stage.GetPrimAtPath(prim_path)
                if prim:
                    try:
                        is_active = prim.IsActive()
                        # Use default timecode for initial check
                        visibility = UsdGeom.Imageable(prim).ComputeVisibility()
                        should_be_visible = is_active and visibility != UsdGeom.Tokens.invisible
                    except Exception as e:
                        should_be_visible = False  # Default to hidden if check fails
                else:
                    should_be_visible = False  # Prim doesn't exist? Hide.
            else:
                should_be_visible = False  # No stage? Hide.

            item.visible = should_be_visible and self._usd_listening_active
            self._icons[prim_path] = item
            self._item_changed(item)

    def remove_sensor_icon(self, prim_path):
        if not isinstance(prim_path, Sdf.Path):
            prim_path = Sdf.Path(prim_path)

        if prim_path in self._icons:
            self._icons[prim_path].removed = True
            self._item_changed(self._icons[prim_path])
            self._icons.pop(prim_path)

    def set_icon_click_fn(self, prim_path, call_back):
        if not isinstance(prim_path, Sdf.Path):
            prim_path = Sdf.Path(prim_path)
        item = self._icons.get(prim_path)
        if item:
            item.on_click = call_back

    def show_sensor_icon(self, prim_path):
        if not isinstance(prim_path, Sdf.Path):
            prim_path = Sdf.Path(prim_path)
        item = self._icons.get(prim_path)
        if item:
            if self._usd_listening_active:
                item.visible = True
                self._item_changed(item)

    def hide_sensor_icon(self, prim_path):
        if not isinstance(prim_path, Sdf.Path):
            prim_path = Sdf.Path(prim_path)
        item = self._icons.get(prim_path)
        if item:
            item.visible = False
            self._item_changed(item)

    def show_all(self):
        # Activate USD listening
        self._usd_listening_active = True

        # Re-register the USD listener if needed
        stage = self._usd_context.get_stage()
        if stage and not self._usd_listener:
            self._usd_listener = Tf.Notice.Register(Usd.Notice.ObjectsChanged, self._on_usd_changed, stage)

        # Refresh all icons from the current USD state
        self._populate_initial_icons()

    def hide_all(self):
        # Deactivate USD listening
        self._usd_listening_active = False

        # Revoke USD listener
        if self._usd_listener:
            self._usd_listener.Revoke()
            self._usd_listener = None

        # Forcefully clear all icons from the model.
        if self._icons:
            self._icons = {}
            self._item_changed(None)

    def refresh_all_icon_visuals(self):
        """Force a refresh notification for all currently tracked icon items."""
        if not self._usd_listening_active:
            return

        # carb.log_info("Forcing refresh of all sensor icon visuals due to simulation state change.")
        # Destroy existing icons and repopulate
        if self._icons:
            self._icons = {}
            self._item_changed(None)
            self._populate_initial_icons()
