# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import weakref

import carb
import isaacsim.core.experimental.utils.stage as stage_utils
import omni.kit.app
import omni.timeline
import omni.usd
from pxr import Usd

_SETTING_PLAY_SIMULATION = "/app/player/playSimulations"
_SETTING_RATE_LIMIT_ENABLED = "/app/runLoops/main/rateLimitEnabled"
_SETTING_RATE_LIMIT_FREQUENCY = "/app/runLoops/main/rateLimitFrequency"

from enum import Enum


class RenderingEvent(Enum):
    """Rendering event types."""

    NEW_FRAME = "isaacsim.rendering.new_frame"
    """Event triggered when a new frame is rendered."""


class RenderingManager:
    """Core class that provides APIs for controlling rendering."""

    _app = omni.kit.app.get_app()
    _callbacks = dict()
    _callback_registry = 0
    _carb_settings = carb.settings.get_settings()
    _event_dispatcher = carb.eventdispatcher.get_eventdispatcher()
    _timeline = omni.timeline.get_timeline_interface()
    try:
        import omni.kit.loop._loop as kit_loop

        _loop_runner = kit_loop.acquire_loop_interface()
    except Exception as e:
        carb.log_warn(f"Isaac Sim's loop runner not found. Its functionalities will not be used: {e}")
        _loop_runner = None

    @classmethod
    def render(cls) -> None:
        """Render the stage.

        This method performs an app update without stepping the simulation or physics.

        Example:

        .. code-block:: python

            >>> from isaacsim.core.rendering_manager import RenderingManager
            >>>
            >>> RenderingManager.render()
        """
        play_simulation = cls._carb_settings.get_as_bool(_SETTING_PLAY_SIMULATION)
        if play_simulation:
            cls._carb_settings.set_bool(_SETTING_PLAY_SIMULATION, False)
        cls._app.update()
        if play_simulation:
            cls._carb_settings.set_bool(_SETTING_PLAY_SIMULATION, True)

    @classmethod
    async def render_async(cls) -> None:
        """Render the stage.

        This method is the asynchronous version of :py:meth:`render`.
        """
        play_simulation = cls._carb_settings.get_as_bool(_SETTING_PLAY_SIMULATION)
        if play_simulation:
            cls._carb_settings.set_bool(_SETTING_PLAY_SIMULATION, False)
        await cls._app.next_update_async()
        if play_simulation:
            cls._carb_settings.set_bool(_SETTING_PLAY_SIMULATION, True)

    @classmethod
    def set_dt(cls, dt: float) -> None:
        """Set the rendering dt.

        Args:
            dt: Rendering dt.

        Example:

        .. code-block:: python

            >>> from isaacsim.core.rendering_manager import RenderingManager
            >>>
            >>> RenderingManager.set_dt(1 / 120.0)  # 120 Hz
        """

        def _set_rate_limit_frequency(frequency):
            cls._carb_settings.set_bool(_SETTING_RATE_LIMIT_ENABLED, True)
            cls._carb_settings.set_float(_SETTING_RATE_LIMIT_FREQUENCY, frequency)
            cls._timeline.set_target_framerate(frequency)

        frequency = 1.0 / dt
        # Set the rate limit only if it is enabled.
        # Otherwise, run the app as fast as possible without specifying a target rate
        if cls._carb_settings.get_as_bool(_SETTING_RATE_LIMIT_ENABLED):
            _set_rate_limit_frequency(frequency)
        # Set the stage's `timeCodesPerSecond` value
        stage = stage_utils.get_current_stage(backend="usd")
        with Usd.EditContext(stage, stage.GetRootLayer()):
            stage.SetTimeCodesPerSecond(frequency)
        cls._timeline.set_time_codes_per_second(frequency)
        # Set Isaac Sim's loop runner-specific configuration. If it is not available, fall back to the app settings
        if hasattr(cls._loop_runner, "set_manual_step_size") and hasattr(cls._loop_runner, "set_manual_mode"):
            cls._loop_runner.set_manual_step_size(dt)
            cls._loop_runner.set_manual_mode(True)
        else:
            carb.log_warn(f"Isaac Sim's loop runner not found. Setting a rate limit instead ({frequency} Hz)")
            _set_rate_limit_frequency(frequency)

    @classmethod
    def get_dt(cls) -> float:
        """Get the rendering dt.

        Returns:
            Rendering dt.

        Example:

        .. code-block:: python

            >>> from isaacsim.core.rendering_manager import RenderingManager
            >>>
            >>> RenderingManager.get_dt()
            0.0166666...
        """

        def _get_rate_limit_dt():
            frequency = cls._carb_settings.get_as_float(_SETTING_RATE_LIMIT_FREQUENCY)
            return 1.0 / frequency if frequency else 0.0

        # Read from the app settings only if the rate limit is enabled in the first instance
        if cls._carb_settings.get_as_bool(_SETTING_RATE_LIMIT_ENABLED):
            return _get_rate_limit_dt()
        # Read from Isaac Sim's loop runner-specific configuration. If it is not available, fall back to the app settings
        if hasattr(cls._loop_runner, "get_manual_step_size") and hasattr(cls._loop_runner, "get_manual_mode"):
            if cls._loop_runner.get_manual_mode():
                return cls._loop_runner.get_manual_step_size()
        return _get_rate_limit_dt()

    @classmethod
    def register_callback(cls, event: RenderingEvent, *, callback: callable, order: int = 0) -> int:
        """Register/subscribe a callback to be triggered when a specific rendering event occurs.

        Args:
            event: The rendering event to subscribe to.
            callback: The callback function.
            order: The subscription order.
                Callbacks registered within the same order will be triggered in the order they were registered.

        Returns:
            The unique identifier of the callback subscription.

        Example:

        .. code-block:: python

            >>> from isaacsim.core.rendering_manager import RenderingEvent, RenderingManager
            >>>
            >>> def callback(event, *args, **kwargs):
            ...     print(event)
            ...
            >>> # subscribe to the NEW_FRAME event
            >>> callback_id = RenderingManager.register_callback(RenderingEvent.NEW_FRAME, callback=callback)
            >>> callback_id
            0
            >>> # perform a rendering step in order to trigger the callback and print the event
            >>> RenderingManager.render()  # doctest: +NO_CHECK
            <carb.eventdispatcher._eventdispatcher.Event object at 0x...>
            >>>
            >>> # deregister all callbacks
            >>> RenderingManager.deregister_all_callbacks()
        """
        if event not in RenderingEvent:
            raise ValueError(f"Invalid rendering event: {event}. Supported events are: {list(RenderingEvent)}")
        # check for weak reference support (when the callback is a method of a class)
        if hasattr(callback, "__self__"):
            on_event = lambda event, obj=weakref.proxy(callback.__self__): getattr(obj, callback.__name__)(event)
        else:
            on_event = callback
        # get a unique id for the callback
        uid = cls._callback_registry
        cls._callback_registry += 1
        # register the callback
        if event in [RenderingEvent.NEW_FRAME]:
            cls._callbacks[uid] = cls._event_dispatcher.observe_event(
                observer_name=f"isaacsim.core.rendering_manager:callback.{event.value}.{uid}",
                event_name=omni.usd.get_context().stage_rendering_event_name(
                    omni.usd.StageRenderingEventType.NEW_FRAME, True
                ),
                on_event=on_event,
                order=order,
            )
        else:
            raise RuntimeError(f"Unable to register callback for event '{event}' with uid '{uid}'")
        return uid

    @classmethod
    def deregister_callback(cls, uid: int) -> None:
        """Deregister a callback registered via :py:meth:`register_callback`.

        Args:
            uid: The unique identifier of the callback to deregister. If the unique identifier does not exist
                or has already been deregistered, a warning is logged and the method does nothing.

        Example:

        .. code-block:: python

            >>> from isaacsim.core.rendering_manager import RenderingManager
            >>>
            >>> # deregister the callback with the unique identifier 0
            >>> RenderingManager.deregister_callback(0)
        """
        if uid not in cls._callbacks:
            carb.log_warn(f"Unable to deregister callback with uid '{uid}'. It might have been already deregistered")
            return
        del cls._callbacks[uid]

    @classmethod
    def deregister_all_callbacks(cls) -> None:
        """Deregister all callbacks registered via :py:meth:`register_callback`.

        Example:

        .. code-block:: python

            >>> from isaacsim.core.rendering_manager import RenderingManager
            >>>
            >>> RenderingManager.deregister_all_callbacks()
        """
        cls._callbacks.clear()
