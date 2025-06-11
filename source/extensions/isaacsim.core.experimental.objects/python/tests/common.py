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

from typing import Callable, Literal

import omni.kit.test
from isaacsim.core.simulation_manager import SimulationManager


def parametrize(
    *,
    devices: list[Literal["cpu", "cuda"]] = ["cpu", "cuda"],
    backends: list[Literal["usd", "usdrt", "fabric", "tensor"]] = ["usd", "usdrt", "fabric", "tensor"],
    instances: list[Literal["one", "many"]] = ["one", "many"],
    operations: list[Literal["wrap", "create"]] = ["wrap", "create"],
    prim_classes: list[type],
    prim_classes_kwargs: list[dict] | None = None,
    populate_stage_func: Callable[[int, Literal["wrap", "create"]], None],
    max_num_prims: int = 5,
):
    def decorator(func):
        async def wrapper(self):
            for device in devices:
                for backend in backends:
                    for instance in instances:
                        for operation in operations:
                            for prim_class, prim_class_kwargs in zip(
                                prim_classes,
                                [{}] * len(prim_classes) if prim_classes_kwargs is None else prim_classes_kwargs,
                            ):
                                assert backend in ["usd", "fabric", "tensor"], f"Invalid backend: {backend}"
                                assert instance in ["one", "many"], f"Invalid instance: {instance}"
                                assert operation in ["wrap", "create"], f"Invalid operation: {operation}"
                                print(
                                    f"  |-- device: {device}, backend: {backend}, instance: {instance}, operation: {operation}, prim_class: {prim_class.__name__}"
                                )
                                # populate stage
                                await populate_stage_func(max_num_prims, operation)
                                # configure simulation manager
                                SimulationManager.set_physics_sim_device(device)
                                # parametrize test
                                if operation == "wrap":
                                    paths = "/World/A_0" if instance == "one" else "/World/A_.*"
                                elif operation == "create":
                                    paths = (
                                        "/World/A_0"
                                        if instance == "one"
                                        else [f"/World/A_{i}" for i in range(max_num_prims)]
                                    )
                                prim = prim_class(paths, **prim_class_kwargs)
                                num_prims = 1 if instance == "one" else max_num_prims
                                # call test method according to backend
                                if backend == "tensor":
                                    omni.timeline.get_timeline_interface().play()
                                    await omni.kit.app.get_app().next_update_async()
                                    await func(self, prim=prim, num_prims=num_prims, device=device, backend=backend)
                                    omni.timeline.get_timeline_interface().stop()
                                elif backend in ["usd", "usdrt", "fabric"]:
                                    await func(self, prim=prim, num_prims=num_prims, device=device, backend=backend)

        return wrapper

    return decorator
