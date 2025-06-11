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

from __future__ import annotations

import contextlib
import threading
from typing import Generator

import carb
import omni.kit.stage_templates
import omni.usd
import usdrt
from omni.metrics.assembler.core import get_metrics_assembler_interface
from pxr import Sdf, Usd, UsdUtils

from . import backend as backend_utils
from . import prim as prim_utils

_context = threading.local()  # thread-local storage to handle nested contexts and concurrent access


@contextlib.contextmanager
def use_stage(stage: Usd.Stage) -> Generator[None, None, None]:
    """Context manager that sets a thread-local stage instance.

    Args:
        stage: The stage to set in the context.

    Raises:
        AssertionError: If the stage is not a USD stage instance.

    Example:

    .. code-block:: python

        >>> from pxr import Usd
        >>> import isaacsim.core.experimental.utils.stage as stage_utils
        >>>
        >>> stage_in_memory = Usd.Stage.CreateInMemory()
        >>> with stage_utils.use_stage(stage_in_memory):
        ...    # operate on the specified stage
        ...    pass
        >>> # operate on the default stage attached to the USD context
    """
    # check stage
    assert isinstance(stage, Usd.Stage), f"Expected a USD stage instance, got: {type(stage)}"
    # store previous context value if it exists
    previous_stage = getattr(_context, "stage", None)
    # set new context value
    try:
        _context.stage = stage
        yield
    # remove context value or restore previous one if it exists
    finally:
        if previous_stage is None:
            delattr(_context, "stage")
        else:
            _context.stage = previous_stage


def is_stage_set() -> bool:
    """Check if a stage is set in the context manager.

    Returns:
        Whether a stage is set in the context manager.
    """
    return getattr(_context, "stage", None) is not None


def get_current_stage(*, backend: str | None = None) -> Usd.Stage | usdrt.Usd.Stage:
    """Get the stage set in the context manager or the default stage attached to the USD context.

    Backends: :guilabel:`usd`, :guilabel:`usdrt`, :guilabel:`fabric`.

    Args:
        backend: Backend to use to get the stage. If not ``None``, it has precedence over the current backend
            set via the :py:func:`~isaacsim.core.experimental.utils.impl.backend.use_backend` context manager.

    Returns:
        The current stage instance or the default stage attached to the USD context if no stage is set.

    Raises:
        ValueError: If the backend is not supported.
        ValueError: If there is no stage (set via context manager or attached to the USD context).

    Example:

    .. code-block:: python

        >>> import isaacsim.core.experimental.utils.stage as stage_utils
        >>>
        >>> stage_utils.get_current_stage()
        Usd.Stage.Open(rootLayer=Sdf.Find('anon:...usd'), ...)
    """
    # get backend
    if backend is None:
        backend = backend_utils.get_current_backend(["usd", "usdrt", "fabric"])
    elif backend not in ["usd", "usdrt", "fabric"]:
        raise ValueError(f"Invalid backend: {backend}")
    # get USD stage
    stage = getattr(_context, "stage", omni.usd.get_context().get_stage())
    if stage is None:
        raise ValueError("No stage found. Create a stage first (see `create_new_stage`).")
    # get USDRT/Fabric stage
    if backend in ["usdrt", "fabric"]:
        stage_cache = UsdUtils.StageCache.Get()
        stage_id = stage_cache.GetId(stage).ToLongInt()
        if stage_id < 0:
            stage_id = stage_cache.Insert(stage).ToLongInt()
        return usdrt.Usd.Stage.Attach(stage_id)
    return stage


def get_stage_id(stage: Usd.Stage) -> int:
    """Get the stage ID of a USD stage.

    Backends: :guilabel:`usd`.

    Args:
        stage: The stage to get the ID of.

    Returns:
        The stage ID.

    Example:

    .. code-block:: python

        >>> import isaacsim.core.experimental.utils.stage as stage_utils
        >>>
        >>> stage = stage_utils.get_current_stage()
        >>> stage_utils.get_stage_id(stage)  # doctest: +NO_CHECK
        9223006
    """
    stage_cache = UsdUtils.StageCache.Get()
    stage_id = stage_cache.GetId(stage).ToLongInt()
    if stage_id < 0:
        stage_id = stage_cache.Insert(stage).ToLongInt()
    return stage_id


def create_new_stage(*, template: str | None = None) -> Usd.Stage:
    """Create a new USD stage attached to the USD context.

    Backends: :guilabel:`usd`.

    .. note::

        At least the following templates should be available.
        Other templates might be available depending on app customizations.

        .. list-table::
            :header-rows: 1

            * - Template
              - Description
            * - ``"default stage"``
              - Stage with a gray gridded plane, dome and distant lights, and the ``/World`` Xform prim.
            * - ``"empty"``
              - Empty stage with the ``/World`` Xform prim.
            * - ``"sunlight"``
              - Stage with a distant light and the ``/World`` Xform prim.

    Args:
        template: The template to use to create the stage. If ``None``, a new stage is created with nothing.

    Returns:
        New USD stage instance.

    Raises:
        ValueError: When the template is not found.

    Example:

    .. code-block:: python

        >>> import isaacsim.core.experimental.utils.stage as stage_utils
        >>>
        >>> # create a new stage from the 'sunlight' template
        >>> stage_utils.create_new_stage(template="sunlight")
        Usd.Stage.Open(rootLayer=Sdf.Find('anon:...usd'), ...)
    """
    # create 'empty' stage
    if template is None:
        omni.usd.get_context().new_stage()
    # create stage from template
    else:
        templates = [name for item in omni.kit.stage_templates.get_stage_template_list() for name in item]
        if not template in templates:
            raise ValueError(f"Template '{template}' not found. Available templates: {templates}")
        omni.kit.stage_templates.new_stage(template=template)
    return omni.usd.get_context().get_stage()


async def create_new_stage_async(*, template: str | None = None) -> Usd.Stage:
    """Create a new USD stage attached to the USD context.

    Backends: :guilabel:`usd`.

    This function is the asynchronous version of :py:func:`create_new_stage`.

    Args:
        template: The template to use to create the stage. If ``None``, a new stage is created with nothing.

    Returns:
        New USD stage instance.

    Raises:
        ValueError: When the template is not found.
    """
    # create 'empty' stage
    if template is None:
        await omni.usd.get_context().new_stage_async()
    # create stage from template
    else:
        templates = [name for item in omni.kit.stage_templates.get_stage_template_list() for name in item]
        if not template in templates:
            raise ValueError(f"Template '{template}' not found. Available templates: {templates}")
        await omni.kit.stage_templates.new_stage_async(template=template)
    await omni.kit.app.get_app().next_update_async()
    return omni.usd.get_context().get_stage()


def open_stage(usd_path: str) -> tuple[bool, Usd.Stage | None]:
    """Open a USD file attached to the USD context.

    Backends: :guilabel:`usd`.

    Args:
        usd_path: USD file path to open.

    Returns:
        Two-elements tuple. 1) Whether the USD file was opened successfully.
        2) Opened USD stage instance or None if the USD file was not opened.

    Raises:
        ValueError: If the USD file does not exist or is not a valid (shallow check).

    Example:

    .. code-block:: python

        >>> import isaacsim.core.experimental.utils.stage as stage_utils
        >>> from isaacsim.storage.native import get_assets_root_path
        >>>
        >>> # open a USD file
        >>> result, stage = stage_utils.open_stage(
        ...     get_assets_root_path() + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
        ... )
        >>> result
        True
        >>> stage
        Usd.Stage.Open(rootLayer=Sdf.Find('...'), ...)
    """
    if not Usd.Stage.IsSupportedFile(usd_path):
        raise ValueError(f"The file ({usd_path}) is not USD open-able")
    usd_context = omni.usd.get_context()
    usd_context.disable_save_to_recent_files()
    result = omni.usd.get_context().open_stage(usd_path)
    usd_context.enable_save_to_recent_files()
    return result, usd_context.get_stage()


async def open_stage_async(usd_path: str) -> tuple[bool, Usd.Stage | None]:
    """Open a USD file attached to the USD context.

    Backends: :guilabel:`usd`.

    This function is the asynchronous version of :py:func:`open_stage`.

    Args:
        usd_path: USD file path to open.

    Returns:
        Two-elements tuple. 1) Whether the USD file was opened successfully.
        2) Opened USD stage instance or None if the USD file was not opened.

    Raises:
        ValueError: If the USD file does not exist or is not a valid (shallow check).
    """
    if not Usd.Stage.IsSupportedFile(usd_path):
        raise ValueError(f"The file ({usd_path}) is not USD open-able")
    usd_context = omni.usd.get_context()
    usd_context.disable_save_to_recent_files()
    (result, error) = await omni.usd.get_context().open_stage_async(usd_path)
    usd_context.enable_save_to_recent_files()
    return result, usd_context.get_stage()


def add_reference_to_stage(
    usd_path: str, path: str, *, prim_type: str = "Xform", variants: list[tuple[str, str]] = []
) -> Usd.Prim:
    """Add a USD file reference to the stage at the specified prim path.

    Backends: :guilabel:`usd`.

    .. note::

        This function handles stage units verification to ensure compatibility.

    Args:
        usd_path: USD file path to reference.
        path: Prim path where the reference will be attached.
        prim_type: Prim type to create if the given ``path`` doesn't exist.
        variants: Variants (variant sets and selections) to author on the USD prim.

    Returns:
        USD prim.

    Raises:
        Exception: The USD file might not exist or might not be a valid USD file.
        ValueError: If a variant set or selection is invalid.

    Example:

    .. code-block:: python

        >>> import isaacsim.core.experimental.utils.stage as stage_utils
        >>> from isaacsim.storage.native import get_assets_root_path
        >>>
        >>> prim = stage_utils.add_reference_to_stage(
        ...     usd_path=get_assets_root_path() + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
        ...     path="/panda",
        ...     variants=[("Gripper", "AlternateFinger"), ("Mesh", "Performance")],
        ... )
    """
    if not Sdf.Path.IsValidPathString(path):
        raise ValueError(f"Prim path ({path}) is not a valid path string")
    sdf_layer = Sdf.Layer.FindOrOpen(usd_path)
    if not sdf_layer:
        raise Exception(
            f"Unable to get Sdf layer. The USD file ({usd_path}) might not exist or is not a valid USD file."
        )
    stage = get_current_stage(backend="usd")
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        prim = stage.DefinePrim(path, prim_type)
    reference = Sdf.Reference(usd_path)
    reference_added = False
    # check for divergent units
    result = get_metrics_assembler_interface().check_layers(
        stage.GetRootLayer().identifier, sdf_layer.identifier, get_stage_id(stage)
    )
    if result["ret_val"]:
        try:
            import omni.kit.commands
            import omni.metrics.assembler.ui

            omni.kit.commands.execute("AddReference", stage=stage, prim_path=path, reference=reference)
            reference_added = True
        except Exception as e:
            carb.log_warn(
                (
                    f"The USD file ({usd_path}) has divergent units. "
                    "Enable the omni.usd.metrics.assembler.ui extension or convert the file into right units."
                )
            )
    # add reference (if not already added during divergent units check)
    if not reference_added:
        result = prim.GetReferences().AddReference(reference)
        if not result:
            raise Exception(f"Unable to add reference to the USD file ({usd_path}).")
    # set variants
    prim_utils.set_prim_variants(prim, variants=variants)
    return prim


def define_prim(path: str, type_name: str = "Xform") -> Usd.Prim | usdrt.Usd.Prim:
    """Attempt to define a prim of the specified type at the given path.

    Backends: :guilabel:`usd`, :guilabel:`usdrt`, :guilabel:`fabric`.

    Common token values for ``type_name`` are:

    * ``"Camera"``, ``"Mesh"``, ``"PhysicsScene"``, ``"Xform"``
    * Shapes (``"Capsule"``, ``"Cone"``, ``"Cube"``, ``"Cylinder"``, ``"Plane"``, ``"Sphere"``)
    * Lights (``"CylinderLight"``, ``"DiskLight"``, ``"DistantLight"``, ``"DomeLight"``, ``"RectLight"``, ``"SphereLight"``)

    Args:
        path: Absolute prim path.
        type_name: Token identifying the prim type.

    Raises:
        ValueError: If the path is not a valid or absolute path string.
        RuntimeError: If there is already a prim at the given path with a different type.

    Returns:
        Defined prim.

    Example:

    .. code-block:: python

        >>> import isaacsim.core.experimental.utils.stage as stage_utils
        >>>
        >>> stage_utils.define_prim("/World/Sphere", type_name="Sphere")
        Usd.Prim(</World/Sphere>)
    """
    if not Sdf.Path.IsValidPathString(path) or not Sdf.Path(path).IsAbsolutePath():
        raise ValueError(f"Prim path ({path}) is not a valid or absolute path string")
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid():
        if prim.GetTypeName() != type_name:
            raise RuntimeError(f"A prim already exists at path ({path}) with type ({prim.GetTypeName()})")
        return prim
    return stage.DefinePrim(path, type_name)
