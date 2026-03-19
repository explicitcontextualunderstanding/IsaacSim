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

"""
Functions for creating, accessing and deleting semantic labels.
"""

from __future__ import annotations

from collections import defaultdict

from pxr import Usd, UsdSemantics

from . import stage as stage_utils


def add_labels(prim: str | Usd.Prim, *, labels: str | list[str], taxonomy: str = "class") -> None:
    """Add semantic labels, given a taxonomy (instance name), to a prim.

    Backends: :guilabel:`usd`.

    Args:
        prim: Prim path or prim instance.
        labels: Label(s) to add to existing ones (if any).
        taxonomy: Name of the taxonomy (instance name).

    Examples:

    .. code-block:: python

        >>> import isaacsim.core.experimental.utils.semantics as semantics_utils
        >>> import isaacsim.core.experimental.utils.stage as stage_utils
        >>>
        >>> stage_utils.define_prim("/World/Cube", "Cube")  # doctest: +NO_CHECK
        >>>
        >>> # add some labels to the default 'class' and 'custom' taxonomies
        >>> semantics_utils.add_labels("/World/Cube", labels=["label_a", "label_b"])
        >>> semantics_utils.add_labels("/World/Cube", labels=["label_c"], taxonomy="custom")
        >>>
        >>> # get the labels
        >>> semantics_utils.get_labels("/World/Cube")
        {'class': ['label_a', 'label_b'], 'custom': ['label_c']}
    """
    prim = stage_utils.get_current_stage(backend="usd").GetPrimAtPath(prim) if isinstance(prim, str) else prim
    labels = [labels] if isinstance(labels, str) else labels
    labels_attribute = UsdSemantics.LabelsAPI.Apply(prim, taxonomy).GetLabelsAttr()
    existing_labels = list(labels_attribute.Get())
    for label in labels:
        if label not in existing_labels:
            existing_labels.append(label)
    labels_attribute.Set(existing_labels)


def get_labels(prim: str | Usd.Prim, *, include_descendants: bool = False) -> dict[str, list[str]]:
    """Get all the semantic labels applied to a prim.

    Backends: :guilabel:`usd`.

    Args:
        prim: Prim path or prim instance.
        include_descendants: Whether to include labels from all descendants of the prim.

    Returns:
        Dictionary mapping taxonomies (instance names) to a list of semantic labels applied to the prim.

    Examples:

    .. code-block:: python

        >>> import isaacsim.core.experimental.utils.semantics as semantics_utils
        >>> import isaacsim.core.experimental.utils.stage as stage_utils
        >>>
        >>> stage_utils.define_prim("/World/Cube", "Cube")  # doctest: +NO_CHECK
        >>>
        >>> # add some labels to the default 'class' and 'custom' taxonomies
        >>> semantics_utils.add_labels("/World/Cube", labels=["label_a", "label_b"])
        >>> semantics_utils.add_labels("/World/Cube", labels=["label_c"], taxonomy="custom")
        >>>
        >>> # get the labels
        >>> semantics_utils.get_labels("/World/Cube")
        {'class': ['label_a', 'label_b'], 'custom': ['label_c']}
    """

    def _get_labels(target_prim: Usd.Prim):
        for schema in target_prim.GetAppliedSchemas():
            if schema.startswith("SemanticsLabelsAPI:"):
                taxonomy = schema.split(":", 1)[-1]
                labels[taxonomy]  # ensure the list is initialized (if not already)
                labels_attribute = UsdSemantics.LabelsAPI(target_prim, taxonomy).GetLabelsAttr()
                if labels_attribute:
                    labels[taxonomy].extend(list(labels_attribute.Get()))

    prim = stage_utils.get_current_stage(backend="usd").GetPrimAtPath(prim) if isinstance(prim, str) else prim
    labels = defaultdict(list)
    if include_descendants:
        for p in Usd.PrimRange(prim):
            _get_labels(p)
    else:
        _get_labels(prim)
    return dict(labels)


def remove_labels(
    prim: str | Usd.Prim, *, labels: str | list[str], taxonomy: str | None = None, include_descendants: bool = False
) -> None:
    """Remove semantic labels from a prim.

    Backends: :guilabel:`usd`.

    This function removes specific labels from a prim (and optionally all its descendants).
    To remove all labels, use the :py:func:`remove_all_labels` function instead.

    Args:
        prim: Prim path or prim instance.
        labels: Label(s) to remove (if any).
        taxonomy: Name of the taxonomy (instance name) to remove labels from.
            If not specified, matching labels from all taxonomies will be removed.
        include_descendants: Whether to remove labels from all descendants of the prim.

    Examples:

    .. code-block:: python

        >>> import isaacsim.core.experimental.utils.semantics as semantics_utils
        >>> import isaacsim.core.experimental.utils.stage as stage_utils
        >>>
        >>> stage_utils.define_prim("/World/Cube", "Cube")  # doctest: +NO_CHECK
        >>>
        >>> # add some labels to the default 'class' and 'custom' taxonomies
        >>> semantics_utils.add_labels("/World/Cube", labels=["label_a", "label_b"])
        >>> semantics_utils.add_labels("/World/Cube", labels=["label_c"], taxonomy="custom")
        >>>
        >>> # remove one of the labels and get the remaining ones
        >>> semantics_utils.remove_labels("/World/Cube", labels="label_a")
        >>> semantics_utils.get_labels("/World/Cube")
        {'class': ['label_b'], 'custom': ['label_c']}
    """

    def _remove_labels(target_prim: Usd.Prim):
        for schema in target_prim.GetAppliedSchemas():
            if schema.startswith("SemanticsLabelsAPI:"):
                current_taxonomy = schema.split(":", 1)[-1]
                if taxonomy is None or current_taxonomy == taxonomy:
                    labels_attribute = UsdSemantics.LabelsAPI(target_prim, current_taxonomy).GetLabelsAttr()
                    if labels_attribute:
                        labels_attribute.Set([item for item in labels_attribute.Get() if item not in labels])

    labels = set([labels] if isinstance(labels, str) else labels)
    prim = stage_utils.get_current_stage(backend="usd").GetPrimAtPath(prim) if isinstance(prim, str) else prim
    if include_descendants:
        for p in Usd.PrimRange(prim):
            _remove_labels(p)
    else:
        _remove_labels(prim)


def remove_all_labels(
    prim: str | Usd.Prim, *, remove_taxonomies: bool = False, include_descendants: bool = False
) -> None:
    """Remove all semantic labels from a prim.

    Backends: :guilabel:`usd`.

    This function removes all labels from a prim (and optionally all its descendants).
    To remove specific labels, use the :py:func:`remove_labels` function instead.

    Args:
        prim: Prim path or prim instance.
        remove_taxonomies: Whether to remove the taxonomies (instance names) along with the labels.
        include_descendants: Whether to remove labels from all descendants of the prim.

    Examples:

    .. code-block:: python

        >>> import isaacsim.core.experimental.utils.semantics as semantics_utils
        >>> import isaacsim.core.experimental.utils.stage as stage_utils
        >>>
        >>> stage_utils.define_prim("/World/Cube", "Cube")  # doctest: +NO_CHECK
        >>>
        >>> # add some labels to the default 'class' and 'custom' taxonomies
        >>> semantics_utils.add_labels("/World/Cube", labels=["label_a", "label_b"])
        >>> semantics_utils.add_labels("/World/Cube", labels=["label_c"], taxonomy="custom")
        >>>
        >>> # remove all labels
        >>> semantics_utils.remove_all_labels("/World/Cube")
        >>> semantics_utils.get_labels("/World/Cube")
        {'class': [], 'custom': []}
        >>>
        >>> # remove all labels and taxonomies
        >>> semantics_utils.remove_all_labels("/World/Cube", remove_taxonomies=True)
        >>> semantics_utils.get_labels("/World/Cube")
        {}
    """

    def _remove_all_labels(target_prim: Usd.Prim):
        for schema in target_prim.GetAppliedSchemas():
            if schema.startswith("SemanticsLabelsAPI:"):
                current_taxonomy = schema.split(":", 1)[-1]
                if remove_taxonomies:
                    target_prim.RemoveAPI(UsdSemantics.LabelsAPI, current_taxonomy)
                else:
                    labels_attribute = UsdSemantics.LabelsAPI(target_prim, current_taxonomy).GetLabelsAttr()
                    if labels_attribute:
                        labels_attribute.Set([])

    prim = stage_utils.get_current_stage(backend="usd").GetPrimAtPath(prim) if isinstance(prim, str) else prim
    if include_descendants:
        for p in Usd.PrimRange(prim):
            _remove_all_labels(p)
    else:
        _remove_all_labels(prim)
