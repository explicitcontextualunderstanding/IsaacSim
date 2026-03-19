# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Browser model implementation for organizing and managing examples in the Isaac Sim Examples Browser."""


import os
from typing import List, Optional

import carb.settings
import omni.kit.commands
import omni.usd
from omni.kit.browser.core import CategoryItem, CollectionItem, DetailItem
from omni.kit.browser.folder.core import FileSystemFolder, TreeFolderBrowserModel
from pxr import Sdf, Tf, Usd

SETTING_FOLDER = "/exts/isaacsim.examples.browser/folders"


class Example:
    """Represents an example that can be registered and displayed in the Isaac Sim Examples Browser.

    This class encapsulates the metadata and functionality for an individual example, including its display
    name, execution logic, UI customization, categorization, and visual representation. Examples are
    organized by category and can be executed through the browser interface.

    Args:
        name: Display name of the example.
        execute_entrypoint: Callable function that executes the example's main logic.
        ui_hook: Callable function for custom UI behavior or modifications.
        category: Category path for organizing the example in the browser hierarchy.
        thumbnail: File path to the example's thumbnail image for visual identification.
    """

    def __init__(
        self,
        name: str = "",
        execute_entrypoint: callable = None,
        ui_hook: callable = None,
        category: str = "",
        thumbnail: Optional[str] = None,
    ):
        self.name = name
        self.category = category if category else "General"
        self.execute_entrypoint = execute_entrypoint
        self.ui_hook = ui_hook
        self.thumbnail = thumbnail


class ExampleCategoryItem(CategoryItem):
    """A category item for organizing examples in the Isaac Sim examples browser.

    This class represents a hierarchical category node that can contain child categories and examples.
    It extends the browser core CategoryItem to provide example-specific functionality for managing
    hierarchical category structures in the examples browser interface.

    Args:
        name: The name of the category.
    """

    def __init__(self, name: str):
        super().__init__(name)

    def add_child(self, child_name: str):
        """Adds a child category item to the current category.

        Creates a new child category with the specified name if it doesn't already exist,
        or returns the existing child category if it does.

        Args:
            child_name: Name of the child category to add.

        Returns:
            The child category item that was created or already existed.
        """
        if child_name not in [c.name for c in self.children]:
            child_category = ExampleCategoryItem(child_name)
            self.children.append(child_category)
            self.count += 1
            child_category.parent = self  # add self to the child's parent
        else:
            child_category = self.children[[c.name for c in self.children].index(child_name)]

        return child_category


class ExampleDetailItem(DetailItem):
    """A detail item representing an individual example in the Isaac Sim examples browser.

    This class extends the browser framework's DetailItem to display example-specific information including
    name, thumbnail, and UI hooks. It serves as the visual representation of an Example object within the
    browser's detail view, enabling users to interact with and execute individual examples.

    Args:
        example: The Example object containing the example's metadata and execution details.
    """

    def __init__(self, example: Example):
        super().__init__(example.name, "", example.thumbnail)
        self.example = example
        self.ui_hook = example.ui_hook


class ExampleBrowserModel(TreeFolderBrowserModel):
    """Represent asset browser model

    Args:
        *args: Variable length argument list passed to the parent class.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(self, *args, **kwargs):
        settings = carb.settings.get_settings()
        self._examples = {}
        super().__init__(
            *args,
            setting_folders=SETTING_FOLDER,
            show_category_subfolders=True,
            hide_file_without_thumbnails=False,
            show_summary_folder=True,
            **kwargs,
        )

    def register_example(self, **kwargs):
        """Registers a new example in the browser.

        Args:
            **kwargs: Example properties including name, execute_entrypoint, ui_hook, category, and thumbnail.
        """
        example = Example(**kwargs)
        if not example.name:
            return
        if example.category not in self._examples:
            self._examples[example.category] = []

        # check if there are already an example with the same name
        if example.name in [e.name for e in self._examples[example.category]]:
            carb.log_warn(f"Example with name {example.name} already exists in category {example.category}.")
        self._examples[example.category].append(ExampleDetailItem(example))
        self.refresh_browser()
        return

    def get_category_items(self, item: CollectionItem) -> List[CategoryItem]:
        """Override to get list of category items

        Args:
            item: Collection item to get categories for.

        Returns:
            List of category items organized hierarchically.
        """
        category_items = []
        for category in self._examples:
            categories = category.split("/")

            if categories[0] not in [c.name for c in category_items]:  # if the category is not already in the list
                current_category = ExampleCategoryItem(categories[0])
                category_items.append(current_category)
            else:
                current_category = category_items[[c.name for c in category_items].index(categories[0])]

            if len(categories) > 1:
                for i in range(1, len(categories)):
                    if categories[i] not in [c.name for c in current_category.children]:
                        child_category = current_category.add_child(categories[i])
                        current_category = child_category
                    else:
                        current_category = current_category.children[
                            [c.name for c in current_category.children].index(categories[i])
                        ]

        self.sort_items(category_items)
        return category_items

    def get_detail_items(self, item: ExampleCategoryItem) -> List[ExampleDetailItem]:
        """Override to get list of detail items

        Args:
            item: Category item to get examples for.

        Returns:
            List of example detail items in the category.
        """
        detail_items = []

        def lookup_category_name(item):
            key_name = item.name
            while item.parent:
                key_name = item.parent.name + "/" + key_name
                item = item.parent
            return key_name

        if item.name == self.SUMMARY_FOLDER_NAME:
            # List all files in sub folders
            for category, examples in self._examples.items:
                detail_items += examples
        else:
            lookup_name = lookup_category_name(item)
            if lookup_name in self._examples:
                detail_items += self._examples[lookup_name]

        self.sort_items(detail_items)
        return detail_items

    def execute(self, item: ExampleDetailItem):
        """Executes the selected example.

        Args:
            item: Example detail item to execute.
        """
        # Create a Reference or payload of the Props in the stage
        example = item.example
        if example.execute_entrypoint:
            example.execute_entrypoint()

    def deregister_example(self, name: str, category: str):
        """Removes an example from the browser.

        Args:
            name: Name of the example to remove.
            category: Category containing the example.
        """
        if category in self._examples:
            self._examples[category] = [e for e in self._examples[category] if e.name != name]
            if len(self._examples[category]) == 0:
                del self._examples[category]
        self.refresh_browser()

    def refresh_browser(self):
        """Refreshes the browser display to reflect current examples."""
        collections = self.get_item_children(None)
        if collections:
            self._item_changed(collections[0])
        else:
            self._item_changed(None)
        return
