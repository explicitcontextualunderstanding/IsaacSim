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


"""UI helper builders for the MJCF importer UI."""

from collections.abc import Callable

import omni.ui as ui
from omni.kit.window.filepicker import FilePickerDialog
from omni.kit.window.property.templates import LABEL_WIDTH

from .style import get_style


def add_line_rect_flourish(draw_line: bool = True) -> None:
    """Add a line and rectangle flourish after UI elements.

    Args:
        draw_line: Whether to draw the line alongside the rectangle.

    Example:

    .. code-block:: python

        >>> from isaacsim.asset.importer.mjcf.ui.impl import ui_utils
        >>> ui_utils.add_line_rect_flourish()  # doctest: +SKIP
    """
    if draw_line:
        ui.Line(style={"color": 0x338A8777}, width=ui.Fraction(1), alignment=ui.Alignment.CENTER)
    ui.Spacer(width=10)
    with ui.Frame(width=0):
        with ui.VStack():
            with ui.Placer(offset_x=0, offset_y=7):
                ui.Rectangle(height=5, width=5, alignment=ui.Alignment.CENTER)
    ui.Spacer(width=5)


def format_tt(tt: str) -> str:
    """Format a tooltip string with capitalization rules.

    Args:
        tt: Tooltip string to format.

    Returns:
        Formatted tooltip string.

    Example:

    .. code-block:: python

        >>> from isaacsim.asset.importer.mjcf.ui.impl import ui_utils
        >>> ui_utils.format_tt("hello world")
        'Hello World '
    """
    import string

    formated = ""
    i = 0
    for w in tt.split():
        if w.isupper():
            formated += w + " "
        elif len(w) > 3 or i == 0:
            formated += string.capwords(w) + " "
        else:
            formated += w.lower() + " "
        i += 1
    return formated


def add_folder_picker_icon(
    on_click_fn: Callable[[str, str], None],
    item_filter_fn: Callable[[str], bool] | None = None,
    bookmark_label: str | None = None,
    bookmark_path: str | None = None,
    dialog_title: str = "Select Output Folder",
    button_title: str = "Select Folder",
    size: int = 24,
) -> Callable[[], None]:
    """Add a folder picker icon button.

    Args:
        on_click_fn: Callback invoked with (filename, path) when a file is selected.
        item_filter_fn: Optional filter function for the file picker.
        bookmark_label: Optional bookmark label.
        bookmark_path: Optional bookmark path.
        dialog_title: Title for the file picker dialog.
        button_title: Title for the apply button.
        size: Size of the icon button in pixels.

    Returns:
        Callable that opens the file picker when invoked.

    Example:

    .. code-block:: python

        >>> from isaacsim.asset.importer.mjcf.ui.impl import ui_utils
        >>> ui_utils.add_folder_picker_icon(lambda a, b: None)  # doctest: +SKIP
        <...>
    """

    def open_file_picker() -> None:
        """Open the file picker dialog and wire callbacks."""

        def on_selected(filename: str, path: str) -> None:
            """Handle file picker selection.

            Args:
                filename: Selected filename.
                path: Selected directory path.
            """
            on_click_fn(filename, path)
            file_picker.hide()

        def on_canceled(a: object, b: object) -> None:
            """Handle file picker cancellation.

            Args:
                a: Unused callback argument.
                b: Unused callback argument.
            """
            file_picker.hide()

        file_picker = FilePickerDialog(
            dialog_title,
            allow_multi_selection=False,
            apply_button_label=button_title,
            click_apply_handler=lambda a, b: on_selected(a, b),
            click_cancel_handler=lambda a, b: on_canceled(a, b),
            item_filter_fn=item_filter_fn,
            enable_versioning_pane=True,
        )
        if bookmark_label and bookmark_path:
            file_picker.toggle_bookmark_from_path(bookmark_label, bookmark_path, True)

    with ui.Frame(width=0, tooltip=button_title):
        ui.Button(
            name="IconButton",
            width=size,
            height=size,
            clicked_fn=open_file_picker,
            style=get_style()["IconButton.Image::FolderPicker"],
            alignment=ui.Alignment.RIGHT_CENTER,
        )

    # Return the function so it can be called from other contexts (e.g., clicking text field)
    return open_file_picker


def dropdown_builder(
    label: str = "",
    type: str = "dropdown",
    default_val: int = 0,
    items: list[str] | None = None,
    tooltip: str = "",
    on_clicked_fn: Callable[[str], None] | None = None,
    identifier: str | None = None,
) -> ui.AbstractItemModel:
    """Create a styled dropdown combobox and return its model.

    Args:
        label: Label to the left of the UI element.
        type: Type of UI element.
        default_val: Default index of dropdown items.
        items: List of items for the dropdown.
        tooltip: Tooltip to display over the label.
        on_clicked_fn: Call-back function when clicked.
        identifier: Optional identifier to simplify UI queries.

    Returns:
        Combo box model.

    Example:

    .. code-block:: python

        >>> from isaacsim.asset.importer.mjcf.ui.impl import ui_utils
        >>> model = ui_utils.dropdown_builder(items=["A", "B"])  # doctest: +SKIP
        >>> model is not None
        True
    """
    if items is None:
        items = ["Option 1", "Option 2", "Option 3"]
    with ui.HStack():
        ui.Label(label, width=LABEL_WIDTH, alignment=ui.Alignment.LEFT_CENTER, tooltip=format_tt(tooltip))
        combo_box_widget = ui.ComboBox(
            default_val,
            *items,
            name="ComboBox",
            width=ui.Fraction(1),
            alignment=ui.Alignment.LEFT_CENTER,
            identifier=identifier,
        )
        combo_box = combo_box_widget.model
        add_line_rect_flourish(False)

        def on_clicked_wrapper(model: ui.AbstractItemModel, val: object) -> None:
            """Forward selected item text to the callback.

            Args:
                model: Combo box model that triggered the change.
                val: Changed value payload.
            """
            if on_clicked_fn is None:
                return
            on_clicked_fn(items[model.get_item_value_model().as_int])

        if on_clicked_fn is not None:
            combo_box.add_item_changed_fn(on_clicked_wrapper)

    return combo_box


def checkbox_builder(
    label: str = "",
    type: str = "checkbox",
    default_val: bool = False,
    tooltip: str = "",
    on_clicked_fn: Callable[[bool], None] | None = None,
    identifier: str | None = None,
) -> ui.SimpleBoolModel:
    """Create a styled checkbox and return its model.

    Args:
        label: Label to the left of the UI element.
        type: Type of UI element.
        default_val: Initial state of the checkbox.
        tooltip: Tooltip to display over the label.
        on_clicked_fn: Call-back function when clicked.
        identifier: Optional identifier to simplify UI queries.

    Returns:
        Checkbox model.

    Example:

    .. code-block:: python

        >>> from isaacsim.asset.importer.mjcf.ui.impl import ui_utils
        >>> model = ui_utils.checkbox_builder()  # doctest: +SKIP
        >>> model is not None
        True
    """
    with ui.HStack():
        check_box = ui.CheckBox(width=10, height=0, identifier=identifier)
        ui.Spacer(width=8)
        check_box.model.set_value(default_val)

        def on_click(value_model: ui.AbstractValueModel) -> None:
            """Forward checkbox changes to the callback.

            Args:
                value_model: Value model for the checkbox.
            """
            if on_clicked_fn is None:
                return
            on_clicked_fn(value_model.get_value_as_bool())

        if on_clicked_fn:
            check_box.model.add_value_changed_fn(on_click)
        ui.Label(label, width=0, height=0, tooltip=tooltip)
        return check_box.model


def string_filed_builder(
    default_val: str = " ",
    tooltip: str = "",
    read_only: bool = False,
    item_filter_fn: Callable[[str], bool] | None = None,
    folder_dialog_title: str = "Select Output Folder",
    folder_button_title: str = "Select Folder",
    identifier: str | None = None,
) -> ui.AbstractValueModel:
    """Create a styled string field widget.

    Args:
        default_val: Text to initialize in the string field.
        tooltip: Tooltip to display over the UI elements.
        read_only: Prevent editing when True.
        item_filter_fn: Filter function to pass to the file picker.
        folder_dialog_title: Title for the file picker dialog.
        folder_button_title: Title for the file picker apply button.
        identifier: Optional identifier to simplify UI queries.

    Returns:
        String field model.

    Example:

    .. code-block:: python

        >>> from isaacsim.asset.importer.mjcf.ui.impl import ui_utils
        >>> model = ui_utils.string_filed_builder()  # doctest: +SKIP
        >>> model is not None
        True
    """
    with ui.HStack():
        str_field = ui.StringField(
            name="StringField",
            tooltip=format_tt(tooltip),
            width=ui.Fraction(1),
            height=0,
            alignment=ui.Alignment.LEFT_CENTER,
            read_only=read_only,
            identifier=identifier,
        )
        str_field.enabled = False
        str_field.model.set_value(default_val)

        def update_field(filename: str, path: str) -> None:
            """Update the string field with the chosen path.

            Args:
                filename: Selected filename.
                path: Selected directory path.
            """
            if filename == "":
                val = path
            elif filename[0] != "/" and path[-1] != "/":
                val = path + "/" + filename
            elif filename[0] == "/" and path[-1] == "/":
                val = path + filename[1:]
            else:
                val = path + filename
            str_field.model.set_value(val)

        ui.Spacer(width=4)
        file_pick_fn = add_folder_picker_icon(
            update_field, item_filter_fn, dialog_title=folder_dialog_title, button_title=folder_button_title, size=16
        )
        ui.Spacer(width=2)
        str_field.set_mouse_pressed_fn(lambda a, b, c, d: file_pick_fn())
        return str_field.model
