# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""UI utility helpers for the URDF importer."""

import os
import subprocess
import sys
import typing
from cmath import inf

import carb
import carb.settings
import omni.appwindow
import omni.ext
import omni.ui as ui
from omni.kit.window.extensions import SimpleCheckBox
from omni.kit.window.filepicker import FilePickerDialog
from omni.kit.window.property.templates import LABEL_HEIGHT, LABEL_WIDTH

from .style import BUTTON_WIDTH, COLOR_W, COLOR_X, COLOR_Y, COLOR_Z, get_option_style, get_style


def on_copy_to_clipboard(to_copy: str) -> None:
    """Copy text to the system clipboard.

    Args:
        to_copy: Text to copy.
    """
    try:
        import pyperclip
    except ImportError:
        carb.log_warn("Could not import pyperclip.")
        return
    try:
        pyperclip.copy(to_copy)
    except pyperclip.PyperclipException:
        carb.log_warn(pyperclip.EXCEPT_MSG)
        return


def on_open_ide_clicked(ext_path: str, file_path: str) -> None:
    """Open the current directory and file in VSCode.

    Args:
        ext_path: Extension root path.
        file_path: Source file path to open.
    """
    if sys.platform == "win32":
        try:
            subprocess.Popen(["code", os.path.abspath(ext_path), os.path.abspath(file_path)], shell=True)
        except Exception:
            carb.log_warn(
                "Could not open in VSCode. See Troubleshooting help here: https://code.visualstudio.com/docs/editor/command-line#_common-questions"
            )
    else:
        try:
            subprocess.run(["code", ext_path, file_path], check=True)
        except Exception:
            carb.log_warn(
                "Could not open in VSCode. See Troubleshooting help here: https://code.visualstudio.com/docs/editor/command-line#_common-questions"
            )


on_open_IDE_clicked = on_open_ide_clicked


def on_open_folder_clicked(file_path: str) -> None:
    """Open the current directory in a file browser.

    Args:
        file_path: Source file path to reveal.
    """
    if sys.platform == "win32":
        try:
            subprocess.Popen(["start", os.path.abspath(os.path.dirname(file_path))], shell=True)
        except OSError:
            carb.log_warn("Could not open file browser.")
    else:
        try:
            subprocess.run(["xdg-open", os.path.abspath(file_path.rpartition("/")[0])], check=True)
        except Exception:
            carb.log_warn("could not open file browser")


def on_docs_link_clicked(doc_link: str) -> None:
    """Open the extension documentation in a web browser.

    Args:
        doc_link: Documentation URL to open.
    """
    import webbrowser

    try:
        webbrowser.open(doc_link, new=2)
    except Exception as e:
        carb.log_warn(f"Could not open browswer with url: {doc_link}, {e}")


def btn_builder(
    label: str = "",
    type: str = "button",
    text: str = "button",
    tooltip: str = "",
    on_clicked_fn: typing.Callable | None = None,
) -> ui.Button:
    """Creates a stylized button.

    Args:
        label: Label to the left of the UI element. Defaults to "".
        type: Type of UI element. Defaults to "button".
        text: Text rendered on the button. Defaults to "button".
        tooltip: Tooltip to display over the label. Defaults to "".
        on_clicked_fn: Callback function when clicked. Defaults to None.

    Returns:
        Button instance.
    """
    with ui.HStack():
        # ui.Label(label, width=LABEL_WIDTH, alignment=ui.Alignment.LEFT_CENTER, tooltip=format_tt(tooltip))
        ui.Spacer()
        btn = ui.Button(
            text.upper(),
            name="Button",
            width=BUTTON_WIDTH,
            clicked_fn=on_clicked_fn,
            style=get_style(),
            alignment=ui.Alignment.LEFT_CENTER,
        )
        # ui.Spacer(width=5)
        # add_line_rect_flourish(True)
        # ui.Spacer(width=ui.Fraction(1))
        # ui.Spacer(width=10)
        # with ui.Frame(width=0):
        #     with ui.VStack():
        #         with ui.Placer(offset_x=0, offset_y=7):
        #             ui.Rectangle(height=5, width=5, alignment=ui.Alignment.CENTER)
        # ui.Spacer(width=5)
    return btn


def state_btn_builder(
    label: str = "",
    type: str = "state_button",
    a_text: str = "STATE A",
    b_text: str = "STATE B",
    tooltip: str = "",
    on_clicked_fn: typing.Callable[[bool], None] | None = None,
) -> ui.Button:
    """Creates a State Change Button that changes text when pressed.

    Args:
        label: Label to the left of the UI element. Defaults to "".
        type: Type of UI element. Defaults to "button".
        a_text: Text rendered on the button for State A. Defaults to "STATE A".
        b_text: Text rendered on the button for State B. Defaults to "STATE B".
        tooltip: Tooltip to display over the label. Defaults to "".
        on_clicked_fn: Callback function when clicked. Defaults to None.

    Returns:
        Button instance.
    """
    callback = on_clicked_fn or (lambda *_: None)

    def toggle():
        if btn.text == a_text.upper():
            btn.text = b_text.upper()
            callback(True)
        else:
            btn.text = a_text.upper()
            callback(False)

    with ui.HStack():
        ui.Label(label, width=LABEL_WIDTH, alignment=ui.Alignment.LEFT_CENTER, tooltip=format_tt(tooltip))
        btn = ui.Button(
            a_text.upper(),
            name="Button",
            width=BUTTON_WIDTH,
            clicked_fn=toggle,
            style=get_style(),
            alignment=ui.Alignment.LEFT_CENTER,
        )
        ui.Spacer(width=5)
        # add_line_rect_flourish(False)
        ui.Spacer(width=ui.Fraction(1))
        ui.Spacer(width=10)
        with ui.Frame(width=0):
            with ui.VStack():
                with ui.Placer(offset_x=0, offset_y=7):
                    ui.Rectangle(height=5, width=5, alignment=ui.Alignment.CENTER)
        ui.Spacer(width=5)
    return btn


def cb_builder(
    label: str = "",
    type: str = "checkbox",
    default_val: bool = False,
    tooltip: str = "",
    on_clicked_fn: typing.Callable[[bool], None] | None = None,
) -> ui.SimpleBoolModel:
    """Creates a stylized checkbox.

    Args:
        label: Label to the left of the UI element. Defaults to "".
        type: Type of UI element. Defaults to "checkbox".
        default_val: Checked is True, Unchecked is False. Defaults to False.
        tooltip: Tooltip to display over the label. Defaults to "".
        on_clicked_fn: Callback function when clicked. Defaults to None.

    Returns:
        Checkbox value model.
    """
    with ui.HStack():
        ui.Label(
            label,
            width=ui.Fraction(1),
            elided_text=True,
            alignment=ui.Alignment.LEFT_CENTER,
            tooltip=format_tt(tooltip),
        )
        model = ui.SimpleBoolModel()
        callable = on_clicked_fn
        if callable is None:
            callable = lambda x: None
        SimpleCheckBox(default_val, callable, model=model)

        add_line_rect_flourish()
        return model


def multi_btn_builder(
    label: str = "",
    type: str = "multi_button",
    count: int = 2,
    text: list[str] | None = None,
    tooltip: list[str] | None = None,
    on_clicked_fn: list[typing.Callable | None] | None = None,
) -> list[ui.Button]:
    """Creates a row of stylized buttons.

    Args:
        label: Label to the left of the UI element. Defaults to "".
        type: Type of UI element. Defaults to "multi_button".
        count: Number of UI elements to create. Defaults to 2.
        text: List of text rendered on the UI elements. Defaults to ["button", "button"].
        tooltip: List of tooltips to display over the UI elements. Defaults to ["", "", ""].
        on_clicked_fn: List of callback functions when clicked. Defaults to [None, None].

    Returns:
        List of button instances.
    """
    if text is None:
        text = ["button"] * count
    if tooltip is None:
        tooltip = [""] * (count + 1)
    if on_clicked_fn is None:
        on_clicked_fn = [None] * count

    btns = []
    with ui.HStack():
        ui.Label(label, width=LABEL_WIDTH, alignment=ui.Alignment.LEFT_CENTER, tooltip=format_tt(tooltip[0]))
        for i in range(count):
            btn = ui.Button(
                text[i].upper(),
                name="Button",
                width=BUTTON_WIDTH,
                clicked_fn=on_clicked_fn[i],
                tooltip=format_tt(tooltip[i + 1]),
                style=get_style(),
                alignment=ui.Alignment.LEFT_CENTER,
            )
            btns.append(btn)
            if i < count:
                ui.Spacer(width=5)
        add_line_rect_flourish()
    return btns


def multi_cb_builder(
    label: str = "",
    type: str = "multi_checkbox",
    count: int = 2,
    text: list[str] | None = None,
    default_val: list[bool] | None = None,
    tooltip: list[str] | None = None,
    on_clicked_fn: list[typing.Callable | None] | None = None,
) -> list[ui.SimpleBoolModel]:
    """Creates a Row of Stylized Checkboxes.

    Args:
        label: Label to the left of the UI element. Defaults to "".
        type: Type of UI element. Defaults to "multi_checkbox".
        count: Number of UI elements to create. Defaults to 2.
        text: List of text rendered on the UI elements. Defaults to [" ", " "].
        default_val: List of default values. Checked is True, unchecked is False.
        tooltip: List of tooltips to display over the UI elements. Defaults to ["", "", ""].
        on_clicked_fn: List of callback functions when clicked. Defaults to [None, None].

    Returns:
        List of checkbox value models.
    """
    if text is None:
        text = [" "] * count
    if default_val is None:
        default_val = [False] * count
    if tooltip is None:
        tooltip = [""] * (count + 1)
    if on_clicked_fn is None:
        on_clicked_fn = [None] * count

    cbs = []
    with ui.HStack():
        ui.Label(label, width=LABEL_WIDTH - 12, alignment=ui.Alignment.LEFT_CENTER, tooltip=format_tt(tooltip[0]))
        for i in range(count):
            cb = ui.SimpleBoolModel(default_value=default_val[i])
            callable = on_clicked_fn[i]
            if callable is None:
                callable = lambda x: None
            SimpleCheckBox(default_val[i], callable, model=cb)
            ui.Label(
                text[i], width=BUTTON_WIDTH / 2, alignment=ui.Alignment.LEFT_CENTER, tooltip=format_tt(tooltip[i + 1])
            )
            if i < count - 1:
                ui.Spacer(width=5)
            cbs.append(cb)
        add_line_rect_flourish()
    return cbs


def str_builder(
    label: str = "",
    type: str = "stringfield",
    default_val: str = " ",
    tooltip: str = "",
    on_clicked_fn: typing.Callable | None = None,
    use_folder_picker: bool = False,
    read_only: bool = False,
    item_filter_fn: typing.Callable | None = None,
    bookmark_label: str | None = None,
    bookmark_path: str | None = None,
    folder_dialog_title: str = "Select Output Folder",
    folder_button_title: str = "Select Folder",
    style: dict | None = None,
    identifier: str | None = None,
) -> ui.AbstractValueModel:
    """Creates a stylized string field widget.

    Args:
        label: Label to the left of the UI element. Defaults to "".
        type: Type of UI element. Defaults to "stringfield".
        default_val: Text to initialize in Stringfield. Defaults to " ".
        tooltip: Tooltip to display over the UI elements. Defaults to "".
        on_clicked_fn: Callback invoked when the field is clicked. Defaults to None.
        use_folder_picker: Add a folder picker button to the right. Defaults to False.
        read_only: Prevents editing. Defaults to False.
        item_filter_fn: Filter function to pass to the FilePicker.
        bookmark_label: Bookmark label to pass to the FilePicker.
        bookmark_path: Bookmark path to pass to the FilePicker.
        folder_dialog_title: Title for the folder picker dialog.
        folder_button_title: Label for the folder picker button.
        style: Optional style overrides for the string field.
        identifier: Optional identifier to simplify UI queries.

    Returns:
        Model for the string field.
    """
    with ui.HStack():
        ui.Label(label, width=LABEL_WIDTH, alignment=ui.Alignment.LEFT_CENTER, tooltip=format_tt(tooltip))
        str_field = ui.StringField(
            name="StringField",
            width=ui.Fraction(1),
            height=0,
            alignment=ui.Alignment.LEFT_CENTER,
            read_only=read_only,
            identifier=identifier,
        )
        str_field.model.set_value(default_val)

        if use_folder_picker:

            def update_field(filename, path):
                if filename == "":
                    val = path
                elif filename[0] != "/" and path[-1] != "/":
                    val = path + "/" + filename
                elif filename[0] == "/" and path[-1] == "/":
                    val = path + filename[1:]
                else:
                    val = path + filename
                str_field.model.set_value(val)

            file_pick_fn = add_folder_picker_icon(
                update_field,
                item_filter_fn,
                bookmark_label,
                bookmark_path,
                dialog_title=folder_dialog_title,
                button_title=folder_button_title,
            )
            str_field.set_mouse_pressed_fn(file_pick_fn)
        else:
            add_line_rect_flourish(False)
        return str_field.model


def checkbox_builder(
    label: str = "",
    type: str = "checkbox",
    default_val: bool = False,
    tooltip: str = "",
    on_clicked_fn: typing.Callable[[bool], None] | None = None,
    identifier: str | None = None,
) -> ui.SimpleBoolModel:
    """Creates a stylized checkbox.

    Args:
        label: Label to the left of the UI element. Defaults to "".
        type: Type of UI element. Defaults to "checkbox".
        default_val: Checked is True, Unchecked is False. Defaults to False.
        tooltip: Tooltip to display over the label. Defaults to "".
        on_clicked_fn: Callback function when clicked. Defaults to None.
        identifier: Optional identifier to simplify UI queries.

    Returns:
        Checkbox value model.
    """
    with ui.HStack():
        check_box = ui.CheckBox(width=10, height=0, identifier=identifier)
        ui.Spacer(width=8)
        check_box.model.set_value(default_val)

        callback = on_clicked_fn
        if callback is not None:

            def on_click(value_model, cb=callback):
                cb(value_model.get_value_as_bool())

            check_box.model.add_value_changed_fn(on_click)
        ui.Label(label, width=0, height=0, tooltip=tooltip)
        return check_box.model


def float_field_builder(
    label: str = "",
    default_val: float = 0,
    tooltip: str = "",
    format: str = "%.2f",
) -> ui.AbstractValueModel:
    """Creates a stylized float field widget.

    Args:
        label: Label to the left of the UI element. Defaults to "".
        default_val: Default value of the UI element. Defaults to 0.
        tooltip: Tooltip to display over the UI elements. Defaults to "".
        format: Format string for the float field. Defaults to "%.2f".

    Returns:
        Value model for the float field.
    """
    with ui.HStack(spacing=10, style=get_option_style()):
        ui.Label(label, width=ui.Fraction(0.5), alignment=ui.Alignment.LEFT_CENTER, tooltip=format_tt(tooltip))
        with ui.ZStack():
            float_field = ui.FloatField(
                name="FloatField",
                width=ui.Fraction(1),
                height=0,
                alignment=ui.Alignment.LEFT,
                format=format,
            ).model
            float_field.set_value(default_val)
            with ui.HStack():
                ui.Spacer()
                ui.Label("Kg/m", name="density", alignment=ui.Alignment.RIGHT_CENTER)
                ui.Label("3", name="exponent", alignment=ui.Alignment.RIGHT_TOP, width=0)
                ui.Spacer(width=1)
        return float_field


def string_filed_builder(
    default_val: str = " ",
    tooltip: str = "",
    read_only: bool = False,
    item_filter_fn: typing.Callable | None = None,
    folder_dialog_title: str = "Select Output Folder",
    folder_button_title: str = "Select Folder",
    bookmark_label: str = "",
    bookmark_path: str = "",
    use_folder_picker: bool = True,
    identifier: str | None = None,
) -> ui.AbstractValueModel:
    """Creates a stylized string field widget.

    Args:
        default_val: Text to initialize in Stringfield. Defaults to " ".
        tooltip: Tooltip to display over the UI elements. Defaults to "".
        read_only: Prevents editing. Defaults to False.
        item_filter_fn: Filter function to pass to the FilePicker.
        folder_dialog_title: Title for the folder picker dialog.
        folder_button_title: Label for the folder picker button.
        bookmark_label: Bookmark label to pass to the FilePicker.
        bookmark_path: Bookmark path to pass to the FilePicker.
        use_folder_picker: Whether to show the folder picker button.
        identifier: Optional identifier to simplify UI queries.

    Returns:
        Model for the string field.
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
        if use_folder_picker:

            def update_field(filename, path):
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
                update_field,
                item_filter_fn,
                dialog_title=folder_dialog_title,
                button_title=folder_button_title,
                bookmark_label=bookmark_label,
                bookmark_path=bookmark_path,
                size=16,
            )
            ui.Spacer(width=2)
            str_field.set_mouse_pressed_fn(lambda a, b, c, d: file_pick_fn())
        return str_field.model


def int_builder(
    label: str = "",
    type: str = "intfield",
    default_val: int = 0,
    tooltip: str = "",
    min: int = sys.maxsize * -1,
    max: int = sys.maxsize,
) -> ui.AbstractValueModel:
    """Creates a stylized int field widget.

    Args:
        label: Label to the left of the UI element. Defaults to "".
        type: Type of UI element. Defaults to "intfield".
        default_val: Default value of the UI element. Defaults to 0.
        tooltip: Tooltip to display over the UI elements. Defaults to "".
        min: Minimum limit for int field. Defaults to sys.maxsize * -1.
        max: Maximum limit for int field. Defaults to sys.maxsize.

    Returns:
        Value model for the int field.
    """
    with ui.HStack():
        ui.Label(label, width=LABEL_WIDTH, alignment=ui.Alignment.LEFT_CENTER, tooltip=format_tt(tooltip))
        int_field = ui.IntDrag(
            name="Field", height=LABEL_HEIGHT, min=min, max=max, alignment=ui.Alignment.LEFT_CENTER
        ).model
        int_field.set_value(default_val)
        add_line_rect_flourish(False)
    return int_field


def float_builder(
    label: str = "",
    type: str = "floatfield",
    default_val: float = 0,
    tooltip: str = "",
    min: float = -inf,
    max: float = inf,
    step: float = 0.1,
    format: str = "%.2f",
) -> ui.AbstractValueModel:
    """Creates a stylized float field widget.

    Args:
        label: Label to the left of the UI element. Defaults to "".
        type: Type of UI element. Defaults to "floatfield".
        default_val: Default value of the UI element. Defaults to 0.
        tooltip: Tooltip to display over the UI elements. Defaults to "".
        min: Minimum float value. Defaults to -inf.
        max: Maximum float value. Defaults to inf.
        step: Step size for the drag control. Defaults to 0.1.
        format: Format string for the float field. Defaults to "%.2f".

    Returns:
        Value model for the float field.
    """
    with ui.HStack():
        ui.Label(label, width=LABEL_WIDTH, alignment=ui.Alignment.LEFT_CENTER, tooltip=format_tt(tooltip))
        float_field = ui.FloatDrag(
            name="FloatField",
            width=ui.Fraction(1),
            height=0,
            alignment=ui.Alignment.LEFT_CENTER,
            min=min,
            max=max,
            step=step,
            format=format,
        ).model
        float_field.set_value(default_val)
        add_line_rect_flourish(False)
        return float_field


def combo_cb_str_builder(
    label: str = "",
    type: str = "checkbox_stringfield",
    default_val: list | None = None,
    tooltip: str = "",
    on_clicked_fn: typing.Callable[[bool], None] | None = None,
    use_folder_picker: bool = False,
    read_only: bool = False,
    folder_dialog_title: str = "Select Output Folder",
    folder_button_title: str = "Select Folder",
) -> tuple[ui.SimpleBoolModel, ui.AbstractValueModel]:
    """Creates a stylized checkbox and string field widget.

    Args:
        label: Label to the left of the UI element. Defaults to "".
        type: Type of UI element. Defaults to "checkbox_stringfield".
        default_val: Text to initialize in Stringfield. Defaults to [False, " "].
        tooltip: Tooltip to display over the UI elements. Defaults to "".
        on_clicked_fn: Checkbox callback function. Defaults to a no-op.
        use_folder_picker: Add a folder picker button to the right. Defaults to False.
        read_only: Prevents editing. Defaults to False.
        folder_dialog_title: Title for the folder picker dialog.
        folder_button_title: Label for the folder picker button.

    Returns:
        Checkbox model and string field model.
    """
    if default_val is None:
        default_val = [False, " "]
    if on_clicked_fn is None:
        on_clicked_fn = lambda x: None

    with ui.HStack():
        ui.Label(label, width=LABEL_WIDTH - 12, alignment=ui.Alignment.LEFT_CENTER, tooltip=format_tt(tooltip))
        cb = ui.SimpleBoolModel(default_value=default_val[0])
        SimpleCheckBox(default_val[0], on_clicked_fn, model=cb)
        str_field = ui.StringField(
            name="StringField", width=ui.Fraction(1), height=0, alignment=ui.Alignment.LEFT_CENTER, read_only=read_only
        ).model
        str_field.set_value(default_val[1])

        if use_folder_picker:

            def update_field(filename, path):
                if filename == "":
                    val = path
                elif filename[0] != "/" and path[-1] != "/":
                    val = path + "/" + filename
                elif filename[0] == "/" and path[-1] == "/":
                    val = path + filename[1:]
                else:
                    val = path + filename
                str_field.set_value(val)

            add_folder_picker_icon(update_field, dialog_title=folder_dialog_title, button_title=folder_button_title)
        else:
            add_line_rect_flourish(False)
        return cb, str_field


def dropdown_builder(
    label: str = "",
    type: str = "dropdown",
    default_val: int = 0,
    items: list[str] | None = None,
    tooltip: str = "",
    on_clicked_fn: typing.Callable | None = None,
    identifier: str | None = None,
) -> ui.AbstractItemModel:
    """Creates a stylized dropdown combobox.

    Args:
        label: Label to the left of the UI element. Defaults to "".
        type: Type of UI element. Defaults to "dropdown".
        default_val: Default index of dropdown items. Defaults to 0.
        items: List of items for dropdown box. Defaults to ["Option 1", "Option 2", "Option 3"].
        tooltip: Tooltip to display over the label. Defaults to "".
        on_clicked_fn: Callback function when clicked. Defaults to None.
        identifier: Optional identifier to simplify UI queries.

    Returns:
        Combo box model.
    """
    if items is None:
        items = ["Option 1", "Option 2", "Option 3"]

    with ui.HStack():
        ui.Label(label, width=LABEL_WIDTH, alignment=ui.Alignment.LEFT_CENTER, tooltip=format_tt(tooltip))
        combo_box = ui.ComboBox(
            default_val,
            *items,
            name="ComboBox",
            width=ui.Fraction(1),
            alignment=ui.Alignment.LEFT_CENTER,
            identifier=identifier,
        ).model
        add_line_rect_flourish(False)

        callback = on_clicked_fn
        if callback is not None:

            def on_clicked_wrapper(model, val, cb=callback):
                cb(items[model.get_item_value_model().as_int])

            combo_box.add_item_changed_fn(on_clicked_wrapper)

    return combo_box


def combo_intfield_slider_builder(
    label: str = "",
    type: str = "intfield_stringfield",
    default_val: float = 0.5,
    min: float = 0,
    max: float = 1,
    step: float = 0.01,
    tooltip: list[str] | None = None,
) -> tuple[ui.AbstractValueModel, ui.IntSlider]:
    """Creates a stylized int field and slider widget.

    Args:
        label: Label to the left of the UI element. Defaults to "".
        type: Type of UI element. Defaults to "intfield_stringfield".
        default_val: Default value. Defaults to 0.5.
        min: Minimum value. Defaults to 0.
        max: Maximum value. Defaults to 1.
        step: Step size. Defaults to 0.01.
        tooltip: List of tooltips. Defaults to ["", ""].

    Returns:
        Field model and slider widget.
    """
    if tooltip is None:
        tooltip = ["", ""]

    with ui.HStack():
        ui.Label(label, width=LABEL_WIDTH, alignment=ui.Alignment.LEFT_CENTER, tooltip=format_tt(tooltip[0]))
        ff = ui.IntDrag(
            name="Field", width=BUTTON_WIDTH / 2, alignment=ui.Alignment.LEFT_CENTER, tooltip=format_tt(tooltip[1])
        ).model
        ff.set_value(default_val)
        ui.Spacer(width=5)
        fs = ui.IntSlider(
            width=ui.Fraction(1), alignment=ui.Alignment.LEFT_CENTER, min=min, max=max, step=step, model=ff
        )

        add_line_rect_flourish(False)
        return ff, fs


def combo_floatfield_slider_builder(
    label: str = "",
    type: str = "floatfield_stringfield",
    default_val: float = 0.5,
    min: float = 0,
    max: float = 1,
    step: float = 0.01,
    tooltip: list[str] | None = None,
) -> tuple[ui.AbstractValueModel, ui.FloatSlider]:
    """Creates a stylized float field and slider widget.

    Args:
        label: Label to the left of the UI element. Defaults to "".
        type: Type of UI element. Defaults to "floatfield_stringfield".
        default_val: Default value. Defaults to 0.5.
        min: Minimum value. Defaults to 0.
        max: Maximum value. Defaults to 1.
        step: Step size. Defaults to 0.01.
        tooltip: List of tooltips. Defaults to ["", ""].

    Returns:
        Field model and slider widget.
    """
    if tooltip is None:
        tooltip = ["", ""]

    with ui.HStack():
        ui.Label(label, width=LABEL_WIDTH, alignment=ui.Alignment.LEFT_CENTER, tooltip=format_tt(tooltip[0]))
        ff = ui.FloatField(
            name="Field", width=BUTTON_WIDTH / 2, alignment=ui.Alignment.LEFT_CENTER, tooltip=format_tt(tooltip[1])
        ).model
        ff.set_value(default_val)
        ui.Spacer(width=5)
        fs = ui.FloatSlider(
            width=ui.Fraction(1), alignment=ui.Alignment.LEFT_CENTER, min=min, max=max, step=step, model=ff
        )

        add_line_rect_flourish(False)
        return ff, fs


def multi_dropdown_builder(
    label: str = "",
    type: str = "multi_dropdown",
    count: int = 2,
    default_val: list[int] | None = None,
    items: list[list[str]] | None = None,
    tooltip: str = "",
    on_clicked_fn: list[typing.Callable | None] | None = None,
) -> list[ui.ComboBox]:
    """Creates a stylized multi-dropdown combobox.

    Args:
        label: Label to the left of the UI element. Defaults to "".
        type: Type of UI element. Defaults to "multi_dropdown".
        count: Number of UI elements. Defaults to 2.
        default_val: List of default indices of dropdown items. Defaults to [0, 0].
        items: List of item lists for dropdown boxes.
        tooltip: Tooltip to display over the label. Defaults to "".
        on_clicked_fn: List of callback functions when clicked. Defaults to [None, None].

    Returns:
        List of combo box widgets.
    """
    if default_val is None:
        default_val = [0] * count
    if items is None:
        default_items = ["Option 1", "Option 2", "Option 3"]
        items = [default_items[:] for _ in range(count)]
    if on_clicked_fn is None:
        on_clicked_fn = [None] * count

    elems = []
    with ui.HStack():
        ui.Label(label, width=LABEL_WIDTH, alignment=ui.Alignment.LEFT_CENTER, tooltip=format_tt(tooltip))
        for i in range(count):
            elem = ui.ComboBox(
                default_val[i], *items[i], name="ComboBox", width=ui.Fraction(1), alignment=ui.Alignment.LEFT_CENTER
            )

            def on_clicked_wrapper(model, val, index):
                callback = on_clicked_fn[index]
                if callback is not None:
                    callback(items[index][model.get_item_value_model().as_int])

            elem.model.add_item_changed_fn(lambda m, v, index=i: on_clicked_wrapper(m, v, index))
            elems.append(elem)
            if i < count - 1:
                ui.Spacer(width=5)
        add_line_rect_flourish(False)
        return elems


def combo_cb_dropdown_builder(
    label: str = "",
    type: str = "checkbox_dropdown",
    default_val: list | None = None,
    items: list[str] | None = None,
    tooltip: str = "",
    on_clicked_fn: list[typing.Callable | None] | None = None,
) -> tuple[ui.SimpleBoolModel, ui.ComboBox]:
    """Creates a stylized dropdown combobox with an enable checkbox.

    Args:
        label: Label to the left of the UI element. Defaults to "".
        type: Type of UI element. Defaults to "checkbox_dropdown".
        default_val: List of checkbox and dropdown defaults. Defaults to [False, 0].
        items: List of items for dropdown box. Defaults to ["Option 1", "Option 2", "Option 3"].
        tooltip: Tooltip to display over the label. Defaults to "".
        on_clicked_fn: List of callback functions. Defaults to [lambda x: None, None].

    Returns:
        Checkbox model and combobox widget.
    """
    if default_val is None:
        default_val = [False, 0]
    if items is None:
        items = ["Option 1", "Option 2", "Option 3"]
    if on_clicked_fn is None:
        on_clicked_fn = [None, None]

    with ui.HStack():
        ui.Label(label, width=LABEL_WIDTH - 12, alignment=ui.Alignment.LEFT_CENTER, tooltip=format_tt(tooltip))
        cb = ui.SimpleBoolModel(default_value=default_val[0])
        cb_callback = on_clicked_fn[0] or (lambda *_: None)
        SimpleCheckBox(default_val[0], cb_callback, model=cb)
        combo_box = ui.ComboBox(
            default_val[1], *items, name="ComboBox", width=ui.Fraction(1), alignment=ui.Alignment.LEFT_CENTER
        )

        def on_clicked_wrapper(model, val):
            callback = on_clicked_fn[1]
            if callback is not None:
                callback(items[model.get_item_value_model().as_int])

        combo_box.model.add_item_changed_fn(on_clicked_wrapper)

        add_line_rect_flourish(False)

        return cb, combo_box


def scrolling_frame_builder(
    label: str = "",
    type: str = "scrolling_frame",
    default_val: str = "No Data",
    tooltip: str = "",
) -> ui.Label:
    """Creates a labeled scrolling frame with copy-to-clipboard button.

    Args:
        label: Label to the left of the UI element. Defaults to "".
        type: Type of UI element. Defaults to "scrolling_frame".
        default_val: Default text. Defaults to "No Data".
        tooltip: Tooltip to display over the label. Defaults to "".

    Returns:
        Label widget used for the scrolling content.
    """
    with ui.VStack(style=get_style(), spacing=5):
        with ui.HStack():
            ui.Label(label, width=LABEL_WIDTH, alignment=ui.Alignment.LEFT_TOP, tooltip=format_tt(tooltip))
            with ui.ScrollingFrame(
                height=LABEL_HEIGHT * 5,
                style_type_name_override="ScrollingFrame",
                alignment=ui.Alignment.LEFT_TOP,
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
            ):
                text = ui.Label(
                    default_val,
                    style_type_name_override="Label::label",
                    word_wrap=True,
                    alignment=ui.Alignment.LEFT_TOP,
                )
            with ui.Frame(width=0, tooltip="Copy To Clipboard"):
                ui.Button(
                    name="IconButton",
                    width=20,
                    height=20,
                    clicked_fn=lambda: on_copy_to_clipboard(to_copy=text.text),
                    style=get_style()["IconButton.Image::CopyToClipboard"],
                    alignment=ui.Alignment.RIGHT_TOP,
                )
    return text


def combo_cb_scrolling_frame_builder(
    label: str = "",
    type: str = "cb_scrolling_frame",
    default_val: list | None = None,
    tooltip: str = "",
    on_clicked_fn: typing.Callable[[bool], None] | None = None,
) -> tuple[ui.SimpleBoolModel, ui.Label]:
    """Creates a labeled, checkbox-enabled scrolling frame with copy-to-clipboard button.

    Args:
        label: Label to the left of the UI element. Defaults to "".
        type: Type of UI element. Defaults to "cb_scrolling_frame".
        default_val: List of checkbox and frame defaults. Defaults to [False, "No Data"].
        tooltip: Tooltip to display over the label. Defaults to "".
        on_clicked_fn: Callback function when clicked. Defaults to a no-op.

    Returns:
        Checkbox model and label widget.
    """
    if default_val is None:
        default_val = [False, "No Data"]
    if on_clicked_fn is None:
        on_clicked_fn = lambda x: None

    with ui.VStack(style=get_style(), spacing=5):
        with ui.HStack():
            ui.Label(label, width=LABEL_WIDTH - 12, alignment=ui.Alignment.LEFT_TOP, tooltip=format_tt(tooltip))
            with ui.VStack(width=0):
                cb = ui.SimpleBoolModel(default_value=default_val[0])
                SimpleCheckBox(default_val[0], on_clicked_fn, model=cb)
                ui.Spacer(height=18 * 4)
            with ui.ScrollingFrame(
                height=18 * 5,
                style_type_name_override="ScrollingFrame",
                alignment=ui.Alignment.LEFT_TOP,
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
            ):
                text = ui.Label(
                    default_val[1],
                    style_type_name_override="Label::label",
                    word_wrap=True,
                    alignment=ui.Alignment.LEFT_TOP,
                )

            with ui.Frame(width=0, tooltip="Copy to Clipboard"):
                ui.Button(
                    name="IconButton",
                    width=20,
                    height=20,
                    clicked_fn=lambda: on_copy_to_clipboard(to_copy=text.text),
                    style=get_style()["IconButton.Image::CopyToClipboard"],
                    alignment=ui.Alignment.RIGHT_TOP,
                )
    return cb, text


def xyz_builder(
    label: str = "",
    tooltip: str = "",
    axis_count: int = 3,
    default_val: list[float] | None = None,
    min: float = float("-inf"),
    max: float = float("inf"),
    step: float = 0.001,
    on_value_changed_fn: list[typing.Callable | None] | None = None,
) -> list[ui.AbstractValueModel]:
    """Create a multi-axis float drag widget with X, Y, Z, W labels.

    Args:
        label: Label to the left of the UI element. Defaults to "".
        tooltip: Tooltip text for the widget. Defaults to "".
        axis_count: Number of axes to display (1-4). Defaults to 3.
        default_val: List of default values. Defaults to [0.0, 0.0, 0.0, 0.0].
        min: Minimum float value. Defaults to float("-inf").
        max: Maximum float value. Defaults to float("inf").
        step: Drag step size. Defaults to 0.001.
        on_value_changed_fn: List of callback functions for each axis. Defaults to [None, None, None, None].

    Returns:
        List of value models for each axis.
    """
    if default_val is None:
        default_val = [0.0, 0.0, 0.0, 0.0]
    if on_value_changed_fn is None:
        on_value_changed_fn = [None, None, None, None]

    # These styles & colors are taken from omni.kit.property.transform_builder.py _create_multi_float_drag_matrix_with_labels
    if axis_count <= 0 or axis_count > 4:
        import builtins

        carb.log_warn("Invalid axis_count: must be in range 1 to 4. Clamping to default range.")
        axis_count = builtins.max(builtins.min(axis_count, 4), 1)

    field_labels = [("X", COLOR_X), ("Y", COLOR_Y), ("Z", COLOR_Z), ("W", COLOR_W)]
    field_tooltips = ["X Value", "Y Value", "Z Value", "W Value"]
    RECT_WIDTH = 13
    # SPACING = 4
    val_models: list[ui.AbstractValueModel] = []
    with ui.HStack():
        ui.Label(
            label,
            width=ui.Fraction(0.5),
            elided_text=True,
            alignment=ui.Alignment.LEFT_CENTER,
            tooltip=format_tt(tooltip),
        )
        with ui.ZStack():
            with ui.HStack():
                ui.Spacer(width=RECT_WIDTH)
                for i in range(axis_count):
                    value_model = ui.FloatDrag(
                        name="Field",
                        height=LABEL_HEIGHT,
                        min=min,
                        max=max,
                        step=step,
                        alignment=ui.Alignment.LEFT_CENTER,
                        tooltip=field_tooltips[i],
                    ).model
                    val_models.append(value_model)
                    value_model.set_value(default_val[i])
                    if on_value_changed_fn[i] is not None:
                        value_model.add_value_changed_fn(on_value_changed_fn[i])
                    if i != axis_count - 1:
                        ui.Spacer(width=19)
            with ui.HStack():
                for i in range(axis_count):
                    if i != 0:
                        ui.Spacer()  # width=BUTTON_WIDTH - 1)
                    field_label = field_labels[i]
                    with ui.ZStack(width=RECT_WIDTH + 2 * i):
                        ui.Rectangle(name="vector_label", style={"background_color": field_label[1]})
                        ui.Label(field_label[0], name="vector_label", alignment=ui.Alignment.CENTER)
                ui.Spacer()
        add_line_rect_flourish(False)
        return val_models


def color_picker_builder(
    label: str = "",
    type: str = "color_picker",
    default_val: list[float] | None = None,
    tooltip: str = "Color Picker",
) -> ui.AbstractItemModel:
    """Creates a color picker widget.

    Args:
        label: Label to the left of the UI element. Defaults to "".
        type: Type of UI element. Defaults to "color_picker".
        default_val: List of (R,G,B,A) default values.
        tooltip: Tooltip to display over the label. Defaults to "Color Picker".

    Returns:
        Color widget model.
    """
    if default_val is None:
        default_val = [1.0, 1.0, 1.0, 1.0]

    with ui.HStack():
        ui.Label(label, width=LABEL_WIDTH, alignment=ui.Alignment.LEFT_CENTER, tooltip=format_tt(tooltip))
        model = ui.ColorWidget(*default_val, width=BUTTON_WIDTH).model
        ui.Spacer(width=5)
        add_line_rect_flourish()
    return model


def progress_bar_builder(
    label: str = "",
    type: str = "progress_bar",
    default_val: int = 0,
    tooltip: str = "Progress",
) -> ui.AbstractValueModel:
    """Creates a progress bar widget.

    Args:
        label: Label to the left of the UI element. Defaults to "".
        type: Type of UI element. Defaults to "progress_bar".
        default_val: Starting value. Defaults to 0.
        tooltip: Tooltip to display over the label. Defaults to "Progress".

    Returns:
        Progress bar model.
    """
    with ui.HStack():
        ui.Label(label, width=LABEL_WIDTH, alignment=ui.Alignment.LEFT_CENTER)
        model = ui.ProgressBar().model
        model.set_value(default_val)
        add_line_rect_flourish(False)
    return model


def plot_builder(
    label: str = "",
    data: list[float] | None = None,
    min: float = -1,
    max: float = 1,
    type: ui.Type = ui.Type.LINE,
    value_stride: int = 1,
    color: int | None = None,
    tooltip: str = "",
) -> ui.Plot:
    """Creates a stylized static plot.

    Args:
        label: Label to the left of the UI element. Defaults to "".
        data: Data to plot. Defaults to None.
        min: Minimum Y value. Defaults to -1.
        max: Maximum Y value. Defaults to 1.
        type: Plot type. Defaults to ui.Type.LINE.
        value_stride: Width of plot stride. Defaults to 1.
        color: Plot color. Defaults to None.
        tooltip: Tooltip to display over the label. Defaults to "".

    Returns:
        Plot widget instance.
    """
    with ui.VStack(spacing=5):
        with ui.HStack():
            ui.Label(label, width=LABEL_WIDTH, alignment=ui.Alignment.LEFT_TOP, tooltip=format_tt(tooltip))

            plot_height = LABEL_HEIGHT * 2 + 13
            plot_width = ui.Fraction(1)
            with ui.ZStack():
                ui.Rectangle(width=plot_width, height=plot_height)
                if not color:
                    color = 0xFFDDDDDD
                plot = ui.Plot(
                    type,
                    min,
                    max,
                    *data,
                    value_stride=value_stride,
                    width=plot_width,
                    height=plot_height,
                    style={"color": color, "background_color": 0x0},
                )

            def update_min(model):
                plot.scale_min = model.as_float

            def update_max(model):
                plot.scale_max = model.as_float

            ui.Spacer(width=5)
            with ui.Frame(width=0):
                with ui.VStack(spacing=5):
                    max_model = ui.FloatDrag(
                        name="Field", width=40, alignment=ui.Alignment.LEFT_BOTTOM, tooltip="Max"
                    ).model
                    max_model.set_value(max)
                    min_model = ui.FloatDrag(
                        name="Field", width=40, alignment=ui.Alignment.LEFT_TOP, tooltip="Min"
                    ).model
                    min_model.set_value(min)

                    min_model.add_value_changed_fn(update_min)
                    max_model.add_value_changed_fn(update_max)

            ui.Spacer(width=20)
        add_separator()

    return plot


def xyz_plot_builder(
    label: str = "",
    data: list[list[float]] | None = None,
    min: float = -1,
    max: float = 1,
    tooltip: str = "",
) -> list[ui.Plot]:
    """Creates a stylized static XYZ plot.

    Args:
        label: Label to the left of the UI element. Defaults to "".
        data: Data to plot. Defaults to [].
        min: Minimum Y value. Defaults to -1.
        max: Maximum Y value. Defaults to 1.
        tooltip: Tooltip to display over the label. Defaults to "".

    Returns:
        List of X, Y, and Z plot widgets.
    """
    if data is None:
        data = [[], [], []]

    with ui.VStack(spacing=5):
        with ui.HStack():
            ui.Label(label, width=LABEL_WIDTH, alignment=ui.Alignment.LEFT_TOP, tooltip=format_tt(tooltip))

            plot_height = LABEL_HEIGHT * 2 + 13
            plot_width = ui.Fraction(1)
            with ui.ZStack():
                ui.Rectangle(width=plot_width, height=plot_height)

                plot_0 = ui.Plot(
                    ui.Type.LINE,
                    min,
                    max,
                    *data[0],
                    width=plot_width,
                    height=plot_height,
                    style=get_style()["PlotLabel::X"],
                )
                plot_1 = ui.Plot(
                    ui.Type.LINE,
                    min,
                    max,
                    *data[1],
                    width=plot_width,
                    height=plot_height,
                    style=get_style()["PlotLabel::Y"],
                )
                plot_2 = ui.Plot(
                    ui.Type.LINE,
                    min,
                    max,
                    *data[2],
                    width=plot_width,
                    height=plot_height,
                    style=get_style()["PlotLabel::Z"],
                )

            def update_min(model):
                plot_0.scale_min = model.as_float
                plot_1.scale_min = model.as_float
                plot_2.scale_min = model.as_float

            def update_max(model):
                plot_0.scale_max = model.as_float
                plot_1.scale_max = model.as_float
                plot_2.scale_max = model.as_float

            ui.Spacer(width=5)
            with ui.Frame(width=0):
                with ui.VStack(spacing=5):
                    max_model = ui.FloatDrag(
                        name="Field", width=40, alignment=ui.Alignment.LEFT_BOTTOM, tooltip="Max"
                    ).model
                    max_model.set_value(max)
                    min_model = ui.FloatDrag(
                        name="Field", width=40, alignment=ui.Alignment.LEFT_TOP, tooltip="Min"
                    ).model
                    min_model.set_value(min)

                    min_model.add_value_changed_fn(update_min)
                    max_model.add_value_changed_fn(update_max)
            ui.Spacer(width=20)

        add_separator()
        return [plot_0, plot_1, plot_2]


def combo_cb_plot_builder(
    label: str = "",
    default_val: bool = False,
    on_clicked_fn: typing.Callable[[bool], None] | None = None,
    data: list[float] | None = None,
    min: float = -1,
    max: float = 1,
    type: ui.Type = ui.Type.LINE,
    value_stride: int = 1,
    color: int | None = None,
    tooltip: str = "",
) -> tuple[ui.SimpleBoolModel, ui.Plot]:
    """Creates a checkbox-enabled dynamic plot.

    Args:
        label: Label to the left of the UI element. Defaults to "".
        default_val: Checkbox default. Defaults to False.
        on_clicked_fn: Checkbox callback function. Defaults to a no-op.
        data: Data to plot. Defaults to None.
        min: Minimum Y value. Defaults to -1.
        max: Maximum Y value. Defaults to 1.
        type: Plot type. Defaults to ui.Type.LINE.
        value_stride: Width of plot stride. Defaults to 1.
        color: Plot color. Defaults to None.
        tooltip: Tooltip to display over the label. Defaults to "".


    Returns:
        Checkbox model and plot widget.
    """
    if on_clicked_fn is None:
        on_clicked_fn = lambda x: None

    with ui.VStack(spacing=5):
        with ui.HStack():
            # Label
            ui.Label(label, width=LABEL_WIDTH, alignment=ui.Alignment.LEFT_TOP, tooltip=format_tt(tooltip))
            # Checkbox
            with ui.Frame(width=0):
                with ui.Placer(offset_x=-10, offset_y=0):
                    with ui.VStack():
                        SimpleCheckBox(default_val, on_clicked_fn)
                        ui.Spacer(height=ui.Fraction(1))
                        ui.Spacer()
            # Plot
            plot_height = LABEL_HEIGHT * 2 + 13
            plot_width = ui.Fraction(1)
            with ui.ZStack():
                ui.Rectangle(width=plot_width, height=plot_height)
                if not color:
                    color = 0xFFDDDDDD
                plot = ui.Plot(
                    type,
                    min,
                    max,
                    *data,
                    value_stride=value_stride,
                    width=plot_width,
                    height=plot_height,
                    style={"color": color, "background_color": 0x0},
                )

            # Min/Max Helpers
            def update_min(model):
                plot.scale_min = model.as_float

            def update_max(model):
                plot.scale_max = model.as_float

            ui.Spacer(width=5)
            with ui.Frame(width=0):
                with ui.VStack(spacing=5):
                    # Min/Max Fields
                    max_model = ui.FloatDrag(
                        name="Field", width=40, alignment=ui.Alignment.LEFT_BOTTOM, tooltip="Max"
                    ).model
                    max_model.set_value(max)
                    min_model = ui.FloatDrag(
                        name="Field", width=40, alignment=ui.Alignment.LEFT_TOP, tooltip="Min"
                    ).model
                    min_model.set_value(min)

                    min_model.add_value_changed_fn(update_min)
                    max_model.add_value_changed_fn(update_max)
            ui.Spacer(width=20)
        with ui.HStack():
            ui.Spacer(width=LABEL_WIDTH + 29)
            # Current Value Field (disabled by default)
            val_model = ui.FloatDrag(
                name="Field",
                width=BUTTON_WIDTH,
                height=LABEL_HEIGHT,
                enabled=False,
                alignment=ui.Alignment.LEFT_CENTER,
                tooltip="Value",
            ).model
        add_separator()
    return plot, val_model


def combo_cb_xyz_plot_builder(
    label: str = "",
    default_val: bool = False,
    on_clicked_fn: typing.Callable[[bool], None] | None = None,
    data: list[list[float]] | None = None,
    min: float = -1,
    max: float = 1,
    type: ui.Type = ui.Type.LINE,
    value_stride: int = 1,
    tooltip: str = "",
) -> tuple[list[ui.Plot], list[ui.AbstractValueModel]]:
    """Create a checkbox-enabled XYZ plot widget.

    Args:
        label: Label to the left of the UI element. Defaults to "".
        default_val: Checkbox default state. Defaults to False.
        on_clicked_fn: Checkbox callback function. Defaults to lambda x: None.
        data: Data to plot for each axis. Defaults to [].
        min: Minimum Y value. Defaults to -1.
        max: Maximum Y value. Defaults to 1.
        type: Plot type. Defaults to ui.Type.LINE.
        value_stride: Width of plot stride. Defaults to 1.
        tooltip: Tooltip to display over the Label. Defaults to "".

    Returns:
        Tuple of plot list and value model list.
    """
    if on_clicked_fn is None:
        on_clicked_fn = lambda x: None
    if data is None:
        data = [[], [], []]

    with ui.VStack(spacing=5):
        with ui.HStack():
            ui.Label(label, width=LABEL_WIDTH, alignment=ui.Alignment.LEFT_TOP, tooltip=format_tt(tooltip))
            # Checkbox
            with ui.Frame(width=0):
                with ui.Placer(offset_x=-10, offset_y=0):
                    with ui.VStack():
                        SimpleCheckBox(default_val, on_clicked_fn)
                        ui.Spacer(height=ui.Fraction(1))
                        ui.Spacer()
            # Plots
            plot_height = LABEL_HEIGHT * 2 + 13
            plot_width = ui.Fraction(1)
            with ui.ZStack():
                ui.Rectangle(width=plot_width, height=plot_height)

                plot_0 = ui.Plot(
                    type,
                    min,
                    max,
                    *data[0],
                    value_stride=value_stride,
                    width=plot_width,
                    height=plot_height,
                    style=get_style()["PlotLabel::X"],
                )
                plot_1 = ui.Plot(
                    type,
                    min,
                    max,
                    *data[1],
                    value_stride=value_stride,
                    width=plot_width,
                    height=plot_height,
                    style=get_style()["PlotLabel::Y"],
                )
                plot_2 = ui.Plot(
                    type,
                    min,
                    max,
                    *data[2],
                    value_stride=value_stride,
                    width=plot_width,
                    height=plot_height,
                    style=get_style()["PlotLabel::Z"],
                )

            def update_min(model):
                plot_0.scale_min = model.as_float
                plot_1.scale_min = model.as_float
                plot_2.scale_min = model.as_float

            def update_max(model):
                plot_0.scale_max = model.as_float
                plot_1.scale_max = model.as_float
                plot_2.scale_max = model.as_float

            ui.Spacer(width=5)
            with ui.Frame(width=0):
                with ui.VStack(spacing=5):
                    max_model = ui.FloatDrag(
                        name="Field", width=40, alignment=ui.Alignment.LEFT_BOTTOM, tooltip="Max"
                    ).model
                    max_model.set_value(max)
                    min_model = ui.FloatDrag(
                        name="Field", width=40, alignment=ui.Alignment.LEFT_TOP, tooltip="Min"
                    ).model
                    min_model.set_value(min)

                    min_model.add_value_changed_fn(update_min)
                    max_model.add_value_changed_fn(update_max)
            ui.Spacer(width=20)

        # with ui.HStack():
        #     ui.Spacer(width=40)
        #     val_models = xyz_builder()#**{"args":args})

        field_labels = [("X", COLOR_X), ("Y", COLOR_Y), ("Z", COLOR_Z), ("W", COLOR_W)]
        RECT_WIDTH = 13
        # SPACING = 4
        with ui.HStack():
            ui.Spacer(width=LABEL_WIDTH + 29)

            with ui.ZStack():
                with ui.HStack():
                    ui.Spacer(width=RECT_WIDTH)
                    # value_widget = ui.MultiFloatDragField(
                    #     *args, name="multivalue", min=min, max=max, step=step, h_spacing=RECT_WIDTH + SPACING, v_spacing=2
                    # ).model
                    val_model_x = ui.FloatDrag(
                        name="Field",
                        width=BUTTON_WIDTH - 5,
                        height=LABEL_HEIGHT,
                        enabled=False,
                        alignment=ui.Alignment.LEFT_CENTER,
                        tooltip="X Value",
                    ).model
                    ui.Spacer(width=19)
                    val_model_y = ui.FloatDrag(
                        name="Field",
                        width=BUTTON_WIDTH - 5,
                        height=LABEL_HEIGHT,
                        enabled=False,
                        alignment=ui.Alignment.LEFT_CENTER,
                        tooltip="Y Value",
                    ).model
                    ui.Spacer(width=19)
                    val_model_z = ui.FloatDrag(
                        name="Field",
                        width=BUTTON_WIDTH - 5,
                        height=LABEL_HEIGHT,
                        enabled=False,
                        alignment=ui.Alignment.LEFT_CENTER,
                        tooltip="Z Value",
                    ).model
                with ui.HStack():
                    for i in range(3):
                        if i != 0:
                            ui.Spacer(width=BUTTON_WIDTH - 1)
                        field_label = field_labels[i]
                        with ui.ZStack(width=RECT_WIDTH + 1):
                            ui.Rectangle(name="vector_label", style={"background_color": field_label[1]})
                            ui.Label(field_label[0], name="vector_label", alignment=ui.Alignment.CENTER)

        add_separator()
        return [plot_0, plot_1, plot_2], [val_model_x, val_model_y, val_model_z]


def add_line_rect_flourish(draw_line: bool = True) -> None:
    """Aesthetic element that adds a Line + Rectangle after all UI elements in the row.

    Args:
        draw_line: Set false to only draw rectangle. Defaults to True.
    """
    if draw_line:
        ui.Line(style={"color": 0x338A8777}, width=ui.Fraction(1), alignment=ui.Alignment.CENTER)
    ui.Spacer(width=10)
    with ui.Frame(width=0):
        with ui.VStack():
            with ui.Placer(offset_x=0, offset_y=7):
                ui.Rectangle(height=5, width=5, alignment=ui.Alignment.CENTER)
    ui.Spacer(width=5)


def add_separator():
    """Aesthetic element to adds a Line Separator."""
    with ui.VStack(spacing=5):
        ui.Spacer()
        with ui.HStack():
            ui.Spacer(width=LABEL_WIDTH)
            ui.Line(style={"color": 0x338A8777}, width=ui.Fraction(1))
            ui.Spacer(width=20)
        ui.Spacer()


def add_folder_picker_icon(
    on_click_fn: typing.Callable[[str, str], None],
    item_filter_fn: typing.Callable | None = None,
    bookmark_label: str | None = None,
    bookmark_path: str | None = None,
    dialog_title: str = "Select Output Folder",
    button_title: str = "Select Folder",
    size: int = 24,
) -> typing.Callable[[], None]:
    """Add a folder picker icon button.

    Args:
        on_click_fn: Callback invoked with (filename, path) when a folder is selected.
        item_filter_fn: Optional filter function for the file picker.
        bookmark_label: Optional bookmark label to highlight.
        bookmark_path: Optional bookmark path to highlight.
        dialog_title: Title for the picker dialog.
        button_title: Label for the picker button.
        size: Icon button size in pixels.

    Returns:
        Callback to open the folder picker dialog.
    """

    def open_file_picker():
        def on_selected(filename, path):
            on_click_fn(filename, path)
            file_picker.hide()

        def on_canceled(a, b):
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

    with ui.VStack(width=0, tooltip=button_title):
        ui.Spacer()
        ui.Button(
            name="FolderPicker",
            style_type_name_override="IconButton",
            width=size,
            height=size,
            clicked_fn=open_file_picker,
            style=get_style(),
            alignment=ui.Alignment.RIGHT_CENTER,
        )
        ui.Spacer()

    return open_file_picker


def add_folder_picker_btn(on_click_fn: typing.Callable[[str, str], None]) -> None:
    """Add a folder picker button.

    Args:
        on_click_fn: Callback invoked with (filename, path) when a folder is selected.
    """

    def open_folder_picker():
        def on_selected(a, b):
            on_click_fn(a, b)
            folder_picker.hide()

        def on_canceled(a, b):
            folder_picker.hide()

        folder_picker = FilePickerDialog(
            "Select Output Folder",
            allow_multi_selection=False,
            apply_button_label="Select Folder",
            click_apply_handler=lambda a, b: on_selected(a, b),
            click_cancel_handler=lambda a, b: on_canceled(a, b),
        )

    with ui.Frame(width=0):
        ui.Button("SELECT", width=BUTTON_WIDTH, clicked_fn=open_folder_picker, tooltip="Select Folder")


def format_tt(tt: str) -> str:
    """Format tooltip text for display.

    Args:
        tt: Tooltip text to format.

    Returns:
        Formatted tooltip string.
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


def setup_ui_headers(
    ext_id: str,
    file_path: str,
    title: str = "My Custom Extension",
    doc_link: str = "https://docs.isaacsim.omniverse.nvidia.com/latest/index.html",
    overview: str = "",
) -> None:
    """Creates the Standard UI Elements at the top of each Isaac Extension.

    Args:
        ext_id: Extension ID.
        file_path: File path to source code.
        title: Name of extension. Defaults to "My Custom Extension".
        doc_link: Hyperlink to documentation.
        overview: Overview text explaining the extension. Defaults to "".
    """
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    extension_path = ext_manager.get_extension_path(ext_id)
    ext_path = os.path.dirname(extension_path) if os.path.isfile(extension_path) else extension_path
    build_header(ext_path, file_path, title, doc_link)
    build_info_frame(overview)


def build_header(
    ext_path: str,
    file_path: str,
    title: str = "My Custom Extension",
    doc_link: str = "https://docs.isaacsim.omniverse.nvidia.com/latest/index.html",
) -> None:
    """Title header with quick access utility buttons.

    Args:
        ext_path: Extension root path.
        file_path: Source file path to open.
        title: Header title text.
        doc_link: Documentation URL to open.
    """

    def build_icon_bar():
        """Add the utility buttons to the title header."""
        with ui.Frame(style=get_style(), width=0):
            with ui.VStack():
                with ui.HStack():
                    icon_size = 24
                    with ui.Frame(tooltip="Open Source Code"):
                        ui.Button(
                            name="IconButton",
                            width=icon_size,
                            height=icon_size,
                            clicked_fn=lambda: on_open_IDE_clicked(ext_path, file_path),
                            style=get_style()["IconButton.Image::OpenConfig"],
                            # style_type_name_override="IconButton.Image::OpenConfig",
                            alignment=ui.Alignment.LEFT_CENTER,
                            # tooltip="Open in IDE",
                        )
                    with ui.Frame(tooltip="Open Containing Folder"):
                        ui.Button(
                            name="IconButton",
                            width=icon_size,
                            height=icon_size,
                            clicked_fn=lambda: on_open_folder_clicked(file_path),
                            style=get_style()["IconButton.Image::OpenFolder"],
                            alignment=ui.Alignment.LEFT_CENTER,
                        )
                    with ui.Placer(offset_x=0, offset_y=3):
                        with ui.Frame(tooltip="Link to Docs"):
                            ui.Button(
                                name="IconButton",
                                width=icon_size - icon_size * 0.25,
                                height=icon_size - icon_size * 0.25,
                                clicked_fn=lambda: on_docs_link_clicked(doc_link),
                                style=get_style()["IconButton.Image::OpenLink"],
                                alignment=ui.Alignment.LEFT_TOP,
                            )

    with ui.ZStack():
        ui.Rectangle(style={"border_radius": 5})
        with ui.HStack():
            ui.Spacer(width=5)
            ui.Label(title, width=0, name="title", style={"font_size": 16})
            ui.Spacer(width=ui.Fraction(1))
            build_icon_bar()
            ui.Spacer(width=5)


def build_info_frame(overview: str = "") -> None:
    """Build an info frame with overview and metadata.

    Args:
        overview: Overview text to display.

    Returns:
        None.
    """
    frame = ui.CollapsableFrame(
        title="Information",
        height=0,
        collapsed=True,
        horizontal_clipping=False,
        style=get_style(),
        style_type_name_override="CollapsableFrame",
        horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
        vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
    )
    with frame:
        label = "Overview"
        default_val = overview
        tooltip = "Overview"
        with ui.VStack(style=get_style(), spacing=5):
            with ui.HStack():
                ui.Label(label, width=LABEL_WIDTH / 2, alignment=ui.Alignment.LEFT_TOP, tooltip=format_tt(tooltip))
                with ui.ScrollingFrame(
                    height=LABEL_HEIGHT * 5,
                    style_type_name_override="ScrollingFrame",
                    alignment=ui.Alignment.LEFT_TOP,
                    horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
                ):
                    text = ui.Label(
                        default_val,
                        style_type_name_override="Label::label",
                        word_wrap=True,
                        alignment=ui.Alignment.LEFT_TOP,
                    )
                with ui.Frame(width=0, tooltip="Copy To Clipboard"):
                    ui.Button(
                        name="IconButton",
                        width=20,
                        height=20,
                        clicked_fn=lambda: on_copy_to_clipboard(to_copy=text.text),
                        style=get_style()["IconButton.Image::CopyToClipboard"],
                        alignment=ui.Alignment.RIGHT_TOP,
                    )
    return


# def build_settings_frame(log_filename="extension.log", log_to_file=False, save_settings=False):
#     """Settings Frame for Common Utilities Functions"""
#     frame = ui.CollapsableFrame(
#         title="Settings",
#         height=0,
#         collapsed=True,
#         horizontal_clipping=False,
#         style=get_style(),
#         style_type_name_override="CollapsableFrame",
#     )

#     def on_log_to_file_enabled(val):
#         # TO DO
#         carb.log_info(f"Logging to {model.get_value_as_string()}:", val)

#     def on_save_out_settings(val):
#         # TO DO
#         carb.log_info("Save Out Settings?", val)


#     with frame:
#         with ui.VStack(style=get_style(), spacing=5):

#             # # Log to File Settings
#             # default_output_path = os.path.realpath(os.getcwd())
#             # kwargs = {
#             #     "label": "Log to File",
#             #     "type": "checkbox_stringfield",
#             #     "default_val": [log_to_file, default_output_path + "/" + log_filename],
#             #     "on_clicked_fn": on_log_to_file_enabled,
#             #     "tooltip": "Log Out to File",
#             #     "use_folder_picker": True,
#             # }
#             # model = combo_cb_str_builder(**kwargs)[1]

#             # Save Settings on Exit
#             # kwargs = {
#             #     "label": "Save Settings",
#             #     "type": "checkbox",
#             #     "default_val": save_settings,
#             #     "on_clicked_fn": on_save_out_settings,
#             #     "tooltip": "Save out GUI Settings on Exit.",
#             # }
#             # cb_builder(**kwargs)


class SearchListItem(ui.AbstractItem):
    """Item wrapper for search list entries.

    Args:
        text: Display text for the item.
    """

    def __init__(self, text: str) -> None:
        super().__init__()
        self.name_model = ui.SimpleStringModel(text)

    def __repr__(self) -> str:
        """Return a readable representation for debugging.

        Returns:
            String representation of the item.
        """
        return f'"{self.name_model.as_string}"'

    def name(self) -> str:
        """Return the display name of the item.

        Returns:
            Item display name.
        """
        return self.name_model.as_string


class SearchListItemModel(ui.AbstractItemModel):
    """Model for search list items.

    Args:
        *args: Item labels to include in the model.

    Example:

    .. code-block:: python

        string_list = ["Hello", "World"]
        model = SearchListItemModel(*string_list)
        ui.TreeView(model)
    """

    def __init__(self, *args: str) -> None:
        super().__init__()
        self._children = [SearchListItem(t) for t in args]
        self._filtered = [SearchListItem(t) for t in args]

    def get_item_children(self, item: SearchListItem | None):
        """Return children for the requested item.

        Args:
            item: Item whose children are requested.

        Returns:
            List of child items for the given item.
        """
        if item is not None:
            # Since we are doing a flat list, we return the children of root only.
            # If it's not root we return.
            return []

        return self._filtered

    def filter_text(self, text: str) -> None:
        """Filter items based on the provided text.

        Args:
            text: Filter text to apply.
        """
        import fnmatch

        self._filtered = []
        if len(text) == 0:
            for c in self._children:
                self._filtered.append(c)
        else:
            parts = text.split()
            # for i in range(len(parts) - 1, -1, -1):
            #     w = parts[i]

            leftover = " ".join(parts)
            if len(leftover) > 0:
                filter_str = f"*{leftover.lower()}*"
                for c in self._children:
                    if fnmatch.fnmatch(c.name().lower(), filter_str):
                        self._filtered.append(c)

        # This tells the Delegate to update the TreeView
        self._item_changed(None)

    def get_item_value_model_count(self, item: SearchListItem | None) -> int:
        """Return the number of columns.

        Args:
            item: Item being queried.

        Returns:
            Number of value models for the item.
        """
        return 1

    def get_item_value_model(self, item: SearchListItem, column_id: int) -> ui.AbstractValueModel:
        """Return the value model for the item.

        Args:
            item: Item being queried.
            column_id: Column index for the value model.

        Returns:
            Value model for the requested item.
        """
        return item.name_model


class SearchListItemDelegate(ui.AbstractItemDelegate):
    """Delegate that renders search list items.

    Args:
        on_double_click_fn: Optional double-click handler.
    """

    def __init__(self, on_double_click_fn: typing.Callable | None = None) -> None:
        super().__init__()
        self._on_double_click_fn = on_double_click_fn

    def build_branch(
        self,
        model: ui.AbstractItemModel,
        item: ui.AbstractItem,
        column_id: int,
        level: int,
        expanded: bool,
    ) -> None:
        """Create a branch widget that opens or closes subtree.

        Args:
            model: Item model for the tree view.
            item: Current item to render.
            column_id: Column index.
            level: Tree depth level.
            expanded: Whether the row is expanded.
        """

    def build_widget(
        self,
        model: ui.AbstractItemModel,
        item: ui.AbstractItem,
        column_id: int,
        level: int,
        expanded: bool,
    ) -> None:
        """Create a widget per column per item.

        Args:
            model: Item model for the tree view.
            item: Current item to render.
            column_id: Column index.
            level: Tree depth level.
            expanded: Whether the row is expanded.
        """
        stack = ui.ZStack(height=20, style=get_style())
        with stack:
            with ui.HStack():
                ui.Spacer(width=5)
                value_model = model.get_item_value_model(item, column_id)
                label = ui.Label(value_model.as_string, name="TreeView.Item")

        if not self._on_double_click_fn:
            self._on_double_click_fn = self.on_double_click

        # Set a double click function
        stack.set_mouse_double_clicked_fn(lambda x, y, b, m, l=label: self._on_double_click_fn(b, m, l))

    def on_double_click(self, button: int, model: ui.AbstractItemModel, label: ui.Label) -> None:
        """Handle double-click events on the tree view item.

        Args:
            button: Mouse button identifier.
            model: Item model for the tree view.
            label: Label widget that was clicked.

        Returns:
            None.
        """
        if button != 0:
            return


def build_simple_search(
    label: str = "",
    type: str = "search",
    model: ui.AbstractItemModel | None = None,
    delegate: ui.AbstractItemDelegate | None = None,
    tooltip: str = "",
) -> tuple["SearchWidget", ui.TreeView]:
    r"""Build a simple search bar and tree view.

    Pass a list of items through the model, and a custom on_click_fn through the delegate.
    Returns the SearchWidget so users can destroy it on shutdown.

    Args:
        label: Label to the left of the UI element. Defaults to "".
        type: Type of UI element. Defaults to "search".
        model: Item model for search. Defaults to None.
        delegate: Item delegate for search. Defaults to None.
        tooltip: Tooltip to display over the label. Defaults to "".

    Returns:
        Tuple(Search Widget, Treeview):

    Raises:
        ValueError: If model is None.
    """
    if model is None:
        raise ValueError("model is required for build_simple_search")

    if delegate is None:
        delegate = SearchListItemDelegate()
    with ui.HStack():
        ui.Label(label, width=LABEL_WIDTH, alignment=ui.Alignment.LEFT_TOP, tooltip=format_tt(tooltip))

        with ui.VStack(spacing=5):

            def filter_text(item):
                model.filter_text(item)

            from omni.kit.window.extensions.ext_components import SearchWidget

            search_bar = SearchWidget(filter_text)

            with ui.ScrollingFrame(
                height=LABEL_HEIGHT * 5,
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
                style=get_style(),
                style_type_name_override="TreeView.ScrollingFrame",
            ):
                treeview = ui.TreeView(
                    model,
                    delegate=delegate,
                    root_visible=False,
                    header_visible=False,
                    style={
                        "TreeView.ScrollingFrame": {"background_color": 0xFFE0E0E0},
                        "TreeView.Item": {"color": 0xFF535354, "font_size": 16},
                        "TreeView.Item:selected": {"color": 0xFF23211F},
                        "TreeView:selected": {"background_color": 0x409D905C},
                    },
                    # name="TreeView",
                    # style_type_name_override="TreeView",
                )
        add_line_rect_flourish(False)
    return search_bar, treeview


class RosPackageItem(ui.AbstractItem):
    """Row item for ROS package table entries.

    Args:
        name: Package name.
        path: Package path.
        row_index: Optional row index.
    """

    def __init__(self, name: str, path: str, row_index: int | None = None) -> None:
        super().__init__()
        self.name_model = ui.SimpleStringModel(name)
        self.path_model = ui.SimpleStringModel(path)
        self.row_index = row_index


class RosPackageModel(ui.AbstractItemModel):
    """Model for ROS package list entries.

    Args:
        rows: Optional list of (name, path) tuples.
    """

    def __init__(self, rows: list[tuple[str, str]] | None = None) -> None:
        super().__init__()
        self._items: list[RosPackageItem] = []
        self._row_index = 0
        if rows:
            for name, path in rows:
                self._items.append(RosPackageItem(name, path, self._row_index))
                self._row_index += 1

    def get_item_children(self, item: RosPackageItem | None) -> list[RosPackageItem]:
        """Return children for the requested item.

        Args:
            item: Item whose children are requested.

        Returns:
            List of child items for the requested item.
        """
        if item is not None:
            return []
        return self._items

    def get_item_value_model_count(self, item: RosPackageItem | None) -> int:
        """Return the number of value models per item.

        Args:
            item: Item being queried.

        Returns:
            Number of value models for the item.
        """
        return 2

    def get_item_value_model(self, item: RosPackageItem | None, column_id: int) -> ui.AbstractValueModel | None:
        """Return the value model for the given item and column.

        Args:
            item: Item whose value model is requested.
            column_id: Column index.

        Returns:
            Value model for the requested column, or None if item is missing.
        """
        if not item:
            return None
        if column_id == 0:
            return item.name_model
        return item.path_model

    def add_row(self, name: str = "", path: str = "") -> None:
        """Add a new row to the model.

        Args:
            name: Package name.
            path: Package path.
        """
        self._items.append(RosPackageItem(name, path, self._row_index))
        self._item_changed(None)

    def remove_row(self, item: RosPackageItem) -> None:
        """Remove a row from the model.

        Args:
            item: Row item to remove.
        """
        if item in self._items:
            self._items.remove(item)
            self._item_changed(None)

    def get_rows(self) -> list[tuple[str, str]]:
        """Return the list of package rows.

        Returns:
            List of (name, path) tuples.
        """
        return [(item.name_model.as_string, item.path_model.as_string) for item in self._items]


class RosPackageDelegate(ui.AbstractItemDelegate):
    """Delegate that renders ROS package table rows.

    Args:
        row_height: Row height in pixels.
        border_width: Border width in pixels.
        on_delete: Callback invoked when a row is deleted.
    """

    def __init__(self, row_height: int, border_width: int, on_delete: typing.Callable) -> None:
        super().__init__()
        self._row_height = row_height
        self._border_width = border_width
        self._on_delete = on_delete
        self._row_style = {
            "background_color": 0xFF111111,
            "border_color": 0xFF5A5A5A,
            "border_width": border_width,
        }

    def build_branch(
        self,
        model: ui.AbstractItemModel,
        item: ui.AbstractItem,
        column_id: int,
        level: int,
        expanded: bool,
    ) -> None:
        """Build the branch cell for tree rows.

        Args:
            model: Tree view model.
            item: Current item to render.
            column_id: Column index.
            level: Tree depth level.
            expanded: Whether the row is expanded.
        """

    def build_header(self, column_id: int) -> None:
        """Build the header cell for the given column.

        Args:
            column_id: Column index for the header.
        """
        label_text = ""
        if column_id == 0:
            label_text = "Package"
        else:
            label_text = "Path"

        row_height = self._row_height
        border = self._border_width
        header_model = ui.SimpleStringModel(label_text)

        with ui.ZStack(height=row_height):
            ui.Rectangle(height=row_height, style=self._row_style)
            with ui.HStack(height=row_height, spacing=0):
                ui.Spacer(width=border, height=border)
                with ui.VStack(width=ui.Fraction(1), height=row_height):
                    ui.Spacer(height=border)
                    ui.StringField(
                        header_model,
                        width=ui.Fraction(1),
                        height=row_height - border * 2,
                        alignment=ui.Alignment.LEFT_CENTER,
                        read_only=True,
                        enabled=False,
                        identifier=f"ros_package_table_header_{label_text}",
                    )
                    ui.Spacer(height=border)
                ui.Spacer(width=border, height=border)
            # Overlay a border so header widgets don't visually cover it.
            ui.Rectangle(
                height=row_height,
                style={"background_color": 0x0, "border_color": 0xFF5A5A5A, "border_width": border},
            )

    def build_widget(
        self,
        model: ui.AbstractItemModel,
        item: ui.AbstractItem,
        column_id: int,
        level: int,
        expanded: bool,
    ) -> None:
        """Build a widget for the given row and column.

        Args:
            model: Tree view model.
            item: Current item to render.
            column_id: Column index.
            level: Tree depth level.
            expanded: Whether the row is expanded.

        Returns:
            None.
        """
        if not item:
            return

        row_height = self._row_height
        border = self._border_width

        with ui.ZStack(height=row_height):
            ui.Rectangle(height=row_height, style=self._row_style)
            with ui.HStack(height=row_height, spacing=0):
                ui.Spacer(width=border, height=border)
                if column_id == 0:
                    with ui.VStack(width=ui.Fraction(1), height=row_height):
                        ui.Spacer(height=border)
                        value_model = model.get_item_value_model(item, column_id)
                        ui.StringField(
                            value_model,
                            width=ui.Fraction(1),
                            height=row_height - 10,
                            alignment=ui.Alignment.LEFT_CENTER,
                            identifier=f"ros_package_table_name_field_{item.row_index}",
                        )
                        ui.Spacer(height=border)
                else:
                    with ui.VStack(width=ui.Fraction(1), height=row_height):
                        ui.Spacer(height=border)
                        value_model = model.get_item_value_model(item, column_id)
                        with ui.HStack(height=0, spacing=2):
                            ui.StringField(
                                value_model,
                                width=ui.Fraction(1),
                                height=row_height - 10,
                                alignment=ui.Alignment.LEFT_CENTER,
                                identifier=f"ros_package_table_path_field_{item.row_index}",
                            )

                            def update_field(filename, path, vm=value_model):
                                if filename == "":
                                    val = path
                                elif filename[0] != "/" and path[-1] != "/":
                                    val = path + "/" + filename
                                elif filename[0] == "/" and path[-1] == "/":
                                    val = path + filename[1:]
                                else:
                                    val = path + filename
                                vm.set_value(val)

                            add_folder_picker_icon(
                                update_field,
                                dialog_title="Select ROS Package Folder",
                                button_title="Select Folder",
                                size=16,
                            )
                            ui.Button(
                                name="Remove",
                                style_type_name_override="IconButton",
                                style=get_style(),
                                width=ui.Pixel(18),
                                height=row_height - border * 2,
                                clicked_fn=lambda i=item: self._on_delete(i),
                                alignment=ui.Alignment.CENTER,
                                identifier=f"ros_package_table_remove_button_{item.row_index}",
                            )
                        ui.Spacer(height=border)
                ui.Spacer(width=border, height=border)
