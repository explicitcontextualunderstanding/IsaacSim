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

"""Callback functions for GUI component interactions including clipboard operations, file browser, IDE integration, and documentation links."""


import os
import subprocess
import sys

import carb


def on_copy_to_clipboard(to_copy: str):
    """Copy text to system clipboard.

    Args:
        to_copy: Text content to copy to the system clipboard.
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


def on_open_IDE_clicked(ext_path: str, file_path: str):
    """Opens the current directory and file in VSCode.

    Args:
        ext_path: Path to the extension directory.
        file_path: Path to the specific file to open.
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


def on_open_folder_clicked(file_path: str):
    """Opens the current directory in a file browser.

    Args:
        file_path: Path to the file whose directory should be opened.
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


def on_docs_link_clicked(doc_link: str):
    """Opens an extension's documentation in a web browser.

    Args:
        doc_link: URL of the documentation to open.
    """
    import webbrowser

    try:
        webbrowser.open(doc_link, new=2)
    except Exception as e:
        carb.log_warn(f"Could not open browswer with url: {doc_link}, {e}")
