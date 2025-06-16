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
import omni.kit.app
import omni.kit.menu.utils
from omni.kit.menu.utils import LayoutSourceSearch, MenuItemDescription, MenuLayout, add_menu_items


class HelpMenuExtension:
    def __init__(self, ext_id):
        self.__menu_layout = [
            MenuLayout.Menu(
                "Help",
                [
                    MenuLayout.Seperator("Examples"),
                    MenuLayout.Item("Physics Examples"),
                    MenuLayout.Item("Robotics Examples"),
                    MenuLayout.Item("Warp Sample Scenes"),
                    MenuLayout.Seperator("Isaac Sim Reference"),
                    MenuLayout.Item("About"),
                    MenuLayout.Item("Online Guide", source="Isaac Sim Online Guide"),
                    MenuLayout.Item("Online Forums", source="Isaac Sim Online Forums"),
                    MenuLayout.Item("Scripting Manual", source="Isaac Sim Scripting Manual"),
                    MenuLayout.Seperator("Omniverse Reference"),
                    MenuLayout.Item("Kit Programming Manual"),
                    MenuLayout.Item("Omni UI Docs"),
                    MenuLayout.Seperator("Physics Reference"),
                    MenuLayout.Item("Physics Programming Manual"),
                    MenuLayout.Seperator("Warp Reference"),
                    MenuLayout.Item("Getting Started", source="Window/Warp/Getting Started"),
                    MenuLayout.Item("Documentation", source="Window/Warp/Documentation"),
                    MenuLayout.Seperator("USD"),
                    MenuLayout.Item("USD Reference Guide", source="Help/USD Reference Guide"),
                    MenuLayout.Seperator(),
                ],
            )
        ]
        omni.kit.menu.utils.add_layout(self.__menu_layout)

        ## hack to have examples in two places
        robotics_demo = MenuItemDescription(
            name="Robotics Examples",
            onclick_action=(
                "isaacsim.examples.browser",
                "open_isaac_sim_examples_browser",
            ),
        )
        physics_demo = MenuItemDescription(
            name="Physics Examples",
            onclick_action=("omni.physxuicommon.windowmenuitem", "WindowMenuItemAction_PhysicsDemoScenes"),
        )
        warp_demo = MenuItemDescription(
            name="Warp Sample Scenes",
            onclick_action=("omni.warp", "browse_scenes"),
        )

        demo_items = [robotics_demo, physics_demo, warp_demo]

        add_menu_items(demo_items, "Help")

        # physics menu item
        url = resolve_physics_ref_url()
        physics_menu_item = MenuItemDescription(
            name="Physics Programming Manual", onclick_fn=lambda: self.open_ref_url(url)
        )

        add_menu_items([physics_menu_item], "Help")

    def shutdown(self):
        omni.kit.menu.utils.remove_layout(self.__menu_layout)

    def open_ref_url(self, url):
        import webbrowser

        webbrowser.open(url)


def resolve_physics_ref_url():
    # get URL by physx extensions version
    try:
        manager = omni.kit.app.get_app().get_extension_manager()
        ext_id = manager.get_extension_id_by_module("omni.physx")
        ext_version = manager.get_extension_dict(ext_id)["package"]["version"]
        version = ".".join(ext_version.split(".")[:2])
        url = f"https://docs.omniverse.nvidia.com/kit/docs/omni_physics/{version}/index.html"

        # check if website exists
        import requests

        response = requests.head(url, timeout=5)
        if response.status_code > 400:
            raise Exception()
    except:
        url = "https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/index.html"

    return url
