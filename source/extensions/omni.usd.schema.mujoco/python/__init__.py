# register the mjcPhysics schema plugin

import os

from pxr import Plug

pluginsRoot = os.path.join(os.path.dirname(__file__), "../../../plugins")

mujocoSchemaPath = pluginsRoot
Plug.Registry().RegisterPlugins(mujocoSchemaPath)
