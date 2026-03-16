# Public API for module isaacsim.gui.property:

## Classes

- class ArrayPropertiesWidget(UsdPropertiesWidget)
  - def on_new_payload(self, payload)
  - def build_property_item(self, stage, ui_prop: UsdPropertyUiEntry, prim_paths: List[Sdf.Path])

- class CustomDataWidget(SimplePropertyWidget)
  - def on_new_payload(self, payload)
  - def build_items(self)
  - def build_property_item(self, stage, ui_prop: UsdPropertyUiEntry, prim_paths: List[Sdf.Path])

- class MotionPlanningAPIWidget(_RobotSchemaWidgetBase)
  - def __init__(self, title: str, collapsed: bool = False)

- class NameOverrideWidget(UsdPropertiesWidget)
  - def __init__(self, title: str, collapsed: bool = False)
  - def destroy(self)
  - def on_new_payload(self, payload) -> Usd.Prim | bool
  - def on_remove_attr(self)
  - def build_items(self)

- class NamespaceWidget(UsdPropertiesWidget)
  - def __init__(self, title: str, collapsed: bool = False)
  - def destroy(self)
  - def on_new_payload(self, payload) -> Usd.Prim | bool
  - def on_remove_attr(self)
  - def build_items(self)

- class JointAPIWidget(_RobotSchemaWidgetBase)
  - def __init__(self, title: str, collapsed: bool = False)

- class LinkAPIWidget(_RobotSchemaWidgetBase)
  - def __init__(self, title: str, collapsed: bool = False)

- class RobotAPIWidget(_RobotSchemaWidgetBase)
  - def __init__(self, title: str, collapsed: bool = False)

- class IsaacPropertyWidgets(omni.ext.IExt)
  - def __init__(self)
  - def on_startup(self, ext_id)
  - def on_shutdown(self)
