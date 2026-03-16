# Public API for module isaacsim.gui.sensors.icon:

## Classes

- class IconModel(sc.AbstractManipulatorModel)
  - SENSOR_TYPES: List
  - class IconItem(sc.AbstractManipulatorItem)
    - def __init__(self, prim_path, icon_url)
  - def __init__(self)
  - def get_world_unit(self) -> float
  - def destroy(self)
  - def get_item(self, identifier) -> IconItem | None
  - def get_prim_paths(self) -> list[Sdf.Path]
  - def get_position(self, prim_path) -> Gf.Vec3d | None
  - def get_on_click(self, prim_path) -> Callable | None
  - def get_icon_url(self, prim_path) -> str
  - def clear(self)
  - def add_sensor_icon(self, prim_path, icon_url = None)
  - def remove_sensor_icon(self, prim_path)
  - def set_icon_click_fn(self, prim_path, call_back)
  - def show_sensor_icon(self, prim_path)
  - def hide_sensor_icon(self, prim_path)
  - def show_all(self)
  - def hide_all(self)
  - def refresh_all_icon_visuals(self)

- class IconScene
  - def __init__(self, title: str = None, icon_scale: float = 1.0, **kwargs)
  - [property] def visible(self) -> bool
  - [visible.setter] def visible(self, value: bool)
  - def destroy(self)
  - def clear(self)

## Functions

- def get_instance() -> SensorIconExtension | None
