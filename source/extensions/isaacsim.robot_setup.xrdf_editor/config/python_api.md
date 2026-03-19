# Public API for module isaacsim.robot_setup.xrdf_editor:

## Classes

- class CollisionSphereEditor
  - def __init__(self)
  - def clear_spheres(self, store_op: bool = True)
  - def clear_link_spheres(self, link_path: str, store_op: bool = True)
  - def delete_sphere(self, sphere_path: str)
  - def set_sphere_colors(self, filter: str, color_in: np.ndarray | None = None, color_out: np.ndarray | None = None)
  - def set_sphere_color(self, sphere_path: str, ensure_visual_material: bool = True)
  - def copy_all_sphere_data(self)
  - def undo(self)
  - def redo(self)
  - def generate_spheres(self, link_path, points, face_inds, vert_cts, num_spheres, radius_offset, is_preview)
  - def clear_preview(self)
  - def add_sphere(self, link_path, center, radius, store_op = True)
  - def load_xrdf_spheres(self, robot_prim_path, parsed_file: dict)
  - def load_spheres(self, robot_prim_path, robot_description_file_path)
  - def interpolate_spheres(self, path1, path2, num_spheres)
  - def scale_spheres(self, path, factor)
  - def get_sphere_names_by_link(self, link_path: str) -> list[str]
  - def write_spheres_to_dict(self, robot_prim_path: str, link_to_spheres: dict)
  - def save_spheres(self, robot_prim_path: str, f)
  - def on_shutdown(self)

## Functions

- def is_yaml_file(path: str) -> bool
- def is_xrdf_file(path: str) -> bool
- def on_filter_xrdf_item(item) -> bool
- def on_filter_item(item) -> bool
