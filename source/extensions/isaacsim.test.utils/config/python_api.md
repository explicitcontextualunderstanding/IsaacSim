# Public API for module isaacsim.test.utils:

## Classes

- class MenuUITestCase(OmniUiTest)
  - async def setUp(self)
  - async def tearDown(self)
  - async def wait_for_stage_loading(self)
  - async def new_stage(self)
  - async def wait_n_frames(self, n: int = 10)
  - async def get_viewport_context_menu(self, max_attempts: int = 5) -> dict[str, Any]
  - async def menu_click_with_retry(self, menu_path: str, delays: list[int] = None, window_name: str = None, wait_n_frames: int = 10)
  - def count_prims_by_type(self, prim_type: str) -> int
  - async def run_timeline_frames(self, n: int = 50)
  - def select_prim(self, prim_path: str)

- class TimedAsyncTestCase(omni.kit.test.AsyncTestCase)
  - async def setUp(self)
  - async def tearDown(self)

## Functions

- def validate_folder_contents(path: str | Path, expected_counts: dict[str, int]) -> bool
- def get_folder_file_summary(path: str | Path) -> dict[str, int | dict[str, list]]
- def validate_file_list(file_paths: list[str | Path]) -> dict[str, bool | list[str]]
- async def capture_annotator_data_async(annotator_name: str, camera_position: tuple[float, float, float] = (5, 5, 5), camera_look_at: tuple[float, float, float] = (0, 0, 0), resolution: tuple[int, int] = (1280, 720), camera_prim_path: str | None = None, render_product: Any = None, do_array_copy: bool = True) -> Any
- async def capture_rgb_data_async(camera_position: tuple[float, float, float] = (5, 5, 5), camera_look_at: tuple[float, float, float] = (0, 0, 0), resolution: tuple[int, int] = (1280, 720), camera_prim_path: str | None = None, render_product: Any = None) -> np.ndarray
- async def capture_depth_data_async(depth_type: str = 'distance_to_camera', camera_position: tuple[float, float, float] = (5, 5, 5), camera_look_at: tuple[float, float, float] = (0, 0, 0), resolution: tuple[int, int] = (1280, 720), camera_prim_path: str | None = None, render_product: Any = None) -> np.ndarray
- async def capture_viewport_annotator_data_async(viewport_api, annotator_name = 'rgb') -> Any
- def compute_difference_metrics(golden_array: np.ndarray, test_array: np.ndarray) -> dict[str, object]
- def print_difference_statistics(metrics: dict[str, object])
- def compare_arrays_within_tolerances(golden_array: np.ndarray, test_array: np.ndarray, allclose_rtol: float | None = 1e-05, allclose_atol: float | None = 1e-08, mean_tolerance: float | None = None, max_tolerance: float | None = None, absolute_tolerance: float | None = None, percentile_tolerance: tuple | None = None, rmse_tolerance: float | None = None, print_all_stats: bool = False) -> dict[str, object]
- def compare_images_within_tolerances(golden_file_path: str, test_file_path: str, allclose_rtol: float | None = 1e-05, allclose_atol: float | None = 1e-08, mean_tolerance: float | None = None, max_tolerance: float | None = None, absolute_tolerance: float | None = None, percentile_tolerance: tuple | None = None, rmse_tolerance: float | None = None, print_all_stats: bool = False) -> dict[str, object]
- def compare_images_in_directories(golden_dir: str, test_dir: str, path_pattern: str | None = None, allclose_rtol: float | None = 1e-05, allclose_atol: float | None = 1e-08, mean_tolerance: float | None = None, max_tolerance: float | None = None, absolute_tolerance: float | None = None, percentile_tolerance: tuple | None = None, rmse_tolerance: float | None = None, print_all_stats: bool = False, print_per_file_results: bool = True) -> dict[str, object]
- def save_rgb_image(rgb_data: np.ndarray, out_dir: str, file_name: str)
- def save_depth_image(depth_data: np.ndarray, out_dir: str, file_name: str, normalize: bool = False)
- def read_image_as_array(file_path: str, squeeze_singleton_channel: bool = True) -> np.ndarray
- async def menu_click_with_retry(menu_path: str, delays: list[int] = None, window_name: str = None, wait_n_frames: int = 10)
- def get_all_menu_paths(menu_dict: dict, current_path: str = '', root_path: str = '') -> list[str]
- def count_menu_items(menu_dict: dict) -> int
