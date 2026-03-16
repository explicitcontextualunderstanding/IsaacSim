# Changelog

## [0.11.1] - 2026-03-07
### Fixed
- Add `omni.kit.material.library` as a dependency

## [0.11.0] - 2026-03-06
### Added
- Move `find_widget_with_retry` from `MenuUITestCase` to `menu_utils` as a standalone function
- Add `find_enabled_widget_with_retry` to poll for a widget that is both found and enabled
- Add `wait_for_widget_enabled` to poll until an already-found widget becomes enabled

## [0.10.1] - 2026-03-04
### Changed
- Fix API errors

## [0.10.0] - 2026-03-04
### Changed
- Added Overview.md and python_api.md and updated docstrings

## [0.9.1] - 2026-02-22
### Changed
- Add `omni.kit.material.library.get_mdl_list_async` and `omni.kit.menu.utils.rebuild_menus` to `MenuUITestCase.wait_for_stage_loading` to fix menu rebuild issues

## [0.9.0] - 2026-02-21
### Changed
- Add `find_widget_with_retry` to `MenuUITestCase` to find a widget with retry

## [0.8.2] - 2026-02-16
### Changed
- Replace `omni.kit.ui_test.menu_click` with custom step-by-step menu navigation that polls for each submenu to become findable and visible before proceeding, avoiding the `carb.log_error` and `AttributeError` that `menu_click` produces when submenus are slow to appear
- Add `carb.log_info` diagnostics throughout `menu_click_with_retry` for log debugging

## [0.8.1] - 2026-02-13
### Changed
- Suppress transient error logs from `omni.kit.ui_test.query` during intermediate retries in `menu_click_with_retry`; errors are only surfaced on the final retry attempt

## [0.8.0] - 2026-02-09
### Changed
- Add pycoverage patch for numpy `_CopyMode.__bool__` to prevent `ValueError` when scipy imports trigger `_CopyMode.IF_NEEDED` evaluation under coverage

## [0.7.3] - 2026-02-02
### Changed
- Fix pycoverage compatibility issue with numpy sum and prod functions

## [0.7.2] - 2026-02-02
### Changed
- Add a pycoverage compatible amin and amax implementation that is monkeypatched into numpy on extension startup. Removed from image_comparison.py as it is no longer needed.
- This is only used if --/exts/omni.kit.test/pyCoverageEnabled=1 is set

## [0.7.1] - 2026-01-24
### Changed
- Refactor menu_click_with_retry into a separate function
- Add new_stage to MenuUITestCase

## [0.7.0] - 2025-12-22
### Added
- Add new utility functions (`get_all_menu_paths`, `count_menu_items`) and `MenuUITestCase` base class for menu UI tests.

## [0.6.4] - 2025-12-07
### Changed
- Update description

## [0.6.3] - 2025-12-02
### Added
- Specify --/app/settings/fabricDefaultStageFrameHistoryCount=3 for startup test

### Removed
- Remove omni.replicator.core as an explicit test dependency

## [0.6.2] - 2025-11-20
### Changed
- Add omni.replicator.core as an explicit dependency for image capture utils

## [0.6.1] - 2025-11-10
### Changed
- Fix invalid escape sequence

## [0.6.0] - 2025-11-07
### Added
- Added `compare_images_in_directories()` function to compare images in two directories

## [0.5.1] - 2025-09-26
### Changed
- Update license headers

## [0.5.0] - 2025-09-22
### Changed
- Can now optionally exclude blank pixels from image_comparison.compute_difference_metrics.

## [0.4.0] - 2025-09-19
### Added
- Added `image_io.py` module with image I/O utilities:
  - `save_rgb_image()` function for saving RGB/RGBA image data to disk
  - `save_depth_image()` function for saving depth data as TIFF (float32) or grayscale visualization
  - `read_image_as_array()` function for reading images as numpy arrays compatible with comparison utilities
- Added `capture_viewport_annotator_data_async()` function to capture annotator data from existing viewports

### Changed
- Moved image saving functions (`save_rgb_image`, `save_depth_image`) from `image_capture.py` to new `image_io.py` module
- Added optional `render_product` parameter to `capture_annotator_data_async()`, `capture_rgb_data_async()`, `capture_depth_data_async()

## [0.3.0] - 2025-09-15
### Changed
- Added file_validation.py for utilities to validate folder contents and file lists

## [0.2.1] - 2025-09-08
### Added
- Base test class stores the asset root path using fast path

## [0.2.0] - 2025-08-28
### Added
- Added image comparison utilities
- Added image capture utilities
- Added test utilities

## [0.1.0] - 2025-08-11
### Added
- Initial release
