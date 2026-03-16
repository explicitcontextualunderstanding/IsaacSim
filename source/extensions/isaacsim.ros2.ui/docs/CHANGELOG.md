# Changelog

## [1.4.2] - 2026-03-06
### Changed
- Migrated test_menu_graphs imports to use experimental prims and stage utilities (XformPrim, define_prim, add_reference_to_stage)
- Fixed articulation root path in test_joint_states_data_flow and test_odometry_data_flow to use chassis_link
- Fixed SimpleCheckBox widget path for "Publish Robot's TF?" in test_odometry_data_flow

## [1.4.1] - 2026-03-05
### Changed
- Fix api and docs syntax issues


## [1.4.0] - 2026-02-08
### Added
- Added PointCloud2 metadata options to RTX Lidar OG tool
- Added test_menu_graphs

## [1.3.0] - 2026-02-07
### Changed
- Remove Asset Browser from menu
- Use menu.open_content_browser_to_path to open the Content Browser to a specific path as a replacement

## [1.2.0] - 2025-11-25
### Changed
- Set ResetOnStop to True for all Simulation Time OG nodes referenced by ROS Bridge

## [1.0.1] - 2025-11-10
### Fixed
- Replaced deprecated `onclick_fn` with `onclick_action` in menu items to eliminate deprecation warnings
- Registered proper actions for Nova Carter, Leatherback, and iw.hub ROS robot creation menu items

## [1.0.0] - 2025-11-03
### Changed
- Moved all ui components of isaacsim.ros2.bridge to this extension
