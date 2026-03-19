# Changelog

## [1.5.1] - 2026-03-09
### Fixed
- Hardened subprocess call to avoid shell=True with string concatenation

## [1.5.0] - 2026-02-26
### Added
- CompressedImage message backend

## [1.4.0] - 2026-02-24
### Changed
-  Removed hardcoded ROS 2 distribution checks. Added experimental support for any ROS 2 distribution beyond Jazzy to be sourced and used with Isaac Sim.

## [1.3.1] - 2026-02-20
### Changed
- Fixed ROS 2 service request polling so `takeRequest()` can receive a pending request on its first poll call.
- Added a regression doctest covering first-poll request/response behavior for `Ros2Service`.

## [1.3.0] - 2026-02-01
### Changed
- Removed isaacsim.sensors.experimental.physics dependency

## [1.2.7] - 2026-01-23
### Changed
- Set publish_with_queue_thread extension setting to true

## [1.2.6] - 2026-01-21
### Added
- Added ros2 image buffer utils

## [1.2.5] - 2025-12-23
### Changed
- Added a simulate_until_condition method to simplify test cases

## [1.2.4] - 2025-12-11
### Changed
- Update ros2 test case to wait for viewport to be ready

## [1.2.3] - 2025-12-07
### Changed
- Fix clang tidy issues in cpp code

## [1.2.2] - 2025-11-26
### Changed
- PointCloud2 message backend now optionally supports host-pinned buffer

## [1.2.1] - 2025-11-26
### Added
 - Default setting values to control ROS2PublishImage queue thread optimization.

## [1.2.0] - 2025-11-24
### Added
- Add flag to generateBuffer() to allocate CUDA pinned memory for image buffers

## [1.1.0] - 2025-11-24
### Changed
- Moved handle interface from isaacsim.core.nodes extension to this extension.

## [1.0.2] - 2025-11-20
### Changed
- Use set_lens_distortion_model in TestCameraInfoUtils

## [1.0.1] - 2025-11-07
### Changed
- Update to Kit 109 and Python 3.12

## [1.0.0] - 2025-11-02
### Added
- Initial release of `isaacsim.ros2.core` extension
- Moved core ROS 2 libraries and backend functionality from isaacsim.ros2.bridge to this extension
