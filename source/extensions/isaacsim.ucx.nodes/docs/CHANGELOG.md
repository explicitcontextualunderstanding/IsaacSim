# Changelog
## [1.3.3] - 2026-03-05
### Fixed
- Fixed flaky `test_sim_clock` test by increasing UCX send timeout from 1ms to 5000ms and using non-blocking `asyncio.sleep` in the receive wait loop

## [1.3.2] - 2026-03-02
### Changed
- Add Overview.md, public python_api.md and update docstrings

## [1.3.1] - 2026-02-14
### Fixed
- Fixed flaky UCX joint state and odometry tests by increasing connection wait time and adding retry logic

## [1.3.0] - 2025-12-15
### Changed
- Migrate extension implementation to core experimental API

## [1.2.0] - 2025-12-12
### Changed
- Consolidate common functionality (publishMessage) into UcxNode base class.
- Introduce per-sensor data structs (ClockData, ImuData, OdometryData, JointStateData, JointCommandData, ImageMetadata) to separate data extraction from serialization.

## [1.1.3] - 2025-12-11
### Changed
- Try removing listener from registry after node resets
- Fix issues with unit tests
- Reset simulation time on stop by default for UCXCameraHelper

## [1.1.2] - 2025-12-07
### Changed
- Fix issues found by clang tidy

## [1.1.1] - 2025-12-02
### Changed
- Updated deprecated imports to isaacsim.storage.native

## [1.1.0] - 2025-11-25
### Added
- Joint state publishing node
- Joint command subscribing node
- Odometry publishing node
- IMU publishing node
- RGB image publishing node
- Camera helper node for connecting the replicator pipeline with image publishing

## [1.0.0] - 2025-11-18
### Added
- Initial release of UCX Nodes extension
- OmniGraph nodes for high-performance UCX communication
