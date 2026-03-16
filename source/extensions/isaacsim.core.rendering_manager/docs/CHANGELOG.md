# Changelog

## [0.2.3] - 2025-12-20
### Changed
- Move import of kit loop runner to a try-except block to handle import exceptions if omni.kit.loop-isaac is not enabled

## [0.2.2] - 2025-11-28
### Fixed
- Fix camera view test case by comparing whether the quaternions represent the same orientation

## [0.2.1] - 2025-11-25
### Changed
- Handle Isaac Sim's loop runner import exceptions

## [0.2.0] - 2025-11-21
### Added
- Add method to set a camera view

### Changed
- Split the manager implementation into two: `RenderingManager` and `ViewportManager`

## [0.1.0] - 2025-11-11
### Added
- Initial release
