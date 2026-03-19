# Changelog

## [1.3.0] - 2026-03-05
### Added
- Add Isaac Read Joint State node that reads full joint state (names, positions, velocities, efforts, DOF types, stage units) from an articulation prim

## [1.2.0] - 2026-03-04
### Changed
- Added Overview.md and python_api.md and updated docstrings

## [1.1.0] - 2026-02-27
### Changed
- Migrate nodes to use C++ core experimental prims APIs

## [1.0.1] - 2026-02-10
### Changed
- IMU and Contact sensor creation commands renamed to include Experimental in their name to avoid name collision with deprecated sensor commands

## [1.0.0] - 2026-02-01
### Changed
- Moved nodes from isaacsim.sensors.physics extension to this extension
- Updated to use interfaces from isaacsim.sensors.experimental.physics extension
- Updated contact and IMU examples to use the new sensor command APIs and legacy Python interfaces
