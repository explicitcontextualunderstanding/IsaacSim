# Changelog
## [0.3.1] - 2026-03-06
### Changed
- Robot Inspector performance and stability improvements.
- Schema UI tests: tearDown now waits for stage assets to finish loading before cleanup to reduce flakiness.

## [0.3.0] - 2026-03-04
### Changed
- Added Overview.md, python_api.md and updated docstrings

## [0.2.0] - 2026-02-23
### Added
- Robot masking: deactivate, bypass, and anchor controls for joints and links via stage columns
- Masking state management with session-scoped USD masking layer
- Robot Inspector window replacing Robot Hierarchy with component inspection and masking UI
- Change robot hierarchy based on different view modes (Flat, Tree, Mujoco-style)
- Custom stage column delegates for deactivate, bypass, and anchor toggles
- Automatic masking layer cleanup on stage open/close

### Changed
- Renamed "Robot Hierarchy" window to "Robot Inspector"

## [0.1.0] - 2026-02-12
### Added
- Initial version
