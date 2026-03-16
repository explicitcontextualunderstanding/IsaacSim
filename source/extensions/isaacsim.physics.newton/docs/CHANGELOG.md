# Changelog

## [0.6.0] - 2026-03-10
### Changed
- Upgraded Newton pip package from 1.0.0rc3 to 1.0.0.

## [0.5.3] - 2026-03-06
### Changed
- Upgraded Newton pip package from 1.0.0rc2 to 1.0.0rc3. 

## [0.5.2] - 2026-03-05
### Changed
- Fix api and docs syntax issues

## [0.5.1] - 2026-03-04
### Changed
- Fix api errors

## [0.5.0] - 2026-03-04
### Changed
- Add Overview.md, python_api.md, SETTINGS.md and update docstrings

## [0.5.0] - 2026-03-03
### Changed
- Upgraded Newton pip package from 1.0.0rc1 to 1.0.0rc2.
- Enabled a set of Newton unit tests that are implemented and pass

## [0.4.1] - 2026-02-28
### Changed
- Updated SimulationFns registration to match new omni.physics.core API (`initialize`/`close` replaces removed `attach_stage`/`detach_stage`)

## [0.4.0] - 2026-02-27
### Changed
- Pinned mujoco-warp to 3.5.0.2 for compatibility with newton[sim] 1.0.0rc1

## [0.3.0] - 2026-02-27
### Removed
- Removed omni.warp extension dependency; only omni.warp.core is required

### Changed
- Upgraded Newton to 1.0.0rc1 with updated solver and USD schema handling

## [0.2.0] - 2026-02-24
### Changed
- Moved newton prebundle to isaacsim.pip.newton extension

## [0.1.2] - 2026-02-20
- Fixed broken newton usecases after upgrading the newton/mujoco packages

## [0.1.1] - 2026-02-06
### Added
- Capability check

### Removed
- Data handling from the physics umbrella

## [0.1.0] - 2025-12-23
### Added
- Initial release of isaacsim.physics.newton extension
- Integration with Newton physics engine
- Support for XPBD and MJC integrators
- Stage update node for physics simulation
