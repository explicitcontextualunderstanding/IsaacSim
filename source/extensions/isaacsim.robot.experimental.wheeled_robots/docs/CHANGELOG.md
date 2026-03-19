# Changelog
## [0.2.3] - 2026-03-05
### Changed
- Linting: `__all__` re-exports, docstrings, and type/doc fixes for ruff, mypy, darglint, pydoclint.


## [0.2.2] - 2026-03-05
### Changed
- Fix api and docs syntax issues


## [0.2.1] - 2026-03-02
### Changed
- Removed deprecated libraries and integrated the latest replacements.


## [0.2.0] - 2026-03-01
### Removed
- Unused C++ plugin, bindings, and IWheeledRobots interface (extension is Python-only).
- OmniGraph nodes (DifferentialController, HolonomicController, AckermannController, etc.); use Python API only. For graph nodes use the stable `isaacsim.robot.wheeled_robots` extension.


## [0.1.0] - 2026-02-25
### Added
- Wheeled Robot for Warp based APIs.

