# Changelog
## [0.8.8] - 2026-03-05
### Changed
- Fix api and docs syntax issues


## [0.8.7] - 2026-01-20
### Removed
- Remove deprecated PhysX residual reporting APIs (`enable_residual_reports`, `get_position_residuals`, `get_velocity_residuals`) from `Articulation` and `SingleArticulation`.

## [0.8.6] - 2026-01-19
### Fixed
- Fixed missing argument `context` in `_reset_fabric_selection` method.

## [0.8.5] - 2026-01-08
### Fixed
- Fixed incorrect method call `set_wind` to `set_winds` in `ParticleSystem` constructor.
- Fixed incorrect attribute name `solverPositionIteration` to `solverPositionIterationCount` in `ParticleSystem.get_solver_position_iteration_counts`.

## [0.8.4] - 2026-01-06
### Changed
- Migrate more events to Events 2.0.

## [0.8.3] - 2025-12-05
### Changed
- Migrate to Events 2.0.

## [0.8.2] - 2025-12-03
### Changed
- Remove TODOs.

## [0.8.1] - 2025-12-01
### Fixed
- Fix the `PhysxCollisionAPI` schema when checking for collision properties

## [0.8.0] - 2025-10-27
### Changed
- Replace import statements with the deprecation function when importing PyTorch
- Make omni.isaac.ml_archive an explicit test dependency

## [0.7.0] - 2025-10-17
### Changed
- Migrate PhysX subscription and simulation control interfaces to Omni Physics

## [0.6.1] - 2025-09-16
### Fixed
- Fix ill-formed SdfPath when querying XFormPrim's applied visual materials

## [0.6.0] - 2025-08-27
### Changed
- Re-write XFormPrim's `get_world_poses`/`set_world_poses` fabric implemenattion using Fabric Scene Delegate and IFabricHierarchy

## [0.5.2] - 2025-08-13
### Fixed
- Auto-compute joint indices when specifying joint names to get articulation's measured joint reaction forces/torques

## [0.5.1] - 2025-07-21
### Changed
- Added explicit `destroy()` method to `Prim`

## [0.5.0] - 2025-07-18
### Changed
- Expose the `reset_xform_properties` argument in single prim classes

## [0.4.4] - 2025-07-07
### Fixed
- Correctly enable omni.kit.loop-isaac in test dependency (fixes issue from 0.4.3)

## [0.4.3] - 2025-07-03
### Changed
- Make omni.kit.loop-isaac an explicit test dependency

## [0.4.2] - 2025-07-01
### Fixed
- Added deprecation note for deformable prims

## [0.4.1] - 2025-06-25
### Changed
- Add --reset-user to test args

## [0.4.0] - 2025-06-09
### Changed
- Move the `_impl/single_prim_wrapper` script to the `impl` folder and update import statements

## [0.3.18] - 2025-05-31
### Changed
- Use default nucleus server for all tests

## [0.3.17] - 2025-05-19
### Changed
- Update copyright and license to apache v2.0

## [0.3.16] - 2025-05-16
### Fixed
- Fix rigid prim's angular velocity unit for USD implementation

## [0.3.15] - 2025-05-10
### Changed
- Enable FSD in test settings

## [0.3.14] - 2025-04-21
### Changed
- Updated franka robot example code

## [0.3.13] - 2025-04-14
### Changed
- Update Isaac Sim robot asset path

## [0.3.12] - 2025-04-09
### Changed
- Update all test args to be consistent

## [0.3.11] - 2025-04-08
### Changed
- Update the get_measured_joint_forces method's docstring to clarify which frame the forces and torques are reported in.

## [0.3.10] - 2025-04-04
### Changed
- Version bump to fix extension publishing issues

## [0.3.9] - 2025-03-26
### Changed
- Cleanup and standardize extension.toml, update code formatting for all code

## [0.3.8] - 2025-03-11
### Changed
- Switch asset root for tests to internal nucleus

## [0.3.7] - 2025-01-26
### Changed
- Update test settings

## [0.3.6] - 2025-01-21
### Changed
- Update extension description and add extension specific test settings

## [0.3.5] - 2024-12-30
### Fixed
- Fixed cleanup for Prim class if it was not initialized correctly

## [0.3.4] - 2024-12-18
### Fixed
- handles_initialized returning the wrong value for a single articulation when articulation controller not initialized

## [0.3.3] - 2024-12-12
### Fixed
- Updated the implementation of articulation inverse dynamic functions to respect the type of articulation (floating base vs fixed based)

## [0.3.2] - 2024-11-19
### Fixed
- XformPrim._set_xform_properties to take in consideration scale:unitsResolve attribute if authored.

## [0.3.1] - 2024-11-08
### Changed
- Changed Articulation.invalidate_physics_callback and RigidPrim.invalidate_physics_callback to weakref callbacks.

## [0.3.0] - 2024-11-05
### Added
- Added joint_names argument to all the joint related methods for ease of specifying which joints to manipulate/ query.

## [0.2.0] - 2024-10-31
### Changed
- Changed .initialize and .post_reset methods to be event triggered
- Changed physX warmup and simulation context creation to be event trigerred

## [0.1.1] - 2024-10-24
### Changed
- Updated dependencies and imports after renaming

## [0.1.0] - 2024-10-18
### Added
- Initial release
