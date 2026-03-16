# Changelog

## [1.2.0] - 2026-03-09
### Added
- PhysicsJointPoseFixRule to correct joint local poses after body world transforms change
- Idempotency test that runs the Isaac Sim profile twice and asserts stable output

### Fixed
- VariantRoutingRule leaving self-referencing payload arcs on extracted variant layers
- GeometriesRoutingRule non-idempotent merge decisions when VisualMaterials scopes or empty siblings remain after geometry extraction
- GeometriesRoutingRule floating-point precision drift in xformOp values across consecutive serialization round-trips
- GeometriesRoutingRule retaining root-level prims outside the defaultPrim hierarchy after flattening
- GeometriesRoutingRule quaternion sign ambiguity (q vs -q) causing idempotency test failure on double-transform
- SchemaRoutingRule, PropertyRoutingRule, and PrimRoutingRule using root layer defaultPrim instead of composed stage defaultPrim
- RobotSchemaRule not setting defaultPrim when no matching Isaac Robot schemas are found

## [1.1.2] - 2026-03-02
### Fixed
- Ensure all generated relative paths (references, payloads, sublayers) use explicit `./` or `../` prefixes

## [1.1.1] - 2026-02-26
### Changed
- Updated isaacsim_structure.json to delete Newton Mimic api in physx to avoid conflict

## [1.1.0] - 2026-02-17
### Added
- Overwrite schema rule

## [1.0.2] - 2026-02-16
### Fixed
- Fix geometry rule for negative scale and proper deduplicate

## [1.0.1] - 2026-02-13
### Fixed
- Tests failing due to misconfiguration of tests

## [1.0.0] - 2026-02-12
### Added
- First version of Asset Transformer core rules
