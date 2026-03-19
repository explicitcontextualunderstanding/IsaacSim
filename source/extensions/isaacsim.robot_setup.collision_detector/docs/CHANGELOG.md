# Changelog

## [0.1.0] - 2026-02-20
### Added
- Initial release of the Robot Self Collision Detector panel.
- Docks next to the Content window by default.
- Automatic robot detection from stage selection.
- **Check Collisions** enumerates all non-adjacent rigid body link pairs.
- Flat two-column TreeView with static "Rigid Body A" / "Rigid Body B" headers.
- **Filtered Pair** checkbox column to toggle `UsdPhysics.FilteredPairsAPI` per pair.
- Per-column sorting (A-Z / Z-A toggle) for all three columns.
- Search bar filtering across body names.
- Per-body **Select Collision Prim** button to inspect collision geometry in the Property panel.
- Viewport highlighting of selected pairs via session-layer display color overrides.
- Stage lifecycle handling (clears on stage close, listens for selection changes).
