# Changelog
## [1.3.1] - 2026-03-02
### Changed
- Add Overview.md, public python_api.md and update docstrings

## [1.3.0] - 2025-12-12
### Added
- Add UCXListenerRegistry::tryRemoveListener for reference-counted listener cleanup

## [1.2.2] - 2025-12-07
### Changed
- Fix issues found by clang tidy

## [1.2.1] - 2025-11-28
### Changed
- Regenerate pip prebundle

## [1.2.0] - 2025-11-25
### Added
- Added `UCXListener::tagSendWithRequest` for better monitoring of send requests.

## [1.1.0] - 2025-11-24
### Added
- Added `UcxUtils.h`.
- Added UCX Python dependencies.

### Changed
- Refactored UCXListener's tag messaging functions.

## [1.0.0] - 2025-10-15
### Added
- **UCX Communication Wrappers**
  - **UCXConnection**: High-performance point-to-point communication
    - Tagged send/receive operations for efficient message routing
    - Multi-buffer operations supporting CPU and CUDA memory transfers
    - Thread-safe connection lifecycle management

  - **UCXListener**: Server-side listener for incoming connections
    - Configurable port binding with timeout support
    - Thread-safe client connection tracking

  - **UCXListenerRegistry**: Centralized listener management
    - Singleton pattern ensuring one listener per port
    - Thread-safe listener creation and retrieval

- **Plugin Interface**: Carbonite plugin that publishes simulation clock and physics steps

- **Testing**: Unit tests for core components using doctest framework
