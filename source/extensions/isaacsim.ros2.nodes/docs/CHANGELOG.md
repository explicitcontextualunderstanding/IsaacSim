# Changelog

## [1.10.1] - 2026-03-10
### Changed
- Updated test_spinning_camera_golden_images unit test with new golden USD

## [1.10.0] - 2026-03-05
### Changed
- ROS2PublishJointState node now publishes from sensor inputs (e.g. IsaacReadJointState) for joint state data

## [1.9.0] - 2026-03-02
### Changed
- Shifted RTX sensor scan accumulation and post-processing back to host by default to reduce GPU resource contention and improve frametime & frametime consistency. Post-processing-on-device still available as option by setting app.sensors.nv.[modality].outputBufferOnGPU=true.

## [1.8.0] - 2026-02-27
### Changed
- Converted OgnROS2QoSProfile node from Python to C++ for improved performance and consistency with other C++ OG nodes

## [1.7.0] - 2026-02-26
### Added
- ROS 2 H.264 Compressed Image support with H.264 RGB unit tests
- omni.replicator.nv extension automatically enabled by omni.replicator.core and implictily used for HW H.264 encoding

## [1.6.2] - 2026-02-18
### Fixed
- Fix `eFloat` and `eUnknown` cases in `writeNodeAttributeFromMessage` incorrectly hardcoding `"outputs:"` prefix instead of using `inputOutput(isOutput)` and `prependStr`

### Changed
- Move some test assertions into node subscription callbacks

## [1.6.1] - 2026-02-16
### Changed
- Add missing dependency for isaacsim.sensors.physics.nodes

## [1.6.0] - 2026-02-05
### Changed
- Update isaacsim.sensors.physics dependency to isaacsim.sensors.experimental.physics

## [1.5.10] - 2026-02-05
### Removed
- Moved test_menu_graphs to isaacsim.ros2.ui
- Simplified test dependencies
- Replaced fields_to_dtype with sensors_msgs_py.point_cloud2.read_points where applicable
- Cleaned up setUp and tearDown methods in tests

## [1.5.9] - 2026-01-30
### Changed
- Update waypoint follower action graph to use ReadPrimLocalTransform node
- Fix quaternion normalization in ROS2 waypoint follower tests

## [1.5.8] - 2026-01-26
### Added
- Also run ros2 image buffer tests with async rendering handshake enabled

## [1.5.7] - 2026-01-24
### Changed
- Fix issues with menu click and context menu tests being flaky

## [1.5.6] - 2026-01-21
### Added
- Added ros2 image buffer util unit tests

## [1.5.5] - 2025-12-23
### Changed
- Update unit tests to reduce overall test time

## [1.5.4] - 2025-12-19
### Removed
- Remove rendering manager as a test dependency since the extension already depends on it indirectly

## [1.5.3] - 2025-12-11
### Changed
- Update ros2 test case to wait for viewport to be ready
- Increase wait time for test_camera_info.py to 1.25 seconds
- Set resetSimulationTimeOnStop to True for ROS2CameraHelper, ROS2CameraInfoHelper, ROS2RtxLidarHelper

## [1.5.2] - 2025-12-02
### Fixed
- ROS2PublishObjectIdMap correctly generates 128-bit unsigned integer strings from 4xuint32

## [1.5.1] - 2025-12-01
### Added
- ros2_common.build_rtx_sensor_pointcloud_writer to build Writer for ROS2 PointCloud2 for user-selected metadata

### Changed
- ROS2RtxLidarHelper uses ros2_common.build_rtx_sensor_pointcloud_writer

## [1.5.0] - 2025-11-26
### Added
- OgnROS2RtxLidarPointCloudConfig to simplify setting metadata to include in RTX Lidar PointCloud2

### Changed
- OgnROS2RtxLidarHelper now takes selectedMetadata input to specify which metadata to include in RTX Lidar PointCloud2
- OgnROS2PublishPointCloud updated with new CUDA kernel to add RTX Lidar and Radar metadata to PointCloud2 message when data is on device

## [1.4.0] - 2025-11-26
### Changed
- Optimized to use a single publish thread rather than tasks. Can be controlled with /exts/isaacsim.ros2.bridge/publish_with_queue_thread=true|false

## [1.3.0] - 2025-11-24
### Added
- Added support for pinned memory buffers to increase memcpy performance

## [1.2.0] - 2025-11-24
### Changed
- Update code to use new handle interface from isaacsim.ros2.core extension.

## [1.1.1] - 2025-11-20
### Changed
- Cleaned up test_camera_info.py: removed unused imports and centralized visualization flag

## [1.1.0] - 2025-11-10
### Changed
- Removed "viewport" input from ROS2 camera helper node

## [1.0.1] - 2025-11-07
### Changed
- Update to Kit 109 and Python 3.12

## [1.0.0] - 2025-11-02
### Added
- Initial release of `isaacsim.ros2.nodes` extension
- Moved ROS 2 OmniGraph nodes and components from isaacsim.ros2.bridge to this extension
