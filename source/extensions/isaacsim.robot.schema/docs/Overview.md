```{csv-table}
**Extension**: {{ extension_version }},**Documentation Generated**: {sub-ref}`today`
```

# Overview

The isaacsim.robot.schema extension provides comprehensive USD schemas for robot modeling and sensor simulation in Isaac Sim. This extension defines formal USD schemas for range sensors, Isaac-specific sensors, and robot structural components, enabling standardized representation of robotic systems in USD stages. The extension is loaded early in the Kit startup sequence to ensure all robot-related schemas are available to other extensions.

<div align="center">

```{mermaid}
graph TD
    %% Inheritance relationships
    Generic --> RangeSensor
    Lidar --> RangeSensor
    UltrasonicArray --> RangeSensor
```

</div>

## Key Components

### Range Sensor Schemas

The `**omni.isaac.RangeSensorSchema**` module provides schemas for distance-measuring sensors. The base [RangeSensor](omni.isaac.RangeSensorSchema/omni.isaac.RangeSensorSchema.RangeSensor) class defines common attributes for range detection, visualization controls, and detection thresholds.

**[Lidar](omni.isaac.RangeSensorSchema/omni.isaac.RangeSensorSchema.Lidar)** extends the range sensor with sophisticated scanning capabilities including field-of-view configuration, rotation rates, and semantic data collection. It supports both continuous scanning and single-shot operation modes.

**[UltrasonicArray](omni.isaac.RangeSensorSchema/omni.isaac.RangeSensorSchema.UltrasonicArray)** implements multi-emitter ultrasonic sensing with advanced acoustic modeling. It manages collections of emitters through relationships, supports material-based reflection calculations using BRDF models, and implements Beer-Lambert attenuation for realistic acoustic behavior.

**[UltrasonicEmitter](omni.isaac.RangeSensorSchema/omni.isaac.RangeSensorSchema.UltrasonicEmitter)** represents individual transducer elements within ultrasonic arrays, managing per-ray intensity and adjacency relationships for multi-path reflection calculations.

**[UltrasonicFiringGroup](omni.isaac.RangeSensorSchema/omni.isaac.RangeSensorSchema.UltrasonicFiringGroup)** defines coordinated firing patterns for ultrasonic arrays, specifying which emitters fire and which receivers collect data during each measurement cycle.

**[UltrasonicMaterialAPI](omni.isaac.RangeSensorSchema/omni.isaac.RangeSensorSchema.UltrasonicMaterialAPI)** provides acoustic material properties for ultrasonic simulation, including reflectance, roughness, and metallic characteristics that affect acoustic reflection behavior.

**[Generic](omni.isaac.RangeSensorSchema/omni.isaac.RangeSensorSchema.Generic)** offers a simplified range sensor implementation with basic sampling rate and streaming controls for custom sensor implementations.

### Isaac Sensor Schemas

The `**omni.isaac.IsaacSensorSchema**` module provides specialized sensors for robotics simulation. [IsaacBaseSensor](omni.isaac.IsaacSensorSchema/omni.isaac.IsaacSensorSchema.IsaacBaseSensor) serves as the foundation class with basic enable/disable functionality.

**[IsaacContactSensor](omni.isaac.IsaacSensorSchema/omni.isaac.IsaacSensorSchema.IsaacContactSensor)** detects contact forces with configurable thresholds and measurement periods, providing visual feedback through colored sphere representation.

**[IsaacImuSensor](omni.isaac.IsaacSensorSchema/omni.isaac.IsaacSensorSchema.IsaacImuSensor)** simulates inertial measurement units with configurable filter widths for linear acceleration, angular velocity, and orientation measurements.

**[IsaacLightBeamSensor](omni.isaac.IsaacSensorSchema/omni.isaac.IsaacSensorSchema.IsaacLightBeamSensor)** creates light curtain sensors with multiple rays, configurable beam direction, and range detection capabilities for industrial automation applications.

**[IsaacRtxLidarSensorAPI](omni.isaac.IsaacSensorSchema/omni.isaac.IsaacSensorSchema.IsaacRtxLidarSensorAPI)** and **[IsaacRtxRadarSensorAPI](omni.isaac.IsaacSensorSchema/omni.isaac.IsaacSensorSchema.IsaacRtxRadarSensorAPI)** provide API schema extensions for RTX-accelerated sensor implementations.

### Robot Structural Schemas

The `usd.schema.isaac` module defines the robot modeling framework through several interconnected schemas:

**IsaacRobotAPI** serves as the primary robot schema, managing metadata like description, version, and license information. It maintains ordered relationships for robot links and joints that define the kinematic structure.

**IsaacLinkAPI** and **IsaacJointAPI** provide specialized schemas for robot components with name overrides and configuration options. The joint API includes DOF offset ordering and actuator specifications.

**IsaacSiteAPI** defines reference points within the robot structure with directional information for attachment and measurement purposes.

**IsaacSurfaceGripper** implements specialized gripping mechanisms with force limits, attachment point management, and status tracking for object manipulation.

**IsaacAttachmentPointAPI** configures individual contact points within surface grippers with clearance offsets and directional constraints.

## Functionality

The extension provides utility functions for robot structure management and schema migration. `PopulateRobotSchemaFromArticulation` and `RecalculateRobotSchema` automatically discover and organize robot components from physics articulations. These functions traverse joint connections to build ordered link and joint relationships while applying appropriate API schemas.

The schema migration system handles deprecated schema updates, replacing obsolete APIs like `IsaacReferencePointAPI` with current equivalents. Joint migration specifically handles deprecated per-axis DOF offset attributes, converting them to the modern token array format.

Robot tree generation creates hierarchical representations of robot structures through `RobotLinkNode` objects, enabling kinematic analysis and visualization. The tree structure supports parent-child relationships and joint connectivity information.

## Integration

The extension loads early in the Kit startup sequence to ensure schema availability for other robot-related extensions. It integrates with USD's schema system through proper inheritance hierarchies, with range sensors inheriting from `UsdGeom.Xformable` and robot APIs extending `UsdAPISchemaBase`.

The schemas work together to create complete robot descriptions - robot APIs organize structural components, sensor schemas define sensing capabilities, and utility functions maintain consistency across the robot definition. This integrated approach enables comprehensive robot modeling from kinematic structure to sensor simulation within a unified USD framework.
