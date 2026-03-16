# USD Schemas

## rangeSensorSchema

### RangeSensor

#### enabled
  Flag used to enable or disable this sensor

#### drawPoints
  Set to true to draw debug points on sensor hit locations

#### drawLines
  Set to true to draw debug lines representing sensor ray casts

#### minRange
  Minimum range for sensor to detect a hit

#### maxRange
  Maximum range for sensor to detect a hit


### Lidar

#### yawOffset
  Offset along yaw axis to account for sensor having a different forward direction

#### rotationRate
  Rotation rate of sensor in Hz, set to zero to make sensor fire all rays at once

#### highLod
  Enable High Lod for 3D Lidar sensor

#### horizontalFov
  Horizontal Field of View in degrees

#### verticalFov
  Vertical Field of View in degrees

#### horizontalResolution
  Degrees in between rays for horizontal axis

#### verticalResolution
  Degrees in between rays for vertical axis

#### enableSemantics
  Set to true to get semantic information of sensor hit locations


### UltrasonicArray

#### numBins
  Number of bins that emitters in this array outputs

#### useBRDF
  Use angle of emitter/receiver relative to normal to compute intensity response

#### useUSSMaterialsForBRDF
  Use Ultrasonic materials for BRDF calculation

#### useDistAttenuation
  Use simplified Beer-Lambert model, negative exponential attenuation

#### attenuationAlpha
  Single attenuation parameter for simplified Beer-Lambert model

#### horizontalFov
  Horizontal Field of View in degrees

#### verticalFov
  Vertical Field of View in degrees

#### horizontalResolution
  Degrees in between rays for horizontal axis

#### verticalResolution
  Degrees in between rays for vertical axis


### UltrasonicEmitter

#### perRayIntensity
  The base value that is attenuated based on distance from sensor and angle of reflection

#### yawOffset
  Offset along yaw axis to account for sensor having a different forward direction

#### adjacencyList
  List of emitter ids for adjacent emitters, used to compute indirects when receiving. Zero indexed and must match the order in the array


### UltrasonicFiringGroup

#### emitterModes
  List of (emitter id, firing mode) pairs for each sensor in this group to emit from. emitter id is zero indexed and must match the order in the array

#### receiverModes
  List of (receiver id, firing mode) pairs to record envelopes for. Receiver id is zero indexed and must match the order in the array


### Generic

#### samplingRate
  sampling rate of the custom sensor data in Hz

#### streaming
  Streaming lidar point data. Default to true


### UltrasonicMaterialAPI
Defines Ultrasonic (USS) material properties.

#### uss:perceptualRoughness
  Perceptual Roughness. Unitless.

#### uss:reflectance
  Reflectance. Unitless.

#### uss:metallic
  Metallic. Unitless.

#### uss:base_color
  Base Color. Unitless.


## isaacSensorSchema

### IsaacBaseSensor

#### enabled
  Set to True to enable this sensor, False to Disable


### IsaacRtxLidarSensorAPI


### IsaacRtxRadarSensorAPI


### IsaacContactSensor

#### threshold
  Min, Max force that is detected by this sensor, units in (kg) * (stage length unit) / (second)^2

#### radius
  Radius of the contact sensor, unit in stage length unit

#### color
  Color of the contact sensor sphere, R, G, B, A

#### sensorPeriod
  Time between each sensor measurement, unit in simulator seconds


### IsaacImuSensor

#### sensorPeriod
  Time between each sensor measurement, unit in simulator seconds

#### linearAccelerationFilterWidth
  Number of linear acceleration measurements used in the rolling average

#### angularVelocityFilterWidth
  Number of angular velocity measurements used in the rolling average

#### orientationFilterWidth
  Number of orientation measurements used in the rolling average


### IsaacLightBeamSensor

#### numRays
  Number of rays for the light curtain, default 1

#### curtainLength
  Total length of the light curtain

#### forwardAxis
  Direction to shoot the light beam in [AxisX, AxisY, AxisZ]

#### curtainAxis
  Direction to expand the light curtain in [AxisX, AxisY, AxisZ]

#### minRange
  Minimum range for sensor to detect a hit

#### maxRange
  Maximum range for sensor to detect a hit


## robot_schema

### IsaacRobotAPI

#### isaac:changelog

#### isaac:description

#### isaac:license

#### isaac:namespace

#### isaac:robotType

#### isaac:source

#### isaac:version


### IsaacLinkAPI

#### isaac:nameOverride


### IsaacSiteAPI

#### isaac:Description

#### isaac:forwardAxis


### IsaacJointAPI

#### isaac:actuator

#### isaac:NameOverride

#### isaac:physics:DofOffsetOpOrder


### IsaacSurfaceGripper

#### isaac:coaxialForceLimit

#### isaac:maxGripDistance

#### isaac:retryInterval

#### isaac:shearForceLimit

#### isaac:status


### IsaacAttachmentPointAPI

#### isaac:clearanceOffset

#### isaac:forwardAxis

