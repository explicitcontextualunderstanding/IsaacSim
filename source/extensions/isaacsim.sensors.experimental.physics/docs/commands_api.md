# Commands
Public command API for module **isaacsim.sensors.experimental.physics**:

- [IsaacSensorExperimentalCreateContactSensor](#isaacsensorexperimentalcreatecontactsensor)
- [IsaacSensorExperimentalCreateImuSensor](#isaacsensorexperimentalcreateimusensor)
- [IsaacSensorExperimentalCreatePrim](#isaacsensorexperimentalcreateprim)


## IsaacSensorExperimentalCreateContactSensor
Command for creating a contact sensor prim.

Creates an IsaacContactSensor prim under the specified parent with
configurable threshold, radius, and color. Also applies the
PhysxContactReportAPI to the parent prim to enable contact reporting.

### Arguments
- path: Relative path for the sensor (appended to parent).
- parent: Parent prim path. Must have a collision-enabled prim.
- min_threshold: Minimum force threshold in Newtons.
- max_threshold: Maximum force threshold in Newtons.
- color: Visualization color as [r, g, b, a].
- radius: Contact detection radius. Negative disables radius filtering.
- translation: Local translation offset.

### Usage

```python
import omni.kit.commands
from pxr import Gf

# Create a contact sensor on a parent prim with custom settings
omni.kit.commands.execute(
    "IsaacSensorExperimentalCreateContactSensor",
    path="/Contact_Sensor",
    parent="/World/Cube",
    min_threshold=10.0,
    max_threshold=5000.0,
    color=Gf.Vec4f(1.0, 0.0, 0.0, 1.0),  # Red color
    radius=0.5,
    translation=Gf.Vec3d(0.0, 0.0, 1.0)
)
```

## IsaacSensorExperimentalCreateImuSensor
Command for creating an IMU sensor prim.

Creates an IsaacImuSensor prim under the specified parent with
configurable filter widths.

### Arguments
- path: Relative path for the sensor (appended to parent).
- parent: Parent prim path.
- translation: Local translation offset.
- orientation: Local orientation as quaternion [w, x, y, z].
- linear_acceleration_filter_size: Rolling average window for acceleration.
- angular_velocity_filter_size: Rolling average window for angular velocity.
- orientation_filter_size: Rolling average window for orientation.

### Usage

```python
import omni.kit.commands
from pxr import Gf

# Create an IMU sensor on a parent prim with custom settings
omni.kit.commands.execute(
    "IsaacSensorExperimentalCreateImuSensor",
    path="/Imu_Sensor",
    parent="/World/Robot/base_link",
    translation=Gf.Vec3d(0.1, 0.0, 0.05),
    orientation=Gf.Quatd(1, 0, 0, 0),
    linear_acceleration_filter_size=5,
    angular_velocity_filter_size=3,
    orientation_filter_size=2
)
```

## IsaacSensorExperimentalCreatePrim
Base command for creating Isaac Sensor prims.

Creates a sensor prim at the specified path with the given schema type
and transform. This is a base command used by specific sensor creation
commands.

### Arguments
- path: Relative path for the new prim (appended to parent).
- parent: Parent prim path. If empty, path is used as absolute.
- translation: Local translation offset.
- orientation: Local orientation as quaternion [w, x, y, z].
- schema_type: USD schema type for the sensor prim.

### Usage

```python
import omni.kit.commands
import omni.isaac.IsaacSensorSchema as IsaacSensorSchema
from pxr import Gf

# Create a basic Isaac sensor prim
omni.kit.commands.execute(
    "IsaacSensorExperimentalCreatePrim",
    path="/MySensor",
    parent="/World/Robot",
    translation=Gf.Vec3d(0, 0, 1),
    orientation=Gf.Quatd(1, 0, 0, 0),
    schema_type=IsaacSensorSchema.IsaacBaseSensor
)
```

