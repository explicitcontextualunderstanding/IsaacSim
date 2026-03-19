```{csv-table}
**Extension**: {{ extension_version }},**Documentation Generated**: {sub-ref}`today`
```

# Overview

The isaacsim.robot_motion.experimental.motion_generation extension provides APIs for interfacing external motion generation code with Isaac Sim. This extension enables developers to build motion planning systems by providing core components for trajectory generation, controller composition, obstacle representation, and world synchronization.

<div align="center">

```{mermaid}
graph TD
    %% Inheritance relationships
    BaseController --> ControllerContainer
    BaseController --> ParallelController
    BaseController --> SequentialController
    BaseController --> TrajectoryFollower
```

</div>

## Key Components

### Controllers

The extension provides a flexible controller framework built around the [BaseController](isaacsim.robot_motion.experimental.motion_generation/isaacsim.robot_motion.experimental.motion_generation.BaseController) interface. The [ControllerContainer](isaacsim.robot_motion.experimental.motion_generation/isaacsim.robot_motion.experimental.motion_generation.ControllerContainer) enables runtime switching between different controllers using selection keys, while [ParallelController](isaacsim.robot_motion.experimental.motion_generation/isaacsim.robot_motion.experimental.motion_generation.ParallelController) runs multiple controllers simultaneously and combines their outputs. The [SequentialController](isaacsim.robot_motion.experimental.motion_generation/isaacsim.robot_motion.experimental.motion_generation.SequentialController) chains controllers together, feeding the output of one controller as the setpoint to the next.

The [TrajectoryFollower](isaacsim.robot_motion.experimental.motion_generation/isaacsim.robot_motion.experimental.motion_generation.TrajectoryFollower) controller implements trajectory execution, requiring a trajectory to be set before use and handling timing bounds automatically.

### State Representation

Robot state is represented through composite data structures. The [RobotState](isaacsim.robot_motion.experimental.motion_generation/isaacsim.robot_motion.experimental.motion_generation.RobotState) class aggregates joint states, root link states, rigid body states, and site states. [JointState](isaacsim.robot_motion.experimental.motion_generation/isaacsim.robot_motion.experimental.motion_generation.JointState) manages joint positions, velocities, and efforts using warp arrays for efficient computation. [SpatialState](isaacsim.robot_motion.experimental.motion_generation/isaacsim.robot_motion.experimental.motion_generation.SpatialState) handles pose and twist data for reference frames, while [RootState](isaacsim.robot_motion.experimental.motion_generation/isaacsim.robot_motion.experimental.motion_generation.RootState) specifically manages the robot's root link state.

### [Trajectory](isaacsim.robot_motion.experimental.motion_generation/isaacsim.robot_motion.experimental.motion_generation.Trajectory) and [Path](isaacsim.robot_motion.experimental.motion_generation/isaacsim.robot_motion.experimental.motion_generation.Path) Planning

The [Path](isaacsim.robot_motion.experimental.motion_generation/isaacsim.robot_motion.experimental.motion_generation.Path) class represents joint-space waypoints connected linearly and provides conversion to minimal-time trajectories using velocity and acceleration constraints. The abstract [Trajectory](isaacsim.robot_motion.experimental.motion_generation/isaacsim.robot_motion.experimental.motion_generation.Trajectory) interface defines continuous-time trajectories with duration and state sampling capabilities.

### Obstacle Management

The [ObstacleStrategy](isaacsim.robot_motion.experimental.motion_generation/isaacsim.robot_motion.experimental.motion_generation.ObstacleStrategy) system manages obstacle representations for planning. [ObstacleConfiguration](isaacsim.robot_motion.experimental.motion_generation/isaacsim.robot_motion.experimental.motion_generation.ObstacleConfiguration) pairs representation types with safety tolerances, while [ObstacleRepresentation](isaacsim.robot_motion.experimental.motion_generation/isaacsim.robot_motion.experimental.motion_generation.ObstacleRepresentation) enumerates supported geometric primitives including spheres, cubes, capsules, meshes, and oriented bounding boxes.

### World Interface

The [WorldBinding](isaacsim.robot_motion.experimental.motion_generation/isaacsim.robot_motion.experimental.motion_generation.WorldBinding) class synchronizes USD stage objects with planning world implementations through the [WorldInterface](isaacsim.robot_motion.experimental.motion_generation/isaacsim.robot_motion.experimental.motion_generation.WorldInterface). It uses USDRT change tracking to efficiently mirror tracked prims into planning representations, handling transforms, collision states, and shape properties.

The [SceneQuery](isaacsim.robot_motion.experimental.motion_generation/isaacsim.robot_motion.experimental.motion_generation.SceneQuery) class provides spatial queries against the USD stage, enabling searches for objects with specific APIs within axis-aligned bounding boxes.

## Integration

The extension integrates with isaacsim.core.experimental.objects for shape definitions and isaacsim.core.experimental.utils for core functionality. The [WorldBinding](isaacsim.robot_motion.experimental.motion_generation/isaacsim.robot_motion.experimental.motion_generation.WorldBinding) component specifically bridges USD scene data with motion planning algorithms by tracking CollisionAPI and RigidBodyAPI objects defined by [TrackableApi](isaacsim.robot_motion.experimental.motion_generation/isaacsim.robot_motion.experimental.motion_generation.TrackableApi).
