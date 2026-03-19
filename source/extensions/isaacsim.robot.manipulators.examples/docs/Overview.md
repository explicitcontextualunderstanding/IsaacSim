```{csv-table}
**Extension**: {{ extension_version }},**Documentation Generated**: {sub-ref}`today`
```

# Overview

The isaacsim.robot.manipulators.examples extension provides practical implementations for Franka and Universal Robots manipulators demonstrating common robotics operations in Isaac Sim. This extension serves as a collection of reference implementations showcasing pick-and-place, follow target behaviors, stacking operations, and motion planning tasks that can be used as starting points for robotics applications.

## Key Components

### Robot Implementations

The extension includes two main robot platforms with comprehensive control interfaces:

**Franka Panda Robot** - Features a 7-DOF arm with parallel gripper, supporting inverse kinematics, gripper control, and multiple task implementations including pick-and-place and stacking operations.

**Universal Robots UR10** - Provides a 6-DOF industrial arm with optional surface gripper attachment, offering follow target capabilities and inverse kinematics control with multiple solver methods.

### Interactive Examples

Two interactive examples provide hands-on demonstrations accessible through Isaac Sim's examples browser:

**[FrankaPickPlaceInteractive](isaacsim.robot.manipulators.examples.interactive.pick_place/isaacsim.robot.manipulators.examples.interactive.pick_place.FrankaPickPlaceInteractive)** - Demonstrates a complete pick-and-place operation with the Franka robot, allowing users to execute and observe the multi-phase manipulation sequence through an intuitive UI interface.

**[UR10FollowTargetInteractive](isaacsim.robot.manipulators.examples.interactive.follow_target/isaacsim.robot.manipulators.examples.interactive.follow_target.UR10FollowTargetInteractive)** - Showcases continuous target following behavior where the UR10 robot tracks a movable cube target using real-time inverse kinematics, with selectable IK methods and live status monitoring.

### Task Collections

The extension organizes functionality into specific task categories:

**Pick and Place Tasks** - Implement complete pick-and-place workflows with state machine control, supporting both Franka and UR10 robots with configurable cube positions, target locations, and gripper operations.

**Follow Target Tasks** - Provide continuous tracking behaviors where robots follow moving targets using inverse kinematics, with real-time position updates and distance threshold monitoring.

**Stacking Operations** - Enable multi-object stacking sequences with ordered cube manipulation, demonstrating complex multi-step robotic operations.

## Functionality

### Inverse Kinematics Methods

Both robot platforms support multiple inverse kinematics solution methods including damped-least-squares, pseudoinverse, transpose, and singular-value-decomposition approaches, allowing users to compare different IK solving strategies for their specific applications.

### Motion Planning Integration

The extension integrates with Isaac Sim's motion planning systems, providing RMPFlow controllers for both Franka and UR10 robots that enable collision-aware motion generation and smooth trajectory execution.

### Gripper Control Systems

**Parallel Gripper** (Franka) - Supports precise finger positioning with open/closed states and custom position control for object manipulation.

**Surface Gripper** (UR10) - Implements suction-based grasping with force and torque limit demonstrations, particularly useful for bin filling and similar applications.

## Integration

The extension builds upon Isaac Sim's core robotics framework, utilizing `isaacsim.core.api` for fundamental robot representations and `isaacsim.robot_motion.motion_generation` for advanced motion planning capabilities. The interactive examples integrate with `isaacsim.examples.browser` to provide discoverable demonstrations within Isaac Sim's examples collection.
