```{csv-table}
**Extension**: {{ extension_version }},**Documentation Generated**: {sub-ref}`today`
```

# Overview

The `isaacsim.robot.manipulators` extension provides high-level Python classes for controlling robotic manipulators in Isaac Sim. This extension focuses on single-arm manipulator systems that can include end effectors and optional grippers for robotic manipulation tasks.

## Key Components

### [SingleManipulator](isaacsim.robot.manipulators/isaacsim.robot.manipulators.SingleManipulator)

The [SingleManipulator](isaacsim.robot.manipulators/isaacsim.robot.manipulators.SingleManipulator) class serves as the primary interface for manipulator control, built on top of the articulation system. It encapsulates a complete manipulator system consisting of the robotic arm, end effector, and optional gripper components.

Key features include:
- **End Effector Tracking**: Monitors the rigid body corresponding to the end effector through the `end_effector` property, enabling access to world pose and motion data
- **Gripper Integration**: Optional gripper attachment through the `gripper` property for pick-and-place operations
- **Physics Integration**: Utilizes PhysX tensor API through the `initialize()` method for efficient physics simulation
- **State Management**: Provides `post_reset()` functionality to restore default states after timeline resets

The class inherits from `SingleArticulation`, extending it with manipulator-specific functionality while maintaining compatibility with the broader articulation framework.

## Functionality

### Manipulator Configuration

The [SingleManipulator](isaacsim.robot.manipulators/isaacsim.robot.manipulators.SingleManipulator) supports flexible configuration during instantiation:
- Prim path specification for USD stage integration
- End effector path definition for tracking specific rigid bodies
- Transform properties including position, orientation, and scale
- Visual properties and gripper attachment options

### Physics Simulation

Integration with Isaac Sim's physics system occurs through:
- Physics simulation view creation and management
- Articulation view setup using PhysX tensor API
- Reset handling for timeline stop/play cycles

### Component Access

The extension provides property-based access to manipulator components:
- End effector access for pose and velocity queries
- Gripper control for opening, closing, and state monitoring
- Integration with the broader Isaac Sim robotics ecosystem
