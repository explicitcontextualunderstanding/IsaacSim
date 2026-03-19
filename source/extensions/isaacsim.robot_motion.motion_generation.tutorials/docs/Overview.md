```{csv-table}
**Extension**: {{ extension_version }},**Documentation Generated**: {sub-ref}`today`
```

# Overview

The isaacsim.robot_motion.motion_generation.tutorials extension provides interactive UI-based tutorials for robot motion generation concepts within Isaac Sim. This extension demonstrates various motion planning algorithms and trajectory generation techniques through hands-on examples with Franka and UR10 robots.

## Key Components

### Tutorial Modules

The extension consists of four specialized tutorial modules, each focusing on different aspects of robot motion generation:

#### RMP Flow Tutorial
The RMP Flow module demonstrates Riemannian Motion Policies (RMP) for reactive motion generation. It provides an interactive interface for loading Franka robot scenarios and running RMP-based motion planning algorithms that generate smooth, collision-aware trajectories in real-time.

#### RRT Tutorial  
The RRT module showcases Rapidly-exploring Random Tree path planning algorithms. Users can experiment with probabilistic sampling-based motion planning techniques that efficiently explore the robot's configuration space to find feasible paths from start to goal configurations.

#### Kinematics Tutorial
The Kinematics module focuses on fundamental robot kinematics concepts. It provides demonstrations of forward and inverse kinematics calculations, joint space manipulations, and the relationship between joint configurations and end-effector poses.

#### Trajectory Generator Tutorial
The Trajectory Generator module presents comprehensive trajectory planning capabilities for UR10 robots. It covers configuration space trajectories, task space trajectories, and advanced trajectory planning methods, demonstrating how to generate smooth, time-parameterized paths for robot motion.

### [UIBuilder](isaacsim.robot_motion.motion_generation.tutorials.rmp_flow/isaacsim.robot_motion.motion_generation.tutorials.rmp_flow.UIBuilder) Architecture

Each tutorial module implements a [UIBuilder](isaacsim.robot_motion.motion_generation.tutorials.rmp_flow/isaacsim.robot_motion.motion_generation.tutorials.rmp_flow.UIBuilder) class that creates a standardized interface with two main control sections:

**World Controls** - Provides Load and Reset buttons for scene management, allowing users to initialize robot scenarios and restore default states.

**Scenario Controls** - Contains interactive buttons for starting and stopping motion generation demonstrations, with state-aware feedback to indicate execution status.

The [UIBuilder](isaacsim.robot_motion.motion_generation.tutorials.rmp_flow/isaacsim.robot_motion.motion_generation.tutorials.rmp_flow.UIBuilder) classes handle timeline integration, physics step callbacks, and stage event management to ensure proper synchronization between the UI and Isaac Sim's simulation systems.

## Functionality

### Interactive Demonstrations

Each tutorial provides hands-on exploration of motion generation concepts through visual demonstrations. Users can load pre-configured robot scenarios, adjust parameters, and observe real-time motion execution within the Isaac Sim environment.

### Event System Integration

The extension integrates with Isaac Sim's timeline and physics systems to provide responsive tutorials. Physics step callbacks ensure continuous updates during simulation, while timeline and stage event handling maintains proper state management throughout the tutorial lifecycle.

### Automated Scene Setup

The tutorials automatically handle scene initialization including lighting setup, camera positioning, and robot asset loading. This ensures optimal visualization conditions for each demonstration without requiring manual configuration.

## Dependencies

The extension uses `isaacsim.robot_motion.lula` for accessing the Lula planning library functionality that powers the motion generation algorithms. It integrates with `isaacsim.robot_motion.motion_generation` to leverage the core motion planning infrastructure. The `isaacsim.gui.components` dependency provides the UI element wrappers used throughout the tutorial interfaces.
