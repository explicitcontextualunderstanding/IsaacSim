```{csv-table}
**Extension**: {{ extension_version }},**Documentation Generated**: {sub-ref}`today`
```

# Overview

The isaacsim.examples.interactive extension provides a comprehensive collection of interactive robotics examples for Isaac Sim, demonstrating key concepts in robot manipulation, path planning, multi-robot coordination, and behavior-driven programming. Each example includes both programmatic APIs and interactive user interfaces, making them valuable for learning robotics concepts, testing implementations, and serving as templates for custom applications.

```{image} ../../../../source/extensions/isaacsim.examples.interactive/data/preview.png
---
align: center
---
```


## Key Components

### Robot Manipulation Examples

**[FollowTarget](isaacsim.examples.interactive.follow_target/isaacsim.examples.interactive.follow_target.FollowTarget)** demonstrates real-time target following using a Franka robot with RMP Flow control. The robot continuously tracks a movable target cube while avoiding dynamically added obstacles, showcasing reactive motion planning and collision avoidance.

**[PathPlanning](isaacsim.examples.interactive.path_planning/isaacsim.examples.interactive.path_planning.PathPlanning)** showcases RRT-based path planning with a Franka robot navigating through complex static environments. Users can add and remove wall obstacles to create challenging scenarios that require sophisticated motion planning algorithms.

**[BinFilling](isaacsim.examples.interactive.bin_filling/isaacsim.examples.interactive.bin_filling.BinFilling)** illustrates industrial automation scenarios with a UR10 robot performing bin filling tasks. The example includes realistic gripper failure conditions and demonstrates surface gripper behavior under various load conditions.

### Input Device Integration

**[KayaGamepad](isaacsim.examples.interactive.kaya_gamepad/isaacsim.examples.interactive.kaya_gamepad.KayaGamepad)** provides gamepad control integration for the NVIDIA Kaya robot, demonstrating how to connect physical input devices with robotic systems in simulation for teleoperation scenarios.

**[OmnigraphKeyboard](isaacsim.examples.interactive.omnigraph_keyboard/isaacsim.examples.interactive.omnigraph_keyboard.OmnigraphKeyboard)** shows keyboard input processing through Omni Graph nodes, allowing users to control object properties (like cube size) using keyboard inputs, illustrating the integration between user input and graph-based programming.

### Multi-Robot Coordination

**[RoboParty](isaacsim.examples.interactive.robo_party/isaacsim.examples.interactive.robo_party.RoboParty)** demonstrates concurrent operation of multiple robot types including Franka manipulators, UR10 arms, Kaya holonomic robots, and Jetbot differential drive robots, each performing specialized tasks simultaneously.

**[RoboFactory](isaacsim.examples.interactive.robo_factory/isaacsim.examples.interactive.robo_factory.RoboFactory)** showcases multiple robots performing stacking operations in a coordinated manner, illustrating multi-robot task coordination and workspace sharing.

### Behavior Framework Examples

**FrankaCortex** integrates the Cortex behavior framework with Franka robots, providing examples of block stacking, state machines, decider networks, and interactive games using behavior-driven programming paradigms.

### Educational and Template Examples

**[HelloWorld](isaacsim.examples.interactive.hello_world/isaacsim.examples.interactive.hello_world.HelloWorld)** serves as the foundational template demonstrating basic Isaac Sim sample structure and scene setup patterns.

**[GettingStarted](isaacsim.examples.interactive.getting_started/isaacsim.examples.interactive.getting_started.GettingStarted)** and **[GettingStartedRobot](isaacsim.examples.interactive.getting_started/isaacsim.examples.interactive.getting_started.GettingStartedRobot)** provide step-by-step tutorials for Isaac Sim fundamentals, covering scene creation, physics setup, robot integration, and basic simulation concepts.

**[ReplayFollowTarget](isaacsim.examples.interactive.replay_follow_target/isaacsim.examples.interactive.replay_follow_target.ReplayFollowTarget)** demonstrates data logging and replay capabilities, showing how to record robot trajectories and scene states for analysis and reproduction.

## Functionality

Each example follows a consistent architecture pattern with scene setup, interactive UI controls, data logging capabilities, and proper cleanup handling. The examples support both programmatic access through their respective sample classes and interactive operation through integrated user interfaces.

Most manipulation examples include obstacle management systems allowing users to dynamically add and remove barriers during simulation. Path planning examples demonstrate collision avoidance algorithms, while multi-robot examples showcase coordination strategies and workspace management.

The behavior-based examples leverage the Cortex framework for decision-making, providing real-time diagnostic information and decision stack visualization for understanding robot behavior execution.

## Integration

The extension integrates with several Isaac Sim frameworks including the examples browser system for easy access, the Cortex behavior framework for advanced robot programming, and various robot manipulation libraries for control algorithms. Examples register themselves with the browser system and provide consistent UI patterns through shared base classes.
