```{csv-table}
**Extension**: {{ extension_version }},**Documentation Generated**: {sub-ref}`today`
```

# Overview

The isaacsim.robot.policy.examples extension provides interactive demonstrations of reinforcement learning policy deployment for robotic systems in Isaac Sim. This extension showcases trained policies running on different robot types including manipulator arms, humanoid robots, and quadruped robots, with real-time keyboard control interfaces and GPU-accelerated physics simulation.

## Key Components

### Interactive Robot Examples

The extension provides three main robot demonstration categories through specialized example classes:

**[FrankaExample](isaacsim.robot.policy.examples.interactive.franka/isaacsim.robot.policy.examples.interactive.franka.FrankaExample)** demonstrates a Franka Emika Panda robot performing an autonomous drawer opening task. The robot uses a learned policy to interact with a cabinet, attempting to open and hold a drawer in position. The simulation runs at 200 Hz physics with 60 Hz rendering and automatically resets every 10 seconds for continuous demonstration.

**[HumanoidExample](isaacsim.robot.policy.examples.interactive.humanoid/isaacsim.robot.policy.examples.interactive.humanoid.HumanoidExample)** showcases a Unitree H1 humanoid robot executing flat terrain locomotion policies. The simulation operates with GPU-accelerated physics at 200 Hz and 25 Hz rendering, providing keyboard controls for forward movement and rotational commands. Users can control the robot using arrow keys or numpad inputs for directional movement.

**[QuadrupedExample](isaacsim.robot.policy.examples.interactive.quadruped/isaacsim.robot.policy.examples.interactive.quadruped.QuadrupedExample)** features a Boston Dynamics Spot robot running flat terrain locomotion policies trained in Isaac Lab. The example provides comprehensive keyboard control including forward/backward movement, lateral motion, and yaw rotation commands, running at 500 Hz physics with 50 Hz rendering for smooth real-time interaction.

### Policy Controller Framework

The extension includes a comprehensive `PolicyController` base class that manages the lifecycle of policy-based robot control. This framework handles policy loading from files, robot initialization with configurable control modes (position, velocity, effort), and physics simulation integration. Each robot implementation extends this base controller with specific policy execution logic.

### Configuration System

A flexible configuration system processes environment parameters from YAML files, handling robot joint properties, physics settings, and simulation parameters. The system supports both scalar and per-joint property specifications with pattern matching for joint names, enabling easy customization of robot behaviors and simulation characteristics.

## Integration

The extension integrates with the Isaac Sim examples browser system, registering each robot demonstration under the "Policy" category. Each example provides a complete UI interface with documentation links and keyboard control instructions, making the demonstrations accessible through the standard Isaac Sim interface.

The examples utilize GPU-accelerated physics simulation with PyTorch backend integration for high-performance policy inference, demonstrating the deployment of reinforcement learning policies trained in Isaac Lab within the Isaac Sim environment.
