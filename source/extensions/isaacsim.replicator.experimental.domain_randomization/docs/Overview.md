```{csv-table}
**Extension**: {{ extension_version }},**Documentation Generated**: {sub-ref}`today`
```

# Overview

The isaacsim.replicator.experimental.domain_randomization extension provides domain randomization capabilities for reinforcement learning and simulation-to-real transfer applications in Isaac Sim. This extension offers specialized OmniGraph nodes that enable dynamic randomization of physics simulation parameters across multiple environments, supporting both individual environment resets and interval-based randomization patterns.

## Key Components

### OmniGraph Nodes

The extension provides several specialized OmniGraph nodes for domain randomization workflows:

**OgnOnRLFrame** manages frame-by-frame execution in reinforcement learning scenarios. It tracks environment states and outputs frame numbers and reset indices for environments that need randomization.

**OgnIntervalFiltering** controls randomization timing by filtering execution based on configurable frame intervals. Users can specify intervals and ignore patterns to create sophisticated randomization schedules.

**OgnWritePhysicsSimulationContext** applies randomization to global simulation parameters like gravity using different operation types (direct assignment, additive modification, or scaling).

**OgnWritePhysicsRigidPrimView** randomizes properties of rigid body objects including position, orientation, velocity, mass, material properties, and contact parameters.

**OgnWritePhysicsArticulationView** handles randomization for articulated bodies, supporting joint properties like stiffness, damping, limits, and tendon parameters for complex robotic systems.

**OgnCountIndices** provides utility functionality to count the number of indices being processed, useful for debugging and monitoring randomization operations.

### Physics View Integration

The extension integrates with Isaac Sim's physics view system to enable efficient batch randomization operations. Users register RigidPrim and Articulation views by name, then reference these registered views in randomization functions.

The physics view integration supports three operation modes:
- **Direct**: Replace existing values with randomized samples
- **Additive**: Add randomized values to current properties
- **Scaling**: Multiply current values by randomized factors

### Context Management

The ReplicatorIsaacContext class manages randomization state across multiple environments. It tracks reset indices, coordinates trigger events, and maintains execution context for complex randomization workflows involving tendon properties and multi-environment scenarios.

## Functionality

### Trigger Gates

The extension provides gate functions that determine when randomization should occur. The `on_interval()` function creates interval-based triggers that activate randomization at specified frequencies. The `on_env_reset()` function triggers randomization whenever environments are reset, ensuring consistent initial conditions.

### Distribution Support

All randomization functions accept ReplicatorItem distributions from the **omni.replicator.core** system. This enables sophisticated sampling strategies including uniform, normal, and custom distributions with configurable parameters.

### Attribute Randomization

The extension supports randomization of numerous physics attributes organized into categories:

**Simulation Context**: Global parameters like gravity that affect the entire simulation environment.

**Rigid Body Properties**: Individual object characteristics including transforms, dynamics, and material properties.

**Articulation Properties**: Joint-specific parameters, body properties, and tendon configurations for complex mechanisms.

## Relationships

This extension builds upon **omni.replicator.core** for distribution sampling and **omni.graph.core** for OmniGraph node infrastructure. It integrates with isaacsim.core.experimental.prims for physics view management and isaacsim.core.simulation_manager for simulation context control. The extension coordinates with these systems to provide domain randomization capabilities that work seamlessly within Isaac Sim's physics simulation pipeline.
