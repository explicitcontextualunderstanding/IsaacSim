```{csv-table}
**Extension**: {{ extension_version }},**Documentation Generated**: {sub-ref}`today`
```

# Overview

The isaacsim.replicator.examples extension provides example implementations and demonstrations of Replicator functionality within Isaac Sim for synthetic data generation workflows in robotics simulation and machine learning.

## Key Components

### OmniGraph Nodes

The extension includes three specialized sampling nodes built on the OmniGraph framework that enable spatial positioning operations for objects in 3D space:

#### OgnSampleBetweenSpheres

**Samples objects in the region between two concentric spheres.** This node takes a collection of prims and positions them randomly within the annular space defined by two sphere radii. It accepts `prims` (objects to position), `radius1` and `radius2` (inner and outer sphere boundaries), and provides execution flow control through `execIn` and `execOut` ports.

#### OgnSampleInSphere  

**Samples objects within a spherical volume.** This node positions prims randomly inside a sphere of specified radius. It uses `prims` (target objects), `radius` (sphere boundary), and execution ports for integration into graph workflows.

#### OgnSampleOnSphere

**Samples objects on a spherical surface.** This node distributes prims randomly across the surface of a sphere at a given radius. It operates with `prims` (objects to position), `radius` (sphere surface distance), and execution flow ports.

### Database Classes

Each sampling node includes a corresponding database class (`OgnSampleBetweenSpheresDatabase`, `OgnSampleInSphereDatabase`, `OgnSampleOnSphereDatabase`) that provides simplified access to node data through the OmniGraph framework. These classes handle attribute management, input/output properties, and node lifecycle operations.

## Functionality

The extension demonstrates key Replicator concepts through practical spatial sampling operations commonly used in robotics and machine learning data generation:

- **Spatial Randomization**: Shows how to implement different spatial distribution patterns for objects in 3D scenes
- **Graph-Based Workflows**: Demonstrates integration with OmniGraph for building complex data generation pipelines  
- **Execution Flow Control**: Provides examples of managing execution order in graph-based synthetic data workflows

## Integration

The extension integrates with Isaac Sim's synthetic data generation ecosystem through its dependencies on **omni.replicator.core** and isaacsim.replicator.writers. The OmniGraph nodes can be combined with other Replicator components to create sophisticated randomization workflows for training data generation in robotics applications.
