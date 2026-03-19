```{csv-table}
**Extension**: {{ extension_version }},**Documentation Generated**: {sub-ref}`today`
```

# Overview

The isaacsim.ucx.nodes extension provides OmniGraph nodes for high-performance communication using UCX (Unified Communication X) technology. This extension enables distributed simulation scenarios with low-latency data transfer capabilities through specialized OmniGraph nodes that can be integrated into computational graphs.

## Key Components

### UCXBridgeExtension

The `UCXBridgeExtension` class manages the lifecycle of UCX communication nodes within the OmniGraph system. It handles the registration and deregistration of UCX-specific nodes when the extension starts up and shuts down.

### UCX Camera Helper Node

The primary node provided is the UCX Camera Helper (`OgnUCXCameraHelper`), which specializes in transmitting camera data over UCX networks. This node extends the base writer functionality to handle camera-specific data streaming with configurable parameters.

#### Node Inputs
- **execIn**: Execution input for triggering node computation
- **frameSkipCount**: Controls frame rate by skipping specified number of frames
- **port**: Network port for UCX communication
- **renderProductPath**: Path to the render product for camera data
- **resetSimulationTimeOnStop**: Option to reset simulation timing when stopping
- **tag**: Identifier tag for the communication stream
- **useSystemTime**: Toggle between system time and simulation time

#### Node Outputs  
- **execOut**: Execution output for chaining with other nodes

### Internal State Management

The `OgnUCXCameraHelperInternalState` class manages the runtime state of the UCX Camera Helper node, inheriting from `BaseWriterNode` to provide writer-specific functionality. It tracks parameters like reset behavior, publish step size, and maintains the connection state with render products.

## Functionality

The extension integrates UCX communication capabilities into OmniGraph workflows, allowing users to:

- Stream camera data with low-latency communication over distributed networks
- Configure frame rates and timing behavior for optimal performance
- Chain UCX nodes with other OmniGraph nodes through execution ports
- Manage multiple communication streams using port and tag identifiers

## Relationships

The extension builds upon `isaacsim.core.nodes` for base node functionality and integrates with `isaacsim.ucx.core` for the underlying UCX communication infrastructure. It connects with `**omni.replicator.core**` and `**omni.syntheticdata**` to access camera and synthetic data generation capabilities within the Isaac Sim ecosystem.
