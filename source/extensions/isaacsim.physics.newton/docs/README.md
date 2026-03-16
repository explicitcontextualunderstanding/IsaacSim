# Isaac Sim Newton Physics Extension

This extension integrates the Newton physics engine into Isaac Sim, providing GPU-accelerated physics simulation with support for multiple integrators (XPBD, MuJoCo, Featherstone, Semi-Implicit).

## Features

- Integration with the unified physics interface (`omni.physics.core`)
- Tensor-based API for NumPy, PyTorch, and Warp backends
- Support for articulations, rigid bodies, and contacts
- CUDA graph capture for optimized GPU performance

## Usage

```python
import isaacsim.physics.newton

# Switch to Newton physics engine
from isaacsim.core.simulation_manager import SimulationManager
SimulationManager.switch_physics_engine("newton")

# Get the Newton stage
newton_stage = isaacsim.physics.newton.acquire_stage()

# Create tensor views for physics data access
import isaacsim.physics.newton.tensors as newton_tensors
sim_view = newton_tensors.create_simulation_view("warp", newton_stage)
```

## Configuration

The extension can be configured via carb settings:

- `/exts/isaacsim.physics.newton/capture_graph_physics_step`: Enable CUDA graph capture for physics stepping (default: true)
- `/exts/isaacsim.physics.newton/auto_switch_on_startup`: Automatically switch to Newton when extension loads (default: true)

## Submodules

- `isaacsim.physics.newton.tensors`: Tensor-based interface for physics data access
