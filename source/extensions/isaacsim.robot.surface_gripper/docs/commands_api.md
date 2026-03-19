# Commands
Public command API for module **isaacsim.robot.surface_gripper**:

- [CreateSurfaceGripper](#createsurfacegripper)


## CreateSurfaceGripper
Creates Action graph containing a Surface Gripper node, and all prims to facilitate its creation

Typical usage example:

.. code-block:: python

result, prim  = omni.kit.commands.execute(
"CreateSurfaceGripper",
prim_path="/SurfaceGripper",
)

### Arguments
- prim_path

### Usage

```python
import omni.kit.commands
import omni.usd

# Create a surface gripper at a specific path
result, prim = omni.kit.commands.execute(
    "CreateSurfaceGripper",
    prim_path="/World/SurfaceGripper"
)
```

