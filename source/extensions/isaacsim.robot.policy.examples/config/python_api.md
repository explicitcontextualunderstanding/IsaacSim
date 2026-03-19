# Public API for module isaacsim.robot.policy.examples:

No public API

# Public API for module isaacsim.robot.policy.examples.interactive.franka:

## Classes

- class FrankaExample(BaseSample)
  - def __init__(self)
  - def setup_scene(self)
  - async def setup_post_load(self)
  - async def setup_pre_reset(self)
  - async def setup_post_reset(self)
  - async def setup_post_clear(self)
  - def on_physics_step(self, dt, context)
  - def physics_cleanup(self)

- class FrankaExampleExtension(omni.ext.IExt)
  - def on_startup(self, ext_id: str)
  - def on_shutdown(self)

# Public API for module isaacsim.robot.policy.examples.interactive.humanoid:

## Classes

- class HumanoidExample(BaseSample)
  - def __init__(self)
  - def setup_scene(self)
  - async def setup_post_load(self)
  - async def setup_pre_reset(self)
  - async def setup_post_reset(self)
  - async def setup_post_clear(self)
  - def on_physics_step(self, dt, context)
  - def physics_cleanup(self)

# Public API for module isaacsim.robot.policy.examples.interactive.quadruped:

## Classes

- class QuadrupedExample(BaseSample)
  - def __init__(self)
  - def setup_scene(self)
  - async def setup_post_load(self)
  - async def setup_pre_reset(self)
  - async def setup_post_reset(self)
  - async def setup_post_clear(self)
  - def on_physics_step(self, dt, context)
  - def physics_cleanup(self)
