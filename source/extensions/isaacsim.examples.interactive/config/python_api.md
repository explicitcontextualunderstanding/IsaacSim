
# Public API for module isaacsim.examples.interactive.kaya_gamepad:

## Classes

- class KayaGamepad(BaseSample)
  - def __init__(self)
  - def setup_scene(self)
  - async def setup_post_load(self)
  - async def setup_pre_reset(self)
  - async def setup_post_reset(self)
  - async def setup_post_clear(self)
  - def physics_cleanup(self)

# Public API for module isaacsim.examples.interactive.omnigraph_keyboard:

## Classes

- class OmnigraphKeyboard(BaseSample)
  - def __init__(self)
  - def setup_scene(self)
  - async def setup_post_load(self)
  - async def setup_pre_reset(self)
  - async def setup_post_reset(self)
  - async def setup_post_clear(self)

# Public API for module isaacsim.examples.interactive.follow_target:

## Classes

- class FollowTarget(BaseSample)
  - def __init__(self)
  - def setup_scene(self)
  - async def setup_pre_reset(self)
  - def world_cleanup(self)
  - async def setup_post_load(self)

# Public API for module isaacsim.examples.interactive.path_planning:

## Classes

- class PathPlanning(BaseSample)
  - def __init__(self)
  - def setup_scene(self)
  - async def setup_pre_reset(self)
  - def world_cleanup(self)
  - async def setup_post_load(self)

- class PathPlannerController(BaseController)
  - def __init__(self, name: str, path_planner_visualizer: PathPlannerVisualizer, cspace_trajectory_generator: LulaCSpaceTrajectoryGenerator, physics_dt = 1 / 60.0, rrt_interpolation_max_dist = 0.01)
  - def forward(self, target_end_effector_position: np.ndarray, target_end_effector_orientation: Optional[np.ndarray] = None) -> ArticulationAction
  - def add_obstacle(self, obstacle: isaacsim.core.api.objects, static: bool = False)
  - def remove_obstacle(self, obstacle: isaacsim.core.api.objects)
  - def reset(self)
  - def get_path_planner(self) -> PathPlanner

- class PathPlanningTask(BaseTask)
  - def __init__(self, name: str, target_prim_path: Optional[str] = None, target_name: Optional[str] = None, target_position: Optional[np.ndarray] = None, target_orientation: Optional[np.ndarray] = None, offset: Optional[np.ndarray] = None)
  - def set_up_scene(self, scene: Scene)
  - def set_params(self, target_prim_path: Optional[str] = None, target_name: Optional[str] = None, target_position: Optional[np.ndarray] = None, target_orientation: Optional[np.ndarray] = None)
  - def get_params(self) -> dict
  - def get_task_objects(self) -> dict
  - def get_observations(self) -> dict
  - def target_reached(self) -> bool
  - def pre_step(self, time_step_index: int, simulation_time: float)
  - def add_obstacle(self, position: np.ndarray = None, orientation = None)
  - def remove_obstacle(self, name: Optional[str] = None)
  - def get_obstacles(self) -> List
  - def get_obstacle_to_delete(self)
  - def obstacles_exist(self) -> bool
  - def cleanup(self)
  - def get_custom_gains(self) -> Tuple[np.array, np.array]

# Public API for module isaacsim.examples.interactive.bin_filling:

## Classes

- class BinFilling(BaseSample)
  - def __init__(self)
  - def setup_scene(self)
  - async def setup_post_load(self)
  - async def on_fill_bin_event_async(self)
  - async def setup_pre_reset(self)
  - def world_cleanup(self)

# Public API for module isaacsim.examples.interactive.franka_cortex:

No public API

# Public API for module isaacsim.examples.interactive.robo_factory:

## Classes

- class RoboFactory(BaseSample)
  - def __init__(self)
  - def setup_scene(self)
  - async def setup_post_load(self)
  - async def setup_pre_reset(self)
  - async def setup_post_reset(self)
  - async def setup_post_clear(self)
  - def physics_cleanup(self)

# Public API for module isaacsim.examples.interactive.robo_party:

## Classes

- class RoboParty(BaseSample)
  - def __init__(self)
  - def setup_scene(self)
  - async def setup_post_load(self)
  - async def setup_pre_reset(self)
  - def world_cleanup(self)

# Public API for module isaacsim.examples.interactive.hello_world:

## Classes

- class HelloWorld(BaseSample)
  - def __init__(self)
  - def setup_scene(self)
  - async def setup_post_load(self)
  - async def setup_pre_reset(self)
  - async def setup_post_reset(self)
  - def world_cleanup(self)

- class HelloWorldExtension(omni.ext.IExt)
  - def on_startup(self, ext_id: str)
  - def on_shutdown(self)

# Public API for module isaacsim.examples.interactive.replay_follow_target:

## Classes

- class ReplayFollowTarget(BaseSample)
  - def __init__(self)
  - def setup_scene(self)
  - async def setup_post_load(self)
  - async def setup_pre_reset(self)
  - async def setup_post_reset(self)
  - async def setup_post_clear(self)
  - def physics_cleanup(self)

# Public API for module isaacsim.examples.interactive.surface_gripper:

No public API

# Public API for module isaacsim.examples.interactive.ur10_palletizing:

No public API

# Public API for module isaacsim.examples.interactive.user_examples:

No public API

# Public API for module isaacsim.examples.interactive.getting_started:

## Classes

- class GettingStarted(BaseSample)
  - def __init__(self)
  - [property] def name(self) -> str
  - def setup_scene(self)
  - async def setup_post_load(self)
  - async def setup_pre_reset(self)
  - async def setup_post_reset(self)
  - async def setup_post_clear(self)
  - def physics_cleanup(self)

- class GettingStartedRobot(BaseSample)
  - def __init__(self)
  - def setup_scene(self)
  - async def setup_post_load(self)
  - def on_physics_step(self, step_size, context)
  - async def setup_pre_reset(self)
  - async def setup_post_reset(self)
  - async def setup_post_clear(self)
  - def physics_cleanup(self)
