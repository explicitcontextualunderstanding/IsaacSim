# Public API for module isaacsim.examples.base:

## Classes

- class BaseSample(object)
  - def __init__(self)
  - def set_world_settings(self, physics_dt = None, stage_units_in_meters = None, rendering_dt = None)
  - async def load_world_async(self)
  - async def reset_async(self)
  - def setup_scene(self)
  - async def setup_post_load(self)
  - async def setup_pre_reset(self)
  - async def setup_post_reset(self)
  - async def setup_post_clear(self)
  - def log_info(self, info)
  - def physics_cleanup(self)
  - async def clear_async(self)

- class BaseSampleUITemplate
  - def __init__(self, *args, **kwargs)
  - [property] def sample(self)
  - [sample.setter] def sample(self, sample)
  - def build_ui(self)
  - def build_default_frame(self)
  - def get_extra_frames_handle(self)
  - def build_extra_frames(self)
  - def post_reset_button_event(self)
  - def post_load_button_event(self)
  - def post_clear_button_event(self)
  - def on_shutdown(self)
  - def on_stage_event(self, event)
