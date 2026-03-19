# Public API for module isaacsim.robot_setup.wizard:

## Classes

- class RobotWizardWindow
  - def __init__(self, window_title)
  - def reset_progress(self)
  - def destroy(self)
  - def set_visible(self, visible: bool)
  - def set_visibility_changed_listener(self, listener: callable)
  - def on_build_steps(self)
  - def on_build_content(self)
  - def update_page(self, page_name)

## Functions

- def get_window() -> RobotWizardWindow | None
