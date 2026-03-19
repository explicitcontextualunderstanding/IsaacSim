# Public API for module isaacsim.examples.browser:

## Classes

- class ExampleBrowserModel(TreeFolderBrowserModel)
  - def __init__(self, *args, **kwargs)
  - def register_example(self, **kwargs)
  - def get_category_items(self, item: CollectionItem) -> List[CategoryItem]
  - def get_detail_items(self, item: ExampleCategoryItem) -> List[ExampleDetailItem]
  - def execute(self, item: ExampleDetailItem)
  - def deregister_example(self, name: str, category: str)
  - def refresh_browser(self)

- class ExampleBrowserWindow(ui.Window)
  - WINDOW_TITLE: str
  - def __init__(self, model: ExampleBrowserModel, visible = True)

## Functions

- def get_instance() -> Optional[ExampleBrowserExtension]
