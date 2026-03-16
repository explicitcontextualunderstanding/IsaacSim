```{csv-table}
**Extension**: {{ extension_version }},**Documentation Generated**: {sub-ref}`today`
```

# Overview

The isaacsim.gui.content_browser extension enhances the Omniverse content browser by adding an Isaac Sim collection that provides quick access to curated robot models, environments, materials, and sensor assets. This extension creates a dedicated "Isaac Sim" collection within the content browser interface, allowing users to browse and access Isaac Sim-specific assets without manually navigating to remote servers.

## Key Components

### [IsaacCollection](isaacsim.gui.content_browser/isaacsim.gui.content_browser.IsaacCollection)

The [IsaacCollection](isaacsim.gui.content_browser/isaacsim.gui.content_browser.IsaacCollection) class creates the main Isaac Sim collection that appears in the content browser with a cloud icon. This collection automatically detects the appropriate protocol (Omniverse or HTTPS) based on the default asset root configuration and populates itself with configured asset folders from the extension settings.

The collection is read-only and does not support adding new connections. Asset folders are loaded asynchronously from the application settings and displayed as browsable items within the file browser interface.

```python
# The collection automatically populates with configured folders
collection = IsaacCollection()
# Asset folders are loaded from settings and displayed as child items
```

### IsaacConnectionItem

Individual asset folder connections are represented by `IsaacConnectionItem` instances, which are specialized `NucleusItem` objects designed for Isaac Sim asset directories. These items handle the display and navigation of specific asset categories like robots, environments, materials, and sensors.

### [ExtendedFileInfo](isaacsim.gui.content_browser/isaacsim.gui.content_browser.ExtendedFileInfo)

The [ExtendedFileInfo](isaacsim.gui.content_browser/isaacsim.gui.content_browser.ExtendedFileInfo) class extends the detail view controller to provide enhanced file information display within the content browser. This component builds custom headers and presents detailed metadata about selected assets, improving the browsing experience for Isaac Sim content.

## Configuration

The extension uses specific settings to define which asset folders appear in the Isaac Sim collection:

```toml
[settings]
exts."isaacsim.gui.content_browser".folders = [
    "https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Robots",
    "https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Environments",
    "https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Materials",
    # Additional configured folders...
]
```

The `timeout` setting controls the resolution time for content browser settings, ensuring reliable asset discovery and display.

## Integration

This extension integrates with `**omni.kit.window.content_browser**` to register the Isaac Sim collection and uses `**omni.kit.widget.filebrowser**` components for the underlying file browsing functionality. The integration with `**omni.kit.window.filepicker**` provides the collection framework that enables the Isaac Sim assets to appear alongside other content browser collections.
