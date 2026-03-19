# Usage

To enable this extension, go to the Extension Manager menu and enable isaacsim.asset.gen.omap extension.

## Generating an Occupancy Map

1. Open the panel via **Tools > Robotics > Occupancy Map**.
2. Set the origin, upper/lower bounds, and cell size to define the mapping region.
3. Click **Calculate** to compute the occupancy grid from collision geometry in the stage.
4. Click **Visualize** to open the visualization window.

## Visualization Window

The visualization window lets you configure the output before saving:

- **Occupied / Freespace / Unknown Color** — choose the colors used to render each cell type in the image.
- **Rotate Image** — apply a clockwise rotation (0°, 90°, −90°, 180°) to the output image.
- **Coordinate Type** — select the format of the configuration data displayed in the text field:
  - *ROS Occupancy Map Parameters File (YAML)* — generates a ROS-compatible `.yaml` file referencing the image and its map parameters (resolution, origin, thresholds).
  - *Coordinates in Stage Space* — displays the corner coordinates of the map in stage units.
- **Re-Generate Image** — recompute the image after changing any of the above settings.

## Saving the Outputs

After generating the image, two save buttons appear below the map preview:

- **Save Image** — opens a file dialog to save the occupancy map as a `.png` file.
- **Save YAML** — opens a file dialog to save the ROS occupancy map parameters as a `.yaml` file. The YAML file references the image by name and includes the map resolution, origin, and occupancy thresholds. This file can be used directly with ROS navigation tools such as `map_server`.

> **Note:** The YAML file is always saved in ROS format regardless of which Coordinate Type is selected in the dropdown.
