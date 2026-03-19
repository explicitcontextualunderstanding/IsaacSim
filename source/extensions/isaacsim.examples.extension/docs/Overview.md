```{csv-table}
**Extension**: {{ extension_version }},**Documentation Generated**: {sub-ref}`today`
```

# Overview

The isaacsim.examples.extension provides a code generation tool for creating Isaac Sim extension templates. It offers a user interface that streamlines the development of custom extensions by generating boilerplate code and proper file structures for different workflow types.

```{image} ../../../../source/extensions/isaacsim.examples.extension/data/preview.png
---
align: center
---
```


## Functionality

The extension creates a dockable window with forms that allow developers to generate four different types of extension templates:

- **Configuration Tooling Template** - For building configuration-based workflow extensions
- **Loaded Scenario Template** - For creating scenario loading and management extensions  
- **Scripting Template** - For developing script-based automation extensions
- **UI Component Library Template** - For building reusable UI component extensions

Each template generation requires three inputs: the target directory path, extension title, and extension description. The generator automatically converts extension titles into valid Python package names and creates the complete directory structure with all necessary files.

## Key Components

### [TemplateGenerator](isaacsim.examples.extension/isaacsim.examples.extension.TemplateGenerator)

The [TemplateGenerator](isaacsim.examples.extension/isaacsim.examples.extension.TemplateGenerator) class handles the core template generation functionality. It manages file system operations including directory creation, template file copying, and keyword replacement within template files. The generator maintains consistent extension metadata across all generated templates and ensures proper package naming conventions.

```python
from isaacsim.examples.extension import TemplateGenerator

generator = TemplateGenerator()
generator.generate_scripting_template(
    file_path="/path/to/extensions", 
    extension_title="My Custom Extension",
    extension_description="Extension for custom automation workflows"
)
```

### Template Types

Each template type provides specialized starter code patterns:

#### Configuration Tooling
Generates templates focused on building configuration interfaces and tooling workflows, providing the foundation for extensions that manage settings and parameters.

#### Loaded Scenario  
Creates templates for scenario loading and management, suitable for extensions that handle environment setup and scene configuration.

#### Scripting
Provides templates optimized for automation and scripting workflows, ideal for extensions that automate repetitive tasks or provide programmatic interfaces.

#### Component Library
Generates templates for creating reusable UI components, designed for extensions that contribute widgets or interface elements to the Isaac Sim ecosystem.

## Integration

The extension window integrates into Isaac Sim's docking system, allowing developers to keep the template generator accessible while working on other tasks. The status panel provides real-time feedback during template generation, showing progress and completion status with configurable visibility controls.
