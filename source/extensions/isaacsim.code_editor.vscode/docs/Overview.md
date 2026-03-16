```{csv-table}
**Extension**: {{ extension_version }},**Documentation Generated**: {sub-ref}`today`
```

# Overview

The isaacsim.code_editor.vscode extension enables Visual Studio Code integration with Isaac Sim by creating a bridge for real-time Python code execution. The extension establishes socket-based communication that allows VS Code to send Python code directly to Isaac Sim for execution within its environment, making it possible to develop and test Isaac Sim scripts from within VS Code.

## Functionality

### Code Execution Bridge

The extension creates a socket server that listens for incoming Python code from VS Code extensions or plugins. When code is received, it executes within Isaac Sim's Python environment using the Executor class, which handles both Python statements and expressions. Results, including output, errors, and tracebacks, are transmitted back to VS Code for display.

### Carbonite Log Broadcasting

An optional UDP-based logging feature allows Isaac Sim's Carbonite logging messages to be broadcast to VS Code. This provides developers with real-time access to Isaac Sim's internal logging output directly within their VS Code development environment.

## Key Components

### Executor

The Executor class manages Python code execution within Isaac Sim's environment. It accepts source code as strings and executes them asynchronously, capturing standard output, exceptions, and tracebacks. The executor maintains separate global and local namespaces for code execution scope management.

### UI Builder

The UIBuilder creates menu integration within Isaac Sim, providing users with access to connection status and configuration options. It manages the lifecycle of menu items that allow users to monitor and control the VS Code integration.

## Configuration

The extension provides three key configuration settings:

- `host`: Configures the IP address where the socket server listens for VS Code connections (default: 127.0.0.1)
- `port`: Sets the port number for socket communication (default: 8226)  
- `carb_logs`: Controls whether Carbonite logging messages are broadcast to VS Code (default: false, with warnings about potential application freezing)

## Integration

The extension integrates with **omni.kit.notification_manager** for user notifications and **omni.kit.uiapp** for UI functionality. It also depends on isaacsim.core.deprecation_manager for compatibility management within the Isaac Sim ecosystem.
