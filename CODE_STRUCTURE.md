# AI Server - Code Structure

This document describes the refactored code structure for the AI Server project.

## File Organization

The original monolithic `start_llama_webui.py` file has been separated into modular components for better maintainability and organization:

### Core Files

1. **`start_llama_webui.py`** - Main entry point and orchestration
   - Contains the main function and argument parsing
   - Imports and coordinates all other modules
   - Handles Docker operations and service coordination

2. **`service_config.py`** - Configuration management
   - `ServiceConfig` class with all model and service configurations
   - Quantization options and interactive selection methods
   - Model recommendations for different hardware setups

3. **`utility_manager.py`** - Utility functions
   - `UtilityManager` class with subprocess execution helpers
   - Command execution with error handling and output control

4. **`llama_server_manager.py`** - Llama server management
   - `LlamaServerManager` class for llama.cpp server operations
   - Model downloading, validation, and API testing
   - Model file management and legacy compatibility

5. **`ollama_manager.py`** - Ollama service management
   - `OllamaManager` class for Ollama operations
   - Model pulling and API health checking
   - Docker container interaction

6. **`webui_manager.py`** - Web UI management
   - `WebUIManager` class for Open WebUI operations
   - Health checking and readiness validation

## Benefits of This Structure

1. **Modularity**: Each class has a single responsibility
2. **Maintainability**: Easier to modify specific components
3. **Testability**: Individual classes can be tested in isolation
4. **Readability**: Code is organized by functionality
5. **Reusability**: Classes can be imported and used in other projects

## Usage

The main script works exactly as before:

```bash
python start_llama_webui.py --auto
python start_llama_webui.py --cpu-only
python start_llama_webui.py --list-models
```

All functionality remains the same, but the code is now better organized and easier to maintain.

## Dependencies

Each file imports only what it needs:
- `service_config.py`: `os`, `sys`
- `utility_manager.py`: `subprocess`, `sys`
- `llama_server_manager.py`: `os`, `sys`, `time`, `requests`
- `ollama_manager.py`: `time`, `requests`, `subprocess`, `docker`
- `webui_manager.py`: `time`, `requests`
- `start_llama_webui.py`: All the above classes plus `docker`, `argparse`, `webbrowser`
