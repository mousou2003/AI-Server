# OllamaCustomModel - Generalized AI Assistant Framework

The `OllamaCustomModel` class provides a flexible framework for creating specialized AI assistants using Ollama models with custom system prompts and configurations. This system generalizes the original `QwenChurnAssistantManager` to work with any template-based AI assistant.

## Features

- 🔧 **Template-based Configuration**: Load custom system prompts and model parameters from JSON templates
- 🐳 **Docker Infrastructure Management**: Automated Docker Compose setup with project-specific configurations  
- 🧠 **Model Customization**: Create specialized models with embedded system prompts
- ⚡ **CPU/GPU Flexibility**: Support for both CPU-only and GPU-accelerated deployments
- 🔄 **Factory Methods**: Easy creation of specific assistant types
- ✅ **Template Validation**: Ensure template files have proper structure and valid parameters

## Quick Start

### Using the Factory Methods

```python
from ollama_custom_model import OllamaCustomModel

# Create a yoga sequence assistant
yoga_assistant = OllamaCustomModel.create_yoga_assistant(cpu_mode=True)
yoga_assistant.start_infrastructure()

# Create a churn analysis assistant  
churn_assistant = OllamaCustomModel.create_churn_assistant(cpu_mode=False)
churn_assistant.start_infrastructure()
```

### Using Custom Templates

```python
# Create an assistant from any template file
custom_assistant = OllamaCustomModel(
    template_path="templates/my_custom_template.json",
    cpu_mode=False,
    project_name="my-assistant"
)
custom_assistant.start_infrastructure()
```

### Command Line Usage

```bash
# Start yoga assistant
python start_yoga_assistant.py --cpu

# Start any custom assistant
python start_custom_assistant.py templates/yoga_sequence_system_prompt.template.json --cpu

# Check status
python start_custom_assistant.py templates/yoga_sequence_system_prompt.template.json --status

# Stop infrastructure
python start_custom_assistant.py templates/yoga_sequence_system_prompt.template.json --stop
```

## Template File Format

Templates are JSON files that define the assistant's behavior and configuration:

```json
{
  "name": "My Specialized Assistant",
  "description": "A specialized AI assistant for specific tasks",
  "system_prompt": "You are a specialized assistant that...",
  "model_parameters": {
    "temperature": 0.2,
    "top_k": 40,
    "top_p": 0.9,
    "repeat_penalty": 1.1
  },
  "recommended_model": "qwen2.5:7b-instruct",
  "usage_instructions": "Tips for using this assistant effectively"
}
```

### Required Fields

- `name`: Display name for the assistant
- `description`: Brief description of the assistant's purpose  
- `system_prompt`: The system prompt that defines the assistant's behavior

### Optional Fields

- `model_parameters`: Ollama model parameters (temperature, top_k, etc.)
- `recommended_model`: Suggested base model to use
- `usage_instructions`: Instructions shown to users after startup
- Additional fields for specialized configurations

## Architecture

### Core Components

1. **TemplateLoader**: Loads and validates JSON template files
2. **OllamaCustomModel**: Main class managing the complete infrastructure
3. **Docker Integration**: Automated container orchestration with project-specific configurations
4. **Model Management**: Automatic custom model creation with embedded system prompts

### File Structure

```
AI-Server/
├── ollama_custom_model.py              # Main framework class
├── start_yoga_assistant.py             # Yoga assistant starter script
├── start_custom_assistant.py           # Generic assistant starter script
├── templates/
│   ├── docker-compose.generic-override.template.yml  # Docker template
│   ├── yoga_sequence_system_prompt.template.json
│   └── your_custom_template.json
└── docker-compose.{project}-override.yml  # Auto-generated project files
```

## Docker Infrastructure

The system automatically manages Docker infrastructure with:

- **Base Services**: Ollama API server and Open WebUI interface
- **Project Isolation**: Each assistant gets its own containers and volumes
- **Auto-Generated Overrides**: Project-specific Docker configurations created automatically
- **Network Management**: External AI network for service communication
- **GPU Support**: Conditional GPU acceleration based on system capabilities

### Container Naming Convention

- Ollama: `ollama-{project-name}`
- WebUI: `open-webui-{project-name}`
- Volumes: `{project-name}-data`, `{project-name}-workspace`

## Examples

### Example 1: Yoga Sequence Assistant

The yoga sequence assistant generates complete yoga class sequences with proper flow, balance calculations, and instructor notes.

**Template**: `templates/yoga_sequence_system_prompt.template.json`

**Usage**:
```bash
python start_yoga_assistant.py --cpu
```

**Example Prompts**:
- "Create a 60-minute restorative yoga class for beginners"
- "Generate a power yoga sequence focusing on core strength"
- "Design a hip-opening flow for tight office workers"

### Example 2: Custom Business Assistant

Create a template for business analysis:

```json
{
  "name": "Business Strategy Assistant", 
  "description": "Specialized assistant for business strategy and analysis",
  "system_prompt": "You are a business strategy expert who provides clear, actionable insights for strategic decision-making. Focus on market analysis, competitive positioning, and growth strategies. Always provide specific recommendations with supporting rationale.",
  "model_parameters": {
    "temperature": 0.3,
    "top_k": 50,
    "top_p": 0.85
  }
}
```

**Usage**:
```bash
python start_custom_assistant.py templates/business_strategy.template.json
```

### Example 3: Educational Tutor

```json
{
  "name": "Math Tutor Assistant",
  "description": "Patient and encouraging math tutor for students",
  "system_prompt": "You are a patient, encouraging math tutor. Break down complex problems into manageable steps, provide clear explanations, and offer positive reinforcement. Adapt your teaching style to the student's level and learning pace.",
  "model_parameters": {
    "temperature": 0.1,
    "top_k": 30,
    "top_p": 0.9
  },
  "usage_instructions": "Ask math questions at any level - from basic arithmetic to advanced calculus"
}
```

## Advanced Configuration

### Model Selection

The system intelligently selects appropriate models based on:

1. **Explicit specification**: Via `model_name` parameter or `recommended_model` in template
2. **Template analysis**: Automatic inference based on template content
3. **Hardware constraints**: CPU vs GPU mode considerations
4. **Size preferences**: Large model flag for enhanced capabilities

### CPU vs GPU Mode

- **CPU Mode** (`--cpu`): Uses smaller models, slower but works without GPU
- **GPU Mode** (default): Uses GPU acceleration when available
- **Large Model** (`--large`): Forces larger model variants for enhanced capabilities

### Project Isolation

Each assistant instance is completely isolated with:

- Separate Docker containers
- Independent data volumes  
- Project-specific configurations
- Isolated model storage

## Troubleshooting

### Common Issues

1. **Template not found**: Ensure template file exists and path is correct
2. **Docker not running**: Start Docker Desktop/service before running assistants
3. **Port conflicts**: Each project uses the same ports but different containers
4. **Memory issues**: Use CPU mode or smaller models on resource-constrained systems

### Checking Status

```bash
# Check infrastructure status
python start_custom_assistant.py templates/your_template.json --status

# View container logs
docker logs ollama-your-project-name
docker logs open-webui-your-project-name
```

### Complete Cleanup

```bash
# Stop with volume cleanup
python start_custom_assistant.py templates/your_template.json --stop --remove-volumes
```

## System Architecture

The OllamaCustomModel framework uses a modular approach:

### Core Components:
- **Template System**: JSON templates define assistant behavior and configuration
- **Docker Compose Overlay**: Modular compose files for different deployment scenarios
- **Model Management**: Automated model pulling and custom model creation
- **Infrastructure Orchestration**: Coordinated startup of Ollama and WebUI services

### Template-Driven Approach:
```python
from ollama_custom_model import OllamaCustomModel

# Create any type of assistant using templates
assistant = OllamaCustomModel(
    template_path="templates/your_assistant.template.json",
    cpu_mode=True,
    project_name="your-assistant"
)
assistant.start_infrastructure()
```

This generalized approach supports unlimited assistant types while maintaining consistent infrastructure management.

## API Reference

### OllamaCustomModel Class

#### Constructor
```python
OllamaCustomModel(
    template_path: str,
    model_name: str = None,
    cpu_mode: bool = False,
    large_model: bool = False,
    quiet_mode: bool = False,
    project_name: str = None
)
```

#### Key Methods
- `start_infrastructure()`: Start the complete infrastructure
- `stop_infrastructure(remove_volumes=False)`: Stop infrastructure 
- `status()`: Check status of all components
- `create_custom_model()`: Create model with embedded system prompt
- `wait_for_services()`: Wait for services to be ready

#### Factory Methods
- `create_yoga_assistant(**kwargs)`: Create yoga sequence assistant
- `create_churn_assistant(**kwargs)`: Create churn analysis assistant
- `create_from_template(template_path, **kwargs)`: Generic factory method

### TemplateLoader Class

#### Static Methods
- `load_template(template_path)`: Load and parse template JSON
- `validate_template(template_data)`: Validate template structure