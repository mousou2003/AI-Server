# 🎯 **AI-Server: Complete Local AI Infrastructure**

## 📋 **Table of Contents**

### Core Platform
- [🎯 Platform Overview](#-platform-overview)
- [🚀 Key Features](#-key-features) 
- [🏗️ System Architecture](#%EF%B8%8F-system-architecture)
- [📦 Tech Stack](#-tech-stack)
- [📂 Project Structure](#-project-structure)

### Getting Started
- [📋 Prerequisites](#-prerequisites)
- [🧪 Development Assistant Mode](#-development-assistant-mode)
- [📊 Business Analytics Mode](#-business-analytics-mode)

### New Modular Workflow
- [🔄 Modular Workflow Overview](#-modular-workflow-overview)
- [🚀 Quick Start Guide](#-quick-start-guide)
- [📋 Management Commands](#-management-commands)
- [🔧 Troubleshooting](#-troubleshooting)

### Framework Documentation
- [🔧 OllamaCustomModel Framework](#-ollamacustommodel-framework)
- [📝 Template System](#-template-system)
- [🏗️ Architecture Details](#%EF%B8%8F-architecture-details)

### Additional Resources


## 🎯 **Platform Overview**


## 🧰 Utilities: PDF ➜ Markdown OCR (New)

Convert one or many PDFs to Markdown using local OCR via Ollama's Qwen2.5-VL model.

Prerequisites:
- Ollama running and the model `qwen2.5vl` pulled and available
- This repo's Python dependencies installed (requests, PyMuPDF, Pillow)

Examples:

```powershell
# 1) Single PDF ➜ per-page Markdown files (default behavior)
python .\create_mkd_from_PDF.py .\docs\report.pdf

# 2) Single PDF ➜ one combined Markdown file
python .\create_mkd_from_PDF.py .\docs\report.pdf .\docs\report.md --combined

# 3) Folder of PDFs ➜ one combined Markdown file
python .\create_mkd_from_PDF.py .\docs .\docs\all-docs.md --combined

# Optional page range (1-indexed)
python .\create_mkd_from_PDF.py .\docs\report.pdf --start 2 --end 10 --combined

# Optional different extension when targeting a folder (default: pdf)
python .\create_mkd_from_PDF.py .\scans .\scans\all.md --combined --ext pdf
```

Notes:
- Without `--combined`, the script saves one Markdown file per page next to the source PDF(s).
- With `--combined` and a folder input, all matching PDFs are merged into a single Markdown.
- With `--combined` and a single PDF, all pages are merged into one Markdown.
- If no `output_md` is provided when combining, the script uses a sensible default name.
- Ensure Ollama is running at the configured URL and `qwen2.5vl` is available.

### Images ➜ Markdown via OCR

You can also process images directly (png, jpg, jpeg, webp, bmp, tiff):

```powershell
# Single image ➜ Markdown
python .\create_mkd_from_PDF.py .\images\scan_01.png

# Single image ➜ specific output path
python .\create_mkd_from_PDF.py .\images\scan_01.jpg .\images\scan_01.md

# Folder of images (png) ➜ one combined Markdown
python .\create_mkd_from_PDF.py .\images .\images\all-images.md --combined --ext png

# Folder of images (jpg) ➜ per-file Markdown (no --combined)
python .\create_mkd_from_PDF.py .\images --ext jpg
```

## 🧰 Utilities: Merge Markdown ➜ One Markdown (New)

Combine multiple Markdown files in a folder into a single Markdown file.

Examples:

```powershell
# Combine all .md files in a folder into one file
python .\create_mkd_from_PDF.py .\notes .\notes\combined.md --combine-md

# Use a different extension (e.g., .mkd)
python .\create_mkd_from_PDF.py .\notes .\notes\combined.md --combine-md --md-ext mkd

# Skip per-file headers in the combined output
python .\create_mkd_from_PDF.py .\notes .\notes\combined.md --combine-md --no-file-headers
```

Notes:
- When using `--combine-md`, the positional path must be a folder.
- Files are included in sorted order by filename.
- By default, the output includes a section header per file. Disable with `--no-file-headers`.
- If `output_md` is omitted, the default is `<foldername>.md` inside the folder.

- 💬 Inline prompt support in VS Code via **Continue**
- ⚙️ **DeepSeek Coder V2 Lite** served by **llama.cpp** standalone server
- 🧠 Code generation, editing, and autocomplete capabilities

### 2. **Business Analytics Mode** 📊
- 🔍 **Qwen Churn Assistant**: Specialized customer churn analysis
- 📈 Natural language data analysis (no coding required)
- 🎯 Business-focused insights and recommendations
- 🤖 Powered by Qwen2.5-Instruct with specialized system prompts

### 3. **Modular Component System** 🔄 **NEW!**
- 🎛️ **Manual Control**: Start Ollama, deploy custom models, launch WebUI independently
- 🔧 **Better Debugging**: Isolate issues to specific components
- ⚡ **Reliable Startup**: No complex dependency chains
- 🔄 **Flexible Deployment**: Mix and match components as needed

## 🚀 Key Features

- **Private & Secure**: All computation happens locally on your hardware
- **🚀 CUDA-Enabled Performance**: GPU acceleration for optimal inference speed
- **🔐 Secure Remote Access**: Multi-platform access via **Tailscale** (browser, IDE, mobile)
- **Template-Driven Architecture**: Modular, maintainable configuration system
- **Multi-Model Support**: Code generation, chat, and specialized business analytics
- **GPU/CPU Flexibility**: Optimized for both GPU acceleration and CPU-only mode
- **Business Intelligence**: Transform raw data into actionable insights

---

## 🏗️ System Architecture

### Development Assistant Mode
```
[freedom (VS Code + Continue)] <---> [last-nuc (llama.cpp server)]
                 ↖
               Mobile (via Tailscale)
```

### Business Analytics Mode
```
[Business User] <---> [Open WebUI] <---> [Qwen Churn Assistant]
                            ↓
                    [Customer Data CSV] ---> [Natural Language Insights]
```

- **freedom**: HP EliteBook G9, Windows 11 Pro, 32GB RAM
- **last-nuc**: Intel NUC13RNGi9, RTX 3060 Ti, 64GB RAM

---

## 📦 Tech Stack

| Component     | Tool                                                     | Description                              |
| ------------- | -------------------------------------------------------- | ---------------------------------------- |
| **Development Mode** |                                                  |                                          |
| LLM Inference | [`llama.cpp`](https://github.com/ggerganov/llama.cpp)   | CUDA-enabled standalone inference server for GGUF models |
| Primary Model | [`DeepSeek Coder V2 Lite`](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct) | 15.7B parameter coding specialist model |
| VSCode Plugin | [`Continue`](https://continue.dev)                       | Inline chat & refactor tools             |
| **Business Analytics Mode** |                                          |                                          |
| Analytics Engine | [`Qwen2.5-Instruct`](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) | 14B/7B parameter business analysis specialist |
| System Prompts | Template-driven business intelligence prompts            | Specialized for churn analysis and business insights |
| **Shared Infrastructure** |                                              |                                          |
| Web Interface | [`Open WebUI`](https://github.com/open-webui/open-webui) | OpenAI-compatible UI & proxy             |
| Containerization | [`Docker Compose`](https://docs.docker.com/compose/)  | Template-based container orchestration    |
| VPN Layer     | [`Tailscale`](https://tailscale.com)                     | Encrypted access across devices          |
| Architecture  | **Template-Driven Design**                               | Modular, maintainable configuration system |

---

## 📂 Project Structure

This repository uses a **template-driven architecture** for maintainable, modular configuration:

### 🚀 **Entry Points (New Modular Workflow)**
- **`start_ollama.py`**: Start Ollama service manually (Step 1)
- **`start_custom_assistant.py`**: Deploy custom models to running Ollama (Step 2)
- **`start_webui.py`**: Start WebUI separately to connect to Ollama (Step 3)
- **`start_llama_server.py`**: llama.cpp server mode (alternative to Ollama)

### ⚙️ **Core Management Modules**
- **`utility_manager.py`**: Centralized subprocess operations and error handling
- **`llama_server_manager.py`**: llama.cpp server operations, model management, and enhanced health checks
- **`ollama_manager.py`**: Ollama service management and model pulling
- **`webui_manager.py`**: Open WebUI health checking and management
- **`ollama_custom_model.py`**: Generalized framework for specialized AI assistants
- **`network_manager.py`**: Network utilities and Docker Compose management

### 🔄 **New Modular Workflow**

The new workflow separates each component for maximum flexibility:

```bash
# Step 1: Start Ollama service
python start_ollama.py              # or: python start_ollama.py --cpu

# Step 2: Deploy custom models (optional)
python start_custom_assistant.py templates/yoga_sequence_system_prompt.template.json
python start_custom_assistant.py templates/qwen_churn_system_prompt.template.json

# Step 3: Start WebUI separately  
python start_webui.py

# Access: http://localhost:3000
```

### 🏗️ **Architecture Benefits**
- **� Component Independence**: Start/stop services individually
- **🧪 Easier Debugging**: Isolate issues to specific components
- **📖 Clear Separation**: Ollama → Custom Models → WebUI
- **♻️ Flexible Deployment**: Mix and match components as needed
- **⚙️ Manual Control**: Full control over each service lifecycle
- **🔄 Reliable Startup**: No complex dependency chains

### 📋 **Template System** 
```
templates/
├── 🤖 AI Model Configuration
│   └── qwen_churn_system_prompt.template.json   # Business intelligence system prompt
└── 🛠️ Open WebUI Integration
    └── open-webui_tools.py                     # CSV analysis tools for Open WebUI
```

### 📁 **Runtime Directories**
- **`.ollama/`**: Ollama data and models (auto-created, gitignored)
- **`.llama/`**: Llama.cpp models (auto-created, gitignored)
- **`.datasets/`**: Analysis workspace (persistent, user data)

### 📚 **Documentation**
- **`readme.md`**: This comprehensive guide (single source of truth)

### ⚙️ **Configuration Files**
- **`docker-compose.ollama.yml`**: Base Ollama service (CPU-optimized)
- **`docker-compose.webui.yml`**: Base Open WebUI service
- **`docker-compose.llama.yml`**: Base Llama.cpp server
- **`docker-compose.gpu-override.yml`**: GPU acceleration overlay (Ollama + WebUI)
- **`docker-compose.llama-gpu-override.yml`**: GPU acceleration for llama.cpp server
- **`docker-compose.churn-assistant-override.yml`**: Auto-generated Qwen Churn Assistant configuration (created by OllamaCustomModel)
- **`.gitignore`**: Excludes models, runtime files, preserves templates

---

## 📋 Prerequisites

### Required Software
- **Docker Desktop**: For running containerized services
- **Python 3.7+**: For running the orchestration script
- **Git**: For cloning the repository

### Python Dependencies
Install required packages:
```bash
pip install docker requests
```

### Hardware Requirements
- **GPU Mode**: NVIDIA GPU with 8GB+ VRAM (recommended for Q4_K_M and higher)
- **CPU Mode**: 8GB+ RAM (16GB+ recommended for Q4_K_M, 8GB sufficient for Q2_K)
- **Storage**: 6-17GB free space (depending on model quantization)
  - Q2_K: ~6.4GB (CPU mode default)
  - Q4_K_M: ~10.4GB (GPU mode default)
  - Q6_K: ~14.1GB (high quality)
  - Q8_0: ~16.7GB (maximum quality)

### Model Quantization Guide
The system uses **DeepSeek Coder V2 Lite** with various quantization levels:

| Quantization | Size   | Quality | GPU Recommendation | CPU Recommendation |
|-------------|--------|---------|-------------------|-------------------|
| Q8_0        | 16.7GB | Extreme | 24GB+ VRAM        | 32GB+ RAM         |
| Q6_K        | 14.1GB | Very High | 16GB+ VRAM      | 24GB+ RAM         |
| Q5_K_M      | 11.9GB | High    | 12GB+ VRAM        | 16GB+ RAM         |
| **Q4_K_M**  | 10.4GB | **Good (Default)** | **8GB+ VRAM** | **16GB+ RAM** |
| Q4_K_S      | 9.53GB | Good    | 8GB VRAM          | 12GB+ RAM         |
| IQ4_XS      | 8.57GB | Decent  | 6-8GB VRAM        | 12GB RAM          |
| Q3_K_L      | 8.45GB | Lower   | 6GB VRAM          | 10GB RAM          |
| Q3_K_M      | 8.12GB | Lower   | 4-6GB VRAM        | 8-10GB RAM        |
| **Q2_K**    | 6.43GB | **Usable (CPU Default)** | **4GB VRAM** | **8GB+ RAM** |

✅ **Tested Configuration**: Q2_K quantization successfully tested on WSL2 with 12 CPUs and 15.44GB RAM in CPU-only mode.

---

## 🧪 **Development Assistant Mode**

### 1. **Clone the Repository**
```bash
git clone https://github.com/mousou2003/AI-Server
cd AI-Server
```

### 2. **Verify Setup (Recommended)**
```bash
# Test basic functionality
python start_qwen_churn_assistant.py --help
python start_llama_server.py --help

# Verify Docker is running
docker --version
```

### 3. **Start llama.cpp Server**

#### GPU Mode (Recommended - CUDA-enabled)
```bash
python start_llama_server.py
```

#### CPU-Only Mode (No GPU Required)
```bash
python start_llama_server.py --cpu-only
```

**Quick Examples:**
- `python start_llama_server.py --auto` - GPU mode with default Q4_K_M
- `python start_llama_server.py --cpu-only --auto` - CPU mode with default Q2_K
- `python start_llama_server.py --cpu-only -q Q3_K_M` - CPU mode with better quality
- `python start_llama_server.py --list-quants` - Show all available quantization options
- `python start_llama_server.py --list-models` - List existing models in your .llama folder
- `python start_llama_server.py --cleanup` - Stop all containers after testing

This script will automatically:
- ✅ Download the DeepSeek Coder V2 Lite model if needed (~6-17GB depending on quantization)
- 🚀 Start llama.cpp server with CUDA support and the selected model/quantization
- 🐳 Launch Docker container using `server-cuda` image with enhanced health checks
- 🧪 Test API endpoints with comprehensive readiness verification
- 🎯 Handle quantization selection interactively or via command-line options
- ⚙️ Configure CPU-only or GPU-accelerated mode automatically

**Available Services:**
- **llama.cpp server**: High-performance inference engine on port 11435 (CUDA-enabled)
- **API Endpoints**: OpenAI-compatible REST API for integration

The script uses a streamlined Docker Compose architecture:
- **Base Service**: `docker-compose.llama.yml` with server-cuda image
- **GPU Acceleration**: `docker-compose.llama-gpu-override.yml` adds GPU runtime support
- **Health Monitoring**: Comprehensive health checks at both container and API level

**CPU vs GPU Mode:**
- **GPU Mode**: Faster inference (~50+ tokens/sec), uses CUDA-enabled server image
- **CPU Mode**: Slower inference (~20 tokens/sec), works on any system with adequate RAM

### Advanced Options

```bash
# Interactive quantization selection
python start_llama_server.py --cpu-only

# List all available quantization options
python start_llama_server.py --list-quants

# List existing models in your .llama directory
python start_llama_server.py --list-models

# Use specific quantization
python start_llama_server.py --cpu-only -q Q3_K_M

# Test deployment and cleanup afterwards
python start_llama_server.py --cleanup
```

### 4. **Start Business Analytics (Alternative Mode)**
```bash
# Quick business analytics setup
python start_qwen_churn_assistant.py --cpu     # CPU mode (7B model)
python start_qwen_churn_assistant.py           # GPU mode (14B model)

# Then visit http://localhost:3000 and upload your CSV files
```

### 5. **Configure VS Code (Development Mode)**

Install the Continue extension for VS Code: https://www.continue.dev/

Create or update your Continue configuration file at `~/.continue/config.yaml`:

```yaml
name: Local Assistant
version: 1.0.0
schema: v1
models:
  # Primary model: DeepSeek Coder V2 Lite via standalone llama.cpp server
  - name: deepseek-coder-v2-lite
    provider: openai
    model: DeepSeek-Coder-V2-Lite-Instruct-Q2_K.gguf  # Adjust based on your quantization
    apiBase: http://localhost:11435/v1  # Standalone llama.cpp server
    apiKey: fake  # Required but not used
    roles:
      - chat
      - edit
      - apply
      - autocomplete
    defaultCompletionOptions:
      contextLength: 163840  # 160K context window
      maxTokens: 8192

context:
  - provider: code
  - provider: docs
  - provider: diff
  - provider: terminal
  - provider: problems
  - provider: folder
  - provider: codebase
```

**For remote access via Tailscale**, replace `localhost` with your Tailscale hostname:
```yaml
apiBase: http://your-tailscale-hostname.ts.net:11435/v1  # For standalone llama.cpp server
```

---

## 🔄 **Modular Workflow Overview**

The AI-Server now uses a **modular workflow** where each component runs independently:

1. **Ollama Service** (runs models)
2. **Custom Model Deployment** (optional specialization)  
3. **WebUI Interface** (user interaction)

This provides better control, easier debugging, and more reliable startups.

### 🚀 **Quick Start Guide**

#### Step 1: Start Ollama
```bash
python start_ollama.py              # GPU mode
python start_ollama.py --cpu        # CPU mode
```

#### Step 2: Deploy Custom Models (Optional)
```bash
# Deploy yoga assistant
python start_custom_assistant.py templates/yoga_sequence_system_prompt.template.json

# Deploy business assistant  
python start_custom_assistant.py templates/qwen_churn_system_prompt.template.json

# Custom deployment
python start_custom_assistant.py templates/my_template.json --model qwen2.5:14b-instruct --name my-assistant
```

#### Step 3: Start WebUI
```bash
python start_webui.py
```

#### Access Points
- **WebUI**: http://localhost:3000
- **Ollama API**: http://localhost:11434

### 📋 **Management Commands**

#### Ollama Management
```bash
python start_ollama.py --status     # Check if running
python start_ollama.py --pull       # Pull default models
python start_ollama.py --list       # List available models
python start_ollama.py --logs       # Show logs
python start_ollama.py --stop       # Stop service
```

#### Custom Model Management
```bash
python start_custom_assistant.py templates/template.json --list    # List custom models
python start_custom_assistant.py templates/template.json --test    # Test model
python start_custom_assistant.py templates/template.json --remove  # Remove model
```

#### WebUI Management  
```bash
python start_webui.py --status      # Check status
python start_webui.py --open        # Open in browser
python start_webui.py --logs        # Show logs
python start_webui.py --stop        # Stop WebUI
```

### 🔧 **Troubleshooting**

#### Common Issues
1. **"Ollama is not running"** → Start Ollama first: `python start_ollama.py`
2. **"Model not found"** → Deploy custom model or pull base model
3. **"WebUI won't start"** → Check Ollama status: `python start_ollama.py --status`

#### Debug Steps
```bash
# Check all services
python start_ollama.py --status
python start_webui.py --status

# View logs
python start_ollama.py --logs
python start_webui.py --logs

# Clean restart
python start_ollama.py --stop
python start_webui.py --stop
python start_ollama.py
python start_webui.py
```

#### Available Templates
- `templates/yoga_sequence_system_prompt.template.json` - Yoga sequence generation
- `templates/qwen_churn_system_prompt.template.json` - Business churn analysis
- Create your own templates following the same JSON structure

#### Benefits of New Workflow
✅ **Independent Services** - Start/stop components individually  
✅ **Better Debugging** - Isolate issues to specific components  
✅ **Flexible Deployment** - Mix and match as needed  
✅ **Reliable Startup** - No complex dependency chains  
✅ **Manual Control** - Full control over service lifecycle  
✅ **Easier Development** - Test components in isolation  

#### Migration from Old Scripts
**Old Way (Removed)**
```bash
# These scripts are no longer available
python start_yoga_assistant.py      # ❌ Removed
python start_qwen_churn_assistant.py # ❌ Removed
python start_ollama_server.py       # ❌ Removed
```

**New Way**
```bash
# Modular approach
python start_ollama.py              # ✅ Start Ollama
python start_custom_assistant.py templates/yoga_sequence_system_prompt.template.json  # ✅ Deploy yoga model
python start_webui.py               # ✅ Start WebUI
```

---

## 🔧 **OllamaCustomModel Framework**

The `OllamaCustomModel` class provides a flexible framework for creating specialized AI assistants using Ollama models with custom system prompts and configurations. This system generalizes the original approach to work with any template-based AI assistant.

### Framework Features

- 🔧 **Template-based Configuration**: Load custom system prompts and model parameters from JSON templates
- 🐳 **Docker Infrastructure Management**: Automated Docker Compose setup with project-specific configurations  
- 🧠 **Model Customization**: Create specialized models with embedded system prompts
- ⚡ **CPU/GPU Flexibility**: Support for both CPU-only and GPU-accelerated deployments
- 🔄 **Factory Methods**: Easy creation of specific assistant types
- ✅ **Template Validation**: Ensure template files have proper structure and valid parameters

### Quick Framework Usage

#### Using Factory Methods
```python
from ollama_custom_model import OllamaCustomModel

# Create a yoga sequence assistant
yoga_assistant = OllamaCustomModel.create_yoga_assistant(cpu_mode=True)
yoga_assistant.start_infrastructure()

# Create a churn analysis assistant  
churn_assistant = OllamaCustomModel.create_churn_assistant(cpu_mode=False)
churn_assistant.start_infrastructure()
```

#### Using Custom Templates
```python
# Create an assistant from any template file
custom_assistant = OllamaCustomModel(
    template_path="templates/my_custom_template.json",
    cpu_mode=False,
    project_name="my-assistant"
)
custom_assistant.start_infrastructure()
```

### 📝 **Template System**

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

#### Required Template Fields
- `name`: Display name for the assistant
- `description`: Brief description of the assistant's purpose  
- `system_prompt`: The system prompt that defines the assistant's behavior

#### Optional Template Fields
- `model_parameters`: Ollama model parameters (temperature, top_k, etc.)
- `recommended_model`: Suggested base model to use
- `usage_instructions`: Instructions shown to users after startup
- Additional fields for specialized configurations

### Framework Examples

#### Example 1: Business Strategy Assistant
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

#### Example 2: Educational Tutor
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

### Advanced Framework Configuration

#### Model Selection
The system intelligently selects appropriate models based on:
1. **Explicit specification**: Via `model_name` parameter or `recommended_model` in template
2. **Template analysis**: Automatic inference based on template content
3. **Hardware constraints**: CPU vs GPU mode considerations
4. **Size preferences**: Large model flag for enhanced capabilities

#### Project Isolation
Each assistant instance is completely isolated with:
- Separate Docker containers
- Independent data volumes  
- Project-specific configurations
- Isolated model storage

#### Framework API Reference

**OllamaCustomModel Class Constructor**
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

**Key Methods**
- `start_infrastructure()`: Start the complete infrastructure
- `stop_infrastructure(remove_volumes=False)`: Stop infrastructure 
- `status()`: Check status of all components
- `create_custom_model()`: Create model with embedded system prompt
- `wait_for_services()`: Wait for services to be ready

**Factory Methods**
- `create_yoga_assistant(**kwargs)`: Create yoga sequence assistant
- `create_churn_assistant(**kwargs)`: Create churn analysis assistant
- `create_from_template(template_path, **kwargs)`: Generic factory method

---

## 🏗️ **Template-Driven Architecture**

This project pioneered a **template-based approach** to AI infrastructure deployment, providing several key advantages:

### 🎯 **Benefits**
- **� Maintainable**: All configuration in version-controlled templates
- **🚀 Scalable**: Easy to add new AI modes and configurations  
- **🧹 Clean**: Clear separation between templates and runtime
- **⚙️ Flexible**: Support for different hardware modes (CPU/GPU)
- **📦 Portable**: Templates work across different environments

### 📋 **Template Categories**

| Template Type | Purpose | Example |
|---------------|---------|---------|
| **🤖 AI Configuration** | System prompts, model parameters | `qwen_churn_system_prompt.template.json` |
| **�️ Tool Integration** | WebUI tools and utilities | `open-webui_tools.py` |

### 🔄 **Template Processing Flow**
```
Template Files (version controlled)
    ↓ (read by managers)
Runtime Environment (active configuration)
    ↓ (cleanup on exit)
Clean State (templates preserved)
```

### 🎨 **Customization**
Want to create your own AI assistant mode? Simply:
1. Create new templates in `templates/`
2. Add template variable substitution in the manager
3. Create a new entry point script
4. Templates automatically handle CPU/GPU modes and standard Open WebUI integration

---

## 🔍 **Verification & Testing**

### **Quick Validation**
```bash
# Verify Docker is running
docker --version

# Check available disk space (models are large)
dir .llama
dir .ollama

# Test basic functionality
python start_qwen_churn_assistant.py --help
```

### **llama.cpp Server Testing**
```bash
# Start standalone llama.cpp server
python start_llama_server.py --cpu-only --auto

# Verify endpoints
curl http://localhost:11435/v1/models     # llama.cpp API
curl http://localhost:11435/health        # Health check endpoint

# Test VS Code integration with Continue extension
```

### **Business Analytics Mode Testing**
```bash
# Start business analytics
python start_qwen_churn_assistant.py --cpu

# Verify services  
python start_qwen_churn_assistant.py --status

# Test WebUI at http://localhost:3000
# Upload a CSV file and ask: "What patterns do you see in this data?"
```

### **Performance Verification**
Recent deployment test results:

**llama.cpp Server Mode** (DeepSeek Coder V2 Lite):
- ✅ **CPU Mode**: Q2_K quantization at ~20 tokens/second
- ✅ **GPU Mode**: CUDA-enabled server-cuda image for optimal performance
- ✅ **Model Size**: 6.4GB loaded successfully  
- ✅ **Context Length**: 163,840 tokens (160K context window)
- ✅ **Parameters**: 15.7B parameters with enhanced health checks
- ✅ **Health Monitoring**: Multi-endpoint validation and container health checks

**Business Analytics Mode** (Qwen2.5-Instruct):
- ✅ **CPU Mode**: 7B model at ~15-20 tokens/second
- ✅ **GPU Mode**: 14B model at ~30-50 tokens/second  
- ✅ **Memory System**: Persistent conversation memory
- ✅ **Workspace**: CSV file analysis capabilities

### Remote Access
For remote access via Tailscale, replace `localhost` with your Tailscale hostname:
- llama.cpp API: `http://your-tailscale-hostname.ts.net:11435`

---

## 📊 **Business Analytics Mode**

**🎯 Purpose**: Transform customer data into actionable business insights through natural language conversations.

The Business Analytics Mode provides specialized business intelligence capabilities that analyze customer churn data without requiring any coding skills. Simply upload your CSV files and ask questions in plain English!

### � **Key Features**
- **🗣️ Natural Language Interface**: Ask business questions in plain English
- **📈 Zero-Code Analysis**: No Python, SQL, or technical skills required
- **🤖 Specialized AI**: Qwen2.5-Instruct optimized with business intelligence prompts
- **🔒 Local & Private**: All analysis happens on your hardware
- **💡 Actionable Insights**: Get practical recommendations, not just statistics
- **📝 Memory-Enabled**: Remembers context across analysis sessions
- **📁 Workspace Integration**: Upload and analyze CSV files directly
- **🛠️ Built-in Tools**: Custom CSV analysis tools integrated with Open WebUI

### 🚀 **Quick Start**
```bash
# GPU mode (recommended - uses 14B model for comprehensive analysis)
python start_qwen_churn_assistant.py

# CPU mode (uses optimized 7B model - much faster on CPU)
python start_qwen_churn_assistant.py --cpu

# CPU mode with large model (14B - slower but more comprehensive)
python start_qwen_churn_assistant.py --cpu --large-model

# Management commands
python start_qwen_churn_assistant.py --status    # Check status
python start_qwen_churn_assistant.py --stop      # Stop infrastructure  
python start_qwen_churn_assistant.py --logs      # View container logs
python start_qwen_churn_assistant.py --cleanup   # Clean temporary files
```

### �️ **Open WebUI Tools Integration**

The system includes custom Python tools that extend Open WebUI's capabilities for CSV data analysis:

**Available Tools:**
- **`analyze_wine_club_csv`**: Direct CSV upload analysis for business insights
- **`load_raw_csv_from_storage`**: Preview CSV files from Open WebUI storage  
- **`analyze_wine_club_from_knowledge`**: Advanced analysis of stored CSV files
- **File Resolution**: Smart file matching from partial filenames

**Tool Features:**
- **Automatic File Discovery**: Upload CSV files and reference them by partial name
- **Data Sampling**: Configurable row limits for large datasets
- **Business Context**: Tools designed for business analysis, not technical processing
- **Error Handling**: Comprehensive error messages and fallback behaviors
- **Storage Integration**: Works with Open WebUI's persistent file storage

**Usage Example:**
```
1. Upload a CSV file named "customer-churn-data.csv" to Open WebUI
2. Ask: "Use the analyze_wine_club_from_knowledge tool to analyze churn-data.csv and tell me about customer retention patterns"
3. The tool will automatically find and analyze your uploaded file
```

**Tool Installation:**
The tools are automatically available when using the Qwen Churn Assistant mode. They're located in `templates/open-webui_tools.py` and integrate seamlessly with the business analytics workflow.

### �💬 **Example Business Questions**
Once your CSV data is uploaded, ask questions like:

**Initial Data Exploration:**
- *"Can you analyze this customer churn data and give me an overview?"*
- *"What's the overall churn rate in this dataset?"*
- *"Which customer segments are represented?"*

**Churn Pattern Analysis:**
- *"Which customer segments have the highest churn rate?"*
- *"What factors seem to correlate with customer churn?"*
- *"Are there critical thresholds that predict churn risk?"*

**Risk Assessment:**
- *"Which current customers have the highest churn risk?"*
- *"What early warning signs should we watch for?"*
- *"How does churn risk vary by customer demographics?"*

**Business Strategy:**
- *"What retention strategies would you recommend?"*
- *"Which customers should we prioritize for intervention?"*
- *"How can we reduce churn in high-risk segments?"*

### 🎯 **What Makes It Special**
- **Business-Focused**: Responses avoid technical jargon and focus on actionable insights
- **Template-Driven**: Uses specialized system prompts for consistent business analysis
- **Standard Architecture**: Proper Open WebUI + Ollama integration (models auto-discovered)
- **Memory & Workspace**: Persistent conversations and file analysis via Docker volumes
- **CPU/GPU Optimized**: Automatically selects appropriate model size for your hardware
- **Integrated Tools**: Custom CSV analysis tools built into Open WebUI
- **File Management**: Smart file resolution and storage integration

### 🏗️ **Template Architecture**
The system uses a sophisticated template-driven approach:

```
User Question → Business AI (Qwen + Custom Prompt) → Actionable Insight
     ↑                           ↓
CSV Upload ←→ WebUI Tools ←→ Workspace ←→ Memory System
```

**Template Components:**
- **System Prompt**: `templates/qwen_churn_system_prompt.template.json`
- **CSV Tools**: `templates/open-webui_tools.py`
- **WebUI Integration**: Standard Ollama connection (auto-discovery)

### � **Data Requirements**
**Supported Formats:** CSV files with customer records

**Upload Methods:**
- **Direct Upload**: Use Open WebUI's file upload feature
- **Tool Integration**: Files are automatically processed by built-in CSV analysis tools
- **Partial Naming**: Reference files by partial names (e.g., "churn-data.csv" matches "customer-churn-data.csv")

**Recommended Structure:**
- **Customer ID**: Unique identifier
- **Churn Status**: Binary indicator (churned/retained)  
- **Demographics**: Age, location, segment, etc.
- **Behavioral Data**: Usage patterns, engagement metrics
- **Account Info**: Tenure, value, payment history
- **Interaction Data**: Support contacts, satisfaction scores

**Privacy:** All processing happens locally - no data sent to external services.

### 🚨 **Troubleshooting**
```bash
# Check system status
python start_qwen_churn_assistant.py --status

# View container logs for debugging
python start_qwen_churn_assistant.py --logs

# Restart if needed
python start_qwen_churn_assistant.py --stop
python start_qwen_churn_assistant.py --cpu

# Clean up temporary files only
python start_qwen_churn_assistant.py --cleanup

# Complete cleanup (removes ALL data including Docker volumes)
python start_qwen_churn_assistant.py --cleanup-all
```

**Common Issues:**
- **Memory issues**: Use `--cpu` mode or close other applications
- **Model download fails**: Check internet connection and disk space
- **Containers won't start**: Ensure Docker is running and has sufficient resources
- **Persistent volume issues**: Use `--cleanup-all` to remove all Docker volumes and start fresh

**Cleanup Options:**
- **`--cleanup-all`**: Removes everything including Docker volumes (⚠️ **WARNING**: deletes all conversation history and WebUI settings)

### 💡 **Best Practices**
- **Ask specific questions**: Reference particular segments or behaviors
- **Build on insights**: Use follow-up questions to dive deeper  
- **Provide context**: Explain your business model and customer segments
- **Clean your data**: Remove duplicates and handle missing values before upload

### 🧪 **Comprehensive Testing Guide**

This testing suite ensures your custom `qwen2.5:7b-instruct-churn` model is properly configured with the specialized churn analysis system prompt and behaves according to business-focused constraints.

#### **Prerequisites**
1. **Verify Model Availability**
   ```bash
   docker exec ollama-qwen-churn ollama list
   ```
   - Confirm `qwen2.5:7b-instruct-churn` appears in the list
   - Note the creation/modification date

2. **Access Web UI**
   - Open http://localhost:3000
   - Select the `qwen2.5:7b-instruct-churn` model from the dropdown
   - Ensure you're testing the custom model, not the base model

#### **Test Suite**

**Test 1: System Prompt Integration & Role Recognition**

*Test Prompt:*
```
Hello! I have a customer dataset and I'm concerned about churn. Can you help me understand what I should be looking for?
```

*Expected Response Characteristics:*
- ✅ Should identify itself as a "specialized churn analysis assistant"
- ✅ Should focus on business insights and patterns
- ✅ Should ask business-focused clarifying questions about:
  - Industry/business type
  - Customer segments
  - Available data fields
  - Current business concerns
- ✅ Should mention analyzing CSV files through conversation
- ❌ Should NOT offer to write code or provide technical implementation

**Test 2: Code Generation Constraint Adherence**

*Test Prompt:*
```
Can you write Python code to analyze my customer churn data?
```

*Expected Response:*
- ✅ Should politely decline to write code
- ✅ Should explain the conversational analysis approach
- ✅ Should redirect to discussing data patterns through dialogue
- ✅ Should maintain helpful, business-focused tone
- ❌ Should NOT provide any Python, SQL, or programming syntax
- ❌ Should NOT suggest technical tools or libraries

**Test 3: Business Focus & Analysis Framework**

*Test Prompt:*
```
I have customer data with columns like customer_id, tenure, monthly_charges, total_charges, contract_type, and churn_status. What insights can you provide?
```

*Expected Response Structure:*
1. **Key Finding**: Should identify important business patterns from the described fields
2. **Supporting Evidence**: Should explain what these fields typically reveal (in business terms)
3. **Business Implication**: Should discuss what patterns mean for the company
4. **Recommended Action**: Should suggest specific, actionable next steps
5. **Follow-up Questions**: Should ask about business context and goals

**Test 4: Technical Implementation Avoidance**

*Test Prompt:*
```
Show me SQL queries to find churned customers with high monthly charges.
```

*Expected Response:*
- ✅ Should decline to provide SQL queries
- ✅ Should offer to discuss the business question behind the request
- ✅ Should ask about the business goal (why high-charge churned customers matter)
- ✅ Should suggest conversational analysis of this customer segment
- ❌ Should NOT provide any SQL syntax or database queries

**Test 5: Statistical Formula Constraint**

*Test Prompt:*
```
Calculate the churn rate formula and show me the statistical significance tests I should run.
```

*Expected Response:*
- ✅ Should explain churn rate in simple business terms (customers lost vs total customers)
- ✅ Should avoid mathematical formulas and statistical notation
- ✅ Should focus on what churn rate means for business decisions
- ✅ Should ask about business thresholds and concerns
- ❌ Should NOT provide mathematical formulas or equations
- ❌ Should NOT mention specific statistical tests or procedures

**Test 6: Business Strategy Focus**

*Test Prompt:*
```
I notice customers with month-to-month contracts have higher churn rates. What should I do about this?
```

*Expected Response Elements:*
- **Business Analysis**: Should discuss why month-to-month customers might churn more
- **Segment Understanding**: Should ask about different customer types and their needs
- **Retention Strategies**: Should suggest practical business actions like:
  - Contract incentives
  - Customer engagement programs
  - Early warning systems
  - Targeted communication
- **Risk Assessment**: Should help identify which month-to-month customers are highest risk
- **Business Impact**: Should discuss revenue implications

#### **Comparison Test with Base Model**

To verify your customization is working, also test the base model `qwen2.5:7b-instruct`:

*Test Prompt:*
```
I have customer churn data and need help analyzing it. What's the best approach?
```

**Base Model Expected Behavior:**
- May offer to write code or provide technical solutions
- Might suggest specific tools, libraries, or programming approaches
- Could provide more general-purpose analysis suggestions

**Custom Model Expected Behavior:**
- Should focus on business conversation and insights
- Should ask about business context and goals
- Should avoid technical implementation suggestions

#### **Quick Validation Checklist**

✅ Model responds as churn analysis expert  
✅ Refuses code generation requests  
✅ Uses business language, not technical terms  
✅ Follows 5-step response format  
✅ Asks business-focused follow-up questions  
✅ Provides actionable recommendations  
✅ Avoids statistical formulas and technical details  
✅ Maintains conversational, helpful tone  

#### **Test Results Template**

```
Date: ___________
Model: qwen2.5:7b-instruct-churn

Test 1 - System Prompt Integration: ✅/❌
Test 2 - Code Generation Constraint: ✅/❌  
Test 3 - Business Focus: ✅/❌
Test 4 - Technical Implementation Avoidance: ✅/❌
Test 5 - Statistical Formula Constraint: ✅/❌
Test 6 - Business Strategy Focus: ✅/❌

Overall Assessment: ✅ PASS / ❌ FAIL

Notes:
_________________________________
_________________________________
```

#### **Troubleshooting Test Failures**

If tests fail, check:
1. **Model Selection**: Ensure you're using `qwen2.5:7b-instruct-churn`, not the base model
2. **Model Creation**: Verify the custom model was created successfully
3. **Modelfile Content**: Check that the system prompt was properly integrated
4. **Container Status**: Ensure Ollama container is running and responsive

**Remember**: The goal is to ensure your custom model behaves as a business consultant, not a technical developer, when discussing customer churn analysis.

---

## 🏗️ Code Architecture

The project uses a **modular architecture** with separated concerns:

- **Main Scripts**: 
  - `start_llama_server.py`: Standalone llama.cpp server with CUDA support
  - `start_qwen_churn_assistant.py`: Specialized business analytics mode
- **Service Managers**: Individual managers for each service component
  - `llama_server_manager.py`: llama.cpp operations, model management, and enhanced health checks
  - `ollama_manager.py`: Ollama service and model pulling
  - `webui_manager.py`: Open WebUI health checks
  - `utility_manager.py`: Subprocess and utility functions
  - `network_manager.py`: Network utilities and Docker Compose management

All components follow the template-driven architecture with clear separation of concerns.

---

## 🙌 Acknowledgments

This **template-driven AI infrastructure platform** builds on the work of:

**Core AI Infrastructure:**
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - High-performance LLM inference
- [Open WebUI](https://github.com/open-webui/open-webui) - Modern web interface with memory/workspace
- [Ollama](https://ollama.ai) - Model backend and management
- [Docker](https://docker.com) - Containerized deployment system

**AI Models:**
- [DeepSeek](https://huggingface.co/deepseek-ai) - DeepSeek Coder V2 Lite (Development Mode)
- [Qwen Team](https://huggingface.co/Qwen) - Qwen2.5-Instruct (Business Analytics Mode)

**Development Tools:**
- [Continue](https://continue.dev) - VS Code AI assistant extension
- [Tailscale](https://tailscale.com) - Secure remote access

**Architecture Innovation:**
This project introduces a **template-driven approach** to AI infrastructure that makes it easy to create specialized AI assistants for different business domains while maintaining clean, maintainable code.

---

## ✨ Motivation (from LinkedIn post)

> Over the past few days, I’ve been designing and deploying a local, private coding assistant setup — entirely powered by open-source tools and hosted on my own machine...

> It's surprisingly easy now to build your own dev assistant:
>
> - No OpenAI key required
> - No cloud dependencies
> - No latency overhead

Whether you’re an indie dev, engineer, or researcher, this gives you full control over your AI stack.

---

## 📌 Repo

[https://github.com/mousou2003/AI-Server](https://github.com/mousou2003/AI-Server)

