# 🧠 Local AI-Powered Development & Business Analytics Platform

This project provides a **fully local AI infrastructure** with two specialized modes:

## 🎯 **Dual-Mode AI Platform**

### 1. **Development Assistant Mode** 💻
- 💬 Inline prompt support in VS Code via **Continue**
- ⚙️ High-performance LLMs served by *### 🎯 **Benefits**
- **🔧 Maintainable**: All configuration in version-controlled templatesllama** & **llama.cpp**
- 🌐 Web interface via **Open WebUI**
- 🔐 Secure remote access with **Tailscale**

### 2. **Business Analytics Mode** 📊 **NEW!**
- 🔍 **Qwen Churn Assistant**: Specialized customer churn analysis
- 📈 Natural language data analysis (no coding required)
- 🎯 Business-focused insights and recommendations
- 🤖 Powered by Qwen2.5-Coder with specialized system prompts

## 🚀 Key Features

- **Private & Secure**: All computation happens locally on your hardware
- **Template-Driven Architecture**: Modular, maintainable configuration system
- **Multi-Model Support**: Code generation, chat, and specialized business analytics
- **Multi-Platform Access**: Browser, IDE, or mobile (via Tailscale)
- **GPU/CPU Flexibility**: Optimized for both GPU acceleration and CPU-only mode
- **Business Intelligence**: Transform raw data into actionable insights

---

## 🏗️ System Architecture

### Development Assistant Mode
```
[freedom (VS Code + Continue)] <---> [last-nuc (llama.cpp + Open WebUI)]
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
| LLM Inference | [`llama.cpp`](https://github.com/ggerganov/llama.cpp)   | High-performance inference engine for GGUF models |
| LLM Backend   | [`Ollama`](https://ollama.ai)                           | Secondary backend for qwen2.5-coder:7b and gemma2:9b |
| Primary Model | [`DeepSeek Coder V2 Lite`](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct) | 15.7B parameter coding specialist model |
| VSCode Plugin | [`Continue`](https://continue.dev)                       | Inline chat & refactor tools             |
| **Business Analytics Mode** |                                          |                                          |
| Analytics Engine | [`Qwen2.5-Coder`](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct) | 32B/7B parameter business analysis specialist |
| System Prompts | Template-driven business intelligence prompts            | Specialized for churn analysis and business insights |
| **Shared Infrastructure** |                                              |                                          |
| Web Interface | [`Open WebUI`](https://github.com/open-webui/open-webui) | OpenAI-compatible UI & proxy             |
| Containerization | [`Docker Compose`](https://docs.docker.com/compose/)  | Template-based container orchestration    |
| VPN Layer     | [`Tailscale`](https://tailscale.com)                     | Encrypted access across devices          |
| Architecture  | **Template-Driven Design**                               | Modular, maintainable configuration system |

---

## 📂 Project Structure

This repository uses a **template-driven architecture** for maintainable, modular configuration:

### 🚀 **Entry Points**
- **`start_llama_webui.py`**: Development assistant mode (coding, VS Code integration)
- **`start_qwen_churn_assistant.py`**: Business analytics mode (churn analysis, natural language)
- **`test_qwen_setup.py`**: Validation script for setup verification

### ⚙️ **Core Management Modules**
- **`utility_manager.py`**: Centralized subprocess operations and error handling
- **`llama_server_manager.py`**: llama.cpp server operations and model management
- **`ollama_manager.py`**: Ollama service management and model pulling
- **`webui_manager.py`**: Open WebUI health checking and management
- **`service_config.py`**: Configuration management and quantization options

### 🏗️ **Modular Architecture Benefits**
- **🔧 Single Responsibility**: Each manager handles one specific service
- **🧪 Testable**: Individual components can be tested in isolation
- **📖 Readable**: Code organized by functionality, not mixed concerns
- **♻️ Reusable**: Managers can be imported and used in other projects
- **⚙️ Self-Contained**: Each manager contains its own configuration
- **🔄 Maintainable**: Modify specific components without affecting others

### 📋 **Template System** 
```
templates/
├── 🤖 AI Model Configuration
│   └── qwen_churn_system_prompt.template.md     # Business intelligence system prompt
├── 🐳 Docker Infrastructure  
│   ├── docker-compose.qwen-churn.template.yml   # GPU-accelerated business analytics
│   └── docker-compose.qwen-churn.cpu.template.yml # CPU-only business analytics
└── 🧠 Memory & Workspace
    └── churn_memory_template.md                 # Analysis framework and memory structure
```

### 📁 **Runtime Directories**
- **`models/`**: Downloaded model files (auto-created, gitignored)
- **`staging/`**: Temporary generated files (auto-created, auto-cleaned)
- **`memory/`**: Persistent conversation memory (created from templates)
- **`workspace/`**: File analysis workspace (persistent, user data)

### 📚 **Documentation**
- **`readme.md`**: This comprehensive guide (single source of truth)

### ⚙️ **Configuration Files**
- **`docker-compose.yml`**: Development mode GPU configuration
- **`docker-compose.cpu.yml`**: Development mode CPU-only configuration
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

## 🧪 **How to Use** - Development Assistant Mode

### 1. **Clone the Repository**
```bash
git clone https://github.com/mousou2003/AI-Server
cd AI-Server
```

### 2. **Verify Setup (Recommended)**
```bash
# Run comprehensive setup validation
python test_qwen_setup.py

# This checks:
# ✅ Docker availability and configuration
# ✅ Required files and template structure  
# ✅ Network connectivity for model downloads
# ✅ GPU availability (optional)
# ✅ Directory structure and permissions
```

### 3. **Start Development Assistant**

#### GPU Mode (Recommended)
```bash
python start_llama_webui.py
```

#### CPU-Only Mode (No GPU Required)
```bash
python start_llama_webui.py --cpu-only
```

**Quick Examples:**
- `python start_llama_webui.py --auto` - GPU mode with default Q4_K_M
- `python start_llama_webui.py --cpu-only --auto` - CPU mode with default Q2_K
- `python start_llama_webui.py --cpu-only -q Q3_K_M` - CPU mode with better quality
- `python start_llama_webui.py --list-quants` - Show all available quantization options
- `python start_llama_webui.py --list-models` - List existing models in your models folder
- `python start_llama_webui.py --cleanup` - Stop all containers after testing

This script will automatically:
- ✅ Download the DeepSeek Coder V2 Lite model if needed (~6-17GB depending on quantization)
- 🚀 Start llama.cpp server with the selected model and quantization
- 🐳 Launch Docker containers (Open WebUI + Ollama) using the appropriate compose file
- 🧪 Test all API endpoints to ensure everything works
- 🎯 Handle quantization selection interactively or via command-line options
- ⚙️ Configure CPU-only or GPU-accelerated mode automatically

**Available Services:**
- **llama.cpp server**: Primary inference engine on port 11435
- **Ollama**: Secondary model backend on port 11434  
- **Open WebUI**: Web interface on port 3000

The script intelligently selects between `docker-compose.yml` (GPU mode) and `docker-compose.cpu.yml` (CPU mode) based on your configuration.

**CPU vs GPU Mode:**
- **GPU Mode**: Faster inference (~50+ tokens/sec), requires NVIDIA GPU with sufficient VRAM
- **CPU Mode**: Slower inference (~20 tokens/sec), works on any system with adequate RAM

### Advanced Options

```bash
# Interactive quantization selection
python start_llama_webui.py --cpu-only

# List all available quantization options
python start_llama_webui.py --list-quants

# List existing models in your models directory
python start_llama_webui.py --list-models

# Use specific quantization
python start_llama_webui.py --cpu-only -q Q3_K_M

# Test deployment and cleanup afterwards
python start_llama_webui.py --cleanup
```

### 4. **Start Business Analytics (Alternative Mode)**
```bash
# Quick business analytics setup
python start_qwen_churn_assistant.py --cpu     # CPU mode (7B model)
python start_qwen_churn_assistant.py           # GPU mode (32B model)

# Then visit http://localhost:3000 and upload your CSV files
```

### 5. **Configure VS Code (Development Mode Only)**

Install the Continue extension for VS Code: https://www.continue.dev/

Create or update your Continue configuration file at `~/.continue/config.yaml`:

```yaml
name: Local Assistant
version: 1.0.0
schema: v1
models:
  # Primary model: DeepSeek Coder V2 Lite via llama.cpp
  - name: deepseek-coder-v2-lite
    provider: openai
    model: DeepSeek-Coder-V2-Lite-Instruct-Q2_K.gguf  # Adjust based on your quantization
    apiBase: http://localhost:11435/v1  # llama.cpp server
    apiKey: fake  # Required but not used
    roles:
      - chat
      - edit
      - apply
      - autocomplete
    defaultCompletionOptions:
      contextLength: 163840  # 160K context window
      maxTokens: 8192
      
  # Secondary models via Ollama
  - name: qwen2.5-coder 7b
    provider: ollama
    model: qwen2.5-coder:7b
    apiBase: http://localhost:11434
    roles:
      - chat
      - edit
      - apply
      - autocomplete
    defaultCompletionOptions:
      contextLength: 32768
      maxTokens: 8192
      
  - name: gemma2 9b
    provider: ollama
    model: gemma2:9b
    apiBase: http://localhost:11434
    roles:
      - chat
      - edit
      - apply
    defaultCompletionOptions:
      contextLength: 8192
      maxTokens: 4096

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
apiBase: http://your-tailscale-hostname.ts.net:11435/v1  # For llama.cpp
apiBase: http://your-tailscale-hostname.ts.net:11434    # For Ollama
```

---

## 🏗️ **Template-Driven Architecture**

This project pioneered a **template-based approach** to AI infrastructure deployment, providing several key advantages:

### 🎯 **Benefits**
- **� Maintainable**: All configuration in version-controlled templates
- **🚀 Scalable**: Easy to add new AI modes and configurations  
- **🧹 Clean**: Clear separation between templates, staging, and runtime
- **⚙️ Flexible**: Support for different hardware modes (CPU/GPU)
- **📦 Portable**: Templates work across different environments

### 📋 **Template Categories**

| Template Type | Purpose | Example |
|---------------|---------|---------|
| **🤖 AI Configuration** | System prompts, model parameters | `qwen_churn_system_prompt.template.md` |
| **🐳 Infrastructure** | Docker Compose configurations | `docker-compose.qwen-churn.template.yml` |
| **🧠 Memory Systems** | Conversation memory frameworks | `churn_memory_template.md` |

### 🔄 **Template Processing Flow**
```
Template Files (version controlled)
    ↓ (read by managers)
Staging Directory (temporary substituted files)  
    ↓ (deployed to containers)
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
Get-ChildItem models -Recurse | Measure-Object -Property Length -Sum

# Test basic functionality
python start_qwen_churn_assistant.py --help
```

### **Development Mode Testing**
```bash
# Start development assistant
python start_llama_webui.py --cpu-only --auto

# Verify endpoints
curl http://localhost:11435/v1/models     # llama.cpp API
curl http://localhost:11434/api/tags      # Ollama API  
Start-Process "http://localhost:3000"     # Open WebUI in browser

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

**Development Mode** (DeepSeek Coder V2 Lite):
- ✅ **CPU Mode**: Q2_K quantization at ~20 tokens/second
- ✅ **Model Size**: 6.4GB loaded successfully  
- ✅ **Context Length**: 163,840 tokens (160K context window)
- ✅ **Parameters**: 15.7B parameters in CPU-only mode

**Business Analytics Mode** (Qwen2.5-Coder):
- ✅ **CPU Mode**: 7B model at ~15-20 tokens/second
- ✅ **GPU Mode**: 32B model at ~30-50 tokens/second  
- ✅ **Memory System**: Persistent conversation memory
- ✅ **Workspace**: CSV file analysis capabilities

### Remote Access
For remote access via Tailscale, replace `localhost` with your Tailscale hostname:
- Open WebUI: `http://your-tailscale-hostname.ts.net:3000`
- llama.cpp API: `http://your-tailscale-hostname.ts.net:11435`
- Ollama API: `http://your-tailscale-hostname.ts.net:11434`

---

## 📊 **Qwen Churn Assistant** - Business Analytics Mode

**🎯 Purpose**: Transform customer data into actionable business insights through natural language conversations.

The Qwen Churn Assistant is a specialized business intelligence system that analyzes customer churn data without requiring any coding skills. Simply upload your CSV files and ask questions in plain English!

### � **Key Features**
- **🗣️ Natural Language Interface**: Ask business questions in plain English
- **📈 Zero-Code Analysis**: No Python, SQL, or technical skills required
- **🤖 Specialized AI**: Qwen2.5-Coder optimized with business intelligence prompts
- **🔒 Local & Private**: All analysis happens on your hardware
- **💡 Actionable Insights**: Get practical recommendations, not just statistics
- **📝 Memory-Enabled**: Remembers context across analysis sessions
- **📁 Workspace Integration**: Upload and analyze CSV files directly

### 🚀 **Quick Start**
```bash
# GPU mode (recommended - uses 32B model for comprehensive analysis)
python start_qwen_churn_assistant.py

# CPU mode (uses optimized 7B model - much faster on CPU)
python start_qwen_churn_assistant.py --cpu

# CPU mode with large model (32B - very slow but most comprehensive)
python start_qwen_churn_assistant.py --cpu --large-model

# Management commands
python start_qwen_churn_assistant.py --status    # Check status
python start_qwen_churn_assistant.py --stop      # Stop infrastructure  
python start_qwen_churn_assistant.py --logs      # View container logs
python start_qwen_churn_assistant.py --cleanup   # Clean temporary files
```

### 💬 **Example Business Questions**
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

### 🏗️ **Template Architecture**
The system uses a sophisticated template-driven approach:

```
User Question → Business AI (Qwen + Custom Prompt) → Actionable Insight
     ↑                           ↓
CSV Upload ←→ Workspace ←→ Memory System
```

**Template Components:**
- **System Prompt**: `templates/qwen_churn_system_prompt.template.md`
- **Docker Config**: `templates/docker-compose.qwen-churn.template.yml`  
- **Memory Framework**: `templates/churn_memory_template.md`
- **WebUI Integration**: Standard Ollama connection (auto-discovery)

### � **Data Requirements**
**Supported Formats:** CSV files with customer records

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
- **`--cleanup`**: Removes only temporary staging files (safe, preserves your data)
- **`--cleanup-all`**: Removes everything including Docker volumes (⚠️ **WARNING**: deletes all conversation history and WebUI settings)

### 💡 **Best Practices**
- **Ask specific questions**: Reference particular segments or behaviors
- **Build on insights**: Use follow-up questions to dive deeper  
- **Provide context**: Explain your business model and customer segments
- **Clean your data**: Remove duplicates and handle missing values before upload

---

## 🏗️ Code Architecture

The project uses a **modular architecture** with separated concerns:

- **Main Script** (`start_llama_webui.py`): Entry point and orchestration
- **Configuration** (`service_config.py`): Model settings and quantization options  
- **Service Managers**: Individual managers for each service component
  - `llama_server_manager.py`: llama.cpp operations and model management
  - `ollama_manager.py`: Ollama service and model pulling
  - `webui_manager.py`: Open WebUI health checks
  - `utility_manager.py`: Subprocess and utility functions

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
- [Qwen Team](https://huggingface.co/Qwen) - Qwen2.5-Coder (Business Analytics Mode)

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

