# 🧠 Local AI-Powered Dev Assistant

This project sets up a **fully local AI coding assistant** with:

- 💬 Inline prompt support in VS Code via **Continue**
- ⚙️ High-performance LLMs served by **Ollama** (qwen2.5-coder:7b & gemma3:4B)
- 🌐 Web interface via **Open WebUI**
- 🔐 Secure remote access with **Tailscale**

## 🚀 Key Features

- **Private & Secure**: All computation happens locally on your hardware
- **Fast Inference**: Optimized models (qwen2.5-coder:7b for coding, gemma3:4B with image support)
- **Multi-Model Support**: Code generation and chat capabilities with visual understanding
- **Multi-platform Access**: Chat from browser, IDE, or mobile (via Tailscale)
- **Code-Centric Models**: Optimized for various programming languages and tasks

---

## 🏗️ System Architecture

```
[freedom (VS Code + Continue)] <---> [last-nuc (llama.cpp + Open WebUI)]
                 ↖
               Mobile (via Tailscale)
```

- **freedom**: HP EliteBook G9, Windows 11 Pro, 32GB RAM
- **last-nuc**: Intel NUC13RNGi9, RTX 3060 Ti, 64GB RAM

---

## 📦 Tech Stack

| Component     | Tool                                                     | Description                              |
| ------------- | -------------------------------------------------------- | ---------------------------------------- |
| LLM Inference | [`llama.cpp`](https://github.com/ggerganov/llama.cpp)   | High-performance inference engine for GGUF models |
| LLM Backend   | [`Ollama`](https://ollama.ai)                           | Secondary backend for qwen2.5-coder:7b and gemma2:9b |
| Primary Model | [`DeepSeek Coder V2 Lite`](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct) | 15.7B parameter coding specialist model |
| VSCode Plugin | [`Continue`](https://continue.dev)                       | Inline chat & refactor tools             |
| Web Interface | [`Open WebUI`](https://github.com/open-webui/open-webui) | OpenAI-compatible UI & proxy             |
| VPN Layer     | [`Tailscale`](https://tailscale.com)                     | Encrypted access across devices          |

---

## 📂 Project Structure

This repository includes:

### Core Files
- `start_llama_webui.py`: Main orchestration script and entry point
- `service_config.py`: Configuration management and quantization options
- `llama_server_manager.py`: Llama.cpp server operations and model management
- `ollama_manager.py`: Ollama service management and model pulling
- `webui_manager.py`: Open WebUI health checking and management
- `utility_manager.py`: Utility functions for subprocess operations

### Configuration Files
- `docker-compose.yml`: GPU-accelerated mode with NVIDIA container runtime
- `docker-compose.cpu.yml`: CPU-only version for systems without GPU
- `CODE_STRUCTURE.md`: Detailed documentation of the modular code architecture

### Directories
- `models/`: Directory for storing downloaded GGUF models (auto-created)
- `readme.md`: This documentation file
- `.gitignore`: Excludes models, cache, and test output from version control

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

## 🧪 How to Use

### 1. Clone the Repo

```bash
git clone https://github.com/mousou2003/AI-Server
cd AI-Server
```

### 2. Start Everything

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

### 3. Configure VS Code

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

## 🔎 Verify It Works

### Local Testing
- Visit `http://localhost:3000` to access the Open WebUI in your browser
- Test the llama.cpp API: `http://localhost:11435/v1/models`
- Test the Ollama API: `http://localhost:11434/api/tags`
- Use `Ctrl+K` in VS Code with Continue extension to prompt your models

### Performance Verification
Recent deployment test results on Windows 11 with WSL2:
- ✅ **CPU Mode**: Q2_K quantization running at ~20 tokens/second
- ✅ **Model Size**: 6.4GB DeepSeek Coder V2 Lite loaded successfully  
- ✅ **Context Length**: 163,840 tokens (160K context window)
- ✅ **Parameters**: 15.7B parameters running smoothly in CPU-only mode

### Remote Access
For remote access via Tailscale, replace `localhost` with your Tailscale hostname:
- Open WebUI: `http://your-tailscale-hostname.ts.net:3000`
- llama.cpp API: `http://your-tailscale-hostname.ts.net:11435`
- Ollama API: `http://your-tailscale-hostname.ts.net:11434`

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

See `CODE_STRUCTURE.md` for detailed architecture documentation.

---

## 🙌 Acknowledgments

This project builds on the work of:

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - High-performance LLM inference
- [DeepSeek](https://huggingface.co/deepseek-ai) - DeepSeek Coder V2 Lite model
- [Open WebUI](https://github.com/open-webui/open-webui) - Modern web interface
- [Continue](https://continue.dev) - VS Code AI assistant extension
- [Ollama](https://ollama.ai) - Additional model backend

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

