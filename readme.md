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
| LLM Inference | [`Ollama`](https://ollama.ai)                           | Runs qwen2.5-coder:7b and gemma3:4B locally |
| VSCode Plugin | [`Continue`](https://continue.dev)                       | Inline chat & refactor tools             |
| Web Interface | [`Open WebUI`](https://github.com/open-webui/open-webui) | OpenAI-compatible UI & proxy             |
| VPN Layer     | [`Tailscale`](https://tailscale.com)                     | Encrypted access across devices          |

---

## 📂 Project Structure

This repository includes:

- `docker-compose.yml`: Runs Open WebUI and Ollama-compatible backend (GPU mode)
- `docker-compose.cpu.yml`: CPU-only version for systems without GPU
- `start_llama_webui.py`: Comprehensive Python script that handles model downloading, Docker orchestration, and service management
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
- **GPU Mode**: NVIDIA GPU with 8GB+ VRAM (recommended)
- **CPU Mode**: 8GB+ RAM (16GB+ recommended for larger quantizations)
- **Storage**: 6-17GB free space (depending on model quantization)

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
- ✅ Download the DeepSeek Coder V2 model if needed (~6-17GB depending on quantization)
- 🚀 Start llama.cpp server with GPU acceleration (or CPU-only mode)  
- 🐳 Launch Docker containers (Open WebUI + Ollama) using the appropriate compose file
- 🧪 Test the endpoints to ensure everything works
- 🎯 Handle quantization selection interactively or via command-line options

The script intelligently selects between `docker-compose.yml` (GPU mode) and `docker-compose.cpu.yml` (CPU mode) based on your chosen configuration.

**CPU vs GPU Mode:**
- **GPU Mode**: Faster inference, requires NVIDIA GPU with 8GB+ VRAM
- **CPU Mode**: Slower inference, works on any system with 8GB+ RAM

### 3. Configure VS Code

Install the Continue extension for VS Code: https://www.continue.dev/

Create or update your Continue configuration file at `~/.continue/config.yaml`:

```yaml
name: Local Assistant
version: 1.0.0
schema: v1
models:
  - name: qwen2.5-coder 7b
    provider: ollama
    model: qwen2.5-coder:7b
    apiBase: http://your-tailscale-hostname.ts.net:11434
    roles:
      - chat
      - edit
      - apply
      - autocomplete
    defaultCompletionOptions:
      contextLength: 32768
      maxTokens: 8192
  - name: gemma3 4B
    provider: ollama
    model: gemma3:4B
    apiBase: http://your-tailscale-hostname.ts.net:11434
    roles:
      - chat
      - edit
      - apply
    capabilities:
      - image_input
    defaultCompletionOptions:
      contextLength: 128000
context:
  - provider: code
  - provider: docs
  - provider: diff
  - provider: terminal
  - provider: problems
  - provider: folder
  - provider: codebase
docs:
  - name: continue
    faviconUrl: ""
    startUrl: https://docs.continue.dev/
```

**For local access**, update the `apiBase` to use `localhost`:
```yaml
apiBase: http://localhost:11434
```

**For remote access via Tailscale**, use your Tailscale hostname as shown in the config above:
```yaml
apiBase: http://your-tailscale-hostname.ts.net:11434
```

---

## 🔎 Verify It Works

- Visit `http://localhost:3000` to access the Open WebUI in your browser
- Test the Ollama API directly at `http://localhost:11434/api/tags`
- Use `Ctrl+K` in VS Code with Continue extension to prompt your model
- For remote access: Use your Tailscale hostname (e.g., `http://your-tailscale-hostname.ts.net:11434`) to access from any device

---

## 🙌 Acknowledgments

This project builds on the work of:

- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [Open WebUI](https://github.com/open-webui/open-webui)
- [Continue](https://continue.dev)
- [DeepSeek](https://huggingface.co/deepseek-ai)

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

