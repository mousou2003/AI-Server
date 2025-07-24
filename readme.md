# 🧠 Local AI-Powered Dev Assistant

This project sets up a **fully local AI coding assistant** with:

- 💬 Inline prompt support in VS Code via **Continue**
- ⚙️ High-performance LLMs served by **llama.cpp** (DeepSeek-Coder V2)
- 🌐 Web interface via **Open WebUI**
- 🔐 Secure remote access with **Tailscale**

## 🚀 Key Features

- **Private & Secure**: All computation happens locally on your hardware
- **Fast Inference**: Quantized GGUF model (Q4\_K\_M) accelerated via GPU
- **Multi-platform Access**: Chat from browser, IDE, or mobile (via Tailscale)
- **Code-Centric Models**: Optimized for Python, Rust, and general code generation

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
| LLM Inference | [`llama.cpp`](https://github.com/ggerganov/llama.cpp)    | Runs quantized DeepSeek-Coder V2 locally |
| VSCode Plugin | [`Continue`](https://continue.dev)                       | Inline chat & refactor tools             |
| Web Interface | [`Open WebUI`](https://github.com/open-webui/open-webui) | OpenAI-compatible UI & proxy             |
| VPN Layer     | [`Tailscale`](https://tailscale.com)                     | Encrypted access across devices          |

---

## 📂 Project Structure

This repository includes:

- `docker-compose.yml`: Runs Open WebUI and Ollama-compatible backend
- `start_llama_webui.py`: Starts llama.cpp server with the correct model
- `config.yaml`: Model & endpoint configuration
- `continue.json`: Cursor (or Continue) configuration to link to local model

---

## 🧪 How to Use

### 1. Clone the Repo

```bash
git clone https://github.com/mousou2003/AI-Server
cd AI-Server
```

### 2. Pull the Model

Use any GGUF model compatible with `llama.cpp`. Example:

```bash
curl -L -o models/deepseek-v2.gguf https://huggingface.co/...
```

### 3. Start the LLM Backend

```bash
python start_llama_webui.py
```

### 4. Start WebUI Proxy (with GPU acceleration)

```bash
docker run -d -p 3000:8080 --gpus=all \
  -v ollama:/root/.ollama \
  -v open-webui:/app/backend/data \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:ollama
```

### 5. Configure VS Code

Update `~/.cursor/config.json`:

```json
{
  "providers": {
    "local-llama": {
      "baseURL": "http://<last-nuc tailscale IP>:8080",
      "model": "deepseek-coder-v2",
      "apiKey": "anystring"
    }
  },
  "defaultProvider": "local-llama"
}
```

---

## 🔎 Verify It Works

- Visit `http://<last-nuc IP>:3000` to chat in browser
- Use `Ctrl+K` in VS Code to prompt your model
- Try it from mobile with Tailscale active

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

