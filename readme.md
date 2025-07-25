# 🧠 Local AI-Powered Dev Assistant

This project sets up a **fully local AI coding assistant** with:

- 💬 Inline prompt support in VS Code via **Continue**
- ⚙️ High-performance LLMs served by **llama.cpp** (DeepSeek-Coder V2)
- 🌐 Web interface via **Open WebUI**
- 🔐 **HTTPS support** with nginx reverse proxy (new!)
- 🔐 Secure remote access with **Tailscale**

## 🚀 Key Features

- **Private & Secure**: All computation happens locally on your hardware
- **HTTPS Encryption**: Secure web access with SSL/TLS termination
- **Fast Inference**: Quantized GGUF model (Q4\_K\_M) accelerated via GPU
- **Multi-platform Access**: Chat from browser, IDE, or mobile (via Tailscale)
- **Code-Centric Models**: Optimized for Python, Rust, and general code generation
- **Easy Setup**: Automated SSL certificate generation for development

---

## 🏗️ System Architecture

```
[freedom (VS Code + Continue)] <---> [last-nuc (nginx → Open WebUI + llama.cpp)]
                 ↖
               Mobile (via Tailscale + HTTPS)
```

- **freedom**: HP EliteBook G9, Windows 11 Pro, 32GB RAM
- **last-nuc**: Intel NUC13RNGi9, RTX 3060 Ti, 64GB RAM
- **nginx**: Reverse proxy with SSL termination for secure HTTPS access

---

## 📦 Tech Stack

| Component       | Tool                                                     | Description                              |
| --------------- | -------------------------------------------------------- | ---------------------------------------- |
| LLM Inference   | [`llama.cpp`](https://github.com/ggerganov/llama.cpp)    | Runs quantized DeepSeek-Coder V2 locally |
| VSCode Plugin   | [`Continue`](https://continue.dev)                       | Inline chat & refactor tools             |
| Web Interface   | [`Open WebUI`](https://github.com/open-webui/open-webui) | OpenAI-compatible UI & proxy             |
| Reverse Proxy   | [`nginx`](https://nginx.org)                            | SSL termination & security headers        |
| VPN Layer       | [`Tailscale`](https://tailscale.com)                     | Encrypted access across devices          |

---

## 📂 Project Structure

This repository includes:

- `docker-compose.yml`: Multi-service setup with nginx, Open WebUI, and Ollama
- `start_llama_webui.py`: Automated startup script for llama.cpp server
- `nginx/nginx.conf`: Reverse proxy configuration with SSL/TLS
- `generate-ssl-certs.ps1`: PowerShell script for SSL certificate generation
- `generate-ssl-certs.sh`: Bash script for SSL certificate generation  
- `HTTPS-README.md`: Detailed HTTPS setup and troubleshooting guide
- `config.yaml`: Model & endpoint configuration
- `continue.json`: Cursor (or Continue) configuration for local model integration

## 🔀 Branch Structure

- **`main`**: Original HTTP-only setup (port 3000)
- **`feature/https-support`**: Enhanced HTTPS setup with nginx reverse proxy

---

## 🧪 How to Use

### Option 1: HTTPS Setup (Recommended) 🔒

```bash
git clone https://github.com/mousou2003/AI-Server
cd AI-Server
git checkout feature/https-support
```

**Generate SSL certificates:**
```powershell
# Windows (PowerShell)
.\generate-ssl-certs.ps1

# Linux/Mac (Bash)  
chmod +x generate-ssl-certs.sh && ./generate-ssl-certs.sh
```

**Start the services:**
```bash
python start_llama_webui.py
```

**Access the interface:**
- 🌐 **HTTPS**: https://localhost (recommended)
- 🔄 **HTTP**: http://localhost (auto-redirects to HTTPS)

### Option 2: HTTP Setup (Simple)

```bash
git clone https://github.com/mousou2003/AI-Server
cd AI-Server
# Stay on main branch for HTTP setup
python start_llama_webui.py
```

**Access the interface:**
- 🌐 **HTTP**: http://localhost:3000

### What the startup script does:

- ✅ Download the DeepSeek Coder V2 model if needed (~10GB)
- 🚀 Start llama.cpp server with GPU acceleration
- 🐳 Launch Docker containers (nginx + Open WebUI + Ollama)
- 🧪 Test the endpoints to ensure everything works

### 3. Configure VS Code

Update your Continue configuration (`~/.continue/config.yaml`):

Install Continue extension for VS Code: https://www.continue.dev/
Modify the config yaml file in your personal folder.

**For HTTPS setup:**
```yaml
name: Local Assistant
version: 1.0.0
schema: v1
models:
  - name: DeepSeek Coder
    provider: openai
    model: DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M
    apiBase: https://<last-nuc tailscale IP>:11435/v1  # Note: HTTPS
    roles:
      - chat
      - edit
      - apply
      - autocomplete
context:
  - provider: code
  - provider: docs
  - provider: diff
  - provider: terminal
  - provider: problems
  - provider: folder
  - provider: codebase
```

**For HTTP setup:**
```yaml
name: Local Assistant
version: 1.0.0
schema: v1
models:
  - name: DeepSeek Coder
    provider: openai
    model: DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M
    apiBase: http://<last-nuc tailscale IP>:11435/v1   # Note: HTTP
    roles:
      - chat
      - edit
      - apply
      - autocomplete
context:
  - provider: code
  - provider: docs
  - provider: diff
  - provider: terminal
  - provider: problems
  - provider: folder
  - provider: codebase
```

---

## 🔎 Verify It Works

**HTTPS Setup:**
- Visit `https://<last-nuc IP>` to chat in browser (secure)
- Use `Ctrl+K` in VS Code to prompt your model
- Try it from mobile with Tailscale active

**HTTP Setup:**
- Visit `http://<last-nuc IP>:3000` to chat in browser
- Use `Ctrl+K` in VS Code to prompt your model
- Try it from mobile with Tailscale active

**Note**: For HTTPS setup, your browser may show a security warning for self-signed certificates. Click "Advanced" → "Proceed to localhost" to continue.

---

## 🔒 Security Features (HTTPS Branch)

- **SSL/TLS Encryption**: All web traffic encrypted
- **Automatic HTTPS Redirect**: HTTP requests redirect to HTTPS
- **Security Headers**: HSTS, X-Frame-Options, X-Content-Type-Options
- **Rate Limiting**: Protection against API abuse
- **Modern SSL Configuration**: TLS 1.2+ with secure cipher suites

For detailed HTTPS setup and troubleshooting, see `HTTPS-README.md`.

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

