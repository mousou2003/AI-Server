# Separated Docker Compose Files Guide

This guide explains how to use the separated docker-compose files for individual services in the AI Server setup.

## Overview

The AI Server uses a clean modular Docker Compose architecture with base files and overrides for maximum flexibility.

### Base Service Files (CPU-optimized by default)
- `docker-compose.ollama.yml` - Ollama service
- `docker-compose.webui.yml` - Open WebUI service (enhanced with features)
- `docker-compose.llama.yml` - Llama.cpp server service

### Universal Override
- `docker-compose.gpu-override.yml` - Adds GPU acceleration to any service combination

### Script-Specific Overrides
- `docker-compose.qwen-churn-override.yml` - Qwen churn assistant customizations
- `docker-compose.llama-webui-override.yml` - Llama WebUI script customizations

## Network Setup

All services use an external Docker network called `ai_network`. You must create this network before using any compose files:

```bash
# Create the network (do this once)
python network_manager.py create

# Or manually:
docker network create ai_network
```

## Usage Examples

### 1. Start Individual Services (GPU Mode - Default)

Start only Ollama (GPU mode):
```bash
docker compose -f docker-compose.ollama.yml up -d
```

Start only WebUI (connects to any Ollama instance):
```bash
docker compose -f docker-compose.webui.yml up -d
```

Start only Llama.cpp server:
```bash
docker compose -f docker-compose.llama.yml up -d
```

### 2. Start Individual Services (CPU Mode - Using Overrides)

Start only Ollama (CPU mode):
```bash
docker compose -f docker-compose.ollama.yml -f docker-compose.ollama.cpu-override.yml up -d
```

Start only WebUI (CPU mode):
```bash
docker compose -f docker-compose.webui.yml -f docker-compose.webui.cpu-override.yml up -d
```

Start only Llama.cpp server (CPU mode):
```bash  
docker compose -f docker-compose.llama.yml -f docker-compose.llama.cpu-override.yml up -d
```

### 3. Using the Dynamic Script (Easiest)

```bash
# GPU mode (default)
python dynamic_compose.py ollama
python dynamic_compose.py webui
python dynamic_compose.py llama

# CPU mode
python dynamic_compose.py ollama --cpu
python dynamic_compose.py webui --cpu
python dynamic_compose.py llama --cpu

# Stop services
python dynamic_compose.py ollama --stop
```

### 4. Mix and Match Services

Start Ollama (GPU) and WebUI (CPU) separately:
```bash
docker compose -f docker-compose.ollama.yml up -d
docker compose -f docker-compose.webui.yml -f docker-compose.webui.cpu-override.yml up -d
```

Start Llama.cpp server (CPU) and WebUI (GPU):
```bash
docker compose -f docker-compose.llama.yml -f docker-compose.llama.cpu-override.yml up -d
OLLAMA_BASE_URL=http://llama-server:11435 docker compose -f docker-compose.webui.yml up -d
```

### 3. Environment Variables

The WebUI services support the `OLLAMA_BASE_URL` environment variable to connect to different backends:

```bash
# Connect WebUI to a different Ollama instance
OLLAMA_BASE_URL=http://localhost:11434 docker compose -f docker-compose.webui.yml up -d

# Connect WebUI to Llama.cpp server instead of Ollama
OLLAMA_BASE_URL=http://llama-server:11435 docker compose -f docker-compose.webui.yml up -d
```

### 4. Managing Services

Stop individual services:
```bash
docker compose -f docker-compose.ollama.yml down
docker compose -f docker-compose.webui.yml down
```

View logs for specific service:
```bash
docker compose -f docker-compose.ollama.yml logs -f
```

Restart specific service:
```bash
docker compose -f docker-compose.ollama.yml restart
```

## Network Management

Check which containers are connected to the network:
```bash
python network_manager.py list
```

Remove the network (stops all services first):
```bash
# Stop all services first
docker compose -f docker-compose.ollama.yml down
docker compose -f docker-compose.webui.yml down
docker compose -f docker-compose.llama.yml down

# Then remove network
python network_manager.py remove
```

## Benefits of Separation

1. **Modularity**: Start only the services you need
2. **Resource Management**: Better control over resource allocation
3. **Development**: Easier to test individual components
4. **Debugging**: Isolate issues to specific services
5. **Scaling**: Can run multiple instances of services on different ports
6. **Maintenance**: Update or restart individual services without affecting others

## Troubleshooting

### Network Issues
If services can't communicate, ensure the network exists:
```bash
docker network ls | grep ai_network
```

If missing, recreate it:
```bash
python network_manager.py create
```

### Port Conflicts
Each service uses different ports:
- Ollama: 11434
- Llama.cpp: 11435
- WebUI: 3000

To use different ports, override in the compose command:
```bash
# Run WebUI on port 3001
docker compose -f docker-compose.webui.yml -p custom up -d
# Then manually edit the port mapping in the file or use overrides
```

### Volume Persistence
Each service maintains its own volumes:
- Ollama: `./models/.ollama`
- Llama.cpp: `./models/.llama`
- WebUI: `open-webui` Docker volume

Data persists across container restarts automatically.
