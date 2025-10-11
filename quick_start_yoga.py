#!/usr/bin/env python3
"""
Quick Start Yoga Assistant (No Health Checks)

This script starts the yoga assistant without relying on Docker health checks,
which can sometimes cause startup issues.
"""

import sys
import time
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from utility_manager import UtilityManager
from ollama_manager import OllamaManager
from webui_manager import WebUIManager


def main():
    print("🧘 Quick Start Yoga Assistant (Simplified)")
    print("=" * 50)
    
    # Stop any existing containers
    print("🧹 Cleaning up existing containers...")
    UtilityManager.run_subprocess(
        "docker stop ollama-yoga-assistant open-webui-yoga-assistant 2>nul || true",
        check=False,
        timeout=30
    )
    UtilityManager.run_subprocess(
        "docker rm ollama-yoga-assistant open-webui-yoga-assistant 2>nul || true",
        check=False,
        timeout=30
    )
    
    # Ensure network exists
    UtilityManager.ensure_docker_network("ai_network")
    
    # Start Ollama first (without health check dependency)
    print("🚀 Starting Ollama container...")
    
    # Use docker run instead of compose for more control
    ollama_cmd = (
        "docker run -d --name ollama-yoga-assistant "
        "--network ai_network "
        "-p 11434:11434 "
        "-v ./.ollama/yoga-assistant:/root/.ollama "
        "-e OLLAMA_KEEP_ALIVE=24h "
        "-e OLLAMA_HOST=0.0.0.0 "
        "ollama/ollama:latest"
    )
    
    result = UtilityManager.run_subprocess(ollama_cmd, check=False, timeout=60)
    if result.returncode != 0:
        print(f"❌ Failed to start Ollama: {result.stderr}")
        return False
    
    print("✅ Ollama container started")
    
    # Wait for Ollama to initialize
    print("⏳ Waiting for Ollama to initialize...")
    ollama_manager = OllamaManager()
    ollama_manager.config["name"] = "ollama-yoga-assistant"
    
    if not ollama_manager.wait_for_api(retries=24):  # 24 retries = ~2 minutes
        print("❌ Ollama failed to start properly")
        return False
    
    print("✅ Ollama is ready")
    
    # Start WebUI
    print("🌐 Starting WebUI container...")
    webui_cmd = (
        'docker run -d --name open-webui-yoga-assistant '
        '--network ai_network '
        '-p 3000:8080 '
        '-v ./.webui/yoga-assistant/data:/app/backend/data '
        '-v ./.webui/yoga-assistant/workspace:/app/backend/workspace '
        '-e "WEBUI_NAME=Yoga Sequence Generator System Prompt" '
        '-e OLLAMA_BASE_URL=http://ollama-yoga-assistant:11434 '
        '-e WEBUI_AUTH=False '
        'ghcr.io/open-webui/open-webui:main'
    )
    
    result = UtilityManager.run_subprocess(webui_cmd, check=False, timeout=60)
    if result.returncode != 0:
        print(f"❌ Failed to start WebUI: {result.stderr}")
        return False
    
    print("✅ WebUI container started")
    
    # Wait for WebUI to initialize
    print("⏳ Waiting for WebUI to initialize...")
    webui_manager = WebUIManager()
    webui_manager.config["name"] = "open-webui-yoga-assistant"
    
    if not webui_manager.wait_for_api_with_progress(retries=36):  # 36 retries = ~3 minutes
        print("❌ WebUI failed to start properly")
        return False
    
    print("✅ WebUI is ready")
    
    # Pull the base model if needed
    print("🤖 Ensuring base model is available...")
    pull_result = UtilityManager.run_subprocess(
        "docker exec ollama-yoga-assistant ollama pull qwen2.5:7b-instruct",
        check=False,
        timeout=600  # 10 minutes for model download
    )
    
    if pull_result.returncode == 0:
        print("✅ Base model available")
    else:
        print("⚠️  Base model pull failed, but continuing...")
    
    print()
    print("🎉 Yoga Assistant is ready!")
    print()
    print("🌐 Access WebUI at: http://localhost:3000")
    print("🤖 Ollama API at: http://localhost:11434")
    print()
    print("💡 Next steps:")
    print("   1. Open http://localhost:3000 in your browser")
    print("   2. Select 'qwen2.5:7b-instruct' model")
    print("   3. Start creating yoga sequences!")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)