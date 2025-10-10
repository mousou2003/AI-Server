#!/usr/bin/env python3
"""
Standalone WebUI Starter

This script starts Ollama and Open WebUI without any specialized assistant.
This gives you a generic AI chat interface with access to any models you have.
"""

import sys
import argparse
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from utility_manager import UtilityManager
from ollama_manager import OllamaManager
from webui_manager import WebUIManager


def main():
    parser = argparse.ArgumentParser(
        description="Start standalone Ollama + WebUI (no specialized assistant)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_webui.py                 # Start with GPU acceleration
  python start_webui.py --cpu           # Use CPU-only mode
  python start_webui.py --stop          # Stop the services
  python start_webui.py --status        # Check status
  python start_webui.py --open          # Open WebUI in browser
  python start_webui.py --logs          # Show container logs

Access:
  WebUI: http://localhost:3000
  Ollama API: http://localhost:11434

Notes:
  - This provides generic AI chat without specialized prompts
  - You can chat with any Ollama models you have installed
  - Use assistant scripts for specialized functionality (yoga, churn analysis)
        """
    )
    
    parser.add_argument("--cpu", action="store_true", 
                       help="Use CPU-only mode (no GPU acceleration)")
    parser.add_argument("--stop", action="store_true", 
                       help="Stop the services")
    parser.add_argument("--status", action="store_true", 
                       help="Check status of running services")
    parser.add_argument("--open", action="store_true", 
                       help="Open WebUI in default browser")
    parser.add_argument("--logs", action="store_true", 
                       help="Show container logs")
    parser.add_argument("--remove-volumes", action="store_true", 
                       help="Remove Docker volumes when stopping (complete cleanup)")
    
    args = parser.parse_args()
    
    # Build compose files list
    compose_files = ["docker-compose.ollama.yml", "docker-compose.webui.yml"]
    if not args.cpu:
        compose_files.append("docker-compose.gpu-override.yml")
    
    # Create managers for status checking
    ollama_manager = OllamaManager()
    webui_manager = WebUIManager()
    
    if args.stop:
        print("🛑 Stopping Ollama + WebUI services...")
        additional_args = "-v" if args.remove_volumes else ""
        cmd = UtilityManager.build_compose_command(
            compose_files=compose_files,
            project_name="webui-standalone",
            action="down",
            additional_args=additional_args
        )
        UtilityManager.run_subprocess(cmd, show_output=True)
        return
    
    if args.status:
        print("🔍 Checking service status...")
        print("\n📊 Container Status:")
        UtilityManager.run_subprocess("docker ps --filter name=ollama --filter name=webui", show_output=True)
        return
    
    if args.open:
        print("🌐 Opening WebUI in browser...")
        webui_manager.open_in_browser()
        return
    
    if args.logs:
        print("📋 Showing container logs...")
        print("\n🤖 Ollama logs:")
        UtilityManager.run_subprocess("docker logs ollama --tail 20", show_output=True)
        print("\n🌐 WebUI logs:")
        UtilityManager.run_subprocess("docker logs open-webui --tail 20", show_output=True)
        return
    
    # Start services
    mode_info = "CPU-only mode" if args.cpu else "GPU-accelerated mode"
    print(f"🚀 Starting Ollama + WebUI ({mode_info})...")
    print("=" * 50)
    
    # Check system requirements
    if not UtilityManager.check_system_requirements(
        model_name="Generic Ollama Models",
        model_description="Standalone Ollama + WebUI setup",
        vram_requirement="8GB+ RAM (CPU mode)" if args.cpu else "8GB+ VRAM (GPU mode)"
    ):
        return False
    
    # Ensure the external network exists
    UtilityManager.ensure_docker_network("ai_network")
    
    # Start services
    print("🐳 Starting Docker containers...")
    cmd = UtilityManager.build_compose_command(
        compose_files=compose_files,
        project_name="webui-standalone",
        action="up",
        additional_args="-d"
    )
    
    result = UtilityManager.run_subprocess(cmd, check=False)
    if result.returncode != 0:
        print(f"❌ Failed to start containers: {result.stderr}")
        return False
    
    print("✅ Containers started successfully")
    print("⏳ Waiting for services to be ready...")
    
    # Wait for services
    if not ollama_manager.wait_for_ready(timeout=120):
        print("❌ Ollama service failed to start")
        return False
    
    if not webui_manager.wait_for_ready(timeout=180):
        print("❌ WebUI service failed to start")
        return False
    
    print()
    print("🎉 Ollama + WebUI is ready!")
    print()
    print("🌐 Access WebUI at: http://localhost:3000")
    print("🤖 Ollama API at: http://localhost:11434")
    print()
    print("💡 What you can do:")
    print("   • Chat with any Ollama models you have installed")
    print("   • Pull new models through the WebUI")
    print("   • Upload and analyze documents")
    print("   • Use as a general-purpose AI assistant")
    print()
    print("🎯 For specialized assistants, use:")
    print("   • python start_yoga_assistant.py (yoga sequences)")
    print("   • python start_qwen_churn_assistant.py (business analysis)")
    
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
        sys.exit(1)