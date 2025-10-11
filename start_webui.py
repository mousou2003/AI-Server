#!/usr/bin/env python3
"""
Standalone WebUI Starter

This script starts Open WebUI that connects to a running Ollama instance.
Part of the new modular workflow: Ollama + Custom Models + WebUI (all separate).

Prerequisites:
- Ollama must be running (use: python start_ollama.py)
- Custom models can be deployed separately (use: python start_custom_assistant.py)
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
        description="Start WebUI for running Ollama instance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_webui.py                 # Start WebUI
  python start_webui.py --stop          # Stop WebUI
  python start_webui.py --status        # Check status
  python start_webui.py --open          # Open WebUI in browser
  python start_webui.py --logs          # Show WebUI logs

New Modular Workflow:
  1. python start_ollama.py              # Start Ollama first
  2. python start_custom_assistant.py templates/your_template.json  # Deploy custom model
  3. python start_webui.py               # Start WebUI (this script)

Access:
  WebUI: http://localhost:3000
  Ollama API: http://localhost:11434

Prerequisites:
  - Ollama must be running (check with: python start_ollama.py --status)
        """
    )
    
    parser.add_argument("--stop", action="store_true", 
                       help="Stop WebUI service")
    parser.add_argument("--status", action="store_true", 
                       help="Check status of WebUI service")
    parser.add_argument("--open", action="store_true", 
                       help="Open WebUI in default browser")
    parser.add_argument("--logs", action="store_true", 
                       help="Show WebUI container logs")
    parser.add_argument("--remove-volumes", action="store_true", 
                       help="Remove WebUI volumes when stopping (complete cleanup)")
    
    args = parser.parse_args()
    
    # Only WebUI compose file needed (Ollama runs separately)
    compose_files = ["docker-compose.webui.yml"]
    
    # Create managers for status checking
    from ollama_manager import OllamaManager
    ollama_manager = OllamaManager()
    webui_manager = WebUIManager()
    
    if args.stop:
        print("🛑 Stopping WebUI service...")
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
        print("🔍 Checking WebUI service status...")
        print("\n📊 WebUI Container Status:")
        UtilityManager.run_subprocess("docker ps --filter name=webui", show_output=True)
        
        # Check if Ollama is running
        is_running, status_msg = ollama_manager.get_api_status()
        print(f"\n🤖 Ollama Status: {status_msg}")
        return
    
    if args.open:
        print("🌐 Opening WebUI in browser...")
        webui_manager.open_in_browser()
        return
    
    if args.logs:
        print("📋 Showing WebUI container logs...")
        UtilityManager.run_subprocess("docker logs open-webui --tail 20", show_output=True)
        return
    
    # Check if Ollama is running first
    is_running, status_msg = ollama_manager.get_api_status()
    if not is_running:
        print("❌ Ollama is not running!")
        print("💡 Start Ollama first: python start_ollama.py")
        return False
    
    print(f"✅ Ollama detected: {status_msg}")
    
    # Start WebUI service
    print("🚀 Starting WebUI Service...")
    print("=" * 50)
    
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
    
    print("✅ WebUI container started")
    print("⏳ Waiting for WebUI to be ready...")
    
    # Wait for WebUI
    if not webui_manager.wait_for_api_with_progress(retries=36):  # 3 minutes
        print("❌ WebUI service failed to start")
        return False
    
    print()
    print("🎉 WebUI is ready!")
    print()
    print("🌐 Access WebUI at: http://localhost:3000")
    print("🤖 Ollama API at: http://localhost:11434")
    print()
    print("💡 Available models:")
    models = ollama_manager.list_models()
    if models:
        for model in models:
            print(f"   • {model}")
    else:
        print("   No models found - deploy some with: python start_custom_assistant.py")
    print()
    print("🎯 Deploy custom assistants:")
    print("   • python start_custom_assistant.py templates/yoga_sequence_system_prompt.template.json")
    print("   • python start_custom_assistant.py templates/qwen_churn_system_prompt.template.json")
    
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