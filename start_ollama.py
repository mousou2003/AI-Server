#!/usr/bin/env python3
"""
Ollama Standalone Manager

This script provides direct control over the Ollama service without WebUI or specialized assistants.
Use this for manual Ollama management in the new modular workflow.
"""

import sys
import argparse
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from ollama_manager import OllamaManager
from utility_manager import UtilityManager


def main():
    parser = argparse.ArgumentParser(
        description="Manage Ollama service standalone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_ollama.py                # Start Ollama with GPU acceleration
  python start_ollama.py --cpu          # Start Ollama in CPU-only mode
  python start_ollama.py --stop         # Stop Ollama service
  python start_ollama.py --status       # Check Ollama status
  python start_ollama.py --pull         # Pull default models
  python start_ollama.py --list         # List available models
  python start_ollama.py --logs         # Show Ollama logs

New Modular Workflow:
  1. python start_ollama.py             # Start Ollama first
  2. python start_custom_assistant.py templates/your_template.json  # Deploy custom model
  3. python start_webui.py              # Start WebUI separately

Access:
  Ollama API: http://localhost:11434
        """
    )
    
    parser.add_argument("--cpu", action="store_true", 
                       help="Use CPU-only mode (no GPU acceleration)")
    parser.add_argument("--stop", action="store_true", 
                       help="Stop Ollama service")
    parser.add_argument("--status", action="store_true", 
                       help="Check status of Ollama service")
    parser.add_argument("--pull", action="store_true", 
                       help="Pull default models after starting")
    parser.add_argument("--list", action="store_true", 
                       help="List all available models")
    parser.add_argument("--logs", action="store_true", 
                       help="Show Ollama container logs")
    parser.add_argument("--model", type=str,
                       help="Pull a specific model (use with --pull)")
    parser.add_argument("--override", type=str,
                       help="Specify a custom Docker Compose override file (e.g., docker-compose.yoga-assistant-override.yml)")
    
    args = parser.parse_args()
    
    # Build compose files list
    compose_files = ["docker-compose.ollama.yml"]
    if args.override:
        compose_files.append(args.override)
    elif not args.cpu:
        compose_files.append("docker-compose.gpu-override.yml")
    
    # Create Ollama manager
    ollama_manager = OllamaManager()
    
    if args.stop:
        print("🛑 Stopping Ollama service...")
        cmd = UtilityManager.build_compose_command(
            compose_files=compose_files,
            project_name="ollama-standalone",
            action="down"
        )
        UtilityManager.run_subprocess(cmd, show_output=True)
        return
    
    if args.status:
        print("🔍 Checking Ollama service status...")
        print("\n📊 Container Status:")
        UtilityManager.run_subprocess("docker ps --filter name=ollama", show_output=True)
        
        # Check API status
        is_running, status_msg = ollama_manager.get_api_status()
        print(f"\n🌐 API Status: {status_msg}")
        return
    
    if args.logs:
        print("📋 Showing Ollama container logs...")
        UtilityManager.run_subprocess("docker logs ollama --tail 20", show_output=True)
        return
    
    if args.list:
        print("📋 Available models in Ollama:")
        models = ollama_manager.list_models()
        if models:
            for model in models:
                print(f"   • {model}")
        else:
            print("   No models found or Ollama not running")
        return
    
    if args.pull:
        if not args.model:
            print("🤖 Pulling default models...")
            ollama_manager.pull_models()
        else:
            print(f"🤖 Pulling model: {args.model}")
            try:
                result = UtilityManager.run_subprocess(
                    f"docker exec ollama ollama pull {args.model}",
                    check=False,
                    timeout=600
                )
                if result.returncode == 0:
                    print(f"✅ Successfully pulled {args.model}")
                else:
                    print(f"❌ Failed to pull {args.model}: {result.stderr}")
            except Exception as e:
                print(f"❌ Error pulling model: {e}")
        return
    
    # Start Ollama service
    mode_info = "CPU-only mode" if args.cpu else "GPU-accelerated mode"
    print(f"🚀 Starting Ollama Service ({mode_info})...")
    print("=" * 50)
    
    # Check system requirements
    if not UtilityManager.check_system_requirements(
        model_name="Ollama Service",
        model_description="Standalone Ollama API server",
        vram_requirement="8GB+ RAM (CPU mode)" if args.cpu else "8GB+ VRAM (GPU mode)"
    ):
        return False
    
    # Ensure the external network exists
    UtilityManager.ensure_docker_network("ai_network")
    
    # Start Ollama
    print("🐳 Starting Ollama container...")
    cmd = UtilityManager.build_compose_command(
        compose_files=compose_files,
        project_name="ollama-standalone",
        action="up",
        additional_args="-d"
    )
    
    result = UtilityManager.run_subprocess(cmd, check=False)
    if result.returncode != 0:
        print(f"❌ Failed to start Ollama: {result.stderr}")
        return False
    
    print("✅ Ollama container started")
    print("⏳ Waiting for Ollama API to be ready...")
    
    # Wait for Ollama to be ready
    if not ollama_manager.wait_for_api(retries=24):  # 2 minutes
        print("❌ Ollama API failed to start")
        return False
    
    print("✅ Ollama API is ready")
    print()
    print("🎉 Ollama Service is running!")
    print()
    print("🌐 Ollama API: http://localhost:11434")
    print()
    print("💡 Next steps:")
    print("   1. Deploy custom models: python start_custom_assistant.py templates/your_template.json")
    print("   2. Start WebUI separately: python start_webui.py")
    print("   3. Or pull base models: python start_ollama.py --pull")
    
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