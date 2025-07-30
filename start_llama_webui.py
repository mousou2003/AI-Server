import os
import docker
import argparse
import sys
import webbrowser

# Import the separated classes
from service_config import ServiceConfig
from utility_manager import UtilityManager
from llama_server_manager import LlamaServerManager
from ollama_manager import OllamaManager
from webui_manager import WebUIManager

def main(cleanup=False, quantization=None, auto_select=False, cpu_only=False):
    # Select quantization if not provided via command line
    if not quantization and not auto_select:
        quantization = ServiceConfig.select_quantization(cpu_only=cpu_only)
    elif not quantization:
        quantization = "Q2_K" if cpu_only else "Q4_K_M"  # Default for auto-select
    
    # Create configuration with selected quantization and CPU mode
    config = ServiceConfig(selected_quant=quantization, cpu_only=cpu_only)
    
    # Display selected model info
    llama_config = config.llama_server
    mode_text = "CPU Mode" if cpu_only else "GPU Mode"
    print(f"\n🎯 Using DeepSeek Coder V2 Lite - {llama_config['quantization']} ({mode_text})")
    print(f"   📦 Model: {llama_config['model_file']}")
    print(f"   💾 Size: {llama_config['size']}")
    print(f"   📝 {llama_config['description']}")
    if cpu_only:
        print(f"   🖥️  Mode: CPU-only (no GPU acceleration)")
    else:
        print(f"   🎮 Mode: GPU-accelerated")
    print()
    
    # Initialize service managers
    llama_manager = LlamaServerManager(config)
    ollama_manager = OllamaManager(config)
    webui_manager = WebUIManager(config)
    
    # Ensure model exists for llama-server
    llama_manager.ensure_model_exists()

    # Set environment variable for docker-compose
    os.environ['LLAMA_MODEL_FILE'] = llama_config['model_file']

    compose_file = "docker-compose.cpu.yml" if cpu_only else "docker-compose.yml"
    print(f"🚀 Starting llama.cpp + Open WebUI stack ({'CPU mode' if cpu_only else 'GPU mode'})...")
    UtilityManager.run_subprocess(f"docker compose -f {compose_file} up -d", show_output=True)

    client = docker.from_env()
    for name in ["llama-server", "ollama", "open-webui"]:
        try:
            client.containers.get(name).restart()
            print(f"🔄 Restarted container: {name}")
        except docker.errors.NotFound:
            print(f"⚠️ Container not found: {name}")

    # Wait for services to be ready
    webui_manager.wait_for_api()
    llama_ready = llama_manager.wait_for_api()
    ollama_ready = ollama_manager.wait_for_api()
    
    # Pull Ollama models after services are ready
    if ollama_ready:
        ollama_manager.pull_models()
    
    # Test completions if llama-server is ready
    if llama_ready:
        llama_manager.test_completion()

    if not cleanup:
        if input("🌐 Open WebUI in browser now? (y/n): ").strip().lower() == "y":
            webbrowser.open("http://localhost:3000")

    if cleanup:
        print("🧹 Cleaning up...")
        UtilityManager.run_subprocess(f"docker compose -f {compose_file} down", show_output=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deploy DeepSeek Coder V2 Lite with llama.cpp and Open WebUI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_llama_webui.py                    # Interactive quantization selection (GPU mode)
  python start_llama_webui.py --auto             # Use default Q4_K_M (GPU mode)
  python start_llama_webui.py --cpu-only         # CPU-only mode with interactive selection
  python start_llama_webui.py --cpu-only --auto  # CPU-only mode with default Q2_K
  python start_llama_webui.py -q Q6_K            # Use specific quantization (GPU mode)
  python start_llama_webui.py --cpu-only -q Q3_K_M # CPU mode with specific quantization
  python start_llama_webui.py --list-models      # Show existing models
  python start_llama_webui.py --list-quants      # Show available quantizations
  python start_llama_webui.py --cleanup          # Stop containers after test

Quantization recommendations:
  GPU Mode:
  • Q4_K_M or higher for best code quality
  • Q4_K_M default for balanced performance (8GB+ VRAM)
  • Q3_K_L or Q2_K for limited VRAM (4-6GB)
  • Q6_K or Q8_0 for maximum quality (16GB+ VRAM)

  CPU Mode:
  • Q2_K recommended for CPU-only mode (8GB+ RAM)
  • Q3_K_M for better quality but slower (12GB+ RAM)
  • Q4_K_M for best quality but much slower (16GB+ RAM)
        """)
    
    parser.add_argument("--cleanup", action="store_true", 
                       help="Stop containers after test")
    
    parser.add_argument("--quantization", "-q", 
                       choices=["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_K_S", "IQ4_XS", "Q3_K_L", "Q3_K_M", "Q2_K"], 
                       help="Select quantization level directly")
    
    parser.add_argument("--auto", action="store_true", 
                       help="Use default quantization (Q4_K_M for GPU, Q2_K for CPU) without prompting")
    
    parser.add_argument("--cpu-only", action="store_true",
                       help="Run in CPU-only mode (no GPU acceleration required)")
    
    parser.add_argument("--list-quants", action="store_true",
                       help="List all available quantization options and exit")
    
    parser.add_argument("--list-models", action="store_true",
                       help="List existing models in models directory and exit")
    
    args = parser.parse_args()
    
    # Handle list commands
    if args.list_quants:
        ServiceConfig.list_quantizations()
        sys.exit(0)
        
    if args.list_models:
        # Create a temporary config and manager to list models properly
        temp_config = ServiceConfig()
        temp_manager = LlamaServerManager(temp_config)
        temp_manager.list_existing_models()
        sys.exit(0)
    
    main(cleanup=args.cleanup, quantization=args.quantization, auto_select=args.auto, cpu_only=args.cpu_only)
