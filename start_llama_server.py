import os
import argparse
import sys

# Import the separated classes
from utility_manager import UtilityManager
from llama_server_manager import LlamaServerManager

def main(cleanup=False, quantization=None, auto_select=False, cpu_only=False):
    # Select quantization if not provided via command line
    if not quantization and not auto_select:
        quantization = LlamaServerManager.select_quantization(cpu_only=cpu_only)
    elif not quantization:
        quantization = "Q2_K" if cpu_only else "Q4_K_M"  # Default for auto-select
    
    # Create llama server manager with selected quantization and CPU mode
    llama_manager = LlamaServerManager(selected_quant=quantization, cpu_only=cpu_only)
    
    # Display selected model info
    llama_config = llama_manager.config
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
    
    # Ensure model exists for llama-server
    llama_manager.ensure_model_exists()

    # Set environment variable for docker-compose
    os.environ['LLAMA_MODEL_FILE'] = llama_config['model_file']

    # Build compose command - just llama server with GPU override if needed
    base_files = ["docker-compose.llama.yml"]
    if not cpu_only:
        base_files.append("docker-compose.llama-gpu-override.yml")  # Add GPU acceleration for llama-server
    
    compose_cmd = "docker compose " + " ".join([f"-f {f}" for f in base_files])
    print(f"🚀 Starting llama.cpp server ({'CPU mode' if cpu_only else 'GPU mode'})...")
    print(f"   Using: {' + '.join(base_files)}")
    
    UtilityManager.run_subprocess(f"{compose_cmd} up -d", show_output=True)

    # Restart container to ensure it's fresh
    container_names = ["llama-server"]
    UtilityManager.restart_containers(container_names)

    # Wait for llama server to be ready with enhanced progress monitoring
    llama_ready = llama_manager.wait_for_api_with_progress()
    
    # Test completions if llama-server is ready
    if llama_ready:
        llama_manager.test_completion()
        print(f"\n✅ llama.cpp server is running at http://localhost:11435")
        print(f"📖 API documentation: http://localhost:11435/docs")

    if not cleanup:
        print(f"\n💡 The llama.cpp server is now available for API calls")
        print(f"   Example: curl -X POST http://localhost:11435/v1/completions \\")
        print(f"     -H 'Content-Type: application/json' \\")
        print(f"     -d '{{\"prompt\": \"Hello\", \"max_tokens\": 50}}'")
        input("\n⏸️  Press Enter to continue...")

    if cleanup:
        print("🧹 Cleaning up...")
        cleanup_cmd = "docker compose " + " ".join([f"-f {f}" for f in base_files])
        UtilityManager.run_subprocess(f"{cleanup_cmd} down", show_output=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deploy DeepSeek Coder V2 Lite with llama.cpp server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_llama_server.py                    # Interactive quantization selection (GPU mode)
  python start_llama_server.py --auto             # Use default Q4_K_M (GPU mode)
  python start_llama_server.py --cpu-only         # CPU-only mode with interactive selection
  python start_llama_server.py --cpu-only --auto  # CPU-only mode with default Q2_K
  python start_llama_server.py -q Q6_K            # Use specific quantization (GPU mode)
  python start_llama_server.py --cpu-only -q Q3_K_M # CPU mode with specific quantization
  python start_llama_server.py --list-models      # Show existing models
  python start_llama_server.py --list-quants      # Show available quantizations
  python start_llama_server.py --cleanup          # Stop containers after test

Architecture:
  Base files: docker-compose.llama.yml (CPU-optimized)
  GPU mode: + docker-compose.llama-gpu-override.yml (adds GPU acceleration)

API Endpoint:
  Server: http://localhost:11435
  OpenAI-compatible API: /v1/completions, /v1/chat/completions

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
        LlamaServerManager.list_quantizations()
        sys.exit(0)
        
    if args.list_models:
        # Create a temporary manager to list models properly
        temp_manager = LlamaServerManager()
        temp_manager.list_existing_models()
        sys.exit(0)
    
    main(cleanup=args.cleanup, quantization=args.quantization, auto_select=args.auto, cpu_only=args.cpu_only)
