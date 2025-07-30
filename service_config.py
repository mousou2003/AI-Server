import os
import sys


class ServiceConfig:
    def __init__(self, selected_quant=None, cpu_only=False):
        self.model_dir = os.path.join(os.getcwd(), "models")
        self.inference_timeout = 60
        self.cpu_only = cpu_only
        
        # Llama.cpp server configuration
        # Using DeepSeek Coder V2 Lite - excellent coding model with good balance of performance/size
        # V2 Lite: 15.7B params (~10GB) vs V3: 411B params (155GB+) - V2 is much more practical
        # 
        # IMPORTANT: DeepSeek Coder models are specialized for CODE GENERATION, not agent/tool use
        # - Original 6.7B model had NO agent/function calling capabilities
        # - V2 Lite has improved instruction following but still primarily for coding tasks
        # - For true agent capabilities, consider models like Claude, GPT-4, or agent-specific models
        
        # Available quantization options for DeepSeek Coder V2 Lite (quality vs size trade-off)
        self.quantization_options = {
            "Q8_0": {
                "size": "16.7GB",
                "description": "Extremely high quality, generally unneeded but max available quant",
                "recommendation": "High-end systems with 24GB+ VRAM" if not cpu_only else "High-end systems with 32GB+ RAM",
                "filename": "DeepSeek-Coder-V2-Lite-Instruct-Q8_0.gguf"
            },
            "Q6_K": {
                "size": "14.1GB", 
                "description": "Very high quality, near perfect",
                "recommendation": "High-end setups with 16GB+ VRAM" if not cpu_only else "High-end systems with 24GB+ RAM",
                "filename": "DeepSeek-Coder-V2-Lite-Instruct-Q6_K.gguf"
            },
            "Q5_K_M": {
                "size": "11.9GB",
                "description": "High quality, excellent balance",
                "recommendation": "Good balance for 12GB+ VRAM" if not cpu_only else "Good balance for 16GB+ RAM",
                "filename": "DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf"
            },
            "Q4_K_M": {
                "size": "10.4GB",
                "description": "Good quality, recommended default",
                "recommendation": "Best balance for most users (8GB+ VRAM)" if not cpu_only else "Best balance for CPU mode (16GB+ RAM)",
                "filename": "DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf"
            },
            "Q4_K_S": {
                "size": "9.53GB",
                "description": "Slightly lower quality with more space savings",
                "recommendation": "Space-conscious users with 8GB+ VRAM" if not cpu_only else "Space-conscious users with 12GB+ RAM",
                "filename": "DeepSeek-Coder-V2-Lite-Instruct-Q4_K_S.gguf"
            },
            "IQ4_XS": {
                "size": "8.57GB",
                "description": "Decent quality, smaller than Q4_K_S",
                "recommendation": "Limited VRAM (6-8GB)" if not cpu_only else "Limited RAM (12GB)",
                "filename": "DeepSeek-Coder-V2-Lite-Instruct-IQ4_XS.gguf"
            },
            "Q3_K_L": {
                "size": "8.45GB",
                "description": "Lower quality but usable",
                "recommendation": "Low VRAM availability (6GB)" if not cpu_only else "Low RAM availability (10GB)",
                "filename": "DeepSeek-Coder-V2-Lite-Instruct-Q3_K_L.gguf"
            },
            "Q3_K_M": {
                "size": "8.12GB",
                "description": "Even lower quality",
                "recommendation": "Very limited VRAM (4-6GB)" if not cpu_only else "Very limited RAM (8-10GB)",
                "filename": "DeepSeek-Coder-V2-Lite-Instruct-Q3_K_M.gguf"
            },
            "Q2_K": {
                "size": "6.43GB",
                "description": "Very low quality but surprisingly usable",
                "recommendation": "Minimal VRAM (4GB or less)" if not cpu_only else "Recommended for CPU mode (8GB+ RAM)",
                "filename": "DeepSeek-Coder-V2-Lite-Instruct-Q2_K.gguf"
            }
        }
        
        # Select quantization (default to Q4_K_M if not specified)
        self.selected_quantization = selected_quant or "Q4_K_M"
        selected_model = self.quantization_options[self.selected_quantization]
        
        self.llama_server = {
            "name": "llama-server",
            "port": 11435,
            "url": "http://localhost:11435",
            "model_file": selected_model["filename"],
            "download_url": f"https://huggingface.co/bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF/resolve/main/{selected_model['filename']}",
            "quantization": self.selected_quantization,
            "size": selected_model["size"],
            "description": selected_model["description"]
        }
        
        # Ollama configuration - Optimized for RTX 3060 Ti (8GB VRAM)
        self.ollama = {
            "name": "ollama",
            "port": 11434,
            "url": "http://localhost:11434",
            "models": [
                "qwen2.5-coder:7b",   # Qwen2.5-Coder 7B - Best coding model that fits 8GB VRAM
                "gemma2:9b",          # Experimental - might work on 8GB but could be tight
                # "qwen2.5-coder:32b",  # Too large for 8GB VRAM - disabled
                # "gemma2:27b",         # Too large for 8GB VRAM - disabled
            ]
        }
        
        # Open WebUI configuration
        self.webui = {
            "name": "open-webui",
            "port": 3000,
            "url": "http://localhost:3000"
        }

    @staticmethod
    def list_quantizations():
        """List all available quantization options"""
        print("\n🎯 Available DeepSeek Coder V2 Lite Quantizations:")
        print("=" * 80)
        
        print("🎮 GPU Mode Recommendations:")
        gpu_config = ServiceConfig(cpu_only=False)
        for quant, info in gpu_config.quantization_options.items():
            print(f"{quant:6s} - {info['size']:>7s} - {info['description']}")
            print(f"        💡 {info['recommendation']}")
        
        print("\n🖥️  CPU Mode Recommendations:")
        cpu_config = ServiceConfig(cpu_only=True)
        for quant, info in cpu_config.quantization_options.items():
            print(f"{quant:6s} - {info['size']:>7s} - {info['description']}")
            print(f"        💡 {info['recommendation']}")
        print()

    @staticmethod
    def select_quantization(cpu_only=False):
        """Interactive quantization selection for DeepSeek Coder V2 Lite"""
        mode_text = "CPU Mode" if cpu_only else "GPU Mode"
        print(f"\n🎯 DeepSeek Coder V2 Lite - Quantization Options ({mode_text})")
        print("=" * 80)
        if cpu_only:
            print("🖥️  CPU-only mode: Models will run entirely on CPU (slower but no GPU required)")
            print("💡 For CPU mode, consider Q2_K or Q3_K_M for better performance")
        else:
            print("🎮 GPU mode: Models will utilize GPU acceleration")
        print("Choose the quantization level based on your hardware capabilities:")
        print()
        
        # Create a temporary config to access quantization options
        temp_config = ServiceConfig(cpu_only=cpu_only)
        options = temp_config.quantization_options
        
        # Display options in a nice table format
        for i, (quant, info) in enumerate(options.items(), 1):
            print(f"{i:2d}. {quant:6s} - {info['size']:>7s} - {info['description']}")
            print(f"    💡 {info['recommendation']}")
            print()
        
        print("💡 Recommendations:")
        if cpu_only:
            print("   • For CPU mode: Q2_K or Q3_K_M (faster inference)")
            print("   • For better quality: Q4_K_M or Q4_K_S")
            print("   • Note: CPU inference is slower but works without GPU")
            print("   • Ensure sufficient RAM (model size + 2-4GB overhead)")
        else:
            print("   • For best code quality: Q4_K_M or higher (Q5_K_M, Q6_K)")
            print("   • For balanced performance: Q4_K_M (default)")
            print("   • For limited VRAM: Q3_K_L or Q2_K")
            print("   • For maximum quality: Q6_K or Q8_0")
        print()
        
        while True:
            try:
                default_quant = "Q2_K" if cpu_only else "Q4_K_M"
                choice = input(f"🔢 Select quantization (1-9, or press Enter for default {default_quant}): ").strip()
                
                if not choice:  # Default choice
                    return default_quant
                
                choice_num = int(choice)
                if 1 <= choice_num <= len(options):
                    selected_quant = list(options.keys())[choice_num - 1]
                    selected_info = options[selected_quant]
                    
                    print(f"\n✅ Selected: {selected_quant} ({selected_info['size']})")
                    print(f"   📝 {selected_info['description']}")
                    
                    # Confirm selection for larger models
                    if choice_num <= 3:  # Q8_0, Q6_K, Q5_K_M
                        confirm = input(f"⚠️  This is a large model ({selected_info['size']}). Continue? (y/n): ").strip().lower()
                        if confirm != 'y':
                            continue
                    
                    return selected_quant
                else:
                    print(f"❌ Invalid choice. Please select 1-{len(options)}")
                    
            except ValueError:
                print("❌ Invalid input. Please enter a number or press Enter for default.")
            except KeyboardInterrupt:
                print("\n\n❌ Selection cancelled.")
                sys.exit(1)
