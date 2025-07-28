import os
import subprocess
import time
import requests
import docker
import argparse
import sys
import webbrowser

# Configuration for different services
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

config = ServiceConfig()

class UtilityManager:
    """Utility functions for the application"""
    
    @staticmethod
    def run_subprocess(cmd, check=True, show_output=False):
        if show_output:
            # Show output in real-time for commands like docker compose
            print(f"🔧 Running: {cmd}")
            result = subprocess.run(cmd, shell=True, text=True, encoding='utf-8', errors='ignore')
            if check and result.returncode != 0:
                print(f"❌ Command failed: {cmd}")
                sys.exit(1)
            return ""
        else:
            # Capture output for other commands
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            if check and result.returncode != 0:
                print(f"❌ Command failed: {cmd}")
                print(result.stderr.strip())
                sys.exit(1)
            return result.stdout.strip()

class LlamaServerManager:
    def __init__(self, config):
        self.config = config.llama_server
        self.model_dir = config.model_dir+"/.llama"
        self.model_path = os.path.join(self.model_dir, self.config["model_file"])
        self.legacy_path = os.path.join(self.model_dir, "model.gguf")

    def list_existing_models(self):
        """List existing models in the models directory"""
        # Check both the main models directory and the llama-specific subdirectory
        main_models_dir = os.path.dirname(self.model_dir)  # Get parent directory (models/)
        
        all_models = []
        
        # Check main models directory
        if os.path.exists(main_models_dir):
            gguf_files = [f for f in os.listdir(main_models_dir) if f.endswith('.gguf')]
            for file in gguf_files:
                file_path = os.path.join(main_models_dir, file)
                all_models.append((file, file_path, "main"))
        
        # Check llama-specific directory
        if os.path.exists(self.model_dir):
            gguf_files = [f for f in os.listdir(self.model_dir) if f.endswith('.gguf')]
            for file in gguf_files:
                file_path = os.path.join(self.model_dir, file)
                all_models.append((file, file_path, "llama"))
        
        if not all_models:
            print("📁 No GGUF models found in models directories")
            return
        
        print(f"\n📁 Existing models:")
        print("=" * 70)
        
        for file, file_path, location in all_models:
            file_size = os.path.getsize(file_path) / (1024**3)  # GB
            location_text = f" (in {location})" if location == "llama" else ""
            
            # Check if it's a DeepSeek model and extract quantization
            if 'DeepSeek-Coder-V2-Lite' in file:
                # Extract quantization from filename
                for quant in ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_K_S", "IQ4_XS", "Q3_K_L", "Q3_K_M", "Q2_K"]:
                    if quant in file:
                        print(f"✅ {file} ({file_size:.1f}GB) - {quant}{location_text}")
                        break
                else:
                    print(f"❓ {file} ({file_size:.1f}GB) - Unknown quantization{location_text}")
            else:
                print(f"📄 {file} ({file_size:.1f}GB) - Other model{location_text}")
        print()

    def get_actual_model_file(self):
        """Get the actual model file that exists in the models directory"""
        if os.path.exists(self.model_path):
            return os.path.basename(self.model_path)
        
        if os.path.exists(self.model_dir):
            gguf_files = [f for f in os.listdir(self.model_dir) if f.endswith('.gguf')]
            if gguf_files:
                self.model_path = os.path.join(self.model_dir, gguf_files[0])
                return gguf_files[0]
        
        return self.config["model_file"]
    
    def ensure_model_exists(self):
        """Ensure a model file exists for llama-server"""
        if os.path.exists(self.model_path):
            print(f"✅ Model file found: {self.config['model_file']} ({self.config['size']})")
            return
            
        # Check for legacy model.gguf
        if os.path.exists(self.legacy_path):
            print(f"🛠 Renaming legacy model.gguf to {self.config['model_file']}")
            os.rename(self.legacy_path, self.model_path)
            return

        # Check for existing GGUF files
        if os.path.exists(self.model_dir):
            existing_files = [f for f in os.listdir(self.model_dir) if f.endswith('.gguf')]
            
            # Check if we already have the exact model we want
            if self.config['model_file'] in existing_files:
                print(f"✅ Required model already exists: {self.config['model_file']}")
                return
            
            # Check for other DeepSeek Coder V2 Lite models
            deepseek_files = [f for f in existing_files if 'DeepSeek-Coder-V2-Lite' in f]
            
            if deepseek_files:
                print(f"📁 Found existing DeepSeek Coder V2 Lite models:")
                for i, file in enumerate(deepseek_files, 1):
                    file_size = os.path.getsize(os.path.join(self.model_dir, file)) / (1024**3)  # GB
                    print(f"   {i}. {file} (~{file_size:.1f}GB)")
                
                print(f"\n🎯 You selected: {self.config['model_file']} ({self.config['size']})")
                
                choice = input(f"❓ Use existing model (1-{len(deepseek_files)}) or download new? (1-{len(deepseek_files)}/d/n): ").strip().lower()
                
                if choice == 'd':
                    pass  # Download new model
                elif choice == 'n':
                    print("❌ Cannot continue without a model.")
                    sys.exit(1)
                else:
                    try:
                        choice_num = int(choice)
                        if 1 <= choice_num <= len(deepseek_files):
                            selected_file = deepseek_files[choice_num - 1]
                            self.model_path = os.path.join(self.model_dir, selected_file)
                            print(f"✅ Using existing model: {selected_file}")
                            return
                    except ValueError:
                        pass
            
            elif existing_files:
                print(f"📁 Found existing GGUF files (not DeepSeek Coder V2 Lite): {', '.join(existing_files)}")
                choice = input(f"❓ Use one of these files or download DeepSeek Coder V2 Lite? (use/download): ").strip().lower()
                if choice == "use" and len(existing_files) == 1:
                    existing_file = os.path.join(self.model_dir, existing_files[0])
                    print(f"✅ Using existing model: {existing_file}")
                    self.model_path = existing_file
                    return

        # No suitable model found, offer to download
        print(f"❗ Model not found: {self.config['model_file']}")
        print(f"   📦 Quantization: {self.config['quantization']}")
        print(f"   💾 Size: {self.config['size']}")
        print(f"   📝 {self.config['description']}")
        
        os.makedirs(self.model_dir, exist_ok=True)
        
        choice = input(f"❓ Download {self.config['model_file']} ({self.config['size']})? (y/n): ").strip().lower()
        if choice == "y":
            self._download_model()
        else:
            print("❌ Cannot continue without a GGUF model.")
            sys.exit(1)
    
    def _download_model(self):
        """Download the model file with progress indication"""
        print(f"⬇️ Downloading {self.config['model_file']} ({self.config['size']})...")
        print(f"   🔗 From: {self.config['download_url']}")
        
        try:
            with requests.get(self.config["download_url"], stream=True) as r:
                r.raise_for_status()
                
                # Get total file size
                total_size = int(r.headers.get('content-length', 0))
                downloaded = 0
                
                with open(self.model_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            # Show progress every 100MB
                            if downloaded % (100 * 1024 * 1024) == 0:
                                if total_size > 0:
                                    progress = (downloaded / total_size) * 100
                                    print(f"   📊 Progress: {downloaded / (1024**3):.1f}GB / {total_size / (1024**3):.1f}GB ({progress:.1f}%)")
                                else:
                                    print(f"   📊 Downloaded: {downloaded / (1024**3):.1f}GB")
                
                final_size = os.path.getsize(self.model_path) / (1024**3)
                print(f"✅ Download completed! Final size: {final_size:.1f}GB")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Download failed: {e}")
            print("💡 You can manually download the model file and place it in the models/ directory")
            print(f"💡 Expected filename: {self.config['model_file']}")
            sys.exit(1)
    
    def wait_for_api(self, retries=30):
        """Wait for llama-server API to be ready"""
        print(f"⏳ Waiting for {self.config['name']} API...")
        for _ in range(retries):
            try:
                r = requests.get(f"{self.config['url']}/v1/models", timeout=2)
                if r.status_code == 200:
                    data = r.json()
                    if "data" in data and len(data["data"]) > 0:
                        # Handle different response structures
                        model_info = data['data'][0]
                        model_name = model_info.get('id', model_info.get('name', 'Unknown'))
                        print(f"✅ {self.config['name']} is ready! Model: {model_name}")
                        return True
            except requests.RequestException:
                pass
            time.sleep(1)
        print(f"❌ {self.config['name']} not responding or no model loaded.")
        return False
    
    def test_completion(self, timeout=120):
        """Test the llama-server completion endpoint"""
        actual_model = self.get_actual_model_file()
        print(f"🧪 Testing llama.cpp endpoint with {actual_model}...")
        
        # Wait for completion endpoint to be ready (not just the models endpoint)
        print("⏳ Waiting for completion endpoint to be ready...")
        
        headers = {"Content-Type": "application/json", "Authorization": "Bearer fake"}
        payload = {
            "model": actual_model,
            "messages": [{"role": "user", "content": "Hello, who are you? What model are you?"}]
        }
        
        # Try multiple times with increasing delays
        for attempt in range(6):  # 6 attempts with exponential backoff
            try:
                wait_time = 2 ** attempt  # 1, 2, 4, 8, 16, 32 seconds
                if attempt > 0:
                    print(f"⏳ Retrying in {wait_time} seconds... (attempt {attempt + 1}/6)")
                    time.sleep(wait_time)
                
                r = requests.post(f"{self.config['url']}/v1/chat/completions", headers=headers, json=payload, timeout=timeout)
                
                if r.status_code == 503:
                    if attempt < 5:  # Not the last attempt
                        print(f"⚠️ Server unavailable (503), will retry...")
                        continue
                    else:
                        print("⚠️ Server is temporarily unavailable (503) - model may still be loading")
                        print("💡 Try again in a few moments, or check container logs with: docker logs llama-server")
                        return
                
                r.raise_for_status()
                data = r.json()
                if "choices" in data:
                    print("✅ Inference succeeded:")
                    print(data["choices"][0]["message"]["content"])
                    return  # Success, exit the retry loop
                else:
                    print("⚠️ 'choices' not found in response:")
                    print(data)
                    return
                    
            except requests.exceptions.Timeout:
                print(f"⏰ Request timed out after {timeout}s - model may be processing")
                if attempt < 5:
                    continue
                else:
                    return
            except Exception as e:
                if attempt < 5:
                    print(f"❌ API call failed: {e}, retrying...")
                    continue
                else:
                    print(f"❌ Failed API call: {e}")
                    return

class OllamaManager:
    def __init__(self, config):
        self.config = config.ollama
        
    def wait_for_api(self, retries=30):
        """Wait for Ollama API to be ready"""
        print(f"⏳ Waiting for {self.config['name']} API...")
        for _ in range(retries):
            try:
                r = requests.get(f"{self.config['url']}/api/tags", timeout=2)
                if r.status_code == 200:
                    print(f"✅ {self.config['name']} is ready!")
                    return True
            except requests.RequestException:
                pass
            time.sleep(1)
        print(f"❌ {self.config['name']} not responding.")
        return False
    
    def pull_models(self):
        """Pull models into Ollama"""
        print("🤖 Pulling Ollama models...")
        
        try:
            client = docker.from_env()
            ollama_container = client.containers.get(self.config["name"])
            if ollama_container.status != "running":
                print("❌ Ollama container is not running")
                return
        except docker.errors.NotFound:
            print("❌ Ollama container not found")
            return
        
        for model in self.config["models"]:
            print(f"⬇️ Pulling {model}...")
            try:
                cmd = f'powershell -Command "docker exec {self.config["name"]} ollama pull {model}"'
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, 
                                      encoding='utf-8', errors='ignore', timeout=300)
                
                if result.returncode == 0:
                    print(f"✅ {model} successfully pulled")
                else:
                    check_cmd = f'powershell -Command "docker exec {self.config["name"]} ollama list"'
                    check_result = subprocess.run(check_cmd, shell=True, capture_output=True, 
                                                text=True, encoding='utf-8', errors='ignore')
                    if model.split(':')[0] in check_result.stdout:
                        print(f"✅ {model} already exists")
                    else:
                        print(f"⚠️ {model} pull failed (exit code: {result.returncode})")
                        
            except subprocess.TimeoutExpired:
                print(f"⏰ {model} pull timed out (300s)")
            except Exception as e:
                print(f"❌ Failed to pull {model}: {e}")
        
        print("✅ Ollama model pulling completed")

class WebUIManager:
    def __init__(self, config):
        self.config = config.webui
        
    def wait_for_api(self, retries=30):
        """Wait for WebUI to be ready"""
        print(f"⏳ Waiting for {self.config['name']}...")
        for _ in range(retries):
            try:
                r = requests.get(self.config["url"], timeout=2)
                if r.status_code == 200:
                    print(f"✅ {self.config['name']} is ready!")
                    return True
            except requests.RequestException:
                pass
            time.sleep(1)
        print(f"❌ {self.config['name']} not responding.")
        return False

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
