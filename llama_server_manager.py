import os
import sys
import time
import requests


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
