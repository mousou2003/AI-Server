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
    def __init__(self):
        self.model_dir = os.path.join(os.getcwd(), "models")
        self.inference_timeout = 60
        
        # Llama.cpp server configuration
        # Using DeepSeek Coder V2 Lite - excellent coding model with good balance of performance/size
        # V2 Lite: 15.7B params (~10GB) vs V3: 411B params (155GB+) - V2 is much more practical
        # 
        # IMPORTANT: DeepSeek Coder models are specialized for CODE GENERATION, not agent/tool use
        # - Original 6.7B model had NO agent/function calling capabilities
        # - V2 Lite has improved instruction following but still primarily for coding tasks
        # - For true agent capabilities, consider models like Claude, GPT-4, or agent-specific models
        # 
        # Available quantization options (quality vs size trade-off):
        # Q8_0: 16.7GB - Extremely high quality, generally unneeded but max available quant
        # Q6_K: 14.1GB - Very high quality, near perfect, recommended for high-end setups
        # Q5_K_M: 11.9GB - High quality, recommended for good balance
        # Q4_K_M: 10.4GB - Good quality, recommended (CURRENT CHOICE) - best balance
        # Q4_K_S: 9.53GB - Slightly lower quality with more space savings
        # IQ4_XS: 8.57GB - Decent quality, smaller than Q4_K_S with similar performance
        # Q3_K_L: 8.45GB - Lower quality but usable, good for low RAM availability
        # Q3_K_M: 8.12GB - Even lower quality
        # Q2_K: 6.43GB - Very low quality but surprisingly usable
        # 
        # For CODE GENERATION: Q4_K_M+ recommended for best code quality and completion
        # Q4_K_M chosen for optimal balance of code quality vs file size (fits most 12GB+ GPUs)
        self.llama_server = {
            "name": "llama-server",
            "port": 11435,
            "url": "http://localhost:11435",
            "model_file": "DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf",  # 10.36GB - fits most GPUs
            "download_url": "https://huggingface.co/bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF/resolve/main/DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf"
        }
        
        # Ollama configuration
        self.ollama = {
            "name": "ollama",
            "port": 11434,
            "url": "http://localhost:11434",
            "models": [
                "mistral:7b",     # General purpose chat
                "wizardcoder",    # Premium coding model (no :13b tag)
                "phi3:mini",      # Lightweight & fast
                "gemma:7b",       # Google's model
            ]
        }
        
        # Open WebUI configuration
        self.webui = {
            "name": "open-webui",
            "port": 443,
            "url": "https://localhost"
        }

config = ServiceConfig()

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
        self.model_dir = config.model_dir
        self.model_path = os.path.join(self.model_dir, self.config["model_file"])
        self.legacy_path = os.path.join(self.model_dir, "model.gguf")
        
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
            print(f"✅ Model file found: {self.model_path}")
            return
            
        if os.path.exists(self.legacy_path):
            print(f"🛠 Renaming legacy model.gguf to {self.config['model_file']}")
            os.rename(self.legacy_path, self.model_path)
            return

        if os.path.exists(self.model_dir):
            existing_files = [f for f in os.listdir(self.model_dir) if f.endswith('.gguf')]
            if existing_files:
                print(f"📁 Found existing GGUF files: {', '.join(existing_files)}")
                if len(existing_files) == 1:
                    existing_file = os.path.join(self.model_dir, existing_files[0])
                    choice = input(f"❓ Use existing file '{existing_files[0]}' as the model? (y/n): ").strip().lower()
                    if choice == "y":
                        print(f"✅ Using existing model: {existing_file}")
                        self.model_path = existing_file
                        return

        print(f"❗ No suitable model file found in {self.model_dir}")
        os.makedirs(self.model_dir, exist_ok=True)
        choice = input(f"❓ Do you want to download {self.config['model_file']} (~4GB)? (y/n): ").strip().lower()
        if choice == "y":
            self._download_model()
        else:
            print("❌ Cannot continue without a GGUF model.")
            sys.exit(1)
    
    def _download_model(self):
        """Download the model file"""
        print("⬇️ Downloading model...")
        try:
            with requests.get(self.config["download_url"], stream=True) as r:
                r.raise_for_status()
                with open(self.model_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print("✅ Model downloaded.")
        except requests.exceptions.RequestException as e:
            print(f"❌ Download failed: {e}")
            print("💡 You can manually place a GGUF model file in the models/ directory")
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

def main(cleanup=False):
    # Create configuration
    config = ServiceConfig()
    
    # Initialize service managers
    llama_manager = LlamaServerManager(config)
    ollama_manager = OllamaManager(config)
    webui_manager = WebUIManager(config)
    
    # Ensure model exists for llama-server
    llama_manager.ensure_model_exists()

    print("🚀 Starting llama.cpp + Open WebUI stack...")
    run_subprocess("docker compose up -d", show_output=True)

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
            webbrowser.open("https://localhost")

    if cleanup:
        print("🧹 Cleaning up...")
        run_subprocess("docker compose down", show_output=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleanup", action="store_true", help="Stop containers after test")
    args = parser.parse_args()
    main(cleanup=args.cleanup)
