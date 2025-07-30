import time
import requests
import subprocess
import docker


class OllamaManager:
    def __init__(self):
        # Ollama configuration - Optimized for RTX 3060 Ti (8GB VRAM)
        self.config = {
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
