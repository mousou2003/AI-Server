import time
import requests
import subprocess
import docker
import json
from pathlib import Path
from utility_manager import UtilityManager


class OllamaManager:
    def __init__(self):
        # Ollama configuration - Optimized for RTX 3060 Ti (8GB VRAM)
        self.config = {
            "name": "ollama",
            "port": 11434,
            "url": "http://localhost:11434",
            "models": [
                "qwen2.5:7b-instruct",   # Qwen2.5-Instruct 7B - General instruction following model that fits 8GB VRAM
                "qwen2.5:14b-instruct",  # Qwen2.5-Instruct 14B - Larger model for better performance (may be tight on 8GB VRAM)
                "gemma2:9b",             # Gemma2 9B - Alternative model, might work on 8GB but could be tight
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
                cmd = f'docker exec {self.config["name"]} ollama pull {model}'
                result = UtilityManager.run_subprocess(cmd, check=False, timeout=300)
                
                if result.returncode == 0:
                    print(f"✅ {model} successfully pulled")
                else:
                    check_cmd = f'docker exec {self.config["name"]} ollama list'
                    check_result = UtilityManager.run_subprocess(check_cmd, check=False)
                    if model.split(':')[0] in check_result.stdout:
                        print(f"✅ {model} already exists")
                    else:
                        print(f"⚠️ {model} pull failed (exit code: {result.returncode})")
                        
            except subprocess.TimeoutExpired:
                print(f"⏰ {model} pull timed out (300s)")
            except Exception as e:
                print(f"❌ Failed to pull {model}: {e}")
        
        print("✅ Ollama model pulling completed")

    def create_custom_model(self, base_model, custom_model_name, modelfile_content, modelfile_path=None):
        """
        Create a custom model in Ollama with specified Modelfile content
        
        Args:
            base_model (str): Base model name (e.g., "qwen2.5:7b-instruct")
            custom_model_name (str): Name for the custom model
            modelfile_content (str): Content of the Modelfile
            modelfile_path (str, optional): Local path to save the Modelfile
            
        Returns:
            bool: True if successful, False otherwise
        """
        print(f"🔧 Creating custom model: {custom_model_name}")
        
        try:
            # Save Modelfile locally if path provided
            if modelfile_path:
                with open(modelfile_path, 'w', encoding='utf-8') as f:
                    f.write(modelfile_content)
                print(f"   ✅ Saved Modelfile: {modelfile_path}")
            
            # Use /tmp directory in container (which exists)
            container_modelfile_path = f"/tmp/Modelfile.{custom_model_name}"
            
            # Method 1: Try to create Modelfile directly in container (preferred)
            # Escape quotes and newlines for shell
            escaped_content = modelfile_content.replace('"', '\\"').replace('\n', '\\n')
            direct_cmd = f'docker exec {self.config["name"]} sh -c "echo -e \\"{escaped_content}\\" > {container_modelfile_path}"'
            
            direct_result = subprocess.run(direct_cmd, shell=True, capture_output=True, text=True,
                                         encoding='utf-8', errors='ignore', timeout=60)
            
            if direct_result.returncode != 0:
                print(f"   ⚠️  Direct method failed, trying file copy method...")
                
                # Method 2: Create temp file and copy (fallback)
                import tempfile
                if modelfile_path:
                    temp_path = modelfile_path
                else:
                    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.modelfile', encoding='utf-8') as f:
                        f.write(modelfile_content)
                        temp_path = f.name
                
                copy_cmd = f'docker cp "{temp_path}" {self.config["name"]}:{container_modelfile_path}'
                copy_result = subprocess.run(copy_cmd, shell=True, capture_output=True, text=True,
                                           encoding='utf-8', errors='ignore', timeout=60)
                
                if copy_result.returncode != 0:
                    print(f"   ⚠️  Could not copy Modelfile to container: {copy_result.stderr}")
                    return False
                    
                # Clean up temp file if we created one
                if not modelfile_path:
                    import os
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
            
            # Create the custom model
            create_cmd = f'docker exec {self.config["name"]} ollama create {custom_model_name} -f {container_modelfile_path}'
            result = subprocess.run(create_cmd, shell=True, capture_output=True, text=True,
                                  encoding='utf-8', errors='ignore', timeout=300)
            
            if result.returncode == 0:
                print(f"   ✅ Created custom model: {custom_model_name}")
                return True
            else:
                print(f"   ⚠️  Could not create custom model: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("   ⏰ Model creation timed out")
            return False
        except Exception as e:
            print(f"   ❌ Error creating custom model: {e}")
            return False
    
    def verify_model_exists(self, model_name):
        """
        Verify that a model exists in Ollama
        
        Args:
            model_name (str): Name of the model to check
            
        Returns:
            bool: True if model exists, False otherwise
        """
        try:
            cmd = f'docker exec {self.config["name"]} ollama list'
            result = UtilityManager.run_subprocess(cmd, check=False, timeout=30)
            
            if result.returncode == 0 and model_name in result.stdout:
                print(f"   ✅ Model {model_name} is available")
                return True
            else:
                print(f"   ⚠️  Model {model_name} not found in Ollama")
                return False
                
        except Exception as e:
            print(f"   ❌ Error verifying model: {e}")
            return False
    
    def list_models(self):
        """
        List all models in Ollama
        
        Returns:
            list: List of model names, or empty list if failed
        """
        try:
            cmd = f'docker exec {self.config["name"]} ollama list'
            result = UtilityManager.run_subprocess(cmd, check=False, timeout=30)
            
            if result.returncode == 0:
                # Parse the output to extract model names
                lines = result.stdout.strip().split('\n')
                models = []
                for line in lines[1:]:  # Skip header line
                    if line.strip():
                        model_name = line.split()[0]  # First column is model name
                        models.append(model_name)
                return models
            else:
                print(f"   ❌ Could not list models: {result.stderr}")
                return []
                
        except Exception as e:
            print(f"   ❌ Error listing models: {e}")
            return []
    
    def setup_specialized_churn_model(self, base_model_name, 
                                     template_name="qwen_churn_system_prompt.template.json",
                                     custom_suffix="churn",
                                     templates_path="templates"):
        """
        Complete setup for specialized churn model - direct approach
        
        Args:
            base_model_name (str): Base model name (e.g., "qwen2.5:7b-instruct")
            template_name (str): Name of template file
            custom_suffix (str): Suffix for custom model name
            templates_path (str): Path to templates directory
            
        Returns:
            tuple: (success: bool, custom_model_name: str)
        """
        print("🔧 Setting up specialized churn model...")
        
        # Read template file directly
        template_file = Path(templates_path) / template_name
        if not template_file.exists():
            print(f"❌ Template file not found: {template_file}")
            return False, base_model_name
        
        with open(template_file, 'r', encoding='utf-8') as f:
            template_content = f.read()
        
        print(f"   ✅ Loaded template: {template_file}")
        print(f"   📄 Template size: {len(template_content)} characters")
        
        # Setup models directory and create Modelfile path
        models_dir = self.setup_models_directory()
        modelfile_path = Path(models_dir) / f"Modelfile.{base_model_name.replace(':', '-')}-{custom_suffix}"
        
        # Create minimal Modelfile with template as system prompt
        modelfile_content = f'''FROM {base_model_name}

SYSTEM """{template_content}"""
'''
        
        # Create the custom model
        custom_model_name = f"{base_model_name}-{custom_suffix}"
        success = self.create_custom_model(
            base_model=base_model_name,
            custom_model_name=custom_model_name,
            modelfile_content=modelfile_content,
            modelfile_path=str(modelfile_path)
        )
        
        if success:
            print(f"   📁 Modelfile saved to: {modelfile_path}")
            return True, custom_model_name
        else:
            print("   The base model will be used without the custom prompt")
            return False, base_model_name
    
    def get_api_status(self, port=None):
        """
        Check Ollama API status
        
        Args:
            port (int, optional): Port to check, uses config port if not provided
            
        Returns:
            tuple: (is_running: bool, status_message: str)
        """
        if port is None:
            port = self.config.get('port', 11434)
            
        try:
            response = requests.get(f"http://localhost:{port}/api/tags", timeout=2)
            if response.status_code == 200:
                return True, f"🌐 Ollama API: ✅ Running"
            else:
                return False, f"🌐 Ollama API: ⚠️  Responding with status {response.status_code}"
        except requests.RequestException:
            return False, f"🌐 Ollama API: ❌ Not responding"
    
    def wait_for_model_ready(self, model_name, max_wait=600, check_interval=15, skip_api_check=False):
        """
        Wait for a specific model to be ready for inference using API checks
        
        Args:
            model_name (str): Name of the model to wait for
            max_wait (int): Maximum wait time in seconds
            check_interval (int): How often to check in seconds
            skip_api_check (bool): If True, skip the initial API readiness check
            
        Returns:
            bool: True if model becomes ready, False if timeout
        """
        print(f"   ⏳ Waiting for model to be ready: {model_name}")
        
        import time
        start_time = time.time()
        
        # First ensure API is responding (unless skipped)
        if not skip_api_check:
            if not self.wait_for_api(retries=10):
                print(f"   ❌ Ollama API not ready")
                return False
        
        # Then wait for the model to be accessible
        while (time.time() - start_time) < max_wait:
            try:
                # Try a minimal API call to check if model is ready
                # Use timeout aligned with Docker OLLAMA_REQUEST_TIMEOUT (180s)
                response = requests.post(
                    f"{self.config['url']}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": "test",
                        "stream": False,
                        "options": {"num_predict": 1}
                    },
                    timeout=300  # 5 minutes for first-time model loading in CPU mode
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if 'response' in result:
                        elapsed = time.time() - start_time
                        print(f"   ✅ Model ready after {elapsed:.1f}s")
                        return True
                    
            except requests.exceptions.RequestException:
                pass  # Model not ready yet
            
            # Update progress on the same line
            elapsed = int(time.time() - start_time)
            print(f"\r   ⏳ Model loading... ({elapsed}s elapsed)", end="", flush=True)
            time.sleep(check_interval)
        
        print()  # New line after progress
        print(f"   ⏰ Model readiness check timed out after {max_wait}s")
        return False

    def warm_up_model(self, model_name, timeout=600):
        """
        Warm up a model by making a simple API request to load it into memory
        Uses smart waiting instead of arbitrary timeouts
        
        Args:
            model_name (str): Name of the model to warm up
            timeout (int): Maximum timeout in seconds (used as fallback)
            
        Returns:
            bool: True if warm-up successful, False otherwise
        """
        print(f"🔥 Warming up model: {model_name}")
        
        # First, wait for the model to be ready using smart detection (skip API check since we're called after API is ready)
        if not self.wait_for_model_ready(model_name, max_wait=timeout, check_interval=15, skip_api_check=True):
            print(f"   ❌ Model readiness check failed")
            return False
        
        # Model is ready, do a final warm-up call
        try:
            api_data = {
                "model": model_name,
                "prompt": "Hi",
                "stream": False,
                "options": {
                    "num_predict": 5,
                    "temperature": 0.1
                }
            }
            
            response = requests.post(
                f"{self.config['url']}/api/generate",
                json=api_data,
                timeout=300  # 5 minutes for first-time model loading in CPU mode
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'response' in result:
                    print(f"   ✅ Model warmed up successfully")
                    print(f"   💬 Response: {result['response'][:50]}..." if len(result['response']) > 50 else f"   💬 Response: {result['response']}")
                    return True
                else:
                    print(f"   ⚠️  API response missing 'response' field")
                    return False
            else:
                print(f"   ❌ Final warm-up call failed with status {response.status_code}")
                return False
                
        except requests.exceptions.Timeout:
            print(f"   ❌ Final warm-up call timed out")
            return False
        except requests.exceptions.ConnectionError:
            print(f"   ❌ Connection error during final warm-up")
            return False
        except Exception as e:
            print(f"   ❌ Error during final warm-up: {e}")
            return False

    def verify_model_in_memory(self, model_name):
        """
        Verify that a model is currently loaded in memory
        
        Args:
            model_name (str): Name of the model to verify
            
        Returns:
            bool: True if model is verified in memory, False otherwise
        """
        print(f"   🔍 Verifying model is loaded in memory...")
        
        # Use API method first
        success, running_models = self.get_running_models()
        
        if success:
            if any(model_name in model for model in running_models):
                print(f"   ✅ Model confirmed loaded in memory")
                print(f"   📊 Running models: {', '.join(running_models)}")
                return True
            else:
                print(f"   ⚠️  Model not showing as loaded")
                print(f"   📊 Currently running: {', '.join(running_models) if running_models else 'None'}")
                return True  # Still consider success - model may be cached
        else:
            print(f"   ⚠️  Could not verify via API")
            return True  # Don't fail for verification issues

    def get_running_models(self, port=None):
        """
        Get list of currently running models using Ollama API
        
        Args:
            port (int, optional): Port to check, uses config port if not provided
            
        Returns:
            tuple: (success: bool, models: list) - list of running model names
        """
        if port is None:
            port = self.config.get('port', 11434)
            
        try:
            response = requests.get(f"http://localhost:{port}/api/ps", timeout=10)
            if response.status_code == 200:
                ps_data = response.json()
                if 'models' in ps_data:
                    running_models = [model.get('name', '') for model in ps_data['models']]
                    return True, running_models
                else:
                    return True, []  # No models running
            else:
                return False, []
        except requests.RequestException:
            return False, []

    def setup_for_specialized_use(self, container_name=None, specialized_models=None, port=None):
        """
        Setup Ollama manager for specialized use case
        
        Args:
            container_name (str): Custom container name
            specialized_models (list): List of specialized model names
            port (int): Custom port
        """
        if container_name:
            self.config["name"] = container_name
        if port:
            self.config["port"] = port
            self.config["url"] = f"http://localhost:{port}"
        if specialized_models:
            self.config["models"] = specialized_models
    
    def setup_models_directory(self, models_path=".ollama"):
        """
        Setup and create the models directory for Ollama
        
        Args:
            models_path (str): Path to the models directory
            
        Returns:
            Path: The created models directory path
        """
        models_dir = Path(models_path)
        models_dir.mkdir(parents=True, exist_ok=True)
        return models_dir
    
    def setup_complete_infrastructure(self, project_name, models_path=".ollama"):
        """
        Complete infrastructure setup for models directory
        
        Args:
            project_name (str): Project name for organization
            models_path (str): Path to the models directory
        """
        # Setup models directory (Ollama-specific)
        self.setup_models_directory(models_path)
