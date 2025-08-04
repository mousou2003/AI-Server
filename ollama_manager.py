import time
import requests
import subprocess
import docker
from pathlib import Path


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

    def create_custom_model(self, base_model, custom_model_name, modelfile_content, modelfile_path=None):
        """
        Create a custom model in Ollama with specified Modelfile content
        
        Args:
            base_model (str): Base model name (e.g., "qwen2.5-coder:7b")
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
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                                  encoding='utf-8', errors='ignore', timeout=30)
            
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
            str: Output from ollama list command, or empty string if failed
        """
        try:
            cmd = f'docker exec {self.config["name"]} ollama list'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                                  encoding='utf-8', errors='ignore', timeout=30)
            
            if result.returncode == 0:
                return result.stdout
            else:
                print(f"   ❌ Could not list models: {result.stderr}")
                return ""
                
        except Exception as e:
            print(f"   ❌ Error listing models: {e}")
            return ""
    
    def load_system_prompt_from_template(self, templates_dir, template_name="qwen_churn_system_prompt.template.md"):
        """
        Load system prompt from template file
        
        Args:
            templates_dir (Path): Path to templates directory
            template_name (str): Name of template file
            
        Returns:
            str: System prompt content
        """
        prompt_file = Path(templates_dir) / template_name
        
        # Read the system prompt template file
        with open(prompt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract the prompt content (skip the markdown header)
        lines = content.split('\n')
        # Find the first line after the main title and start from there
        start_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('# ') and i == 0:
                continue
            elif line.strip() == '' and i <= 3:
                continue
            else:
                start_idx = i
                break
        
        prompt_content = '\n'.join(lines[start_idx:]).strip()
        print(f"   ✅ Loaded system prompt: {prompt_file}")
        return prompt_content
    
    def configure_specialized_model(self, base_model_name, system_prompt, models_dir, 
                                  custom_suffix="churn", temperature=0.3, top_k=40, 
                                  top_p=0.9, repeat_penalty=1.1):
        """
        Configure a specialized model with custom system prompt and parameters
        
        Args:
            base_model_name (str): Base model name (e.g., "qwen2.5-coder:7b")
            system_prompt (str): Custom system prompt
            models_dir (Path): Directory to save the Modelfile
            custom_suffix (str): Suffix for custom model name
            temperature (float): Model temperature setting
            top_k (int): Top-k parameter
            top_p (float): Top-p parameter
            repeat_penalty (float): Repeat penalty parameter
            
        Returns:
            tuple: (success: bool, custom_model_name: str)
        """
        print("🔧 Configuring specialized model with custom prompt...")
        
        # Save Modelfile in models directory for persistence
        modelfile_path = Path(models_dir) / f"Modelfile.{base_model_name.replace(':', '-')}-{custom_suffix}"
        
        # Create a Modelfile for Ollama with the system prompt
        modelfile_content = f'''FROM {base_model_name}

# Set parameters optimized for specialized use
PARAMETER temperature {temperature}
PARAMETER top_k {top_k}
PARAMETER top_p {top_p}
PARAMETER repeat_penalty {repeat_penalty}

# System prompt for specialized analysis
SYSTEM """{system_prompt}"""

# Template for consistent responses
TEMPLATE """{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
"""
'''
        
        # Create the customized model
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
    
    def setup_models_directory(self, models_path="models/.ollama"):
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
    
    def setup_complete_infrastructure(self, webui_manager, project_name="churn", 
                                    models_path="models/.ollama",
                                    workspace_path="workspace", 
                                    memory_path="memory", 
                                    templates_path="templates",
                                    datasets_path="workspace/churn_analysis"):
        """
        Complete infrastructure setup for models and WebUI directories
        
        Args:
            webui_manager: WebUIManager instance for directory operations
            project_name (str): Project name for subdirectories and memory files
            models_path (str): Path to the models directory
            workspace_path (str): Path to workspace directory
            memory_path (str): Path to memory directory
            templates_path (str): Path to templates directory
            datasets_path (str): Path to datasets directory
        """
        # Setup models directory (Ollama-specific)
        self.setup_models_directory(models_path)
        
        # Setup base directories
        workspace_dir = Path(workspace_path)
        memory_dir = Path(memory_path)
        templates_dir = Path(templates_path)
        datasets_dir = Path(datasets_path)
        
        # Create datasets directory
        datasets_dir.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created datasets directory: {datasets_dir}")
        
        # Create persistent directories using WebUIManager
        webui_manager.create_persistent_directories(
            workspace_dir=workspace_dir,
            memory_dir=memory_dir,
            templates_dir=templates_dir,
            project_name=project_name
        )
    
    def setup_specialized_churn_model(self, base_model_name, 
                                     template_name="qwen_churn_system_prompt.template.md",
                                     custom_suffix="churn", temperature=0.3, top_k=40, 
                                     top_p=0.9, repeat_penalty=1.1,
                                     templates_path="templates"):
        """
        Complete setup for specialized churn model including prompt loading and model configuration
        
        Args:
            base_model_name (str): Base model name (e.g., "qwen2.5-coder:7b")
            template_name (str): Name of template file
            custom_suffix (str): Suffix for custom model name
            temperature (float): Model temperature setting
            top_k (int): Top-k parameter
            top_p (float): Top-p parameter
            repeat_penalty (float): Repeat penalty parameter
            templates_path (str): Path to templates directory
            
        Returns:
            tuple: (success: bool, custom_model_name: str)
        """
        # Get templates directory path
        templates_dir = Path(templates_path)
        
        # Load system prompt
        system_prompt = self.load_system_prompt_from_template(templates_dir, template_name)
        
        # Use internal models directory (create it if needed)
        models_dir = self.setup_models_directory()
        
        # Configure specialized model
        success, custom_model_name = self.configure_specialized_model(
            base_model_name=base_model_name,
            system_prompt=system_prompt,
            models_dir=models_dir,
            custom_suffix=custom_suffix,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repeat_penalty
        )
        
        return success, custom_model_name
    
    def preload_sample_dataset(self, datasets_path="workspace/churn_analysis", 
                              create_sample=True):
        """
        Preload sample datasets for churn analysis
        
        Args:
            datasets_path (str): Path to datasets directory
            create_sample (bool): Whether to create a sample dataset if none exists
            
        Returns:
            bool: True if datasets are available, False otherwise
        """
        datasets_dir = Path(datasets_path)
        datasets_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for existing datasets
        csv_files = list(datasets_dir.glob("*.csv"))
        
        if csv_files:
            print(f"   ✅ Found {len(csv_files)} dataset(s) in {datasets_dir}")
            for csv_file in csv_files:
                print(f"      📊 {csv_file.name}")
            return True
        
        if create_sample:
            print(f"   📊 Creating sample churn dataset...")
            sample_data = """customer_id,tenure,monthly_charges,total_charges,contract_type,payment_method,churn_status
CUST001,12,29.85,358.2,Month-to-month,Electronic check,Yes
CUST002,34,56.95,1937.3,One year,Mailed check,No
CUST003,2,53.85,107.7,Month-to-month,Electronic check,Yes
CUST004,45,42.30,1899.5,Two year,Bank transfer,No
CUST005,8,70.70,565.6,Month-to-month,Credit card,No
CUST006,22,99.65,2191.3,One year,Credit card,No
CUST007,10,29.75,297.5,Month-to-month,Electronic check,Yes
CUST008,28,84.80,2372.4,Two year,Bank transfer,No
CUST009,7,56.15,393.05,Month-to-month,Mailed check,Yes
CUST010,36,42.65,1535.4,Two year,Credit card,No"""
            
            sample_file = datasets_dir / "sample_churn_data.csv"
            with open(sample_file, 'w', encoding='utf-8') as f:
                f.write(sample_data)
            
            print(f"   ✅ Created sample dataset: {sample_file}")
            print(f"   📝 Sample contains basic churn analysis fields")
            return True
        
        print(f"   ⚠️  No datasets found in {datasets_dir}")
        return False
