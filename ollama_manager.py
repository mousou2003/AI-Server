import time
import requests
import subprocess
import docker
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
    
    def load_system_prompt_from_template(self, templates_dir, template_name="qwen_churn_system_prompt.template.json"):
        """
        Load system prompt from template file (supports both JSON and Markdown)
        
        Args:
            templates_dir (Path): Path to templates directory
            template_name (str): Name of template file
            
        Returns:
            str: System prompt content
        """
        prompt_file = Path(templates_dir) / template_name
        
        if not prompt_file.exists():
            raise FileNotFoundError(f"Template file not found: {prompt_file}")
        
        # Read the template file
        with open(prompt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Handle JSON template files
        if template_name.endswith('.json'):
            import json
            try:
                template_data = json.loads(content)
                if 'system_prompt' in template_data:
                    prompt_content = template_data['system_prompt']
                    print(f"   ✅ Loaded system prompt from JSON template: {prompt_file}")
                    return prompt_content
                else:
                    raise ValueError("JSON template missing 'system_prompt' field")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in template file: {e}")
        
        # Handle Markdown template files (legacy support)
        elif template_name.endswith('.md'):
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
            print(f"   ✅ Loaded system prompt from Markdown template: {prompt_file}")
            return prompt_content
        
        else:
            # Treat as plain text
            prompt_content = content.strip()
            print(f"   ✅ Loaded system prompt from text template: {prompt_file}")
            return prompt_content
    
    def configure_specialized_model(self, base_model_name, system_prompt, models_dir, 
                                  custom_suffix="churn", temperature=0.3, top_k=40, 
                                  top_p=0.9, repeat_penalty=1.1):
        """
        Configure a specialized model with custom system prompt and parameters
        
        Args:
            base_model_name (str): Base model name (e.g., "qwen2.5:7b-instruct")
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
    
    def setup_complete_infrastructure(self, project_name, 
                                    models_path=".ollama"):
        """
        Complete infrastructure setup for models directory
        
        Args:
            project_name (str): Project name for organization
            models_path (str): Path to the models directory
        """
        # Setup models directory (Ollama-specific)
        self.setup_models_directory(models_path)
        

    
    def setup_specialized_churn_model(self, base_model_name, 
                                     template_name="qwen_churn_system_prompt.template.json",
                                     custom_suffix="churn", temperature=0.3, top_k=40, 
                                     top_p=0.9, repeat_penalty=1.1,
                                     templates_path="templates"):
        """
        Complete setup for specialized churn model including prompt loading and model configuration
        
        Args:
            base_model_name (str): Base model name (e.g., "qwen2.5:7b-instruct")
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
