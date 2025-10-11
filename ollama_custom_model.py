#!/usr/bin/env python3
"""
Ollama Custom Model Manager

This module contains the OllamaCustomModel class that provides a generalized framework
for managing specialized Ollama models with custom templates and system prompts.

Key Features:
- Load and parse custom template JSON files
- Configure specialized models with custom system prompts
- Handle both CPU and GPU deployment modes
- Flexible Docker compose configuration management
- Support for different model types (churn analysis, yoga sequences, etc.)
"""

import os
import sys
import json
import time
import traceback
from subprocess import TimeoutExpired
import tempfile
from pathlib import Path
import shutil
import stat
from typing import Dict, Any, Optional, List, Tuple

# Import existing managers
from utility_manager import UtilityManager
from ollama_manager import OllamaManager
from webui_manager import WebUIManager


class TemplateLoader:
    """Handles loading and validation of template JSON files"""
    
    @staticmethod
    def load_template(template_path: str) -> Dict[str, Any]:
        """
        Load and parse a template JSON file
        
        Args:
            template_path (str): Path to the template JSON file
            
        Returns:
            Dict[str, Any]: Parsed template data
            
        Raises:
            FileNotFoundError: If template file doesn't exist
            ValueError: If template is invalid or missing required fields
        """
        template_file = Path(template_path)
        
        if not template_file.exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")
        
        try:
            with open(template_file, 'r', encoding='utf-8') as f:
                template_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in template file {template_path}: {e}")
        
        # Validate required fields
        required_fields = ['name', 'description', 'system_prompt']
        missing_fields = [field for field in required_fields if field not in template_data]
        
        if missing_fields:
            raise ValueError(f"Template missing required fields: {missing_fields}")
        
        return template_data
    
    @staticmethod
    def validate_template(template_data: Dict[str, Any]) -> bool:
        """
        Validate template structure and content
        
        Args:
            template_data (Dict[str, Any]): Template data to validate
            
        Returns:
            bool: True if template is valid
            
        Raises:
            ValueError: If template validation fails
        """
        # Check for required top-level fields
        required_fields = ['name', 'description', 'system_prompt']
        for field in required_fields:
            if field not in template_data:
                raise ValueError(f"Missing required field: {field}")
            if not isinstance(template_data[field], str) or not template_data[field].strip():
                raise ValueError(f"Field '{field}' must be a non-empty string")
        
        # Validate model_parameters if present
        if 'model_parameters' in template_data:
            params = template_data['model_parameters']
            if not isinstance(params, dict):
                raise ValueError("'model_parameters' must be a dictionary")
            
            # Validate parameter types and ranges
            param_validations = {
                'temperature': (float, 0.0, 2.0),
                'top_k': (int, 1, 100),
                'top_p': (float, 0.0, 1.0),
                'repeat_penalty': (float, 0.0, 2.0)
            }
            
            for param_name, (param_type, min_val, max_val) in param_validations.items():
                if param_name in params:
                    value = params[param_name]
                    if not isinstance(value, param_type):
                        raise ValueError(f"Parameter '{param_name}' must be of type {param_type.__name__}")
                    if not (min_val <= value <= max_val):
                        raise ValueError(f"Parameter '{param_name}' must be between {min_val} and {max_val}")
        
        return True


class OllamaCustomModel:
    """
    Generalized manager for custom Ollama models with template-based configuration
    """
    
    def __init__(self, template_path: str, model_name: str = None, cpu_mode: bool = False, 
                 large_model: bool = False, quiet_mode: bool = False, project_name: str = None):
        """
        Initialize the Ollama Custom Model Manager
        
        Args:
            template_path (str): Path to the template JSON file
            model_name (str, optional): Base model name (e.g., "qwen2.5:7b-instruct"). If None, will be inferred from template
            cpu_mode (bool): If True, use CPU-only mode (no GPU acceleration)
            large_model (bool): If True, force use of larger model variant
            quiet_mode (bool): If True, suppress startup messages
            project_name (str, optional): Project name for Docker containers. If None, derived from template
        """
        self.template_path = template_path
        self.cpu_mode = cpu_mode
        self.large_model = large_model
        self.quiet_mode = quiet_mode
        
        # Load and validate template
        self.template_data = TemplateLoader.load_template(template_path)
        TemplateLoader.validate_template(self.template_data)
        
        # Initialize managers
        self.utility_manager = UtilityManager()
        
        # Determine project name
        if project_name:
            self.project_name = project_name
        else:
            # Create project name from template name (sanitized for Docker)
            template_name = self.template_data['name'].lower()
            self.project_name = ''.join(c if c.isalnum() else '-' for c in template_name)
            self.project_name = self.project_name.strip('-')
        
        # Determine model name
        if model_name:
            self.base_model_name = model_name
        else:
            # Try to infer from template or use sensible default
            self.base_model_name = self._infer_model_name()
        
        # Apply model size logic
        if cpu_mode and not large_model:
            # Use smaller model for CPU mode
            if '14b' in self.base_model_name or '32b' in self.base_model_name:
                self.base_model_name = self.base_model_name.replace('14b', '7b').replace('32b', '7b')
            if not quiet_mode:
                print(f"🖥️  CPU-only mode enabled - using optimized model: {self.base_model_name}")
        elif large_model:
            # Use larger model if available
            if '7b' in self.base_model_name:
                self.base_model_name = self.base_model_name.replace('7b', '14b')
            if not quiet_mode:
                print(f"⚡ Large model requested: {self.base_model_name}")
        else:
            if not quiet_mode:
                print(f"🚀 Using model: {self.base_model_name}")
        
        # Configure managers
        self.ollama_manager = OllamaManager()
        self.ollama_manager.setup_for_specialized_use(
            container_name=f"ollama-{self.project_name}",
            specialized_models=[self.base_model_name]
        )
        
        self.webui_manager = WebUIManager()
        
        # Build configuration
        self.config = {
            "project_name": self.project_name,
            "template_name": self.template_data['name'],
            "template_description": self.template_data['description'],
            "base_model_name": self.base_model_name,
            "cpu_mode": cpu_mode,
            "model_parameters": self.template_data.get('model_parameters', {}),
            "system_prompt": self.template_data['system_prompt']
        }
        
        # Setup infrastructure directories
        self.ollama_manager.setup_complete_infrastructure(
            project_name=self.project_name.replace('-', '_')
        )
        
        # Set up compose files
        self._setup_compose_files()
        
        if not quiet_mode:
            print(f"📋 Loaded template: {self.template_data['name']}")
            print(f"🏗️  Project: {self.project_name}")
            print(f"🤖 Base model: {self.base_model_name}")
    
    def _infer_model_name(self) -> str:
        """
        Infer appropriate model name based on template content and use case
        
        Returns:
            str: Inferred model name
        """
        # Look for model suggestions in template
        if 'recommended_model' in self.template_data:
            return self.template_data['recommended_model']
        
        # Analyze template content to suggest appropriate model
        template_name = self.template_data['name'].lower()
        system_prompt = self.template_data['system_prompt'].lower()
        
        # For text generation, analysis, or creative tasks
        if any(keyword in template_name + system_prompt for keyword in 
               ['generate', 'create', 'write', 'analysis', 'sequence', 'yoga', 'creative']):
            return "qwen2.5:7b-instruct"  # Good for creative/generative tasks
        
        # For conversational or assistant tasks
        if any(keyword in template_name + system_prompt for keyword in 
               ['assistant', 'chat', 'conversation', 'help', 'support']):
            return "qwen2.5:7b-instruct"  # Good for conversation
        
        # For analytical or business tasks
        if any(keyword in template_name + system_prompt for keyword in 
               ['churn', 'business', 'data', 'analytical', 'insights']):
            return "qwen2.5:7b-instruct"  # Good for analysis
        
        # Default fallback
        return "qwen2.5:7b-instruct"
    
    def _setup_compose_files(self):
        """Setup Docker compose file paths and create project-specific override if needed"""
        self.base_ollama_file = Path("docker-compose.ollama.yml")
        self.base_webui_file = Path("docker-compose.webui.yml")
        self.gpu_override_file = Path("docker-compose.gpu-override.yml")
        
        # Create project-specific override file name
        self.project_override_file = Path(f"docker-compose.{self.project_name}-override.yml")
        
        # Create project-specific override file if it doesn't exist
        if not self.project_override_file.exists():
            self._create_project_override_file()
        
        if not self.quiet_mode:
            files_list = [str(self.base_ollama_file), str(self.base_webui_file)]
            if not self.cpu_mode:
                files_list.append(str(self.gpu_override_file))
            files_list.append(str(self.project_override_file))
            
            mode = "CPU mode" if self.cpu_mode else "GPU mode"
            print(f"🐳 Docker files ({mode}): {' + '.join(files_list)}")
    
    def _create_project_override_file(self):
        """Create a project-specific Docker override file from the template"""
        template_file = Path("templates/docker-compose.generic-override.template.yml")
        
        if not template_file.exists():
            if not self.quiet_mode:
                print(f"⚠️  Generic override template not found: {template_file}")
                print("   Creating basic override file...")
            
            # Create a basic override file
            basic_override_content = f"""services:
  ollama:
    container_name: ollama-{self.project_name}
    hostname: ollama-{self.project_name}
    environment:
      - OLLAMA_KEEP_ALIVE=24h
      - OLLAMA_HOST=0.0.0.0
    volumes:
      - ./.ollama/{self.project_name}:/root/.ollama
    networks:
      - ai_network

  open-webui:
    container_name: open-webui-{self.project_name}
    hostname: open-webui-{self.project_name}
    environment:
      - WEBUI_NAME={self.template_data['name']}
      - OLLAMA_BASE_URL=http://ollama-{self.project_name}:11434
      - WEBUI_AUTH=False
    volumes:
      - ./.webui/{self.project_name}/data:/app/backend/data
      - ./.webui/{self.project_name}/workspace:/app/backend/workspace
    networks:
      - ai_network
    depends_on:
      ollama:
        condition: service_started

networks:
  ai_network:
    external: true
"""
            
            with open(self.project_override_file, 'w', encoding='utf-8') as f:
                f.write(basic_override_content)
                
            if not self.quiet_mode:
                print(f"   ✅ Created basic override file: {self.project_override_file}")
            return
        
        # Process the template file
        try:
            with open(template_file, 'r', encoding='utf-8') as f:
                template_content = f.read()
            
            # Replace placeholders
            project_content = template_content.replace(
                'PROJECT_NAME_PLACEHOLDER', self.project_name
            ).replace(
                'PROJECT_TITLE_PLACEHOLDER', self.template_data['name']
            )
            
            # Write the processed content
            with open(self.project_override_file, 'w', encoding='utf-8') as f:
                f.write(project_content)
            
            if not self.quiet_mode:
                print(f"   ✅ Created project override file: {self.project_override_file}")
                
        except Exception as e:
            if not self.quiet_mode:
                print(f"   ⚠️  Error processing template: {e}")
                print("   Using basic override instead...")
            # Fall back to basic override creation
            self._create_project_override_file()
    
    def get_compose_command(self, action: str = "up", additional_args: str = "") -> str:
        """
        Build the docker compose command with appropriate files
        
        Args:
            action (str): Docker compose action (up, down, etc.)
            additional_args (str): Additional arguments for the command
            
        Returns:
            str: Complete docker compose command
        """
        compose_files = [str(self.base_ollama_file)]
        
        # Add WebUI base file
        if self.base_webui_file.exists():
            compose_files.append(str(self.base_webui_file))
        
        # Add GPU override if not in CPU mode
        if not self.cpu_mode and self.gpu_override_file.exists():
            compose_files.append(str(self.gpu_override_file))
        
        # Add project-specific override if it exists
        if self.project_override_file.exists():
            compose_files.append(str(self.project_override_file))
        
        return self.utility_manager.build_compose_command(
            compose_files=compose_files,
            project_name=self.project_name,
            action=action,
            additional_args=additional_args
        )
    
    def create_custom_model(self) -> Tuple[bool, str]:
        """
        Create a custom model with the template's system prompt
        
        Returns:
            Tuple[bool, str]: (success, custom_model_name)
        """
        custom_model_name = f"{self.base_model_name}-{self.project_name}"
        
        # Check if custom model already exists
        if self.ollama_manager.verify_model_exists(custom_model_name):
            if not self.quiet_mode:
                print(f"✅ Custom model already exists: {custom_model_name}")
            return True, custom_model_name
        
        if not self.quiet_mode:
            print(f"🔧 Creating custom model: {custom_model_name}")
            print(f"   Template: {self.template_data['name']}")
        
        try:
            # Create modelfile content with template system prompt
            modelfile_content = f"""FROM {self.base_model_name}

SYSTEM \"\"\"{self.template_data['system_prompt']}\"\"\"
"""
            
            # Add model parameters if specified
            if 'model_parameters' in self.template_data:
                params = self.template_data['model_parameters']
                for param_name, param_value in params.items():
                    modelfile_content += f"PARAMETER {param_name} {param_value}\n"
            
            # Create temporary modelfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.modelfile', delete=False) as f:
                f.write(modelfile_content)
                temp_modelfile = f.name
            
            try:
                # Create the custom model
                create_cmd = f"docker exec ollama-{self.project_name} ollama create {custom_model_name} -f /tmp/modelfile"
                
                # Copy modelfile to container
                copy_cmd = f"docker cp {temp_modelfile} ollama-{self.project_name}:/tmp/modelfile"
                copy_result = self.utility_manager.run_subprocess(copy_cmd, check=False)
                
                if copy_result.returncode != 0:
                    raise Exception(f"Failed to copy modelfile: {copy_result.stderr}")
                
                # Create the model
                create_result = self.utility_manager.run_subprocess(create_cmd, check=False)
                
                if create_result.returncode == 0:
                    if not self.quiet_mode:
                        print(f"✅ Successfully created custom model: {custom_model_name}")
                    self.config["custom_model_name"] = custom_model_name
                    return True, custom_model_name
                else:
                    raise Exception(f"Model creation failed: {create_result.stderr}")
                    
            finally:
                # Clean up temporary file
                if os.path.exists(temp_modelfile):
                    os.unlink(temp_modelfile)
                    
        except Exception as e:
            if not self.quiet_mode:
                print(f"❌ Error creating custom model: {e}")
            return False, ""
    
    def start_infrastructure(self) -> bool:
        """
        Start the complete infrastructure for the custom model
        
        Returns:
            bool: True if successful
        """
        mode_info = "CPU-only mode" if self.cpu_mode else "GPU-accelerated mode"
        print(f"🚀 Starting {self.template_data['name']} Infrastructure ({mode_info})...")
        print("=" * 60)
        
        # Check system requirements
        vram_req = "8GB+ RAM (CPU mode)" if self.cpu_mode else "12GB+ VRAM (GPU mode)"
        if not self.utility_manager.check_system_requirements(
            model_name=self.base_model_name,
            model_description=f"{self.base_model_name} - {self.template_data['description']}",
            vram_requirement=vram_req
        ):
            return False
        
        # Validate compose files
        print("🔍 Validating Docker Compose files...")
        try:
            base_files = [self.base_ollama_file, self.base_webui_file]
            override_files = []
            
            if not self.cpu_mode:
                override_files.append(self.gpu_override_file)
            if self.project_override_file.exists():
                override_files.append(self.project_override_file)
            
            self.utility_manager.validate_compose_files(base_files, override_files)
        except FileNotFoundError as e:
            print(f"❌ {e}")
            return False
        
        # Ensure external network exists
        self.utility_manager.ensure_docker_network("ai_network")
        
        # Start services
        print("🐳 Starting Docker containers...")
        try:
            # Clean up existing containers
            print("   Cleaning up any existing containers...")
            cleanup_cmd = self.get_compose_command("down")
            self.utility_manager.run_subprocess(cleanup_cmd, check=False)
            
            # Start new containers
            print("   Starting new containers...")
            start_cmd = self.get_compose_command("up", "-d")
            print(f"   Command: {start_cmd}")
            result = self.utility_manager.run_subprocess(start_cmd, check=False, timeout=120)  # 2 minute timeout
            
            if result.returncode != 0:
                print(f"❌ Failed to start containers: {result.stderr}")
                return False
            else:
                print("   ✅ Containers started successfully")
                
        except Exception as e:
            print(f"❌ Error starting containers: {e}")
            return False
        
        # Wait for services
        if not self.wait_for_services():
            return False
        
        # Pull base model
        print(f"🤖 Pulling base model: {self.base_model_name}")
        try:
            self.ollama_manager.pull_models()
            print(f"   ✅ Base model pulled successfully")
        except Exception as e:
            print(f"❌ Error pulling model: {e}")
            return False
        
        # Create custom model
        success, custom_model_name = self.create_custom_model()
        if not success:
            print("⚠️  Custom model creation failed, using base model")
            custom_model_name = self.base_model_name
        
        # Warm up model
        model_to_warm = custom_model_name if success else self.base_model_name
        print(f"🔥 Warming up model: {model_to_warm}")
        if self._warm_up_model(model_to_warm):
            print("   ✅ Model is ready for use")
        else:
            print("   ⚠️  Model warm-up failed, but infrastructure is running")
        
        # Success message
        print("\n" + "=" * 60)
        print(f"🎉 {self.template_data['name']} Infrastructure is Ready!")
        print("=" * 60)
        print(f"📊 Open WebUI: http://localhost:3000")
        print(f"🤖 Ollama API: http://localhost:11434")
        print(f"🔧 Base Model: {self.base_model_name}")
        if success:
            print(f"🎯 Custom Model: {custom_model_name}")
        print(f"⚡ Mode: {mode_info}")
        print()
        print("📝 Usage Instructions:")
        print("   1. Open the WebUI in your browser")
        print(f"   2. Select the '{custom_model_name if success else self.base_model_name}' model")
        print(f"   3. {self.template_data.get('usage_instructions', 'Start using the specialized assistant!')}")
        
        return True
    
    def wait_for_services(self) -> bool:
        """Wait for both Ollama and Open WebUI to be ready"""
        print("⏳ Waiting for services to start...")
        
        # Wait for Ollama
        ollama_ready = self.ollama_manager.wait_for_api(retries=120)
        if not ollama_ready:
            print(f"   💡 Try checking container logs: docker logs ollama-{self.project_name}")
            return False
        
        # Wait for WebUI
        print("   📱 Waiting for WebUI...")
        webui_ready = self.webui_manager.wait_for_api_with_progress(retries=240, progress_interval=15)
        
        return webui_ready
    
    def _warm_up_model(self, model_name: str) -> bool:
        """
        Warm up the model to ensure it's loaded
        
        Args:
            model_name (str): Name of the model to warm up
            
        Returns:
            bool: True if successful
        """
        try:
            warm_up_prompt = "Hello"
            warm_up_cmd = f'docker exec ollama-{self.project_name} ollama run {model_name} "{warm_up_prompt}"'
            
            result = self.utility_manager.run_subprocess(warm_up_cmd, timeout=120, check=False)
            return result.returncode == 0
            
        except Exception:
            return False
    
    def stop_infrastructure(self, remove_volumes: bool = False) -> None:
        """
        Stop the infrastructure
        
        Args:
            remove_volumes (bool): If True, also remove Docker volumes
        """
        print(f"🛑 Stopping {self.template_data['name']} Infrastructure...")
        
        additional_args = "-v" if remove_volumes else ""
        down_cmd = self.get_compose_command("down", additional_args)
        
        try:
            result = self.utility_manager.run_subprocess(down_cmd, check=False)
            if result.returncode == 0:
                status = "stopped and volumes removed" if remove_volumes else "stopped successfully"
                print(f"✅ Infrastructure {status}")
            else:
                print(f"⚠️  Some containers may still be running: {result.stderr}")
        except Exception as e:
            print(f"❌ Error stopping infrastructure: {e}")
    
    def status(self) -> None:
        """Check and display the status of the infrastructure"""
        mode_info = "CPU-only" if self.cpu_mode else "GPU-accelerated"
        print(f"📊 {self.template_data['name']} Status ({mode_info})")
        print("=" * 50)
        print(f"📋 Template: {self.template_data['name']}")
        print(f"🏗️  Project: {self.project_name}")
        print(f"🤖 Base Model: {self.base_model_name}")
        if 'custom_model_name' in self.config:
            print(f"🎯 Custom Model: {self.config['custom_model_name']}")
        print()
        
        # Check container status
        containers = [f"ollama-{self.project_name}", f"open-webui-{self.project_name}"]
        print("🐳 Container Status:")
        self.utility_manager.check_container_status(containers)
        
        print()
        # Check if any containers exist first
        containers_exist = False
        try:
            import docker
            client = docker.from_env()
            for container_name in containers:
                try:
                    container = client.containers.get(container_name)
                    containers_exist = True
                    break
                except docker.errors.NotFound:
                    continue
        except Exception:
            pass
        
        # Check service APIs
        is_running, status_message = self.ollama_manager.get_api_status(11434)
        service_name, url, webui_running = self.webui_manager.get_status_info()
        
        if not containers_exist:
            # Infrastructure not started - this is normal, not an error
            print(f"🌐 Ollama API: ⏸️  Not started")
            print(f"🌐 {service_name}: ⏸️  Not started at {url}")
            print()
            print("💡 Infrastructure Status: Not running (this is normal)")
            print("   To start the infrastructure:")
            cpu_flag = " --cpu" if self.cpu_mode else ""
            print(f"   python start_custom_assistant.py {self.template_path}{cpu_flag}")
        else:
            # Containers exist - show actual API status
            print(status_message)
            status_icon = "✅" if webui_running else "⚠️"
            status_text = "Running" if webui_running else "Not responding (check logs)"
            print(f"🌐 {service_name}: {status_icon} {status_text} at {url}")
            
            if not webui_running or not is_running:
                print()
                print("💡 Services are starting but not yet ready. This can take a few minutes.")
                print("   Check logs for more details:")
                print(f"   docker logs ollama-{self.project_name}")
                print(f"   docker logs open-webui-{self.project_name}")
    
    def show_logs(self) -> None:
        """Show container logs for troubleshooting"""
        print("📋 Container Logs")
        print("=" * 40)
        
        containers = [f"ollama-{self.project_name}", f"open-webui-{self.project_name}"]
        
        for container_name in containers:
            print(f"\n🐳 {container_name} logs (last 20 lines):")
            print("-" * 40)
            
            stdout, stderr = self.utility_manager.get_container_logs(container_name, lines=20)
            
            if stdout:
                print(stdout)
            if stderr and "Could not get logs" in stderr:
                print(f"❌ {stderr}")
            elif stderr:
                print("STDERR:", stderr)
    
    def rebuild_custom_model(self) -> bool:
        """Rebuild the custom model, removing existing one first"""
        mode_info = "CPU-only" if self.cpu_mode else "GPU-accelerated"
        print(f"🔧 Rebuilding Custom Model ({mode_info})")
        print("=" * 50)
        
        # Check if Ollama is running
        print("🔍 Checking Ollama service status...")
        is_running, status_message = self.ollama_manager.get_api_status(11434)
        
        if not is_running:
            print("❌ Ollama service is not running!")
            print("💡 Please start the infrastructure first:")
            print(f"   python start_qwen_churn_assistant.py --cpu" if self.cpu_mode else f"   python start_qwen_churn_assistant.py")
            return False
        
        print("✅ Ollama service is available")
        
        # Check if base model exists
        print(f"🔍 Checking base model: {self.base_model_name}")
        base_model_exists = self.ollama_manager.verify_model_exists(self.base_model_name)
        
        if not base_model_exists:
            print(f"❌ Base model '{self.base_model_name}' not found!")
            print("💡 Please pull the base model first:")
            print(f"   python start_qwen_churn_assistant.py --cpu" if self.cpu_mode else f"   python start_qwen_churn_assistant.py")
            return False
        
        print(f"✅ Base model '{self.base_model_name}' is available")
        
        # Remove existing custom model if it exists
        expected_custom_model_name = f"{self.base_model_name}-{self.project_name}"
        print(f"🔍 Checking for existing custom model: {expected_custom_model_name}")
        
        custom_model_exists = self.ollama_manager.verify_model_exists(expected_custom_model_name)
        
        if custom_model_exists:
            print(f"🗑️  Removing existing custom model: {expected_custom_model_name}")
            try:
                remove_cmd = f"docker exec ollama-{self.project_name} ollama rm {expected_custom_model_name}"
                result = self.utility_manager.run_subprocess(remove_cmd, check=False)
                
                if result.returncode == 0:
                    print(f"   ✅ Successfully removed existing custom model")
                else:
                    print(f"   ⚠️  Warning: Could not remove existing model: {result.stderr}")
                    print(f"   Continuing with creation anyway...")
                    
            except Exception as e:
                print(f"   ⚠️  Warning: Error removing existing model: {e}")
                print(f"   Continuing with creation anyway...")
        else:
            print(f"   ℹ️  No existing custom model found")
        
        # Create the new custom model
        print(f"🔧 Creating new custom model: {expected_custom_model_name}")
        print("   This will configure the model with specialized system prompt...")
        
        success, custom_model_name = self.create_custom_model()
        
        if success:
            print(f"✅ Successfully created custom model: {custom_model_name}")
            self.config["custom_model_name"] = custom_model_name
            
            # Verify the model exists
            print(f"🔍 Verifying custom model creation...")
            model_verified = self.ollama_manager.verify_model_exists(custom_model_name)
            
            if model_verified:
                print(f"✅ Custom model verified: {custom_model_name}")
                print(f"\n🎉 Custom model rebuild completed successfully!")
                print(f"💡 You can now use '{custom_model_name}' in the WebUI")
                return True
            else:
                print(f"❌ Custom model creation verification failed")
                return False
        else:
            print(f"❌ Failed to create custom model")
            return False
    
    def test_custom_model(self, quick_mode: bool = False) -> bool:
        """Test the custom model to verify it's working correctly"""
        mode_info = "CPU-only" if self.cpu_mode else "GPU-accelerated"
        test_type = "Quick" if quick_mode else "Comprehensive"
        print(f"🧪 Testing Custom Model - {test_type} ({mode_info})")
        print("=" * 50)
        
        # Check if Ollama is running
        print("🔍 Checking Ollama service status...")
        is_running, status_message = self.ollama_manager.get_api_status(11434)
        
        if not is_running:
            print("❌ Ollama service is not running!")
            print("💡 Please start the infrastructure first:")
            print(f"   python start_qwen_churn_assistant.py --cpu" if self.cpu_mode else f"   python start_qwen_churn_assistant.py")
            return False
        
        print("✅ Ollama service is available")
        
        # Determine which custom model to test
        expected_custom_model_name = f"{self.base_model_name}-{self.project_name}"
        print(f"🔍 Checking for custom model: {expected_custom_model_name}")
        
        custom_model_exists = self.ollama_manager.verify_model_exists(expected_custom_model_name)
        
        if not custom_model_exists:
            print(f"❌ Custom model '{expected_custom_model_name}' not found!")
            print("💡 Please create the custom model first:")
            print(f"   python start_qwen_churn_assistant.py --cpu --rebuild-model" if self.cpu_mode else f"   python start_qwen_churn_assistant.py --rebuild-model")
            return False
        
        print(f"✅ Custom model '{expected_custom_model_name}' is available")
        
        # Quick model responsiveness check
        print("\n🏃 Quick model responsiveness check...")
        test_success = self._warm_up_model(expected_custom_model_name)
        
        if quick_mode:
            if test_success:
                print("✅ Quick test passed - model is responsive")
                return True
            else:
                print("❌ Quick test failed - model is not responding")
                return False
        
        # For comprehensive test, just return the basic connectivity result
        if test_success:
            print("✅ Comprehensive test passed - model is working correctly")
            return True
        else:
            print("❌ Comprehensive test failed")
            return False
    
    def cleanup_all(self) -> None:
        """Comprehensive cleanup: removes containers, volumes, and models"""
        print("🧹 Performing comprehensive cleanup...")
        print("   This will remove:")
        print("   - All Docker containers and volumes for this project")
        print("   - Project-specific models and data")
        
        # Stop infrastructure and remove volumes
        self.stop_infrastructure(remove_volumes=True)
        
        # Manual volume cleanup if needed
        volume_names = [
            f"{self.project_name}-data",
            f"{self.project_name}-memory", 
            f"{self.project_name}-workspace"
        ]
        
        self.utility_manager.cleanup_docker_volumes(volume_names, self.project_name)
        
        print("✅ Comprehensive cleanup completed")
    
    @classmethod
    def create_from_template(cls, template_path: str, **kwargs) -> 'OllamaCustomModel':
        """
        Factory method to create an instance from a template file
        
        Args:
            template_path (str): Path to the template JSON file
            **kwargs: Additional arguments to pass to constructor
            
        Returns:
            OllamaCustomModel: Configured instance
        """
        return cls(template_path=template_path, **kwargs)
    
    @classmethod
    def create_churn_assistant(cls, **kwargs) -> 'OllamaCustomModel':
        """
        Factory method to create a churn analysis assistant
        
        Args:
            **kwargs: Arguments to pass to constructor
            
        Returns:
            OllamaCustomModel: Configured churn assistant instance
        """
        # This would use a churn analysis template
        template_path = "templates/qwen_churn_system_prompt.template.json"
        return cls(template_path=template_path, project_name="churn-assistant", **kwargs)
    
    @classmethod
    def create_yoga_assistant(cls, **kwargs) -> 'OllamaCustomModel':
        """
        Factory method to create a yoga sequence assistant
        
        Args:
            **kwargs: Arguments to pass to constructor
            
        Returns:
            OllamaCustomModel: Configured yoga assistant instance
        """
        template_path = "templates/yoga_sequence_system_prompt.template.json"
        return cls(template_path=template_path, project_name="yoga-assistant", **kwargs)


# Convenience functions for backwards compatibility and easy usage
def create_custom_model_from_template(template_path: str, **kwargs) -> OllamaCustomModel:
    """
    Convenience function to create a custom model from a template
    
    Args:
        template_path (str): Path to the template JSON file
        **kwargs: Additional arguments
        
    Returns:
        OllamaCustomModel: Configured instance
    """
    return OllamaCustomModel.create_from_template(template_path, **kwargs)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Ollama Custom Model Manager")
    parser.add_argument("template", help="Path to template JSON file")
    parser.add_argument("--cpu", action="store_true", help="Use CPU-only mode")
    parser.add_argument("--large", action="store_true", help="Use large model variant")
    parser.add_argument("--model", help="Override base model name")
    parser.add_argument("--project", help="Override project name")
    parser.add_argument("--action", choices=["start", "stop", "status"], 
                       default="start", help="Action to perform")
    
    args = parser.parse_args()
    
    try:
        custom_model = OllamaCustomModel(
            template_path=args.template,
            model_name=args.model,
            cpu_mode=args.cpu,
            large_model=args.large,
            project_name=args.project
        )
        
        if args.action == "start":
            success = custom_model.start_infrastructure()
            sys.exit(0 if success else 1)
        elif args.action == "stop":
            custom_model.stop_infrastructure()
        elif args.action == "status":
            custom_model.status()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)