#!/usr/bin/env python3
"""
Qwen Churn Assistant Manager

This module contains the QwenChurnAssistantManager class that handles the infrastructure
for the Qwen Churn Assistant, including Docker container management, model configuration,
and service orchestration.

Key Features:
- Manages Ollama and Open WebUI containers for churn analysis
- Configures specialized Qwen models with churn analysis prompts
- Handles both CPU and GPU deployment modes
- Provides comprehensive infrastructure management and monitoring
"""

import os
import sys
import time
import traceback
from subprocess import TimeoutExpired
import tempfile
from pathlib import Path

# Import existing managers
from utility_manager import UtilityManager
from ollama_manager import OllamaManager
from webui_manager import WebUIManager
from qwen_config_loader import QwenConfig


class QwenChurnAssistantManager:
    def __init__(self, cpu_mode=False, large_model=False):
        """
        Initialize the Qwen Churn Assistant Manager
        
        Args:
            cpu_mode (bool): If True, use CPU-only mode (no GPU acceleration)
            large_model (bool): If True, force use of 32B model even in CPU mode
        """
        self.cpu_mode = cpu_mode
        self.large_model = large_model
        self.utility_manager = UtilityManager()
        
        # Load centralized configuration
        self.config_loader = QwenConfig()
        
        # Get model configuration from config file
        self.selected_model = self.config_loader.get_model_config(cpu_mode, large_model)
        
        # Display mode information
        if cpu_mode and not large_model:
            print("🖥️  CPU-only mode enabled - using 7B model for better performance")
        elif cpu_mode and large_model:
            print("🖥️  CPU-only mode with 32B model - this will be very slow!")
            print("     Consider using the default 7B model for CPU mode")
        elif cpu_mode:
            print("🖥️  CPU-only mode enabled")
        
        # Get container configurations
        ollama_config = self.config_loader.get_container_config('ollama')
        webui_config = self.config_loader.get_container_config('webui')
        
        # Configure OllamaManager for our specific setup
        self.ollama_manager = OllamaManager()
        self.ollama_manager.setup_for_specialized_use(
            container_name=ollama_config['name'],
            specialized_models=[self.selected_model["name"]],
            port=ollama_config['port']
        )
        
        # Configure WebUIManager for our specific setup
        self.webui_manager = WebUIManager()
        urls = self.config_loader.get_urls()
        self.webui_manager.config.update({
            "name": webui_config['name'],
            "port": webui_config['external_port'],
            "url": urls['webui']
        })
            
        # Build unified configuration
        self.config = {
            "project_name": self.config_loader.project_name,
            "ollama_port": ollama_config['port'],
            "webui_port": webui_config['external_port'],
            "model_name": self.selected_model["name"],
            "description": self.selected_model["description"],
            "cpu_mode": cpu_mode
        }
        
        # Setup complete infrastructure directories using OllamaManager
        self.ollama_manager.setup_complete_infrastructure(
            project_name="churn"
        )
        
        # Set up compose files using config loader
        compose_files = self.config_loader.get_compose_files(cpu_mode)
        self.base_ollama_file = Path(compose_files[0])
        self.base_webui_file = Path(compose_files[1])
        
        # Handle optional override files
        if len(compose_files) > 2:
            if not cpu_mode and "gpu-override" in compose_files[2]:
                self.gpu_override_file = Path(compose_files[2])
                self.qwen_override_file = Path(compose_files[3]) if len(compose_files) > 3 else None
            else:
                self.gpu_override_file = None
                self.qwen_override_file = Path(compose_files[2])
        else:
            self.gpu_override_file = None
            self.qwen_override_file = None
        
        # Display compose file usage
        files_used = compose_files[:2]  # Base files
        if not cpu_mode and self.gpu_override_file:
            files_used.append(str(self.gpu_override_file))
        if self.qwen_override_file:
            files_used.append(str(self.qwen_override_file))
            
        mode_suffix = " (CPU mode)" if cpu_mode else " (GPU mode)"
        files_display = " + ".join(files_used) + mode_suffix
        
        if cpu_mode:
            print(f"🖥️  Using: {files_display}")
        else:
            print(f"🚀 Using: {files_display}")
        
    def wait_for_services(self):
        """Wait for both Ollama and Open WebUI to be ready using managers"""
        print("⏳ Waiting for services to start...")
        
        # Use OllamaManager's wait_for_api method (already configured for our container)
        ollama_ready = self.ollama_manager.wait_for_api(retries=120)
        
        if not ollama_ready:
            ollama_config = self.config_loader.get_container_config('ollama')
            print(f"   💡 Try checking container logs: docker logs {ollama_config['name']}")
            return False
        
        # Use WebUIManager's wait_for_api_with_progress method with smart health checks
        # WebUI can take 3-5 minutes to initialize on first startup due to file downloads
        print("   📱 WebUI initialization can take 3-5 minutes on first startup...")
        webui_ready = self.webui_manager.wait_for_api_with_progress(retries=240, progress_interval=15)
        
        if not webui_ready:
            return False
            
        return True
    
    def get_compose_command(self, action="up", additional_args=""):
        """Build the docker compose command with appropriate files using UtilityManager"""
        compose_files = [str(self.base_ollama_file)]
        
        # Add WebUI base file (needed for the qwen override to work properly)
        if self.base_webui_file.exists():
            compose_files.append(str(self.base_webui_file))
        
        # Add GPU override if not in CPU mode
        if not self.cpu_mode and self.gpu_override_file and self.gpu_override_file.exists():
            compose_files.append(str(self.gpu_override_file))
        
        # Always add Qwen-specific override
        if self.qwen_override_file and self.qwen_override_file.exists():
            compose_files.append(str(self.qwen_override_file))
        
        return self.utility_manager.build_compose_command(
            compose_files=compose_files,
            project_name=self.config_loader.project_name,
            action=action,
            additional_args=additional_args
        )
    
    def cleanup_all(self, remove_volumes=False):
        """Comprehensive cleanup including containers, images, and optionally volumes"""
        print("🧹 Performing comprehensive cleanup...")
        
        # Stop infrastructure and optionally remove volumes
        self.stop_infrastructure(remove_volumes=remove_volumes)
        
        # If we didn't remove volumes via compose, try manual removal
        if remove_volumes:
            volume_names = [
                "qwen-churn-assistant-data",
                "qwen-churn-assistant-memory", 
                "qwen-churn-assistant-workspace"
            ]
            
            self.utility_manager.cleanup_docker_volumes(volume_names, "qwen-churn-assistant")
        
        print("✅ Comprehensive cleanup completed")
    
    def start_infrastructure(self):
        """Start the complete Qwen Churn Assistant infrastructure"""
        mode_info = "CPU-only mode" if self.config['cpu_mode'] else "GPU-accelerated mode"
        print(f"🚀 Starting Qwen Churn Assistant Infrastructure ({mode_info})...")
        print("=" * 60)
        
        # Check system requirements
        if not self.utility_manager.check_system_requirements(
            model_name=self.config['model_name'],
            model_description=self.config['description'],
            vram_requirement=self.selected_model['vram_requirement']
        ):
            return False
        
        # Validate base compose files exist using UtilityManager
        print("🔍 Validating Docker Compose files...")
        try:
            base_files = [self.base_ollama_file, self.base_webui_file]
            override_files = []
            
            if not self.cpu_mode:
                override_files.append(self.gpu_override_file)
            override_files.append(self.qwen_override_file)
            
            self.utility_manager.validate_compose_files(base_files, override_files)
        except FileNotFoundError as e:
            print(f"❌ {e}")
            return False
        
        # Ensure the external network exists using UtilityManager
        self.utility_manager.ensure_docker_network("ai_network")
        
        # Start services
        print("🐳 Starting Docker containers...")
        try:
            # First, stop any existing containers to avoid conflicts
            print("   Cleaning up any existing containers...")
            cleanup_cmd = self.get_compose_command("down")
            cleanup_result = self.utility_manager.run_subprocess(cleanup_cmd, check=False)
            
            # Start the containers
            print("   Starting new containers...")
            start_cmd = self.get_compose_command("up", "-d")
            result = self.utility_manager.run_subprocess(start_cmd, check=False)
            
            if result.returncode != 0:
                print(f"❌ Failed to start containers:")
                print(f"   {result.stderr}")
                if "is unhealthy" in result.stderr:
                    print("\n💡 Health check issue detected. This might be normal during first startup.")
                    print("   Ollama may take extra time to initialize. Checking service status...")
                    # Continue to wait_for_services which has its own retry logic
                else:
                    return False
            else:
                print("   ✅ Containers started successfully")
                print("   ℹ️  Note: WebUI may take 3-5 minutes to initialize on first startup")
            
        except Exception as e:
            print(f"❌ Error starting containers: {e}")
            return False
        
        # Wait for services
        if not self.wait_for_services():
            return False
        
        # Pull Qwen model using OllamaManager
        print(f"🤖 Pulling Qwen model: {self.config['model_name']}")
        print("   This may take several minutes depending on your internet connection...")
        print(f"   Model: {self.selected_model['description']}")
        print(f"   Size: {'~3GB' if '7b' in self.config['model_name'] else '~20GB'}")
        
        try:
            self.ollama_manager.pull_models()
            model_pulled = True
        except Exception as e:
            print(f"❌ Error pulling model via OllamaManager: {e}")
            if '32b' in self.config['model_name'] and self.cpu_mode:
                print("\n💡 Suggestion: Try using the default 7B model for CPU mode:")
                print(f"   python {sys.argv[0]} --cpu")
            model_pulled = False
            
        if not model_pulled:
            print("⚠️  Model pull failed, but services are running")
            print("   You can try pulling the model manually later")
            return False
        
        # Check if custom churn model already exists, create if not
        expected_custom_model_name = f"{self.config['model_name']}-churn"
        print(f"🔍 Checking if custom churn model exists: {expected_custom_model_name}")
        
        custom_model_exists = self.ollama_manager.verify_model_exists(expected_custom_model_name)
        
        if custom_model_exists:
            print(f"   ✅ Custom model already exists: {expected_custom_model_name}")
            prompt_configured = True
            custom_model_name = expected_custom_model_name
        else:
            print(f"   🔧 Custom model not found, creating: {expected_custom_model_name}")
            # Configure model with system prompt using OllamaManager
            prompt_configured, custom_model_name = self.ollama_manager.setup_specialized_churn_model(
                base_model_name=self.config["model_name"]
            )
            
            if prompt_configured:
                print(f"   ✅ Successfully created custom model: {custom_model_name}")
            else:
                print(f"   ⚠️  Failed to create custom model, will use base model")
        
        # Update config if custom model is available
        if prompt_configured:
            self.config["custom_model_name"] = custom_model_name
        
        # Success message
        mode_info = "CPU-only" if self.config['cpu_mode'] else "GPU-accelerated"
        print("\n" + "=" * 60)
        print(f"🎉 Qwen Churn Assistant Infrastructure is Ready! ({mode_info})")
        print("=" * 60)
        print(f"📊 Open WebUI: http://localhost:{self.config['webui_port']}")
        print(f"🤖 Ollama API: http://localhost:{self.config['ollama_port']}")
        print(f"🔧 Base Model: {self.config['model_name']}")
        print(f"⚡ Mode: {mode_info}")
        if prompt_configured and 'custom_model_name' in self.config:
            print(f"🎯 Custom Model: {self.config['custom_model_name']} (with churn analysis prompt)")
        print()
        print("📝 Next Steps:")
        print("   1. Open the WebUI in your browser")
        if prompt_configured and 'custom_model_name' in self.config:
            print(f"   2. Select the '{self.config['custom_model_name']}' model in the WebUI")
        else:
            print(f"   2. Select the '{self.config['model_name']}' model in the WebUI")
        print("   3. Start asking natural language questions about churn patterns")
        print("   4. Example: 'Which customer segments have the highest churn?'")
        print("   5. Provide specific data examples in your conversations")
        print()
        print("🧠 Features Enabled:")
        print("   📝 Memory: Conversations persist via Docker volumes")
        print("   � Interactive: Natural language churn analysis conversations")
        print("   🎯 Specialized: Business-focused churn analysis prompt")
        if prompt_configured and 'custom_model_name' in self.config:
            print(f"   🤖 Custom Model: {self.config['custom_model_name']} with embedded system prompt")
        print()
        print("💡 Remember: This assistant focuses on business insights, not code!")
        
        return True
    
    def stop_infrastructure(self, remove_volumes=False):
        """Stop the Qwen Churn Assistant infrastructure
        
        Args:
            remove_volumes (bool): If True, also remove Docker volumes
        """
        print("🛑 Stopping Qwen Churn Assistant Infrastructure...")
        
        # Build the docker compose down command
        additional_args = "-v" if remove_volumes else ""
        down_cmd = self.get_compose_command("down", additional_args)
        
        if remove_volumes:
            print("🗑️  Also removing Docker volumes...")
        
        try:
            result = self.utility_manager.run_subprocess(down_cmd, check=False)
            
            if result.returncode == 0:
                if remove_volumes:
                    print("✅ Infrastructure stopped and volumes removed")
                else:
                    print("✅ Infrastructure stopped successfully")
            else:
                print(f"⚠️  Some containers may still be running: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Error stopping infrastructure: {e}")
    
    def status(self):
        """Check the status of the infrastructure using managers"""
        mode_info = "CPU-only" if self.cpu_mode else "GPU-accelerated"
        print(f"📊 Qwen Churn Assistant Status ({mode_info})")
        print("=" * 50)
        print(f"🐳 Base Ollama file: {self.base_ollama_file}")
        print(f"🌐 Base WebUI file: {self.base_webui_file}")
        if not self.cpu_mode and self.gpu_override_file.exists():
            print(f"🎮 GPU override file: {self.gpu_override_file}")
        if self.qwen_override_file.exists():
            print(f"🎯 Qwen override file: {self.qwen_override_file}")
        print(f"⚡ Mode: {mode_info}")
        print(f"🤖 Model: {self.config['model_name']}")
        print()
        
        try:
            # Check containers using UtilityManager
            containers = [self.ollama_manager.config["name"], self.webui_manager.config["name"]]
            print("🐳 Container Status:")
            self.utility_manager.check_container_status(containers)
            
            # Check services using managers
            print()
            
            # Check Ollama API using OllamaManager
            is_running, status_message = self.ollama_manager.get_api_status(self.config['ollama_port'])
            print(status_message)
            
            # Check WebUI using WebUIManager
            service_name, url, is_running = self.webui_manager.get_status_info()
            if is_running:
                print(f"🌐 {service_name}: ✅ Running at {url}")
            else:
                print(f"🌐 {service_name}: ❌ Not responding at {url}")
                    
        except Exception as e:
            print(f"❌ Error checking status: {e}")

    def rebuild_custom_model(self):
        """Rebuild the custom churn model, removing existing one first"""
        mode_info = "CPU-only" if self.cpu_mode else "GPU-accelerated"
        print(f"🔧 Rebuilding Custom Churn Model ({mode_info})")
        print("=" * 50)
        
        # Check if Ollama is running
        print("🔍 Checking Ollama service status...")
        is_running, status_message = self.ollama_manager.get_api_status(self.config['ollama_port'])
        
        if not is_running:
            print("❌ Ollama service is not running!")
            print("💡 Please start the infrastructure first:")
            print(f"   python {sys.argv[0]} --cpu" if self.cpu_mode else f"   python {sys.argv[0]}")
            return False
        
        print("✅ Ollama service is available")
        
        # Check if base model exists
        print(f"🔍 Checking base model: {self.config['model_name']}")
        base_model_exists = self.ollama_manager.verify_model_exists(self.config["model_name"])
        
        if not base_model_exists:
            print(f"❌ Base model '{self.config['model_name']}' not found!")
            print("💡 Please pull the base model first:")
            print(f"   python {sys.argv[0]} --cpu" if self.cpu_mode else f"   python {sys.argv[0]}")
            return False
        
        print(f"✅ Base model '{self.config['model_name']}' is available")
        
        # Check if custom model already exists and remove it
        expected_custom_model_name = f"{self.config['model_name']}-churn"
        print(f"🔍 Checking for existing custom model: {expected_custom_model_name}")
        
        custom_model_exists = self.ollama_manager.verify_model_exists(expected_custom_model_name)
        
        if custom_model_exists:
            print(f"🗑️  Removing existing custom model: {expected_custom_model_name}")
            try:
                remove_cmd = f"docker exec {self.ollama_manager.config['name']} ollama rm {expected_custom_model_name}"
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
        print(f"🔧 Creating new custom churn model: {expected_custom_model_name}")
        print("   This will configure the model with specialized churn analysis prompt...")
        
        try:
            prompt_configured, custom_model_name = self.ollama_manager.setup_specialized_churn_model(
                base_model_name=self.config["model_name"]
            )
            
            if prompt_configured:
                print(f"✅ Successfully created custom model: {custom_model_name}")
                
                # Update config
                self.config["custom_model_name"] = custom_model_name
                
                # Verify the model exists
                print(f"🔍 Verifying custom model creation...")
                model_verified = self.ollama_manager.verify_model_exists(custom_model_name)
                
                if model_verified:
                    print(f"✅ Custom model verified: {custom_model_name}")
                    
                    # List all models to show current state
                    print(f"\n📋 Current models in Ollama:")
                    try:
                        models = self.ollama_manager.list_models()
                        if isinstance(models, list):
                            for model in models:
                                if "churn" in model:
                                    print(f"   🎯 {model} (custom)")
                                else:
                                    print(f"   - {model}")
                        else:
                            print(f"   {models}")
                    except Exception as e:
                        print(f"   ⚠️  Could not list models: {e}")
                    
                    print(f"\n🎉 Custom model rebuild completed successfully!")
                    print(f"💡 You can now use '{custom_model_name}' in the WebUI")
                    return True
                else:
                    print(f"❌ Custom model creation verification failed")
                    return False
            else:
                print(f"❌ Failed to create custom model")
                return False
                
        except Exception as e:
            print(f"❌ Error creating custom model: {e}")
            traceback.print_exc()
            return False

    def show_logs(self):
        """Show container logs for troubleshooting using UtilityManager"""
        print("📋 Container Logs")
        print("=" * 40)
        
        containers = [self.ollama_manager.config["name"], self.webui_manager.config["name"]]
        
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
