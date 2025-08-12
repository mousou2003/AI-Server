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


class QwenChurnAssistantManager:
    def __init__(self, cpu_mode=False, large_model=False, quiet_mode=False):
        """
        Initialize the Qwen Churn Assistant Manager
        
        Args:
            cpu_mode (bool): If True, use CPU-only mode (no GPU acceleration)
            large_model (bool): If True, force use of 14B model even in CPU mode
            quiet_mode (bool): If True, suppress startup messages (useful for status checks)
        """
        self.cpu_mode = cpu_mode
        self.large_model = large_model
        self.quiet_mode = quiet_mode
        self.utility_manager = UtilityManager()
        
        # Qwen model selection for churn analysis
        # Optimized for RTX 3060 Ti (8GB VRAM) - use 7B model for both modes
        if cpu_mode and not large_model:
            # Use smaller model for CPU mode (much faster)
            self.model_name = "qwen2.5:7b-instruct"
            if not quiet_mode:
                print("🖥️  CPU-only mode enabled - using 7B model for better performance")
        elif large_model:
            # Use full model only when explicitly requested
            self.model_name = "qwen2.5:14b-instruct"
            if not quiet_mode:
                print("⚠️  14B model requested - may be tight on RTX 3060 Ti (8GB VRAM)")
        else:
            # Use 7B model for GPU mode too (RTX 3060 Ti optimized)
            self.model_name = "qwen2.5:7b-instruct"
            if not quiet_mode:
                print("🚀 GPU mode enabled - using 7B model optimized for RTX 3060 Ti")
                if cpu_mode and large_model:
                    print("🖥️  CPU-only mode with 14B model - this will be slower!")
                    print("     Consider using the default 7B model for CPU mode")
                elif cpu_mode:
                    print("🖥️  CPU-only mode enabled")
        
        # Configure OllamaManager for our specific setup
        self.ollama_manager = OllamaManager()
        self.ollama_manager.setup_for_specialized_use(
            container_name="ollama-qwen-churn",
            specialized_models=[self.model_name]
        )
        
        # Configure WebUIManager for our specific setup
        self.webui_manager = WebUIManager()
            
        # Build minimal configuration - just the essentials
        self.config = {
            "project_name": "qwen-churn-assistant",
            "model_name": self.model_name,
            "cpu_mode": cpu_mode
        }
        
        # Setup complete infrastructure directories using OllamaManager
        self.ollama_manager.setup_complete_infrastructure(
            project_name="churn"
        )
        
        # Set up compose files - use modular override approach
        self.base_ollama_file = Path("docker-compose.ollama.yml")
        self.base_webui_file = Path("docker-compose.webui.yml")
        self.gpu_override_file = Path("docker-compose.gpu-override.yml")
        self.qwen_override_file = Path("docker-compose.qwen-churn-override.yml")
        
        if cpu_mode:
            if not quiet_mode:
                print(f"🖥️  Using: {self.base_ollama_file} + {self.base_webui_file} + {self.qwen_override_file} (CPU mode)")
        else:
            if not quiet_mode:
                print(f"🚀 Using: {self.base_ollama_file} + {self.base_webui_file} + {self.gpu_override_file} + {self.qwen_override_file} (GPU mode)")
        
    def wait_for_services(self):
        """Wait for both Ollama and Open WebUI to be ready using managers"""
        print("⏳ Waiting for services to start...")
        
        # Use OllamaManager's wait_for_api method (already configured for our container)
        ollama_ready = self.ollama_manager.wait_for_api(retries=120)
        
        if not ollama_ready:
            print("   💡 Try checking container logs: docker logs ollama-qwen-churn")
            return False
        
        # Additional check: Wait for tensor loading to complete
        print("   🧠 Checking for model tensor loading completion...")
        tensor_loading_complete = self.wait_for_tensor_loading()
        
        if not tensor_loading_complete:
            print("   ⚠️  Tensor loading check failed, but API is responsive")
            print("   💡 Model may still be loading in background")
            print("   💡 Try checking container logs: docker logs ollama-qwen-churn")
        
        # Force model loading by making a test API call
        print("   🔥 Triggering model loading with test API call...")
        model_loaded = self.trigger_model_loading()
        
        if not model_loaded:
            print("   ⚠️  Test API call failed, but basic API is responsive")
            print("   💡 Model may still be initializing - check performance in WebUI")
        else:
            print("   ✅ Model successfully loaded and ready for inference")
        
        # Use WebUIManager's wait_for_api_with_progress method with smart health checks
        # WebUI can take 3-5 minutes to initialize on first startup due to file downloads
        print("   📱 WebUI initialization can take 3-5 minutes on first startup...")
        webui_ready = self.webui_manager.wait_for_api_with_progress(retries=240, progress_interval=15)
        
        if not webui_ready:
            return False
            
        return True
    
    def wait_for_tensor_loading(self):
        """
        Wait for tensor loading to complete by monitoring container logs
        
        Returns:
            bool: True if tensor loading completes or is not detected, False if timeout
        """
        print("   🔍 Monitoring tensor loading progress...")
        
        import time
        max_wait_time = 300  # 5 minutes max wait for tensor loading
        check_interval = 5   # Check every 5 seconds
        start_time = time.time()
        
        tensor_loading_detected = False
        
        while (time.time() - start_time) < max_wait_time:
            try:
                # Get recent container logs
                stdout, stderr = self.utility_manager.get_container_logs("ollama-qwen-churn", lines=50)
                
                if stdout:
                    # Check for tensor loading messages
                    if "loading model tensors" in stdout.lower():
                        if not tensor_loading_detected:
                            print("   📊 Tensor loading detected, waiting for completion...")
                            tensor_loading_detected = True
                    
                    # Check for completion indicators
                    completion_indicators = [
                        "load_tensors: model tensors loaded",
                        "model loaded successfully",
                        "llama runner started",
                        "server listening"
                    ]
                    
                    if any(indicator in stdout.lower() for indicator in completion_indicators):
                        if tensor_loading_detected:
                            print("   ✅ Tensor loading completed successfully")
                        return True
                    
                    # Check for error conditions
                    error_indicators = [
                        "failed to load",
                        "out of memory",
                        "cuda error",
                        "tensor loading failed"
                    ]
                    
                    if any(error in stdout.lower() for error in error_indicators):
                        print(f"   ❌ Error detected during tensor loading")
                        return False
                
                # If no tensor loading detected within first 30 seconds, assume not needed
                if not tensor_loading_detected and (time.time() - start_time) > 30:
                    print("   ℹ️  No tensor loading detected - model may already be cached")
                    return True
                
                time.sleep(check_interval)
                
            except Exception as e:
                print(f"   ⚠️  Error monitoring tensor loading: {e}")
                # Don't fail the entire startup for monitoring issues
                return True
        
        if tensor_loading_detected:
            print(f"   ⚠️  Tensor loading timeout after {max_wait_time} seconds")
            print("   💡 Model may still be loading - check performance in WebUI")
            return False
        else:
            # No tensor loading detected, probably fine
            return True
    
    def trigger_model_loading(self):
        """
        Actively trigger model loading by making a test API call to Ollama
        
        Returns:
            bool: True if model loads successfully, False otherwise
        """
        import requests
        import json
        
        try:
            # Use a simple, fast test prompt to trigger model loading
            test_data = {
                "model": self.model_name,
                "prompt": "Hi",
                "stream": False,
                "options": {
                    "num_predict": 1,  # Only generate 1 token to make it fast
                    "temperature": 0.1
                }
            }
            
            # Make API call to trigger model loading
            print(f"   📡 Making test API call to load {self.model_name}...")
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=test_data,
                timeout=120  # 2 minutes timeout for model loading
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'response' in result:
                    print("   🎯 Test API call successful - model is loaded and responsive")
                    return True
                else:
                    print("   ⚠️  Test API call returned unexpected format")
                    return False
            else:
                print(f"   ❌ Test API call failed with status {response.status_code}")
                if response.status_code == 404:
                    print(f"   💡 Model '{self.model_name}' may not be available")
                return False
                
        except requests.exceptions.Timeout:
            print("   ⏰ Test API call timed out - model may be loading slowly")
            print("   💡 This is normal for first-time model loading")
            return False
        except requests.exceptions.ConnectionError:
            print("   🔌 Connection error during test API call")
            return False
        except Exception as e:
            print(f"   ❌ Error during test API call: {e}")
            return False
    
    def get_compose_command(self, action="up", additional_args=""):
        """Build the docker compose command with appropriate files using UtilityManager"""
        compose_files = [str(self.base_ollama_file)]
        
        # Add WebUI base file (needed for the qwen override to work properly)
        if self.base_webui_file.exists():
            compose_files.append(str(self.base_webui_file))
        
        # Add GPU override if not in CPU mode
        if not self.cpu_mode and self.gpu_override_file.exists():
            compose_files.append(str(self.gpu_override_file))
        
        # Always add Qwen-specific override
        if self.qwen_override_file.exists():
            compose_files.append(str(self.qwen_override_file))
        
        return self.utility_manager.build_compose_command(
            compose_files=compose_files,
            project_name="qwen-churn-assistant",
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
            model_description=f"{self.model_name} - Churn analysis model",
            vram_requirement="8GB+ RAM (CPU mode)" if self.cpu_mode else "12GB+ VRAM (GPU mode)"
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
        print(f"   Model: {self.model_name} - Churn analysis model")
        print(f"   Size: {'~3GB' if '7b' in self.config['model_name'] else '~8GB'}")
        
        try:
            self.ollama_manager.pull_models()
            model_pulled = True
        except Exception as e:
            print(f"❌ Error pulling model via OllamaManager: {e}")
            if '14b' in self.config['model_name'] and self.cpu_mode:
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
        print(f"📊 Open WebUI: http://localhost:3000")
        print(f"🤖 Ollama API: http://localhost:11434")
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
        # First check if containers exist
        containers = ["ollama-qwen-churn", "open-webui-qwen-churn"]
        containers_exist = False
        
        try:
            import docker
            client = docker.from_env()
            for container_name in containers:
                try:
                    container = client.containers.get(container_name)
                    containers_exist = True
                    break  # At least one container exists
                except docker.errors.NotFound:
                    continue
        except Exception:
            pass
        
        if containers_exist:
            # Try to detect the actual running mode from containers
            detected_mode = self.utility_manager.detect_running_mode("ollama-qwen-churn")
            
            if detected_mode != "unknown":
                # Use detected mode as the primary mode information
                actual_cpu_mode = (detected_mode == "cpu")
                mode_info = "CPU-only" if actual_cpu_mode else "GPU-accelerated"
                mode_note = ""
                
                # Only add a note if there's a significant mismatch that might confuse users
                # Don't show the note for normal operation (when mode detection works)
                # if actual_cpu_mode != self.cpu_mode:
                #     configured_mode = "CPU-only" if self.cpu_mode else "GPU-accelerated"
                #     mode_note = f" (configured as {configured_mode}, but detecting {mode_info})"
            else:
                # Fallback to configured mode
                actual_cpu_mode = self.cpu_mode
                mode_info = "CPU-only" if self.cpu_mode else "GPU-accelerated"
                mode_note = " (mode detection failed, using configured mode)"
        else:
            # No containers exist - infrastructure is not running
            actual_cpu_mode = self.cpu_mode
            mode_info = "CPU-only" if self.cpu_mode else "GPU-accelerated"
            mode_note = " (infrastructure not running)"
        
        print(f"📊 Qwen Churn Assistant Status ({mode_info}{mode_note})")
        print("=" * 50)
        print(f"🐳 Base Ollama file: {self.base_ollama_file}")
        print(f"🌐 Base WebUI file: {self.base_webui_file}")
        if not actual_cpu_mode and self.gpu_override_file.exists():
            print(f"🎮 GPU override file: {self.gpu_override_file}")
        if self.qwen_override_file.exists():
            print(f"🎯 Qwen override file: {self.qwen_override_file}")
        print(f"⚡ Mode: {mode_info}")
        print(f"🤖 Model: {self.config['model_name']}")
        print()
        
        try:
            # Check containers using UtilityManager
            print("🐳 Container Status:")
            self.utility_manager.check_container_status(containers)
            
            # Only check services if containers exist
            if containers_exist:
                print()
                
                # Check Ollama API using OllamaManager
                is_running, status_message = self.ollama_manager.get_api_status(11434)
                print(status_message)
                
                # Check WebUI using WebUIManager
                service_name, url, is_running = self.webui_manager.get_status_info()
                if is_running:
                    print(f"🌐 {service_name}: ✅ Running at {url}")
                else:
                    print(f"🌐 {service_name}: ❌ Not responding at {url}")
            else:
                print()
                print("🌐 Ollama API: ❌ Not running (container not found)")
                print("🌐 open-webui: ❌ Not running (container not found)")
                print()
                print("💡 To start the infrastructure:")
                cpu_flag = " --cpu" if self.cpu_mode else ""
                print(f"   python start_qwen_churn_assistant.py{cpu_flag}")
                    
        except Exception as e:
            print(f"❌ Error checking status: {e}")

    def rebuild_custom_model(self):
        """Rebuild the custom churn model, removing existing one first"""
        mode_info = "CPU-only" if self.cpu_mode else "GPU-accelerated"
        print(f"🔧 Rebuilding Custom Churn Model ({mode_info})")
        print("=" * 50)
        
        # Check if Ollama is running
        print("🔍 Checking Ollama service status...")
        is_running, status_message = self.ollama_manager.get_api_status(11434)
        
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
                remove_cmd = f"docker exec ollama-qwen-churn ollama rm {expected_custom_model_name}"
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
        
        containers = ["ollama-qwen-churn", "open-webui-qwen-churn"]
        
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

    def test_custom_model(self):
        """Test the custom churn model to verify it's working correctly"""
        mode_info = "CPU-only" if self.cpu_mode else "GPU-accelerated"
        print(f"🧪 Testing Custom Churn Model ({mode_info})")
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
        expected_custom_model_name = f"{self.model_name}-churn"
        print(f"🔍 Checking for custom model: {expected_custom_model_name}")
        
        custom_model_exists = self.ollama_manager.verify_model_exists(expected_custom_model_name)
        
        if not custom_model_exists:
            print(f"❌ Custom model '{expected_custom_model_name}' not found!")
            print("💡 Please create the custom model first:")
            print(f"   python start_qwen_churn_assistant.py --cpu --rebuild-model" if self.cpu_mode else f"   python start_qwen_churn_assistant.py --rebuild-model")
            return False
        
        print(f"✅ Custom model '{expected_custom_model_name}' is available")
        
        # Run the test
        test_prompt = "I have customer churn data and need help analyzing termination risk. Can you help?"
        
        print(f"\n🧪 Testing with prompt:")
        print(f"   '{test_prompt}'")
        print("\n📝 Response:")
        print("-" * 40)
        
        try:
            import subprocess
            
            # Increase timeout for large models
            timeout_seconds = 120 if "14b" in expected_custom_model_name else 60
            print(f"   ⏰ Using {timeout_seconds}s timeout for {'large' if '14b' in expected_custom_model_name else 'standard'} model")
            
            # Run the test command with proper encoding handling
            result = subprocess.run([
                "docker", "exec", "ollama-qwen-churn", 
                "ollama", "run", expected_custom_model_name, 
                test_prompt
            ], capture_output=True, text=True, timeout=timeout_seconds, encoding='utf-8', errors='replace')
            
            if result.returncode == 0:
                response = result.stdout.strip()
                print(response)
                
                # Analyze the response
                business_keywords = [
                    "business", "customer", "churn", "analysis", 
                    "retention", "segments", "patterns", "risk",
                    "data", "insights", "recommendations"
                ]
                
                privacy_keywords = [
                    "privacy", "personal data", "confidential", 
                    "can't provide", "protected under", "sorry"
                ]
                
                business_score = sum(1 for keyword in business_keywords if keyword.lower() in response.lower())
                privacy_score = sum(1 for keyword in privacy_keywords if keyword.lower() in response.lower())
                
                print(f"\n📊 Analysis Results:")
                print(f"   Business Keywords Found: {business_score}")
                print(f"   Privacy Keywords Found: {privacy_score}")
                
                if business_score >= 3 and privacy_score == 0:
                    print("   ✅ PASS: Model is responding as business churn analyst")
                    print("\n🎉 Test completed successfully!")
                    print(f"💡 You can now use '{expected_custom_model_name}' in the WebUI")
                    return True
                elif privacy_score > 0:
                    print("   ❌ FAIL: Model still showing privacy/HR response behavior")
                    print("💡 Try rebuilding the custom model:")
                    print(f"   python start_qwen_churn_assistant.py --cpu --rebuild-model" if self.cpu_mode else f"   python start_qwen_churn_assistant.py --rebuild-model")
                    return False
                else:
                    print("   ⚠️  UNCERTAIN: Response unclear or insufficient business keywords")
                    print("💡 Manual review recommended - check the response above")
                    return False
                    
            else:
                print(f"❌ Error running test: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            model_size = "14B" if "14b" in expected_custom_model_name else "7B"
            timeout_used = 120 if "14b" in expected_custom_model_name else 60
            print(f"❌ Test timed out after {timeout_used} seconds")
            if "14b" in expected_custom_model_name:
                print("💡 14B model may be tight on RTX 3060 Ti (8GB VRAM)")
                print("💡 Try using the optimized 7B model instead:")
                print(f"   python start_qwen_churn_assistant.py --test")
            else:
                print("💡 The model might be loading for the first time, try again")
            return False
        except Exception as e:
            print(f"❌ Error running test: {e}")
            return False
