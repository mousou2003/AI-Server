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
import shutil
import stat

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
        print("   🧠 Checking for tensor loading...")
        tensor_loading_complete = self.wait_for_tensor_loading()
        
        if not tensor_loading_complete:
            print("   ⚠️  Tensor loading check failed, but API is responsive")
        
        # Note: Model warm-up will happen after model setup in start_infrastructure
        print("   ℹ️  Model warm-up will occur after model setup")
        
        # Use WebUIManager's wait_for_api_with_progress method with smart health checks
        # WebUI can take 3-5 minutes to initialize on first startup due to file downloads
        print("   📱 Waiting for WebUI (can take 3-5 minutes on first startup)...")
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
        print("   🔍 Monitoring for tensor loading...", end="", flush=True)
        
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
                            print("   📊 Tensor loading detected...")
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
                            print()  # New line after progress
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
                elapsed = int(time.time() - start_time)
                if not tensor_loading_detected and elapsed > 30:
                    print()  # New line after monitoring message
                    print("   ℹ️  No tensor loading detected - model may already be cached")
                    return True
                
                # Update progress on same line if tensor loading detected
                if tensor_loading_detected:
                    print(f"\r   📊 Tensor loading in progress... ({elapsed}s elapsed)", end="", flush=True)
                
                time.sleep(check_interval)
                
            except Exception as e:
                print(f"   ⚠️  Error monitoring tensor loading: {e}")
                # Don't fail the entire startup for monitoring issues
                return True
        
        if tensor_loading_detected:
            print()  # New line after progress
            print(f"   ⚠️  Tensor loading timeout after {max_wait_time} seconds")
            print("   💡 Model may still be loading - check performance in WebUI")
            return False
        else:
            # No tensor loading detected, probably fine
            return True
    
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
    
    def cleanup_all(self):
        """Comprehensive cleanup: removes containers, volumes, and ALL Ollama models"""
        print("🧹 Performing comprehensive cleanup...")
        print("   This will remove:")
        print("   - All Docker containers and volumes")
        print("   - ALL Ollama models (including base models)")
        print("   - Complete .ollama directory")
        
        # Remove all Ollama models first
        print("🗑️  Removing ALL Ollama models...")
        self._cleanup_ollama_models()
        
        # Stop infrastructure and remove volumes
        self.stop_infrastructure(remove_volumes=True)
        
        # Manual volume cleanup if needed
        volume_names = [
            "qwen-churn-assistant-data",
            "qwen-churn-assistant-memory", 
            "qwen-churn-assistant-workspace"
        ]
        
        self.utility_manager.cleanup_docker_volumes(volume_names, "qwen-churn-assistant")
        
        # Remove .ollama directory for clean slate
        print("🗂️  Removing .ollama directory...")
        self._cleanup_ollama_directory()
        
        print("✅ Comprehensive cleanup completed (containers, volumes, and models removed)")
    
    def _cleanup_ollama_models(self):
        """Remove all Ollama models from the running container"""
        try:
            # Check if container is running first
            is_running, _ = self.ollama_manager.get_api_status(11434)
            
            if not is_running:
                print("   ℹ️  Ollama container not running - models will be removed with directory cleanup")
                return
            
            print("   📋 Listing current models...")
            try:
                models = self.ollama_manager.list_models()
                if isinstance(models, list) and models:
                    print(f"   📊 Found {len(models)} models to remove")
                    
                    for model in models:
                        print(f"   🗑️  Removing model: {model}")
                        try:
                            remove_cmd = f"docker exec ollama-qwen-churn ollama rm {model}"
                            result = self.utility_manager.run_subprocess(remove_cmd, check=False)
                            
                            if result.returncode == 0:
                                print(f"      ✅ Removed: {model}")
                            else:
                                print(f"      ⚠️  Warning: Could not remove {model}: {result.stderr}")
                        except Exception as e:
                            print(f"      ❌ Error removing {model}: {e}")
                else:
                    print("   ℹ️  No models found to remove")
            except Exception as e:
                print(f"   ❌ Error listing models: {e}")
                
        except Exception as e:
            print(f"   ❌ Error during model cleanup: {e}")
    
    def _cleanup_ollama_directory(self):
        """Remove the entire .ollama directory to ensure clean slate"""
        ollama_dir = Path(".ollama")
        
        if ollama_dir.exists():
            print(f"   🗂️  Removing .ollama directory: {ollama_dir.absolute()}")
            try:
                # On Windows, we might need to handle permissions carefully
                if os.name == 'nt':  # Windows
                    def handle_remove_readonly(func, path, exc):
                        if os.path.exists(path):
                            os.chmod(path, stat.S_IWRITE)
                            func(path)
                    
                    shutil.rmtree(ollama_dir, onerror=handle_remove_readonly)
                else:
                    shutil.rmtree(ollama_dir)
                
                print(f"      ✅ Removed .ollama directory")
            except Exception as e:
                print(f"      ❌ Error removing .ollama directory: {e}")
                print(f"      💡 You may need to remove it manually: {ollama_dir.absolute()}")
        else:
            print(f"   ℹ️  .ollama directory not found (already clean)")
    
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
        
        # Warm up the model to ensure it's loaded and ready for immediate use
        model_to_warm = custom_model_name if prompt_configured else self.config["model_name"]
        cpu_timing = " (may take 3-5 minutes in CPU mode)" if self.cpu_mode else ""
        print(f"\n🔥 Warming up model: {model_to_warm}{cpu_timing}")
        warm_up_success = self._warm_up_model(model_to_warm)
        
        if warm_up_success:
            print("   ✅ Model is loaded and ready for immediate use")
        else:
            print("   ⚠️  Model warm-up failed, but infrastructure is running")
            print("   💡 First question may take longer to respond")

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

    def test_custom_model(self, quick_mode=False):
        """Test the custom churn model to verify it's working correctly with our streamlined prompt template
        
        Args:
            quick_mode (bool): If True, run only basic connectivity and responsiveness tests
        """
        mode_info = "CPU-only" if self.cpu_mode else "GPU-accelerated"
        test_type = "Quick" if quick_mode else "Comprehensive"
        print(f"🧪 Testing Custom Churn Model - {test_type} ({mode_info})")
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
        
        # Validate template file exists and is valid
        print("\n🔍 Validating prompt template...")
        if not self._validate_prompt_template():
            return False
        
        # Quick model responsiveness check before running full tests
        print("\n🏃 Quick model responsiveness check...")
        if not self._quick_model_check(expected_custom_model_name):
            print("💡 Model appears to be slow or unresponsive. This could be due to:")
            print("   - First-time model loading (can take 2-5 minutes)")
            print("   - System resource constraints")
            print("   - Model not fully initialized")
            print("\n🔧 Try these solutions:")
            print("   1. Wait a few minutes and run the test again")
            print("   2. Check container logs: docker logs ollama-qwen-churn")
            print("   3. Restart the infrastructure if needed")
            return False
        
        # If quick mode, just return success after basic checks
        if quick_mode:
            print("\n✅ Quick test completed successfully!")
            print("💡 Basic model connectivity and responsiveness confirmed")
            print(f"💡 You can now use '{expected_custom_model_name}' in the WebUI")
            print("\n📝 To run comprehensive tests:")
            print("   python start_qwen_churn_assistant.py --test")
            return True
        
        # Run comprehensive tests
        print("\n🧪 Running comprehensive model tests...")
        test_results = []
        
        # Test 1: Basic role understanding
        test_results.append(self._test_basic_role_understanding(expected_custom_model_name))
        
        # Test 2: Question type recognition (exploratory)
        test_results.append(self._test_exploratory_questions(expected_custom_model_name))
        
        # Test 3: Question type recognition (specific queries)
        test_results.append(self._test_specific_queries(expected_custom_model_name))
        
        # Test 4: Question type recognition (strategic)
        test_results.append(self._test_strategic_questions(expected_custom_model_name))
        
        # Test 5: Business definitions understanding
        test_results.append(self._test_business_definitions(expected_custom_model_name))
        
        # Test 6: Statistical insights capability
        test_results.append(self._test_statistical_insights(expected_custom_model_name))
        
        # Calculate overall results
        passed_tests = sum(1 for result in test_results if result)
        total_tests = len(test_results)
        
        print(f"\n📊 Test Results Summary:")
        print("=" * 50)
        print(f"✅ Passed: {passed_tests}/{total_tests} tests")
        print(f"❌ Failed: {total_tests - passed_tests}/{total_tests} tests")
        
        if passed_tests == total_tests:
            print("\n🎉 All tests passed! The model is working correctly.")
            print(f"💡 You can now use '{expected_custom_model_name}' in the WebUI")
            print("\n📝 Next steps:")
            print("   1. Open WebUI at http://localhost:3000")
            print(f"   2. Select '{expected_custom_model_name}' as your model")
            print("   3. Start analyzing your churn data!")
            return True
        elif passed_tests >= total_tests * 0.7:  # 70% pass rate
            print("\n⚠️  Most tests passed, but some issues detected.")
            print("� The model should work for basic churn analysis.")
            print("🔧 Consider rebuilding if you encounter issues:")
            print(f"   python start_qwen_churn_assistant.py --cpu --rebuild-model" if self.cpu_mode else f"   python start_qwen_churn_assistant.py --rebuild-model")
            return True
        else:
            print("\n❌ Multiple test failures detected.")
            print("🔧 Please rebuild the custom model:")
            print(f"   python start_qwen_churn_assistant.py --cpu --rebuild-model" if self.cpu_mode else f"   python start_qwen_churn_assistant.py --rebuild-model")
            print("\n💡 If tests keep timing out, try:")
            print("   1. Restart the infrastructure to clear any stuck processes")
            print("   2. Check available system resources")
            print("   3. Run a basic model verification instead")
            return False
    
    def _validate_prompt_template(self):
        """Validate that the prompt template file exists and is properly structured"""
        template_path = Path("templates/qwen_churn_system_prompt.template.json")
        
        if not template_path.exists():
            print(f"❌ Template file not found: {template_path}")
            return False
        
        try:
            import json
            with open(template_path, 'r', encoding='utf-8') as f:
                template = json.load(f)
            
            # Check for required sections
            required_sections = ['name', 'description', 'system_prompt', 'dataset_knowledge', 'constraints', 'response_format']
            missing_sections = [section for section in required_sections if section not in template]
            
            if missing_sections:
                print(f"❌ Template missing required sections: {missing_sections}")
                return False
            
            # Check response_format structure
            response_format = template.get('response_format', {})
            required_rf_sections = ['analysis_workflow', 'style_guidelines', 'structure', 'response_guidelines']
            missing_rf_sections = [section for section in required_rf_sections if section not in response_format]
            
            if missing_rf_sections:
                print(f"❌ response_format missing sections: {missing_rf_sections}")
                return False
            
            # Check response_guidelines has the expected question types
            response_guidelines = response_format.get('response_guidelines', {})
            expected_question_types = ['exploratory_questions', 'specific_queries', 'strategic_business_questions', 'pattern_analysis']
            missing_question_types = [qt for qt in expected_question_types if qt not in response_guidelines]
            
            if missing_question_types:
                print(f"❌ response_guidelines missing question types: {missing_question_types}")
                return False
            
            print("✅ Template validation passed")
            return True
            
        except json.JSONDecodeError as e:
            print(f"❌ Template JSON parsing error: {e}")
            return False
        except Exception as e:
            print(f"❌ Template validation error: {e}")
            return False
    
    def _quick_model_check(self, model_name):
        """Quick check to see if the model is responsive before running full tests"""
        print(f"   🎯 Testing basic model responsiveness...")
        
        try:
            import subprocess
            
            # Use a very simple prompt that should respond quickly
            simple_prompt = "Hi"
            
            # Short timeout for quick check
            result = subprocess.run([
                "docker", "exec", "ollama-qwen-churn", 
                "ollama", "run", model_name, simple_prompt
            ], capture_output=True, text=True, timeout=30, encoding='utf-8', errors='replace')
            
            if result.returncode == 0 and result.stdout.strip():
                print(f"   ✅ Model is responsive")
                return True
            else:
                print(f"   ❌ Model not responding or returned empty response")
                if result.stderr:
                    print(f"   📋 Error: {result.stderr.strip()}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"   ❌ Model responsiveness check timed out (30s)")
            return False
        except Exception as e:
            print(f"   ❌ Error during responsiveness check: {e}")
            return False
    
    def _warm_up_model(self, model_name):
        """Warm up the model during deployment to ensure immediate responsiveness
        
        Args:
            model_name (str): Name of the model to warm up
            
        Returns:
            bool: True if warm-up successful, False otherwise
        """
        # Use ollama_manager's smart warm_up_model method (no log checking needed)
        # Use longer timeout for CPU mode since model loading can take 4-5 minutes
        timeout = 600 if self.cpu_mode else 300  # 10 minutes for CPU, 5 minutes for GPU
        if self.ollama_manager.warm_up_model(model_name, timeout=timeout):
            # Verify model stays in memory using ollama_manager
            return self.ollama_manager.verify_model_in_memory(model_name)
        else:
            return False
    
    def _verify_model_in_memory(self, model_name):
        """Verify model is loaded and stays in memory"""
        try:
            import subprocess
            
            print(f"   � Verifying model is loaded in memory...")
            verify_result = subprocess.run([
                "docker", "exec", "ollama-qwen-churn", 
                "ollama", "ps"
            ], capture_output=True, text=True, timeout=10, encoding='utf-8', errors='replace')
            
            if verify_result.returncode == 0:
                if model_name in verify_result.stdout:
                    print(f"   ✅ Model confirmed loaded in memory")
                    return True
                else:
                    print(f"   ⚠️  Model not showing as loaded (may have unloaded quickly)")
                    return True  # Still consider success if initial load worked
            
            return True
            
        except Exception as e:
            print(f"   ⚠️  Could not verify model in memory: {e}")
            return True  # Don't fail the entire warm-up for verification issues
    
    def _run_model_test(self, model_name, prompt, timeout=180):
        """Helper method to run a single test against the model"""
        try:
            import subprocess
            
            # Create optimized test command with shorter response limits
            optimized_prompt = f"{prompt} (Please provide a brief, focused response in 2-3 sentences.)"
            
            # Use ollama run with options for faster inference
            cmd = [
                "docker", "exec", "ollama-qwen-churn", 
                "ollama", "run", model_name,
                "--verbose",  # Add verbose to see what's happening
                optimized_prompt
            ]
            
            print(f"   🔄 Running test (timeout: {timeout}s)...")
            
            result = subprocess.run(
                cmd,
                capture_output=True, 
                text=True, 
                timeout=timeout, 
                encoding='utf-8', 
                errors='replace'
            )
            
            if result.returncode == 0:
                response = result.stdout.strip()
                if response:
                    print(f"   📝 Response received ({len(response)} chars)")
                    return response
                else:
                    print(f"   ⚠️  Empty response received")
                    return None
            else:
                error_msg = result.stderr.strip()
                print(f"   ❌ Model execution error: {error_msg}")
                
                # Check for specific error types
                if "model not found" in error_msg.lower():
                    print(f"   💡 Model '{model_name}' might not be properly loaded")
                elif "connection refused" in error_msg.lower():
                    print(f"   💡 Ollama service might not be fully ready")
                
                return None
                
        except subprocess.TimeoutExpired:
            print(f"   ⏰ Test timed out after {timeout} seconds")
            print(f"   💡 Model might be loading for the first time or under heavy load")
            print(f"   💡 Current CPU usage suggests the system isn't fully utilized")
            return None
        except Exception as e:
            print(f"   ❌ Test execution error: {e}")
            return None
    
    def _test_basic_role_understanding(self, model_name):
        """Test 1: Basic role understanding"""
        print("\n1️⃣ Testing basic role understanding...")
        
        prompt = "What is your role and how can you help me?"
        response = self._run_model_test(model_name, prompt)
        
        if not response:
            print("   ❌ No response received")
            return False
        
        # Check for key role indicators
        role_keywords = [
            "churn analysis", "churn analyst", "customer churn", 
            "retention", "business insights", "wine club", "membership"
        ]
        
        found_keywords = [kw for kw in role_keywords if kw.lower() in response.lower()]
        
        if len(found_keywords) >= 2:
            print(f"   ✅ Role understanding confirmed (found: {', '.join(found_keywords)})")
            return True
        else:
            print(f"   ❌ Role understanding unclear (found: {', '.join(found_keywords)})")
            print(f"   📄 Response: {response[:200]}...")
            return False
    
    def _test_exploratory_questions(self, model_name):
        """Test 2: Exploratory question type recognition"""
        print("\n2️⃣ Testing exploratory question recognition...")
        
        prompt = "What does the data show about customer churn patterns?"
        response = self._run_model_test(model_name, prompt)
        
        if not response:
            print("   ❌ No response received")
            return False
        
        # Should focus on steps 1-2 (Key Finding + Supporting Evidence)
        # Should NOT include forced business implications or actions
        unwanted_phrases = [
            "recommended action", "next steps", "you should implement",
            "strategy", "intervention", "retention program"
        ]
        
        found_unwanted = [phrase for phrase in unwanted_phrases if phrase.lower() in response.lower()]
        
        if found_unwanted:
            print(f"   ❌ Found forced business advice in exploratory response: {found_unwanted}")
            return False
        
        # Should focus on data and evidence
        data_phrases = [
            "data shows", "pattern", "analysis", "finding", "evidence", 
            "percentage", "rate", "segment", "trend"
        ]
        
        found_data = [phrase for phrase in data_phrases if phrase.lower() in response.lower()]
        
        if len(found_data) >= 2:
            print(f"   ✅ Appropriate exploratory response (found: {', '.join(found_data)})")
            return True
        else:
            print(f"   ❌ Response doesn't focus on data/evidence enough")
            print(f"   📄 Response: {response[:200]}...")
            return False
    
    def _test_specific_queries(self, model_name):
        """Test 3: Specific query type recognition"""
        print("\n3️⃣ Testing specific query recognition...")
        
        prompt = "Give me the membership IDs of customers at risk of churning"
        response = self._run_model_test(model_name, prompt)
        
        if not response:
            print("   ❌ No response received")
            return False
        
        # Should provide direct answer without forced implications
        direct_response_indicators = [
            "membership id", "customer id", "guid", "list", "members",
            "specific", "following", "these customers", "at risk"
        ]
        
        found_indicators = [ind for ind in direct_response_indicators if ind.lower() in response.lower()]
        
        # Should NOT include forced business steps
        business_forcing = [
            "business implication", "recommended action", "strategy",
            "what this means for the company", "next steps"
        ]
        
        found_forcing = [phrase for phrase in business_forcing if phrase.lower() in response.lower()]
        
        if len(found_indicators) >= 2 and not found_forcing:
            print(f"   ✅ Direct query response confirmed (found: {', '.join(found_indicators)})")
            return True
        else:
            print(f"   ❌ Response not appropriately direct")
            if found_forcing:
                print(f"   ❌ Found forced business content: {found_forcing}")
            print(f"   📄 Response: {response[:200]}...")
            return False
    
    def _test_strategic_questions(self, model_name):
        """Test 4: Strategic question type recognition"""
        print("\n4️⃣ Testing strategic question recognition...")
        
        prompt = "What should we do about high churn rates in our premium segment?"
        response = self._run_model_test(model_name, prompt)
        
        if not response:
            print("   ❌ No response received")
            return False
        
        # Should include strategic elements (steps 1-5 when relevant)
        strategic_elements = [
            "recommendation", "action", "strategy", "implement", 
            "should", "could", "suggest", "consider", "plan"
        ]
        
        found_strategic = [elem for elem in strategic_elements if elem.lower() in response.lower()]
        
        # Should also include analytical foundation
        analytical_elements = [
            "data", "analysis", "finding", "evidence", "pattern", "segment"
        ]
        
        found_analytical = [elem for elem in analytical_elements if elem.lower() in response.lower()]
        
        if len(found_strategic) >= 2 and len(found_analytical) >= 2:
            print(f"   ✅ Strategic response confirmed (strategic: {', '.join(found_strategic[:3])}, analytical: {', '.join(found_analytical[:3])})")
            return True
        else:
            print(f"   ❌ Strategic response incomplete")
            print(f"   📊 Strategic elements: {found_strategic}")
            print(f"   📈 Analytical elements: {found_analytical}")
            return False
    
    def _test_business_definitions(self, model_name):
        """Test 5: Business definitions understanding"""
        print("\n5️⃣ Testing business definitions understanding...")
        
        prompt = "What's the difference between churn rate and termination rate?"
        response = self._run_model_test(model_name, prompt)
        
        if not response:
            print("   ❌ No response received")
            return False
        
        # Should mention key differences from our definitions
        definition_elements = [
            "churn rate", "termination rate", "cancelled", "onhold", 
            "percentage", "customers", "active", "formal", "subset"
        ]
        
        found_elements = [elem for elem in definition_elements if elem.lower() in response.lower()]
        
        # Should demonstrate understanding of wine club context
        wine_context = [
            "wine club", "membership", "shipment", "winery"
        ]
        
        found_context = [ctx for ctx in wine_context if ctx.lower() in response.lower()]
        
        if len(found_elements) >= 4 and len(found_context) >= 1:
            print(f"   ✅ Business definitions confirmed (definitions: {', '.join(found_elements[:4])}, context: {', '.join(found_context)})")
            return True
        else:
            print(f"   ❌ Business definitions understanding unclear")
            print(f"   📚 Definition elements: {found_elements}")
            print(f"   🍷 Wine context: {found_context}")
            return False
    
    def _test_statistical_insights(self, model_name):
        """Test 6: Statistical insights capability"""
        print("\n6️⃣ Testing statistical insights capability...")
        
        prompt = "If I have 1000 customers and 150 churned, what insights can you provide?"
        response = self._run_model_test(model_name, prompt)
        
        if not response:
            print("   ❌ No response received")
            return False
        
        # Should include statistical analysis
        statistical_elements = [
            "15%", "percentage", "rate", "ratio", "150", "1000",
            "proportion", "metric", "calculate", "analysis"
        ]
        
        found_statistical = [elem for elem in statistical_elements if elem.lower() in response.lower()]
        
        # Should provide business context
        business_elements = [
            "benchmark", "industry", "segment", "pattern", "trend",
            "business", "insight", "finding", "implication"
        ]
        
        found_business = [elem for elem in business_elements if elem.lower() in response.lower()]
        
        # Should NOT refuse to do calculations or provide statistics
        refusal_phrases = [
            "can't calculate", "unable to provide", "don't have access",
            "need more context", "cannot determine", "insufficient data"
        ]
        
        found_refusal = [phrase for phrase in refusal_phrases if phrase.lower() in response.lower()]
        
        if len(found_statistical) >= 3 and len(found_business) >= 2 and not found_refusal:
            print(f"   ✅ Statistical insights confirmed (stats: {', '.join(found_statistical[:3])}, business: {', '.join(found_business[:2])})")
            return True
        else:
            print(f"   ❌ Statistical insights inadequate")
            if found_refusal:
                print(f"   ❌ Found statistical refusal: {found_refusal}")
            print(f"   📊 Statistical elements: {found_statistical}")
            print(f"   💼 Business elements: {found_business}")
            return False
