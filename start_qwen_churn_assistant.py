#!/usr/bin/env python3
"""
Qwen Churn Assistant Infrastructure Starter

This script sets up the infrastructure for the Qwen Churn Assistant as described in churn_qwen.md.
It deploys Qwen2.5-Coder-32B-Instruct via Ollama and Open WebUI for churn analysis conversations.

Key Features:
- Deploys Qwen2.5-Coder-32B-Instruct model via Ollama
- Sets up Open WebUI for natural language churn analysis
- Configures the environment for CSV file analysis
- No code execution - purely conversational analysis
- Business-focused insights and recommendations

Requirements:
- Docker and Docker Compose
- For GPU mode: NVIDIA GPU with sufficient VRAM (recommended: 24GB+ for 32B model)
- For CPU mode: Sufficient RAM (slower but works on any hardware)
- CPU-only mode available with --cpu flag
"""

import os
import sys
import docker
import argparse
import webbrowser
import time
import requests
from subprocess import TimeoutExpired
import tempfile
import shutil
from pathlib import Path

# Import existing managers
from utility_manager import UtilityManager
from ollama_manager import OllamaManager
from webui_manager import WebUIManager


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
        
        # Qwen model configuration for churn analysis
        if cpu_mode and not large_model:
            # Use smaller model for CPU mode (much faster)
            self.selected_model = {
                "name": "qwen2.5-coder:7b",
                "description": "Qwen2.5-Coder-7B-Instruct - Optimized for CPU churn analysis",
                "vram_requirement": "8GB+ RAM (CPU mode)"
            }
            print("🖥️  CPU-only mode enabled - using 7B model for better performance")
        else:
            # Use full model for GPU mode or when explicitly requested
            self.selected_model = {
                "name": "qwen2.5-coder:32b",
                "description": "Qwen2.5-Coder-32B-Instruct - Comprehensive churn analysis model",
                "vram_requirement": "24GB+ VRAM (GPU mode) or 32GB+ RAM (CPU mode)"
            }
            if cpu_mode and large_model:
                print("🖥️  CPU-only mode with 32B model - this will be very slow!")
                print("     Consider using the default 7B model for CPU mode")
            elif cpu_mode:
                print("🖥️  CPU-only mode enabled")
            
        # Configuration
        self.config = {
            "project_name": "qwen-churn-assistant",
            "ollama_port": 11434,
            "webui_port": 3000,
            "model_name": self.selected_model["name"],
            "description": self.selected_model["description"],
            "cpu_mode": cpu_mode
        }
        
        # Create models directory if it doesn't exist
        self.models_dir = Path("models/.ollama")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Create staging directory for temporary files
        self.staging_dir = Path("staging")
        self.staging_dir.mkdir(exist_ok=True)
        
        # Create workspace and memory directories
        self.workspace_dir = Path("workspace")
        self.memory_dir = Path("memory")
        self.templates_dir = Path("templates")
        self.create_persistent_directories()
        
    def check_system_requirements(self):
        """Check if system meets requirements for the selected model"""
        print("🔍 Checking system requirements...")
        
        try:
            # Check if Docker is available
            docker_client = docker.from_env()
            print("✅ Docker is available")
            
            # Check if NVIDIA GPU is available
            try:
                result = self.utility_manager.run_subprocess('nvidia-smi', check=False)
                if result.returncode == 0:
                    print("✅ NVIDIA GPU detected")
                    # Try to extract VRAM info (basic check)
                    if 'MiB' in result.stdout:
                        print("✅ GPU memory information available")
                else:
                    print("⚠️  NVIDIA GPU not detected or nvidia-smi not available")
                    print("   Ollama will run in CPU mode (slower performance)")
            except Exception:
                print("⚠️  nvidia-smi not found - GPU acceleration may not be available")
                
        except Exception as e:
            print(f"❌ Docker not available: {e}")
            return False
            
        print(f"📋 Selected Model: {self.config['model_name']}")
        print(f"   {self.config['description']}")
        print(f"   VRAM Requirement: {self.selected_model['vram_requirement']}")
        print()
        
        return True
    
    def create_persistent_directories(self):
        """Create workspace and memory directories for Open WebUI"""
        # Create workspace directory for file analysis
        self.workspace_dir.mkdir(exist_ok=True)
        churn_workspace = self.workspace_dir / "churn_analysis"
        churn_workspace.mkdir(exist_ok=True)
        
        # Create memory directory
        self.memory_dir.mkdir(exist_ok=True)
        
        # Create initial memory file from template if it doesn't exist
        memory_file = self.memory_dir / "churn_qwen.md"
        if not memory_file.exists():
            self._create_memory_from_template(memory_file)
        
        print(f"   ✅ Created workspace: {churn_workspace}")
        print(f"   ✅ Created memory: {memory_file}")
    
    def _create_memory_from_template(self, memory_file):
        """Create memory file from template"""
        template_file = self.templates_dir / "churn_memory_template.md"
        
        # Copy from template
        with open(template_file, 'r', encoding='utf-8') as f:
            template_content = f.read()
        
        with open(memory_file, 'w', encoding='utf-8') as f:
            f.write(template_content)
        
        print(f"   ✅ Memory created from template: {template_file}")
    
    def create_docker_compose(self):
        """Create a specialized docker-compose file for Qwen Churn Assistant from template"""
        # Choose template based on CPU/GPU mode
        if self.config['cpu_mode']:
            template_file = self.templates_dir / "docker-compose.qwen-churn.cpu.template.yml"
        else:
            template_file = self.templates_dir / "docker-compose.qwen-churn.template.yml"
        
        # Load template and substitute variables
        with open(template_file, 'r', encoding='utf-8') as f:
            template_content = f.read()
        
        # Substitute template variables
        compose_content = template_content.replace('{{OLLAMA_PORT}}', str(self.config['ollama_port']))
        compose_content = compose_content.replace('{{WEBUI_PORT}}', str(self.config['webui_port']))
        
        compose_file = self.staging_dir / "docker-compose.qwen-churn.yml"
        with open(compose_file, 'w', encoding='utf-8') as f:
            f.write(compose_content)
            
        mode_info = "CPU-only" if self.config['cpu_mode'] else "GPU-accelerated"
        print(f"   ✅ Docker Compose created from template: {template_file} ({mode_info})")
        return str(compose_file)
    
    def pull_qwen_model(self):
        """Pull the selected Qwen model into Ollama"""
        print(f"🤖 Pulling Qwen model: {self.config['model_name']}")
        print("   This may take several minutes depending on your internet connection...")
        print(f"   Model: {self.selected_model['description']}")
        print(f"   Size: {'~3GB' if '7b' in self.config['model_name'] else '~20GB'}")
        
        try:
            # Pull model using docker exec
            cmd = f'docker exec ollama-qwen-churn ollama pull {self.config["model_name"]}'
            print(f"   Running: {cmd}")
            
            # Use longer timeout for 32B model
            timeout = 3600 if '32b' in self.config['model_name'] else 1800  # 60min vs 30min
            
            result = self.utility_manager.run_subprocess(cmd, check=False, timeout=timeout)
            
            if result.returncode == 0:
                print(f"✅ Successfully pulled {self.config['model_name']}")
                return True
            else:
                print(f"❌ Failed to pull {self.config['model_name']}")
                print(f"   Error: {result.stderr}")
                if '32b' in self.config['model_name'] and self.cpu_mode:
                    print("\n💡 Suggestion: Try using the default 7B model for CPU mode:")
                    print(f"   python {sys.argv[0]} --cpu")
                else:
                    print("\n💡 Suggestion: Check your internet connection and try again")
                return False
                
        except TimeoutExpired:
            timeout_min = timeout // 60
            print(f"⏰ Model pull timed out ({timeout_min} minutes)")
            if '32b' in self.config['model_name']:
                print("   The 32B model is very large. Consider using the 7B model for CPU mode.")
            return False
        except Exception as e:
            print(f"❌ Error pulling model: {e}")
            return False
    
    def wait_for_services(self):
        """Wait for both Ollama and Open WebUI to be ready"""
        print("⏳ Waiting for services to start...")
        
        # Wait for Ollama
        print("   Checking Ollama...")
        ollama_ready = False
        for i in range(120):  # 120 second timeout (2 minutes)
            try:
                response = requests.get(f"http://localhost:{self.config['ollama_port']}/api/tags", timeout=3)
                if response.status_code == 200:
                    print("   ✅ Ollama is ready")
                    ollama_ready = True
                    break
            except requests.RequestException:
                pass
            
            # Print progress every 15 seconds
            if i > 0 and i % 15 == 0:
                print(f"   ⏳ Still waiting for Ollama... ({i}s elapsed)")
            
            time.sleep(1)
        
        if not ollama_ready:
            print("   ❌ Ollama failed to start within timeout")
            print("   💡 Try checking container logs: docker logs ollama-qwen-churn")
            return False
        
        # Wait for Open WebUI
        print("   Checking Open WebUI...")
        webui_ready = False
        for i in range(60):  # 60 second timeout
            try:
                response = requests.get(f"http://localhost:{self.config['webui_port']}", timeout=3)
                if response.status_code == 200:
                    print("   ✅ Open WebUI is ready")
                    webui_ready = True
                    break
            except requests.RequestException:
                pass
            
            # Print progress every 15 seconds
            if i > 0 and i % 15 == 0:
                print(f"   ⏳ Still waiting for Open WebUI... ({i}s elapsed)")
                
            time.sleep(1)
        
        if not webui_ready:
            print("   ❌ Open WebUI failed to start within timeout")
            print("   💡 Try checking container logs: docker logs open-webui-qwen-churn")
            return False
            
        return True
    
    def create_churn_analysis_prompt(self):
        """Load the churn analysis system prompt from the template file"""
        prompt_file = self.templates_dir / "qwen_churn_system_prompt.template.md"
        
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
    
    def configure_model_with_prompt(self, system_prompt):
        """Configure the Qwen model with the churn analysis system prompt"""
        print("🔧 Configuring model with churn analysis prompt...")
        
        # Use staging directory for temporary files
        modelfile_path = self.staging_dir / "Modelfile.qwen-churn"
        
        try:
            # Create a Modelfile for Ollama with the system prompt
            modelfile_content = f'''FROM {self.config["model_name"]}

# Set parameters optimized for churn analysis
PARAMETER temperature 0.3
PARAMETER top_k 40
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1

# System prompt for churn analysis
SYSTEM """{system_prompt}"""

# Template for consistent responses
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
"""
'''
            
            # Save the Modelfile in staging directory
            with open(modelfile_path, 'w') as f:
                f.write(modelfile_content)
            
            print(f"   ✅ Created Modelfile: {modelfile_path}")
            
            # Create the customized model in Ollama
            custom_model_name = f"{self.config['model_name']}-churn"
            cmd = f'docker exec ollama-qwen-churn ollama create {custom_model_name} -f /app/Modelfile.qwen-churn'
            
            # Copy Modelfile to container first
            copy_cmd = f'docker cp {modelfile_path} ollama-qwen-churn:/app/Modelfile.qwen-churn'
            copy_result = self.utility_manager.run_subprocess(copy_cmd, check=False)
            
            if copy_result.returncode != 0:
                print(f"   ⚠️  Could not copy Modelfile to container: {copy_result.stderr}")
                return False
            
            # Create the custom model
            result = self.utility_manager.run_subprocess(cmd, check=False, timeout=300)
            
            if result.returncode == 0:
                print(f"   ✅ Created custom model: {custom_model_name}")
                # Update config to use the custom model
                self.config["custom_model_name"] = custom_model_name
                return True
            else:
                print(f"   ⚠️  Could not create custom model: {result.stderr}")
                print("   The base model will be used without the custom prompt")
                return False
                
        except TimeoutExpired:
            print("   ⏰ Model configuration timed out")
            return False
        except Exception as e:
            print(f"   ❌ Error configuring model: {e}")
            return False
    
    def verify_custom_model(self):
        """Verify that the custom churn analysis model is available"""
        if 'custom_model_name' not in self.config:
            return False
            
        try:
            # List models in Ollama to verify custom model exists
            cmd = 'docker exec ollama-qwen-churn ollama list'
            result = self.utility_manager.run_subprocess(cmd, check=False, timeout=30)
            
            if result.returncode == 0 and self.config['custom_model_name'] in result.stdout:
                print(f"   ✅ Custom model {self.config['custom_model_name']} is available")
                return True
            else:
                print(f"   ⚠️  Custom model {self.config['custom_model_name']} not found in Ollama")
                return False
                
        except Exception as e:
            print(f"   ❌ Error verifying custom model: {e}")
            return False
    
    def cleanup_staging(self):
        """Clean up the staging directory and temporary files (preserve memory and workspace)"""
        try:
            if self.staging_dir.exists():
                # Only remove files in staging directory, not the persistent directories
                for item in self.staging_dir.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                # Remove staging directory itself if empty
                try:
                    self.staging_dir.rmdir()
                    print("🧹 Cleaned up staging directory")
                except OSError:
                    # Directory not empty, that's fine
                    print("🧹 Cleaned up staging files")
        except Exception as e:
            print(f"⚠️  Could not clean up staging directory: {e}")
    
    def cleanup_all(self, remove_volumes=False):
        """Comprehensive cleanup including containers, images, and optionally volumes"""
        print("🧹 Performing comprehensive cleanup...")
        
        # Stop infrastructure and optionally remove volumes
        self.stop_infrastructure(remove_volumes=remove_volumes)
        
        # If we didn't remove volumes via compose, try manual removal
        if remove_volumes:
            print("🗑️  Ensuring all related volumes are removed...")
            volume_names = [
                "qwen-churn-assistant-data",
                "qwen-churn-assistant-memory", 
                "qwen-churn-assistant-workspace"
            ]
            
            for volume_name in volume_names:
                try:
                    result = self.utility_manager.run_subprocess(
                        f"docker volume rm {volume_name}",
                        check=False
                    )
                    if result.returncode == 0:
                        print(f"   ✅ Removed volume: {volume_name}")
                    else:
                        # Volume may have been removed by compose down -v
                        print(f"   ℹ️  Volume {volume_name} already removed or doesn't exist")
                except Exception as e:
                    print(f"   ❌ Error removing volume {volume_name}: {e}")
        
        print("✅ Comprehensive cleanup completed")
    
    def ensure_staging_cleanup(self):
        """Ensure staging directory is cleaned up on exit"""
        import atexit
        atexit.register(self.cleanup_staging)
    
    def start_infrastructure(self):
        """Start the complete Qwen Churn Assistant infrastructure"""
        mode_info = "CPU-only mode" if self.config['cpu_mode'] else "GPU-accelerated mode"
        print(f"🚀 Starting Qwen Churn Assistant Infrastructure ({mode_info})...")
        print("=" * 60)
        
        # Set up cleanup for staging directory
        self.ensure_staging_cleanup()
        
        # Check system requirements
        if not self.check_system_requirements():
            return False
        
        # Create docker-compose file
        compose_file = self.create_docker_compose()
        
        # Create system prompt
        system_prompt = self.create_churn_analysis_prompt()
        
        # Start services
        print("🐳 Starting Docker containers...")
        try:
            # First, stop any existing containers to avoid conflicts
            print("   Cleaning up any existing containers...")
            cleanup_result = self.utility_manager.run_subprocess(
                f"docker compose -p qwen-churn-assistant -f {compose_file} down",
                check=False
            )
            
            # Start the containers
            print("   Starting new containers...")
            result = self.utility_manager.run_subprocess(
                f"docker compose -p qwen-churn-assistant -f {compose_file} up -d",
                check=False
            )
            
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
            
        except Exception as e:
            print(f"❌ Error starting containers: {e}")
            return False
        
        # Wait for services
        if not self.wait_for_services():
            return False
        
        # Pull Qwen model
        model_pulled = self.pull_qwen_model()
        if not model_pulled:
            print("⚠️  Model pull failed, but services are running")
            print("   You can try pulling the model manually later")
            return False
        
        # Configure model with system prompt
        prompt_configured = self.configure_model_with_prompt(system_prompt)
        
        # Verify custom model if created
        if prompt_configured:
            self.verify_custom_model()
        
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
        print("   3. Upload your churn CSV file to the workspace")
        print("   4. Start asking natural language questions about churn patterns")
        print("   5. Example: 'Which customer segments have the highest churn?'")
        print()
        print("🧠 Features Enabled:")
        print("   📝 Memory: Conversations persist via Docker volumes")
        print("   🗂️  Workspace: Upload and analyze CSV files directly in WebUI")
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
        
        compose_file = self.staging_dir / "docker-compose.qwen-churn.yml"
        
        # Build the docker compose down command
        down_cmd = f"docker compose -p qwen-churn-assistant -f {compose_file} down"
        if remove_volumes:
            down_cmd += " -v"
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
        
        # Clean up staging directory only (preserve memory and workspace)
        self.cleanup_staging()
    
    def status(self):
        """Check the status of the infrastructure"""
        print("📊 Qwen Churn Assistant Status")
        print("=" * 40)
        
        try:
            client = docker.from_env()
            
            # Check containers
            containers = ["ollama-qwen-churn", "open-webui-qwen-churn"]
            for container_name in containers:
                try:
                    container = client.containers.get(container_name)
                    status = container.status
                    print(f"🐳 {container_name}: {status}")
                except docker.errors.NotFound:
                    print(f"🐳 {container_name}: Not found")
            
            # Check services
            services = [
                ("Ollama API", f"http://localhost:{self.config['ollama_port']}/api/tags"),
                ("Open WebUI", f"http://localhost:{self.config['webui_port']}")
            ]
            
            for service_name, url in services:
                try:
                    response = requests.get(url, timeout=2)
                    if response.status_code == 200:
                        print(f"🌐 {service_name}: ✅ Running")
                    else:
                        print(f"🌐 {service_name}: ⚠️  Responding with status {response.status_code}")
                except requests.RequestException:
                    print(f"🌐 {service_name}: ❌ Not responding")
                    
        except Exception as e:
            print(f"❌ Error checking status: {e}")

    def show_logs(self):
        """Show container logs for troubleshooting"""
        print("📋 Container Logs")
        print("=" * 40)
        
        containers = ["ollama-qwen-churn", "open-webui-qwen-churn"]
        
        for container_name in containers:
            print(f"\n🐳 {container_name} logs (last 20 lines):")
            print("-" * 40)
            try:
                result = self.utility_manager.run_subprocess(
                    f"docker logs --tail 20 {container_name}",
                    check=False
                )
                if result.returncode == 0:
                    print(result.stdout)
                    if result.stderr:
                        print("STDERR:", result.stderr)
                else:
                    print(f"❌ Could not get logs: {result.stderr}")
            except Exception as e:
                print(f"❌ Error getting logs: {e}")


def main():
    """Main function to handle command line arguments and start the assistant"""
    parser = argparse.ArgumentParser(
        description="Start Qwen Churn Assistant Infrastructure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_qwen_churn_assistant.py              # Start with GPU acceleration (32B model)
  python start_qwen_churn_assistant.py --cpu        # Use CPU-only mode (7B model)
  python start_qwen_churn_assistant.py --cpu --large-model  # CPU with 32B model (slow)
  python start_qwen_churn_assistant.py --stop       # Stop the infrastructure
  python start_qwen_churn_assistant.py --status     # Check status
  python start_qwen_churn_assistant.py --logs       # Show container logs
  python start_qwen_churn_assistant.py --open       # Open WebUI in browser
  python start_qwen_churn_assistant.py --cleanup    # Clean up temporary files only
  python start_qwen_churn_assistant.py --cleanup-all  # Clean up everything including Docker volumes
        """
    )
    
    parser.add_argument('--cpu', action='store_true',
                       help='Use CPU-only mode (no GPU acceleration, uses 7B model)')
    parser.add_argument('--large-model', action='store_true',
                       help='Force use of 32B model even in CPU mode (very slow)')
    parser.add_argument('--stop', action='store_true',
                       help='Stop the Qwen Churn Assistant infrastructure')
    parser.add_argument('--status', action='store_true',
                       help='Check the status of running services')
    parser.add_argument('--logs', action='store_true',
                       help='Show container logs for troubleshooting')
    parser.add_argument('--open', action='store_true',
                       help='Open WebUI in default browser')
    parser.add_argument('--cleanup', action='store_true',
                       help='Clean up staging directory and temporary files')
    parser.add_argument('--cleanup-all', action='store_true',
                       help='Comprehensive cleanup including Docker volumes (WARNING: removes all churn assistant data)')
    
    args = parser.parse_args()
    
    # Create manager instance
    manager = QwenChurnAssistantManager(cpu_mode=args.cpu, large_model=args.large_model)
    
    if args.stop:
        manager.stop_infrastructure()
    elif args.status:
        manager.status()
    elif args.logs:
        manager.show_logs()
    elif args.open:
        webbrowser.open(f"http://localhost:{manager.config['webui_port']}")
        print(f"🌐 Opening WebUI: http://localhost:{manager.config['webui_port']}")
    elif args.cleanup:
        manager.cleanup_staging()
        print("🧹 Staging cleanup completed")
    elif args.cleanup_all:
        print("⚠️  WARNING: This will remove ALL churn assistant data including Docker volumes!")
        try:
            response = input("Are you sure you want to proceed? (yes/no): ")
            if response.lower() == 'yes':
                manager.cleanup_all(remove_volumes=True)
            else:
                print("❌ Cleanup cancelled")
        except KeyboardInterrupt:
            print("\n❌ Cleanup cancelled")
    else:
        # Start infrastructure
        success = manager.start_infrastructure()
        
        if success:
            # Optionally open browser
            try:
                response = input("\n🌐 Would you like to open the WebUI in your browser? (y/N): ")
                if response.lower() in ['y', 'yes']:
                    webbrowser.open(f"http://localhost:{manager.config['webui_port']}")
            except KeyboardInterrupt:
                print("\n👋 Setup complete!")
        else:
            print("❌ Failed to start infrastructure")
            sys.exit(1)


if __name__ == "__main__":
    main()
