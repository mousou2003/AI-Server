import subprocess
import sys
import docker
from pathlib import Path


class UtilityManager:
    """Utility functions for the application"""
    
    @staticmethod
    def run_subprocess(cmd, check=True, show_output=False, timeout=None):
        if show_output:
            # Show output in real-time for commands like docker compose
            print(f"🔧 Running: {cmd}")
            result = subprocess.run(cmd, shell=True, text=True, encoding='utf-8', errors='ignore', timeout=timeout)
            if check and result.returncode != 0:
                print(f"❌ Command failed: {cmd}")
                sys.exit(1)
            return result
        else:
            # Capture output for other commands
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='utf-8', errors='ignore', timeout=timeout)
            if check and result.returncode != 0:
                print(f"❌ Command failed: {cmd}")
                print(result.stderr.strip())
                sys.exit(1)
            return result
    
    @staticmethod
    def check_system_requirements(model_name, model_description, vram_requirement):
        """Check if system meets requirements for the selected model"""
        print("🔍 Checking system requirements...")
        
        try:
            # Check if Docker is available
            docker_client = docker.from_env()
            print("✅ Docker is available")
            
            # Check if NVIDIA GPU is available
            try:
                result = UtilityManager.run_subprocess('nvidia-smi', check=False)
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
            
        print(f"📋 Selected Model: {model_name}")
        print(f"   {model_description}")
        print(f"   VRAM Requirement: {vram_requirement}")
        print()
        
        return True
    
    @staticmethod
    def validate_docker_compose_file(compose_file_path):
        """Validate that a docker-compose file exists"""
        compose_file = Path(compose_file_path)
        if not compose_file.exists():
            raise FileNotFoundError(f"Docker compose file not found: {compose_file}")
        
        print(f"   ✅ Using existing docker-compose file: {compose_file}")
        return str(compose_file)
    
    @staticmethod
    def cleanup_docker_volumes(volume_names, project_name):
        """Clean up Docker volumes for a project"""
        print("🗑️  Ensuring all related volumes are removed...")
        
        for volume_name in volume_names:
            try:
                result = UtilityManager.run_subprocess(
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
        
        print("✅ Volume cleanup completed")
    
    @staticmethod
    def check_container_status(container_names):
        """Check the status of Docker containers"""
        try:
            client = docker.from_env()
            
            for container_name in container_names:
                try:
                    container = client.containers.get(container_name)
                    status = container.status
                    print(f"🐳 {container_name}: {status}")
                except docker.errors.NotFound:
                    print(f"🐳 {container_name}: Not found")
                    
        except Exception as e:
            print(f"❌ Error checking container status: {e}")
            return False
        
        return True
    
    @staticmethod
    def restart_containers(container_names):
        """Restart Docker containers"""
        try:
            client = docker.from_env()
            for name in container_names:
                try:
                    client.containers.get(name).restart()
                    print(f"🔄 Restarted container: {name}")
                except docker.errors.NotFound:
                    print(f"⚠️ Container not found: {name}")
                    
        except Exception as e:
            print(f"❌ Error restarting containers: {e}")
            return False
        
        return True
