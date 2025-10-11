import subprocess
import sys
import docker
from pathlib import Path


class UtilityManager:
    """Utility functions for the application"""
    
    @staticmethod
    def run_subprocess(cmd, check=True, show_output=False, timeout=300):  # Default 5 minute timeout
        try:
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
        except subprocess.TimeoutExpired:
            print(f"⏰ Command timed out after {timeout} seconds: {cmd}")
            print("💡 This usually means Docker containers are taking too long to start")
            print("💡 Try using --cpu mode or check Docker resource allocation")
            if check:
                sys.exit(1)
            # Return a mock result for timeout cases
            return subprocess.CompletedProcess(cmd, 124, "", "Command timed out")
    
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
    def validate_compose_files(base_files, override_files=None):
        """Validate multiple Docker Compose files exist"""
        missing_files = []
        
        # Check base files (required)
        for file_path in base_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
                print(f"❌ Base file not found: {file_path}")
            else:
                print(f"✅ Base file found: {file_path}")
        
        # Check override files (optional)
        if override_files:
            for file_path in override_files:
                if not Path(file_path).exists():
                    print(f"⚠️  Override file not found: {file_path}")
                else:
                    print(f"✅ Override file found: {file_path}")
        
        if missing_files:
            raise FileNotFoundError(f"Required Docker Compose files not found: {missing_files}")
        
        return True
    
    @staticmethod
    def ensure_docker_network(network_name="ai_network"):
        """Ensure Docker network exists, create if not"""
        print(f"🌐 Ensuring Docker network exists: {network_name}")
        try:
            network_check = UtilityManager.run_subprocess(
                f"docker network inspect {network_name}", 
                check=False
            )
            
            if network_check.returncode != 0:
                print(f"   Creating network: {network_name}")
                network_create = UtilityManager.run_subprocess(
                    f"docker network create {network_name}",
                    check=True
                )
                print(f"   ✅ Network created: {network_name}")
            else:
                print(f"   ✅ Network already exists: {network_name}")
            return True
        except Exception as e:
            print(f"   ⚠️  Warning: Could not create network {network_name}: {e}")
            print("   Services may not be able to communicate properly")
            return False
    
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
    
    @staticmethod
    def build_compose_command(compose_files, project_name, action="up", additional_args=""):
        """Build a docker compose command with multiple files"""
        if isinstance(compose_files, str):
            compose_files = [compose_files]
        
        # Build command
        cmd_parts = ["docker", "compose"]
        
        if project_name:
            cmd_parts.extend(["-p", project_name])
            
        for compose_file in compose_files:
            cmd_parts.extend(["-f", str(compose_file)])
        
        cmd_parts.append(action)
        if additional_args:
            cmd_parts.extend(additional_args.split())
            
        return " ".join(cmd_parts)
    
    @staticmethod
    def list_docker_containers(filter_pattern=None):
        """List Docker containers, optionally filtered by name pattern"""
        try:
            cmd = "docker ps -a --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'"
            if filter_pattern:
                cmd += f" --filter name={filter_pattern}"
            
            result = UtilityManager.run_subprocess(cmd, check=False)
            if result.returncode == 0:
                return result.stdout
            else:
                return "No containers found or Docker not available"
        except Exception as e:
            return f"Error listing containers: {e}"
    
    @staticmethod
    def get_container_logs(container_name, lines=20):
        """Get logs from a Docker container"""
        try:
            result = UtilityManager.run_subprocess(
                f"docker logs --tail {lines} {container_name}",
                check=False
            )
            if result.returncode == 0:
                return result.stdout, result.stderr
            else:
                return None, f"Could not get logs: {result.stderr}"
        except Exception as e:
            return None, f"Error getting logs: {e}"
    
    @staticmethod
    def detect_running_mode(container_name):
        """
        Detect whether a container is running in CPU or GPU mode
        by checking container environment variables and runtime
        
        Args:
            container_name (str): Name of the container to check
        
        Returns:
            str: "cpu" or "gpu" or "unknown"
        """
        try:
            client = docker.from_env()
            
            # Check the specified container
            try:
                container = client.containers.get(container_name)
                
                # Check for GPU runtime
                host_config = container.attrs.get('HostConfig', {})
                device_requests = host_config.get('DeviceRequests', [])
                
                # If container has GPU device requests, it's GPU mode
                if device_requests:
                    for device_request in device_requests:
                        if device_request.get('Driver') == 'nvidia':
                            return "gpu"
                
                # Check environment variables for CUDA/GPU indicators
                env_vars = container.attrs.get('Config', {}).get('Env', [])
                for env_var in env_vars:
                    if 'CUDA' in env_var or 'GPU' in env_var:
                        return "gpu"
                
                # If no GPU indicators found, assume CPU mode
                return "cpu"
                
            except docker.errors.NotFound:
                # Container doesn't exist - this is expected after cleanup
                return "unknown"
                
        except Exception:
            # Docker not available or other issues - don't print error messages
            # as this is often expected (e.g., after cleanup)
            return "unknown"
