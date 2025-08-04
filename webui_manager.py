import time
import requests
import webbrowser
import docker
from pathlib import Path


class WebUIManager:
    def __init__(self):
        # Open WebUI configuration
        self.config = {
            "name": "open-webui",
            "port": 3000,
            "url": "http://localhost:3000"
        }
        
    def wait_for_api(self, retries=30):
        """Wait for WebUI to be ready"""
        print(f"⏳ Waiting for {self.config['name']}...")
        for _ in range(retries):
            try:
                r = requests.get(self.config["url"], timeout=2)
                if r.status_code == 200:
                    print(f"✅ {self.config['name']} is ready!")
                    return True
            except requests.RequestException:
                pass
            time.sleep(1)
        print(f"❌ {self.config['name']} not responding.")
        return False
    
    def wait_for_api_with_progress(self, retries=60, progress_interval=15):
        """
        Wait for WebUI to be ready with progress reporting and smart health checks
        
        Args:
            retries (int): Number of retries (default 60 for 60 seconds)
            progress_interval (int): Interval to show progress messages
            
        Returns:
            bool: True if WebUI is ready, False if timeout
        """
        print(f"   Checking {self.config['name']}...")
        
        container_name = self.config.get("name", "open-webui")
        
        for i in range(retries):
            # Check multiple readiness indicators
            container_ready = self._check_container_health(container_name)
            api_ready = self._check_api_ready()
            
            if api_ready:
                print(f"   ✅ {self.config['name']} is ready and responding")
                return True
            elif container_ready and i > 30:  # After 30 seconds, also check if container is healthy
                # Container is running but API not ready - check for common startup issues
                startup_status = self._check_startup_progress(container_name)
                if "ERROR" in startup_status:
                    print(f"   ❌ {self.config['name']} startup error detected: {startup_status}")
                    return False
            
            # Print progress every interval seconds with more detail
            if i > 0 and i % progress_interval == 0:
                status = self._get_detailed_status(container_name)
                print(f"   ⏳ Still waiting for {self.config['name']}... ({i}s elapsed) - {status}")
                
            time.sleep(1)
        
        print(f"   ❌ {self.config['name']} failed to start within timeout")
        print(f"   💡 Try checking container logs: docker logs {container_name}")
        return False
    
    def _check_container_health(self, container_name):
        """Check if container is running and healthy"""
        try:
            client = docker.from_env()
            container = client.containers.get(container_name)
            
            # Check basic running status
            if container.status != 'running':
                return False
            
            # Check health status if available
            health = container.attrs.get('State', {}).get('Health', {})
            if health:
                health_status = health.get('Status', 'none')
                if health_status == 'healthy':
                    return True
                elif health_status == 'unhealthy':
                    return False
                # If starting or no health check, continue with other checks
            
            return True  # Running but no definitive health info
        except (docker.errors.NotFound, Exception):
            return False
    
    def _check_api_ready(self):
        """Check if API endpoints are responding"""
        try:
            # Try multiple endpoints to ensure full readiness
            endpoints = [
                self.config["url"],  # Main page
                f"{self.config['url']}/api/v1/models",  # API endpoint
                f"{self.config['url']}/health"  # Health check endpoint
            ]
            
            for endpoint in endpoints:
                try:
                    response = requests.get(endpoint, timeout=3)
                    if response.status_code == 200:
                        return True
                except requests.RequestException:
                    continue
            return False
        except Exception:
            return False
    
    def _check_startup_progress(self, container_name):
        """Check container logs for startup progress"""
        try:
            client = docker.from_env()
            container = client.containers.get(container_name)
            logs = container.logs(tail=10).decode('utf-8')
            
            # Look for key startup indicators
            if "Started server process" in logs:
                return "Server started"
            elif "Waiting for application startup" in logs:
                return "Starting application"
            elif "Installing external dependencies" in logs:
                return "Installing dependencies"
            elif "Fetching" in logs and "files" in logs:
                return "Downloading files"
            elif "ERROR" in logs or "Error" in logs:
                return f"ERROR in logs"
            else:
                return "Initializing"
        except Exception:
            return "Unknown"
    
    def _get_detailed_status(self, container_name):
        """Get detailed status for progress reporting"""
        try:
            client = docker.from_env()
            container = client.containers.get(container_name)
            
            # Get recent logs to understand what's happening
            logs = container.logs(tail=5).decode('utf-8')
            
            if "Fetching" in logs and "files" in logs:
                # Extract progress from fetching logs
                lines = logs.split('\n')
                for line in reversed(lines):
                    if "Fetching" in line and "%" in line:
                        return f"Downloading: {line.split('|')[1].strip() if '|' in line else 'in progress'}"
                return "Downloading files"
            elif "Installing external dependencies" in logs:
                return "Installing dependencies"
            elif "Started server process" in logs:
                return "Server starting"
            elif "Waiting for application startup" in logs:
                return "Starting application"
            else:
                return f"Container: {container.status}"
        except Exception:
            return "Status unknown"
    
    def open_in_browser(self):
        """Open WebUI in the default browser"""
        try:
            webbrowser.open(self.config["url"])
            print(f"🌐 Opening {self.config['name']}: {self.config['url']}")
            return True
        except Exception as e:
            print(f"❌ Could not open browser: {e}")
            return False
    
    def get_status_info(self):
        """
        Get status information for WebUI
        
        Returns:
            tuple: (service_name, url, is_running)
        """
        try:
            response = requests.get(self.config["url"], timeout=2)
            is_running = response.status_code == 200
            return (self.config["name"], self.config["url"], is_running)
        except requests.RequestException:
            return (self.config["name"], self.config["url"], False)
    
    def create_persistent_directories(self, workspace_dir, memory_dir, templates_dir, project_name):
        """
        Create workspace and memory directories for Open WebUI
        
        Args:
            workspace_dir (Path): Workspace directory path
            memory_dir (Path): Memory directory path  
            templates_dir (Path): Templates directory path
            project_name (str): Project name for subdirectories and memory files
        """
        # Create workspace directory for file analysis
        workspace_dir.mkdir(exist_ok=True)
        project_workspace = workspace_dir / f"{project_name}_analysis"
        project_workspace.mkdir(exist_ok=True)
        
        # Create memory directory
        memory_dir.mkdir(exist_ok=True)
        
        # Create initial memory file from template if it doesn't exist
        memory_file = memory_dir / f"{project_name}.md"
        if not memory_file.exists():
            self._create_memory_from_template(memory_file, templates_dir, project_name)
        
        print(f"   ✅ Created workspace: {project_workspace}")
        print(f"   ✅ Created memory: {memory_file}")
        
        return project_workspace, memory_file
    
    def _create_memory_from_template(self, memory_file, templates_dir, project_name):
        """Create memory file from template"""
        template_file = templates_dir / f"{project_name}_memory_template.md"
        
        # Copy from template
        with open(template_file, 'r', encoding='utf-8') as f:
            template_content = f.read()
        
        with open(memory_file, 'w', encoding='utf-8') as f:
            f.write(template_content)
        
        print(f"   ✅ Memory created from template: {template_file}")
    
    def setup_base_directories(self, workspace_path="workspace", memory_path="memory", templates_path="templates"):
        """
        Setup base directories for WebUI operations
        
        Args:
            workspace_path (str): Path to workspace directory
            memory_path (str): Path to memory directory  
            templates_path (str): Path to templates directory
            
        Returns:
            tuple: (workspace_dir, memory_dir, templates_dir) as Path objects
        """
        workspace_dir = Path(workspace_path)
        memory_dir = Path(memory_path)
        templates_dir = Path(templates_path)
        
        return workspace_dir, memory_dir, templates_dir
