#!/usr/bin/env python3
"""
AI Server Network Manager

This script manages the shared Docker network for AI Server components.
Creates an external network that all services can connect to, allowing
for flexible deployment of individual services.
"""

import subprocess
import sys
from utility_manager import UtilityManager

class NetworkManager:
    def __init__(self):
        self.network_name = "ai_network"
        self.utility_manager = UtilityManager()
    
    def create_network(self):
        """Create the external AI network if it doesn't exist"""
        print(f"🌐 Creating external network: {self.network_name}")
        
        # Check if network already exists
        try:
            result = self.utility_manager.run_subprocess(
                f"docker network inspect {self.network_name}",
                check=False,
                capture_output=True
            )
            
            if result.returncode == 0:
                print(f"   ✅ Network '{self.network_name}' already exists")
                return True
                
        except Exception as e:
            print(f"   ℹ️  Checking network existence: {e}")
        
        # Create the network
        try:
            result = self.utility_manager.run_subprocess(
                f"docker network create {self.network_name}",
                check=True
            )
            print(f"   ✅ Created network: {self.network_name}")
            return True
            
        except Exception as e:
            print(f"   ❌ Failed to create network: {e}")
            return False
    
    def remove_network(self):
        """Remove the external AI network"""
        print(f"🗑️  Removing network: {self.network_name}")
        
        try:
            result = self.utility_manager.run_subprocess(
                f"docker network rm {self.network_name}",
                check=False
            )
            
            if result.returncode == 0:
                print(f"   ✅ Removed network: {self.network_name}")
                return True
            else:
                print(f"   ⚠️  Network may not exist or be in use: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"   ❌ Failed to remove network: {e}")
            return False
    
    def list_network_containers(self):
        """List containers connected to the AI network"""
        print(f"📋 Containers in network: {self.network_name}")
        
        try:
            result = self.utility_manager.run_subprocess(
                f"docker network inspect {self.network_name} --format '{{{{range .Containers}}}}{{{{.Name}}}} {{{{end}}}}'",
                check=False,
                capture_output=True
            )
            
            if result.returncode == 0:
                containers = result.stdout.strip().split()
                if containers and containers != ['']:
                    for container in containers:
                        print(f"   - {container}")
                else:
                    print("   (no containers connected)")
                return True
            else:
                print(f"   ❌ Network not found: {self.network_name}")
                return False
                
        except Exception as e:
            print(f"   ❌ Error listing containers: {e}")
            return False

def main():
    """Main function for network management"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Manage AI Server Docker Network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python network_manager.py create     # Create the ai_network
  python network_manager.py remove     # Remove the ai_network
  python network_manager.py list       # List containers in network
        """
    )
    
    parser.add_argument('action', choices=['create', 'remove', 'list'],
                       help='Action to perform on the network')
    
    args = parser.parse_args()
    
    manager = NetworkManager()
    
    if args.action == 'create':
        success = manager.create_network()
        sys.exit(0 if success else 1)
    elif args.action == 'remove':
        success = manager.remove_network()
        sys.exit(0 if success else 1)
    elif args.action == 'list':
        success = manager.list_network_containers()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
