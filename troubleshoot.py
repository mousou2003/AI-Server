#!/usr/bin/env python3
"""
AI-Server Troubleshooting Tool

This script helps diagnose and fix common issues with the AI-Server infrastructure.
"""

import sys
import argparse
import time
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from utility_manager import UtilityManager


def check_docker_status():
    """Check if Docker is running and accessible"""
    print("🔍 Checking Docker status...")
    
    try:
        result = UtilityManager.run_subprocess("docker version", check=False)
        if result.returncode == 0:
            print("✅ Docker is running and accessible")
            return True
        else:
            print("❌ Docker is not accessible")
            print("💡 Solution: Start Docker Desktop or Docker service")
            return False
    except Exception as e:
        print(f"❌ Error checking Docker: {e}")
        print("💡 Solution: Install Docker Desktop or start Docker service")
        return False


def check_containers_health():
    """Check the health of running containers"""
    print("\n🔍 Checking container health...")
    
    # Get all AI-Server related containers
    result = UtilityManager.run_subprocess(
        "docker ps -a --filter name=ollama --filter name=webui --format 'table {{.Names}}\\t{{.Status}}\\t{{.Ports}}'",
        check=False
    )
    
    if result.returncode == 0 and result.stdout.strip():
        print("📋 Container Status:")
        print(result.stdout)
        
        # Check for unhealthy containers
        unhealthy_result = UtilityManager.run_subprocess(
            "docker ps --filter health=unhealthy --format '{{.Names}}'",
            check=False
        )
        
        if unhealthy_result.returncode == 0 and unhealthy_result.stdout.strip():
            unhealthy_containers = unhealthy_result.stdout.strip().split('\n')
            print(f"\n⚠️  Unhealthy containers found: {unhealthy_containers}")
            
            for container in unhealthy_containers:
                if 'ollama' in container:
                    print(f"\n🔧 Troubleshooting {container}:")
                    print("   1. Check if Ollama service is starting properly")
                    print("   2. Verify health check endpoint is accessible")
                    print("   3. Try restarting the container")
                    
                    # Show container logs
                    print(f"\n📋 Recent logs for {container}:")
                    log_result = UtilityManager.run_subprocess(f"docker logs {container} --tail 10", check=False)
                    if log_result.returncode == 0:
                        print(log_result.stdout)
                    else:
                        print("❌ Could not retrieve logs")
            
            return False
        else:
            print("✅ All containers are healthy")
            return True
    else:
        print("ℹ️  No AI-Server containers found")
        return True


def check_network():
    """Check if the ai_network exists"""
    print("\n🔍 Checking Docker network...")
    
    result = UtilityManager.run_subprocess("docker network ls --filter name=ai_network", check=False)
    if result.returncode == 0 and 'ai_network' in result.stdout:
        print("✅ ai_network exists")
        return True
    else:
        print("❌ ai_network not found")
        print("🔧 Creating ai_network...")
        
        create_result = UtilityManager.run_subprocess("docker network create ai_network", check=False)
        if create_result.returncode == 0:
            print("✅ ai_network created successfully")
            return True
        else:
            print(f"❌ Failed to create network: {create_result.stderr}")
            return False


def fix_health_checks():
    """Restart containers with health check issues"""
    print("\n🔧 Attempting to fix health check issues...")
    
    # Get unhealthy containers
    result = UtilityManager.run_subprocess(
        "docker ps --filter health=unhealthy --format '{{.Names}}'",
        check=False
    )
    
    if result.returncode == 0 and result.stdout.strip():
        unhealthy_containers = result.stdout.strip().split('\n')
        
        for container in unhealthy_containers:
            print(f"🔄 Restarting {container}...")
            restart_result = UtilityManager.run_subprocess(f"docker restart {container}", check=False)
            
            if restart_result.returncode == 0:
                print(f"✅ {container} restarted")
                
                # Wait a bit for the container to start
                print("⏳ Waiting for container to initialize...")
                time.sleep(10)
                
                # Check health again
                health_result = UtilityManager.run_subprocess(
                    f"docker inspect {container} --format '{{{{.State.Health.Status}}}}'",
                    check=False
                )
                
                if health_result.returncode == 0:
                    health_status = health_result.stdout.strip()
                    print(f"📊 {container} health status: {health_status}")
                
            else:
                print(f"❌ Failed to restart {container}: {restart_result.stderr}")
    else:
        print("ℹ️  No unhealthy containers found")


def cleanup_and_restart():
    """Clean up all containers and restart fresh"""
    print("\n🧹 Performing complete cleanup and restart...")
    
    # Stop all AI-Server containers
    print("🛑 Stopping all AI-Server containers...")
    UtilityManager.run_subprocess(
        "docker stop $(docker ps -q --filter name=ollama --filter name=webui) 2>/dev/null || true",
        check=False,
        shell=True
    )
    
    # Remove containers
    print("🗑️  Removing containers...")
    UtilityManager.run_subprocess(
        "docker rm $(docker ps -aq --filter name=ollama --filter name=webui) 2>/dev/null || true",
        check=False,
        shell=True
    )
    
    print("✅ Cleanup complete")
    print("\n💡 Now you can restart your assistant:")
    print("   python start_yoga_assistant.py --cpu")
    print("   python start_qwen_churn_assistant.py --cpu")
    print("   python start_webui.py --cpu")


def main():
    parser = argparse.ArgumentParser(
        description="AI-Server Troubleshooting Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python troubleshoot.py                # Run all diagnostic checks
  python troubleshoot.py --fix          # Try to fix common issues
  python troubleshoot.py --cleanup      # Clean up and restart fresh
  python troubleshoot.py --health       # Check container health only

Common Issues:
  1. "container is unhealthy" - Health check failure, usually Ollama startup issue
  2. "dependency failed to start" - One container depends on another that's unhealthy
  3. Docker not running - Start Docker Desktop
  4. Network issues - Missing ai_network
        """
    )
    
    parser.add_argument("--fix", action="store_true",
                       help="Attempt to fix detected issues")
    parser.add_argument("--cleanup", action="store_true",
                       help="Perform complete cleanup and restart fresh")
    parser.add_argument("--health", action="store_true",
                       help="Check container health only")
    
    args = parser.parse_args()
    
    print("🛠️  AI-Server Troubleshooting Tool")
    print("=" * 50)
    
    if args.cleanup:
        cleanup_and_restart()
        return
    
    # Run diagnostic checks
    docker_ok = check_docker_status()
    
    if not docker_ok:
        print("\n❌ Cannot proceed without Docker. Please start Docker and try again.")
        return
    
    network_ok = check_network()
    health_ok = check_containers_health()
    
    if args.health:
        return  # Only check health, don't attempt fixes
    
    # Attempt fixes if requested
    if args.fix:
        if not health_ok:
            fix_health_checks()
        else:
            print("\n✅ No issues detected that can be automatically fixed")
    
    # Summary
    print("\n📋 Summary:")
    print(f"   Docker: {'✅' if docker_ok else '❌'}")
    print(f"   Network: {'✅' if network_ok else '❌'}")
    print(f"   Container Health: {'✅' if health_ok else '❌'}")
    
    if not (docker_ok and network_ok and health_ok):
        print("\n💡 Recommendations:")
        if not docker_ok:
            print("   - Start Docker Desktop")
        if not network_ok:
            print("   - Run: docker network create ai_network")
        if not health_ok:
            print("   - Run: python troubleshoot.py --fix")
            print("   - Or: python troubleshoot.py --cleanup")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)