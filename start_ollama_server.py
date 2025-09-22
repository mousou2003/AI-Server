#!/usr/bin/env python3
"""
Ollama Server Starter

This script starts the Ollama server container using Docker Compose.
It does NOT start any churn assistant or WebUI features.

Usage:
  python start_ollama_server.py [--cpu] [--stop] [--status] [--logs]

Options:
  --cpu      Start in CPU-only mode (no GPU acceleration)
  --stop     Stop the Ollama server container
  --status   Show Ollama container status
  --logs     Show Ollama container logs
"""

import os
import sys
import argparse
import time
from utility_manager import UtilityManager


from ollama_manager import OllamaManager

def main(cpu_only=False, cleanup=False, status=False, logs=False, list_models=False, pull_models=False):
    base_files = ["docker-compose.ollama.yml"]
    if not cpu_only:
        base_files.append("docker-compose.gpu-override.yml")

    compose_cmd = "docker compose " + " ".join([f"-f {f}" for f in base_files])
    ollama = OllamaManager()
    container_name = ollama.config["name"]

    if cleanup:
        print("🛑 Stopping Ollama server...")
        UtilityManager.run_subprocess(f"{compose_cmd} down", show_output=True)
        return

    if status:
        print("🔍 Checking Ollama server status...")
        UtilityManager.run_subprocess(f"docker ps --filter name={container_name}", show_output=True)
        return

    if logs:
        print(f"📋 Showing logs for container: {container_name}")
        UtilityManager.run_subprocess(f"docker logs {container_name}", show_output=True)
        return

    if list_models:
        print("📋 Listing models in Ollama:")
        models = ollama.list_models()
        for m in models:
            print(f"  - {m}")
        return

    if pull_models:
        print("⬇️ Pulling models into Ollama...")
        ollama.pull_models()
        return

    print(f"🚀 Starting Ollama server ({'CPU mode' if cpu_only else 'GPU mode'})...")
    print(f"   Using: {' + '.join(base_files)}")
    UtilityManager.run_subprocess(f"{compose_cmd} up -d", show_output=True)
    print(f"✅ Ollama server started. Container name: {container_name}")
    print(f"💡 To stop: python start_ollama_server.py --cleanup")
    print(f"💡 To view logs: python start_ollama_server.py --logs")
    print(f"💡 To check status: python start_ollama_server.py --status")
    print(f"💡 To list models: python start_ollama_server.py --list-models")
    print(f"💡 To pull models: python start_ollama_server.py --pull-models")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Start Ollama server container only (no churn assistant)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_ollama_server.py                # Start with GPU acceleration
  python start_ollama_server.py --cpu-only     # Start in CPU-only mode
  python start_ollama_server.py --cleanup      # Stop Ollama server
  python start_ollama_server.py --status       # Show container status
  python start_ollama_server.py --logs         # Show container logs
  python start_ollama_server.py --list-models  # List models in Ollama
  python start_ollama_server.py --pull-models  # Pull models into Ollama
        """
    )
    parser.add_argument('--cpu-only', action='store_true', help='Run in CPU-only mode (no GPU acceleration required)')
    parser.add_argument('--cleanup', action='store_true', help='Stop containers after test')
    parser.add_argument('--list-models', action='store_true', help='List models in Ollama and exit')
    parser.add_argument('--pull-models', action='store_true', help='Pull models into Ollama and exit')
    parser.add_argument('--status', action='store_true', help='Show Ollama container status')
    parser.add_argument('--logs', action='store_true', help='Show Ollama container logs')
    args = parser.parse_args()

    main(cpu_only=args.cpu_only, cleanup=args.cleanup, list_models=args.list_models, pull_models=args.pull_models, status=args.status, logs=args.logs)
