#!/usr/bin/env python3
"""
Qwen Churn Assistant Infrastructure Starter

This script sets up the infrastructure for the Qwen Churn Assistant as described in churn_qwen.md.
It deploys Qwen2.5-Coder-32B-Instruct via Ollama and Open WebUI for churn analysis conversations.

Key Features:
- Deploys Qwen2.5-Coder-32B-Instruct model via Ollama
- Sets up Open WebUI for natural language churn analysis
- Configures specialized model with business-focused churn analysis prompt
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
import argparse
import time
import traceback
from subprocess import TimeoutExpired
import tempfile
from pathlib import Path

# Import managers
from qwen_churn_assistant_manager import QwenChurnAssistantManager


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
  python start_qwen_churn_assistant.py --rebuild-model  # Rebuild custom churn model only
  python start_qwen_churn_assistant.py --test      # Test the custom churn model
  python start_qwen_churn_assistant.py --cleanup-all  # Clean up everything including Docker volumes

Notes:
  - First startup may take 3-5 minutes as WebUI downloads dependencies
  - GPU mode requires NVIDIA GPU with 24GB+ VRAM for 32B model
  - CPU mode is slower but works on any hardware

Architecture:
  Base files: docker-compose.ollama.yml + docker-compose.webui.yml (CPU-optimized)
  GPU mode: + docker-compose.gpu-override.yml (adds GPU acceleration)
  Qwen churn: + docker-compose.qwen-churn-override.yml (adds churn-specific config)
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
    parser.add_argument('--rebuild-model', action='store_true',
                       help='Rebuild the custom churn model (requires running infrastructure)')
    parser.add_argument('--test', action='store_true',
                       help='Test the custom churn model to verify it\'s working correctly')
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
        manager.webui_manager.open_in_browser()
    elif args.rebuild_model:
        success = manager.rebuild_custom_model()
        if not success:
            print("❌ Failed to rebuild custom model")
            sys.exit(1)
    elif args.test:
        success = manager.test_custom_model()
        if not success:
            print("❌ Test failed")
            sys.exit(1)
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
            # Optionally open browser using WebUIManager
            try:
                response = input("\n🌐 Would you like to open the WebUI in your browser? (y/N): ")
                if response.lower() in ['y', 'yes']:
                    manager.webui_manager.open_in_browser()
            except KeyboardInterrupt:
                print("\n👋 Setup complete!")
        else:
            print("❌ Failed to start infrastructure")
            sys.exit(1)


if __name__ == "__main__":
    main()
