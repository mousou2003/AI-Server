#!/usr/bin/env python3
"""
Qwen Churn Assistant Infrastructure Starter

This script sets up the infrastructure for the Qwen Churn Assistant as described in churn_qwen.md.
It deploys Qwen2.5-Instruct via Ollama and Open WebUI for churn analysis conversations.

Key Features:
- Deploys Qwen2.5-Instruct model via Ollama (7B default, 14B optional)
- Sets up Open WebUI for natural language churn analysis
- Configures specialized model with business-focused churn analysis prompt
- No code execution - purely conversational analysis
- Business-focused insights and recommendations

Requirements:
- Docker and Docker Compose
- For GPU mode: NVIDIA GPU (optimized for RTX 3060 Ti with 8GB VRAM)
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
  python start_qwen_churn_assistant.py              # Start with GPU acceleration (7B model)
  python start_qwen_churn_assistant.py --cpu        # Use CPU-only mode (7B model)
  python start_qwen_churn_assistant.py --cpu --large-model  # CPU with 14B model (slower)
  python start_qwen_churn_assistant.py --stop       # Stop the infrastructure
  python start_qwen_churn_assistant.py --status     # Check status
  python start_qwen_churn_assistant.py --logs       # Show container logs
  python start_qwen_churn_assistant.py --open       # Open WebUI in browser
  python start_qwen_churn_assistant.py --rebuild-model  # Rebuild custom churn model only
  python start_qwen_churn_assistant.py --test      # Test the custom churn model
  python start_qwen_churn_assistant.py --quick-test  # Quick connectivity test
  python start_qwen_churn_assistant.py --cleanup-all  # Clean up everything including Docker volumes

Notes:
  - First startup may take 3-5 minutes as WebUI downloads dependencies
  - GPU mode uses 7B model optimized for RTX 3060 Ti (8GB VRAM)
  - CPU mode is slower but works on any hardware
  - Use --large-model for 14B model (requires more resources)

Architecture:
  Base files: docker-compose.ollama.yml + docker-compose.webui.yml (CPU-optimized)
  GPU mode: + docker-compose.gpu-override.yml (adds GPU acceleration)
  Qwen churn: + docker-compose.qwen-churn-override.yml (adds churn-specific config)
        """
    )
    
    parser.add_argument('--cpu', action='store_true',
                       help='Use CPU-only mode (no GPU acceleration, uses 7B model)')
    parser.add_argument('--large-model', action='store_true',
                       help='Force use of 14B model even in CPU mode (slower)')
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
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test for basic model connectivity and responsiveness')
    parser.add_argument('--cleanup-all', action='store_true',
                       help='Nuclear cleanup: removes containers, volumes, AND all Ollama models (WARNING: removes everything)')
    parser.add_argument('--force', action='store_true',
                       help='Force cleanup without confirmation (use with --cleanup-all for testing)')
    
    args = parser.parse_args()
    
    # Create manager instance with quiet mode for status checks
    quiet_mode = args.status or args.logs  # Suppress output for status and logs commands
    manager = QwenChurnAssistantManager(cpu_mode=args.cpu, large_model=args.large_model, quiet_mode=quiet_mode)
    
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
        success = manager.test_custom_model(quick_mode=False)
        if not success:
            print("❌ Test failed")
            sys.exit(1)
    elif args.quick_test:
        success = manager.test_custom_model(quick_mode=True)
        if not success:
            print("❌ Quick test failed")
            sys.exit(1)
    elif args.cleanup_all:
        print("🚨 NUCLEAR WARNING: This will remove EVERYTHING:")
        print("   - All Docker containers and volumes")
        print("   - ALL Ollama models (including qwen2.5:7b-instruct, qwen2.5-coder:7b, etc.)")
        print("   - Complete .ollama directory")
        print("   - You'll need to re-download all models from scratch")
        
        if args.force:
            print("⚡ Force mode enabled - proceeding without confirmation")
            manager.cleanup_all()
        else:
            try:
                response = input("Are you absolutely sure? Type 'DELETE EVERYTHING' to confirm: ")
                if response == 'DELETE EVERYTHING':
                    manager.cleanup_all()
                else:
                    print("❌ Nuclear cleanup cancelled")
            except KeyboardInterrupt:
                print("\n❌ Nuclear cleanup cancelled")
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
