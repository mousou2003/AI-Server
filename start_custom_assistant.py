#!/usr/bin/env python3
"""
Custom Model Assistant Starter

This script demonstrates how to use the OllamaCustomModel with any template file.
It provides a generic way to start specialized assistants.
"""

import sys
import argparse
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from ollama_custom_model import OllamaCustomModel


def main():
    parser = argparse.ArgumentParser(description="Start Custom Model Assistant")
    parser.add_argument("template", help="Path to template JSON file")
    parser.add_argument("--cpu", action="store_true", 
                       help="Use CPU-only mode (slower but works without GPU)")
    parser.add_argument("--large", action="store_true", 
                       help="Use larger model variant")
    parser.add_argument("--model", help="Override base model name (e.g., 'llama3.1:8b-instruct')")
    parser.add_argument("--project", help="Override project name for Docker containers")
    parser.add_argument("--stop", action="store_true", 
                       help="Stop the infrastructure instead of starting")
    parser.add_argument("--status", action="store_true", 
                       help="Check status of the infrastructure")
    parser.add_argument("--remove-volumes", action="store_true", 
                       help="Remove Docker volumes when stopping (complete cleanup)")
    
    args = parser.parse_args()
    
    template_file = Path(args.template)
    if not template_file.exists():
        print(f"❌ Template file not found: {args.template}")
        return 1
    
    try:
        # Create the custom assistant
        custom_assistant = OllamaCustomModel(
            template_path=args.template,
            model_name=args.model,
            cpu_mode=args.cpu,
            large_model=args.large,
            project_name=args.project
        )
        
        if args.stop:
            custom_assistant.stop_infrastructure(remove_volumes=args.remove_volumes)
        elif args.status:
            custom_assistant.status()
        else:
            print(f"🚀 Starting {custom_assistant.template_data['name']}...")
            print(f"📝 {custom_assistant.template_data['description']}")
            print()
            
            success = custom_assistant.start_infrastructure()
            
            if success:
                print()
                print("🌐 Access the assistant at: http://localhost:3000")
                return 0
            else:
                print("❌ Failed to start assistant")
                return 1
                
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())