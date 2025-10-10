#!/usr/bin/env python3
"""
Yoga Sequence Assistant Starter

This script demonstrates how to use the OllamaCustomModel with the yoga sequence template.
It creates a specialized assistant for generating yoga class sequences.
"""

import sys
import argparse
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from ollama_custom_model import OllamaCustomModel


def main():
    parser = argparse.ArgumentParser(description="Start Yoga Sequence Assistant")
    parser.add_argument("--cpu", action="store_true", 
                       help="Use CPU-only mode (slower but works without GPU)")
    parser.add_argument("--large", action="store_true", 
                       help="Use larger model (14B instead of 7B)")  
    parser.add_argument("--stop", action="store_true", 
                       help="Stop the infrastructure instead of starting")
    parser.add_argument("--status", action="store_true", 
                       help="Check status of the infrastructure")
    parser.add_argument("--open", action="store_true", 
                       help="Open WebUI in default browser")
    parser.add_argument("--remove-volumes", action="store_true", 
                       help="Remove Docker volumes when stopping (complete cleanup)")
    
    args = parser.parse_args()
    
    # Path to the yoga sequence template
    template_path = "templates/yoga_sequence_system_prompt.template.json"
    
    try:
        # Create the yoga assistant using the factory method
        yoga_assistant = OllamaCustomModel.create_yoga_assistant(
            cpu_mode=args.cpu,
            large_model=args.large
        )
        
        if args.stop:
            yoga_assistant.stop_infrastructure(remove_volumes=args.remove_volumes)
        elif args.status:
            yoga_assistant.status()
        elif args.open:
            yoga_assistant.webui_manager.open_in_browser()
        else:
            print("🧘 Starting Yoga Sequence Assistant...")
            print("This assistant specializes in creating personalized yoga class sequences")
            print("with proper flow, balance, and calorie calculations.")
            print()
            
            success = yoga_assistant.start_infrastructure()
            
            if success:
                print()
                print("🎉 Yoga Sequence Assistant is ready!")
                print()
                print("📝 Example questions to try:")
                print("   • 'Create a 60-minute restorative yoga class for beginners'")
                print("   • 'Generate a power yoga sequence focusing on core strength'") 
                print("   • 'Design a hip-opening flow for tight office workers'")
                print("   • 'Make a 45-minute vinyasa class with backbends as the peak'")
                print()
                print("💡 Tips:")
                print("   - Specify duration, intensity, and focus areas")
                print("   - Mention any injuries or limitations")
                print("   - Include room temperature preference (heated/unheated)")
                print("   - Ask for modifications or variations")
                print()
                print("🌐 Access the assistant at: http://localhost:3000")
                
                return 0
            else:
                print("❌ Failed to start Yoga Sequence Assistant")
                return 1
                
    except FileNotFoundError:
        print(f"❌ Template file not found: {template_path}")
        print("Please ensure the yoga sequence template exists in the templates directory.")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())