#!/usr/bin/env python3
"""
Custom Model Assistant Deployer

This script deploys custom models to a running Ollama instance using template files.
Part of the modular workflow: Uses shared Ollama instance, deploys models as Ollama models.

Prerequisites:
- Ollama must be running (use: python start_ollama.py)
- WebUI can be started separately (use: python start_webui.py)
"""

import sys
import argparse
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(
        description="Deploy Custom Model to Running Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_custom_assistant.py templates/yoga_sequence_system_prompt.template.json
  python start_custom_assistant.py templates/qwen_churn_system_prompt.template.json --model qwen2.5:14b-instruct
  python start_custom_assistant.py templates/my_template.json --name my-assistant
  python start_custom_assistant.py templates/yoga_sequence_system_prompt.template.json --list
  python start_custom_assistant.py templates/yoga_sequence_system_prompt.template.json --remove

New Modular Workflow:
  1. python start_ollama.py              # Start Ollama first
  2. python start_custom_assistant.py templates/your_template.json  # Deploy custom model
  3. python start_webui.py               # Start WebUI separately

Prerequisites:
  - Ollama must be running (check with: python start_ollama.py --status)
        """
    )
    parser.add_argument("template", help="Path to template JSON file")
    parser.add_argument("--model", help="Override base model name (e.g., 'qwen2.5:14b-instruct')")
    parser.add_argument("--name", help="Override custom model name")
    parser.add_argument("--list", action="store_true", 
                       help="List all custom models created from this template")
    parser.add_argument("--remove", action="store_true", 
                       help="Remove custom model created from this template")
    parser.add_argument("--test", action="store_true", 
                       help="Test the custom model after deployment")
    
    args = parser.parse_args()
    
    template_file = Path(args.template)
    if not template_file.exists():
        print(f"❌ Template file not found: {args.template}")
        return 1
    
    try:
        # Import required modules
        from ollama_manager import OllamaManager
        from utility_manager import UtilityManager
        import json
        
        # Load template
        template_file = Path(args.template)
        with open(template_file, 'r', encoding='utf-8') as f:
            template_data = json.load(f)
        
        print(f"📋 Template: {template_data['name']}")
        print(f"📝 Description: {template_data['description']}")
        print()
        
        # Check if Ollama is running
        ollama_manager = OllamaManager()
        is_running, status_msg = ollama_manager.get_api_status()
        
        if not is_running:
            print("❌ Ollama is not running!")
            print("💡 Start Ollama first: python start_ollama.py")
            return 1
        
        print(f"✅ Ollama is running: {status_msg}")
        
        # Determine model names
        base_model = args.model or "qwen2.5:7b-instruct"
        
        # Create custom model name
        if args.name:
            custom_model_name = args.name
        else:
            # Use model_name from template if available, otherwise generate from name field
            if 'model_name' in template_data and template_data['model_name']:
                short_name = template_data['model_name']
            else:
                # Extract meaningful words from template name, excluding common filler words
                name_parts = template_data['name'].lower().split()
                filler_words = {'system', 'prompt', 'assistant', 'generator', 'the', 'a', 'an', 'and', 'or', 'for', 'with', 'using'}
                meaningful_words = [word for word in name_parts if word not in filler_words]
                
                # Take first 2-3 meaningful words to keep name short but descriptive
                short_name = '-'.join(meaningful_words[:3]) if meaningful_words else 'custom-assistant'
            
            custom_model_name = f"{base_model.split(':')[0]}:{short_name}"
        
        if args.list:
            print(f"\n📋 Models related to template '{template_data['name']}':")
            models = ollama_manager.list_models()
            template_models = [m for m in models if template_data['name'].lower().replace(' ', '-') in m]
            if template_models:
                for model in template_models:
                    print(f"   • {model}")
            else:
                print("   No custom models found for this template")
            return 0
        
        if args.remove:
            print(f"🗑️  Removing custom model: {custom_model_name}")
            try:
                result = UtilityManager.run_subprocess(
                    f"docker exec ollama ollama rm {custom_model_name}",
                    check=False
                )
                if result.returncode == 0:
                    print(f"✅ Successfully removed {custom_model_name}")
                else:
                    print(f"❌ Failed to remove {custom_model_name}: {result.stderr}")
                return result.returncode
            except Exception as e:
                print(f"❌ Error removing model: {e}")
                return 1
        
        # Deploy custom model
        print(f"🤖 Base model: {base_model}")
        print(f"🎯 Custom model: {custom_model_name}")
        print()
        
        # Check if base model exists
        models = ollama_manager.list_models()
        if base_model not in models:
            print(f"⬇️  Base model {base_model} not found, pulling...")
            try:
                result = UtilityManager.run_subprocess(
                    f"docker exec ollama ollama pull {base_model}",
                    check=False,
                    timeout=600
                )
                if result.returncode != 0:
                    print(f"❌ Failed to pull base model: {result.stderr}")
                    return 1
                print(f"✅ Successfully pulled {base_model}")
            except Exception as e:
                print(f"❌ Error pulling base model: {e}")
                return 1
        
        # Create custom model
        print(f"� Creating custom model...")
        
        success = ollama_manager.create_custom_model(
            base_model=base_model,
            custom_model_name=custom_model_name,
            modelfile_content=f'''FROM {base_model}

SYSTEM """{template_data['system_prompt']}"""
'''
        )
        
        if success:
            print(f"✅ Successfully created custom model: {custom_model_name}")
            
            if args.test:
                print(f"\n🧪 Testing custom model...")
                try:
                    result = UtilityManager.run_subprocess(
                        f'docker exec ollama ollama run {custom_model_name} "Hello, please introduce yourself briefly."',
                        check=False,
                        timeout=60
                    )
                    if result.returncode == 0:
                        print(f"✅ Model test successful")
                        print(f"📝 Response: {result.stdout.strip()}")
                    else:
                        print(f"❌ Model test failed: {result.stderr}")
                except Exception as e:
                    print(f"❌ Error testing model: {e}")
            
            print()
            print("� Custom model deployment complete!")
            print()
            print("💡 Next steps:")
            print("   1. Start WebUI: python start_webui.py")
            print(f"   2. Select model '{custom_model_name}' in the WebUI")
            print("   3. Start chatting with your custom assistant!")
            
            return 0
        else:
            print(f"❌ Failed to create custom model")
            return 1
                
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())