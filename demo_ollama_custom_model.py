#!/usr/bin/env python3
"""
OllamaCustomModel Demo Script

This script demonstrates the key features of the OllamaCustomModel framework,
including template loading, validation, and different ways to create assistants.
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from ollama_custom_model import OllamaCustomModel, TemplateLoader


def demo_template_loading():
    """Demonstrate template loading and validation"""
    print("🔍 Template Loading and Validation Demo")
    print("=" * 50)
    
    # Available templates
    templates = [
        "templates/yoga_sequence_system_prompt.template.json"
    ]
    
    for template_path in templates:
        if Path(template_path).exists():
            try:
                print(f"\n📋 Loading template: {template_path}")
                template_data = TemplateLoader.load_template(template_path)
                
                print(f"   ✅ Name: {template_data['name']}")
                print(f"   📝 Description: {template_data['description'][:100]}...")
                
                # Validate template
                TemplateLoader.validate_template(template_data)
                print(f"   ✅ Template validation passed")
                
                # Show model parameters if present
                if 'model_parameters' in template_data:
                    params = template_data['model_parameters']
                    print(f"   ⚙️  Model parameters: {params}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
        else:
            print(f"\n⚠️  Template not found: {template_path}")


def demo_assistant_creation():
    """Demonstrate different ways to create assistants"""
    print("\n\n🤖 Assistant Creation Demo")
    print("=" * 50)
    
    print("\n1️⃣ Factory Method - Yoga Assistant")
    try:
        yoga_assistant = OllamaCustomModel.create_yoga_assistant(
            cpu_mode=True, 
            quiet_mode=True
        )
        print(f"   ✅ Created: {yoga_assistant.config['template_name']}")
        print(f"   🏗️  Project: {yoga_assistant.project_name}")
        print(f"   🤖 Model: {yoga_assistant.base_model_name}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n2️⃣ Generic Factory Method")
    try:
        yoga_path = "templates/yoga_sequence_system_prompt.template.json"
        if Path(yoga_path).exists():
            generic_assistant = OllamaCustomModel.create_from_template(
                template_path=yoga_path,
                cpu_mode=True,
                quiet_mode=True
            )
            print(f"   ✅ Created: {generic_assistant.config['template_name']}")
            print(f"   🏗️  Project: {generic_assistant.project_name}")
        else:
            print(f"   ⚠️  Template not found: {yoga_path}")
    except Exception as e:
        print(f"   ❌ Error: {e}")


def demo_docker_configuration():
    """Demonstrate Docker configuration generation"""
    print("\n\n🐳 Docker Configuration Demo")
    print("=" * 50)
    
    try:
        # Create an assistant to show Docker setup using yoga template
        yoga_path = "templates/yoga_sequence_system_prompt.template.json"
        if Path(yoga_path).exists():
            assistant = OllamaCustomModel(
                template_path=yoga_path,
                cpu_mode=True,
                quiet_mode=True,
                project_name="demo-assistant"
            )
            
            print(f"📋 Project: {assistant.project_name}")
            print(f"🔧 Base Ollama file: {assistant.base_ollama_file}")
            print(f"🌐 Base WebUI file: {assistant.base_webui_file}")
            print(f"⚡ GPU override file: {assistant.gpu_override_file} (CPU mode: {assistant.cpu_mode})")
            print(f"🎯 Project override file: {assistant.project_override_file}")
            
            # Show compose command
            compose_cmd = assistant.get_compose_command("up", "-d")
            print(f"\n🐳 Docker Compose Command:")
            print(f"   {compose_cmd}")
            
            # Check if project override was created
            if assistant.project_override_file.exists():
                print(f"\n✅ Project override file created: {assistant.project_override_file}")
            else:
                print(f"\n⚠️  Project override file not found: {assistant.project_override_file}")
        else:
            print(f"⚠️  Template not found: {yoga_path}")
            
    except Exception as e:
        print(f"❌ Error: {e}")


def demo_model_inference():
    """Demonstrate model name inference"""
    print("\n\n🧠 Model Inference Demo")
    print("=" * 50)
    
    templates_to_test = [
        ("templates/yoga_sequence_system_prompt.template.json", "Yoga sequence generation")
    ]
    
    for template_path, description in templates_to_test:
        if Path(template_path).exists():
            try:
                print(f"\n📋 Template: {template_path}")
                print(f"📝 Purpose: {description}")
                
                assistant = OllamaCustomModel(
                    template_path=template_path,
                    cpu_mode=True,
                    quiet_mode=True
                )
                
                print(f"🤖 Inferred model: {assistant.base_model_name}")
                print(f"🏗️  Project name: {assistant.project_name}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
        else:
            print(f"\n⚠️  Template not found: {template_path}")


def main():
    """Run all demonstrations"""
    print("🚀 OllamaCustomModel Framework Demo")
    print("=" * 60)
    print("This demo showcases the key features of the generalized")
    print("Ollama Custom Model framework for creating AI assistants.")
    print("=" * 60)
    
    # Run demonstrations
    demo_template_loading()
    demo_assistant_creation()
    demo_docker_configuration()
    demo_model_inference()
    
    print("\n\n🎉 Demo Complete!")
    print("=" * 60)
    print("📚 For more information, see: OLLAMA_CUSTOM_MODEL_README.md")
    print("🚀 To start an assistant: python start_yoga_assistant.py --cpu")
    print("🔧 To create custom templates, see: templates/ directory")


if __name__ == "__main__":
    main()