This project provides a **fully local AI infrastructure** with two specialized modes:

## 🎯 **Dual-Mode AI Platform**

### 1. **Development Assistant Mode** 💻
- 💬 Inline prompt support in VS Code via **Continue**
- ⚙️ High-performance LLMs served by **llama.cpp** standalone server
- � CUDA-enabled inference engine for optimal performance
- 🔐 Secure remote access with **Tailscale**

### 2. **Business Analytics Mode** 📊 **NEW!**
- 🔍 **Qwen Churn Assistant**: Specialized customer churn analysis
- 📈 Natural language data analysis (no coding required)
- 🎯 Business-focused insights and recommendations
- 🤖 Powered by Qwen2.5-Coder with specialized system prompts

## 🚀 Key Features

- **Private & Secure**: All computation happens locally on your hardware
- **Template-Driven Architecture**: Modular, maintainable configuration system
- **Multi-Model Support**: Code generation, chat, and specialized business analytics
- **Multi-Platform Access**: Browser, IDE, or mobile (via Tailscale)
- **GPU/CPU Flexibility**: Optimized for both GPU acceleration and CPU-only mode
- **Business Intelligence**: Transform raw data into actionable insights

---

## 🏗️ System Architecture

### Development Assistant Mode
```
[freedom (VS Code + Continue)] <---> [last-nuc (llama.cpp server)]
                 ↖
               Mobile (via Tailscale)
```

### Business Analytics Mode
```
[Business User] <---> [Open WebUI] <---> [Qwen Churn Assistant]
                            ↓
                    [Customer Data CSV] ---> [Natural Language Insights]
```

- **freedom**: HP EliteBook G9, Windows 11 Pro, 32GB RAM
- **last-nuc**: Intel NUC13RNGi9, RTX 3060 Ti, 64GB RAM

---

## 📦 Tech Stack

| Component     | Tool                                                     | Description                              |
| ------------- | -------------------------------------------------------- | ---------------------------------------- |
| **Development Mode** |                                                  |                                          |
| LLM Inference | [`llama.cpp`](https://github.com/ggerganov/llama.cpp)   | CUDA-enabled standalone inference server for GGUF models |
| Primary Model | [`DeepSeek Coder V2 Lite`](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct) | 15.7B parameter coding specialist model |
| VSCode Plugin | [`Continue`](https://continue.dev)                       | Inline chat & refactor tools             |
| **Business Analytics Mode** |                                          |                                          |
| Analytics Engine | [`Qwen2.5-Coder`](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct) | 32B/7B parameter business analysis specialist |
| System Prompts | Template-driven business intelligence prompts            | Specialized for churn analysis and business insights |
| **Shared Infrastructure** |                                              |                                          |
| Web Interface | [`Open WebUI`](https://github.com/open-webui/open-webui) | OpenAI-compatible UI & proxy             |
| Containerization | [`Docker Compose`](https://docs.docker.com/compose/)  | Template-based container orchestration    |
| VPN Layer     | [`Tailscale`](https://tailscale.com)                     | Encrypted access across devices          |
| Architecture  | **Template-Driven Design**                               | Modular, maintainable configuration system |

---

## 📂 Project Structure

This repository uses a **template-driven architecture** for maintainable, modular configuration:

### 🚀 **Entry Points**
- **`start_llama_server.py`**: llama.cpp server mode (CUDA-enabled, standalone inference engine)
- **`start_qwen_churn_assistant.py`**: Business analytics mode (churn analysis, natural language)

### ⚙️ **Core Management Modules**
- **`utility_manager.py`**: Centralized subprocess operations and error handling
- **`llama_server_manager.py`**: llama.cpp server operations, model management, and enhanced health checks
- **`ollama_manager.py`**: Ollama service management and model pulling
- **`webui_manager.py`**: Open WebUI health checking and management
- **`qwen_churn_assistant_manager.py`**: Specialized business analytics infrastructure management
- **`network_manager.py`**: Network utilities and Docker Compose management

### 🏗️ **Modular Architecture Benefits**
- **🔧 Single Responsibility**: Each manager handles one specific service
- **🧪 Testable**: Individual components can be tested in isolation
- **📖 Readable**: Code organized by functionality, not mixed concerns
- **♻️ Reusable**: Managers can be imported and used in other projects
- **⚙️ Self-Contained**: Each manager contains its own configuration
- **🔄 Maintainable**: Modify specific components without affecting others

### 📋 **Template System** 
```
templates/
├── 🤖 AI Model Configuration
│   └── qwen_churn_system_prompt.template.json   # Business intelligence system prompt
└── 🛠️ Open WebUI Integration
    └── open-webui_tools.py                     # CSV analysis tools for Open WebUI
```

### 📁 **Runtime Directories**
- **`.ollama/`**: Ollama data and models (auto-created, gitignored)
- **`.llama/`**: Llama.cpp models (auto-created, gitignored)
- **`.datasets/`**: Analysis workspace (persistent, user data)

### 📚 **Documentation**
- **`readme.md`**: This comprehensive guide (single source of truth)

### ⚙️ **Configuration Files**
- **`docker-compose.ollama.yml`**: Base Ollama service (CPU-optimized)
- **`docker-compose.webui.yml`**: Base Open WebUI service
- **`docker-compose.llama.yml`**: Base Llama.cpp server
- **`docker-compose.gpu-override.yml`**: GPU acceleration overlay (Ollama + WebUI)
- **`docker-compose.llama-gpu-override.yml`**: GPU acceleration for llama.cpp server
- **`docker-compose.qwen-churn-override.yml`**: Qwen Churn Assistant specialization
- **`.gitignore`**: Excludes models, runtime files, preserves templates

---

## 📋 Prerequisites

### Required Software
- **Docker Desktop**: For running containerized services
- **Python 3.7+**: For running the orchestration script
- **Git**: For cloning the repository

### Python Dependencies
Install required packages:
```bash
pip install docker requests
```

### Hardware Requirements
- **GPU Mode**: NVIDIA GPU with 8GB+ VRAM (recommended for Q4_K_M and higher)
- **CPU Mode**: 8GB+ RAM (16GB+ recommended for Q4_K_M, 8GB sufficient for Q2_K)
- **Storage**: 6-17GB free space (depending on model quantization)
  - Q2_K: ~6.4GB (CPU mode default)
  - Q4_K_M: ~10.4GB (GPU mode default)
  - Q6_K: ~14.1GB (high quality)
  - Q8_0: ~16.7GB (maximum quality)

### Model Quantization Guide
The system uses **DeepSeek Coder V2 Lite** with various quantization levels:

| Quantization | Size   | Quality | GPU Recommendation | CPU Recommendation |
|-------------|--------|---------|-------------------|-------------------|
| Q8_0        | 16.7GB | Extreme | 24GB+ VRAM        | 32GB+ RAM         |
| Q6_K        | 14.1GB | Very High | 16GB+ VRAM      | 24GB+ RAM         |
| Q5_K_M      | 11.9GB | High    | 12GB+ VRAM        | 16GB+ RAM         |
| **Q4_K_M**  | 10.4GB | **Good (Default)** | **8GB+ VRAM** | **16GB+ RAM** |
| Q4_K_S      | 9.53GB | Good    | 8GB VRAM          | 12GB+ RAM         |
| IQ4_XS      | 8.57GB | Decent  | 6-8GB VRAM        | 12GB RAM          |
| Q3_K_L      | 8.45GB | Lower   | 6GB VRAM          | 10GB RAM          |
| Q3_K_M      | 8.12GB | Lower   | 4-6GB VRAM        | 8-10GB RAM        |
| **Q2_K**    | 6.43GB | **Usable (CPU Default)** | **4GB VRAM** | **8GB+ RAM** |

✅ **Tested Configuration**: Q2_K quantization successfully tested on WSL2 with 12 CPUs and 15.44GB RAM in CPU-only mode.

---

## 🧪 **How to Use** - Development Assistant Mode

### 1. **Clone the Repository**
```bash
git clone https://github.com/mousou2003/AI-Server
cd AI-Server
```

### 2. **Verify Setup (Recommended)**
```bash
# Test basic functionality
python start_qwen_churn_assistant.py --help
python start_llama_server.py --help

# Verify Docker is running
docker --version
```

### 3. **Start llama.cpp Server**

#### GPU Mode (Recommended - CUDA-enabled)
```bash
python start_llama_server.py
```

#### CPU-Only Mode (No GPU Required)
```bash
python start_llama_server.py --cpu-only
```

**Quick Examples:**
- `python start_llama_server.py --auto` - GPU mode with default Q4_K_M
- `python start_llama_server.py --cpu-only --auto` - CPU mode with default Q2_K
- `python start_llama_server.py --cpu-only -q Q3_K_M` - CPU mode with better quality
- `python start_llama_server.py --list-quants` - Show all available quantization options
- `python start_llama_server.py --list-models` - List existing models in your .llama folder
- `python start_llama_server.py --cleanup` - Stop all containers after testing

This script will automatically:
- ✅ Download the DeepSeek Coder V2 Lite model if needed (~6-17GB depending on quantization)
- 🚀 Start llama.cpp server with CUDA support and the selected model/quantization
- 🐳 Launch Docker container using `server-cuda` image with enhanced health checks
- 🧪 Test API endpoints with comprehensive readiness verification
- 🎯 Handle quantization selection interactively or via command-line options
- ⚙️ Configure CPU-only or GPU-accelerated mode automatically

**Available Services:**
- **llama.cpp server**: High-performance inference engine on port 11435 (CUDA-enabled)
- **API Endpoints**: OpenAI-compatible REST API for integration

The script uses a streamlined Docker Compose architecture:
- **Base Service**: `docker-compose.llama.yml` with server-cuda image
- **GPU Acceleration**: `docker-compose.llama-gpu-override.yml` adds GPU runtime support
- **Health Monitoring**: Comprehensive health checks at both container and API level

**CPU vs GPU Mode:**
- **GPU Mode**: Faster inference (~50+ tokens/sec), uses CUDA-enabled server image
- **CPU Mode**: Slower inference (~20 tokens/sec), works on any system with adequate RAM

### Advanced Options

```bash
# Interactive quantization selection
python start_llama_server.py --cpu-only

# List all available quantization options
python start_llama_server.py --list-quants

# List existing models in your .llama directory
python start_llama_server.py --list-models

# Use specific quantization
python start_llama_server.py --cpu-only -q Q3_K_M

# Test deployment and cleanup afterwards
python start_llama_server.py --cleanup
```

### 4. **Start Business Analytics (Alternative Mode)**
```bash
# Quick business analytics setup
python start_qwen_churn_assistant.py --cpu     # CPU mode (7B model)
python start_qwen_churn_assistant.py           # GPU mode (32B model)

# Then visit http://localhost:3000 and upload your CSV files
```

### 5. **Configure VS Code (Development Mode)**

Install the Continue extension for VS Code: https://www.continue.dev/

Create or update your Continue configuration file at `~/.continue/config.yaml`:

```yaml
name: Local Assistant
version: 1.0.0
schema: v1
models:
  # Primary model: DeepSeek Coder V2 Lite via standalone llama.cpp server
  - name: deepseek-coder-v2-lite
    provider: openai
    model: DeepSeek-Coder-V2-Lite-Instruct-Q2_K.gguf  # Adjust based on your quantization
    apiBase: http://localhost:11435/v1  # Standalone llama.cpp server
    apiKey: fake  # Required but not used
    roles:
      - chat
      - edit
      - apply
      - autocomplete
    defaultCompletionOptions:
      contextLength: 163840  # 160K context window
      maxTokens: 8192

context:
  - provider: code
  - provider: docs
  - provider: diff
  - provider: terminal
  - provider: problems
  - provider: folder
  - provider: codebase
```

**For remote access via Tailscale**, replace `localhost` with your Tailscale hostname:
```yaml
apiBase: http://your-tailscale-hostname.ts.net:11435/v1  # For standalone llama.cpp server
```

---

## 🏗️ **Template-Driven Architecture**

This project pioneered a **template-based approach** to AI infrastructure deployment, providing several key advantages:

### 🎯 **Benefits**
- **� Maintainable**: All configuration in version-controlled templates
- **🚀 Scalable**: Easy to add new AI modes and configurations  
- **🧹 Clean**: Clear separation between templates and runtime
- **⚙️ Flexible**: Support for different hardware modes (CPU/GPU)
- **📦 Portable**: Templates work across different environments

### 📋 **Template Categories**

| Template Type | Purpose | Example |
|---------------|---------|---------|
| **🤖 AI Configuration** | System prompts, model parameters | `qwen_churn_system_prompt.template.json` |
| **�️ Tool Integration** | WebUI tools and utilities | `open-webui_tools.py` |

### 🔄 **Template Processing Flow**
```
Template Files (version controlled)
    ↓ (read by managers)
Runtime Environment (active configuration)
    ↓ (cleanup on exit)
Clean State (templates preserved)
```

### 🎨 **Customization**
Want to create your own AI assistant mode? Simply:
1. Create new templates in `templates/`
2. Add template variable substitution in the manager
3. Create a new entry point script
4. Templates automatically handle CPU/GPU modes and standard Open WebUI integration

---

## 🔍 **Verification & Testing**

### **Quick Validation**
```bash
# Verify Docker is running
docker --version

# Check available disk space (models are large)
dir .llama
dir .ollama

# Test basic functionality
python start_qwen_churn_assistant.py --help
```

### **llama.cpp Server Testing**
```bash
# Start standalone llama.cpp server
python start_llama_server.py --cpu-only --auto

# Verify endpoints
curl http://localhost:11435/v1/models     # llama.cpp API
curl http://localhost:11435/health        # Health check endpoint

# Test VS Code integration with Continue extension
```

### **Business Analytics Mode Testing**
```bash
# Start business analytics
python start_qwen_churn_assistant.py --cpu

# Verify services  
python start_qwen_churn_assistant.py --status

# Test WebUI at http://localhost:3000
# Upload a CSV file and ask: "What patterns do you see in this data?"
```

### **Performance Verification**
Recent deployment test results:

**llama.cpp Server Mode** (DeepSeek Coder V2 Lite):
- ✅ **CPU Mode**: Q2_K quantization at ~20 tokens/second
- ✅ **GPU Mode**: CUDA-enabled server-cuda image for optimal performance
- ✅ **Model Size**: 6.4GB loaded successfully  
- ✅ **Context Length**: 163,840 tokens (160K context window)
- ✅ **Parameters**: 15.7B parameters with enhanced health checks
- ✅ **Health Monitoring**: Multi-endpoint validation and container health checks

**Business Analytics Mode** (Qwen2.5-Coder):
- ✅ **CPU Mode**: 7B model at ~15-20 tokens/second
- ✅ **GPU Mode**: 32B model at ~30-50 tokens/second  
- ✅ **Memory System**: Persistent conversation memory
- ✅ **Workspace**: CSV file analysis capabilities

### Remote Access
For remote access via Tailscale, replace `localhost` with your Tailscale hostname:
- llama.cpp API: `http://your-tailscale-hostname.ts.net:11435`

---

## 📊 **Qwen Churn Assistant** - Business Analytics Mode

**🎯 Purpose**: Transform customer data into actionable business insights through natural language conversations.

The Qwen Churn Assistant is a specialized business intelligence system that analyzes customer churn data without requiring any coding skills. Simply upload your CSV files and ask questions in plain English!

### � **Key Features**
- **🗣️ Natural Language Interface**: Ask business questions in plain English
- **📈 Zero-Code Analysis**: No Python, SQL, or technical skills required
- **🤖 Specialized AI**: Qwen2.5-Coder optimized with business intelligence prompts
- **🔒 Local & Private**: All analysis happens on your hardware
- **💡 Actionable Insights**: Get practical recommendations, not just statistics
- **📝 Memory-Enabled**: Remembers context across analysis sessions
- **📁 Workspace Integration**: Upload and analyze CSV files directly
- **🛠️ Built-in Tools**: Custom CSV analysis tools integrated with Open WebUI

### 🚀 **Quick Start**
```bash
# GPU mode (recommended - uses 32B model for comprehensive analysis)
python start_qwen_churn_assistant.py

# CPU mode (uses optimized 7B model - much faster on CPU)
python start_qwen_churn_assistant.py --cpu

# CPU mode with large model (32B - very slow but most comprehensive)
python start_qwen_churn_assistant.py --cpu --large-model

# Management commands
python start_qwen_churn_assistant.py --status    # Check status
python start_qwen_churn_assistant.py --stop      # Stop infrastructure  
python start_qwen_churn_assistant.py --logs      # View container logs
python start_qwen_churn_assistant.py --cleanup   # Clean temporary files
```

### �️ **Open WebUI Tools Integration**

The system includes custom Python tools that extend Open WebUI's capabilities for CSV data analysis:

**Available Tools:**
- **`analyze_wine_club_csv`**: Direct CSV upload analysis for business insights
- **`load_raw_csv_from_storage`**: Preview CSV files from Open WebUI storage  
- **`analyze_wine_club_from_knowledge`**: Advanced analysis of stored CSV files
- **File Resolution**: Smart file matching from partial filenames

**Tool Features:**
- **Automatic File Discovery**: Upload CSV files and reference them by partial name
- **Data Sampling**: Configurable row limits for large datasets
- **Business Context**: Tools designed for business analysis, not technical processing
- **Error Handling**: Comprehensive error messages and fallback behaviors
- **Storage Integration**: Works with Open WebUI's persistent file storage

**Usage Example:**
```
1. Upload a CSV file named "customer-churn-data.csv" to Open WebUI
2. Ask: "Use the analyze_wine_club_from_knowledge tool to analyze churn-data.csv and tell me about customer retention patterns"
3. The tool will automatically find and analyze your uploaded file
```

**Tool Installation:**
The tools are automatically available when using the Qwen Churn Assistant mode. They're located in `templates/open-webui_tools.py` and integrate seamlessly with the business analytics workflow.

### �💬 **Example Business Questions**
Once your CSV data is uploaded, ask questions like:

**Initial Data Exploration:**
- *"Can you analyze this customer churn data and give me an overview?"*
- *"What's the overall churn rate in this dataset?"*
- *"Which customer segments are represented?"*

**Churn Pattern Analysis:**
- *"Which customer segments have the highest churn rate?"*
- *"What factors seem to correlate with customer churn?"*
- *"Are there critical thresholds that predict churn risk?"*

**Risk Assessment:**
- *"Which current customers have the highest churn risk?"*
- *"What early warning signs should we watch for?"*
- *"How does churn risk vary by customer demographics?"*

**Business Strategy:**
- *"What retention strategies would you recommend?"*
- *"Which customers should we prioritize for intervention?"*
- *"How can we reduce churn in high-risk segments?"*

### 🎯 **What Makes It Special**
- **Business-Focused**: Responses avoid technical jargon and focus on actionable insights
- **Template-Driven**: Uses specialized system prompts for consistent business analysis
- **Standard Architecture**: Proper Open WebUI + Ollama integration (models auto-discovered)
- **Memory & Workspace**: Persistent conversations and file analysis via Docker volumes
- **CPU/GPU Optimized**: Automatically selects appropriate model size for your hardware
- **Integrated Tools**: Custom CSV analysis tools built into Open WebUI
- **File Management**: Smart file resolution and storage integration

### 🏗️ **Template Architecture**
The system uses a sophisticated template-driven approach:

```
User Question → Business AI (Qwen + Custom Prompt) → Actionable Insight
     ↑                           ↓
CSV Upload ←→ WebUI Tools ←→ Workspace ←→ Memory System
```

**Template Components:**
- **System Prompt**: `templates/qwen_churn_system_prompt.template.json`
- **CSV Tools**: `templates/open-webui_tools.py`
- **WebUI Integration**: Standard Ollama connection (auto-discovery)

### � **Data Requirements**
**Supported Formats:** CSV files with customer records

**Upload Methods:**
- **Direct Upload**: Use Open WebUI's file upload feature
- **Tool Integration**: Files are automatically processed by built-in CSV analysis tools
- **Partial Naming**: Reference files by partial names (e.g., "churn-data.csv" matches "customer-churn-data.csv")

**Recommended Structure:**
- **Customer ID**: Unique identifier
- **Churn Status**: Binary indicator (churned/retained)  
- **Demographics**: Age, location, segment, etc.
- **Behavioral Data**: Usage patterns, engagement metrics
- **Account Info**: Tenure, value, payment history
- **Interaction Data**: Support contacts, satisfaction scores

**Privacy:** All processing happens locally - no data sent to external services.

### 🚨 **Troubleshooting**
```bash
# Check system status
python start_qwen_churn_assistant.py --status

# View container logs for debugging
python start_qwen_churn_assistant.py --logs

# Restart if needed
python start_qwen_churn_assistant.py --stop
python start_qwen_churn_assistant.py --cpu

# Clean up temporary files only
python start_qwen_churn_assistant.py --cleanup

# Complete cleanup (removes ALL data including Docker volumes)
python start_qwen_churn_assistant.py --cleanup-all
```

**Common Issues:**
- **Memory issues**: Use `--cpu` mode or close other applications
- **Model download fails**: Check internet connection and disk space
- **Containers won't start**: Ensure Docker is running and has sufficient resources
- **Persistent volume issues**: Use `--cleanup-all` to remove all Docker volumes and start fresh

**Cleanup Options:**
- **`--cleanup-all`**: Removes everything including Docker volumes (⚠️ **WARNING**: deletes all conversation history and WebUI settings)

### 💡 **Best Practices**
- **Ask specific questions**: Reference particular segments or behaviors
- **Build on insights**: Use follow-up questions to dive deeper  
- **Provide context**: Explain your business model and customer segments
- **Clean your data**: Remove duplicates and handle missing values before upload

### 🧪 **Comprehensive Testing Guide**

This testing suite ensures your custom `qwen2.5-coder:7b-churn` model is properly configured with the specialized churn analysis system prompt and behaves according to business-focused constraints.

#### **Prerequisites**
1. **Verify Model Availability**
   ```bash
   docker exec ollama-qwen-churn ollama list
   ```
   - Confirm `qwen2.5-coder:7b-churn` appears in the list
   - Note the creation/modification date

2. **Access Web UI**
   - Open http://localhost:3000
   - Select the `qwen2.5-coder:7b-churn` model from the dropdown
   - Ensure you're testing the custom model, not the base model

#### **Test Suite**

**Test 1: System Prompt Integration & Role Recognition**

*Test Prompt:*
```
Hello! I have a customer dataset and I'm concerned about churn. Can you help me understand what I should be looking for?
```

*Expected Response Characteristics:*
- ✅ Should identify itself as a "specialized churn analysis assistant"
- ✅ Should focus on business insights and patterns
- ✅ Should ask business-focused clarifying questions about:
  - Industry/business type
  - Customer segments
  - Available data fields
  - Current business concerns
- ✅ Should mention analyzing CSV files through conversation
- ❌ Should NOT offer to write code or provide technical implementation

**Test 2: Code Generation Constraint Adherence**

*Test Prompt:*
```
Can you write Python code to analyze my customer churn data?
```

*Expected Response:*
- ✅ Should politely decline to write code
- ✅ Should explain the conversational analysis approach
- ✅ Should redirect to discussing data patterns through dialogue
- ✅ Should maintain helpful, business-focused tone
- ❌ Should NOT provide any Python, SQL, or programming syntax
- ❌ Should NOT suggest technical tools or libraries

**Test 3: Business Focus & Analysis Framework**

*Test Prompt:*
```
I have customer data with columns like customer_id, tenure, monthly_charges, total_charges, contract_type, and churn_status. What insights can you provide?
```

*Expected Response Structure:*
1. **Key Finding**: Should identify important business patterns from the described fields
2. **Supporting Evidence**: Should explain what these fields typically reveal (in business terms)
3. **Business Implication**: Should discuss what patterns mean for the company
4. **Recommended Action**: Should suggest specific, actionable next steps
5. **Follow-up Questions**: Should ask about business context and goals

**Test 4: Technical Implementation Avoidance**

*Test Prompt:*
```
Show me SQL queries to find churned customers with high monthly charges.
```

*Expected Response:*
- ✅ Should decline to provide SQL queries
- ✅ Should offer to discuss the business question behind the request
- ✅ Should ask about the business goal (why high-charge churned customers matter)
- ✅ Should suggest conversational analysis of this customer segment
- ❌ Should NOT provide any SQL syntax or database queries

**Test 5: Statistical Formula Constraint**

*Test Prompt:*
```
Calculate the churn rate formula and show me the statistical significance tests I should run.
```

*Expected Response:*
- ✅ Should explain churn rate in simple business terms (customers lost vs total customers)
- ✅ Should avoid mathematical formulas and statistical notation
- ✅ Should focus on what churn rate means for business decisions
- ✅ Should ask about business thresholds and concerns
- ❌ Should NOT provide mathematical formulas or equations
- ❌ Should NOT mention specific statistical tests or procedures

**Test 6: Business Strategy Focus**

*Test Prompt:*
```
I notice customers with month-to-month contracts have higher churn rates. What should I do about this?
```

*Expected Response Elements:*
- **Business Analysis**: Should discuss why month-to-month customers might churn more
- **Segment Understanding**: Should ask about different customer types and their needs
- **Retention Strategies**: Should suggest practical business actions like:
  - Contract incentives
  - Customer engagement programs
  - Early warning systems
  - Targeted communication
- **Risk Assessment**: Should help identify which month-to-month customers are highest risk
- **Business Impact**: Should discuss revenue implications

#### **Comparison Test with Base Model**

To verify your customization is working, also test the base model `qwen2.5-coder:7b`:

*Test Prompt:*
```
I have customer churn data and need help analyzing it. What's the best approach?
```

**Base Model Expected Behavior:**
- May offer to write code or provide technical solutions
- Might suggest specific tools, libraries, or programming approaches
- Could provide more general-purpose analysis suggestions

**Custom Model Expected Behavior:**
- Should focus on business conversation and insights
- Should ask about business context and goals
- Should avoid technical implementation suggestions

#### **Quick Validation Checklist**

✅ Model responds as churn analysis expert  
✅ Refuses code generation requests  
✅ Uses business language, not technical terms  
✅ Follows 5-step response format  
✅ Asks business-focused follow-up questions  
✅ Provides actionable recommendations  
✅ Avoids statistical formulas and technical details  
✅ Maintains conversational, helpful tone  

#### **Test Results Template**

```
Date: ___________
Model: qwen2.5-coder:7b-churn

Test 1 - System Prompt Integration: ✅/❌
Test 2 - Code Generation Constraint: ✅/❌  
Test 3 - Business Focus: ✅/❌
Test 4 - Technical Implementation Avoidance: ✅/❌
Test 5 - Statistical Formula Constraint: ✅/❌
Test 6 - Business Strategy Focus: ✅/❌

Overall Assessment: ✅ PASS / ❌ FAIL

Notes:
_________________________________
_________________________________
```

#### **Troubleshooting Test Failures**

If tests fail, check:
1. **Model Selection**: Ensure you're using `qwen2.5-coder:7b-churn`, not the base model
2. **Model Creation**: Verify the custom model was created successfully
3. **Modelfile Content**: Check that the system prompt was properly integrated
4. **Container Status**: Ensure Ollama container is running and responsive

**Remember**: The goal is to ensure your custom model behaves as a business consultant, not a technical developer, when discussing customer churn analysis.

---

## 🏗️ Code Architecture

The project uses a **modular architecture** with separated concerns:

- **Main Scripts**: 
  - `start_llama_server.py`: Standalone llama.cpp server with CUDA support
  - `start_qwen_churn_assistant.py`: Specialized business analytics mode
- **Service Managers**: Individual managers for each service component
  - `llama_server_manager.py`: llama.cpp operations, model management, and enhanced health checks
  - `ollama_manager.py`: Ollama service and model pulling
  - `webui_manager.py`: Open WebUI health checks
  - `utility_manager.py`: Subprocess and utility functions
  - `network_manager.py`: Network utilities and Docker Compose management

All components follow the template-driven architecture with clear separation of concerns.

---

## 🙌 Acknowledgments

This **template-driven AI infrastructure platform** builds on the work of:

**Core AI Infrastructure:**
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - High-performance LLM inference
- [Open WebUI](https://github.com/open-webui/open-webui) - Modern web interface with memory/workspace
- [Ollama](https://ollama.ai) - Model backend and management
- [Docker](https://docker.com) - Containerized deployment system

**AI Models:**
- [DeepSeek](https://huggingface.co/deepseek-ai) - DeepSeek Coder V2 Lite (Development Mode)
- [Qwen Team](https://huggingface.co/Qwen) - Qwen2.5-Coder (Business Analytics Mode)

**Development Tools:**
- [Continue](https://continue.dev) - VS Code AI assistant extension
- [Tailscale](https://tailscale.com) - Secure remote access

**Architecture Innovation:**
This project introduces a **template-driven approach** to AI infrastructure that makes it easy to create specialized AI assistants for different business domains while maintaining clean, maintainable code.

---

## ✨ Motivation (from LinkedIn post)

> Over the past few days, I’ve been designing and deploying a local, private coding assistant setup — entirely powered by open-source tools and hosted on my own machine...

> It's surprisingly easy now to build your own dev assistant:
>
> - No OpenAI key required
> - No cloud dependencies
> - No latency overhead

Whether you’re an indie dev, engineer, or researcher, this gives you full control over your AI stack.

---

## 📌 Repo

[https://github.com/mousou2003/AI-Server](https://github.com/mousou2003/AI-Server)

