# Dataset Preloading Guide for Ollama Churn Analysis

This guide explains how to preload datasets with your Ollama churn analysis infrastructure for seamless analysis.

## Overview

Your enhanced Docker Compose setup now includes dataset volumes that make CSV files available to both Ollama and Open WebUI containers. This allows for:

- **Pre-loaded datasets** accessible immediately upon startup
- **Persistent storage** of analysis results
- **Easy dataset management** through mounted volumes
- **Seamless file upload** through Open WebUI interface

## Dataset Deployment Options

### Option 1: Volume Mounting (Recommended)

Your `docker-compose.qwen-churn.yml` now includes dataset volume mounts:

```yaml
# Ollama container - read-only access to datasets
volumes:
  - .\workspace\churn_analysis:/data/datasets:ro

# Open WebUI container - read-write access for uploads
volumes:
  - .\workspace\churn_analysis:/app/backend/data/uploads/datasets:rw
```

**Benefits:**
- Datasets persist between container restarts
- Easy to add new datasets by copying files to the host directory
- Separation between read-only and read-write access
- No need to rebuild containers when adding data

### Option 2: Custom Docker Image (Advanced)

For production deployments, you can create a custom image with datasets baked in:

```dockerfile
FROM ghcr.io/open-webui/open-webui:main

# Copy datasets into the image
COPY ./datasets/*.csv /app/backend/data/uploads/datasets/

# Set proper permissions
RUN chown -R root:root /app/backend/data/uploads/datasets/ && \
    chmod 644 /app/backend/data/uploads/datasets/*.csv
```

## Dataset Directory Structure

```
AI-Server/
├── workspace/
│   └── churn_analysis/           # Main datasets directory
│       ├── sample_churn_data.csv # Auto-generated sample
│       ├── customer_data.csv     # Your actual datasets
│       ├── historical_churn.csv
│       └── segment_analysis.csv
├── memory/
│   └── churn.md                  # Analysis memory/context
└── models/
    └── .ollama/                  # Model storage
```

## Deploying Your Datasets

### Step 1: Prepare Your Dataset Directory

```powershell
# Create the datasets directory (auto-created by infrastructure)
mkdir "workspace\churn_analysis" -Force

# Copy your CSV files
Copy-Item "C:\path\to\your\customer_data.csv" "workspace\churn_analysis\"
Copy-Item "C:\path\to\your\historical_data.csv" "workspace\churn_analysis\"
```

### Step 2: Start Infrastructure with Dataset Support

```powershell
# Start the enhanced infrastructure
python start_qwen_churn_assistant.py --cpu

# Or for GPU mode
python start_qwen_churn_assistant.py
```

### Step 3: Verify Dataset Access

1. **Check dataset directory**:
   ```powershell
   ls "workspace\churn_analysis\"
   ```

2. **Verify container access**:
   ```powershell
   docker exec ollama-qwen-churn ls /data/datasets
   docker exec open-webui-qwen-churn ls /app/backend/data/uploads/datasets
   ```

3. **Access through Web UI**:
   - Open http://localhost:3000
   - Navigate to the file upload section
   - Your datasets should be visible in the uploads/datasets directory

## Dataset Requirements

### Recommended CSV Format for Churn Analysis

```csv
customer_id,tenure,monthly_charges,total_charges,contract_type,payment_method,churn_status
CUST001,12,29.85,358.2,Month-to-month,Electronic check,Yes
CUST002,34,56.95,1937.3,One year,Mailed check,No
```

### Essential Columns
- **customer_id**: Unique identifier
- **churn_status**: Yes/No or 1/0 indicating churn
- **tenure**: Length of customer relationship
- **monetary_fields**: Revenue, charges, or value metrics
- **categorical_fields**: Contract type, payment method, segments

### Optional Enrichment Columns
- Demographics (age, gender, location)
- Usage patterns (login frequency, feature usage)
- Support interactions (tickets, calls)
- Product adoption (services used, upgrades)

## Testing Dataset Preloading

### Automatic Sample Generation

Your OllamaManager can create a sample dataset:

```python
from ollama_manager import OllamaManager

manager = OllamaManager()
success = manager.preload_sample_dataset("workspace/churn_analysis", create_sample=True)
```

### Manual Testing

1. **Upload via Web UI**:
   - Go to http://localhost:3000
   - Click the attachment/upload button in chat
   - Upload your CSV file
   - Ask: "Can you analyze this customer churn data?"

2. **Direct File Access**:
   - Place CSV in `workspace/churn_analysis/`
   - Reference in conversation: "I have customer data in the datasets folder"

## Integration with Custom Model

Your `qwen2.5-coder:7b-churn` model is specifically designed to:

- **Recognize dataset references** in conversations
- **Ask business-focused questions** about the data
- **Provide insights without code** generation
- **Guide analysis through conversation**

### Example Conversation Flow

```
User: "I've uploaded our customer churn data. What should I look for?"

Model: "I can see you have customer data available. Let me help you identify key churn patterns. 

To provide the most relevant insights, I'd like to understand:
1. What time period does this data cover?
2. What's your current overall churn rate?
3. Are there specific customer segments you're most concerned about?
4. What's driving your interest in churn analysis right now?

From the data structure, I can help you explore patterns in tenure, contract types, and payment methods that might correlate with churn risk."
```

## Security Considerations

### File Permissions
- Datasets mounted as **read-only** in Ollama container (`:ro`)
- **Read-write access** only in Open WebUI for uploads (`:rw`)
- Host directory permissions should restrict access appropriately

### Data Privacy
- Consider anonymizing customer IDs in datasets
- Remove or hash sensitive personal information
- Use environment variables for sensitive configuration

### Access Control
- Open WebUI configured with admin user for first setup
- Consider disabling signup after initial configuration
- Monitor file upload logs for security

## Troubleshooting

### Dataset Not Visible

1. **Check volume mounts**:
   ```powershell
   docker inspect ollama-qwen-churn | findstr -A 5 -B 5 "Mounts"
   docker inspect open-webui-qwen-churn | findstr -A 5 -B 5 "Mounts"
   ```

2. **Verify file permissions**:
   ```powershell
   icacls "workspace\churn_analysis"
   ```

3. **Check container logs**:
   ```powershell
   docker logs ollama-qwen-churn
   docker logs open-webui-qwen-churn
   ```

### Performance Issues

- **Large datasets**: Consider splitting into smaller files
- **Memory usage**: Monitor container resource consumption
- **File formats**: Ensure CSV files are properly formatted

### Model Behavior

If the model doesn't recognize datasets:
- Verify custom model creation was successful
- Check that `qwen2.5-coder:7b-churn` is selected in WebUI
- Test with explicit dataset references in conversation

## Production Deployment

### Scaling Considerations
- Use external storage volumes for large datasets
- Implement backup strategies for dataset persistence
- Consider read-only replicas for high-availability setups

### Automation
- Script dataset deployment as part of CI/CD
- Automate dataset validation and preprocessing
- Implement dataset versioning for reproducible analysis

---

Your churn analysis infrastructure now supports seamless dataset preloading, making it easy to start analyzing customer data immediately upon deployment!
