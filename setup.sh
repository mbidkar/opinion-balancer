#!/bin/bash
# Simple OpinionBalancer Setup for PACE-ICE
# Works for both CPU and GPU environments

set -e  # Exit on any error

echo "🚀 OpinionBalancer Setup for PACE-ICE"
echo "======================================"

# Configuration
PROJECT_DIR="/storage/ice1/shared/ece8803cai/mbidkar3/opinion-balancer"
ENV_NAME="opinion-balancer"

# Clean up first
echo "🧹 Cleaning up..."
conda env remove -n $ENV_NAME -y 2>/dev/null || true
conda clean --all -y
rm -rf ~/.cache/pip ~/.local/lib/python3.* 2>/dev/null || true

# Load modules
echo "📦 Loading modules..."
module load anaconda3 2>/dev/null || echo "Anaconda already loaded"

# Create environment
echo "🐍 Creating conda environment..."
cd $PROJECT_DIR
conda env create -f environment.yml

# Activate environment
echo "🔧 Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Verify installation
echo "✅ Testing installation..."
python -c "
import sys
print(f'Python: {sys.version}')

# Test core packages
packages = ['torch', 'transformers', 'langgraph', 'pydantic', 'yaml', 'pandas', 'numpy']
for pkg in packages:
    try:
        if pkg == 'yaml':
            import yaml
        else:
            __import__(pkg)
        print(f'✅ {pkg}')
    except ImportError as e:
        print(f'❌ {pkg}: {e}')
"

# Test GPT-2 model access
echo "🔍 Checking GPT-2 model..."
python -c "
import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

# Check model paths
model_paths = [
    '/storage/data/mod-huggingface-0/gpt2-medium',
    '/storage/data/mod-huggingface-0/openai-community__gpt2-medium'
]

model_path = None
for path in model_paths:
    if os.path.exists(path):
        model_path = path
        break

if model_path:
    print(f'✅ GPT-2 model found: {model_path}')
    try:
        # Quick test load
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        print('✅ Tokenizer loads successfully')
    except Exception as e:
        print(f'⚠️  Model load test failed: {e}')
else:
    print('❌ GPT-2 model not found - will use online model')
"

# Test LangGraph
echo "🧪 Testing LangGraph..."
python -c "
try:
    from langgraph.graph import StateGraph
    print('✅ LangGraph working')
except ImportError as e:
    print(f'❌ LangGraph error: {e}')
"

echo ""
echo "✅ Setup Complete!"
echo ""
echo "🎯 Next steps:"
echo "1. conda activate $ENV_NAME"
echo "2. python test_gpt2.py"
echo "3. langgraph dev"
echo ""
echo "📊 Environment info:"
echo "Name: $ENV_NAME"
echo "Python: $(python --version 2>&1)"
echo "Location: $(which python)"
