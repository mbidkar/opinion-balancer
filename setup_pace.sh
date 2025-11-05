#!/bin/bash
# Setup OpinionBalancer on PACE-ICE with GPU-enabled GPT-2

echo "🚀 Setting up OpinionBalancer on PACE-ICE (GPU Edition)"
echo "======================================================"

# Load required modules
echo "📦 Loading required modules..."
module load anaconda3
module load cuda/12.1  # Load CUDA for GPU support

# Remove existing environment if it exists
echo "🧹 Cleaning up any existing environment..."
conda env remove -n opinion-balancer-gpu -y || true

# Create GPU-enabled conda environment from yml file
echo "🐍 Creating GPU-enabled conda environment..."
conda env create -f env-gpu.yml

# Check if environment was created successfully
if conda env list | grep -q "opinion-balancer-gpu"; then
    echo "✅ Environment created successfully"
else
    echo "❌ Environment creation failed"
    exit 1
fi

# Activate the environment
echo "🔧 Activating opinion-balancer-gpu environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate opinion-balancer-gpu

# Verify activation
if [[ $CONDA_DEFAULT_ENV == "opinion-balancer-gpu" ]]; then
    echo "✅ Environment activated successfully"
else
    echo "❌ Environment activation failed"
    exit 1
fi

# Verify PyTorch installation
echo "🔍 Verifying PyTorch installation..."
if python -c "import torch" 2>/dev/null; then
    echo "✅ PyTorch installed successfully"
else
    echo "❌ PyTorch not found - installing manually..."
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
fi

# Install any missing packages via pip as fallback
echo "📦 Installing any missing packages..."
pip install transformers>=4.35 huggingface-hub tokenizers accelerate

# Verify GPU setup
echo "🖥️  Verifying GPU setup..."
python -c "
try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA version: {torch.version.cuda}')
        print(f'GPU device: {torch.cuda.get_device_name(0)}')
        print(f'GPU count: {torch.cuda.device_count()}')
    else:
        print('⚠️  GPU not available - will use CPU')
except ImportError as e:
    print(f'❌ PyTorch import error: {e}')
"

# Test GPT-2 model access
echo "🧪 Testing GPT-2 model access..."
python -c "
import os
model_path = '/storage/data/mod-huggingface-0/openai-community__gpt2-medium/models--openai-community--gpt2-medium/snapshots/6dcaa7a952f72f9298047fd5137cd6e4f05f41da'
print(f'Model path exists: {os.path.exists(model_path)}')
if os.path.exists(model_path):
    print('✅ GPT-2 model accessible')
else:
    print('❌ GPT-2 model not found')

# Test transformers import
try:
    from transformers import GPT2Tokenizer
    print('✅ Transformers library available')
except ImportError as e:
    print(f'❌ Transformers import error: {e}')
"

# Test the LLM client
echo "🔍 Testing LLM client..."
python llm_client_gpt2.py

echo "✅ GPU Setup complete!"
echo ""
echo "To use OpinionBalancer:"
echo "1. conda activate opinion-balancer-gpu"
echo "2. python run.py --test"
echo "3. python run.py --topic 'Your topic here'"
echo ""
echo "📊 To monitor GPU usage during runs:"
echo "   nvidia-smi"
