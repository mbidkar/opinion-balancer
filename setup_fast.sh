#!/bin/bash
# Fast OpinionBalancer Setup on PACE-ICE

echo "🚀 Fast OpinionBalancer Setup (Manual Installation)"
echo "=================================================="

# Load modules
echo "📦 Loading modules..."
module load anaconda3
module load cuda/12.1

# Create minimal environment
echo "🐍 Creating minimal environment..."
conda create -n opinion-balancer-gpu python=3.10 -y

# Activate environment  
echo "🔧 Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate opinion-balancer-gpu

# Install PyTorch with GPU support (multiple methods for reliability)
echo "⚡ Installing PyTorch GPU..."

# Method 1: Try conda first (more reliable on PACE)
echo "  Trying conda installation..."
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Check if successful, if not try pip
if ! python -c "import torch" 2>/dev/null; then
    echo "  Conda failed, trying pip..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

# Final fallback - CPU version
if ! python -c "import torch" 2>/dev/null; then
    echo "  GPU versions failed, installing CPU version..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install basic dependencies first
echo "� Installing basic dependencies..."
pip install requests urllib3 certifi charset-normalizer idna

# Install core packages
echo "🔬 Installing scientific packages..."
pip install numpy scipy pandas scikit-learn joblib

# Install core utilities
echo "🔧 Installing core utilities..."
pip install pyyaml rich click tqdm

# Install ML packages (now that dependencies are ready)
echo "� Installing ML packages..."
pip install tokenizers
pip install huggingface-hub
pip install transformers>=4.35

# Install NLP packages
echo "📝 Installing NLP packages..."
pip install nltk textstat sentence-transformers

# Install LangChain/LangGraph (after all dependencies)
echo "🔗 Installing LangGraph stack..."
pip install pydantic
pip install langchain-core
pip install langchain  
pip install langgraph

# Install optional accelerate for GPU optimization
echo "⚡ Installing GPU optimization..."
pip install accelerate || echo "⚠️  Accelerate install failed (optional)"

# Verify installation
echo "🧪 Testing installation..."

# Test PyTorch
if python -c "import torch" 2>/dev/null; then
    python -c "
import torch
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ CUDA version: {torch.version.cuda}')
else:
    print('⚠️  Running on CPU mode')
"
else
    echo "❌ PyTorch installation failed!"
    echo "🔧 Attempting manual fix..."
    pip install torch --no-deps
    pip install torchvision --no-deps  
    pip install torchaudio --no-deps
fi

# Test transformers
if python -c "import transformers" 2>/dev/null; then
    python -c "import transformers; print(f'✅ Transformers: {transformers.__version__}')"
else
    echo "❌ Transformers failed, reinstalling..."
    pip install --force-reinstall transformers
fi

# Test LangGraph
if python -c "import langgraph" 2>/dev/null; then
    echo "✅ LangGraph available"
else
    echo "⚠️  LangGraph not available - trying to fix..."
    pip install --force-reinstall langgraph langchain langchain-core
fi

# Test model path
echo "📁 Testing model path..."
python -c "
import os
model_path = '/storage/data/mod-huggingface-0/openai-community__gpt2-medium/models--openai-community--gpt2-medium/snapshots/6dcaa7a952f72f9298047fd5137cd6e4f05f41da'
print(f'Model path exists: {os.path.exists(model_path)}')
"

# Final verification
echo "🔍 Final system check..."
python -c "
import sys
print(f'Python: {sys.version}')

packages = ['torch', 'transformers', 'langgraph', 'numpy', 'pydantic']
for pkg in packages:
    try:
        __import__(pkg)
        print(f'✅ {pkg}')
    except ImportError:
        print(f'❌ {pkg} - missing')
"

echo ""
echo "✅ Fast setup complete!"
echo ""
echo "To use:"
echo "  conda activate opinion-balancer-gpu" 
echo "  python run.py --test"
echo ""
echo "If packages are still missing, try the ordered requirements:"
echo "  conda activate opinion-balancer-gpu"
echo "  pip install -r requirements-ordered.txt"
echo ""
echo "Or install manually in order:"
echo "  pip install requests numpy pyyaml"
echo "  pip install transformers"
echo "  pip install langgraph langchain pydantic"
