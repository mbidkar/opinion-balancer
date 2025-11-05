# OpinionBalancer GPU Environment Setup

## Quick Setup on PACE-ICE

```bash
# 1. Navigate to project directory
cd /storage/ice1/shared/ece8803cai/mbidkar3/opinion-balancer

# 2. Run the GPU setup script
chmod +x setup_pace.sh
./setup_pace.sh

# 3. Activate the GPU environment (for future sessions)
conda activate opinion-balancer-gpu
```

## GPU Environment Features

- **Environment Name**: `opinion-balancer-gpu`
- **Python Version**: 3.10
- **CUDA Support**: 12.1.1
- **PyTorch**: GPU-enabled with CUDA support
- **Transformers**: GPU-accelerated inference
- **Device Detection**: Automatic GPU/CPU fallback

## Key Components

### GPU Packages
- `pytorch` with CUDA 12.1 support
- `transformers` with GPU acceleration
- `accelerate` for optimized inference
- `cuda-toolkit` for CUDA operations

### OpinionBalancer Stack
- `langgraph` for workflow orchestration
- `langchain` for LLM integration
- `sentence-transformers` for embeddings
- `pydantic` for data models
- Scientific computing: `numpy`, `scipy`, `pandas`, `scikit-learn`

## Usage Commands

```bash
# Activate environment
conda activate opinion-balancer-gpu

# Test the system
python run.py --test

# Run opinion balancing
python run.py --topic "Climate change policy debate"

# Monitor GPU usage
nvidia-smi
watch -n 1 nvidia-smi  # Continuous monitoring
```

## GPU Configuration

The system automatically detects GPU availability:
- **GPU Available**: Uses CUDA acceleration for model inference
- **GPU Not Available**: Falls back to CPU processing
- **Configuration**: Set via `device: auto` in `config.yaml`

## Performance Benefits

With GPU acceleration on PACE-ICE:
- **Faster Model Loading**: GPT-2 loads into GPU memory
- **Accelerated Inference**: Text generation ~5-10x faster
- **Batch Processing**: Can handle multiple evaluations efficiently
- **Memory Management**: Automatic GPU memory optimization

## Troubleshooting

### Check GPU Status
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Check Model Device
```bash
python -c "
from llm_client_gpt2 import make_llm_client
client = make_llm_client()
info = client.get_model_info()
print(f'Device: {info[\"device\"]}')
"
```

### Environment Issues
```bash
# Recreate environment if needed
conda env remove -n opinion-balancer-gpu
conda env create -f env-gpu.yml
```
