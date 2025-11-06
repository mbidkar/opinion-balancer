#!/bin/bash
# Quick activation script for OpinionBalancer environment

source $(conda info --base)/etc/profile.d/conda.sh
conda activate opinion-balancer

echo "✅ OpinionBalancer environment activated"
echo "Current environment: $CONDA_DEFAULT_ENV"
echo "Python: $(python --version)"

# Show available commands
echo ""
echo "🛠️  Available commands:"
echo "  python test_gpt2.py     # Test GPT-2 client"
echo "  langgraph dev           # Start LangGraph dev server"
echo "  python run.py --test    # Test full system"
echo "  python run.py --topic 'Your topic'  # Run analysis"
