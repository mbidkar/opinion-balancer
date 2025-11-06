# OpinionBalancer

**A local, multi-agent writing system for balanced opinion pieces**

OpinionBalancer is a KB-free, deterministic system built with LangGraph that automatically drafts and refines balanced opinion pieces using GPT-2. Designed to run locally on PACE-ICE cluster or personal machines with minimal dependencies.

## 🎯 What It Does

The system takes a topic and produces balanced opinion articles through a structured, iterative refinement process:

1. **Topic Intake** → Normalize topic and parameters
2. **Draft Writing** → Generate initial opinion piece
3. **Multi-Metric Evaluation** → Assess bias, framing, readability, coherence
4. **Critique Synthesis** → Generate targeted edit instructions
5. **Editing** → Apply improvements (single pass)
6. **Logging** → Save results and metrics

## 🔧 Key Features

- **🏠 100% Local**: Runs entirely locally using GPT-2 (355M parameter model)
- **⚖️ Bias Detection**: Quantifies and balances political stance
- **🖼️ Frame Diversity**: Ensures multiple perspective types (moral, economic, policy, etc.)
- **📖 Readability Control**: Targets specific grade levels (10-13)
- **🔗 Coherence Scoring**: Maintains logical flow between paragraphs
- **📊 Deterministic Metrics**: All evaluations are measurable and reproducible
- **�️ PACE-ICE Ready**: Optimized for Georgia Tech's computing cluster
- **🚫 No External APIs**: No internet or external services required

## 🚀 Quick Start

### 1. PACE-ICE Setup

```bash
# Clone repository
git clone https://github.com/mbidkar/opinion-balancer.git
cd opinion-balancer

# Run setup script
chmod +x setup.sh
./setup.sh

# Activate environment
source activate.sh
```

### 2. Local Setup

```bash
# Create environment
conda env create -f environment.yml
conda activate opinion-balancer

# Test installation
python test_gpt2.py
```

### 3. Basic Usage

```bash
# Test GPT-2 client
python test_gpt2.py

# Start LangGraph development server
langgraph dev

# Run opinion analysis
python run.py --topic "Universal basic income in the US"

# Test system components
python run.py --test
```

## 📊 Measurement System

### Bias Scoring
- **Method**: Keyword-based stance detection with lexicon matching
- **Output**: Probability distribution over Left/Center/Right positions
- **Target**: Configurable distribution (default: 50/50 L/R)
- **Threshold**: Bias delta ≤ 0.05 (5% deviation from target)

### Frame Diversity 
- **Method**: Shannon entropy over frame categories
- **Categories**: Moral, Economic, Policy, Conflict, Human Interest, Consequence, Attribution
- **Target**: Entropy ≥ 0.6 (encourages multiple frames)

### Readability & Coherence
- **Readability**: Flesch-Kincaid Grade Level targeting 10-13
- **Coherence**: Cosine similarity between paragraph embeddings ≥ 0.7

## 🎛️ Usage Examples

```bash
# Basic run (500 word max, Left=0.5/Right=0.5 target)
python run.py --topic "Universal basic income in the US"

# Test components
python run.py --test

# Start development server
langgraph dev
```

## ⚙️ Fixed Configuration

OpinionBalancer uses fixed settings to ensure consistent results:
- **Length**: Maximum 500 words
- **Target Balance**: 50% Left, 50% Right 
- **Audience**: General US reader
- **Workflow**: Single-pass linear execution (no loops)
- **Grade Level**: 10-13 (high school to college)

## 🏗️ System Architecture

### Core Components

1. **LLM Client** (`llm_client_gpt2.py`) - GPT-2 interface with GPU/CPU auto-detection
2. **State Management** (`state.py`) - Pydantic models for workflow state
3. **LangGraph Workflow** (`graphs/kb_free.py`) - Multi-agent orchestration
4. **Evaluation Nodes** (`nodes/`) - Bias, frame, readability, coherence analysis
5. **Generation Nodes** (`nodes/`) - Draft writing, editing, critique synthesis

### Model Configuration

- **Model**: GPT-2 Medium (355M parameters)
- **Location**: `/storage/data/mod-huggingface-0/gpt2-medium` (PACE-ICE)
- **Fallback**: Online GPT-2 via Hugging Face Hub
- **Device**: Auto-detection (CUDA if available, else CPU)

## 📁 Files Overview

```
opinion-balancer/
├── setup.sh              # Main setup script for PACE-ICE
├── activate.sh            # Environment activation helper
├── environment.yml        # Conda environment specification
├── requirements-simple.txt # Python package dependencies
├── test_gpt2.py          # GPT-2 client test script
├── llm_client_gpt2.py    # GPT-2 LLM client implementation
├── langgraph.json        # LangGraph configuration
├── run.py                # Main CLI interface
├── state.py              # Pydantic state models
├── config.yaml           # System configuration
├── prompts.yaml          # LLM prompts
├── graphs/
│   └── kb_free.py        # LangGraph workflow definition
└── nodes/                # Individual processing nodes
    ├── evaluators/       # Bias, frame, readability evaluators
    └── generators/       # Draft, edit, critique generators
```

## 🧪 Testing

```bash
# Test GPT-2 client
python test_gpt2.py

# Test LangGraph setup
langgraph dev

# Test full pipeline
python run.py --test
```

## 🔧 Configuration

Edit `config.yaml` to customize:
- Model paths and parameters
- Evaluation thresholds
- Target bias distributions
- Output formatting

## 📈 Development

1. Use `langgraph dev` for interactive development
2. Monitor with LangSmith (optional, set `LANGSMITH_API_KEY`)
3. Test individual nodes in `nodes/` directory
4. Modify prompts in `prompts.yaml`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Test on PACE-ICE environment
4. Submit a pull request
