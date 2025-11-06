# OpinionBalancer

**A multi-agent writing system for balanced opinion pieces powered by GPT-5**

OpinionBalancer is a KB-free, deterministic system built with LangGraph that automatically drafts and refines balanced opinion pieces using OpenAI's GPT-5. Designed for high-quality, nuanced opinion generation with built-in bias detection and balance optimization.

## 🎯 What It Does

The system takes a topic and produces balanced opinion articles through a structured, iterative refinement process:

1. **Topic Intake** → Normalize topic and parameters
2. **Draft Writing** → Generate initial opinion piece
3. **Multi-Metric Evaluation** → Assess bias, framing, readability, coherence
4. **Critique Synthesis** → Generate targeted edit instructions
5. **Editing** → Apply improvements (single pass)
6. **Logging** → Save results and metrics

## 🔧 Key Features

- **🤖 GPT-5 Powered**: Uses OpenAI's latest GPT-5 model for superior text generation
- **⚖️ Bias Detection**: Quantifies and balances political stance
- **🖼️ Frame Diversity**: Ensures multiple perspective types (moral, economic, policy, etc.)
- **📖 Readability Control**: Targets specific grade levels (10-13)
- **🔗 Coherence Scoring**: Maintains logical flow between paragraphs
- **📊 Deterministic Metrics**: All evaluations are measurable and reproducible
- **🌐 API-Based**: Leverages OpenAI's powerful infrastructure
- **� Single-Pass**: Linear workflow for consistent, predictable results

## 🚀 Quick Start

### 1. Setup

```bash
# Clone repository
git clone https://github.com/mbidkar/opinion-balancer.git
cd opinion-balancer

# Install dependencies
pip install -r requirements-simple.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 2. Configuration

Add your OpenAI API key to the `.env` file:
```bash
OPENAI_API_KEY=your_openai_api_key_here
LANGSMITH_API_KEY=your_langsmith_key_here  # Optional
LANGSMITH_TRACING=true  # Optional
```

### 3. Basic Usage

```bash
# Test OpenAI connection
python test_openai.py

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
- **Model**: GPT-5 (OpenAI's latest)
- **Length**: Maximum 500 words
- **Target Balance**: 50% Left, 50% Right 
- **Audience**: General US reader
- **Workflow**: Single-pass linear execution (no loops)
- **Grade Level**: 10-13 (high school to college)

## 🏗️ System Architecture

### Core Components

1. **LLM Client** (`llm_client_openai.py`) - OpenAI GPT-5 interface with API management
2. **State Management** (`state.py`) - Pydantic models for workflow state
3. **LangGraph Workflow** (`graphs/kb_free.py`) - Multi-agent orchestration
4. **Evaluation Nodes** (`nodes/`) - Bias, frame, readability, coherence analysis
5. **Generation Nodes** (`nodes/`) - Draft writing, editing, critique synthesis

### Model Configuration

- **Model**: GPT-5 (OpenAI API)
- **API**: Requires OPENAI_API_KEY environment variable
- **Fallback**: GPT-4 or GPT-3.5-turbo (configurable)
- **Temperature**: Task-specific (0.6-0.8 range)

## 📁 Files Overview

```
opinion-balancer/
├── .env                  # Environment variables (API keys)
├── requirements-simple.txt # Python package dependencies (OpenAI focus)
├── environment.yml       # Conda environment (simplified)
├── test_openai.py        # OpenAI API test script
├── llm_client_openai.py  # OpenAI API client implementation
├── langgraph.json        # LangGraph configuration
├── run.py                # Main CLI interface
├── state.py              # Pydantic state models
├── config.yaml           # System configuration (GPT-5 settings)
├── prompts.yaml          # LLM prompts
├── graphs/
│   └── kb_free.py        # LangGraph workflow definition
└── nodes/                # Individual processing nodes
    ├── evaluators/       # Bias, frame, readability evaluators
    └── generators/       # Draft, edit, critique generators
```

## 🧪 Testing

```bash
# Test OpenAI API connectivity
python test_openai.py

# Test LangGraph setup
langgraph dev

# Test full pipeline
python run.py --test
```

## 🔧 Configuration

Edit `config.yaml` to customize:
- Model parameters and API settings
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
