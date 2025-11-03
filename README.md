# OpinionBalancer

**A local, multi-agent writing system for balanced opinion pieces**

OpinionBalancer is a KB-free, deterministic system built with LangGraph that automatically drafts and refines balanced opinion pieces using local LLMs via Ollama. No external APIs, knowledge bases, or internet connection required.

## 🎯 What It Does

The system takes a topic and produces balanced opinion articles through a structured, iterative refinement process:

1. **Topic Intake** → Normalize topic and parameters
2. **Draft Writing** → Generate initial opinion piece
3. **Multi-Metric Evaluation** → Assess bias, framing, readability, coherence
4. **Critique Synthesis** → Generate targeted edit instructions
5. **Editing** → Apply improvements
6. **Convergence Check** → Repeat until quality thresholds met

## 🔧 Key Features

- **🏠 100% Local**: Runs entirely on your machine using Ollama
- **⚖️ Bias Detection**: Quantifies and balances political stance
- **🖼️ Frame Diversity**: Ensures multiple perspective types (moral, economic, policy, etc.)
- **📖 Readability Control**: Targets specific grade levels (10-13)
- **🔗 Coherence Scoring**: Maintains logical flow between paragraphs
- **📊 Deterministic Metrics**: All evaluations are measurable and reproducible
- **🚫 No External Dependencies**: No APIs, knowledge bases, or internet required

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Install Ollama
brew install ollama

# Start Ollama and pull model
ollama serve
ollama pull llama3.2:1b

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Basic Usage

```bash
# Simple opinion piece
python run.py --topic "Universal basic income in the US"

# Custom parameters
python run.py \
  --topic "Climate change policy options" \
  --audience "policymakers" \
  --length 800 \
  --target "Left=0.4,Center=0.2,Right=0.4"

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
# Basic run
python run.py --topic "Universal basic income in the US"

# Test components
python run.py --test

# Custom configuration  
python run.py --topic "Climate policy" --length 800 --target "Left=0.4,Right=0.6"
```

*Complete documentation and examples available in the full README.md*

A new project for balancing opinions.

## Getting Started

This repository was just created. Add your project description and setup instructions here.

## Features

- Add your features here

## Installation

- Add installation instructions here

## Usage

- Add usage instructions here

## Contributing

- Add contributing guidelines here
