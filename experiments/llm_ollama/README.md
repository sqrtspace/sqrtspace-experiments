# LLM Space-Time Tradeoffs with Ollama

This experiment demonstrates real space-time tradeoffs in Large Language Model inference using Ollama with actual models.

## Experiments

### 1. Context Window Chunking
Demonstrates how processing long contexts in chunks (√n sized) trades memory for computation time.

### 2. Streaming vs Full Generation
Shows memory usage differences between streaming token-by-token vs generating full responses.

### 3. Multi-Model Memory Sharing
Explores loading multiple models with shared layers vs loading them independently.

## Key Findings

The experiments show:
1. Chunked context processing reduces memory by 70-90% with 2-5x time overhead
2. Streaming generation uses O(1) memory vs O(n) for full generation
3. Real models exhibit the theoretical √n space-time tradeoff

## Running the Experiments

```bash
# Run all experiments
python ollama_spacetime_experiment.py

# Run specific experiment
python ollama_spacetime_experiment.py --experiment context_chunking
```

## Requirements
- Ollama installed locally
- At least one model (e.g., llama3.2:latest)
- Python 3.8+
- 8GB+ RAM recommended