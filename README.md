# The Ubiquity of Space-Time Tradeoffs: Experiments & Implementation

This repository contains the experimental code, case studies, and interactive dashboard accompanying the paper "The Ubiquity of Space-Time Simulation in Modern Computing: From Theory to Practice".

**Paper Repository**: [github.com/sqrtspace/sqrtspace-paper](https://github.com/sqrtspace/sqrtspace-paper)  
**Interactive Dashboard**: Run locally with `streamlit run dashboard/app.py`  
**Based on**: Ryan Williams' 2025 result that TIME[t] ⊆ SPACE[√(t log t)]

## Overview

This project demonstrates how theoretical space-time tradeoffs manifest in real-world systems through:
- **Controlled experiments** validating the √n relationship
- **Interactive visualizations** exploring memory hierarchies
- **Practical implementations** in production-ready libraries

## Key Findings

- Theory predicts √n slowdown, practice shows 100-10,000× due to constant factors
- Memory hierarchy (L1/L2/L3/RAM/Disk) dominates performance
- Cache-friendly algorithms can be faster with less memory
- The √n pattern appears in our experimental implementations

## Experiments

### 1. Maze Solver (C#)
**Location:** `experiments/maze_solver/`

Demonstrates graph traversal with memory constraints:
- BFS: O(n) memory, 1ms runtime
- Memory-Limited DFS: O(√n) memory, 5ms runtime (5× slower)

```bash
cd experiments/maze_solver
dotnet run
```

### 2. Checkpointed Sorting (Python)
**Location:** `experiments/checkpointed_sorting/`

Shows massive I/O penalties when reducing memory:
- In-memory: O(n) space, 0.0001s
- Checkpointed: O(√n) space, 0.268s (2,680× slower!)

```bash
cd experiments/checkpointed_sorting
python checkpointed_sort.py
```

### 3. Stream Processing (Python)
**Location:** `experiments/stream_processing/`

Reveals when less memory is actually faster:
- Full history: O(n) memory, 0.33s
- Sliding window: O(w) memory, 0.011s (30× faster!)

```bash
cd experiments/stream_processing
python sliding_window.py
```

### 4. Real LLM Inference with Ollama (Python)
**Location:** `experiments/llm_ollama/`

Demonstrates space-time tradeoffs with actual language models:
- Context chunking: 18.3× slowdown for √n chunks
- Streaming generation: 6% overhead vs full generation
- Checkpointing: 7.6% overhead for fault tolerance

```bash
cd experiments/llm_ollama
python ollama_spacetime_experiment.py
```

## Quick Start

### Prerequisites
- Python 3.8+ (for Python experiments)
- .NET Core SDK (for C# maze solver)
- Ollama (optional, for real LLM experiments)
- 2GB free memory for experiments

### Installation
```bash
# Clone repository
git clone https://github.com/sqrtspace/sqrtspace-experiments.git
cd Ubiquity

# Install Python dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run dashboard/app.py
```

### Running All Experiments
```bash
# Run each experiment
cd experiments/maze_solver && dotnet run && cd ../..
cd experiments/checkpointed_sorting && python checkpointed_sort.py && cd ../..
cd experiments/stream_processing && python sliding_window.py && cd ../..
cd experiments/database_buffer_pool && python sqlite_heavy_experiment.py && cd ../..
cd experiments/llm_kv_cache && python llm_kv_cache_experiment.py && cd ../..
cd experiments/llm_ollama && python ollama_spacetime_experiment.py && cd ../..  # Requires Ollama
```

## Repository Structure

```
├── experiments/                    # Core experiments demonstrating tradeoffs
│   ├── maze_solver/               # C# graph traversal with memory limits
│   ├── checkpointed_sorting/      # Python external sorting with O(√n) space
│   ├── stream_processing/         # Python sliding window vs full storage
│   ├── database_buffer_pool/      # SQLite experiments with different cache sizes
│   ├── llm_kv_cache/             # Simulated LLM attention mechanism tradeoffs
│   ├── llm_ollama/               # Real LLM experiments with Ollama models
│   └── measurement_framework.py   # Shared profiling and analysis tools
├── dashboard/                     # Interactive Streamlit visualizations
│   ├── app.py                    # 6-page interactive dashboard
│   └── requirements.txt          # Dashboard dependencies
└── FINDINGS.md                   # Verified experimental results with statistical analysis
```

## Interactive Dashboard

The dashboard (`dashboard/app.py`) includes:
1. **Space-Time Calculator**: Find optimal configurations
2. **Memory Hierarchy Simulator**: Visualize cache effects
3. **Algorithm Comparisons**: See tradeoffs in action
4. **LLM Optimizations**: Flash Attention demonstrations
5. **Implementation Examples**: Library demonstrations

## Measurement Framework

`experiments/measurement_framework.py` provides:
- Continuous memory monitoring (10ms intervals)
- Cache-aware benchmarking
- Statistical analysis across multiple runs
- Automated visualization generation

## Extending the Work

### Adding New Experiments
1. Create folder in `experiments/`
2. Implement space-time tradeoff variants
3. Use `measurement_framework.py` for profiling
4. Document findings in experiment README

## 📚 Citation

If you use this code or build upon our work:

```bibtex
@article{friedel2025ubiquity,
  title={The Ubiquity of Space-Time Simulation in Modern Computing: From Theory to Practice},
  author={Friedel Jr., David H.},
  journal={arXiv preprint arXiv:25XX.XXXXX},
  year={2025}
}
```

## Contact

**Author**: David H. Friedel Jr.  
**Organization**: MarketAlly LLC (USA) & MarketAlly Pte. Ltd. (Singapore)  
**Email**: dfriedel@marketally.ai

## License

This work is licensed under CC BY 4.0. You may share and adapt the material with proper attribution.

## Acknowledgments

- Ryan Williams for the theoretical foundation
- The authors of Flash Attention, PostgreSQL, and Apache Spark
- Early-stage R&D support from MarketAlly LLC and MarketAlly Pte. Ltd.
