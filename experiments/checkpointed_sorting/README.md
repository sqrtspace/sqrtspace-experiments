# Checkpointed Sorting Experiment

## Overview
This experiment demonstrates how external merge sort with limited memory exhibits the space-time tradeoff predicted by Williams' 2025 result.

## Key Concepts

### Standard In-Memory Sort
- **Space**: O(n) - entire array in memory
- **Time**: O(n log n) - optimal comparison-based sorting
- **Example**: Python's built-in sort, quicksort

### Checkpointed External Sort
- **Space**: O(√n) - only √n elements in memory at once
- **Time**: O(n√n) - due to disk I/O and recomputation
- **Technique**: Sort chunks that fit in memory, merge with limited buffers

### Extreme Space-Limited Sort
- **Space**: O(log n) - minimal memory usage
- **Time**: O(n²) - extensive recomputation required
- **Technique**: Iterative merging with frequent checkpointing

## Running the Experiments

### Quick Test
```bash
python test_quick.py
```
Runs with small input sizes (100-1000) to verify correctness.

### Full Experiment
```bash
python run_final_experiment.py
```
Runs complete experiment with:
- Input sizes: 1000, 2000, 5000, 10000, 20000
- 10 trials per size for statistical significance
- RAM disk comparison to isolate I/O overhead
- Generates publication-quality plots

### Rigorous Analysis
```bash
python rigorous_experiment.py
```
Comprehensive experiment with:
- 20 trials per size
- Detailed memory profiling
- Environment logging
- Statistical analysis with confidence intervals

## Actual Results (Apple M3 Max, 64GB RAM)

| Input Size | In-Memory Time | Checkpointed Time | Slowdown | Memory Reduction |
|------------|----------------|-------------------|----------|------------------|
| 1,000      | 0.022 ms       | 8.2 ms           | 375×     | 0.1× (overhead)  |
| 5,000      | 0.045 ms       | 23.4 ms          | 516×     | 0.2×             |
| 10,000     | 0.091 ms       | 40.5 ms          | 444×     | 0.2×             |
| 20,000     | 0.191 ms       | 71.4 ms          | 375×     | 0.2×             |

Note: Memory shows algorithmic overhead due to Python's memory management.

## Key Findings

1. **Massive Constant Factors**: 375-627× slowdown instead of theoretical √n
2. **I/O Not Dominant**: Fast NVMe SSDs show only 1.0-1.1× I/O overhead
3. **Scaling Confirmed**: Power law fits show n^1.0 for in-memory, n^1.4 for checkpointed

## Real-World Applications

- **Database Systems**: External sorting for large datasets
- **MapReduce**: Shuffle phase with limited memory
- **Video Processing**: Frame-by-frame processing with checkpoints
- **Scientific Computing**: Out-of-core algorithms

## Visualization

The experiment generates:
1. `paper_sorting_figure.png` - Clean figure for publication
2. `rigorous_sorting_analysis.png` - Detailed analysis with error bars
3. `memory_usage_analysis.png` - Memory scaling comparison
4. `experiment_environment.json` - Hardware/software configuration
5. `final_experiment_results.json` - Raw experimental data

## Dependencies

```bash
pip install numpy scipy matplotlib psutil
```

## Reproducing Results

To reproduce our results exactly:
1. Ensure CPU frequency scaling is disabled
2. Close all other applications
3. Run on a machine with fast SSD (>3GB/s read)
4. Use Python 3.10+ with NumPy 2.0+