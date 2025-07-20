# Stream Processing Experiment

## Overview
This experiment demonstrates a scenario where space-time tradeoffs are actually BENEFICIAL - reducing memory usage can improve performance!

## The Problem
Computing sliding window statistics (e.g., moving average) over a data stream.

## Approaches

1. **Full Storage** - O(n) space
   - Store entire stream in memory
   - Random access to any element
   - Poor cache locality for large streams

2. **Sliding Window** - O(w) space (w = window size)
   - Only store current window
   - Optimal for streaming
   - Better cache performance

3. **Checkpoint Strategy** - O(√n) space
   - Store periodic checkpoints
   - Recompute from nearest checkpoint
   - Balance between space and recomputation

4. **Extreme Minimal** - O(1) space
   - Recompute everything each time
   - Theoretical minimum space
   - Impractical time complexity

## Key Insight

Unlike sorting, streaming algorithms can benefit from space reduction:
- **Better cache locality** → faster execution
- **Matches data access pattern** → no random access needed
- **Real-world systems** use this approach (Kafka, Flink, Spark Streaming)

## Running the Experiment

```bash
cd experiments/stream_processing
python sliding_window.py
```

## Expected Results

The sliding window approach (less memory) is FASTER than full storage because:
1. All data fits in CPU cache
2. No memory allocation overhead
3. Sequential access pattern

This validates that Williams' space-time tradeoffs aren't always penalties - 
sometimes reducing space improves both memory usage AND performance!