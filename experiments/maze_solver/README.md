# Experiment: Maze Solver with Memory Constraints

## Objective
Demonstrate Ryan Williams' 2025 theoretical result that TIME[t] ⊆ SPACE[√(t log t)] through practical maze-solving algorithms.

## Algorithms Implemented

1. **BFS (Breadth-First Search)**
   - Space: O(n) - stores all visited nodes
   - Time: O(n) - visits each node once
   - Finds shortest path

2. **DFS (Depth-First Search)** 
   - Space: O(n) - standard implementation
   - Time: O(n) - may not find shortest path

3. **Memory-Limited DFS**
   - Space: O(√n) - only keeps √n nodes in memory
   - Time: O(n√n) - must recompute evicted paths
   - Demonstrates the space-time tradeoff

4. **Iterative Deepening DFS**
   - Space: O(log n) - only stores current path
   - Time: O(n²) - recomputes extensively
   - Extreme space efficiency at high time cost

## Key Insight
By limiting memory to O(√n), we force the algorithm to recompute paths, increasing time complexity. This mirrors Williams' theoretical result showing that any time-bounded computation can be simulated with √(t) space.

## Running the Experiment

```bash
dotnet run                    # Run simple demo
dotnet run --property:StartupObject=Program  # Run full experiment
python plot_memory.py         # Visualize results
```

## Expected Results
- BFS uses ~n memory units, completes in ~n time
- Memory-limited DFS uses ~√n memory, takes ~n√n time
- Shows approximately quadratic time increase for square-root memory reduction

This practical demonstration validates the theoretical space-time tradeoff!
