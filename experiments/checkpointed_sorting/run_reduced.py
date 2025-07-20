"""
Run sorting experiments with reduced parameters for faster execution
"""

import sys
sys.path.insert(0, '..')

# Modify the original script to use smaller parameters
from checkpointed_sort import *

def run_reduced_experiments():
    """Run with smaller sizes and fewer trials for quick results"""
    
    print("=== Checkpointed Sorting Experiment (Reduced) ===\n")
    
    # Reduced parameters
    num_trials = 5  # Instead of 20
    sizes = [1000, 2000, 5000, 10000]  # Smaller sizes
    results = []
    
    for size in sizes:
        print(f"\nTesting with {size} elements ({num_trials} trials each):")
        
        # Store times for each trial
        in_memory_times = []
        checkpoint_times = []
        extreme_times = []
        
        for trial in range(num_trials):
            exp = SortingExperiment(size)
            
            # 1. In-memory sort - O(n) space
            start = time.time()
            result1 = exp.in_memory_sort()
            time1 = time.time() - start
            in_memory_times.append(time1)
            
            # 2. Checkpointed sort - O(√n) space
            memory_limit = int(np.sqrt(size) * 4)  # 4 bytes per element
            start = time.time()
            result2 = exp.checkpoint_sort(memory_limit)
            time2 = time.time() - start
            checkpoint_times.append(time2)
            
            # 3. Extreme checkpoint - O(log n) space (only for size 1000)
            if size == 1000 and trial == 0:  # Just once for demo
                print("  Running extreme checkpoint (this will take ~2-3 minutes)...")
                start = time.time()
                result3 = exp.extreme_checkpoint_sort()
                time3 = time.time() - start
                extreme_times.append(time3)
                print(f"  Extreme checkpoint completed: {time3:.1f}s")
            
            # Verify correctness (only on first trial)
            if trial == 0:
                assert np.allclose(result1, result2), "Checkpointed sort produced incorrect result"
            
            exp.cleanup()
            
            # Progress indicator
            if trial == num_trials - 1:
                print(f"    Completed all trials")
        
        # Calculate statistics
        in_memory_mean = np.mean(in_memory_times)
        in_memory_std = np.std(in_memory_times)
        checkpoint_mean = np.mean(checkpoint_times)
        checkpoint_std = np.std(checkpoint_times)
        
        print(f"  In-memory sort: {in_memory_mean:.4f}s ± {in_memory_std:.4f}s")
        print(f"  Checkpointed sort (√n memory): {checkpoint_mean:.4f}s ± {checkpoint_std:.4f}s")
        
        if extreme_times:
            extreme_mean = np.mean(extreme_times)
            extreme_std = 0  # Only one trial
            print(f"  Extreme checkpoint (log n memory): {extreme_mean:.4f}s")
        else:
            extreme_mean = None
            extreme_std = None
        
        # Calculate slowdown factor
        slowdown = checkpoint_mean / in_memory_mean if in_memory_mean > 0.0001 else checkpoint_mean / 0.0001
        
        # Calculate 95% confidence intervals
        if num_trials > 1:
            in_memory_ci = stats.t.interval(0.95, len(in_memory_times)-1, 
                                           loc=in_memory_mean, 
                                           scale=stats.sem(in_memory_times))
            checkpoint_ci = stats.t.interval(0.95, len(checkpoint_times)-1, 
                                            loc=checkpoint_mean, 
                                            scale=stats.sem(checkpoint_times))
        else:
            in_memory_ci = (in_memory_mean, in_memory_mean)
            checkpoint_ci = (checkpoint_mean, checkpoint_mean)
        
        results.append({
            'size': size,
            'in_memory_time': in_memory_mean,
            'in_memory_std': in_memory_std,
            'in_memory_ci': in_memory_ci,
            'checkpoint_time': checkpoint_mean,
            'checkpoint_std': checkpoint_std,
            'checkpoint_ci': checkpoint_ci,
            'extreme_time': extreme_mean,
            'extreme_std': extreme_std,
            'slowdown': slowdown,
            'num_trials': num_trials
        })
    
    # Plot results with error bars
    plot_sorting_results(results)
    
    print("\n=== Summary ===")
    print("Space-time tradeoffs observed:")
    for r in results:
        print(f"  n={r['size']:,}: {r['slowdown']:.0f}x slowdown for √n space reduction")
    
    return results

if __name__ == "__main__":
    results = run_reduced_experiments()