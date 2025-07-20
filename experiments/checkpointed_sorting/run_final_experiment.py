"""
Run final sorting experiment with parameters balanced for:
- Statistical significance (10 trials)
- Reasonable runtime (smaller sizes)
- Demonstrating scaling behavior
"""

from rigorous_experiment import *
import time

def run_final_experiment():
    """Run experiment with balanced parameters"""
    
    print("="*60)
    print("FINAL SORTING EXPERIMENT")
    print("Space-Time Tradeoffs in External Sorting")
    print("="*60)
    
    start_time = time.time()
    
    # Log environment
    env = ExperimentEnvironment.get_environment()
    print("\nExperimental Environment:")
    print(f"  Platform: {env['platform']}")
    print(f"  Python: {env['python_version']}")
    print(f"  CPUs: {env['cpu_count']} physical, {env['cpu_count_logical']} logical")
    print(f"  Memory: {env['memory_total'] / 1e9:.1f} GB total")
    if 'l3_cache' in env:
        print(f"  L3 Cache: {env['l3_cache'] / 1e6:.1f} MB")
    
    # Save environment
    with open('experiment_environment.json', 'w') as f:
        json.dump(env, f, indent=2)
    
    # Run experiments - balanced for paper
    sizes = [1000, 2000, 5000, 10000, 20000]
    num_trials = 10  # Enough for statistical significance
    all_results = []
    
    for size in sizes:
        print(f"\n{'='*40}")
        print(f"Testing n = {size:,}")
        print(f"{'='*40}")
        
        result = run_single_experiment(size, num_trials=num_trials)
        all_results.append(result)
        
        # Print detailed results
        print(f"\nSummary for n={size:,}:")
        print(f"  Algorithm           | Mean Time    | Std Dev      | Memory (peak)")
        print(f"  -------------------|--------------|--------------|---------------")
        print(f"  In-memory O(n)     | {result['in_memory_mean']:10.6f}s | ±{result['in_memory_std']:.6f}s | {result['in_memory_memory_mean']/1024:.1f} KB")
        print(f"  Checkpoint O(√n)   | {result['checkpoint_mean']:10.6f}s | ±{result['checkpoint_std']:.6f}s | {result['checkpoint_memory_mean']/1024:.1f} KB")
        
        if 'checkpoint_ramdisk_mean' in result:
            print(f"  Checkpoint (RAM)   | {result['checkpoint_ramdisk_mean']:10.6f}s | N/A          | {result['checkpoint_ramdisk_memory']/1024:.1f} KB")
            print(f"\n  Slowdown (with I/O): {result['slowdown_disk']:.1f}x")
            print(f"  Slowdown (RAM disk): {result['slowdown_ramdisk']:.1f}x") 
            print(f"  Pure I/O overhead:   {result['io_overhead_factor']:.1f}x")
        else:
            print(f"\n  Slowdown: {result['slowdown_disk']:.1f}x")
        
        print(f"  Memory reduction: {result['in_memory_memory_mean'] / result['checkpoint_memory_mean']:.1f}x")
    
    # Save detailed results
    with open('final_experiment_results.json', 'w') as f:
        json.dump({
            'environment': env,
            'parameters': {
                'sizes': sizes,
                'num_trials': num_trials
            },
            'results': all_results
        }, f, indent=2)
    
    # Create comprehensive plots
    create_comprehensive_plots(all_results)
    
    # Also create a simple summary plot for the paper
    create_paper_figure(all_results)
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE in {elapsed:.1f} seconds")
    print("\nGenerated files:")
    print("  - experiment_environment.json")
    print("  - final_experiment_results.json") 
    print("  - rigorous_sorting_analysis.png")
    print("  - memory_usage_analysis.png")
    print("  - paper_sorting_figure.png")
    print(f"{'='*60}")
    
    return all_results

def create_paper_figure(all_results: List[Dict]):
    """Create a clean figure for the paper"""
    
    sizes = [r['size'] for r in all_results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left plot: Time complexity
    in_memory_means = [r['in_memory_mean'] for r in all_results]
    checkpoint_means = [r['checkpoint_mean'] for r in all_results]
    
    ax1.loglog(sizes, in_memory_means, 'o-', label='In-memory O(n)', 
               color='blue', linewidth=2, markersize=8)
    ax1.loglog(sizes, checkpoint_means, 's-', label='Checkpointed O(√n)', 
               color='red', linewidth=2, markersize=8)
    
    # Add trend lines
    sizes_smooth = np.logspace(np.log10(1000), np.log10(20000), 100)
    
    # Fit actual data
    from scipy.optimize import curve_fit
    def power_law(x, a, b):
        return a * x**b
    
    popt1, _ = curve_fit(power_law, sizes, in_memory_means)
    popt2, _ = curve_fit(power_law, sizes, checkpoint_means)
    
    ax1.loglog(sizes_smooth, power_law(sizes_smooth, *popt1), 
               'b--', alpha=0.5, label=f'Fit: n^{{{popt1[1]:.2f}}}')
    ax1.loglog(sizes_smooth, power_law(sizes_smooth, *popt2), 
               'r--', alpha=0.5, label=f'Fit: n^{{{popt2[1]:.2f}}}')
    
    ax1.set_xlabel('Input Size (n)', fontsize=14)
    ax1.set_ylabel('Time (seconds)', fontsize=14)
    ax1.set_title('(a) Time Complexity', fontsize=16)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Slowdown factor
    slowdowns = [r['slowdown_disk'] for r in all_results]
    
    ax2.loglog(sizes, slowdowns, 'go-', linewidth=2, markersize=8, 
               label='Observed')
    
    # Theoretical √n
    theory = np.sqrt(sizes_smooth / sizes[0]) * slowdowns[0] / np.sqrt(1)
    ax2.loglog(sizes_smooth, theory, 'k--', alpha=0.5, 
               label='Theoretical √n')
    
    ax2.set_xlabel('Input Size (n)', fontsize=14)
    ax2.set_ylabel('Slowdown Factor', fontsize=14)
    ax2.set_title('(b) Cost of Space Reduction', fontsize=16)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('paper_sorting_figure.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    results = run_final_experiment()