"""Test rigorous experiment with small parameters"""

from rigorous_experiment import *

def test_main():
    """Run with very small sizes for testing"""
    
    print("="*60)
    print("TEST RUN - RIGOROUS EXPERIMENT")
    print("="*60)
    
    # Log environment
    env = ExperimentEnvironment.get_environment()
    print("\nExperimental Environment:")
    print(f"  Platform: {env['platform']}")
    print(f"  Python: {env['python_version']}")
    print(f"  CPUs: {env['cpu_count']} physical, {env['cpu_count_logical']} logical")
    print(f"  Memory: {env['memory_total'] / 1e9:.1f} GB total")
    
    # Test with very small sizes
    sizes = [100, 500, 1000]
    num_trials = 3  # Just 3 trials for test
    all_results = []
    
    for size in sizes:
        result = run_single_experiment(size, num_trials=num_trials)
        all_results.append(result)
        
        print(f"\nResults for n={size:,}:")
        print(f"  In-memory: {result['in_memory_mean']:.6f}s")
        print(f"  Checkpoint: {result['checkpoint_mean']:.6f}s")
        print(f"  Slowdown: {result['slowdown_disk']:.1f}x")
    
    print("\n✓ Test completed successfully!")

if __name__ == "__main__":
    test_main()