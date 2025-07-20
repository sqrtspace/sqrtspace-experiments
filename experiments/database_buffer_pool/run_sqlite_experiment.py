"""
Run SQLite buffer pool experiment with realistic parameters
Shows space-time tradeoffs in a production database system
"""

from sqlite_buffer_pool_experiment import *
import matplotlib.pyplot as plt

def run_realistic_experiment():
    """Run experiment with parameters that show clear tradeoffs"""
    
    print("="*60)
    print("SQLite Buffer Pool Space-Time Tradeoff")
    print("Demonstrating Williams' √n pattern in databases")
    print("="*60)
    
    # Use a size that creates meaningful page counts
    num_users = 25000  # Creates ~6MB database
    
    exp = SQLiteExperiment(num_users)
    print(f"\nCreating database with {num_users:,} users...")
    db_size = exp.setup_database()
    stats = exp.analyze_page_distribution()
    
    print(f"\nDatabase Statistics:")
    print(f"  Size: {db_size / 1024 / 1024:.1f} MB")
    print(f"  Pages: {stats['page_count']:,}")
    print(f"  Page size: {stats['page_size']} bytes")
    print(f"  Users: {stats['users_count']:,}")
    print(f"  Posts: {stats['posts_count']:,}")
    
    # Define cache configurations based on theory
    optimal_cache = stats['page_count']  # O(n) - all pages in memory
    sqrt_cache = int(np.sqrt(stats['page_count']))  # O(√n)
    log_cache = max(5, int(np.log2(stats['page_count'])))  # O(log n)
    
    cache_configs = [
        ('O(n) Full Cache', optimal_cache, 'green'),
        ('O(√n) Cache', sqrt_cache, 'orange'),
        ('O(log n) Cache', log_cache, 'red'),
        ('O(1) Minimal', 5, 'darkred')
    ]
    
    print(f"\nCache Configurations:")
    for label, size, _ in cache_configs:
        size_mb = size * stats['page_size'] / 1024 / 1024
        pct = (size / stats['page_count']) * 100
        print(f"  {label}: {size} pages ({size_mb:.1f} MB, {pct:.1f}% of DB)")
    
    # Run experiments with multiple trials
    results = []
    num_trials = 5
    
    for label, cache_size, color in cache_configs:
        print(f"\nTesting {label}...")
        
        trial_results = []
        for trial in range(num_trials):
            if trial > 0:
                # Clear OS cache between trials
                dummy = os.urandom(20 * 1024 * 1024)
                del dummy
            
            result = exp.run_queries(cache_size, num_queries=100)
            trial_results.append(result)
            
            if trial == 0:
                print(f"  Point lookup: {result['avg_point_lookup']*1000:.3f} ms")
                print(f"  Range scan: {result['avg_range_scan']*1000:.3f} ms")
                print(f"  Join query: {result['avg_join']*1000:.3f} ms")
        
        # Average across trials
        avg_result = {
            'label': label,
            'cache_size': cache_size,
            'color': color,
            'point_lookup': np.mean([r['avg_point_lookup'] for r in trial_results]),
            'range_scan': np.mean([r['avg_range_scan'] for r in trial_results]),
            'join': np.mean([r['avg_join'] for r in trial_results]),
            'point_lookup_std': np.std([r['avg_point_lookup'] for r in trial_results]),
            'range_scan_std': np.std([r['avg_range_scan'] for r in trial_results]),
            'join_std': np.std([r['avg_join'] for r in trial_results])
        }
        results.append(avg_result)
    
    # Calculate slowdown factors
    base_time = results[0]['point_lookup']  # O(n) cache baseline
    for r in results:
        r['slowdown'] = r['point_lookup'] / base_time
    
    # Create visualization
    create_paper_quality_plot(results, stats)
    
    # Save results
    exp_data = {
        'database_size_mb': db_size / 1024 / 1024,
        'page_count': stats['page_count'],
        'num_users': num_users,
        'cache_configs': [
            {
                'label': r['label'],
                'cache_pages': r['cache_size'],
                'cache_mb': r['cache_size'] * stats['page_size'] / 1024 / 1024,
                'avg_lookup_ms': r['point_lookup'] * 1000,
                'slowdown': r['slowdown']
            }
            for r in results
        ]
    }
    
    with open('sqlite_experiment_results.json', 'w') as f:
        json.dump(exp_data, f, indent=2)
    
    exp.cleanup()
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    for r in results:
        print(f"{r['label']:20} | Slowdown: {r['slowdown']:6.1f}x | "
              f"Lookup: {r['point_lookup']*1000:6.3f} ms")
    
    print("\nFiles generated:")
    print("  - sqlite_spacetime_tradeoff.png")
    print("  - sqlite_experiment_results.json")
    print("="*60)

def create_paper_quality_plot(results, stats):
    """Create publication-quality figure showing space-time tradeoff"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left plot: Performance vs Cache Size
    cache_sizes = [r['cache_size'] for r in results]
    cache_mb = [c * stats['page_size'] / 1024 / 1024 for c in cache_sizes]
    lookup_times = [r['point_lookup'] * 1000 for r in results]
    colors = [r['color'] for r in results]
    
    # Add error bars
    lookup_errors = [r['point_lookup_std'] * 1000 * 1.96 for r in results]  # 95% CI
    
    ax1.errorbar(cache_mb, lookup_times, yerr=lookup_errors,
                 fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=10)
    
    # Color individual points
    for i, (x, y, c) in enumerate(zip(cache_mb, lookup_times, colors)):
        ax1.scatter(x, y, color=c, s=100, zorder=5)
    
    # Add labels
    for i, r in enumerate(results):
        ax1.annotate(r['label'].split()[0], 
                    (cache_mb[i], lookup_times[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10)
    
    ax1.set_xlabel('Cache Size (MB)', fontsize=14)
    ax1.set_ylabel('Query Time (ms)', fontsize=14)
    ax1.set_title('(a) Query Performance vs Cache Size', fontsize=16)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Slowdown factors
    labels = [r['label'].replace(' Cache', '').replace(' ', '\n') for r in results]
    slowdowns = [r['slowdown'] for r in results]
    
    bars = ax2.bar(range(len(labels)), slowdowns, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, slowdowns):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}×', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=12)
    ax2.set_ylabel('Slowdown Factor', fontsize=14)
    ax2.set_title('(b) Space-Time Tradeoff in SQLite', fontsize=16)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add theoretical √n line
    ax2.axhline(y=np.sqrt(results[0]['cache_size'] / results[1]['cache_size']), 
                color='blue', linestyle='--', alpha=0.5, label='Theoretical √n')
    ax2.legend()
    
    plt.suptitle('SQLite Buffer Pool: Williams\' √n Pattern in Practice', fontsize=18)
    plt.tight_layout()
    plt.savefig('sqlite_spacetime_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    run_realistic_experiment()