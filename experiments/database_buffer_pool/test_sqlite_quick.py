"""Quick test of SQLite experiment with small data"""

from sqlite_buffer_pool_experiment import SQLiteExperiment
import numpy as np

def quick_test():
    print("=== Quick SQLite Test ===")
    
    # Small test
    num_users = 1000
    exp = SQLiteExperiment(num_users)
    
    print(f"\nSetting up database with {num_users} users...")
    db_size = exp.setup_database()
    stats = exp.analyze_page_distribution()
    
    print(f"Database size: {db_size / 1024:.1f} KB")
    print(f"Total pages: {stats['page_count']}")
    
    # Test three cache sizes
    cache_sizes = [
        ('Full', stats['page_count']),
        ('√n', int(np.sqrt(stats['page_count']))),
        ('Minimal', 5)
    ]
    
    for label, cache_size in cache_sizes:
        print(f"\n{label} cache: {cache_size} pages")
        result = exp.run_queries(cache_size, num_queries=10)
        print(f"  Avg lookup: {result['avg_point_lookup']*1000:.2f} ms")
        print(f"  Avg scan: {result['avg_range_scan']*1000:.2f} ms")
    
    exp.cleanup()
    print("\n✓ Test completed successfully!")

if __name__ == "__main__":
    quick_test()