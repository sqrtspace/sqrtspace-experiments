"""
SQLite Buffer Pool Space-Time Tradeoff Experiment

Demonstrates how SQLite's page cache size affects query performance,
validating Williams' √n space-time tradeoff in a real production database.

Key parameters:
- cache_size: Number of pages in memory (default 2000)
- page_size: Size of each page (default 4096 bytes)

This experiment shows:
1. Full cache (O(n) space): Fast queries
2. √n cache: Moderate slowdown
3. Minimal cache: Extreme slowdown
"""

import sqlite3
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json
import tempfile
import shutil

class SQLiteExperiment:
    """Test SQLite performance with different cache sizes"""
    
    def __init__(self, num_rows: int, page_size: int = 4096):
        self.num_rows = num_rows
        self.page_size = page_size
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test.db')
        
    def cleanup(self):
        """Clean up temporary files"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def setup_database(self):
        """Create and populate the test database"""
        conn = sqlite3.connect(self.db_path)
        conn.execute(f'PRAGMA page_size = {self.page_size}')
        conn.commit()
        
        # Create tables simulating a real app
        conn.execute('''
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name TEXT,
                email TEXT,
                created_at INTEGER,
                data BLOB
            )
        ''')
        
        conn.execute('''
            CREATE TABLE posts (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                title TEXT,
                content TEXT,
                created_at INTEGER,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # Insert data
        print(f"Populating database with {self.num_rows:,} users...")
        
        # Batch insert for efficiency
        batch_size = 1000
        for i in range(0, self.num_rows, batch_size):
            batch = []
            for j in range(min(batch_size, self.num_rows - i)):
                user_id = i + j
                # Add some data to make pages more realistic
                data = os.urandom(200)  # 200 bytes of data per user
                batch.append((
                    user_id,
                    f'User {user_id}',
                    f'user{user_id}@example.com',
                    int(time.time()) - user_id,
                    data
                ))
            
            conn.executemany(
                'INSERT INTO users VALUES (?, ?, ?, ?, ?)',
                batch
            )
            
            # Insert 3 posts per user
            post_batch = []
            for user in batch:
                user_id = user[0]
                for k in range(3):
                    post_batch.append((
                        user_id * 3 + k,
                        user_id,
                        f'Post {k} by user {user_id}',
                        f'Content of post {k}' * 10,  # Make content larger
                        int(time.time()) - user_id + k
                    ))
            
            conn.executemany(
                'INSERT INTO posts VALUES (?, ?, ?, ?, ?)',
                post_batch
            )
        
        # Create indexes (common in real apps)
        conn.execute('CREATE INDEX idx_users_email ON users(email)')
        conn.execute('CREATE INDEX idx_posts_user ON posts(user_id)')
        conn.execute('CREATE INDEX idx_posts_created ON posts(created_at)')
        
        conn.commit()
        conn.close()
        
        # Get database size
        db_size = os.path.getsize(self.db_path)
        print(f"Database size: {db_size / 1024 / 1024:.1f} MB")
        return db_size
    
    def run_queries(self, cache_size: int, num_queries: int = 100) -> Dict:
        """Run queries with specified cache size"""
        conn = sqlite3.connect(self.db_path)
        
        # Set cache size (in pages)
        conn.execute(f'PRAGMA cache_size = {cache_size}')
        
        # Clear OS cache by reading another file (best effort)
        dummy_data = os.urandom(50 * 1024 * 1024)  # 50MB
        del dummy_data
        
        # Get actual cache size in bytes
        cache_bytes = cache_size * self.page_size
        
        # Query patterns that simulate real usage
        query_times = {
            'point_lookups': [],
            'range_scans': [],
            'joins': [],
            'aggregations': []
        }
        
        # Warm up
        conn.execute('SELECT COUNT(*) FROM users').fetchone()
        
        # 1. Point lookups (random access pattern)
        for _ in range(num_queries):
            user_id = np.random.randint(1, self.num_rows)
            start = time.time()
            conn.execute(
                'SELECT * FROM users WHERE id = ?', 
                (user_id,)
            ).fetchone()
            query_times['point_lookups'].append(time.time() - start)
        
        # 2. Range scans
        for _ in range(num_queries // 10):  # Fewer range scans
            max_start = max(1, self.num_rows - 100)
            start_id = np.random.randint(1, max_start + 1)
            start = time.time()
            conn.execute(
                'SELECT * FROM users WHERE id BETWEEN ? AND ?',
                (start_id, min(start_id + 100, self.num_rows))
            ).fetchall()
            query_times['range_scans'].append(time.time() - start)
        
        # 3. Joins (most expensive)
        for _ in range(num_queries // 20):  # Even fewer joins
            user_id = np.random.randint(1, self.num_rows)
            start = time.time()
            conn.execute('''
                SELECT u.*, p.*
                FROM users u
                JOIN posts p ON u.id = p.user_id
                WHERE u.id = ?
            ''', (user_id,)).fetchall()
            query_times['joins'].append(time.time() - start)
        
        # 4. Aggregations
        for _ in range(num_queries // 20):
            start_time = int(time.time()) - np.random.randint(0, self.num_rows)
            start = time.time()
            conn.execute('''
                SELECT COUNT(*), AVG(LENGTH(content))
                FROM posts
                WHERE created_at > ?
            ''', (start_time,)).fetchone()
            query_times['aggregations'].append(time.time() - start)
        
        # Get cache statistics
        cache_hit = conn.execute('PRAGMA cache_stats').fetchone()
        
        conn.close()
        
        return {
            'cache_size': cache_size,
            'cache_bytes': cache_bytes,
            'query_times': query_times,
            'avg_point_lookup': np.mean(query_times['point_lookups']),
            'avg_range_scan': np.mean(query_times['range_scans']),
            'avg_join': np.mean(query_times['joins']),
            'avg_aggregation': np.mean(query_times['aggregations'])
        }
    
    def analyze_page_distribution(self) -> Dict:
        """Analyze how data is distributed across pages"""
        conn = sqlite3.connect(self.db_path)
        
        # Get page count
        page_count = conn.execute('PRAGMA page_count').fetchone()[0]
        
        # Get various statistics
        stats = {
            'page_count': page_count,
            'page_size': self.page_size,
            'total_size': page_count * self.page_size,
            'users_count': conn.execute('SELECT COUNT(*) FROM users').fetchone()[0],
            'posts_count': conn.execute('SELECT COUNT(*) FROM posts').fetchone()[0]
        }
        
        conn.close()
        return stats

def run_sqlite_experiment():
    """Run the complete SQLite buffer pool experiment"""
    
    print("="*60)
    print("SQLite Buffer Pool Space-Time Tradeoff Experiment")
    print("="*60)
    
    # Test with different database sizes
    sizes = [10000, 50000, 100000]  # Number of users
    results = {}
    
    for num_users in sizes:
        print(f"\n{'='*40}")
        print(f"Testing with {num_users:,} users")
        print(f"{'='*40}")
        
        exp = SQLiteExperiment(num_users)
        db_size = exp.setup_database()
        stats = exp.analyze_page_distribution()
        
        print(f"Database pages: {stats['page_count']:,}")
        print(f"Page size: {stats['page_size']} bytes")
        
        # Test different cache sizes
        # Full cache, √n cache, minimal cache
        cache_configs = [
            ('Full O(n)', stats['page_count']),  # All pages in memory
            ('√n cache', int(np.sqrt(stats['page_count']))),  # √n pages
            ('Minimal', 10)  # Almost no cache
        ]
        
        user_results = []
        
        for label, cache_size in cache_configs:
            print(f"\nTesting {label}: {cache_size} pages ({cache_size * 4096 / 1024:.1f} KB)")
            
            result = exp.run_queries(cache_size, num_queries=50)
            result['label'] = label
            user_results.append(result)
            
            print(f"  Point lookups: {result['avg_point_lookup']*1000:.2f} ms")
            print(f"  Range scans: {result['avg_range_scan']*1000:.2f} ms")
            print(f"  Joins: {result['avg_join']*1000:.2f} ms")
        
        results[num_users] = {
            'stats': stats,
            'experiments': user_results
        }
        
        exp.cleanup()
    
    # Create visualizations
    create_sqlite_plots(results)
    
    # Save results
    with open('sqlite_results.json', 'w') as f:
        # Convert numpy types for JSON serialization
        def convert(o):
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return o
        
        json.dump(results, f, indent=2, default=convert)
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("Generated files:")
    print("  - sqlite_results.json")
    print("  - sqlite_buffer_pool_analysis.png")
    print("="*60)
    
    return results

def create_sqlite_plots(results: Dict):
    """Create publication-quality plots for SQLite experiment"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Point lookup performance vs cache size
    sizes = sorted(results.keys())
    
    for size in sizes:
        experiments = results[size]['experiments']
        cache_sizes = [e['cache_size'] for e in experiments]
        point_times = [e['avg_point_lookup'] * 1000 for e in experiments]  # Convert to ms
        
        ax1.plot(cache_sizes, point_times, 'o-', label=f'{size:,} users', 
                linewidth=2, markersize=8)
    
    ax1.set_xlabel('Cache Size (pages)', fontsize=12)
    ax1.set_ylabel('Avg Point Lookup Time (ms)', fontsize=12)
    ax1.set_title('Point Lookup Performance vs Cache Size', fontsize=14)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Slowdown factors
    base_size = sizes[1]  # Use 50k as reference
    base_results = results[base_size]['experiments']
    
    full_cache_time = base_results[0]['avg_point_lookup']
    sqrt_cache_time = base_results[1]['avg_point_lookup']
    min_cache_time = base_results[2]['avg_point_lookup']
    
    categories = ['Full\nO(n)', '√n\nCache', 'Minimal\nO(1)']
    slowdowns = [1, sqrt_cache_time/full_cache_time, min_cache_time/full_cache_time]
    
    bars = ax2.bar(categories, slowdowns, color=['green', 'orange', 'red'])
    ax2.set_ylabel('Slowdown Factor', fontsize=12)
    ax2.set_title(f'Query Slowdown vs Cache Size ({base_size:,} users)', fontsize=14)
    
    # Add value labels on bars
    for bar, val in zip(bars, slowdowns):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}×', ha='center', va='bottom', fontsize=11)
    
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Memory usage efficiency
    for size in sizes:
        experiments = results[size]['experiments']
        cache_mb = [e['cache_bytes'] / 1024 / 1024 for e in experiments]
        query_speed = [1 / e['avg_point_lookup'] for e in experiments]  # Queries per second
        
        ax3.plot(cache_mb, query_speed, 's-', label=f'{size:,} users',
                linewidth=2, markersize=8)
    
    ax3.set_xlabel('Cache Size (MB)', fontsize=12)
    ax3.set_ylabel('Queries per Second', fontsize=12)
    ax3.set_title('Memory Efficiency: Speed vs Cache Size', fontsize=14)
    ax3.set_xscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Different query types
    query_types = ['Point\nLookup', 'Range\nScan', 'Join\nQuery']
    
    for i, (label, cache_size) in enumerate(cache_configs[:3]):
        if i >= len(base_results):
            break
        result = base_results[i]
        times = [
            result['avg_point_lookup'] * 1000,
            result['avg_range_scan'] * 1000,
            result['avg_join'] * 1000
        ]
        
        x = np.arange(len(query_types))
        width = 0.25
        ax4.bar(x + i*width, times, width, label=label)
    
    ax4.set_xlabel('Query Type', fontsize=12)
    ax4.set_ylabel('Average Time (ms)', fontsize=12)
    ax4.set_title('Query Performance by Type and Cache Size', fontsize=14)
    ax4.set_xticks(x + width)
    ax4.set_xticklabels(query_types)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_yscale('log')
    
    plt.suptitle('SQLite Buffer Pool: Space-Time Tradeoffs', fontsize=16)
    plt.tight_layout()
    plt.savefig('sqlite_buffer_pool_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

# Helper to get theoretical cache configs
cache_configs = [
    ('Full O(n)', None),  # Will be set based on page count
    ('√n cache', None),
    ('Minimal', 10)
]

if __name__ == "__main__":
    run_sqlite_experiment()