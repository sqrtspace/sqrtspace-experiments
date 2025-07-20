"""
SQLite experiment with heavier workload to demonstrate space-time tradeoffs
Uses larger data and more complex queries to stress the buffer pool
"""

import sqlite3
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import tempfile
import shutil
import gc

class SQLiteHeavyExperiment:
    """SQLite experiment with larger data to force real I/O"""
    
    def __init__(self, scale_factor: int = 100000):
        self.scale_factor = scale_factor
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'heavy.db')
        
    def cleanup(self):
        """Clean up temporary files"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def setup_database(self):
        """Create a database that's too large for small caches"""
        conn = sqlite3.connect(self.db_path)
        
        # Use larger pages for efficiency
        conn.execute('PRAGMA page_size = 8192')
        conn.execute('PRAGMA journal_mode = WAL')  # Write-ahead logging
        conn.commit()
        
        # Create tables that simulate real-world complexity
        conn.execute('''
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                title TEXT,
                content TEXT,
                tags TEXT,
                created_at INTEGER,
                updated_at INTEGER,
                view_count INTEGER,
                data BLOB
            )
        ''')
        
        conn.execute('''
            CREATE TABLE analytics (
                id INTEGER PRIMARY KEY,
                doc_id INTEGER,
                event_type TEXT,
                user_id INTEGER,
                timestamp INTEGER,
                metadata TEXT,
                FOREIGN KEY (doc_id) REFERENCES documents(id)
            )
        ''')
        
        print(f"Populating database (this will take a moment)...")
        
        # Insert documents with realistic data
        batch_size = 1000
        total_docs = self.scale_factor
        
        for i in range(0, total_docs, batch_size):
            batch = []
            for j in range(min(batch_size, total_docs - i)):
                doc_id = i + j
                # Create variable-length content to simulate real documents
                content_length = np.random.randint(100, 2000)
                content = 'x' * content_length  # Simplified for speed
                
                # Random binary data to increase row size
                data_size = np.random.randint(500, 2000)
                data = os.urandom(data_size)
                
                batch.append((
                    doc_id,
                    np.random.randint(1, 10000),  # user_id
                    f'Document {doc_id}',
                    content,
                    f'tag{doc_id % 100},tag{doc_id % 50}',
                    int(time.time()) - doc_id,
                    int(time.time()) - doc_id // 2,
                    np.random.randint(0, 10000),
                    data
                ))
            
            conn.executemany(
                'INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                batch
            )
            
            # Insert analytics events (3-5 per document)
            analytics_batch = []
            for doc in batch:
                doc_id = doc[0]
                num_events = np.random.randint(3, 6)
                for k in range(num_events):
                    analytics_batch.append((
                        doc_id * 5 + k,
                        doc_id,
                        np.random.choice(['view', 'click', 'share', 'like']),
                        np.random.randint(1, 10000),
                        int(time.time()) - np.random.randint(0, 86400 * 30),
                        f'{{"source": "web", "version": {k}}}'
                    ))
            
            conn.executemany(
                'INSERT INTO analytics VALUES (?, ?, ?, ?, ?, ?)',
                analytics_batch
            )
            
            if (i + batch_size) % 10000 == 0:
                print(f"  Inserted {i + batch_size:,} / {total_docs:,} documents...")
                conn.commit()
        
        # Create indexes to make queries more realistic
        print("Creating indexes...")
        conn.execute('CREATE INDEX idx_docs_user ON documents(user_id)')
        conn.execute('CREATE INDEX idx_docs_created ON documents(created_at)')
        conn.execute('CREATE INDEX idx_analytics_doc ON analytics(doc_id)')
        conn.execute('CREATE INDEX idx_analytics_time ON analytics(timestamp)')
        
        conn.commit()
        
        # Analyze to update statistics
        conn.execute('ANALYZE')
        conn.close()
        
        # Get database size
        db_size = os.path.getsize(self.db_path)
        print(f"Database size: {db_size / 1024 / 1024:.1f} MB")
        
        return db_size
    
    def force_cache_clear(self):
        """Try to clear OS cache"""
        # Allocate and access large memory to evict cache
        try:
            dummy = np.zeros((100, 1024, 1024), dtype=np.uint8)  # 100MB
            dummy[:] = np.random.randint(0, 256, size=dummy.shape, dtype=np.uint8)
            del dummy
            gc.collect()
        except:
            pass
    
    def run_heavy_queries(self, cache_pages: int) -> dict:
        """Run queries that stress the cache"""
        conn = sqlite3.connect(self.db_path)
        
        # Set cache size
        conn.execute(f'PRAGMA cache_size = -{cache_pages * 8}')  # Negative = KB
        
        # Disable query optimizer shortcuts
        conn.execute('PRAGMA query_only = ON')
        
        results = {
            'random_reads': [],
            'sequential_scan': [],
            'complex_join': [],
            'aggregation': []
        }
        
        # 1. Random point queries (cache-unfriendly)
        print("  Running random reads...")
        for _ in range(50):
            doc_id = np.random.randint(1, self.scale_factor)
            start = time.time()
            conn.execute(
                'SELECT * FROM documents WHERE id = ?', 
                (doc_id,)
            ).fetchone()
            results['random_reads'].append(time.time() - start)
        
        # 2. Sequential scan with filter
        print("  Running sequential scans...")
        for _ in range(5):
            min_views = np.random.randint(1000, 5000)
            start = time.time()
            conn.execute(
                'SELECT COUNT(*) FROM documents WHERE view_count > ?',
                (min_views,)
            ).fetchone()
            results['sequential_scan'].append(time.time() - start)
        
        # 3. Complex join queries
        print("  Running complex joins...")
        for _ in range(5):
            user_id = np.random.randint(1, 10000)
            start = time.time()
            conn.execute('''
                SELECT d.*, COUNT(a.id) as events
                FROM documents d
                LEFT JOIN analytics a ON d.id = a.doc_id
                WHERE d.user_id = ?
                GROUP BY d.id
                LIMIT 10
            ''', (user_id,)).fetchall()
            results['complex_join'].append(time.time() - start)
        
        # 4. Time-based aggregation
        print("  Running aggregations...")
        for _ in range(5):
            days_back = np.random.randint(1, 30)
            start_time = int(time.time()) - (days_back * 86400)
            start = time.time()
            conn.execute('''
                SELECT 
                    event_type,
                    COUNT(*) as count,
                    COUNT(DISTINCT user_id) as unique_users
                FROM analytics
                WHERE timestamp > ?
                GROUP BY event_type
            ''', (start_time,)).fetchall()
            results['aggregation'].append(time.time() - start)
        
        conn.close()
        
        return {
            'cache_pages': cache_pages,
            'avg_random_read': np.mean(results['random_reads']),
            'avg_sequential': np.mean(results['sequential_scan']),
            'avg_join': np.mean(results['complex_join']),
            'avg_aggregation': np.mean(results['aggregation']),
            'p95_random_read': np.percentile(results['random_reads'], 95),
            'raw_results': results
        }

def run_heavy_experiment():
    """Run the heavy SQLite experiment"""
    
    print("="*60)
    print("SQLite Heavy Workload Experiment")
    print("Demonstrating space-time tradeoffs with real I/O pressure")
    print("="*60)
    
    # Create large database
    scale = 50000  # 50k documents = ~200MB database
    exp = SQLiteHeavyExperiment(scale)
    
    db_size = exp.setup_database()
    
    # Calculate page count
    page_size = 8192
    total_pages = db_size // page_size
    
    print(f"\nDatabase created:")
    print(f"  Documents: {scale:,}")
    print(f"  Size: {db_size / 1024 / 1024:.1f} MB")
    print(f"  Pages: {total_pages:,}")
    
    # Test different cache sizes
    cache_configs = [
        ('O(n) Full', min(total_pages, 10000)),  # Cap at 10k pages for memory
        ('O(√n)', int(np.sqrt(total_pages))),
        ('O(log n)', int(np.log2(total_pages))),
        ('O(1)', 10)
    ]
    
    results = []
    
    for label, cache_pages in cache_configs:
        cache_mb = cache_pages * page_size / 1024 / 1024
        print(f"\nTesting {label}: {cache_pages} pages ({cache_mb:.1f} MB)")
        
        # Clear cache between runs
        exp.force_cache_clear()
        time.sleep(1)  # Let system settle
        
        result = exp.run_heavy_queries(cache_pages)
        result['label'] = label
        result['cache_mb'] = cache_mb
        results.append(result)
        
        print(f"  Random read: {result['avg_random_read']*1000:.2f} ms")
        print(f"  Sequential: {result['avg_sequential']*1000:.2f} ms")
        print(f"  Complex join: {result['avg_join']*1000:.2f} ms")
    
    # Create visualization
    create_heavy_experiment_plot(results, db_size)
    
    # Calculate slowdowns
    base = results[0]['avg_random_read']
    for r in results:
        r['slowdown'] = r['avg_random_read'] / base
    
    # Save results
    with open('sqlite_heavy_results.json', 'w') as f:
        save_data = {
            'scale_factor': scale,
            'db_size_mb': db_size / 1024 / 1024,
            'results': [
                {
                    'label': r['label'],
                    'cache_mb': r['cache_mb'],
                    'avg_random_ms': r['avg_random_read'] * 1000,
                    'slowdown': r['slowdown']
                }
                for r in results
            ]
        }
        json.dump(save_data, f, indent=2)
    
    exp.cleanup()
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    for r in results:
        print(f"{r['label']:15} | Slowdown: {r['slowdown']:6.1f}x | "
              f"Random: {r['avg_random_read']*1000:6.2f} ms | "
              f"Join: {r['avg_join']*1000:6.2f} ms")
    
    print("\nFiles generated:")
    print("  - sqlite_heavy_experiment.png")
    print("  - sqlite_heavy_results.json")
    print("="*60)

def create_heavy_experiment_plot(results, db_size):
    """Create plot for heavy experiment"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract data
    labels = [r['label'] for r in results]
    cache_mb = [r['cache_mb'] for r in results]
    random_times = [r['avg_random_read'] * 1000 for r in results]
    join_times = [r['avg_join'] * 1000 for r in results]
    
    # Plot 1: Random read performance
    colors = ['green', 'orange', 'red', 'darkred']
    ax1.bar(labels, random_times, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Time (ms)', fontsize=12)
    ax1.set_title('Random Read Performance', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(ax1.patches, random_times)):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Join query performance
    ax2.bar(labels, join_times, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Time (ms)', fontsize=12)
    ax2.set_title('Complex Join Performance', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Cache efficiency
    db_mb = db_size / 1024 / 1024
    cache_pct = [(c / db_mb) * 100 for c in cache_mb]
    slowdowns = [r['avg_random_read'] / results[0]['avg_random_read'] for r in results]
    
    ax3.scatter(cache_pct, slowdowns, s=200, c=colors, edgecolor='black', linewidth=2)
    
    # Add theoretical √n curve
    x_theory = np.linspace(0.1, 100, 100)
    y_theory = 1 / np.sqrt(x_theory / 100)
    ax3.plot(x_theory, y_theory, 'b--', alpha=0.5, label='Theoretical 1/√x')
    
    ax3.set_xlabel('Cache Size (% of Database)', fontsize=12)
    ax3.set_ylabel('Slowdown Factor', fontsize=12)
    ax3.set_title('Space-Time Tradeoff', fontsize=14)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: All query types comparison
    query_types = ['Random\nRead', 'Sequential\nScan', 'Complex\nJoin', 'Aggregation']
    
    x = np.arange(len(query_types))
    width = 0.2
    
    for i, r in enumerate(results):
        times = [
            r['avg_random_read'] * 1000,
            r['avg_sequential'] * 1000,
            r['avg_join'] * 1000,
            r['avg_aggregation'] * 1000
        ]
        ax4.bar(x + i*width, times, width, label=r['label'], color=colors[i])
    
    ax4.set_xlabel('Query Type', fontsize=12)
    ax4.set_ylabel('Time (ms)', fontsize=12)
    ax4.set_title('Performance by Query Type', fontsize=14)
    ax4.set_xticks(x + width * 1.5)
    ax4.set_xticklabels(query_types)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_yscale('log')
    
    plt.suptitle('SQLite Buffer Pool: Heavy Workload Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig('sqlite_heavy_experiment.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    run_heavy_experiment()