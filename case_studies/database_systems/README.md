# Database Systems: Space-Time Tradeoffs in Practice

## Overview
Databases are perhaps the most prominent example of space-time tradeoffs in production systems. Every major database makes explicit decisions about trading memory for computation time.

## 1. Query Processing

### Hash Join vs Nested Loop Join

**Hash Join (More Memory)**
- Build hash table: O(n) space
- Probe phase: O(n+m) time
- Used when: Sufficient memory available
```sql
-- PostgreSQL will choose hash join if work_mem is high enough
SET work_mem = '256MB';
SELECT * FROM orders o JOIN customers c ON o.customer_id = c.id;
```

**Nested Loop Join (Less Memory)**
- Space: O(1) 
- Time: O(n×m)
- Used when: Memory constrained
```sql
-- Force nested loop with low work_mem
SET work_mem = '64kB';
```

### Real PostgreSQL Example
```sql
-- Monitor actual memory usage
EXPLAIN (ANALYZE, BUFFERS) 
SELECT * FROM large_table JOIN huge_table USING (id);

-- Output shows:
-- Hash Join: 145MB memory, 2.3 seconds
-- Nested Loop: 64KB memory, 487 seconds
```

## 2. Indexing Strategies

### B-Tree vs Full Table Scan
- **B-Tree Index**: O(n) space, O(log n) lookup
- **No Index**: O(1) extra space, O(n) scan time

### Covering Indexes
Trading more space for zero I/O reads:
```sql
-- Regular index: must fetch row data
CREATE INDEX idx_user_email ON users(email);

-- Covering index: all data in index (more space)
CREATE INDEX idx_user_email_covering ON users(email) INCLUDE (name, created_at);
```

## 3. Materialized Views

Ultimate space-for-time trade:
```sql
-- Compute once, store results
CREATE MATERIALIZED VIEW sales_summary AS
SELECT 
    date_trunc('day', sale_date) as day,
    product_id,
    SUM(amount) as total_sales,
    COUNT(*) as num_sales
FROM sales
GROUP BY 1, 2;

-- Instant queries vs recomputation
SELECT * FROM sales_summary WHERE day = '2024-01-15';  -- 1ms
-- vs
SELECT ... FROM sales GROUP BY ...;  -- 30 seconds
```

## 4. Buffer Pool Management

### PostgreSQL's shared_buffers
```
# Low memory: more disk I/O
shared_buffers = 128MB  # Frequent disk reads

# High memory: cache working set  
shared_buffers = 8GB    # Most data in RAM
```

Performance impact:
- 128MB: TPC-H query takes 45 minutes
- 8GB: Same query takes 3 minutes

## 5. Query Planning

### Bitmap Heap Scan
A perfect example of √n-like behavior:
1. Build bitmap of matching rows: O(√n) space
2. Scan heap in physical order: Better than random I/O
3. Falls between index scan and sequential scan

```sql
EXPLAIN SELECT * FROM orders WHERE status IN ('pending', 'processing');
-- Bitmap Heap Scan on orders
-- Recheck Cond: (status = ANY ('{pending,processing}'::text[]))
-- -> Bitmap Index Scan on idx_status
```

## 6. Write-Ahead Logging (WAL)

Trading write performance for durability:
- **Synchronous commit**: Every transaction waits for disk
- **Asynchronous commit**: Buffer writes, risk data loss
```sql
-- Trade durability for speed
SET synchronous_commit = off;  -- 10x faster inserts
```

## 7. Column Stores vs Row Stores

### Row Store (PostgreSQL, MySQL)
- Store complete rows together
- Good for OLTP, random access
- Space: Stores all columns even if not needed

### Column Store (ClickHouse, Vertica)  
- Store each column separately
- Excellent compression (less space)
- Must reconstruct rows (more time for some queries)

Example compression ratios:
- Row store: 100GB table
- Column store: 15GB (85% space savings)
- But: Random row lookup 100x slower

## 8. Real-World Configuration

### PostgreSQL Memory Settings
```conf
# Total system RAM: 64GB

# Aggressive caching (space for time)
shared_buffers = 16GB          # 25% of RAM
work_mem = 256MB               # Per operation
maintenance_work_mem = 2GB     # For VACUUM, CREATE INDEX

# Conservative (time for space)  
shared_buffers = 128MB         # Minimal caching
work_mem = 4MB                 # Forces disk-based operations
```

### MySQL InnoDB Buffer Pool
```conf
# 75% of RAM for buffer pool
innodb_buffer_pool_size = 48G

# Adaptive hash index (space for time)
innodb_adaptive_hash_index = ON
```

## 9. Distributed Databases

### Replication vs Computation
- **Full replication**: n× space, instant reads
- **No replication**: 1× space, distributed queries

### Cassandra's Space Amplification
- Replication factor 3: 3× space
- Plus SSTables: Another 2-3× during compaction
- Total: ~10× space for high availability

## Key Insights

1. **Every join algorithm** is a space-time tradeoff
2. **Indexes** are precomputed results (space for time)
3. **Buffer pools** cache hot data (space for I/O time)
4. **Query planners** explicitly optimize these tradeoffs
5. **DBAs tune memory** to control space-time balance

## Connection to Williams' Result

Databases naturally implement √n-like algorithms:
- Bitmap indexes: O(√n) space for range queries
- Sort-merge joins: O(√n) memory for external sort
- Buffer pool: Typically sized at √(database size)

The ubiquity of these patterns in database internals validates Williams' theoretical insights about the fundamental nature of space-time tradeoffs in computation.