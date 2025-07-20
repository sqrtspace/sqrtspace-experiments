# Distributed Computing: Space-Time Tradeoffs at Scale

## Overview
Distributed systems make explicit decisions about replication (space) vs computation (time). Every major distributed framework embodies these tradeoffs.

## 1. MapReduce / Hadoop

### Shuffle Phase - The Classic Tradeoff
```java
// Map output: Written to local disk (space for fault tolerance)
map(key, value):
    for word in value.split():
        emit(word, 1)

// Shuffle: All-to-all communication
// Choice: Buffer in memory vs spill to disk
shuffle.memory.ratio = 0.7  // 70% of heap for shuffle
shuffle.spill.percent = 0.8 // Spill when 80% full
```

**Memory Settings Impact:**
- High memory: Fast shuffle, risk of OOM
- Low memory: Frequent spills, 10x slower
- Sweet spot: √(data_size) memory per node

### Combiner Optimization
```java
// Without combiner: Send all data
map: (word, 1), (word, 1), (word, 1)...

// With combiner: Local aggregation (compute for space)
combine: (word, 3)

// Network transfer: 100x reduction
// CPU cost: Local sum computation
```

## 2. Apache Spark

### RDD Persistence Levels
```scala
// MEMORY_ONLY: Fast but memory intensive
rdd.persist(StorageLevel.MEMORY_ONLY)
// Space: Full dataset in RAM
// Time: Instant access

// MEMORY_AND_DISK: Spill to disk when needed
rdd.persist(StorageLevel.MEMORY_AND_DISK)
// Space: Min(dataset, available_ram)
// Time: RAM-speed or disk-speed

// DISK_ONLY: Minimal memory
rdd.persist(StorageLevel.DISK_ONLY)
// Space: O(1) RAM
// Time: Always disk I/O

// MEMORY_ONLY_SER: Serialized in memory
rdd.persist(StorageLevel.MEMORY_ONLY_SER)
// Space: 2-5x reduction via serialization
// Time: CPU cost to deserialize
```

### Broadcast Variables
```scala
// Without broadcast: Send to each task
val bigData = loadBigDataset() // 1GB
rdd.map(x => doSomething(x, bigData))
// Network: 1GB × num_tasks

// With broadcast: Send once per node
val bcData = sc.broadcast(bigData)
rdd.map(x => doSomething(x, bcData.value))
// Network: 1GB × num_nodes
// Memory: Extra copy per node
```

## 3. Distributed Key-Value Stores

### Redis Eviction Policies
```conf
# No eviction: Fail when full (pure space)
maxmemory-policy noeviction

# LRU: Recompute evicted data (time for space)
maxmemory-policy allkeys-lru
maxmemory 10gb

# LFU: Better hit rate, more CPU
maxmemory-policy allkeys-lfu
```

### Memcached Slab Allocation
- Fixed-size slabs: Internal fragmentation (waste space)
- Variable-size: External fragmentation (CPU to compact)
- Typical: √n slab classes for n object sizes

## 4. Kafka / Stream Processing

### Log Compaction
```properties
# Keep all messages (max space)
cleanup.policy=none

# Keep only latest per key (compute to save space)
cleanup.policy=compact
min.compaction.lag.ms=86400000

# Compression (CPU for space)
compression.type=lz4  # 4x space reduction
compression.type=zstd # 6x reduction, more CPU
```

### Consumer Groups
- Replicate processing: Each consumer gets all data
- Partition assignment: Each message processed once
- Tradeoff: Redundancy vs coordination overhead

## 5. Kubernetes / Container Orchestration

### Resource Requests vs Limits
```yaml
resources:
  requests:
    memory: "256Mi"  # Guaranteed (space reservation)
    cpu: "250m"      # Guaranteed (time reservation)
  limits:
    memory: "512Mi"  # Max before OOM
    cpu: "500m"      # Max before throttling
```

### Image Layer Caching
- Base images: Shared across containers (dedup space)
- Layer reuse: Fast container starts
- Tradeoff: Registry space vs pull time

## 6. Distributed Consensus

### Raft Log Compaction
```go
// Snapshot periodically to bound log size
if logSize > maxLogSize {
    snapshot = createSnapshot(stateMachine)
    truncateLog(snapshot.index)
}
// Space: O(snapshot) instead of O(all_operations)
// Time: Recreate state from snapshot + recent ops
```

### Multi-Paxos vs Raft
- Multi-Paxos: Less memory, complex recovery
- Raft: More memory (full log), simple recovery
- Tradeoff: Space vs implementation complexity

## 7. Content Delivery Networks (CDNs)

### Edge Caching Strategy
```nginx
# Cache everything (max space)
proxy_cache_valid 200 30d;
proxy_cache_max_size 100g;

# Cache popular only (compute popularity)
proxy_cache_min_uses 3;
proxy_cache_valid 200 1h;
proxy_cache_max_size 10g;
```

### Geographic Replication
- Full replication: Every edge has all content
- Lazy pull: Fetch on demand
- Predictive push: ML models predict demand

## 8. Batch Processing Frameworks

### Apache Flink Checkpointing
```java
// Checkpoint frequency (space vs recovery time)
env.enableCheckpointing(10000); // Every 10 seconds

// State backend choice
env.setStateBackend(new FsStateBackend("hdfs://..."));
// vs
env.setStateBackend(new RocksDBStateBackend("file://..."));

// RocksDB: Spill to disk, slower access
// Memory: Fast access, limited size
```

### Watermark Strategies
- Perfect watermarks: Buffer all late data (space)
- Heuristic watermarks: Drop some late data (accuracy for space)
- Allowed lateness: Bounded buffer

## 9. Real-World Examples

### Google's MapReduce (2004)
- Problem: Processing 20TB of web data
- Solution: Trade disk space for fault tolerance
- Impact: 1000 machines × 3 hours vs 1 machine × 3000 hours

### Facebook's TAO (2013)
- Problem: Social graph queries
- Solution: Replicate to every datacenter
- Tradeoff: Petabytes of RAM for microsecond latency

### Amazon's Dynamo (2007)
- Problem: Shopping cart availability
- Solution: Eventually consistent, multi-version
- Tradeoff: Space for conflict resolution

## 10. Optimization Patterns

### Hierarchical Aggregation
```python
# Naive: All-to-one
results = []
for worker in workers:
    results.extend(worker.compute())
return aggregate(results)  # Bottleneck!

# Tree aggregation: √n levels
level1 = [aggregate(chunk) for chunk in chunks(workers, sqrt(n))]
level2 = [aggregate(chunk) for chunk in chunks(level1, sqrt(n))]
return aggregate(level2)

# Space: O(√n) intermediate results
# Time: O(log n) vs O(n)
```

### Bloom Filters in Distributed Joins
```java
// Broadcast join with Bloom filter
BloomFilter filter = createBloomFilter(smallTable);
broadcast(filter);

// Each node filters locally
bigTable.filter(row -> filter.mightContain(row.key))
        .join(broadcastedSmallTable);

// Space: O(m log n) bits for filter
// Reduction: 99% fewer network transfers
```

## Key Insights

1. **Every distributed system** trades replication for computation
2. **The √n pattern** appears in:
   - Shuffle buffer sizes
   - Checkpoint frequencies  
   - Aggregation tree heights
   - Cache sizes

3. **Network is the new disk**:
   - Network transfer ≈ Disk I/O in cost
   - Same space-time tradeoffs apply

4. **Failures force space overhead**:
   - Replication for availability
   - Checkpointing for recovery
   - Logging for consistency

## Connection to Williams' Result

Distributed systems naturally implement √n algorithms:
- Shuffle phases: O(√n) memory per node optimal
- Aggregation trees: O(√n) height minimizes time
- Cache sizing: √(total_data) per node common

These patterns emerge independently across systems, validating the fundamental nature of the √(t log t) space bound for time-t computations.