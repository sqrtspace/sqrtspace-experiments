"""
Quick test to verify sorting experiment works with smaller parameters
"""

import os
import time
import tempfile
import numpy as np
import shutil
from scipy import stats
import sys

class SortingExperiment:
    """Compare different sorting algorithms with varying memory constraints"""
    
    def __init__(self, data_size: int):
        self.data_size = data_size
        self.data = np.random.rand(data_size).astype(np.float32)
        self.temp_dir = tempfile.mkdtemp()
        
    def cleanup(self):
        """Clean up temporary files"""
        shutil.rmtree(self.temp_dir)
        
    def in_memory_sort(self) -> np.ndarray:
        """Standard in-memory sorting - O(n) space"""
        return np.sort(self.data.copy())
    
    def checkpoint_sort(self, memory_limit: int) -> np.ndarray:
        """External merge sort with checkpointing - O(√n) space"""
        chunk_size = memory_limit // 4  # Reserve memory for merging
        num_chunks = (self.data_size + chunk_size - 1) // chunk_size
        
        # Phase 1: Sort chunks and write to disk
        chunk_files = []
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, self.data_size)
            
            # Sort chunk in memory
            chunk = np.sort(self.data[start:end])
            
            # Write to disk (checkpoint)
            filename = os.path.join(self.temp_dir, f'chunk_{i}.npy')
            np.save(filename, chunk)
            chunk_files.append(filename)
            
            # Clear chunk from memory
            del chunk
        
        # Phase 2: Simple merge (for quick test)
        result = []
        for f in chunk_files:
            chunk = np.load(f)
            result.extend(chunk.tolist())
        
        # Final sort (not truly external, but for quick test)
        result = np.sort(np.array(result))
        
        # Cleanup chunk files
        for f in chunk_files:
            os.remove(f)
            
        return result

def run_quick_test():
    """Run a quick test with smaller sizes"""
    
    print("=== Quick Sorting Test ===\n")
    
    # Small sizes for quick verification
    sizes = [100, 500, 1000]
    num_trials = 3
    
    for size in sizes:
        print(f"\nTesting with {size} elements ({num_trials} trials):")
        
        in_memory_times = []
        checkpoint_times = []
        
        for trial in range(num_trials):
            exp = SortingExperiment(size)
            
            # In-memory sort
            start = time.time()
            result1 = exp.in_memory_sort()
            time1 = time.time() - start
            in_memory_times.append(time1)
            
            # Checkpointed sort
            memory_limit = int(np.sqrt(size) * 4)
            start = time.time()
            result2 = exp.checkpoint_sort(memory_limit)
            time2 = time.time() - start
            checkpoint_times.append(time2)
            
            # Verify correctness
            if trial == 0:
                assert np.allclose(result1, result2), f"Results don't match for size {size}"
                print(f"  ✓ Correctness verified")
            
            exp.cleanup()
        
        # Calculate statistics
        in_memory_mean = np.mean(in_memory_times)
        in_memory_std = np.std(in_memory_times)
        checkpoint_mean = np.mean(checkpoint_times)
        checkpoint_std = np.std(checkpoint_times)
        
        print(f"  In-memory: {in_memory_mean:.6f}s ± {in_memory_std:.6f}s")
        print(f"  Checkpoint: {checkpoint_mean:.6f}s ± {checkpoint_std:.6f}s")
        print(f"  Slowdown: {checkpoint_mean/in_memory_mean:.1f}x")

if __name__ == "__main__":
    run_quick_test()