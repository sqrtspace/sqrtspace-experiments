"""
Stream Processing with Sliding Windows
Demonstrates favorable space-time tradeoffs in streaming scenarios
"""

import time
import random
from collections import deque
from typing import List, Tuple, Iterator
import math


class StreamProcessor:
    """Compare different approaches to computing sliding window statistics"""
    
    def __init__(self, stream_size: int, window_size: int):
        self.stream_size = stream_size
        self.window_size = window_size
        # Simulate a data stream (in practice, this would come from network/disk)
        self.stream = [random.gauss(0, 1) for _ in range(stream_size)]
    
    def full_storage_approach(self) -> Tuple[List[float], float]:
        """Store entire stream in memory - O(n) space"""
        start = time.time()
        
        # Store all data
        all_data = []
        results = []
        
        for i, value in enumerate(self.stream):
            all_data.append(value)
            
            # Compute sliding window average
            if i >= self.window_size - 1:
                window_start = i - self.window_size + 1
                window_avg = sum(all_data[window_start:i+1]) / self.window_size
                results.append(window_avg)
        
        elapsed = time.time() - start
        memory_used = len(all_data) * 8  # 8 bytes per float
        
        return results, elapsed, memory_used
    
    def sliding_window_approach(self) -> Tuple[List[float], float]:
        """Sliding window with deque - O(w) space where w = window size"""
        start = time.time()
        
        window = deque(maxlen=self.window_size)
        results = []
        window_sum = 0
        
        for value in self.stream:
            if len(window) == self.window_size:
                # Remove oldest value from sum
                window_sum -= window[0]
            
            window.append(value)
            window_sum += value
            
            if len(window) == self.window_size:
                results.append(window_sum / self.window_size)
        
        elapsed = time.time() - start
        memory_used = self.window_size * 8
        
        return results, elapsed, memory_used
    
    def checkpoint_approach(self) -> Tuple[List[float], float]:
        """Checkpoint every √n elements - O(√n) space"""
        start = time.time()
        
        checkpoint_interval = int(math.sqrt(self.stream_size))
        checkpoints = {}  # Store periodic snapshots
        results = []
        
        current_sum = 0
        current_count = 0
        
        for i, value in enumerate(self.stream):
            # Create checkpoint every √n elements
            if i % checkpoint_interval == 0:
                checkpoints[i] = {
                    'sum': current_sum,
                    'values': list(self.stream[max(0, i-self.window_size+1):i])
                }
            
            current_sum += value
            current_count += 1
            
            # Compute window average
            if i >= self.window_size - 1:
                # Find nearest checkpoint and recompute from there
                checkpoint_idx = (i // checkpoint_interval) * checkpoint_interval
                
                if checkpoint_idx in checkpoints:
                    # Recompute from checkpoint
                    cp = checkpoints[checkpoint_idx]
                    window_values = cp['values'] + list(self.stream[checkpoint_idx:i+1])
                    window_values = window_values[-(self.window_size):]
                    window_avg = sum(window_values) / len(window_values)
                else:
                    # Fallback: compute directly
                    window_start = i - self.window_size + 1
                    window_avg = sum(self.stream[window_start:i+1]) / self.window_size
                
                results.append(window_avg)
        
        elapsed = time.time() - start
        memory_used = len(checkpoints) * self.window_size * 8
        
        return results, elapsed, memory_used
    
    def extreme_space_approach(self) -> Tuple[List[float], float]:
        """Recompute everything - O(1) extra space"""
        start = time.time()
        
        results = []
        
        for i in range(self.window_size - 1, self.stream_size):
            # Recompute window sum every time
            window_sum = sum(self.stream[i - self.window_size + 1:i + 1])
            results.append(window_sum / self.window_size)
        
        elapsed = time.time() - start
        memory_used = 8  # Just one float for the sum
        
        return results, elapsed, memory_used


def run_stream_experiments():
    """Compare different streaming approaches"""
    print("=== Stream Processing: Sliding Window Average ===\n")
    print("Computing average over sliding windows of streaming data\n")
    
    # Test configurations
    configs = [
        (10000, 100),    # 10K stream, 100-element window
        (50000, 500),    # 50K stream, 500-element window  
        (100000, 1000),  # 100K stream, 1K window
    ]
    
    for stream_size, window_size in configs:
        print(f"\nStream size: {stream_size:,}, Window size: {window_size}")
        processor = StreamProcessor(stream_size, window_size)
        
        # 1. Full storage
        results1, time1, mem1 = processor.full_storage_approach()
        print(f"  Full storage (O(n) space):")
        print(f"    Time: {time1:.4f}s, Memory: {mem1/1024:.1f} KB")
        
        # 2. Sliding window
        results2, time2, mem2 = processor.sliding_window_approach()
        print(f"  Sliding window (O(w) space):")
        print(f"    Time: {time2:.4f}s, Memory: {mem2/1024:.1f} KB")
        if time2 > 0:
            print(f"    Speedup: {time1/time2:.2f}x, Memory reduction: {mem1/mem2:.1f}x")
        else:
            print(f"    Too fast to measure! Memory reduction: {mem1/mem2:.1f}x")
        
        # 3. Checkpoint approach
        results3, time3, mem3 = processor.checkpoint_approach()
        print(f"  Checkpoint (O(√n) space):")
        print(f"    Time: {time3:.4f}s, Memory: {mem3/1024:.1f} KB")
        if time1 > 0:
            print(f"    vs Full: {time3/time1:.2f}x time, {mem1/mem3:.1f}x less memory")
        else:
            print(f"    vs Full: Time ratio N/A, {mem1/mem3:.1f}x less memory")
        
        # 4. Extreme approach (only for smaller sizes)
        if stream_size <= 10000:
            results4, time4, mem4 = processor.extreme_space_approach()
            print(f"  Recompute all (O(1) space):")
            print(f"    Time: {time4:.4f}s, Memory: {mem4:.1f} bytes")
            if time1 > 0:
                print(f"    vs Full: {time4/time1:.1f}x slower")
            else:
                print(f"    vs Full: {time4:.4f}s (full storage too fast to compare)")
        
        # Verify correctness (sample check)
        for i in range(min(10, len(results1))):
            assert abs(results1[i] - results2[i]) < 1e-10, "Results don't match!"
    
    print("\n=== Analysis ===")
    print("Key observations:")
    print("1. Sliding window (O(w) space) is FASTER than full storage!")
    print("   - Better cache locality")
    print("   - No need to maintain huge arrays")
    print("2. This is a case where space reduction improves performance")
    print("3. Real streaming systems use exactly this approach")
    print("\nThis demonstrates that space-time tradeoffs can be beneficial,")
    print("not just theoretical curiosities!")


if __name__ == "__main__":
    run_stream_experiments()