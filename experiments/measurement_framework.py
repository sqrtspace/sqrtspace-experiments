"""
Standardized measurement framework for space-time tradeoff experiments.
Provides consistent metrics and visualization tools.
"""

import time
import psutil
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from typing import Callable, Any, List, Dict
from datetime import datetime


@dataclass
class Measurement:
    """Single measurement point"""
    timestamp: float
    memory_bytes: int
    cpu_percent: float
    
    
@dataclass
class ExperimentResult:
    """Results from a single experiment run"""
    algorithm: str
    input_size: int
    elapsed_time: float
    peak_memory: int
    average_memory: int
    measurements: List[Measurement]
    output: Any
    metadata: Dict[str, Any]
    

class SpaceTimeProfiler:
    """Profile space and time usage of algorithms"""
    
    def __init__(self, sample_interval: float = 0.01):
        self.sample_interval = sample_interval
        self.process = psutil.Process(os.getpid())
        
    def profile(self, func: Callable, *args, **kwargs) -> ExperimentResult:
        """Profile a function's execution"""
        measurements = []
        
        # Start monitoring in background
        import threading
        stop_monitoring = threading.Event()
        
        def monitor():
            while not stop_monitoring.is_set():
                measurements.append(Measurement(
                    timestamp=time.time(),
                    memory_bytes=self.process.memory_info().rss,
                    cpu_percent=self.process.cpu_percent(interval=0.01)
                ))
                time.sleep(self.sample_interval)
        
        monitor_thread = threading.Thread(target=monitor)
        monitor_thread.start()
        
        # Run the function
        start_time = time.time()
        try:
            output = func(*args, **kwargs)
        finally:
            stop_monitoring.set()
            monitor_thread.join()
        
        elapsed_time = time.time() - start_time
        
        # Calculate statistics
        memory_values = [m.memory_bytes for m in measurements]
        peak_memory = max(memory_values) if memory_values else 0
        average_memory = sum(memory_values) / len(memory_values) if memory_values else 0
        
        return ExperimentResult(
            algorithm=func.__name__,
            input_size=kwargs.get('input_size', 0),
            elapsed_time=elapsed_time,
            peak_memory=peak_memory,
            average_memory=int(average_memory),
            measurements=measurements,
            output=output,
            metadata=kwargs.get('metadata', {})
        )


class ExperimentRunner:
    """Run and compare multiple algorithms"""
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.results: List[ExperimentResult] = []
        self.profiler = SpaceTimeProfiler()
        
    def add_algorithm(self, func: Callable, input_sizes: List[int], 
                     name: str = None, **kwargs):
        """Run algorithm on multiple input sizes"""
        name = name or func.__name__
        
        for size in input_sizes:
            print(f"Running {name} with input size {size}...")
            result = self.profiler.profile(func, input_size=size, **kwargs)
            result.algorithm = name
            result.input_size = size
            self.results.append(result)
            
    def save_results(self, filename: str = None):
        """Save results to JSON file"""
        filename = filename or f"{self.experiment_name}_results.json"
        
        # Convert results to serializable format
        data = {
            'experiment': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'results': [
                {
                    **asdict(r),
                    'measurements': [asdict(m) for m in r.measurements[:100]]  # Limit measurements
                }
                for r in self.results
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
    def plot_space_time_curves(self, save_path: str = None):
        """Generate space-time tradeoff visualization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Group by algorithm
        algorithms = {}
        for r in self.results:
            if r.algorithm not in algorithms:
                algorithms[r.algorithm] = {'sizes': [], 'times': [], 'memory': []}
            algorithms[r.algorithm]['sizes'].append(r.input_size)
            algorithms[r.algorithm]['times'].append(r.elapsed_time)
            algorithms[r.algorithm]['memory'].append(r.peak_memory / 1024 / 1024)  # MB
        
        # Plot time complexity
        for alg, data in algorithms.items():
            ax1.plot(data['sizes'], data['times'], 'o-', label=alg, markersize=8)
        ax1.set_xlabel('Input Size (n)')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Time Complexity')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        
        # Plot space complexity
        for alg, data in algorithms.items():
            ax2.plot(data['sizes'], data['memory'], 's-', label=alg, markersize=8)
        ax2.set_xlabel('Input Size (n)')
        ax2.set_ylabel('Peak Memory (MB)')
        ax2.set_title('Space Complexity')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        
        # Only add theoretical bounds if they make sense for the experiment
        # (removed inappropriate √n bound for sorting algorithms that use O(1) space)
        
        plt.suptitle(f'{self.experiment_name}: Space-Time Tradeoff Analysis')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        else:
            plt.savefig(f"{self.experiment_name}_analysis.png", dpi=150)
        plt.close()
        
    def print_summary(self):
        """Print summary statistics"""
        print(f"\n=== {self.experiment_name} Results Summary ===\n")
        
        # Group by algorithm and size
        summary = {}
        for r in self.results:
            key = (r.algorithm, r.input_size)
            if key not in summary:
                summary[key] = []
            summary[key].append(r)
        
        # Print table
        print(f"{'Algorithm':<20} {'Size':<10} {'Time (s)':<12} {'Memory (MB)':<12} {'Time Ratio':<12}")
        print("-" * 70)
        
        baseline_times = {}
        for (alg, size), results in sorted(summary.items()):
            avg_time = sum(r.elapsed_time for r in results) / len(results)
            avg_memory = sum(r.peak_memory for r in results) / len(results) / 1024 / 1024
            
            # Store baseline (first algorithm) times
            if size not in baseline_times:
                baseline_times[size] = avg_time
            
            time_ratio = avg_time / baseline_times[size]
            
            print(f"{alg:<20} {size:<10} {avg_time:<12.4f} {avg_memory:<12.2f} {time_ratio:<12.2f}x")


# Example usage for testing
if __name__ == "__main__":
    # Test with simple sorting algorithms
    import random
    
    def bubble_sort(input_size: int, **kwargs):
        arr = [random.random() for _ in range(input_size)]
        n = len(arr)
        for i in range(n):
            for j in range(0, n-i-1):
                if arr[j] > arr[j+1]:
                    arr[j], arr[j+1] = arr[j+1], arr[j]
        return arr
    
    def python_sort(input_size: int, **kwargs):
        arr = [random.random() for _ in range(input_size)]
        return sorted(arr)
    
    runner = ExperimentRunner("Sorting Comparison")
    runner.add_algorithm(python_sort, [100, 500, 1000], name="Built-in Sort")
    runner.add_algorithm(bubble_sort, [100, 500, 1000], name="Bubble Sort")
    
    runner.print_summary()
    runner.plot_space_time_curves()
    runner.save_results()