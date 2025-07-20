import pandas as pd
import matplotlib.pyplot as plt

def plot_memory_usage(file_path, label):
    df = pd.read_csv(file_path)
    plt.plot(df['TimeMs'], df['MemoryBytes'] / 1024.0, label=label)  # Convert to KB

# Plot both BFS and DFS memory logs
plot_memory_usage("bfs_memory.csv", "BFS (High Memory)")
plot_memory_usage("dfs_memory.csv", "DFS (Low Memory)")

plt.title("Memory Usage Over Time")
plt.xlabel("Time (ms)")
plt.ylabel("Memory (KB)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("memory_comparison.png")
plt.show()
