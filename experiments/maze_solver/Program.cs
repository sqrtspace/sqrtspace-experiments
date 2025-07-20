using System;

class Program
{
    static void Main(string[] args)
    {
        int size = 30;
        var maze = MazeGenerator.Generate(size, size);

        Console.WriteLine("Running BFS...");
        MemoryLogger.LogMemoryUsage("bfs_memory.csv", () => MazeSolver.BFS(maze));

        Console.WriteLine("Running DFS with recomputation...");
        MemoryLogger.LogMemoryUsage("dfs_memory.csv", () => MazeSolver.DFS(maze));
    }
}
