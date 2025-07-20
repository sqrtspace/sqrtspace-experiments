using System;
using System.Diagnostics;

class SimpleDemo
{
    static void Main()
    {
        Console.WriteLine("=== Space-Time Tradeoff Demo ===\n");
        
        // Create a simple 30x30 maze
        int size = 30;
        var maze = MazeGenerator.Generate(size, size);
        
        // Run BFS (uses more memory, less time)
        Console.WriteLine("1. BFS (O(n) space):");
        var sw1 = Stopwatch.StartNew();
        var bfsResult = MazeSolver.BFS(maze);
        sw1.Stop();
        Console.WriteLine($"   Time: {sw1.ElapsedMilliseconds}ms");
        Console.WriteLine($"   Memory: {bfsResult.MemoryUsage} bytes\n");
        
        // Run memory-limited algorithm (uses less memory, more time)
        Console.WriteLine("2. Memory-Limited DFS (O(√n) space):");
        var sw2 = Stopwatch.StartNew();
        int memLimit = (int)Math.Sqrt(size * size);
        var limitedResult = SpaceEfficientMazeSolver.MemoryLimitedDFS(maze, memLimit);
        sw2.Stop();
        Console.WriteLine($"   Time: {sw2.ElapsedMilliseconds}ms");
        Console.WriteLine($"   Memory: {limitedResult.MemoryUsage} bytes");
        Console.WriteLine($"   Nodes explored: {limitedResult.NodesExplored}");
        
        // Show the tradeoff
        Console.WriteLine("\n=== Analysis ===");
        Console.WriteLine($"Memory reduction: {(1.0 - (double)limitedResult.MemoryUsage / bfsResult.MemoryUsage) * 100:F1}%");
        Console.WriteLine($"Time increase: {((double)sw2.ElapsedMilliseconds / sw1.ElapsedMilliseconds - 1) * 100:F1}%");
        Console.WriteLine("\nThis demonstrates Williams' theoretical result:");
        Console.WriteLine("We can simulate time-bounded algorithms with ~√(t) space!");
    }
}