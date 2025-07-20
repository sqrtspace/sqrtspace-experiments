using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

public static class SpaceEfficientMazeSolver
{
    // Memory-limited DFS that only keeps O(√n) visited nodes in memory
    // Recomputes paths when needed, trading time for space
    public static MazeResult MemoryLimitedDFS(bool[,] maze, int memoryLimit)
    {
        var sw = Stopwatch.StartNew();
        long memBefore = GC.GetTotalMemory(true);

        int rows = maze.GetLength(0);
        int cols = maze.GetLength(1);
        int nodesExplored = 0;
        bool pathFound = false;
        int pathLength = 0;

        // Limited memory for visited nodes - simulates √n space
        var limitedVisited = new HashSet<(int, int)>(memoryLimit);
        var currentPath = new HashSet<(int, int)>(); // Track current recursion path to prevent cycles
        
        bool DfsWithRecomputation(int r, int c, int depth)
        {
            nodesExplored++;
            
            // Goal reached
            if (r == rows - 1 && c == cols - 1)
            {
                pathLength = depth;
                return true;
            }

            var current = (r, c);
            
            // Prevent cycles in current path
            if (currentPath.Contains(current))
                return false;
                
            currentPath.Add(current);

            // Add to limited visited set (may evict old entries)
            if (limitedVisited.Count >= memoryLimit && !limitedVisited.Contains(current))
            {
                // Evict oldest entry (simulate FIFO for simplicity)
                var toRemove = limitedVisited.First();
                limitedVisited.Remove(toRemove);
            }
            limitedVisited.Add(current);

            int[] dr = { 0, 1, 0, -1 };
            int[] dc = { 1, 0, -1, 0 };
            
            for (int i = 0; i < 4; i++)
            {
                int nr = r + dr[i], nc = c + dc[i];
                if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && maze[nr, nc])
                {
                    if (DfsWithRecomputation(nr, nc, depth + 1))
                    {
                        currentPath.Remove(current);
                        pathFound = true;
                        return true;
                    }
                }
            }

            currentPath.Remove(current);
            return false;
        }

        pathFound = DfsWithRecomputation(0, 0, 1);

        sw.Stop();
        long memAfter = GC.GetTotalMemory(true);
        
        return new MazeResult 
        { 
            Elapsed = sw.Elapsed, 
            MemoryUsage = memAfter - memBefore,
            PathFound = pathFound,
            PathLength = pathLength,
            NodesExplored = nodesExplored
        };
    }

    // Iterative deepening DFS - uses O(log n) space but recomputes extensively
    public static MazeResult IterativeDeepeningDFS(bool[,] maze)
    {
        var sw = Stopwatch.StartNew();
        long memBefore = GC.GetTotalMemory(true);

        int rows = maze.GetLength(0);
        int cols = maze.GetLength(1);
        int nodesExplored = 0;
        bool pathFound = false;
        int pathLength = 0;

        // Try increasing depth limits
        for (int maxDepth = 1; maxDepth <= rows * cols; maxDepth++)
        {
            bool DepthLimitedDFS(int r, int c, int depth)
            {
                nodesExplored++;
                
                if (depth > maxDepth) return false;
                
                if (r == rows - 1 && c == cols - 1)
                {
                    pathLength = depth;
                    return true;
                }

                int[] dr = { 0, 1, 0, -1 };
                int[] dc = { 1, 0, -1, 0 };
                
                for (int i = 0; i < 4; i++)
                {
                    int nr = r + dr[i], nc = c + dc[i];
                    if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && maze[nr, nc])
                    {
                        if (DepthLimitedDFS(nr, nc, depth + 1))
                            return true;
                    }
                }
                
                return false;
            }

            if (DepthLimitedDFS(0, 0, 0))
            {
                pathFound = true;
                break;
            }
        }

        sw.Stop();
        long memAfter = GC.GetTotalMemory(true);
        
        return new MazeResult 
        { 
            Elapsed = sw.Elapsed, 
            MemoryUsage = memAfter - memBefore,
            PathFound = pathFound,
            PathLength = pathLength,
            NodesExplored = nodesExplored
        };
    }
}