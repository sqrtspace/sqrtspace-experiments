using System;
using System.Collections.Generic;
using System.Diagnostics;

public static class MazeSolver
{
    public static MazeResult BFS(bool[,] maze)
    {
        var sw = Stopwatch.StartNew();
        long memBefore = GC.GetTotalMemory(true);

        int rows = maze.GetLength(0);
        int cols = maze.GetLength(1);
        var visited = new bool[rows, cols];
        var queue = new Queue<(int, int)>();
        queue.Enqueue((0, 0));
        visited[0, 0] = true;

        int[] dr = { 0, 1, 0, -1 };
        int[] dc = { 1, 0, -1, 0 };

        while (queue.Count > 0)
        {
            var (r, c) = queue.Dequeue();
            for (int i = 0; i < 4; i++)
            {
                int nr = r + dr[i], nc = c + dc[i];
                if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && maze[nr, nc] && !visited[nr, nc])
                {
                    visited[nr, nc] = true;
                    queue.Enqueue((nr, nc));
                }
            }
        }

        sw.Stop();
        long memAfter = GC.GetTotalMemory(true);
        return new MazeResult { Elapsed = sw.Elapsed, MemoryUsage = memAfter - memBefore };
    }

    public static MazeResult DFS(bool[,] maze)
    {
        var sw = Stopwatch.StartNew();
        long memBefore = GC.GetTotalMemory(true);

        int rows = maze.GetLength(0);
        int cols = maze.GetLength(1);

        void DfsVisit(int r, int c, HashSet<(int, int)> visited)
        {
            visited.Add((r, c));
            int[] dr = { 0, 1, 0, -1 };
            int[] dc = { 1, 0, -1, 0 };
            for (int i = 0; i < 4; i++)
            {
                int nr = r + dr[i], nc = c + dc[i];
                if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && maze[nr, nc] && !visited.Contains((nr, nc)))
                {
                    DfsVisit(nr, nc, visited);
                }
            }
        }

        DfsVisit(0, 0, new HashSet<(int, int)>());

        sw.Stop();
        long memAfter = GC.GetTotalMemory(true);
        return new MazeResult { Elapsed = sw.Elapsed, MemoryUsage = memAfter - memBefore };
    }
}
