using System;

public static class MazeGenerator
{
    public static bool[,] Generate(int rows, int cols)
    {
        var maze = new bool[rows, cols];
        var rand = new Random();
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                maze[r, c] = rand.NextDouble() > 0.2; // 80% open
        maze[0, 0] = true;
        maze[rows - 1, cols - 1] = true;
        return maze;
    }
}
