using System;

public class MazeResult
{
    public TimeSpan Elapsed { get; set; }
    public long MemoryUsage { get; set; }
    public bool PathFound { get; set; }
    public int PathLength { get; set; }
    public int NodesExplored { get; set; }
}