using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading;

public static class MemoryLogger
{
    public static void LogMemoryUsage(string filename, Func<MazeResult> simulation, int intervalMs = 50)
    {
        var memoryData = new List<(double, long)>();
        var stopwatch = Stopwatch.StartNew();

        // Start memory polling in background
        var polling = true;
        var thread = new Thread(() =>
        {
            while (polling)
            {
                var time = stopwatch.Elapsed.TotalMilliseconds;
                var memory = GC.GetTotalMemory(false);
                memoryData.Add((time, memory));
                Thread.Sleep(intervalMs);
            }
        });

        thread.Start();

        // Run the simulation
        simulation.Invoke();

        // Stop polling
        polling = false;
        thread.Join();
        stopwatch.Stop();

        // Write CSV
        using var writer = new StreamWriter(filename);
        writer.WriteLine("TimeMs,MemoryBytes");
        foreach (var (time, mem) in memoryData)
        {
            writer.WriteLine($"{time:F2},{mem}");
        }

        Console.WriteLine($"Memory usage written to: {filename}");
    }
}
