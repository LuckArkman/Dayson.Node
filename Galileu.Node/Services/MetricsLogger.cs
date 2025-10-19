using System;
using System.IO;
using System.Text;

namespace Galileu.Node.Services;

public class MetricsLogger : IDisposable
{
    private readonly StreamWriter _writer;
    private readonly object _lock = new object();
    private bool _disposed = false;

    public MetricsLogger(string logFilePath)
    {
        var directory = Path.GetDirectoryName(logFilePath);
        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }
        _writer = new StreamWriter(logFilePath, false, Encoding.UTF8);
        WriteHeader();
    }

    private void WriteHeader()
    {
        var header = "Timestamp,Epoch,Batch,BLEU,ROUGE_L,CosineSimilarity,Input,Reference,Candidate";
        lock (_lock)
        {
            _writer.WriteLine(header);
            _writer.Flush();
        }
    }

    public void LogMetrics(MetricsRecord record)
    {
        Func<string, string> escape = (s) => $"\"{s.Replace("\"", "\"\"")}\"";
        var line = string.Join(",",
            DateTime.UtcNow.ToString("o"), record.Epoch, record.Batch,
            record.BleuScore.ToString("F4"), record.RougeLScore.ToString("F4"), record.CosineSimilarity.ToString("F4"),
            escape(record.InputText), escape(record.ReferenceText), escape(record.CandidateText)
        );
        lock (_lock)
        {
            if (!_disposed) _writer.WriteLine(line);
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        lock (_lock)
        {
            _writer.Flush();
            _writer.Close();
            _writer.Dispose();
            _disposed = true;
        }
    }
}