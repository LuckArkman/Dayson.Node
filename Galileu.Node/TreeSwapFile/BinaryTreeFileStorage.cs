using System;
using System.IO;
using System.Text;
using System.Threading;

namespace Galileu.Node.TreeSwapFile;

public class BinaryTreeFileStorage : IDisposable
{
    private readonly string _filePath;
    private readonly FileStream _fileStream;
    private readonly BinaryWriter _writer;
    private readonly BinaryReader _reader;
    private readonly ReaderWriterLockSlim _lock = new ReaderWriterLockSlim();
    private bool _disposed = false;
    private static readonly Encoding Utf8NoBom = new UTF8Encoding(false);

    public BinaryTreeFileStorage(string filePath)
    {
        _filePath = filePath;
        var directory = Path.GetDirectoryName(_filePath);
        if (directory != null && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }
        _fileStream = new FileStream(_filePath, FileMode.Create, FileAccess.ReadWrite, FileShare.None);
        _writer = new BinaryWriter(_fileStream, Utf8NoBom, true);
        _reader = new BinaryReader(_fileStream, Utf8NoBom, true);
    }
    
    public long StoreData(byte[] dataBytes)
    {
        _lock.EnterWriteLock();
        try
        {
            if (_disposed) throw new ObjectDisposedException(nameof(BinaryTreeFileStorage));
            long offset = _fileStream.Length; 
            _fileStream.Seek(offset, SeekOrigin.Begin);
            _writer.Write(dataBytes.Length);
            _writer.Write(dataBytes);
            return offset;
        }
        catch { return -1; }
        finally { _lock.ExitWriteLock(); }
    }

    public long StoreData(string data)
    {
        byte[] dataBytes = Utf8NoBom.GetBytes(data);
        return StoreData(dataBytes);
    }

    public void Flush()
    {
        _lock.EnterWriteLock();
        try { _writer?.Flush(); }
        finally { _lock.ExitWriteLock(); }
    }
    
    public string GetData(long offset)
    {
        _lock.EnterReadLock();
        try
        {
            if (_disposed) throw new ObjectDisposedException(nameof(BinaryTreeFileStorage));
            _fileStream.Seek(offset, SeekOrigin.Begin);
            int dataLength = _reader.ReadInt32();
            byte[] dataBytes = _reader.ReadBytes(dataLength);
            return Utf8NoBom.GetString(dataBytes);
        }
        finally
        {
            _lock.ExitReadLock();
        }
    }
    
    // MÉTODO CORRIGIDO (ADICIONADO)
    // Este método público implementa a interface IDisposable.
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    // Este método protegido contém a lógica de limpeza.
    protected virtual void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing)
        {
            _lock.EnterWriteLock();
            try
            {
                _writer?.Close();
                _reader?.Close();
                _fileStream?.Close();
            }
            finally { _lock.ExitWriteLock(); }
            _lock?.Dispose();
        }
        _disposed = true;
    }
    
    public void Clear()
    {
        _lock.EnterWriteLock();
        try
        {
            if (_disposed) return;
            _fileStream.SetLength(0);
            _fileStream.Flush();
        }
        finally { _lock.ExitWriteLock(); }
    }
}