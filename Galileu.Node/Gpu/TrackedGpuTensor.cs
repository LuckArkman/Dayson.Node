using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using Galileu.Node.Core;
using Galileu.Node.Interfaces;
using OpenCL.NetCore;
using static OpenCL.NetCore.Cl;
using Exception = System.Exception;

namespace Galileu.Node.Gpu;

/// <summary>
/// Versão rastreada do GpuTensor que automaticamente registra alocações/liberações.
/// Inclui validações de dispose e detecção de uso após liberação.
/// </summary>
public class TrackedGpuTensor : IMathTensor
{
    public int[] Shape { get; private set; }
    public long Length { get; private set; }
    public bool IsGpu => true;
    
    internal Mem Buffer { get; private set; }
    private readonly Context _context;
    private readonly CommandQueue _commandQueue;
    private readonly GpuMemoryTracker? _tracker;
    private readonly int _trackingId;
    private readonly string _allocationLocation;
    private bool _disposed = false;
    
    // Estatísticas de uso
    private int _accessCount = 0;
    private DateTime _creationTime;
    private DateTime _lastAccessTime;
    
    public TrackedGpuTensor(int[] shape, Context context, CommandQueue commandQueue, 
        GpuMemoryTracker? tracker = null, string location = "Unknown")
    {
        Shape = shape;
        Length = shape.Aggregate(1L, (a, b) => a * b);
        _context = context;
        _commandQueue = commandQueue;
        _tracker = tracker;
        _allocationLocation = location;
        _creationTime = DateTime.UtcNow;
        _lastAccessTime = _creationTime;
        
        ErrorCode error;
        Buffer = (Mem)CreateBuffer(_context, MemFlags.ReadWrite, 
            (IntPtr)(Length * sizeof(float)), IntPtr.Zero, out error);
        
        if (error != ErrorCode.Success)
            throw new OpenClException("Falha ao alocar buffer do tensor.", error);
        
        // Registra alocação no tracker
        if (_tracker != null)
        {
            _trackingId = _tracker.TrackAllocation(this, location, shape);
        }
    }
    
    public TrackedGpuTensor(float[] data, int[] shape, Context context, CommandQueue commandQueue,
        GpuMemoryTracker? tracker = null, string location = "Unknown")
    {
        Shape = shape;
        Length = data.Length;
        _context = context;
        _commandQueue = commandQueue;
        _tracker = tracker;
        _allocationLocation = location;
        _creationTime = DateTime.UtcNow;
        _lastAccessTime = _creationTime;
        
        ErrorCode error;
        Buffer = (Mem)CreateBuffer(_context, MemFlags.ReadWrite | MemFlags.CopyHostPtr,
            (IntPtr)(Length * sizeof(float)), data, out error);
        
        if (error != ErrorCode.Success)
            throw new OpenClException("Falha ao alocar e copiar buffer do tensor.", error);
        
        // Registra alocação no tracker
        if (_tracker != null)
        {
            _trackingId = _tracker.TrackAllocation(this, location, shape);
        }
    }
    
    /// <summary>
    /// Valida se o tensor pode ser usado.
    /// </summary>
    private void ValidateNotDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(
                $"TrackedGpuTensor[{_trackingId}]",
                $"Tentativa de usar tensor após Dispose().\n" +
                $"Alocado em: {_allocationLocation}\n" +
                $"Criado em: {_creationTime:HH:mm:ss.fff}\n" +
                $"Acessos: {_accessCount}\n" +
                $"Shape: [{string.Join(", ", Shape)}]");
        }
    }
    
    /// <summary>
    /// Registra acesso ao tensor.
    /// </summary>
    private void RecordAccess()
    {
        _accessCount++;
        _lastAccessTime = DateTime.UtcNow;
    }
    
    public Tensor ToCpuTensor()
    {
        ValidateNotDisposed();
        RecordAccess();
        
        var floatData = new float[Length];
        ErrorCode error = EnqueueReadBuffer(_commandQueue, Buffer, Bool.True, IntPtr.Zero,
            (IntPtr)(Length * sizeof(float)), floatData, 0, null, out _);
        
        if (error != ErrorCode.Success)
            throw new OpenClException("Falha ao ler dados da GPU para a CPU.", error);
        
        return new Tensor(floatData, Shape);
    }
    
    public void UpdateFromCpu(float[] data)
    {
        ValidateNotDisposed();
        RecordAccess();
        
        if (data.Length != Length)
            throw new ArgumentException($"Tamanho incompatível: esperado {Length}, recebido {data.Length}");
        
        ErrorCode error = EnqueueWriteBuffer(_commandQueue, Buffer, Bool.True, IntPtr.Zero,
            (IntPtr)(Length * sizeof(float)), data, 0, null, out _);
        
        if (error != ErrorCode.Success)
            throw new OpenClException("Falha ao escrever dados da CPU para a GPU.", error);
    }
    
    public unsafe void WriteToStream(BinaryWriter writer)
    {
        ValidateNotDisposed();
        RecordAccess();
        
        writer.Write(Length);
        long byteSize = Length * sizeof(float);
        IntPtr hostPtr = Marshal.AllocHGlobal((IntPtr)byteSize);
        
        try
        {
            ErrorCode error = EnqueueReadBuffer(_commandQueue, Buffer, Bool.True, IntPtr.Zero,
                (IntPtr)byteSize, hostPtr, 0, null, out _);
            
            if (error != ErrorCode.Success)
                throw new OpenClException("Falha ao ler buffer da GPU para ponteiro não gerenciado.", error);
            
            float* pFloat = (float*)hostPtr;
            for (int i = 0; i < Length; i++)
            {
                writer.Write(*pFloat);
                pFloat++;
            }
        }
        finally
        {
            Marshal.FreeHGlobal(hostPtr);
        }
    }
    
    /// <summary>
    /// Retorna informações de debug do tensor.
    /// </summary>
    public string GetDebugInfo()
    {
        var age = DateTime.UtcNow - _creationTime;
        var timeSinceLastAccess = DateTime.UtcNow - _lastAccessTime;
        
        return $"TrackedGpuTensor[{_trackingId}]\n" +
               $"  Shape: [{string.Join(", ", Shape)}]\n" +
               $"  Size: {Length * sizeof(float) / (1024.0 * 1024.0):F2} MB\n" +
               $"  Created: {_creationTime:HH:mm:ss.fff} ({age.TotalSeconds:F1}s ago)\n" +
               $"  Last Access: {_lastAccessTime:HH:mm:ss.fff} ({timeSinceLastAccess.TotalSeconds:F1}s ago)\n" +
               $"  Access Count: {_accessCount}\n" +
               $"  Location: {_allocationLocation}\n" +
               $"  Disposed: {_disposed}";
    }
    
    public void Dispose()
    {
        if (_disposed) return;
        
        try
        {
            // Libera buffer OpenCL
            ReleaseMemObject(Buffer);
            
            // Registra liberação no tracker
            if (_tracker != null)
            {
                _tracker.TrackDeallocation(_trackingId);
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[TrackedGpuTensor] Erro ao liberar tensor {_trackingId}: {ex.Message}");
        }
        finally
        {
            _disposed = true;
            GC.SuppressFinalize(this);
        }
    }
    
    ~TrackedGpuTensor()
    {
        if (!_disposed)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"⚠️  TENSOR NÃO LIBERADO: {_trackingId} alocado em {_allocationLocation}");
            Console.ResetColor();
            
            Dispose();
        }
    }
}