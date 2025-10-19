using System;
using System.Linq;
using Galileu.Node.Interfaces;
using Galileu.Node.Core;
using OpenCL.NetCore;
using System.IO;
using System.Runtime.InteropServices;
using static OpenCL.NetCore.Cl;

namespace Galileu.Node.Gpu;

public class GpuTensor : IMathTensor
{
    public int[] Shape { get; }
    public long Length { get; }
    public bool IsGpu => true;

    internal Mem Buffer { get; private set; }
    private readonly Context _context;
    private readonly CommandQueue _commandQueue;
    private bool _disposed = false;

    public GpuTensor(int[] shape, Context context, CommandQueue commandQueue)
    {
        Shape = shape;
        Length = shape.Aggregate(1L, (a, b) => a * b);
        _context = context;
        _commandQueue = commandQueue;

        ErrorCode error;
        Buffer = (Mem)CreateBuffer(_context, MemFlags.ReadWrite, (IntPtr)(Length * sizeof(float)), IntPtr.Zero, out error);
        if (error != ErrorCode.Success) throw new OpenClException("Falha ao alocar buffer do tensor.", error);
    }

    public GpuTensor(double[] data, int[] shape, Context context, CommandQueue commandQueue)
    {
        Shape = shape;
        Length = data.Length;
        _context = context;
        _commandQueue = commandQueue;
        var floatData = Array.ConvertAll(data, d => (float)d);

        ErrorCode error;
        Buffer = (Mem)CreateBuffer(_context, MemFlags.ReadWrite | MemFlags.CopyHostPtr, (IntPtr)(Length * sizeof(float)), floatData, out error);
        if (error != ErrorCode.Success) throw new OpenClException("Falha ao alocar e copiar buffer do tensor.", error);
    }

    public Tensor ToCpuTensor()
    {
        var floatData = new float[Length];
        ErrorCode error = EnqueueReadBuffer(_commandQueue, Buffer, Bool.True, IntPtr.Zero, (IntPtr)(Length * sizeof(float)), floatData, 0, null, out _);
        if (error != ErrorCode.Success) throw new OpenClException("Falha ao ler dados da GPU para a CPU.", error);

        var doubleData = Array.ConvertAll(floatData, f => (double)f);
        return new Tensor(doubleData, Shape);
    }

    public void UpdateFromCpu(double[] data)
    {
        var floatData = Array.ConvertAll(data, d => (float)d);
        ErrorCode error = EnqueueWriteBuffer(_commandQueue, Buffer, Bool.True, IntPtr.Zero, (IntPtr)(Length * sizeof(float)), floatData, 0, null, out _);
        if (error != ErrorCode.Success) throw new OpenClException("Falha ao escrever dados da CPU para a GPU.", error);
    }

    // --- CORREÇÃO 1: Adicionada a palavra-chave 'unsafe' ---
    /// <summary>
    /// Escreve o conteúdo do tensor da GPU diretamente em um stream.
    /// O método agora é 'unsafe' para permitir o uso de ponteiros.
    /// </summary>
    public unsafe void WriteToStream(BinaryWriter writer)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(GpuTensor));

        writer.Write(Length);
        long byteSize = Length * sizeof(float);
        IntPtr hostPtr = Marshal.AllocHGlobal((IntPtr)byteSize);
        try
        {
            ErrorCode error = EnqueueReadBuffer(_commandQueue, Buffer, Bool.True, IntPtr.Zero,
                (IntPtr)byteSize, hostPtr, 0, null, out _);
            if (error != ErrorCode.Success)
                throw new OpenClException("Falha ao ler buffer da GPU para ponteiro não gerenciado.", error);
            
            // O bloco unsafe agora é permitido porque o método é unsafe.
            float* pFloat = (float*)hostPtr;
            for (int i = 0; i < Length; i++)
            {
                writer.Write((double)(*pFloat));
                pFloat++;
            }
        }
        finally
        {
            Marshal.FreeHGlobal(hostPtr);
        }
    }

    public void Dispose()
    {
        if (_disposed) return;

        // --- CORREÇÃO 2: Removida a verificação incorreta 'Buffer.Content' ---
        // A struct Mem não tem uma propriedade 'Content'. A verificação _disposed
        // já previne a chamada dupla de ReleaseMemObject.
        ReleaseMemObject(Buffer);

        _disposed = true;
        GC.SuppressFinalize(this);
    }
}