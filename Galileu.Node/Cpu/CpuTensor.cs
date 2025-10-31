using Galileu.Node.Core;
using Galileu.Node.Interfaces;
using System.IO; // Adicionado para usar BinaryWriter

namespace Galileu.Node.Cpu;

public class CpuTensor : IMathTensor
{
    private float[] _data;
    public int[] Shape { get; }
    public long Length { get; }
    public bool IsGpu => false;

    public CpuTensor(float[] data, int[] shape)
    {
        _data = data;
        Shape = shape;
        Length = data.Length;
    }

    /// <summary>
    /// Retorna uma referência direta ao array de dados interno.
    /// </summary>
    public float[] GetData() => _data;

    /// <summary>
    /// Cria e retorna um objeto Tensor da CPU, copiando os dados.
    /// </summary>
    public Tensor ToCpuTensor() => new Tensor((float[])_data.Clone(), Shape);

    /// <summary>
    /// Implementa o método UpdateFromCpu exigido pela interface.
    /// </summary>
    public void UpdateFromCpu(float[] data)
    {
        if (data.Length != this.Length)
        {
            throw new ArgumentException("Os dados de entrada devem ter o mesmo tamanho do tensor.");
        }

        // Copia os novos dados para o array interno.
        Array.Copy(data, _data, this.Length);
    }

    // --- CORREÇÃO APLICADA ---
    /// <summary>
    /// Escreve o conteúdo do tensor diretamente em um BinaryWriter.
    /// Esta implementação é eficiente, pois os dados já estão na RAM e não
    /// requerem alocações intermediárias.
    /// </summary>
    /// <param name="writer">O stream para onde os dados serão escritos.</param>
    public void WriteToStream(BinaryWriter writer)
    {
        // 1. Escreve o número de elementos para que o leitor saiba o tamanho.
        writer.Write(Length);

        // 2. Escreve cada elemento do array de dados diretamente no stream.
        foreach (float val in _data)
        {
            writer.Write(val);
        }
    }

    /// <summary>
    /// Como o array _data é gerenciado pelo Garbage Collector do .NET,
    /// não há recursos não gerenciados para liberar.
    /// </summary>
    public void Dispose()
    {
        /* Nada a fazer */
    }
}