using System.Text;

namespace Galileu.Node.TreeSwapFile;

public class TreeNode
{
    public const int MaxDataSize = 50 * 1024; // 1 Megabytes por amostra
    public static readonly int NodeSize = sizeof(long) * 3 + MaxDataSize;

    public long LeftOffset;
    public long RightOffset;
    public long LastModified;
    public byte[] Data;

    public TreeNode()
    {
        LeftOffset = -1;
        RightOffset = -1;
        LastModified = DateTime.UtcNow.Ticks;
        Data = new byte[MaxDataSize];
    }

    public byte[] Serialize()
    {
        using (var ms = new MemoryStream(NodeSize))
        using (var bw = new BinaryWriter(ms))
        {
            bw.Write(LeftOffset);
            bw.Write(RightOffset);
            bw.Write(LastModified);
            bw.Write(Data, 0, MaxDataSize);
            return ms.ToArray();
        }
    }

    public static TreeNode Deserialize(byte[] data)
    {
        var node = new TreeNode();
        using (var ms = new MemoryStream(data))
        using (var br = new BinaryReader(ms))
        {
            node.LeftOffset = br.ReadInt64();
            node.RightOffset = br.ReadInt64();
            node.LastModified = br.ReadInt64();
            node.Data = br.ReadBytes(MaxDataSize);
        }
        return node;
    }
}