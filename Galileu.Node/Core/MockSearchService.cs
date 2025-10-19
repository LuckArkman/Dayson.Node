using Galileu.Node.Interfaces;

namespace Galileu.Node.Core;

public class MockSearchService : ISearchService
{
    public List<string> Search(string query)
    {
        return new List<string>
        {
            $"Informação sobre '{query}': Dados recentes indicam que '{query}' é um tópico relevante.",
            $"Contexto adicional para '{query}': Exemplos e casos de uso estão disponíveis online."
        };
    }
}