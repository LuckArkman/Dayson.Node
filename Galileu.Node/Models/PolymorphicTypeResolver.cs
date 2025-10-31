using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.Json.Serialization.Metadata;
using Galileu.Node.Models;

namespace Galileu.Node.Models;

public class PolymorphicTypeResolver : DefaultJsonTypeInfoResolver
{
    public override JsonTypeInfo GetTypeInfo(Type type, JsonSerializerOptions options)
    {
        JsonTypeInfo jsonTypeInfo = base.GetTypeInfo(type, options);

        // CORRIGIDO: Usa o tipo base correto 'Message'
        Type baseMessage = typeof(Message);
        if (jsonTypeInfo.Type == baseMessage)
        {
            jsonTypeInfo.PolymorphismOptions = new JsonPolymorphismOptions
            {
                TypeDiscriminatorPropertyName = "$type",
                IgnoreUnrecognizedTypeDiscriminators = true,
                UnknownDerivedTypeHandling = JsonUnknownDerivedTypeHandling.FailSerialization,
            };

            foreach (var derivedType in baseMessage.GetCustomAttributes(typeof(JsonDerivedTypeAttribute), false)
                         .Cast<JsonDerivedTypeAttribute>())
            {
                // Tratamento de erro para TypeDiscriminator nulo
                if (derivedType.TypeDiscriminator != null)
                {
                    jsonTypeInfo.PolymorphismOptions.DerivedTypes.Add(
                        new JsonDerivedType(derivedType.DerivedType, derivedType.TypeDiscriminator.ToString()!));
                }
            }
        }

        return jsonTypeInfo;
    }
}