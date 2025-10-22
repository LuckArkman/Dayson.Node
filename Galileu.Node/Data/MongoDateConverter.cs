using System;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Galileu.Node.Data;

public class MongoDateConverter : JsonConverter<DateTime>
{
    public override DateTime Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        if (reader.TokenType == JsonTokenType.StartObject)
        {
            reader.Read(); // Move to property name
            if (reader.GetString() == "$date")
            {
                reader.Read(); // Move to date string
                string dateString = reader.GetString();
                reader.Read(); // Move to end object
                return DateTime.Parse(dateString);
            }
            throw new JsonException("Expected MongoDB $date format.");
        }
        return reader.GetDateTime();
    }

    public override void Write(Utf8JsonWriter writer, DateTime value, JsonSerializerOptions options)
    {
        writer.WriteStartObject();
        writer.WriteString("$date", value.ToString("o")); // ISO 8601 format
        writer.WriteEndObject();
    }
}