from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-5.6",
    input="Say hello to NephroScan AI in one short sentence."
)

print(response.output_text)
