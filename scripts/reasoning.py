from openai import OpenAI

client = OpenAI()

result = client.responses.create(
    model="gpt-5-nano",
    input="Write a haiku about code.",
    reasoning={"effort": "low"},
    text={"verbosity": "low"},
)

print(result.to_json(indent=2))
