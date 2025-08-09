from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4.1-nano",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "こんにちは！私はジョンと言います！",
        },
    ],
)
print(response.to_json(indent=2))
