from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4.1-nano",
    messages=[
        {
            "role": "system",
            "content": "ステップバイステップで考えてください。",
        },
        {
            "role": "user",
            "content": "10 + 2 * 3 - 4 * 2",
        },
    ],
)

print(response.to_json(indent=2))
