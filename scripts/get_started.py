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
        # {
        #     "role": "assistant",
        #     "content": "こんにちは、ジョンさん！お会いできてうれしいです。今日はどのようにお手伝いできますか？",
        # },
        # {
        #     "role": "user",
        #     "content": "私の名前が分かりますか？",
        # },
    ],
    # stream=True,
    # logprobs=True,
    # stop=["ジョン"],
    # temperature=1.9,
    # n=2,
)

# for chunk in response:
#     content = chunk.choices[0].delta.content
#     if content is not None:
#         print(content, end="", flush=True)
print(response.to_json(indent=2))
