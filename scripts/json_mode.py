from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4.1-nano",
    messages=[
        {
            "role": "system",
            "content": '人物一覧を次のJSON形式で出力してください。\n{"people": ["aaa", "bbb"]}',
        },
        {
            "role": "user",
            "content": "昔々あるところにおじいさんとおばあさんがいました。"
            "おじいさんは山へ芝刈りに、おばあさんは川へ洗濯に行きました。",
        },
    ],
    response_format={"type": "json_object"},
)

print(response.to_json(indent=2))
