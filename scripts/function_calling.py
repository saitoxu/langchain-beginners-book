import json

from openai import OpenAI


def get_current_weather(location, unit="fahrenheit"):
    if "tokyo" in location.lower():
        return json.dumps(
            {
                "location": "Tokyo",
                "temperature": "10",
                "unit": unit,
            }
        )
    elif "san francisco" in location.lower():
        return json.dumps(
            {"location": "San Francisco", "temperature": "72", "unit": unit}
        )
    elif "paris" in location.lower():
        return json.dumps(
            {
                "location": "Paris",
                "temperature": "22",
                "unit": unit,
            }
        )
    return json.dumps({"location": location, "temperature": "unknown"})


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather for a specific location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get the weather for.",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["location"],
            },
        },
    }
]

messages = [
    {
        "role": "user",
        "content": "東京の天気はどうですか？",
    },
]


client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4.1-nano",
    messages=messages,
    tools=tools,
)

response_message = response.choices[0].message
messages.append(response_message)

available_functions = {"get_current_weather": get_current_weather}

for tool_call in response_message.tool_calls:
    function_name = tool_call.function.name
    function_to_call = available_functions.get(function_name)
    function_args = json.loads(tool_call.function.arguments)
    function_response = function_to_call(**function_args)

    messages.append(
        {
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": function_name,
            "content": function_response,
        }
    )

second_response = client.chat.completions.create(
    model="gpt-4.1-nano",
    messages=messages,
    tools=tools,
)

print(second_response.to_json(indent=2))
