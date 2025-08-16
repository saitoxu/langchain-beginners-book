from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4.1-nano", temperature=0.0)

messages = [
    SystemMessage("You are a helpful assistant."),
    HumanMessage("What is the capital of France?"),
]

for chunk in model.stream(messages):
    print(chunk.content, end="", flush=True)
