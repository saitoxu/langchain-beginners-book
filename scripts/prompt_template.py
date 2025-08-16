from langsmith import Client

client = Client()
prompt = client.pull_prompt("recipe:c37132e4", include_model=False)

prompt_value = prompt.invoke({"dish": "カレーライス"})
print(prompt_value)

# from langchain_core.messages import AIMessage, HumanMessage
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a helpful assistant."),
#         MessagesPlaceholder("chat_history", optional=True),
#         ("human", "{input}"),
#     ]
# )

# prompt_value = prompt.invoke(
#     {
#         "chat_history": [
#             HumanMessage("What is the capital of France?"),
#             AIMessage("The capital of France is Paris."),
#         ],
#         "input": "What is the capital of France?",
#     }
# )
# print(prompt_value)

# from langchain_core.prompts import PromptTemplate

# prompt = PromptTemplate.from_template(
#     """以下の料理のレシピを考えてください。

# 料理名: {dish}
# """.strip()
# )

# prompt_value = prompt.invoke({"dish": "カレーライス"})
# print(prompt_value.to_string())
