from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "ユーザーが入力した料理のレシピを考えてください。"),
        ("human", "{dish}"),
    ]
)

model = ChatOpenAI(model="gpt-4.1-nano", temperature=0.0)

output_parser = StrOutputParser()

chain = prompt | model | output_parser
# output = chain.invoke({"dish": "カレーライス"})
# print(output)

for chunk in chain.stream({"dish": "カレーライス"}):
    print(chunk, end="", flush=True)
