from typing import Iterator

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{input}"),
    ]
)


def upper(input_stream: Iterator[str]) -> Iterator[str]:
    for text in input_stream:
        yield text.upper()


model = ChatOpenAI(model="gpt-4.1-nano", temperature=0.0)

output_parser = StrOutputParser()

chain = prompt | model | output_parser | upper

output = chain.invoke({"input": "Hello! Please generate a long text."})
print(output)

# for chunk in chain.stream({"input": "Hello! Please generate a long text."}):
#     print(chunk, end="", flush=True)
# print()
