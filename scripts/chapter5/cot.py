from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

cot_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "ユーザーの質問にステップバイステップで回答してください。"),
        ("human", "{question}"),
    ]
)

summarize_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "ステップバイステップで考えた回答から結論だけ抽出してください。"),
        ("human", "{text}"),
    ]
)

model = ChatOpenAI(model="gpt-4.1-nano", temperature=0.0)

output_parser = StrOutputParser()

cot_chain = cot_prompt | model | output_parser
summarize_chain = summarize_prompt | model | output_parser

cot_summarize_chain = cot_chain | summarize_chain
output = cot_summarize_chain.invoke({"question": "10 + 2 * 3"})

print(output)
