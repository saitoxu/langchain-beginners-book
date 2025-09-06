from enum import Enum
from typing import Any

from langchain_chroma import Chroma
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
model = ChatOpenAI(model="gpt-4.1-nano", temperature=0.0)
db = Chroma(
    persist_directory="./chapter6_db",
    embedding_function=embeddings,
)
retriever = db.as_retriever()
langchain_document_retriever = retriever.with_config(
    {"run_name": "langchain_document_retriever"}
)

web_retriever = TavilySearchAPIRetriever(k=3).with_config(
    {"run_name": "web_retriever"},
)


class Route(str, Enum):
    langchain_document = "langchain_document"
    web = "web"


class RouteOutput(BaseModel):
    route: Route


route_prompt = ChatPromptTemplate.from_template(
    """
質問に回答するために、適切なRetrieverを選択してください。

質問: {question}
"""
)

route_chain = (
    route_prompt
    | model.with_structured_output(
        RouteOutput,
    )
    | (lambda x: x.route)
)


def routed_retriever(inp: dict[str, Any]) -> list[Document]:
    question = inp["question"]
    route = inp["route"]

    if route == Route.langchain_document:
        return langchain_document_retriever.invoke(question)
    elif route == Route.web:
        return web_retriever.invoke(question)

    raise ValueError(f"Unknown route: {route}")


prompt = ChatPromptTemplate.from_template(
    '''
以下の文脈だけを踏まえて質問に回答してください。

文脈"""
{context}
"""

質問: {question}
'''
)

route_rag_chain = (
    {
        "question": RunnablePassthrough(),
        "route": route_chain,
    }
    | RunnablePassthrough.assign(context=routed_retriever)
    | prompt
    | model
    | StrOutputParser()
)

print(route_rag_chain.invoke("東京の今日の天気は？"))
