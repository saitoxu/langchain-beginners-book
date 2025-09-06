from typing import Any

from langchain_chroma import Chroma
from langchain_cohere import CohereRerank
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def rerank(inp: dict[str, Any], top_n: int = 3) -> list[Document]:
    question = inp["question"]
    documents = inp["documents"]

    cohere_reranker = CohereRerank(
        model="rerank-multilingual-v3.0",
        top_n=top_n,
    )
    return cohere_reranker.compress_documents(
        query=question,
        documents=documents,
    )


prompt = ChatPromptTemplate.from_template(
    '''
以下の文脈だけを踏まえて質問に回答してください。

文脈"""
{context}
"""

質問: {question}
'''
)

model = ChatOpenAI(model="gpt-4.1-nano", temperature=0.0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(
    persist_directory="./chapter6_db",
    embedding_function=embeddings,
)
retriever = db.as_retriever()

rerank_rag_chain = (
    {
        "question": RunnablePassthrough(),
        "documents": retriever,
    }
    | RunnablePassthrough.assign(context=rerank)
    | prompt
    | model
    | StrOutputParser()
)

print(rerank_rag_chain.invoke("LangChainの概要を教えて"))
