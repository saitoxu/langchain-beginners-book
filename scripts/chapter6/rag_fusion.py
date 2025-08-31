from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field


def reciprocal_rank_fusion(
    retriever_outputs: list[list[Document]], k: int = 60, top_k: int = 4
) -> list[str]:
    content_score_mapping = {}

    for docs in retriever_outputs:
        for rank, doc in enumerate(docs):
            content = doc.page_content

            if content not in content_score_mapping:
                content_score_mapping[content] = 0

            content_score_mapping[content] += 1 / (rank + k)

    ranked = sorted(
        content_score_mapping.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    print(ranked[:top_k])
    print(len(ranked[:top_k]))
    return [content for content, _ in ranked[:top_k]]


class QueryGenerationOutput(BaseModel):
    queries: list[str] = Field(..., description="検索クエリのリスト")


query_generation_prompt = ChatPromptTemplate.from_template(
    """\
質問に対してベクターデータベースから関連文書を検索するために、
3つの異なる検索クエリを生成してください。
距離ベースの類似性検索の限界を克服するために、
ユーザーの質問に対して複数の視点を提供することが目標です。

質問: {question}
"""
)

model = ChatOpenAI(model="gpt-4.1-nano", temperature=0.0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(
    persist_directory="./chapter6_db",
    embedding_function=embeddings,
)
retriever = db.as_retriever()

query_generation_chain = (
    query_generation_prompt
    | model.with_structured_output(QueryGenerationOutput)
    | (lambda x: x.queries)
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

context_chain = query_generation_chain
context_chain |= retriever.map()
context_chain |= reciprocal_rank_fusion

rag_fusion_chain = (
    {
        "question": RunnablePassthrough(),
        "context": context_chain,
    }
    | prompt
    | model
    | StrOutputParser()
)

print(rag_fusion_chain.invoke("LangChainの概要を教えて"))
