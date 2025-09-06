from langchain_chroma import Chroma
from langchain_community.document_loaders import GitLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def file_filter(file_path: str) -> bool:
    return file_path.endswith(".mdx")


loader = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain",
    repo_path="./langchain",
    branch="master",
    file_filter=file_filter,
)

documents = loader.load()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
model = ChatOpenAI(model="gpt-4.1-nano", temperature=0.0)
db = Chroma(
    persist_directory="./chapter6_db",
    embedding_function=embeddings,
)
retriever = db.as_retriever()
chroma_retriever = retriever.with_config({"run_name": "chroma_retriever"})

bm25_retriever = BM25Retriever.from_documents(documents=documents).with_config(
    {"run_name": "bm25_retriever"},
)


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
    # print(ranked[:top_k])
    # print(len(ranked[:top_k]))
    return [content for content, _ in ranked[:top_k]]


hybrid_retriever = (
    RunnableParallel(
        {
            "chroma_documents": chroma_retriever,
            "bm25_documents": bm25_retriever,
        }
    )
    | (lambda x: [x["chroma_documents"], x["bm25_documents"]])
    | reciprocal_rank_fusion
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

hybrid_rag_chain = (
    {
        "question": RunnablePassthrough(),
        "context": hybrid_retriever,
    }
    | prompt
    | model
    | StrOutputParser()
)

print(hybrid_rag_chain.invoke("LangChainの概要を教えて"))
