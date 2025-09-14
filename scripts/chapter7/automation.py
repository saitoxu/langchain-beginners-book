from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tracers.context import collect_runs
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith import Client

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

chain = (
    {
        "question": RunnablePassthrough(),
        "context": retriever,
    }
    | prompt
    | model
    | StrOutputParser()
)

client = Client()

with collect_runs() as runs_cb:
    output = chain.invoke("LangChainの概要を教えて")
    print(output)
    run_id = runs_cb.traced_runs[0].id
    feedback = client.create_feedback(
        run_id=run_id,
        key="thumbs",
        score=1,
    )
    print(feedback)
