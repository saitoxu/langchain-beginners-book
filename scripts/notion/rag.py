from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

prompt = ChatPromptTemplate.from_template(
    '''\
以下の文脈だけを踏まえて質問に回答してください。

文脈: """
{context}
"""

質問: {question}
'''
)

model = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0.0)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

retriever = db.as_retriever(
    search_kwargs={
        "filter": {
            "source": "notion",
        },
    }
)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

query = "どういう人たちが在籍していますか？"

output = chain.invoke(query)
print(output)
