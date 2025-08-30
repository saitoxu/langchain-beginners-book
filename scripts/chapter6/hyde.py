from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

prompt = ChatPromptTemplate.from_template(
    '''
以下の文脈だけを踏まえて質問に回答してください。

文脈"""
{context}
"""

質問: {question}
'''
)

hypothetical_prompt = ChatPromptTemplate.from_template(
    """
次の質問に回答する一文を書いてください。

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

hypothetical_chain = hypothetical_prompt | model | StrOutputParser()

hyde_rag_chain = (
    {
        "question": RunnablePassthrough(),
        "context": hypothetical_chain | retriever,
    }
    | prompt
    | model
    | StrOutputParser()
)


print(hyde_rag_chain.invoke("LangChainの概要を教えて"))
