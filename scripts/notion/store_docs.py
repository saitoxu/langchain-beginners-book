import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import NotionDBLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

integration_token = os.getenv("NOTION_API_KEY")
database_id = os.getenv("NOTION_DATABASE_ID")

filter_object = {
    "property": "ステータス",
    "select": {
        "equals": "在籍",
    },
}

model = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0.0)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

loader = NotionDBLoader(
    integration_token=integration_token,
    database_id=database_id,
    filter_object=filter_object,
    request_timeout_sec=30,
)


def update_doc(doc):
    new_metadata = {}
    for k, v in doc.metadata.items():
        if isinstance(v, (str, int, float, bool)):
            new_metadata[k] = v
    new_metadata["source"] = "notion"
    doc.metadata = new_metadata
    return doc


raw_docs = loader.load()
raw_docs = list(map(update_doc, raw_docs))
# print(len(raw_docs))
# print(raw_docs[0])

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(raw_docs)

Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
