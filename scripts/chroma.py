from langchain_chroma import Chroma
from langchain_community.document_loaders import GitLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter


def file_filter(file_path: str) -> bool:
    return file_path.endswith(".mdx")


loader = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain",
    repo_path="./langchain",
    branch="master",
    file_filter=file_filter,
)

raw_docs = loader.load()
raw_docs = raw_docs[:5]
print(len(raw_docs))

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(raw_docs)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

print(len(docs))

db = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")

retriever = db.as_retriever()

query = "AWSのS3からデータを読み込むためのDocument Loaderはありますか？"
context_docs = retriever.invoke(query)
print(len(context_docs))

first_doc = context_docs[0]
print(f"metadata = {first_doc.metadata}")
print(first_doc.page_content)
