from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

query = "AWSのS3からデータを読み込むためのDocument Loaderはありますか？"
vector = embeddings.embed_query(query)
print(len(vector))
print(vector[:5])  # Print the first 5 elements of the vector
