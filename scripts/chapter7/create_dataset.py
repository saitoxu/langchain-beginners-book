from langchain_community.document_loaders import GitLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith import Client
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import (
    MultiHopAbstractQuerySynthesizer,
    SingleHopSpecificQuerySynthesizer,
)


def file_filter(file_path: str) -> bool:
    return file_path.endswith(".mdx")


loader = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain",
    repo_path="./langchain",
    branch="master",
    file_filter=file_filter,
)

documents = loader.load()
print(f"Total documents: {len(documents)}")

# 10件に制限
documents = documents[:10]

# メタデータをシンプルに保つ
for document in documents:
    # 既存のメタデータを削除してシンプルなものだけ残す
    source = document.metadata.get("source", "")
    document.metadata["filename"] = source


generator_llm = LangchainLLMWrapper(
    ChatOpenAI(model="gpt-4.1-mini", temperature=0.0),
)

generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

# ==== 3) TestsetGenerator の生成 ====
generator = TestsetGenerator(
    llm=generator_llm,
    embedding_model=generator_embeddings,
)

# ==== 4) 質問タイプの分布（旧 simple:0.5, reasoning:0.25, multi_context:0.25 に対応）====
query_distribution = [
    (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 0.5),
    (MultiHopAbstractQuerySynthesizer(llm=generator_llm), 0.5),
    # pydanticのvalidation errorが出るため一旦コメントアウト
    # (MultiHopSpecificQuerySynthesizer(llm=generator_llm), 0.25),
]

# ==== 5) テストセット生成（LangChain の Document をそのまま渡せる）====
testset = generator.generate_with_langchain_docs(
    documents,
    testset_size=4,
    query_distribution=query_distribution,
)


dataset_name = "agent-book"
client = Client()

if client.has_dataset(dataset_name=dataset_name):
    client.delete_dataset(dataset_name=dataset_name)

dataset = client.create_dataset(dataset_name=dataset_name)

# inputs = []
# outputs = []
# metadatas = []

examples = []


for testset_record in testset.samples:
    print(testset_record)
    inputs = {"question": testset_record.eval_sample.user_input}
    outputs = {
        "contexts": testset_record.eval_sample.reference_contexts,
        "ground_truth": testset_record.eval_sample.reference,
    }
    metadata = {
        "synthesizer_name": testset_record.synthesizer_name,
    }

    example = {
        "inputs": inputs,
        "outputs": outputs,
        "metadata": metadata,
    }
    examples.append(example)

client.create_examples(
    examples=examples,
    dataset_id=dataset.id,
)
