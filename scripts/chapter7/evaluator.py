from typing import Any

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith.evaluation import evaluate
from langsmith.schemas import Example, Run
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision
from ragas.metrics.base import Metric, MetricWithEmbeddings, MetricWithLLM


class RagasMetricEvaluator:
    def __init__(
        self,
        metric: Metric,
        llm: BaseChatModel,
        embeddings: Embeddings,
    ):
        self.metric = metric

        if isinstance(metric, MetricWithLLM):
            self.metric.llm = LangchainLLMWrapper(llm)
        if isinstance(metric, MetricWithEmbeddings):
            self.metric.embeddings = LangchainEmbeddingsWrapper(embeddings)

    def evaluate(self, run: Run, example: Example) -> dict[str, Any]:
        context_strs = [doc.page_content for doc in run.outputs["contexts"]]

        score = self.metric.score(
            {
                "question": example.inputs["question"],
                "user_input": example.inputs["question"],
                "answer": run.outputs["answer"],
                "response": run.outputs["answer"],
                "contexts": context_strs,
                "retrieved_contexts": context_strs,
                "ground_truth": example.outputs["ground_truth"],
                "reference": example.outputs["ground_truth"],
            }
        )
        return {"key": self.metric.name, "score": score}


metrics = [context_precision, answer_relevancy]
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

evaluators = [
    RagasMetricEvaluator(
        metric,
        llm,
        embeddings,
    ).evaluate
    for metric in metrics
]


db = Chroma(
    persist_directory="./chapter6_db",
    embedding_function=embeddings,
)

prompt = ChatPromptTemplate.from_template(
    """\
以下の文脈だけを踏まえて質問に回答してください。

文脈: \"\"\"
{context}
\"\"\"

質問: {question}
"""
)

retriever = db.as_retriever()

chain = RunnableParallel(
    {
        "question": RunnablePassthrough(),
        "context": retriever,
    }
).assign(answer=prompt | llm | StrOutputParser())


def predict(inputs: dict[str, Any]) -> dict[str, Any]:
    question = inputs["question"]
    output = chain.invoke(question)
    return {
        "contexts": output["context"],
        "answer": output["answer"],
    }


print("Running evaluation...")
evaluate(
    predict,
    data="agent-book",
    evaluators=evaluators,
)
print("Evaluation complete.")
