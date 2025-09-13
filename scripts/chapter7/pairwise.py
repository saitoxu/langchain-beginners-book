from langsmith import evaluate

from langchain import hub
from langchain.chat_models import init_chat_model

prompt = hub.pull("langchain-ai/pairwise-evaluation-2")
model = init_chat_model("gpt-4o-mini")
chain = prompt | model


def ranked_preference(inputs: dict, outputs: list[dict]) -> list:
    # Assumes example inputs have a 'question' key and experiment
    # outputs have an 'answer' key.
    response = chain.invoke(
        {
            "question": inputs["question"],
            "answer_a": outputs[0].get("answer", "N/A"),
            "answer_b": outputs[1].get("answer", "N/A"),
        }
    )
    if response["Preference"] == 1:
        scores = [1, 0]
    elif response["Preference"] == 2:
        scores = [0, 1]
    else:
        scores = [0, 0]
    return scores


evaluate(
    (
        "warm-zinc-69",
        "perfect-sky-48",
    ),  # Replace with the names/IDs of your experiments
    evaluators=[ranked_preference],
    randomize_order=True,
    max_concurrency=4,
)
