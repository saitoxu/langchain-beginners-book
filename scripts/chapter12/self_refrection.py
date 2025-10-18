import json
import operator
import os
import uuid
from datetime import datetime
from typing import Annotated, Any

import faiss
import numpy as np
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from scripts.chapter12.passive_goal_creator import PassiveGoalCreator
from scripts.chapter12.prompt_optimizer import PromptOptimizer
from scripts.chapter12.response_optimizer import ResponseOptimizer


class DecomposedTasks(BaseModel):
    values: list[str] = Field(
        default_factory=list,
        min_length=1,
        max_length=3,
        description="1~3個に分解されたタスク",
    )


class ReflectiveAgentState(BaseModel):
    query: str = Field(..., description="ユーザーが入力したクエリ")
    optimized_goal: str = Field(default="", description="最適化された目標")
    optimized_response: str = Field(
        default="", description="最適化されたレスポンス定義"
    )
    tasks: list[str] = Field(default_factory=list, description="実行するタスクのリスト")
    current_task_index: int = Field(default=0, description="現在実行中のタスクの番号")
    results: Annotated[list[str], operator.add] = Field(
        default_factory=list, description="実行済みタスクの結果リスト"
    )
    reflection_ids: Annotated[list[str], operator.add] = Field(
        default_factory=list, description="リフレクション結果のIDリスト"
    )
    final_output: str = Field(default="", description="最終的な出力結果")
    retry_count: int = Field(default=0, description="タスクの再試行回数")


class ReflectionJudgment(BaseModel):
    needs_retry: bool = Field(
        description="タスクの実行結果は適切だったと思いますか?あなたの判断を真偽値で示してください。"
    )
    confidence: float = Field(
        description="あなたの判断に対するあなたの自信の度合いを0から1までの小数で示してください。"
    )
    reasons: list[str] = Field(
        description="タスクの実行結果の適切性とそれに対する自信度について、判断に至った理由を簡潔に列挙してください。"
    )


class Reflection(BaseModel):
    id: str = Field(description="リフレクション内容に一意性を与えるためのID")
    task: str = Field(description="ユーザーから与えられたタスクの内容")
    reflection: str = Field(
        description="このタスクに取り組んだ際のあなたの思考プロセスを振り返ってください。何か改善できる点はありましたか?"
        "次に同様のタスクに取り組む際に、より良い結果を出すための教訓を2〜3文程度で簡潔に述べてください。"
    )
    judgment: ReflectionJudgment = Field(description="リトライが必要かどうかの判定")


def format_reflections(reflections: list[Reflection]) -> str:
    return (
        "\n\n".join(
            f"<ref_{i}><task>{r.task}</task><reflection>{r.reflection}"
            "</reflection></ref_{i}>"
            for i, r in enumerate(reflections)
        )
        if reflections
        else "No relevant past reflections."
    )


class ReflectionManager:
    def __init__(self, file_path: str = "self_reflection_db.json"):
        self.file_path = file_path
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.reflections: dict[str, Reflection] = {}
        self.embeddings_dict: dict[str, list[float]] = {}
        self.index = None
        self.load_reflections()

    def load_reflections(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as file:
                data = json.load(file)
                for item in data:
                    reflection = Reflection(**item["reflection"])
                    self.reflections[reflection.id] = reflection
                    self.embeddings_dict[reflection.id] = item["embedding"]

            if self.reflections:
                embeddings = list(self.embeddings_dict.values())
                self.index = faiss.IndexFlatL2(len(embeddings[0]))
                self.index.add(np.array(embeddings).astype("float32"))

    def save_reflection(self, reflection: Reflection) -> str:
        reflection.id = str(uuid.uuid4())
        reflection_id = reflection.id
        self.reflections[reflection_id] = reflection
        embedding = self.embeddings.embed_query(reflection.reflection)
        self.embeddings_dict[reflection_id] = embedding

        if self.index is None:
            self.index = faiss.IndexFlatL2(len(embedding))
        self.index.add(np.array([embedding]).astype("float32"))

        with open(self.file_path, "w", encoding="utf-8") as file:
            json.dump(
                [
                    {
                        "reflection": reflection.model_dump(),
                        "embedding": embedding,
                    }
                    for reflection, embedding in zip(
                        self.reflections.values(),
                        self.embeddings_dict.values(),
                    )
                ],
                file,
                ensure_ascii=False,
                indent=4,
            )

        return reflection_id

    def get_reflection(self, reflection_id: str) -> Reflection | None:
        return self.reflections.get(reflection_id)

    def get_relevant_reflections(
        self,
        query: str,
        k: int = 3,
    ) -> list[Reflection]:
        if not self.reflections or self.index is None:
            return []

        query_embedding = self.embeddings.embed_query(query)
        try:
            D, I = self.index.search(  # noqa: E741
                np.array([query_embedding]).astype("float32"),
                min(k, len(self.reflections)),
            )
            reflection_ids = list(self.reflections.keys())
            return [
                self.reflections[reflection_ids[i]]
                for i in I[0]
                if i < len(reflection_ids)
            ]
        except Exception as e:
            print(f"Error during reflection search: {e}")
            return []


class TaskReflector:
    def __init__(self, llm: ChatOpenAI, reflection_manager: ReflectionManager):
        self.llm = llm.with_structured_output(Reflection)
        self.reflection_manager = reflection_manager

    def run(self, task: str, result: str) -> Reflection:
        prompt = ChatPromptTemplate.from_template(
            "与えられたタスクの内容:\n{task}\n\n"
            "タスクを実行した結果:\n{result}\n\n"
            "あなたは高度な推論能力を持つAIエージェントです。上記のタスクを実行した結果を分析し、"
            "このタスクに対するあなたの取り組みが適切だったかどうかを内省してください。\n"
            "以下の項目に沿って、リフレクションの内容を出力してください。\n\n"
            "リフレクション:\n"
            "このタスクに取り組んだ際のあなたの思考プロセスや方法を振り返ってください。何か改善できる点はありましたか?\n"
            "次に同様のタスクに取り組む際に、より良い結果を出すための教訓を2〜3文程度で簡潔に述べてください。\n\n"
            "判定:\n"
            "- 結果の適切性: タスクの実行結果は適切だったと思いますか?あなたの判断を真偽値で示してください。\n"
            "- 判定の自信度: 上記の判断に対するあなたの自信の度合いを0から1までの小数で示してください。\n"
            "- 判定の理由: タスクの実行結果の適切性とそれに対する自信度について、判断に至った理由を簡潔に列挙してください。\n\n"
            "出力は必ず日本語で行ってください。\n\n"
            "Tips: Make sure to answer in the correct format."
        )

        chain = prompt | self.llm

        def invoke_chain() -> Reflection:
            return chain.invoke({"task": task, "result": result})

        reflection = invoke_chain()
        reflection_id = self.reflection_manager.save_reflection(reflection)
        reflection.id = reflection_id

        return reflection


class QueryDecomposer:
    def __init__(self, llm: ChatOpenAI, reflection_manager: ReflectionManager):
        self.llm = llm.with_structured_output(DecomposedTasks)
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        self.reflection_manager = reflection_manager

    def run(self, query: str) -> DecomposedTasks:
        reflections = self.reflection_manager.get_relevant_reflections(query)
        reflection_text = format_reflections(reflections)
        prompt = ChatPromptTemplate.from_template(
            f"CURRENT_DATE: {self.current_date}\n"
            "-----\n"
            "タスク: 与えられた目標を具体的で実行可能なタスクに分解してください。\n"
            "要件:\n"
            "1. 以下の行動だけで目標を達成すること。決して指定された以外の行動をとらないこと。\n"
            "   - インターネットを利用して、目標を達成するための調査を行う。\n"
            "2. 各タスクは具体的かつ詳細に記載されており、単独で実行ならびに検証可能な情報を含めること。一切抽象的な表現を含まないこと。\n"
            "3. タスクは実行可能な順序でリスト化すること。\n"
            "4. タスクは日本語で出力すること。\n"
            "5. タスクを作成する際に以下の過去のふりかえりを考慮すること:\n{reflections}\n\n"
            "目標: {query}"
        )
        chain = prompt | self.llm
        tasks = chain.invoke({"query": query, "reflections": reflection_text})
        return tasks


class TaskExecutor:
    def __init__(self, llm: ChatOpenAI, reflection_manager: ReflectionManager):
        self.llm = llm
        self.reflection_manager = reflection_manager
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        self.tools = [TavilySearchResults(max_results=3)]

    def run(self, task: str) -> str:
        reflections = self.reflection_manager.get_relevant_reflections(task)
        reflection_text = format_reflections(reflections)
        agent = create_react_agent(self.llm, self.tools)
        result = agent.invoke(
            {
                "messages": [
                    (
                        "human",
                        f"CURRENT_DATE: {self.current_date}\n"
                        "-----\n"
                        f"次のタスクを実行し、詳細な回答を提供してください。\n\nタスク: {task}\n\n"
                        "要件:\n"
                        "1. 必要に応じて提供されたツールを使用すること。\n"
                        "2. 実行において徹底的かつ包括的であること。\n"
                        "3. 可能な限り具体的な事実やデータを提供すること。\n"
                        "4. 発見事項を明確に要約すること。\n"
                        f"5. 以下の過去のふりかえりを考慮すること:\n{reflection_text}\n",
                    )
                ]
            }
        )
        return result["messages"][-1].content


class ResultAggregator:
    def __init__(self, llm: ChatOpenAI, reflection_manager: ReflectionManager):
        self.llm = llm
        self.reflection_manager = reflection_manager
        self.current_date = datetime.now().strftime("%Y-%m-%d")

    def run(
        self,
        query: str,
        results: list[str],
        reflection_ids: list[str],
        response_definition: str,
    ) -> str:
        relevant_reflections = [
            self.reflection_manager.get_reflection(
                rid,
            )
            for rid in reflection_ids
        ]
        prompt = ChatPromptTemplate.from_template(
            "与えられた目標:\n{query}\n\n"
            "調査結果:\n{results}\n\n"
            "与えられた目標に対し、調査結果を用いて、以下の指示に基づいてレスポンスを生成してください。\n"
            "{response_definition}\n\n"
            "過去のふりかえりを考慮すること:\n{reflection_text}\n"
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke(
            {
                "query": query,
                "results": "\n\n".join(
                    f"Info {i+1}:\n{result}"
                    for i, result in enumerate(
                        results,
                    )
                ),
                "response_definition": response_definition,
                "reflection_text": format_reflections(relevant_reflections),
            }
        )


class ReflectiveGoalCreator:
    def __init__(self, llm: ChatOpenAI, reflection_manager: ReflectionManager):
        self.llm = llm
        self.reflection_manager = reflection_manager
        self.passive_goal_creator = PassiveGoalCreator(llm=self.llm)
        self.prompt_optimizer = PromptOptimizer(llm=self.llm)

    def run(self, query: str) -> str:
        reflections = self.reflection_manager.get_relevant_reflections(query)
        reflection_text = format_reflections(reflections)

        query = f"{query}\n\n目標設定する際に以下の過去のふりかえりを考慮すること:\n{reflection_text}"
        goal = self.passive_goal_creator.run(query=query)
        optimized_goal = self.prompt_optimizer.run(query=goal.description)
        return optimized_goal.text


class ReflectiveResponseOptimizer:
    def __init__(self, llm: ChatOpenAI, reflection_manager: ReflectionManager):
        self.llm = llm
        self.reflection_manager = reflection_manager
        self.response_optimizer = ResponseOptimizer(llm=llm)

    def run(self, query: str) -> str:
        reflections = self.reflection_manager.get_relevant_reflections(query)
        reflection_text = format_reflections(reflections)

        query = f"{query}\n\nレスポンス最適化に以下の過去のふりかえりを考慮すること:\n{reflection_text}"
        optimized_response: str = self.response_optimizer.run(query=query)
        return optimized_response


class ReflectiveAgent:

    def __init__(
        self,
        reflection_manager: ReflectionManager,
        task_reflector: TaskReflector,
        max_retries: int = 2,
    ):
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)
        self.reflection_manager = reflection_manager
        self.task_reflector = task_reflector
        self.reflective_goal_creator = ReflectiveGoalCreator(
            llm=llm, reflection_manager=self.reflection_manager
        )
        self.reflective_response_optimizer = ReflectiveResponseOptimizer(
            llm=llm, reflection_manager=self.reflection_manager
        )
        self.query_decomposer = QueryDecomposer(
            llm=llm, reflection_manager=self.reflection_manager
        )
        self.task_executor = TaskExecutor(
            llm=llm, reflection_manager=self.reflection_manager
        )
        self.result_aggregator = ResultAggregator(
            llm=llm, reflection_manager=self.reflection_manager
        )
        self.max_retries = max_retries
        self.graph = self._create_graph()

    def run(self, query: str) -> str:
        initial_state = ReflectiveAgentState(query=query)
        final_state = self.graph.invoke(
            initial_state,
            {"recursion_limit": 1000},
        )
        return final_state.get("final_output", "エラー: 出力に失敗しました。")

    def _create_graph(self) -> StateGraph:
        graph = StateGraph(ReflectiveAgentState)
        graph.add_node("goal_setting", self._goal_setting)
        graph.add_node("decompose_query", self._decompose_query)
        graph.add_node("execute_task", self._execute_task)
        graph.add_node("reflect_on_task", self._reflect_on_task)
        graph.add_node("update_task_index", self._update_task_index)
        graph.add_node("aggregate_results", self._aggregate_results)
        graph.set_entry_point("goal_setting")
        graph.add_edge("goal_setting", "decompose_query")
        graph.add_edge("decompose_query", "execute_task")
        graph.add_edge("execute_task", "reflect_on_task")
        graph.add_conditional_edges(
            "reflect_on_task",
            self._should_retry_or_continue,
            {
                "retry": "execute_task",
                "continue": "update_task_index",
                "finish": "aggregate_results",
            },
        )
        graph.add_edge("update_task_index", "execute_task")
        graph.add_edge("aggregate_results", END)
        return graph.compile()

    def _goal_setting(
        self,
        state: ReflectiveAgentState,
    ) -> dict[str, Any]:
        optimized_goal: str = self.reflective_goal_creator.run(
            query=state.query,
        )
        optimized_response: str = self.reflective_response_optimizer.run(
            query=optimized_goal
        )
        return {
            "optimized_goal": optimized_goal,
            "optimized_response": optimized_response,
        }

    def _decompose_query(self, state: ReflectiveAgentState) -> dict[str, Any]:
        tasks: DecomposedTasks = self.query_decomposer.run(
            query=state.optimized_goal,
        )
        return {"tasks": tasks.values}

    def _execute_task(self, state: ReflectiveAgentState) -> dict[str, Any]:
        current_task = state.tasks[state.current_task_index]
        result = self.task_executor.run(task=current_task)
        return {
            "results": [result],
            "current_task_index": state.current_task_index,
        }

    def _reflect_on_task(self, state: ReflectiveAgentState) -> dict[str, Any]:
        current_task = state.tasks[state.current_task_index]
        current_result = state.results[-1]
        reflection = self.task_reflector.run(
            task=current_task,
            result=current_result,
        )
        return {
            "reflection_ids": [reflection.id],
            "retry_count": (
                state.retry_count + 1 if reflection.judgment.needs_retry else 0
            ),
        }

    def _should_retry_or_continue(self, state: ReflectiveAgentState) -> str:
        latest_reflection_id = state.reflection_ids[-1]
        latest_reflection = self.reflection_manager.get_reflection(
            latest_reflection_id,
        )
        if (
            latest_reflection
            and latest_reflection.judgment.needs_retry
            and state.retry_count < self.max_retries
        ):
            return "retry"
        elif state.current_task_index < len(state.tasks) - 1:
            return "continue"
        else:
            return "finish"

    def _update_task_index(
        self,
        state: ReflectiveAgentState,
    ) -> dict[str, Any]:
        return {"current_task_index": state.current_task_index + 1}

    def _aggregate_results(
        self,
        state: ReflectiveAgentState,
    ) -> dict[str, Any]:
        final_output = self.result_aggregator.run(
            query=state.optimized_goal,
            results=state.results,
            reflection_ids=state.reflection_ids,
            response_definition=state.optimized_response,
        )
        return {"final_output": final_output}


if __name__ == "__main__":
    reflection_manager = ReflectionManager()
    task_reflector = TaskReflector(
        llm=ChatOpenAI(model="gpt-4.1-mini", temperature=0.0),
        reflection_manager=reflection_manager,
    )
    agent = ReflectiveAgent(
        reflection_manager=reflection_manager,
        task_reflector=task_reflector,
    )
    query = "猫砂が飛び散らない方法"
    output = agent.run(query)
    print(output)
