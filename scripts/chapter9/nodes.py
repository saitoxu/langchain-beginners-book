from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI

from scripts.chapter9.roles import ROLES
from scripts.chapter9.state import State

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)
llm = llm.configurable_fields(max_tokens=ConfigurableField(id="max_tokens"))


def selection_node(state: State) -> dict[str, Any]:
    query = state.query
    role_options = "\n".join(
        [f"{k}. {v['name']}: {v['description']}" for k, v in ROLES.items()]
    )
    prompt = ChatPromptTemplate.from_template(
        """質問を分析し、最も適切な回答担当ロールを選択してください。

選択肢:
{role_options}

回答は選択肢の番号（1、2、または3）のみを返してください。

質問: {query}
""".strip()
    )

    chain = (
        prompt
        | llm.with_config(
            configurable=dict(max_tokens=1),
        )
        | StrOutputParser()
    )
    role_number = chain.invoke(dict(role_options=role_options, query=query))
    selected_role = ROLES.get(role_number.strip())["name"]
    return {"current_role": selected_role}


def answering_node(state: State) -> dict[str, Any]:
    query = state.query
    role = state.current_role
    role_details = "\n".join(
        [f"- {v['name']}: {v['details']}" for v in ROLES.values()],
    )

    prompt = ChatPromptTemplate.from_template(
        """あなたは{role}として回答してください。以下の質問に対して、あなたの役割に基づいた適切な回答を提供してください。
役割の詳細:
{role_details}

質問: {query}

回答:""".strip()
    )

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke(
        dict(
            role=role,
            role_details=role_details,
            query=query,
        )
    )
    return {"messages": [answer]}


class Judgement(BaseModel):
    reason: str = Field(default="", description="判定理由")
    judge: bool = Field(default=False, description="判定結果")


def check_node(state: State) -> dict[str, Any]:
    query = state.query
    answer = state.messages[-1]

    prompt = ChatPromptTemplate.from_template(
        """以下の回答の品質をチェックし、問題がある場合は'False'、問題がない場合は'True'を回答してください。また、その判断理由も説明してください。

ユーザーからの質問: {query}
回答: {answer}
""".strip()
    )

    chain = prompt | llm.with_structured_output(Judgement)
    result: Judgement = chain.invoke(dict(query=query, answer=answer))
    return {
        "current_judge": result.judge,
        "judgement_reason": result.reason,
    }
