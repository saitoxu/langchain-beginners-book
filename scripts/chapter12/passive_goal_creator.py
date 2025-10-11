from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class Goal(BaseModel):
    description: str = Field(..., description="目標の説明")


class PassiveGoalCreator:

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_template(
            "ユーザーの入力を分析し、明確で実行可能な目標を生成してください。\n"
            "要件\n"
            "1. 目標は具体的かつ明確であり、実行可能なレベルで詳細化されている必要があります。\n"
            "2. あなたが実行可能な行動は以下の行動だけです。\n"
            "  - インターネットを利用して、目標を達成するための調査を行う。\n"
            "  - ユーザーのためのレポートを生成する。\n"
            "3. 決して2.以外の行動を取ってはいけません。\n"
            "ユーザーの入力: {query}"
        )

    def run(self, query: str) -> Goal:
        chain = self.prompt | self.llm.with_structured_output(Goal)
        return chain.invoke({"query": query})


if __name__ == "__main__":
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)
    goal_creator = PassiveGoalCreator(llm=llm)
    user_input = "猫砂が飛び散らないようにしたい"
    goal = goal_creator.run(user_input)
    print(goal)
