from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from scripts.chapter10.models import Personas


class PersonaGenerator:
    def __init__(self, llm: ChatOpenAI, k: int = 5):
        self.llm = llm.with_structured_output(Personas)
        self.k = k

    def run(self, user_request: str) -> Personas:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたはユーザーインタビュー用の多様なペルソナを作成する専門家です。",
                ),
                (
                    "human",
                    f"以下のユーザーリクエストに関するインタビュー用に、{self.k}人の多様なペルソナを生成してください。\n\n"
                    "ユーザーリクエスト: {user_request}\n\n"
                    "各ペルソナには名前と簡単な背景を含めてください。"
                    "年齢、性別、職業、技術的専門知識において多様性を確保してください。",
                ),
            ]
        )
        chain = prompt | self.llm
        return chain.invoke({"user_request": user_request})
