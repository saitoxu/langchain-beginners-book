# from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class Recipe(BaseModel):
    ingredients: list[str] = Field(description="料理に使う材料")
    steps: list[str] = Field(description="料理の手順")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "ユーザーが入力した料理のレシピを考えてください。"),
        ("user", "{dish}"),
    ],
)

model = ChatOpenAI(model="gpt-4.1-nano", temperature=0.0)


# chain = prompt | model | StrOutputParser()
chain = prompt | model.with_structured_output(Recipe)

recipe = chain.invoke({"dish": "カレー"})
print(recipe)
print(type(recipe))
