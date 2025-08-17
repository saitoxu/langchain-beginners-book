from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class Recipe(BaseModel):
    ingredients: list[str] = Field(description="料理に使う材料")
    steps: list[str] = Field(description="料理の手順")


output_parser = PydanticOutputParser(pydantic_object=Recipe)

format_instructions = output_parser.get_format_instructions()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "ユーザーが入力した料理のレシピを考えてください。\n\n"
            "{format_instructions}",
        ),
        ("human", "{dish}"),
    ]
)

prompt_with_format_instructions = prompt.partial(
    format_instructions=format_instructions
)

prompt_value = prompt_with_format_instructions.invoke({"dish": "カレー"})
# print("=== role: system ===")
# print(prompt_value.messages[0].content)
# print("=== role: user ===")
# print(prompt_value.messages[1].content)


model = ChatOpenAI(model="gpt-4.1-nano", temperature=0.0)

ai_message = model.invoke(prompt_value)

recipe = output_parser.invoke(ai_message)
print(recipe)
print(type(recipe))
