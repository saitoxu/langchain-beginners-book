from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4.1-nano", temperature=0.0)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
    ]
)

chain = prompt | model | StrOutputParser()


def respond(session_id: str, human_message: str) -> str:
    chat_message_history = SQLChatMessageHistory(
        session_id=session_id, connection="sqlite:///sqlite.db"
    )
    messages = chat_message_history.get_messages()

    ai_message = chain.invoke(
        {
            "chat_history": messages,
            "input": human_message,
        }
    )

    chat_message_history.add_user_message(human_message)
    chat_message_history.add_ai_message(ai_message)

    return ai_message


session_id = "session-id"

output = respond(session_id, "よろしくお願いします。")
print(output)
