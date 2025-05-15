from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

prompt = ChatPromptTemplate(
    [
        ("system", "あなたは優秀なアシスタントです。"),
        MessagesPlaceholder(variable_name="chat_history"), # ここにチャット履歴が入る
        ("human", "{human_input}"),
    ]
)
chain = prompt | llm

store = {}
def get_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="human_input",
    history_messages_key="chat_history",
)

print(chain_with_history.invoke(
    {"human_input": "こんにちは！私の名前はbirdwatcherです。"},
    config={"configurable": {"session_id": "test-id"}},
).content)
print(chain_with_history.invoke(
    {"human_input": "私の名前を覚えてますか？"},
    config={"configurable": {"session_id": "test-id"}},
).content)
print(store)
