from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

system_message = "あなたは優秀なアシスタントです"

memory = MemorySaver()
langgraph_agent_executor = create_react_agent(
    llm, [], state_modifier=system_message, checkpointer=memory
)

config = {"configurable": {"thread_id": "test-id"}}
print(
    langgraph_agent_executor.invoke(
        {"messages": [("user", "こんにちは！私の名前はbirdwatcherです。")]},
        config,
    )["messages"][-1].content
)
print(
    langgraph_agent_executor.invoke(
        {"messages": [("user", "私の名前を覚えてますか？")]}, config
    )["messages"][-1].content
)
