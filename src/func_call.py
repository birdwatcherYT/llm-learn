from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool


@tool
def func_bird(input_str: str) -> str:
    """鳥に関する質問に答える"""
    print("called func_bird")
    return "それは鳥です。"

@tool
def func_add(a: int, b: int) -> int:
    """足し算をする"""
    print("called func_add")
    return a + b

tools = [func_bird, func_add]

langgraph_agent_executor = create_react_agent(llm, tools)
result = langgraph_agent_executor.invoke(
    {"messages": [("human", "ハトについて教えて")]}
)
print(result["messages"][-1].content)

result = langgraph_agent_executor.invoke(
    {"messages": [("human", "3と4を足すとどうなる？")]}
)
print(result["messages"][-1].content)
