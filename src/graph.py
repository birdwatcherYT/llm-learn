from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

chain = ChatGoogleGenerativeAI(model="gemini-1.5-flash") | StrOutputParser()

from typing_extensions import TypedDict, Literal
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph

# 保持したい情報
class State(TypedDict):
    name: str
    area: str
    feature: str
    names: list[str]

# nodeに使う関数定義（returnした値が更新されます）
def get_name(state: State, config: RunnableConfig):
    return {"name": chain.invoke(f"{state['names']}以外の鳥の名前を1つあげて")}

def get_feature(state: State, config: RunnableConfig):
    return {"feature": chain.invoke(f"{state['name']}の特徴を一言で")}

def get_area(state: State, config: RunnableConfig):
    return {"area": chain.invoke(f"{state['name']}の生息地を一言で")}

def output(state: State, config: RunnableConfig):
    print(state)
    return {"names": state["names"] + [state["name"]]}

def finish(state: State, config: RunnableConfig):
    return {}

# 分岐
def judge(state: State, config: RunnableConfig) -> Literal["name_node", "finish"]:
    return "name_node" if len(state["names"]) < 3 else "finish"

# ワークフロー定義
workflow = StateGraph(State)
# node追加
workflow.add_node("name_node", get_name)
workflow.add_node("feature_node", get_feature)
workflow.add_node("area_node", get_area)
workflow.add_node("output", output)
workflow.add_node("finish", finish)
# edge追加
workflow.add_edge("name_node", "area_node")
workflow.add_edge("name_node", "feature_node")
workflow.add_edge("area_node", "output")
workflow.add_edge("feature_node", "output")
# 始点
workflow.set_entry_point("name_node")
# 終点
workflow.set_finish_point("finish")
# 分岐
workflow.add_conditional_edges("output", judge)

# コンパイル
graph = workflow.compile()
# 実行
print("最終アウトプット:", graph.invoke({"names": []}))
# グラフのmermaid図示
print(graph.get_graph().draw_mermaid())
