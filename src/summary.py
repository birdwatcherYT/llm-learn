from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, RemoveMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph


def call_model(state: MessagesState):
    system_prompt = "あなたは有能なアシスタントです。すべての質問にできる限り答えてください。提供されたチャット履歴には、以前の会話の要約が含まれることがあります。"
    system_message = SystemMessage(content=system_prompt)
    message_history = state["messages"][:-1]
    if len(message_history) >= 4:  # 一定の長さを超えたら要約する
        last_human_message = state["messages"][-1]
        summary_prompt = "上記のチャットメッセージを要約してください。ただし、できるだけ多くの具体的な詳細を含めてください。"
        summary_message = llm.invoke(
            message_history + [HumanMessage(content=summary_prompt)]
        )
        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"]]
        human_message = HumanMessage(content=last_human_message.content)
        response = llm.invoke([system_message, summary_message, human_message])
        message_updates = [summary_message, human_message, response] + delete_messages
    else:
        message_updates = llm.invoke([system_message] + state["messages"])
    return {"messages": message_updates}


workflow = StateGraph(state_schema=MessagesState)
workflow.add_node("llm", call_model)
workflow.add_edge(START, "llm")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

chat_history = [
    HumanMessage(content="こんにちは！私の名前はbirdwatcherです。"),
    AIMessage(content="こんにちは!どうしましたか？"),
    HumanMessage(content="今日の気分は？"),
    AIMessage(content="とてもよいです。"),
]
print(app.invoke(
    {"messages": chat_history + [HumanMessage("私の名前を覚えていますか？")]},
    config={"configurable": {"thread_id": "test-id"}},
))
