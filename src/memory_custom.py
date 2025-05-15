from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

chain = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite") | StrOutputParser()

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

chat_history = [SystemMessage(content="あなたは優秀なアシスタントです")]

print("チャット開始")
while True:
    user_input = input("USER:")
    chat_history.append(HumanMessage(content=user_input))
    ai_answer = chain.invoke(chat_history)
    chat_history.append(AIMessage(content=ai_answer))
    print("AI:", ai_answer)
