from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

from langchain.callbacks.tracers import ConsoleCallbackHandler
result = llm.invoke(
    "LLMについて教えて下さい", 
    config={"callbacks": [ConsoleCallbackHandler()]}
)
