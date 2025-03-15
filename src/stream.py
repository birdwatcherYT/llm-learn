from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

for chunk in llm.stream("LLMについて教えて下さい"):
    print(chunk.content, end="", flush=True)
