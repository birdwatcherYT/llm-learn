from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", thinking_budget=0)

result = llm.invoke("LLMについて教えて下さい")
print(result.content)
