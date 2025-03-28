from dotenv import load_dotenv
load_dotenv()

import os
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek/deepseek-chat-v3-0324:free",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
)

result = llm.invoke("LLMについて教えて下さい")
print(result.content)
