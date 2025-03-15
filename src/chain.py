from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.callbacks.tracers import ConsoleCallbackHandler

prompt1 = PromptTemplate.from_template("次の文章から趣味を抽出して1つの単語で回答してください。\n{user_input}")
chain1 = prompt1 | llm | StrOutputParser()

prompt2 = PromptTemplate.from_template("次の記述を英語に翻訳してください。\n{hobby}")
chain2 = prompt2 | llm | StrOutputParser()

chain = chain1 | chain2

print(chain.invoke(
    "普段はAIエンジニアをしていますが、休日には外に出て鳥を観察しに行き、撮影をします", 
    config={'callbacks': [ConsoleCallbackHandler()]}
))
