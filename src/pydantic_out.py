from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate

PROMPT = """
日本の身近な野鳥を、名前と体長をセットでいくつか列挙してください。

# Format instructions
{format_instructions}
"""
# 出力したい型を定義
class Bird(BaseModel):
    name: str = Field(description="名前")
    length: int = Field(description="体長(cm)")

class BirdList(BaseModel):
    birds: list[Bird] = Field(description="Birdのリスト")

# PydanticOutputParserで行う場合
output_parser = PydanticOutputParser(pydantic_object=BirdList)
prompt_template = PromptTemplate(
    template=PROMPT,
    partial_variables={"format_instructions": output_parser.get_format_instructions()},
)

chain1 = prompt_template | llm | output_parser
print(chain1.invoke({}))
# -> birds=[Bird(name='スズメ', length=14), Bird(name='ハト', length=30), Bird(name='ムクドリ', length=22), Bird(name='カラス', length=45), Bird(name='メジロ', length=11), Bird(name='シジュウカラ', length=14)]

# with_structured_outputで行う場合
chain2 = llm.with_structured_output(BirdList)
print(chain2.invoke("日本の身近な野鳥を、名前と体長をセットでいくつか列挙してください。"))
# -> birds=[Bird(name='スズメ', length=14), Bird(name='ハクセキレイ', length=27), Bird(name='ムクドリ', length=22)]
