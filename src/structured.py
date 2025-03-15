from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import PromptTemplate

PROMPT = """
「入力文」から「趣味」「職業」「年齢」を抽出してください。
該当する項目がなければnullで埋めてください。

# 入力文
{input_str}

# Format instructions
{format_instructions}
"""

response_schemas = [
    ResponseSchema(name="趣味", description="ユーザーの入力文から判断できる趣味", type="string"),
    ResponseSchema(name="職業", description="ユーザーの入力文から判断できる職業", type="string"),
    ResponseSchema(name="年齢", description="ユーザーの入力文から判断できる年齢", type="integer"),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

prompt_template = PromptTemplate(
    template=PROMPT,
    input_variables=["input_str"],
    partial_variables={"format_instructions": output_parser.get_format_instructions()},
)

chain = prompt_template | llm | output_parser
result=chain.invoke("Webデザイナーをしています。山を登ることが趣味です。生まれてから20年経過しました。")
print(result) # -> {'趣味': '山登り', '職業': 'Webデザイナー', '年齢': 20}
print(type(result)) # -> <class 'dict'>

