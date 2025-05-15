from dotenv import load_dotenv
load_dotenv()

from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")

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

prompt_template = PromptTemplate.from_template(
    PROMPT,
    partial_variables={"format_instructions": output_parser.get_format_instructions()},
)

from langchain.output_parsers import OutputFixingParser

output_fixing_parser = OutputFixingParser.from_llm(
    parser=output_parser, llm=llm, max_retries=3,
)

chain = prompt_template | llm | output_fixing_parser
chain_with_retry = chain.with_retry(
    stop_after_attempt=3,
    # wait_exponential_jitter=False # exponential backoffをやめる場合
)
result = chain_with_retry.invoke("普段はAIエンジニアをしていますが、休日には外に出て鳥を観察しに行き、撮影をします")
print(result) # -> {'趣味': '鳥の観察、撮影', '職業': 'AIエンジニア', '年齢': None}
