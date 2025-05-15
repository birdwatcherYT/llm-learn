# uv run python target_for_test/structured.py
from dotenv import load_dotenv

load_dotenv()

from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.language_models.base import BaseLanguageModel
from langchain_google_genai import ChatGoogleGenerativeAI


def get_llm() -> BaseLanguageModel:
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")

def get_prompt_and_parser() -> tuple[PromptTemplate, StructuredOutputParser]:
    response_schemas = [
        ResponseSchema(
            name="趣味", description="ユーザーの入力文から判断できる趣味", type="string"
        ),
        ResponseSchema(
            name="職業", description="ユーザーの入力文から判断できる職業", type="string"
        ),
        ResponseSchema(
            name="年齢",
            description="ユーザーの入力文から判断できる年齢",
            type="integer",
        ),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    prompt = PromptTemplate.from_template(
        template="""
「入力文」から「趣味」「職業」「年齢」を抽出してください。
該当する項目がなければnullで埋めてください。

# 入力文
{input_str}

# Format instructions
{format_instructions}
""",
        partial_variables={
            "format_instructions": output_parser.get_format_instructions()
        },
    )
    return prompt, output_parser


def create_extraction_chain():
    llm = get_llm()
    prompt, output_parser = get_prompt_and_parser()
    return prompt | llm | output_parser


if __name__ == "__main__":
    chain = create_extraction_chain()
    input_text = "Webデザイナーをしています。山を登ることが趣味です。生まれてから20年経過しました。"
    result = chain.invoke(input_text)
    print(result)
