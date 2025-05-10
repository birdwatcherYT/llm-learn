# uv run python -m target_for_test.token
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.runnables import RunnableLambda
from .structured import get_prompt_and_parser, get_llm

from dotenv import load_dotenv

load_dotenv()


def get_token_parser(base_parser: BaseOutputParser[Any]) -> RunnableLambda:
    """LLMの応答からトークン使用量とパースされたコンテンツを取得する RunnableLambda を返します。
    利用例: chain = prompt | llm | get_token_parser(parser)
    """

    def token_parser_func(message: BaseMessage) -> dict[str, Any]:
        """LLMからの応答メッセージを処理し、トークン情報とパースされたコンテンツを返します。

        Args:
            message (BaseMessage): LLMからの応答。通常は AIMessage インスタンス。
                     このオブジェクトは .content 属性と、トークン情報を含む可能性のある .usage_metadata 属性を持つことが期待されます。

        Returns:
            dict[str, Any]: トークン情報 (存在する場合) とパースされたコンテンツを含む辞書。
        """
        # 例: {'input_tokens': 100, 'output_tokens': 30, 'total_tokens': 130}
        if not hasattr(message, "usage_metadata"):
            raise Exception("usage_metadataがありません")
        output_result = {
            "content": base_parser.parse(message.content),
            "token": message.usage_metadata,
        }
        return output_result

    return RunnableLambda(token_parser_func)


def create_extraction_chain():
    llm = get_llm()
    prompt, output_parser = get_prompt_and_parser()
    return prompt | llm | get_token_parser(output_parser)


if __name__ == "__main__":
    chain = create_extraction_chain()
    input_text = "Webデザイナーをしています。山を登ることが趣味です。生まれてから20年経過しました。"
    result = chain.invoke(input_text)
    print(result)
