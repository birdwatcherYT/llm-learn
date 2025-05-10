import pytest
import json
from pytest import MonkeyPatch
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

from target_for_test.token import create_extraction_chain


@pytest.mark.parametrize(
    "input_text, expected_content, expected_token_usage",
    [
        (
            "私はエンジニアで、趣味はサイクリング、30歳です。",
            {"趣味": "サイクリング", "職業": "エンジニア", "年齢": 30},
            {"input_tokens": 50, "output_tokens": 25, "total_tokens": 75},
        ),
        (
            "映画を見るのが好きです。年は22です。",
            {"趣味": "映画鑑賞", "職業": None, "年齢": 22},
            {"input_tokens": 30, "output_tokens": 15, "total_tokens": 45},
        ),
        (
            "特になし。",
            {"趣味": None, "職業": None, "年齢": None},
            {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        ),
    ],
)
def test_create_extraction_chain_with_token_data(
    monkeypatch: MonkeyPatch,
    input_text: str,
    expected_content: dict[str, Any],
    expected_token_usage: dict[str, int],
) -> None:
    """
    token.pyのcreate_extraction_chain関数が、コンテントとトークン使用量を
    正しく抽出できるかをテストします。
    LLMをモック化し、制御されたAIMessageを返すようにします。
    """
    # expected_contentとexpected_token_usageからAIMessageを構築します。
    content_json_string = json.dumps(expected_content, ensure_ascii=False, indent=2)
    mock_aim_message = AIMessage(
        content=f"```json\n{content_json_string}\n```",
        usage_metadata=expected_token_usage,
    )

    # 1. FakeMessagesListChatModelのセットアップ
    # このモデルは、モックのAIMessageを返します。
    fake_llm_instance = FakeMessagesListChatModel(responses=[mock_aim_message])

    # 2. target_for_test.token.get_llmのパッチ適用
    # token.pyのcreate_extraction_chainが呼び出された際に、get_llmがここで検索されます。
    monkeypatch.setattr("target_for_test.token.get_llm", lambda: fake_llm_instance)

    # target_for_test.structuredの実際のget_prompt_and_parserを使用して、

    # 3. token.pyから抽出チェーンを作成します。
    extraction_chain = create_extraction_chain()

    # 4. チェーンを実行します。
    # structured.pyのget_prompt_templateからのプロンプトは "input_text" を使用します。
    actual_result = extraction_chain.invoke(input_text)
    # 5. アサーション
    assert actual_result["content"] == expected_content
    assert actual_result["token"] == expected_token_usage
