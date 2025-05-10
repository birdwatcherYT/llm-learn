import pytest
from pytest import MonkeyPatch
from typing import Any

from langchain_core.language_models.fake import FakeListLLM

from target_for_test.structured import create_extraction_chain

# --- テスト関数の定義 ---


@pytest.mark.parametrize(
    "mock_llm_response, input_text, expected_output",
    [
        (
            # LLMのモック応答
            """```json
{
	"趣味": "山登り",
	"職業": "Webデザイナー",
	"年齢": 20
}
```""",
            "Webデザイナーをしています。山を登ることが趣味です。生まれてから20年経過しました。",  # テスト対象の入力テキスト
            {"趣味": "山登り", "職業": "Webデザイナー", "年齢": 20},  # 期待される出力
        ),
        (
            """```json
{
	"趣味": "読書",
	"職業": null,
	"年齢": 25
}
```""",
            "趣味は読書です。25歳です。",
            {"趣味": "読書", "職業": None, "年齢": 25},
        ),
        (
            """```json
{
	"趣味": null,
	"職業": null,
	"年齢": null
}
```""",
            "特になし。",
            {"趣味": None, "職業": None, "年齢": None},
        ),
    ],
)
def test_extract_info(
    monkeypatch: MonkeyPatch,
    mock_llm_response: str,
    input_text: str,
    expected_output: dict[str, Any],
) -> None:
    """
    `create_extraction_chain`で生成される情報抽出チェーンの動作をテストします。

    各テストケースについて、指定された`mock_llm_response`を使用して`FakeListLLM`を
    セットアップし、`target_for_test.structured.get_llm`をモンキーパッチします。
    これにより、実際のLLM API呼び出しを伴わずにチェーンのロジックを検証します。
    """
    # FakeListLLMインスタンスを作成
    fake_llm_instance = FakeListLLM(responses=[mock_llm_response])

    # `target_for_test.structured.get_llm`関数を置き換える。
    # これにより、テスト対象コード内で`get_llm()`が呼び出された際に、
    # このテストケース用に作成した`fake_llm_instance`が返されるようになる。
    monkeypatch.setattr(
        "target_for_test.structured.get_llm", lambda: fake_llm_instance
    )

    extraction_chain = create_extraction_chain()
    actual_output = extraction_chain.invoke(input_text)
    assert actual_output == expected_output
