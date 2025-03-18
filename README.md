# LangChainコピペ用コード
[Qiita](https://qiita.com/birdwatcher/items/b5cc66ce59095dee5625)で紹介したコードを基本に追加したり減らしたりしているサンプルコード。

ネット上に落ちているコードやChatGPTが出力するコードが非推奨コードも多いので、ここにサンプルを置いておく。

2025年3月の最新langchain=0.3.20でWARNINGが出ないコードたち。

## 環境構築
```sh
uv sync
```

`.env`の作成
```
GOOGLE_API_KEY="Google AI Studioから取得したAPIキー"

# 以下はオプション
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY="LANGSMITHのAPIキー"
LANGSMITH_PROJECT="llm-learn"
```
VerteAIのGeminiを使いたい場合は、`ChatGoogleGenerativeAI`を`ChatVertexAI`に置き換える必要がある

## プログラム
### src
- [基礎](src/basic.py)
    - `uv run python src/basic.py`
- [ストリーム出力](src/stream.py) 
    - `uv run python src/stream.py`
- [デバッグ用出力](src/debug.py)
    - `uv run python src/debug.py`
- [構造化出力](src/structured.py)
    - `uv run python src/structured.py`
- [構造化出力失敗時の自動修正・リトライ](src/autofix.py) 
    - `uv run python src/autofix.py`
- [より複雑な構造化出力](src/pydantic_out.py)
    - `uv run python src/pydantic_out.py`
- [FewShotプロンプト](src/fewshot.py)
    - `uv run python src/fewshot.py`
- [直列に複数のLLMをつなぐ](src/chain.py)
    - `uv run python src/chain.py`
- [並列に複数のLLMをつなぐ](src/chain_parallel.py)
    - `uv run python src/chain_parallel.py`
- [LangGraph](src/graph.py)
    - `uv run python src/graph.py`
- [関数呼び出し](src/func_call.py)
    - `uv run python src/func_call.py`
- [会話履歴の記憶(自前実装)](src/memory_custom.py)
    - `uv run python src/memory_custom.py`
- [会話履歴の記憶(RunnableWithMessageHistory)](src/history.py)
    - `uv run python src/history.py`
- [会話履歴の記憶(MemorySaver)](src/saver.py)
    - `uv run python src/saver.py`
- [会話履歴の要約(ConversationSummaryMemory)](src/summary.py)
    - `uv run python src/summary.py`
- [近似近傍探索](src/ann.py)
    - `uv run python src/ann.py`
- [RAG](src/rag.py)
    - `uv run python src/rag.py`

### sample
- [マルチ対話](sample/multichat.py)
    - `uv run python sample/multichat.py`
