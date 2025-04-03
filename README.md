# LangChainコピペ用コード
[Qiita](https://qiita.com/birdwatcher/items/b5cc66ce59095dee5625)で紹介したコードを基本に追加したり減らしたりしているサンプルコード。

ネット上に落ちているコードやChatGPTが出力するコードが非推奨コードも多いので、ここにサンプルを置いておく。

2025年3月の最新langchain=0.3.20でWARNINGが出ないコードたち。

## Python
### 環境構築
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

### プログラム
#### src
- [基礎](src/basic.py)
    - `uv run python src/basic.py`
- [ストリーム出力](src/stream.py) 
    - `uv run python src/stream.py`
- [デバッグ用出力(ConsoleCallbackHandler)](src/debug.py)
    - `uv run python src/debug.py`
- [構造化出力(StructuredOutputParser)](src/structured.py) with ResponseSchema, PromptTemplate 
    - `uv run python src/structured.py`
- [構造化出力失敗時の自動修正・リトライ(OutputFixingParser)](src/autofix.py) 
    - `uv run python src/autofix.py`
- [より複雑な構造化出力(PydanticOutputParser)](src/pydantic_out.py) with BaseModel, Field
    - `uv run python src/pydantic_out.py`
- [FewShotプロンプト(FewShotPromptTemplate)](src/fewshot.py)
    - `uv run python src/fewshot.py`
- [直列に複数のLLMをつなぐ](src/chain.py) with StrOutputParser
    - `uv run python src/chain.py`
- [並列に複数のLLMをつなぐ(RunnableParallel)](src/chain_parallel.py) with RunnableLambda
    - `uv run python src/chain_parallel.py`
- [LangGraph(StateGraph)](src/graph.py)
    - `uv run python src/graph.py`
- [関数呼び出し(create_react_agent)](src/func_call.py)
    - `uv run python src/func_call.py`
- [会話履歴の記憶(自前実装)](src/memory_custom.py) with AIMessage, HumanMessage, SystemMessage
    - `uv run python src/memory_custom.py`
- [会話履歴の記憶(RunnableWithMessageHistory)](src/history.py) with ChatPromptTemplate
    - `uv run python src/history.py`
- [会話履歴の記憶(MemorySaver)](src/saver.py)
    - `uv run python src/saver.py`
- [会話履歴の要約(ConversationSummaryMemory)](src/summary.py)
    - `uv run python src/summary.py`
- [近似近傍探索](src/ann.py)
    - `uv run python src/ann.py`
- [RAG](src/rag.py)
    - `uv run python src/rag.py`

#### sample
- [マルチ対話](sample/multichat.py)
    - `uv run python sample/multichat.py`
- [ChatPromptTemplateを使うとAIMessageが2連続の場合に空のレスポンスでエラーになる](sample/multichat_error.py)
    - `uv run python sample/multichat_error.py`
- [OpenRouterを使う](sample/openrouter.py)
    - [OpenRouter](https://openrouter.ai/settings/keys)からAPIキーを取得し、`.env`に`OPENROUTER_API_KEY="APIキー"`を記載
    - `uv run python sample/openrouter.py`

## JavaScript
langchain.jsも触ってみた

### 環境構築
```sh
npm install
```

`.env`の作成
```
GOOGLE_API_KEY="Google AI Studioから取得したAPIキー"
```
### プログラム
- [基礎](js/basic.js)
    - `node js/basic.js`
- [構造化出力](js/structured.js)
    - `node js/structured.js`
- [直列に複数のLLMをつなぐ](js/chain.js)
    - `node js/chain.js`
- [並列に複数のLLMをつなぐ](js/chain_parallel.js)
    - `node js/chain_parallel.js`
- [LangGraph](js/graph.js)
    - `node js/graph.js`
