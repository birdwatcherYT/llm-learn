from dotenv import load_dotenv

load_dotenv()

import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

# --- 各エージェント用のチェーン定義 ---
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# エージェントAngel
parser_angel = StructuredOutputParser.from_response_schemas(
    [ResponseSchema(name="angel", description="発言内容", type="string")]
)
prompt_angel = PromptTemplate.from_template(
    """
あなたは相談内容に対して的確なアドバイスをするエージェント(angel)です。
適切なアドバイスをしてあげましょう。

# あなたの性格
- 優しい
- ポジティブ

# 話者一覧
- angel: 優しくポジティブな性格のアドバイザー(あなた)
- devil: 厳しくネガティブな性格のアドバイザー
- human: 相談した人間

# 会話履歴
```json
{history}
```

# 出力形式
{format}
""",
    partial_variables={"format": parser_angel.get_format_instructions()},
)
chain_angel = prompt_angel | llm | parser_angel

# エージェントDevil
parser_devil = StructuredOutputParser.from_response_schemas(
    [ResponseSchema(name="devil", description="発言内容", type="string")]
)
prompt_devil = PromptTemplate.from_template(
    """
あなたは相談内容に対して的確なアドバイスをするエージェント(devil)です。
適切なアドバイスをしてあげましょう。

# あなたの性格
- 厳しい
- ネガティブ

# 話者一覧
- angel: 優しくポジティブな性格のアドバイザー
- devil: 厳しくネガティブな性格のアドバイザー(あなた)
- human: 相談した人間

# 会話履歴
```json
{history}
```

# 出力形式
{format}
""",
    partial_variables={"format": parser_devil.get_format_instructions()},
)
chain_devil = prompt_devil | llm | parser_devil

# 次話者割り当てエージェント
parser_moderator = StructuredOutputParser.from_response_schemas(
    [ResponseSchema(name="next", description="次の話者名", type="string")]
)
prompt_moderator = PromptTemplate.from_template(
    """
あなたは会話を回す司会進行役です。
会話の流れを踏まえて、次に発言すべき話者を決定してください。

# 話者一覧
- angel: 優しくポジティブな性格のアドバイザー
- devil: 厳しくネガティブな性格のアドバイザー
- human: 相談した人間

# 判断の基準
- 同じ人が連続で話さないように会話を回す
- ラウンドロビンにせず、基本はangelとhumanが対話をする
- angelのアドバイスがあまりにも足りていないと判断されたとき、angelの後にdevilに振る
- devilが言い過ぎだと判断したらangelに振る

# 会話履歴
```json
{history}
```

# 出力形式
{format}
""",
    partial_variables={"format": parser_moderator.get_format_instructions()},
)

chain_moderator = prompt_moderator | llm | parser_moderator


# --- ユーザー入力取得 ---
history = []
user_input = input("あなたの相談内容を入力してください: ")
history.append({"human": user_input})
while True:
    history_str = json.dumps(history, ensure_ascii=False, indent=1)
    response = chain_moderator.invoke(history_str)
    print("next speaker:", response)
    if response["next"] == "angel":
        answer = chain_angel.invoke(history_str)
        print("angel:", answer["angel"])
        history.append(answer)
    elif response["next"] == "devil":
        answer = chain_devil.invoke(history_str)
        print("devil:", answer["devil"])
        history.append(answer)
    else:
        user_input = input("返信: ")
        history.append({"human": user_input})
