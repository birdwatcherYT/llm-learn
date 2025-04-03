from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# --- 各エージェント用のチェーン定義 ---
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# エージェントAngel
prompt_angel = ChatPromptTemplate(
    [
        (
            "system",
            """あなたは相談内容に対して的確なアドバイスをするエージェント(angel)です。
適切なアドバイスをしてあげましょう。

# あなたの性格
- 優しい
- ポジティブ

# 話者一覧
- angel: 優しくポジティブな性格のアドバイザー(あなた)
- devil: 厳しくネガティブな性格のアドバイザー
- human: 相談した人間
""",
        ),
        MessagesPlaceholder(variable_name="chat_history"),  # ここにチャット履歴が入る
    ],
)
chain_angel = prompt_angel | llm | StrOutputParser()

# エージェントDevil
prompt_devil = ChatPromptTemplate(
    [
        (
            "system",
            """あなたは相談内容に対して的確なアドバイスをするエージェント(devil)です。
適切なアドバイスをしてあげましょう。

# あなたの性格
- 厳しい
- ネガティブ

# 話者一覧
- angel: 優しくポジティブな性格のアドバイザー
- devil: 厳しくネガティブな性格のアドバイザー(あなた)
- human: 相談した人間
""",
        ),
        MessagesPlaceholder(variable_name="chat_history"),  # ここにチャット履歴が入る
    ],
)
chain_devil = prompt_devil | llm | StrOutputParser()

# 次話者割り当てエージェント
parser_moderator = StructuredOutputParser.from_response_schemas(
    [ResponseSchema(name="next", description="次の話者名", type="string")]
)
prompt_moderator = ChatPromptTemplate(
    [
        (
            "system",
            """あなたは会話を回す司会進行役です。
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

# 出力形式
{format}
""",
        ),
        MessagesPlaceholder(variable_name="chat_history"),  # ここにチャット履歴が入る
    ],
    partial_variables={"format": parser_moderator.get_format_instructions()},
)

chain_moderator = prompt_moderator | llm | parser_moderator


# --- ユーザー入力取得 ---
history = []
user_input = input("あなたの相談内容を入力してください: ")
history.append(HumanMessage(user_input))

while True:
    response = chain_moderator.invoke(history)
    print("next speaker:", response)
    if response["next"] == "angel":
        answer = chain_angel.invoke(history)
        print("angel:", answer)
        history.append(AIMessage(answer, name="angel"))
        # history.append(AIMessage(f"angel: {answer}"))
    elif response["next"] == "devil":
        answer = chain_devil.invoke(history)
        print("devil:", answer)
        history.append(AIMessage(answer, name="devil"))
        # history.append(AIMessage(f"devil: {answer}"))
    else:
        user_input = input("返信: ")
        history.append(HumanMessage(user_input))
