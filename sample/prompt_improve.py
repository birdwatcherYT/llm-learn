from dotenv import load_dotenv

load_dotenv()

import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.output_parsers import StrOutputParser

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# 応答生成
response_prompt = PromptTemplate.from_template(
    """{prompt}

**対話履歴**に続くように適切な応答を自然言語で返してください。

# 対話履歴
```json
{chat_history}
```
"""
)
response_chain = response_prompt | llm | StrOutputParser()

# 入力生成役
human_prompt = PromptTemplate.from_template(
    """あなたは悩み事を持った人(human)です。AIではありません。
**対話履歴**に続くように適当な応答を短い自然言語で返してください。

# 対話履歴
```json
{chat_history}
```
"""
)
human_chain = human_prompt | llm | StrOutputParser()

# プロンプト改善役
improve_prompt = PromptTemplate.from_template(
    """あなたはLLMのプロンプトチューニングのプロです。
与えられた**プロンプト**と**対話履歴**を見て、対話履歴のaiによる出力が**フィードバック**を実現するようにプロンプトを修正してください。
フィードバックはプロンプトの出力結果に対するもので、プロンプト自体に対するものではありません。
対話履歴の出力を実現したいわけではありません。
修正後のプロンプトだけを出力して、余計な説明は付与しないでください。あなたの出力結果がそのまま改善後のプロンプトとして使われます。

# 対話履歴
```json
{chat_history}
```

# フィードバック
{feedback}

# プロンプト
{prompt}
"""
)
improve_chain = improve_prompt | llm | StrOutputParser()

# 改善したいプロンプト
prompt = """あなたはAIです。
ユーザーと以下の順序で対話をして悩み事を解決してください。
1. 悩み事を聞く
2. 抽象度が高い場合は具体性を引き出す
3. アドバイスをする
4. ユーザーの反応に応じて別の視点でのアドバイスをしてみる
"""

history = []
feedback_history = []
while True:
    history_str = json.dumps(history, ensure_ascii=False, indent=1)
    answer = response_chain.invoke({"prompt": prompt, "chat_history": history_str})
    print("ai:", answer)
    print("-" * 10)
    history.append({"ai": answer})
    history_str = json.dumps(history, ensure_ascii=False, indent=1)
    human_input = human_chain.invoke(history_str)
    print("human(ai):", human_input)
    print("-" * 10)
    history.append({"human(ai)": human_input})

    feedback = input("feedback:").strip()
    if feedback == "skip" or feedback == "":
        # 入力がなければそのまま継続
        continue
    elif feedback == "reset":
        history.clear()
        continue
    print(">" * 40)
    print(prompt)
    print("<" * 40)
    feedback_history.append(feedback)
    print("feedback_history:", feedback_history)
    history_str = json.dumps(history, ensure_ascii=False, indent=1)
    while True:
        new_prompt = improve_chain.invoke(
            {
                "feedback": feedback,
                # "feedback": feedback_history,
                "prompt": prompt,
                "chat_history": history_str,
            }
        )
        print(">" * 40)
        print(new_prompt)
        print("<" * 40)
        ok_ng = input("ok?ng?(ok以外はng扱い): ")
        if ok_ng == "ok":
            prompt = new_prompt
            break
    # 対話履歴が残ってると過去の生成結果に影響を受けるのでリセットしておく
    history.clear()
